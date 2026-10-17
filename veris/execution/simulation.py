"""Host lifecycle around Veris' pure single-step physics and scan rollout.

This layer adds no physical equations or numerical tolerances. The maintained
``advance`` transition owns physics, forcing application and boundary halos;
``step`` owns iteration. A Simulation owns the absolute forcing position,
microsecond model clock and borrowed mesh, outside the array-only State.
Use advance/step directly for JIT, gradients, JVP and VJP. The host facade
supports serial initialization and the explicit-mesh modes supported by
initialize_experiment; it never initializes distributed JAX or changes settings.
"""

from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, fields
from datetime import timedelta
from numbers import Integral
from pathlib import Path
from types import TracebackType
from typing import Any, Self

import jax
import jax.numpy as jnp
from jax.sharding import Mesh

from veris.execution.advance import advance
from veris.execution.experiment_initialization import initialize_experiment
from veris.execution.rollout_output import run_timed
from veris.grid.regions import regional_forcing
from veris.io import lifecycle as simulation_io
from veris.io.calendar import duration_us
from veris.io.configuration import OutputSettings
from veris.io.guard import require_host
from veris.io.scan import _FieldSelector
from veris.io.schedule import iter_segments
from veris.model.configuration import Configuration
from veris.model.diagnostics import Diagnostics
from veris.model.experiment import Experiment
from veris.model.physical_constants import PhysicalConstants
from veris.model.types import State

type _Carry = tuple[State, Diagnostics]
type _Observer = Callable[[State, Diagnostics], Any]


@dataclass(frozen=True)
class RunResult:
    """Final numerical fields and host information from one successful run.

    steps counts this call; completed_steps and elapsed count the entire
    Simulation. diagnostics is None until a physical step has succeeded.
    Optional pure observations stack post-step values along a leading time
    axis. Compilation and synchronized execution timings are wall-clock seconds,
    excluding initialization. State/Diagnostics retain packed boundary halos.
    """

    state: State
    diagnostics: Diagnostics | None
    steps: int
    completed_steps: int
    elapsed: timedelta
    observations: Any = None
    compilation_seconds: float = 0.0
    integration_seconds: float = 0.0
    execution_metadata: Mapping[str, Any] | None = None


class Simulation:
    """Mutable host owner of a validated Experiment and its numerical State.

    Construct through initialize. Explicit run counts may exceed the configured
    default target, subject to available forcing and the exact host clock range.
    run(None) advances only the remaining default steps. Input preflight errors
    leave the simulation usable; compilation or execution failures close it and
    preserve its last committed State and time. A supplied mesh is borrowed and
    remains the caller's responsibility. Enabled output owns a lazy file writer and pending means outside State.
    """

    def __init__(
        self,
        experiment: Experiment,
        state: State,
        configuration: Configuration,
        constants: PhysicalConstants,
        mesh: Mesh | None,
        output_path: str | Path | None = None,
        *,
        completed_steps: int = 0,
        elapsed: timedelta = timedelta(0),
        emit_initial_instant: bool = True,
    ) -> None:
        """Bind initialized numerical resources; use initialize as the factory."""
        require_host()
        self._experiment = experiment
        self._state = state
        self._configuration = configuration
        self._physical_constants = constants
        self._mesh = mesh
        self._diagnostics: Diagnostics | None = None
        self._completed_steps = completed_steps
        self._elapsed = elapsed
        self._closed = False
        self._output_manager, self._output_observer = simulation_io.output_resources(
            experiment,
            output_path,
            configuration,
            constants,
            mesh,
            start_step=completed_steps,
            mean_coverage_start_us=duration_us(elapsed),
            emit_initial_instant=emit_initial_instant,
        )
        # Capture only numerical/static values, never this mutable host object.
        # Series replace constants of the same name before entering physics.
        with self._execution_context():
            constant = {
                name: (
                    jnp.asarray(value, dtype=configuration.dtype)
                    if mesh is None
                    else regional_forcing(value, configuration.dtype, mesh)
                )
                for name, value in experiment.resolved_forcing.constant.items()
                if name not in experiment.resolved_forcing.series
            }
        physics = experiment.physics

        def transition(
            carry: _Carry, prescribed: Mapping[str, jax.Array] | None = None
        ) -> _Carry:
            forcing = constant if prescribed is None else {**constant, **prescribed}
            # Keep the first failure visible, and never run more physics on an
            # invalid carry while the compiled rollout finishes its fixed shape.
            return jax.lax.cond(
                jnp.any(carry[1].physics_status != 0),
                lambda _: carry,
                lambda _: advance(
                    carry[0], configuration, constants, forcing, physics=physics
                ),
                operand=None,
            )

        self._transition = transition

    @classmethod
    def initialize(
        cls,
        experiment: Experiment,
        *,
        output_path: str | Path | None = None,
        mesh: Mesh | None = None,
    ) -> Self:
        """Validate, allocate and initialize the experiment through the shared API.

        Set backend/distributed JAX configuration before calling this factory;
        enable x64 explicitly when requesting float64. The initializer validates
        global physical dimensions, masks, inputs, dtype and mesh compatibility.
        No numerical step runs during initialization.
        """
        require_host()
        simulation_io.validate_output(experiment, output_path)
        state, configuration, constants = initialize_experiment(experiment, mesh=mesh)
        return cls(experiment, state, configuration, constants, mesh, output_path)

    def save_restart(self, path: str | Path) -> None:
        """Save all serial State fields and exact clock to a new versioned file.

        Full halos and future/past forcing are retained. The live Simulation,
        pending output and numerical State are unchanged; existing paths reject.
        Multi-device and distributed checkpoints are currently unsupported.
        """
        from veris.io.restart import save_restart

        require_host()
        self._require_open()
        save_restart(
            path, self.state, self.experiment, self.completed_steps, self.elapsed
        )

    @classmethod
    def from_restart(
        cls,
        path: str | Path,
        *,
        output_path: str | Path | None = None,
        output_settings: OutputSettings | None = None,
    ) -> Self:
        """Restore a complete serial checkpoint after cold schema validation.

        Restore the actual saved backend and precision without startup forcing
        or derived-field initialization. Diagnostics are None until the next
        step. Enabled output requires a new file; absolute cadence/calendar
        persist, the checkpoint instant is not replayed, and crossing means
        are discarded. Output overrides must preserve the calendar and origin.
        """
        from veris.io.restart import read_restart

        experiment, state, completed, elapsed = read_restart(
            path, output_path=output_path, output_settings=output_settings
        )
        return cls(
            experiment,
            state,
            experiment.configuration,
            experiment.physical_constants,
            None,
            output_path,
            completed_steps=completed,
            elapsed=elapsed,
            emit_initial_instant=False,
        )

    @property
    def state(self) -> State:
        """Last successfully committed numerical State, including boundary halos."""
        return self._state

    @property
    def experiment(self) -> Experiment:
        """Immutable resolved definition supplied at initialization."""
        return self._experiment

    @property
    def configuration(self) -> Configuration:
        """Static numerical settings, using local physical grid dimensions."""
        return self._configuration

    @property
    def physical_constants(self) -> PhysicalConstants:
        """Immutable physical parameters bound for this host trajectory."""
        return self._physical_constants

    @property
    def completed_steps(self) -> int:
        """Absolute number of successful physics timesteps and forcing cursor."""
        return self._completed_steps

    @property
    def elapsed(self) -> timedelta:
        """Exact accumulated model time, with integer microsecond resolution."""
        return self._elapsed

    @property
    def closed(self) -> bool:
        """Whether the host lifecycle has ended and further execution is rejected."""
        return self._closed

    @contextmanager
    def _execution_context(self) -> Iterator[None]:
        """Honor initialized devices when preparing arrays and tracing physics."""
        if self._mesh is not None:
            with jax.set_mesh(self._mesh):
                yield
        else:
            # A default device alone does not suspend a caller's active mesh.
            # Serial traces must retain one initialized device for output and
            # restart ownership; context exit restores any borrowed mesh.
            with jax.set_mesh(None), jax.default_device(self._state.Area.device):
                yield

    def _require_open(self) -> None:
        """Reject any execution or context entry after lifecycle closure."""
        if self._closed:
            raise RuntimeError("Simulation is closed")

    def _preflight(self, steps: int | None, observe: _Observer | None) -> int:
        """Validate run controls, absolute forcing bounds and exact clock range."""
        if steps is None:
            count = max(0, self._experiment.steps - self._completed_steps)
        else:
            if isinstance(steps, bool) or not isinstance(steps, Integral):
                raise TypeError("steps must be a nonnegative integer or None")
            if steps < 0:
                raise ValueError("steps must be nonnegative")
            count = int(steps)
        if observe is not None and not callable(observe):
            raise TypeError("observe must be a pure callable or None")
        length = self._experiment.resolved_forcing.length
        if length is not None and self._completed_steps + count > length:
            raise ValueError(
                f"requested steps exceed forcing length {length} "
                f"at completed step {self._completed_steps}"
            )
        try:
            _ = self._elapsed + self._experiment.step_duration * count
        except OverflowError as error:
            raise ValueError("elapsed model clock exceeds timedelta range") from error
        if self._output_observer is not None:
            observer = self._output_observer
            iter_segments(
                observer.manager.settings,
                observer.step_seconds,
                count,
                start_step=observer.next_step,
                pending_windows=observer.pending_windows,
            )
        return count

    def step(self) -> RunResult:
        """Advance one physics timestep using the next absolute forcing sample."""
        return self.run(1)

    def run(
        self, steps: int | None = None, observe: _Observer | None = None
    ) -> RunResult:
        """Advance a trajectory and commit only after synchronized device success.

        A pure observe(state, diagnostics) optionally selects post-step histories.
        No default history is allocated. A zero run preserves State/Diagnostics
        identities; with an observer it traces only an identity transition to
        infer empty output shapes, including after forcing exhaustion.
        """
        require_host()
        self._require_open()
        count = self._preflight(steps, observe)
        if count == 0 and observe is None and self._output_observer is None:
            return self._result(0)
        elapsed = self._elapsed + self._experiment.step_duration * count
        try:
            with self._execution_context():
                final, history, compilation, integration = self._execute(count, observe)
        except BaseException:
            self.close()
            raise
        if count:
            self._state, self._diagnostics = final
            self._completed_steps += count
            self._elapsed = elapsed
        return self._result(count, history, compilation, integration)

    def _execute(
        self, count: int, observe: _Observer | None
    ) -> tuple[_Carry, Any, float, float]:
        """Compile existing rollout and synchronize all fields before host commit."""
        diagnostic = self._diagnostics
        if diagnostic is None:
            zeros = jnp.zeros_like(self._state.Area)
            diagnostic = Diagnostics(
                **{item.name: zeros for item in fields(Diagnostics)}
            )
        carry = self._state, diagnostic
        inputs = None
        if count and self._experiment.resolved_forcing.series:
            start = self._completed_steps
            inputs = {
                name: (
                    jnp.asarray(
                        value[start : start + count], dtype=self.configuration.dtype
                    )
                    if self._mesh is None
                    else regional_forcing(
                        value[start : start + count],
                        self.configuration.dtype,
                        self._mesh,
                    )
                )
                for name, value in self._experiment.resolved_forcing.series.items()
            }
        transition = self._transition if count else lambda value: value
        selector = None if observe is None else lambda value: observe(*value)

        output_select = None
        if self._output_observer is not None:
            output_select = _FieldSelector(self._output_observer.names, from_carry=True)
        chunks = []
        final, compilation, integration = run_timed(
            carry,
            transition,
            count,
            inputs=inputs,
            observe=self._output_observer,
            select=output_select,
            history_select=selector,
            history_sink=None if selector is None else chunks.append,
        )
        history = None
        if selector is not None:
            history = jax.tree.map(
                lambda *arrays: jnp.concatenate(arrays, axis=0), *chunks
            )
            jax.block_until_ready(history)
        status = int(jax.device_get(jnp.max(final[1].physics_status)))
        if status:
            reason = (
                "invalid saline-ice formation"
                if status == 2
                else "invalid physical parameters"
            )
            raise ValueError(f"{reason} (physics_status={status})")
        return final, history, compilation, integration

    def snapshot(
        self, path: str | Path, *, variables: Sequence[str] | None = None
    ) -> None:
        """Save selected physical cells and resolved metadata at absolute time.

        Visualization snapshots are selected inputs, not complete restart data.
        Sharded output uses the established collective physical-cell collector.
        """
        require_host()
        self._require_open()
        simulation_io.snapshot(
            path,
            self._state,
            self._experiment,
            self._configuration,
            self._physical_constants,
            self._elapsed,
            self._mesh,
            variables,
        )

    def load_fields(
        self,
        path: str | Path,
        variables: Sequence[str] | None = None,
        *,
        record: int = -1,
        group: str = "snapshot",
    ) -> None:
        """Load serial instantaneous initial/forcing fields transactionally.

        Geometry and derived fields are excluded. Refresh dependent ice/wind
        fields and boundary halos; preserve time, forcing and last Diagnostics.
        Invalid input leaves State identity and lifecycle usability intact.
        """
        require_host()
        self._require_open()
        with self._execution_context():
            candidate = simulation_io.load_fields(
                path,
                self._state,
                self._experiment,
                self._configuration,
                self._physical_constants,
                variables,
                record=record,
                group=group,
            )
        self._state = candidate

    def _result(
        self,
        steps: int,
        observations: Any = None,
        compilation_seconds: float = 0.0,
        integration_seconds: float = 0.0,
    ) -> RunResult:
        """Capture committed fields and absolute time with per-call run metadata."""
        return RunResult(
            state=self._state,
            diagnostics=self._diagnostics,
            steps=steps,
            completed_steps=self._completed_steps,
            elapsed=self._elapsed,
            observations=observations,
            compilation_seconds=compilation_seconds,
            integration_seconds=integration_seconds,
        )

    def close(self) -> None:
        """End the lifecycle idempotently; borrowed meshes stay caller-owned."""
        require_host()
        if self._closed:
            return
        self._closed = True
        try:
            if self._output_manager is not None:
                self._output_manager.close()
        finally:
            if self._output_observer is not None:
                self._output_observer.release()

    def __enter__(self) -> Self:
        """Enter an open simulation's host context."""
        require_host()
        self._require_open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close after success or failure without suppressing user exceptions."""
        self.close()


__all__ = ["RunResult", "Simulation"]
