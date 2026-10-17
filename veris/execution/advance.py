"""One pure sea-ice timestep with prescribed ocean and atmospheric forcing.

Compose the maintained Hibler/Kimmritz dynamics and category thermodynamics in
the standalone reference order. No equations or numerical tolerances change.
State and Diagnostics retain halo-inclusive (x, y) arrays; prescribed inputs use
scalars or global physical arrays without halos. Settings and physics mode are
static; physical constants, State and forcing are dynamic JAX inputs. Host clocks, files
and forcing cursors belong to Simulation, outside this module.
"""

from collections.abc import Mapping
from dataclasses import fields, replace
from functools import partial

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from jax.typing import ArrayLike

from veris.grid.halos import fill_overlap, fill_state_overlap
from veris.model.configuration import Configuration
from veris.model.diagnostics import DIAGNOSTICS, Diagnostics
from veris.model.forcing import FORCING_FIELDS
from veris.model.physical_constants import PhysicalConstants
from veris.model.physical_parameters import _physical_parameters_valid
from veris.model.types import State, jit
from veris.numerics.autodiff import norm_hypot
from veris.physics.dynamics.driver import dynamics_transport
from veris.physics.thermodynamics.growth import _growth_with_status


def _forcing_arrays(
    forcing: Mapping[str, ArrayLike] | None,
    shape: tuple[int, int],
    dtype: str,
) -> dict[str, jax.Array]:
    """Validate static forcing metadata without inspecting dynamic array values."""
    if forcing is None:
        return {}
    if not isinstance(forcing, Mapping):
        raise TypeError("forcing must be a mapping of prescribed fields or None")
    result = {}
    for name, value in forcing.items():
        if name not in FORCING_FIELDS:
            raise ValueError(f"unsupported forcing field {name!r}")
        array = jnp.asarray(value)
        if array.dtype.kind not in "fiu":
            raise TypeError(f"forcing field {name} must contain real numeric values")
        if array.shape not in ((), shape):
            raise ValueError(
                f"forcing field {name} has shape {array.shape}; "
                f"expected scalar or global physical {shape}"
            )
        result[name] = array.astype(dtype)
    return result


def _apply_forcing(
    state: State, forcing: Mapping[str, jax.Array], conf: Configuration
) -> State:
    """Refresh only supplied fields before staggered stencils consume them."""
    values = {}
    for name, value in forcing.items():
        stored = (
            jnp.full_like(getattr(state, name), value)
            if value.ndim == 0
            else jnp.pad(value, 2)
        )
        boundary = (
            "normal" if name == "vOcean" else "zero" if name == "uOcean" else "edge"
        )
        values[name] = fill_overlap(stored, conf, boundary=boundary)
    if values:
        state = replace(state, **values)
    if {"uWind", "vWind"} & forcing.keys() and "wSpeed" not in forcing:
        state = replace(state, wSpeed=norm_hypot(state.uWind, state.vWind))
    return state


def _validate_state(state: State, shape: tuple[int, int], dtype: str) -> None:
    """Check array metadata while tracing, without materializing numerical values."""
    if not isinstance(state, State):
        raise TypeError("state must be a State")
    if jax.dtypes.canonicalize_dtype(dtype) != jnp.dtype(dtype):
        raise ValueError(f"dtype {dtype} requires jax_enable_x64 before execution")
    for item in fields(state):
        value = getattr(state, item.name)
        if not hasattr(value, "shape") or value.shape != shape:
            raise ValueError(f"State field {item.name} must have storage shape {shape}")
        if value.dtype != jnp.dtype(dtype):
            raise ValueError(f"State field {item.name} must have dtype {dtype}")


def _advance_valid(
    state: State,
    conf: Configuration,
    phys: PhysicalConstants,
    forcing: Mapping[str, jax.Array],
    physics: str,
) -> tuple[State, Diagnostics]:
    """Compose reference stages on a local grid, inside shard_map if sharded."""
    state = _apply_forcing(state, forcing, conf)
    zeros = jnp.zeros_like(state.Area)
    stress_u = stress_v = zeros
    if physics != "thermodynamics":
        state, stress_u, stress_v = dynamics_transport(state, conf, phys)
    freshwater = salt = penetrating = zeros
    mass = salt_mass = external_mass = external_heat = status = zeros
    if physics != "dynamics":
        values, status = _growth_with_status(state, conf, phys)
        (
            ice,
            snow,
            area,
            temperature,
            freshwater,
            salt,
            shortwave,
            heat,
            load,
            penetrating,
            inverse,
            mass,
            salt_mass,
            external_mass,
            external_heat,
        ) = values
        state = replace(
            state,
            hIceMean=ice,
            hSnowMean=snow,
            Area=area,
            TSurf=temperature,
            Qsw=shortwave,
            Qnet=heat,
            SeaIceLoad=load,
            recip_hIceMean=inverse,
            os_hIceMean=zeros,
            os_hSnowMean=zeros,
        )
    diagnostics = Diagnostics(
        penetrating,
        stress_u,
        stress_v,
        freshwater,
        salt,
        status,
        mass,
        salt_mass,
        external_mass,
        external_heat,
    )
    # The column reference leaves periodic output halos as produced by Growth.
    # Dynamics/coupled always refresh; closed walls always need field policies.
    if physics != "thermodynamics" or not conf.enable_cyclic_y:
        if conf.use_sharding and jax.process_count() > 1:
            # Keep unchanged fields' final exchanges behind the complete solver
            # result. Otherwise their independent permutes can overlap nested
            # EVP collectives across processes, mismatching Gloo messages.
            state, diagnostics = jax.lax.optimization_barrier((state, diagnostics))
        state = fill_state_overlap(state, conf)
        diagnostics = fill_state_overlap(diagnostics, conf)
    return state, diagnostics


@partial(jit, static_argnames=("conf", "physics"))
def _advance_local(
    state: State,
    conf: Configuration,
    phys: PhysicalConstants,
    forcing: Mapping[str, jax.Array],
    physics: str,
) -> tuple[State, Diagnostics]:
    """Skip invalid physical inputs and return an explicit same-dtype status."""

    def valid(
        operands: tuple[State, PhysicalConstants, Mapping[str, jax.Array]],
    ) -> tuple[State, Diagnostics]:
        current, physical, inputs = operands
        return _advance_valid(current, conf, physical, inputs, physics)

    def invalid(
        operands: tuple[State, PhysicalConstants, Mapping[str, jax.Array]],
    ) -> tuple[State, Diagnostics]:
        current = operands[0]
        zeros = jnp.zeros_like(current.Area)
        diagnostic = Diagnostics(**{name: zeros for name in DIAGNOSTICS})
        return current, replace(diagnostic, physics_status=jnp.ones_like(zeros))

    parameters_valid = _physical_parameters_valid(phys)
    if physics != "thermodynamics" and conf.useFreedrift and not conf.useEVP:
        parameters_valid = parameters_valid & (phys.waterTurnAngle == 0)
    return jax.lax.cond(parameters_valid, valid, invalid, (state, phys, forcing))


def advance(
    state: State,
    configuration: Configuration,
    constants: PhysicalConstants,
    forcing: Mapping[str, ArrayLike] | None = None,
    *,
    physics: str = "coupled",
) -> tuple[State, Diagnostics]:
    """Advance ONE physics timestep and return State and coupling Diagnostics.

    ``physics`` selects coupled, dynamics or thermodynamics. Coupled timesteps
    must agree. Prescribed forcing accepts scalar or global physical (nx, ny)
    arrays, in VARIABLES units. Refresh supplied halos before physics and derive
    wind magnitude when components are supplied without an explicit wSpeed.
    Missing fields retain their State value; in particular, absent Qnet/Qsw keep
    Growth's recursive coupling fluxes. Explicit heat is reapplied on every call.

    This function is pure and supports JIT, grad, JVP and VJP with respect to
    State, independent physical parameters and forcing arrays. Bind configuration
    and physics as static arguments; pass constants as a dynamic PyTree. Dynamic value validation belongs to host Experiment and
    initialization; this path checks names, shapes and real dtypes while tracing.
    For sharding retain the caller's active x/y mesh. ``veris.step`` remains the
    separate multi-step scan rollout and can consume this transition.
    """
    if physics not in ("coupled", "dynamics", "thermodynamics"):
        raise ValueError("physics must be coupled, dynamics or thermodynamics")
    if not isinstance(configuration, Configuration):
        raise TypeError("configuration must be a Configuration")
    if not isinstance(constants, PhysicalConstants):
        raise TypeError("constants must be PhysicalConstants")
    if configuration.dtype != constants.dtype:
        raise ValueError("configuration and constants must have matching dtype")
    if physics == "coupled" and configuration.deltatDyn != configuration.deltatTherm:
        raise ValueError("coupled physics requires equal timesteps")
    conf, phys = configuration, constants
    shape = (conf.nx, conf.ny)
    if conf.use_sharding:
        mesh = jax.sharding.get_abstract_mesh()
        if set(mesh.axis_names) != {"x", "y"}:
            raise ValueError(
                "sharded stepping requires an active mesh with axes x and y"
            )
        if not {"x", "y"} <= set(mesh.manual_axes):
            shape = (conf.nx * mesh.shape["x"], conf.ny * mesh.shape["y"])
            _validate_state(
                state,
                ((conf.nx + 4) * mesh.shape["x"], (conf.ny + 4) * mesh.shape["y"]),
                conf.dtype,
            )
            values = _forcing_arrays(forcing, shape, conf.dtype)
            specs = {
                name: P() if value.ndim == 0 else P("x", "y")
                for name, value in values.items()
            }
            # Forcing formed from replicated differentiable parameters can have
            # replicated explicit types. Partition it before local halo padding;
            # shard_map's in_specs do not perform implicit resharding under JIT.
            values = jax.reshard(
                values,
                {name: NamedSharding(mesh, spec) for name, spec in specs.items()},
            )
            return jax.shard_map(
                lambda local, physical, inputs: _advance_local(
                    local, conf, physical, inputs, physics
                ),
                mesh=mesh,
                in_specs=(P("x", "y"), P(), specs),
                out_specs=(P("x", "y"), P("x", "y")),
            )(state, jax.reshard(phys, NamedSharding(mesh, P())), values)
    _validate_state(state, (conf.nx + 4, conf.ny + 4), conf.dtype)
    values = _forcing_arrays(forcing, shape, conf.dtype)
    return _advance_local(state, conf, phys, values, physics)
