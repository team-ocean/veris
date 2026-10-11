"""One pure sea-ice timestep with prescribed ocean and atmospheric forcing.

Compose the maintained Hibler/Kimmritz dynamics and category thermodynamics in
the standalone reference order. No equations or numerical tolerances change.
State and Diagnostics retain halo-inclusive (x, y) arrays; prescribed inputs use
scalars or global physical arrays without halos. Settings, constants and physics
are static, while State and forcing are dynamic JAX inputs. Host clocks, files
and forcing cursors belong to Simulation, outside this module.
"""

from collections.abc import Mapping
from dataclasses import fields, replace

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from jax.typing import ArrayLike

from veris._ad import norm_hypot
from veris._typing import State
from veris.configuration import Configuration
from veris.diagnostics import Diagnostics
from veris.dynamics import dynamics_transport
from veris.fill_overlap import fill_overlap, fill_state_overlap
from veris.forcing import FORCING_FIELDS
from veris.growth import Growth
from veris.physical_constants import PhysicalConstants


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


def _advance_local(
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
    if physics != "dynamics":
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
        ) = Growth(state, conf, phys)
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
        )
    diagnostics = Diagnostics(penetrating, stress_u, stress_v, freshwater, salt)
    # The column reference leaves periodic output halos as produced by Growth.
    # Dynamics/coupled always refresh; closed walls always need field policies.
    if physics != "thermodynamics" or not conf.enable_cyclic_y:
        state = fill_state_overlap(state, conf)
        diagnostics = fill_state_overlap(diagnostics, conf)
    return state, diagnostics


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
    State and forcing arrays. Bind configuration, constants and physics as
    static arguments. Dynamic value validation belongs to host Experiment and
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
    return _execute_advance(state, configuration, constants, forcing, physics)


def _execute_advance(
    state: State,
    conf: Configuration,
    phys: PhysicalConstants,
    forcing: Mapping[str, ArrayLike] | None,
    physics: str,
) -> tuple[State, Diagnostics]:
    """Dispatch shared physics with the supplied independent numerical timesteps.

    Public advance validates its single coupled clock before calling here.
    The legacy island wrapper retains independent dynamics/thermodynamics
    timesteps; this helper changes neither configuration nor stage arithmetic.
    Both paths share forcing, State validation, sharding and _advance_local.
    """
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
                lambda local, inputs: _advance_local(
                    local, conf, phys, inputs, physics
                ),
                mesh=mesh,
                in_specs=(P("x", "y"), specs),
                out_specs=(P("x", "y"), P("x", "y")),
            )(state, values)
    _validate_state(state, (conf.nx + 4, conf.ny + 4), conf.dtype)
    values = _forcing_arrays(forcing, shape, conf.dtype)
    return _advance_local(state, conf, phys, values, physics)
