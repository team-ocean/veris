"""Precision propagation through alternate dynamics, flux and setup paths."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from veris import heat_flux_CESM as cesm
from veris.dynsolver import IceVelocities
from veris.initialization import initialize
from veris.setups import island
from veris.setups.ocean import OceanGeometry, initialize_from_ocean


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("difference", [-3.0, 3.0])
def test_cesm_stable_and_unstable_fluxes_keep_precision(
    dtype: str, difference: float
) -> None:
    """Both stability branches retain the policy under real atmospheric forcing."""
    _, settings, constants = initialize(2, 3, dtype=dtype)
    ones = jnp.ones((3, 5), dtype=dtype)
    mask = ones.at[0, 0].set(0)
    temperature = 280.0
    saturation = jnp.asarray(
        0.98 * 640380 * np.exp(-5107.4 / temperature) / 1.3, dtype=dtype
    )
    iterative_inputs = (
        mask,
        1.3 * ones,
        10 * ones,
        5 * ones,
        0 * ones,
        saturation * 0.5 * ones,
        (temperature + difference) * ones,
        (temperature + difference) * ones,
        0 * ones,
        0 * ones,
        temperature * ones,
    )
    simple_inputs = (
        mask,
        100000 * ones,
        saturation * 0.5 * ones,
        1.3 * ones,
        5 * ones,
        0 * ones,
        (temperature + difference) * ones,
        0 * ones,
        0 * ones,
        temperature * ones,
    )
    for array in (*iterative_inputs, *simple_inputs):
        assert array.dtype == np.dtype(dtype)
    result = cesm.flux_atmOcn(settings, constants, *iterative_inputs)
    simple = cesm.flux_atmOcn_simple(settings, constants, *simple_inputs)
    for array in (*result, *simple):
        assert array.dtype == np.dtype(dtype)
        assert np.all(np.isfinite(array))
    # Use the precise path for the independent latent-heat equation reference.
    if dtype == "float64":
        np.testing.assert_allclose(
            result[1], constants.latvap * result[3], rtol=1e-13, atol=1e-13
        )


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("solver", ["adaptive_evp", "free_drift"])
def test_alternative_dynamics_keep_precision(dtype: str, solver: str) -> None:
    """Adaptive relaxation and free drift use the same initialized scalar policy."""
    state, settings, constants = island.initialize(4, 5, dtype=dtype)
    settings = replace(
        settings,
        useAdaptiveEVP=solver == "adaptive_evp",
        useFreedrift=solver == "free_drift",
        useEVP=solver == "adaptive_evp",
        nEVPsteps=3,
    )
    ones = jnp.ones_like(state.hIceMean)
    # Uniform wet fields expose precision in the solvers without dry-cell roots.
    state = replace(
        state,
        iceMask=ones,
        iceMaskU=ones,
        iceMaskV=ones,
        Area=ones,
        AreaW=ones,
        AreaS=ones,
        hIceMean=ones,
        SeaIceMassC=900 * ones,
        SeaIceMassU=900 * ones,
        SeaIceMassV=900 * ones,
        SeaIceStrength=100 * ones,
        WindForcingX=0.1 * ones,
        WindForcingY=-0.05 * ones,
    )
    result = IceVelocities(state, settings, constants)
    assert len(result) == 5
    for array in result:
        assert array.dtype == np.dtype(dtype)
        assert np.all(np.isfinite(array))
    assert np.any(np.asarray(result[0]) != 0)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_geometry_inputs_convert_to_initialized_precision(dtype: str) -> None:
    """External float64 metrics and integer masks cannot promote State fields."""
    ones = jnp.ones((6, 7), dtype="float64")
    mask = jnp.ones((6, 7, 2), dtype="int32")
    geometry = OceanGeometry(
        maskT=mask,
        maskU=mask,
        maskV=mask,
        ht=100 * ones,
        coriolis_t=1e-4 * ones,
        dxt=jnp.full((6,), 3.0, dtype="float64"),
        dyt=jnp.full((7,), 4.0, dtype="float64"),
        dxu=jnp.full((6,), 3.0, dtype="float64"),
        dyu=jnp.full((7,), 4.0, dtype="float64"),
        area_t=12 * ones,
        area_u=12 * ones,
        area_v=12 * ones,
    )
    result, _settings, _constants = initialize_from_ocean(geometry, dtype=dtype)
    for array in jax.tree.leaves(result):
        assert array.dtype == np.dtype(dtype)
        assert np.all(np.isfinite(array))


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_state_overrides_and_constants_share_selected_policy(dtype: str) -> None:
    """Mixed external input types convert once and conflicting policies fail."""
    source_dtype = "float64" if dtype == "float32" else "float32"
    temperature = np.full((6, 7), 270.25, dtype=source_dtype)
    state, settings, constants = initialize(
        2,
        3,
        dtype=dtype,
        state_overrides={"theta": temperature, "iceMask": np.ones((6, 7), dtype=int)},
        physical_overrides={
            "rhoIce": np.float64(920),
            "pressReplFac": 0.5,
            "dtype": dtype,
        },
    )
    for array in jax.tree.leaves(state):
        assert array.dtype == np.dtype(dtype)
        assert np.all(np.isfinite(array))
    np.testing.assert_array_equal(state.theta, temperature)
    assert np.asarray(constants.rhoIce).dtype == np.dtype(dtype)
    assert constants.pressReplFac == 0.5
    assert np.asarray(constants.pressReplFac).dtype == np.dtype(dtype)
    assert settings.dtype == constants.dtype == dtype
    with pytest.raises(ValueError, match="physical dtype must match"):
        initialize(2, 3, dtype=dtype, physical_overrides={"dtype": source_dtype})
