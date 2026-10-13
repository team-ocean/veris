"""Reference column initialization and recursive growth-only driver checks."""

from dataclasses import fields, replace
from pathlib import Path

import jax
import numpy as np
import pytest

from veris import Experiment, initialize_experiment
from veris.execution.advance import advance
from veris.io import read_record
from veris.model.diagnostics import Diagnostics
from veris.model.types import State
from veris.physics.thermodynamics.growth import Growth


def test_growth_case_matches_reference_initial_column() -> None:
    """Match the standalone reference initial thermodynamic column."""

    state, conf, phys = initialize_experiment(Experiment.builtin("growth"))
    assert isinstance(state, State)
    assert (conf.nx, conf.ny) == (2, 2)
    assert not conf.use_sharding
    assert conf.deltatTherm == 86400.0
    expected = {
        "hIceMean": 1.3,
        "hSnowMean": 0.1,
        "Area": 0.9,
        "TSurf": 273.0,
        "wSpeed": 2.0,
        "ocSalt": 29.0,
        "theta": phys.celsius2K - 1.66,
        "Qnet": 173.03212617345582,
        "Qsw": 0,
        "SWdown": 0,
        "LWdown": 80,
        "ATemp": 253,
        "precip": 0,
        "aqh": 0,
        "SeaIceLoad": phys.rhoIce * 1.3 + phys.rhoSnow * 0.1,
    }
    for name, value in expected.items():
        array = getattr(state, name)
        assert array.shape == (6, 6)
        np.testing.assert_allclose(array, value, rtol=1e-14, atol=1e-14)
    for name in ("maskInC", "maskInU", "maskInV", "iceMask", "iceMaskU", "iceMaskV"):
        np.testing.assert_array_equal(getattr(state, name), 1)
    # initialize_growth.py explicitly zeroes every unprescribed field.
    for name in ("dxG", "dyG", "recip_dxC", "recip_dyC", "rAz", "recip_rA"):
        np.testing.assert_array_equal(getattr(state, name), 0)


def test_growth_driver_preserves_recursive_fluxes_and_separate_diagnostics() -> None:
    """Retain recursive coupling fluxes without adding diagnostic State leaves."""

    state, conf, phys = initialize_experiment(Experiment.builtin("growth"))
    names = [
        "hIceMean",
        "hSnowMean",
        "Area",
        "TSurf",
        "EmPmR",
        "forc_salt_surface",
        "Qsw",
        "Qnet",
        "SeaIceLoad",
        "IcePenetSW",
        "recip_hIceMean",
    ]
    compiled_advance = jax.jit(
        advance, static_argnames=("configuration", "constants", "physics")
    )
    for _ in range(2):
        expected = dict(zip(names, Growth(state, conf, phys), strict=True))
        actual, diagnostics = advance(state, conf, phys, physics="thermodynamics")
        assert isinstance(diagnostics, Diagnostics)
        for name, value in expected.items():
            owner = actual if hasattr(actual, name) else diagnostics
            np.testing.assert_allclose(
                getattr(owner, name), value, rtol=1e-13, atol=1e-13
            )
        np.testing.assert_array_equal(diagnostics.OceanStressU, 0)
        np.testing.assert_array_equal(diagnostics.OceanStressV, 0)
        compiled = compiled_advance(state, conf, phys, physics="thermodynamics")[0]
        for leaf, reference in zip(
            jax.tree.leaves(compiled), jax.tree.leaves(actual), strict=True
        ):
            np.testing.assert_allclose(leaf, reference, rtol=1e-13, atol=1e-13)
        assert not hasattr(actual, "EmPmR")
        state = actual


def test_growth_cli_records_physical_final_state(tmp_path: Path) -> None:
    """Save the completed State without storage halos in netCDF."""
    from veris.setups import run_growth

    path = tmp_path / "growth.nc"
    run_growth.main(["--steps", "2", "--backend", "cpu", "--output", str(path)])
    initial, conf, phys = initialize_experiment(Experiment.builtin("growth"))
    first = advance(initial, conf, phys, physics="thermodynamics")[0]
    final = advance(first, conf, phys, physics="thermodynamics")[0]
    saved = read_record(path)
    assert saved.time == 2 * conf.deltatTherm
    for field in fields(State):
        np.testing.assert_allclose(
            saved.fields[field.name],
            getattr(final, field.name)[2:-2, 2:-2],
            rtol=1e-12,
            atol=1e-12,
        )


def test_growth_cli_rejects_negative_steps(tmp_path: Path) -> None:
    """Reject an invalid iteration count without producing output."""
    from veris.setups import run_growth

    with pytest.raises(SystemExit):
        run_growth.main(["--steps", "-1", "--output", str(tmp_path / "bad.nc")])


def test_growth_initialize_accepts_model_overrides() -> None:
    """Apply physical overrides when deriving the initial ice load."""

    state, conf, phys = initialize_experiment(
        Experiment.builtin(
            "growth", settings={"deltatTherm": 600.0}, constants={"rhoIce": 920.0}
        )
    )
    assert conf.deltatTherm == 600.0
    np.testing.assert_allclose(state.SeaIceLoad, phys.rhoIce * 1.3 + phys.rhoSnow * 0.1)


def test_growth_cli_zero_steps_saves_initial_column(tmp_path: Path) -> None:
    """Permit initialization-only output and create missing parent directories."""
    from veris.setups import run_growth

    path = tmp_path / "nested" / "outputs" / "initial.nc"
    run_growth.main(["--steps", "0", "--backend", "cpu", "--output", str(path)])
    saved = read_record(path)
    assert saved.time == 0
    assert saved.fields["hIceMean"].shape == (2, 2)
    np.testing.assert_allclose(saved.fields["hIceMean"], 1.3)


def test_growth_cli_rejects_nonfinite_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reject a nonfinite column before writing its output archive."""
    from veris.setups import run_growth

    state, conf, phys = initialize_experiment(Experiment.builtin("growth"))
    invalid = replace(state, hIceMean=state.hIceMean.at[2, 2].set(np.nan))
    from veris.execution import simulation

    monkeypatch.setattr(
        simulation,
        "initialize_experiment",
        lambda *args, **kwargs: (invalid, conf, phys),
    )
    path = tmp_path / "invalid.nc"
    with pytest.raises(ValueError, match="nonfinite"):
        run_growth.main(["--steps", "1", "--output", str(path)])
    assert not path.exists()


def test_growth_compiled_advance_longwave_gradient_matches_finite_difference() -> None:
    """Differentiate the reference column's ice response to longwave forcing."""

    state, conf, phys = initialize_experiment(Experiment.builtin("growth"))

    compiled_advance = jax.jit(
        advance, static_argnames=("configuration", "constants", "physics")
    )

    def final_ice(longwave: jax.Array | float) -> jax.Array:
        """Return interior mean ice after a uniformly perturbed forcing step."""
        forced = replace(state, LWdown=state.LWdown * 0 + longwave)
        result = compiled_advance(forced, conf, phys, physics="thermodynamics")[0]
        return result.hIceMean[2:-2, 2:-2].mean()

    longwave = float(state.LWdown[2, 2])
    delta = 1e-2
    derivative = jax.grad(final_ice)(longwave)
    finite_difference = (final_ice(longwave + delta) - final_ice(longwave - delta)) / (
        2 * delta
    )
    assert np.isfinite(derivative)
    assert abs(float(derivative)) > 1e-8
    np.testing.assert_allclose(derivative, finite_difference, rtol=1e-6, atol=1e-10)
