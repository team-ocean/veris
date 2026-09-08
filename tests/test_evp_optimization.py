"""Pre-optimization EVP value and AD oracles on a nonuniform rectangular grid.

The fixture records the original jax-only equations before insertion of the CPU
fusion barrier. Masked values exercise a coastline; derivatives use open ocean
with nonzero velocity and strain to avoid the model's zero-norm singularities.
Run this file directly only on the provenance source revision to regenerate.
"""

import ast
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.typing import NDArray

import veris
from veris.setup.artificial import initialize
from veris.state import Settings, State

REFERENCE = Path(__file__).parent / "reference_data" / "evp_pre_barrier.npz"
CASES = [
    (False, False, 1),
    (False, True, 4),
    (True, False, 4),
    (True, True, 4),
    (False, False, 400),
    (True, False, 400),
]


def oracle_state(masked: bool) -> tuple[State, Settings]:
    """Construct periodic 6x9 interiors with variable ice, stress and forcing."""
    vs, sett = initialize(6, 9)
    x, y = np.meshgrid(np.arange(6), np.arange(9), indexing="ij")

    def field(values: np.ndarray) -> jax.Array:
        return jnp.asarray(np.pad(values, 2, mode="wrap"), dtype=jnp.float64)

    a = field(np.sin(2 * np.pi * x / 6) + 0.3 * np.cos(2 * np.pi * y / 9))
    b = field(np.cos(2 * np.pi * (x / 6 + y / 9)))
    interior = np.ones((6, 9))
    if masked:
        interior[2, 4] = 0
    mask = field(interior)
    west = mask * jnp.roll(mask, 1, 0)
    south = mask * jnp.roll(mask, 1, 1)
    thickness = (1.1 + 0.12 * a) * mask
    area = (0.86 + 0.025 * b) * mask
    vs = vs._replace(
        maskInC=mask,
        iceMask=mask,
        maskInU=west,
        iceMaskU=west,
        maskInV=south,
        iceMaskV=south,
        hIceMean=thickness,
        Area=area,
        hSnowMean=0.06 * mask,
        uIce=(0.09 + 0.012 * a) * west,
        vIce=(0.06 + 0.013 * b) * south,
        uOcean=-0.03 + 0.003 * b,
        vOcean=-0.02 + 0.004 * a,
        sigma1=(-700 + 35 * a) * mask,
        sigma2=(80 + 12 * b) * mask,
        sigma12=(25 + 3 * a) * mask,
        WindForcingX=(0.08 + 0.006 * b) * west,
        WindForcingY=(-0.04 + 0.007 * a) * south,
        R_low=-8 - a,
        fCori=1e-4 * jnp.where(b > 0, 1, -1),
    )
    from veris.area_mass import AreaWS, SeaIceMass
    from veris.dynamics_routines import SeaIceStrength

    mass_c, mass_u, mass_v = SeaIceMass(vs, sett)
    area_w, area_s = AreaWS(vs, sett)
    return vs._replace(
        SeaIceMassC=mass_c,
        SeaIceMassU=mass_u,
        SeaIceMassV=mass_v,
        AreaW=area_w,
        AreaS=area_s,
        SeaIceStrength=SeaIceStrength(vs, sett),
    ), sett


def evaluate(
    case: tuple[bool, bool, int], masked: bool
) -> dict[str, NDArray[np.float64]]:
    """Return scaled field outputs and derivatives with respect to two forcings."""
    vs, sett = oracle_state(masked)
    from veris.evp_solver import evp_solver

    adaptive, no_slip, steps = case
    sett = sett._replace(useAdaptiveEVP=adaptive, noSlip=no_slip, nEVPsteps=steps)

    def solve(parameters: jax.Array) -> jax.Array:
        state = vs._replace(
            WindForcingX=parameters[0] * vs.WindForcingX,
            SeaIceStrength=parameters[1] * vs.SeaIceStrength,
        )
        result = evp_solver(state, sett)
        # Scale stresses so a single tolerance resolves velocity and stress alike.
        return jnp.stack(result) / jnp.array([1, 1, 1000, 1000, 1000])[:, None, None]

    parameters = jnp.array([1.07, 0.93], dtype=jnp.float64)
    if masked:
        return {"value": np.asarray(solve(parameters))}
    value, tangent = jax.jvp(solve, (parameters,), (jnp.array([0.7, -0.2]),))
    _, pullback = jax.vjp(solve, parameters)
    weights = jnp.linspace(-0.5, 0.7, value.size).reshape(value.shape)
    epsilon = 1e-4
    direction = jnp.array([0.7, -0.2])
    finite_jvp = (
        solve(parameters + epsilon * direction)
        - solve(parameters - epsilon * direction)
    ) / (2 * epsilon)
    finite_vjp = jnp.stack(
        [
            jnp.sum(
                weights
                * (
                    solve(parameters + epsilon * axis)
                    - solve(parameters - epsilon * axis)
                )
            )
            / (2 * epsilon)
            for axis in jnp.eye(2)
        ]
    )
    np.testing.assert_allclose(
        tangent,
        finite_jvp,
        rtol=3e-6,
        atol=2e-10,
        err_msg="ERROR EVP JVP finite-difference mismatch",
    )
    np.testing.assert_allclose(
        pullback(weights)[0],
        finite_vjp,
        rtol=3e-6,
        atol=2e-10,
        err_msg="ERROR EVP VJP finite-difference mismatch",
    )
    return {
        "value": np.asarray(value),
        "jvp": np.asarray(tangent),
        "vjp": np.asarray(pullback(weights)[0]),
    }


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("masked", [False, True], ids=["smooth_ad", "coastline"])
def test_evp_matches_pre_optimization_values_and_derivatives(
    case: tuple[bool, bool, int], masked: bool
) -> None:
    """Keep every velocity/stress cell and both AD directions at baseline values."""
    index = CASES.index(case)
    with np.load(REFERENCE) as reference:
        for quantity, actual in evaluate(case, masked).items():
            expected = reference[f"case{index}_masked{int(masked)}_{quantity}"]
            assert np.all(np.isfinite(actual)), f"ERROR nonfinite EVP {quantity}"
            np.testing.assert_allclose(
                actual,
                expected,
                rtol=2e-10,
                atol=2e-12,
                err_msg=f"ERROR EVP pre-optimization {quantity} mismatch",
            )


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)
    # Resolve the actually imported package: PYTHONPATH can select an immutable
    # reference checkout while the test and output fixture stay in this tree.
    root = Path(veris.__file__).resolve().parents[1]
    source = root / "veris/evp_solver.py"
    if "optimization_barrier" in source.read_text():
        raise RuntimeError("ERROR generate the oracle only before barrier insertion")
    provenance = json.loads(REFERENCE.with_suffix(".json").read_text())
    for name, expected_hash in provenance["source_sha256"].items():
        actual_hash = hashlib.sha256((root / name).read_bytes()).hexdigest()
        if actual_hash != expected_hash:
            raise RuntimeError(f"ERROR oracle source hash mismatch: {name}")
    setup_source = (root / "veris/setup/artificial.py").read_text()
    initialize_node = next(
        node
        for node in ast.parse(setup_source).body
        if isinstance(node, ast.FunctionDef) and node.name == "initialize"
    )
    initialize_source = ast.get_source_segment(setup_source, initialize_node)
    assert initialize_source is not None
    initialize_hash = hashlib.sha256(initialize_source.encode()).hexdigest()
    if initialize_hash != provenance["initialize_sha256"]:
        raise RuntimeError("ERROR oracle initialize source hash mismatch")
    arrays = {}
    for index, case in enumerate(CASES):
        for masked in (False, True):
            for quantity, values in evaluate(case, masked).items():
                if not np.isfinite(values).all():
                    raise RuntimeError(f"ERROR nonfinite baseline: {case} {quantity}")
                arrays[f"case{index}_masked{int(masked)}_{quantity}"] = values
    REFERENCE.parent.mkdir(exist_ok=True)
    np.savez_compressed(REFERENCE, allow_pickle=False, **arrays)
    metadata = {
        "git_commit": provenance["git_commit"],
        "jax": jax.__version__,
        "backend": jax.default_backend(),
        "precision": "float64",
        "initialize_sha256": initialize_hash,
        "cases": CASES,
        "source_sha256": {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in provenance["source_sha256"]
        },
    }
    REFERENCE.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Captured {len(arrays)} finite oracle arrays in {REFERENCE}")
