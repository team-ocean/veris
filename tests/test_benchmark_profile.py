"""Contract checks for synchronized paired profiling with real tiny JAX work."""

import importlib
import json
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import jax
import jax.numpy as jnp
import pytest


def harness() -> ModuleType:
    """Load the standalone harness only when a test runs."""
    return importlib.import_module("benchmarks.profile_veris")


@pytest.mark.parametrize("evolving,expected", [(False, 2.0), (True, 5.0)])
def test_pair_uses_independent_trajectories_and_alternates(
    evolving: bool, expected: float
) -> None:
    profile = harness()
    calls = []
    compiled = jax.jit(lambda x: {"x": x["x"] + 1})

    def variant(name: str) -> Callable[[dict[str, jax.Array]], dict[str, jax.Array]]:
        def apply(state: dict[str, jax.Array]) -> dict[str, jax.Array]:
            calls.append(name)
            return compiled(state)

        return apply

    result = profile.measure_pair(
        {name: variant(name) for name in ("baseline", "candidate")},
        {"x": jnp.ones(3)},
        repeats=4,
        warmup=1,
        evolving=evolving,
    )
    assert calls[-8:] == ["baseline", "candidate", "candidate", "baseline"] * 2
    assert result["validation"]["max_abs_error"] == 0
    assert result["validation"]["schedule"] == "each"
    assert result["validation"]["comparisons"] == 5
    for name in ("baseline", "candidate"):
        assert len(result["variants"][name]["samples_ms"]) == 4
        assert result["variants"][name]["first_call_ms"] > 0
        assert result["variants"][name]["final_checksum"] == expected * 3


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), 0.001])
def test_pair_rejects_nonfinite_or_mismatched_outputs(bad: float) -> None:
    profile = harness()
    with pytest.raises(ValueError, match="ERROR.*equivalence"):
        profile.measure_pair(
            {"baseline": jax.jit(lambda x: x), "candidate": jax.jit(lambda x: x + bad)},
            jnp.ones(2),
            repeats=2,
            warmup=1,
            evolving=False,
        )


@pytest.mark.parametrize(
    "option,value",
    [
        ("--nx", "3"),
        ("--ny", "0"),
        ("--evp-steps", "0"),
        ("--repeats", "0"),
        ("--warmup", "0"),
        ("--backend", "tpu"),
        ("--validation", "unknown"),
    ],
)
def test_cli_rejects_invalid_workload(option: str, value: str, tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        harness().parse_args(["--output", str(tmp_path), option, value])


def test_cli_writes_metadata_and_separate_real_traces(tmp_path: Path) -> None:
    profile = harness()
    profile.main(
        [
            "--backend",
            "cpu",
            "--nx",
            "4",
            "--ny",
            "5",
            "--evp-steps",
            "1",
            "--repeats",
            "1",
            "--warmup",
            "1",
            "--trace",
            "--validation",
            "final",
            "--output",
            str(tmp_path),
        ]
    )
    data = json.loads((tmp_path / "results.json").read_text())
    assert data["metadata"]["backend"] == "cpu"
    assert data["metadata"]["jax_enable_x64"] is True
    assert data["metadata"]["settings"]["nEVPsteps"] == 1
    assert data["metadata"]["revision"]
    assert data["metadata"]["cpu_affinity"]
    assert data["validation"]["passed"] is True
    assert data["validation"]["schedule"] == "final"
    assert data["metadata"]["arguments"]["validation"] == "final"
    for variant in ("baseline", "candidate"):
        assert list((tmp_path / variant).rglob("*.xplane.pb"))
        assert list((tmp_path / variant).rglob("perfetto_trace.json.gz"))


def test_final_validation_allows_intermediate_difference_that_recovers() -> None:
    """Final-only validation deliberately cannot certify intermediate outputs."""
    profile = harness()
    variants = {
        "baseline": jax.jit(
            lambda x: jnp.where(x < 1, 1.0, jnp.where(x < 2, 2.0, 4.0))
        ),
        "candidate": jax.jit(
            lambda x: jnp.where(x < 1, 1.0, jnp.where(x < 2, 3.0, 4.0))
        ),
    }
    result = profile.measure_pair(
        variants,
        jnp.zeros(2),
        repeats=3,
        warmup=1,
        evolving=True,
        validation="final",
    )
    assert result["validation"]["schedule"] == "final"
    assert result["validation"]["comparisons"] == 2
    assert result["validation"]["max_abs_error"] == 0.0
    assert result["variants"]["candidate"]["final_checksum"] == 8.0
    with pytest.raises(ValueError, match="ERROR.*equivalence"):
        profile.measure_pair(
            variants,
            jnp.zeros(2),
            repeats=3,
            warmup=1,
            evolving=True,
            validation="each",
        )


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), 0.001])
def test_final_validation_rejects_final_mismatch(bad: float) -> None:
    """An initially equivalent candidate must still pass the final comparison."""
    with pytest.raises(ValueError, match="ERROR.*equivalence"):
        harness().measure_pair(
            {
                "baseline": jax.jit(lambda x: x + 1),
                "candidate": jax.jit(lambda x: jnp.where(x == 0, 1.0, x + 1 + bad)),
            },
            jnp.zeros(2),
            repeats=2,
            warmup=1,
            evolving=True,
            validation="final",
        )


def test_pair_rejects_unknown_validation_schedule() -> None:
    """Direct callers receive the same schedule validation as CLI users."""
    identity = jax.jit(lambda x: x)
    with pytest.raises(ValueError, match="ERROR.*validation"):
        harness().measure_pair(
            {"baseline": identity, "candidate": identity},
            jnp.ones(2),
            repeats=1,
            warmup=1,
            evolving=False,
            validation="unknown",
        )
