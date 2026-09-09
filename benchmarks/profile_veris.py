"""Profile artificial coupled dynamics/growth with synchronized paired trials.

Run with .venv-latest: python -m benchmarks.profile_veris --backend cpu
--output test_logs/profiling/paired-cpu --trace. The baseline is the Python
body of artificial.step; candidate compiles that entire body. Float64 periodic
(nx+4, ny+4) fields include the artificial island and default prescribed forcing.
EVP iteration count is explicit; this is not a convergence benchmark. First-call
latency includes compilation and execution and can share cached inner kernels
between variants. Steady timing excludes validation, initialization and tracing.
"""

import argparse
import importlib.metadata
import json
import os
import platform
import statistics
import subprocess
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import jax
import numpy as np


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Validate a bounded single-device benchmark configuration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cpu", "gpu"), default="cpu")
    for name, default in (
        ("nx", 64),
        ("ny", 64),
        ("evp-steps", 400),
        ("repeats", 12),
        ("warmup", 3),
    ):
        parser.add_argument("--" + name, type=int, default=default)
    parser.add_argument("--mode", choices=("fixed", "evolving"), default="fixed")
    parser.add_argument("--validation", choices=("each", "final"), default="each")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace", action="store_true")
    args = parser.parse_args(argv)
    for name in ("nx", "ny", "evp_steps", "repeats", "warmup"):
        minimum = 4 if name in ("nx", "ny") else 1
        if getattr(args, name) < minimum:
            parser.error(f"{name} must be at least {minimum}")
    return args


def _equivalence(reference: Any, candidate: Any) -> float:
    if jax.tree.structure(reference) != jax.tree.structure(candidate):
        raise ValueError("ERROR numerical equivalence: different output structures")
    maximum = 0.0
    for index, (left, right) in enumerate(
        zip(jax.tree.leaves(reference), jax.tree.leaves(candidate), strict=True)
    ):
        left, right = np.asarray(left), np.asarray(right)
        if left.shape != right.shape or not (
            np.isfinite(left).all() and np.isfinite(right).all()
        ):
            raise ValueError(
                f"ERROR numerical equivalence: leaf {index} shape or nonfinite output"
            )
        error = float(np.max(np.abs(left - right), initial=0))
        maximum = max(maximum, error)
        if not np.allclose(left, right, rtol=1e-10, atol=1e-12):
            raise ValueError(
                f"ERROR numerical equivalence: leaf {index} max_abs_error={error:.6g}"
            )
    return maximum


def _timed(call: Callable[[Any], Any], state: Any) -> tuple[Any, float]:
    start = time.perf_counter()
    output = jax.block_until_ready(call(state))
    return output, (time.perf_counter() - start) * 1000


def measure_pair(
    variants: Mapping[str, Callable[[Any], Any]],
    initial: Any,
    *,
    repeats: int,
    warmup: int,
    evolving: bool,
    validation: str = "each",
) -> dict[str, Any]:
    """Alternate synchronized calls with off-clock output equivalence checks.

    Warmup and first calls always use initial input; evolving trajectories restart
    from initial for measured samples. Each variant owns its independent state.
    Each mode checks every measured pair; final mode checks only first-call and
    final measured outputs, avoiding host reads between pairs. Intermediate
    outputs in final mode are not certified. All checked values must be finite.
    Tolerances allow float64 fusion roundoff.
    """
    if len(variants) != 2 or repeats < 1 or warmup < 1:
        raise ValueError("ERROR require two variants and positive repeats/warmup")
    if validation not in ("each", "final"):
        raise ValueError("ERROR validation must be each or final")
    jax.block_until_ready(initial)
    names = list(variants)
    records: dict[str, Any] = {}
    first = {}
    for name, call in variants.items():
        first[name], elapsed = _timed(call, initial)
        records[name] = {"first_call_ms": elapsed, "samples_ms": []}
    maximum = _equivalence(first[names[0]], first[names[1]])
    for _ in range(warmup):
        for call in variants.values():
            jax.block_until_ready(call(initial))
    states = dict.fromkeys(names, initial)
    orders = []
    for iteration in range(repeats):
        order = names if iteration % 2 == 0 else names[::-1]
        orders.append(order)
        outputs = {}
        for name in order:
            outputs[name], elapsed = _timed(variants[name], states[name])
            records[name]["samples_ms"].append(elapsed)
        if validation == "each" or iteration == repeats - 1:
            maximum = max(maximum, _equivalence(outputs[names[0]], outputs[names[1]]))
        if evolving:
            states = outputs
    for name in names:
        records[name]["median_ms"] = statistics.median(records[name]["samples_ms"])
        records[name]["final_checksum"] = sum(
            float(np.asarray(x).sum()) for x in jax.tree.leaves(outputs[name])
        )
    return {
        "variants": records,
        "pair_order": orders,
        "validation": {
            "passed": True,
            "schedule": validation,
            "comparisons": repeats + 1 if validation == "each" else 2,
            "rtol": 1e-10,
            "atol": 1e-12,
            "max_abs_error": maximum,
        },
        "median_paired_speedup": statistics.median(
            a / b
            for a, b in zip(
                records[names[0]]["samples_ms"],
                records[names[1]]["samples_ms"],
                strict=True,
            )
        ),
    }


def _metadata(
    args: argparse.Namespace, device: jax.Device, sett: Any, phys: Any
) -> dict[str, Any]:
    cpu_model = platform.processor()
    # Read this single kernel metadata file; never scan the host filesystem.
    if Path("/proc/cpuinfo").exists():
        cpu_model = next(
            (
                line.split(":", 1)[1].strip()
                for line in Path("/proc/cpuinfo").read_text().splitlines()
                if line.startswith("model name")
            ),
            cpu_model,
        )
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    versions = {}
    for package in ("jax", "jaxlib", "numpy", "tensorboard", "xprof"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {
        "backend": device.platform,
        "device": str(device),
        "device_kind": device.device_kind,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_model": cpu_model,
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "revision": revision,
        "tracked_dirty": bool(dirty),
        "versions": versions,
        "jax_enable_x64": jax.config.x64_enabled,
        "settings": asdict(sett),
        "physical_constants": asdict(phys),
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "environment": {
            key: os.environ.get(key)
            for key in (
                "XLA_FLAGS",
                "JAX_PLATFORMS",
                "CUDA_VISIBLE_DEVICES",
                "OMP_NUM_THREADS",
                "JAX_COMPILATION_CACHE_DIR",
            )
        },
        "first_call_note": "compile plus execution; baseline first, inner compilation cache may be shared",
    }


def main(argv: list[str] | None = None) -> None:
    """Measure baseline/fused coupled steps and optionally capture both traces."""
    args = parse_args(argv)
    jax.config.update("jax_enable_x64", True)
    devices = jax.devices(args.backend)
    device = devices[0]
    if device.platform != args.backend:
        raise RuntimeError(f"ERROR requested {args.backend}, got {device.platform}")
    from veris.setup import artificial

    with jax.default_device(device):
        initial, sett, phys = artificial.initialize(args.nx, args.ny)
        sett = replace(sett, nEVPsteps=args.evp_steps)
        body = getattr(artificial.step, "__wrapped__", artificial.step)
        fused = getattr(artificial, "compiled_step", None)
        if fused is None:
            # Historical checkouts predate the explicit compiled driver.
            fused = jax.jit(body, static_argnames=["sett", "phys"])
        variants = {
            "baseline": lambda state: body(state, sett, phys),
            "candidate": lambda state: fused(state, sett, phys),
        }
        result = measure_pair(
            variants,
            initial,
            repeats=args.repeats,
            warmup=args.warmup,
            evolving=args.mode == "evolving",
            validation=args.validation,
        )
        result["metadata"] = _metadata(args, device, sett, phys)
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
        if args.trace:
            for name, call in variants.items():
                state = initial
                with jax.profiler.trace(
                    str(args.output / name), create_perfetto_trace=True
                ):
                    for index in range(3):
                        with jax.profiler.StepTraceAnnotation(
                            "coupled_step", step_num=index
                        ):
                            output = jax.block_until_ready(call(state))
                            if args.mode == "evolving":
                                state = output
        print(
            json.dumps(
                {
                    "output": str(args.output),
                    "median_ms": {
                        name: record["median_ms"]
                        for name, record in result["variants"].items()
                    },
                    "paired_speedup": result["median_paired_speedup"],
                    "max_abs_error": result["validation"]["max_abs_error"],
                }
            )
        )


if __name__ == "__main__":
    main()
