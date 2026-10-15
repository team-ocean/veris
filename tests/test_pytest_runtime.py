"""The test-only compilation cache is private, inherited and reliably cleaned up."""

import os
import subprocess
import sys
from pathlib import Path


def test_session_cache_reuses_real_compilations_and_cleans_up_on_failure() -> None:
    """Fresh child processes reuse executable code while still executing numerics."""
    root = Path(__file__).resolve().parents[1]
    script = r'''
import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import jax
from conftest import _temporary_compilation_cache

os.environ.pop("JAX_COMPILATION_CACHE_DIR", None)
jax.config.update("jax_compilation_cache_dir", None)
jax.config.update("jax_enable_compilation_cache", True)
compile_script = """
import json
import sys
import jax
import jax.numpy as jnp
import numpy as np
assert jax.config.jax_compilation_cache_dir == sys.argv[1]
jax.config.update('jax_enable_compilation_cache', True)
jax.config.update('jax_compilation_cache_check_contents', False)
jax.config.update('jax_compilation_cache_max_size', -1)
jax.config.update('jax_persistent_cache_min_entry_size_bytes', -1)
jax.config.update('jax_persistent_cache_min_compile_time_secs', 0)
events = []
jax.monitoring.register_event_listener(lambda event, **kwargs: events.append(event))
@jax.jit
def cached(values):
    return jnp.sin(values) + values * values
values = np.arange(16, dtype=np.float32) + int(sys.argv[2])
result = np.asarray(cached(values))
np.testing.assert_allclose(result, np.sin(values) + values * values, rtol=1e-6)
print(json.dumps({'hits': events.count('/jax/compilation_cache/cache_hits'), 'sum': float(result.sum())}))
"""
with _temporary_compilation_cache():
    path = Path(os.environ["JAX_COMPILATION_CACHE_DIR"])
    assert path.is_dir() and stat.S_IMODE(path.stat().st_mode) == 0o700
    assert jax.config.jax_compilation_cache_dir == str(path)
    runs = []
    for offset in range(2):
        result = subprocess.run(
            [sys.executable, "-c", compile_script, str(path), str(offset)],
            capture_output=True, text=True, check=True,
        )
        runs.append(json.loads(result.stdout))
    assert runs[0]['sum'] != runs[1]['sum']
    assert runs[0]['hits'] == 0 and runs[1]['hits'] >= 1, runs
assert not path.exists()
assert "JAX_COMPILATION_CACHE_DIR" not in os.environ
assert jax.config.jax_compilation_cache_dir is None
try:
    with _temporary_compilation_cache():
        failed_path = Path(os.environ["JAX_COMPILATION_CACHE_DIR"])
        raise RuntimeError("test failure")
except RuntimeError:
    pass
assert not failed_path.exists()
assert "JAX_COMPILATION_CACHE_DIR" not in os.environ
assert jax.config.jax_compilation_cache_dir is None
print("real cache reuse and cleanup passed")
'''
    environment = dict(
        os.environ,
        JAX_PLATFORMS="cpu",
        PYTHONPATH=os.pathsep.join((str(root / "tests"), str(root))),
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"ERROR cache lifecycle: {result.stderr[-2000:]}"
    assert "real cache reuse and cleanup passed" in result.stdout


def test_session_cache_honors_explicit_directory_and_disabled_cache() -> None:
    """Caller cache settings and files must never be replaced or removed."""
    root = Path(__file__).resolve().parents[1]
    script = r"""
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import jax
from conftest import _temporary_compilation_cache

with TemporaryDirectory() as external:
    marker = Path(external) / "owned-by-caller"
    marker.write_text("preserve")
    os.environ["JAX_COMPILATION_CACHE_DIR"] = external
    jax.config.update("jax_compilation_cache_dir", external)
    with _temporary_compilation_cache():
        assert jax.config.jax_compilation_cache_dir == external
        assert os.environ["JAX_COMPILATION_CACHE_DIR"] == external
    assert marker.read_text() == "preserve"
    assert jax.config.jax_compilation_cache_dir == external
    assert os.environ["JAX_COMPILATION_CACHE_DIR"] == external
os.environ.pop("JAX_COMPILATION_CACHE_DIR")
jax.config.update("jax_compilation_cache_dir", None)
jax.config.update("jax_enable_compilation_cache", False)
with _temporary_compilation_cache():
    assert jax.config.jax_compilation_cache_dir is None
    assert "JAX_COMPILATION_CACHE_DIR" not in os.environ
assert not jax.config.jax_enable_compilation_cache
print("caller cache settings preserved")
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=dict(
            os.environ,
            JAX_PLATFORMS="cpu",
            PYTHONPATH=os.pathsep.join((str(root / "tests"), str(root))),
        ),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"ERROR caller cache settings: {result.stderr[-2000:]}"
    )
    assert "caller cache settings preserved" in result.stdout


def test_session_cache_inherits_subsecond_policy_and_restores_settings() -> None:
    """Cold sessions reuse small compilations without overriding caller policy."""
    root = Path(__file__).resolve().parents[1]
    script = r"""
import os
import jax
from conftest import _temporary_compilation_cache

os.environ.pop("JAX_COMPILATION_CACHE_DIR", None)
os.environ.pop("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", None)
jax.config.update("jax_compilation_cache_dir", None)
jax.config.update("jax_enable_compilation_cache", True)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)
with _temporary_compilation_cache():
    assert jax.config.jax_persistent_cache_min_compile_time_secs == 0.0
    assert os.environ["JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS"] == "0"
assert jax.config.jax_persistent_cache_min_compile_time_secs == 1.0
assert "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS" not in os.environ
try:
    with _temporary_compilation_cache():
        raise RuntimeError("interrupted session")
except RuntimeError:
    pass
assert jax.config.jax_persistent_cache_min_compile_time_secs == 1.0
assert "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS" not in os.environ
os.environ["JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS"] = "0.25"
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.25)
with _temporary_compilation_cache():
    assert jax.config.jax_persistent_cache_min_compile_time_secs == 0.25
    assert os.environ["JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS"] == "0.25"
assert jax.config.jax_persistent_cache_min_compile_time_secs == 0.25
assert os.environ["JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS"] == "0.25"
os.environ.pop("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS")
with _temporary_compilation_cache():
    assert jax.config.jax_persistent_cache_min_compile_time_secs == 0.25
    assert "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS" not in os.environ
print("subsecond policy and caller settings preserved")
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=dict(
            os.environ,
            JAX_PLATFORMS="cpu",
            PYTHONPATH=os.pathsep.join((str(root / "tests"), str(root))),
        ),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"ERROR subsecond cache policy: {result.stderr[-2000:]}"
    )
    assert "subsecond policy and caller settings preserved" in result.stdout
