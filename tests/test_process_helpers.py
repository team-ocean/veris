"""Child-process coordination reports failed peers without execution deadlines."""

import subprocess
import sys
from pathlib import Path

import pytest


def test_wait_for_processes_preserves_successful_child_results(tmp_path: Path) -> None:
    from tests.process_helpers import wait_for_processes

    marker = tmp_path / "completed"
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "from pathlib import Path; import sys; Path(sys.argv[1]).write_text('done')",
            str(marker),
        ]
    )
    try:
        wait_for_processes([process])
        assert process.returncode == 0 and marker.read_text() == "done"
    finally:
        if process.poll() is None:
            process.kill()
        process.wait()


def test_wait_for_processes_reports_failed_peer_before_blocking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.process_helpers import wait_for_processes

    pending = subprocess.Popen(
        [sys.executable, "-c", "import sys; sys.stdin.read()"],
        stdin=subprocess.PIPE,
    )
    original_wait = pending.wait
    failed = None

    def guarded_wait() -> int:
        assert pending.poll() is not None, "ERROR blocking wait hid a failed peer"
        return original_wait()

    try:
        failed = subprocess.Popen([sys.executable, "-c", "raise SystemExit(23)"])
        failed.wait()
        monkeypatch.setattr(pending, "wait", guarded_wait)
        with pytest.raises(ChildProcessError, match="rank1.*23"):
            wait_for_processes([pending, failed])
    finally:
        if pending.poll() is None:
            pending.kill()
        original_wait()
        assert pending.stdin is not None
        pending.stdin.close()
        if failed is not None:
            if failed.poll() is None:
                failed.kill()
            failed.wait()
