"""Coordinate real child probes without imposing execution deadlines.

Launchers retain ownership of child output and failure cleanup. Waiting watches
all ranks so an unsuccessful peer cannot strand the parent behind a live rank.
"""

import subprocess
import time
from collections.abc import Sequence
from typing import Any


def wait_for_processes(processes: Sequence[subprocess.Popen[Any]]) -> None:
    """Wait for all children to succeed, reporting any failed rank immediately."""
    while True:
        statuses = [process.poll() for process in processes]
        for rank, status in enumerate(statuses):
            if status not in (None, 0):
                raise ChildProcessError(f"ERROR rank{rank} exited with status {status}")
        if all(status == 0 for status in statuses):
            return
        time.sleep(0.1)
