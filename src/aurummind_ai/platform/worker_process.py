"""Safe control of the local scheduled worker from the dashboard."""

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from .config import PROJECT_ROOT
from .database import Database

WORKER_MARKER = "aurummind_ai.cli worker"


def _process_matches(pid: int) -> bool:
    """Confirm a stored PID still belongs to the AurumMind worker."""
    if not Path("/proc").is_dir():
        raise RuntimeError("Dashboard worker controls currently require Linux /proc.")
    if pid <= 1:
        return False
    try:
        command = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ")
    except FileNotFoundError:
        return False
    return WORKER_MARKER.encode() in command


def worker_status(database: Database) -> dict[str, int | str | None]:
    """Report worker PID state without trusting a stale PID record."""
    stored = database.get_state("worker_pid")
    if not stored:
        return {"status": "stopped", "pid": None}
    try:
        pid = int(stored)
    except ValueError:
        database.upsert_state("worker_pid", "")
        return {"status": "stopped", "pid": None}
    if _process_matches(pid):
        return {"status": "running", "pid": pid}
    database.upsert_state("worker_pid", "")
    return {"status": "stopped", "pid": None}


def start_worker(database: Database) -> dict[str, int | str | None]:
    """Launch the configured polling worker once; never duplicates a live PID."""
    status = worker_status(database)
    if status["status"] == "running":
        return status
    log_path = PROJECT_ROOT / "logs" / "worker.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_path.open("a", encoding="utf-8")
    try:
        process = subprocess.Popen(
            [sys.executable, "-m", "aurummind_ai.cli", "worker"],
            cwd=PROJECT_ROOT,
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            close_fds=True,
            start_new_session=True,
        )
    except Exception:
        log_handle.close()
        raise
    log_handle.close()
    if process.poll() is not None:
        raise RuntimeError(
            f"Worker exited during startup with status {process.returncode}; "
            f"inspect {log_path} for details."
        )
    database.upsert_state("worker_pid", str(process.pid))
    return {"status": "running", "pid": process.pid}


def stop_worker(database: Database) -> dict[str, int | str | None]:
    """Stop only a PID verified as this application's CLI worker."""
    status = worker_status(database)
    pid = status["pid"]
    if not isinstance(pid, int):
        return status
    if not _process_matches(pid):
        raise RuntimeError("Stored worker PID no longer identifies the AurumMind worker.")
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        database.upsert_state("worker_pid", "")
        return {"status": "stopped", "pid": None}
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and _process_matches(pid):
        time.sleep(0.1)
    if _process_matches(pid):
        return {"status": "stopping", "pid": pid}
    database.upsert_state("worker_pid", "")
    return {"status": "stopped", "pid": None}
