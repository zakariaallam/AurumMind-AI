"""Durable job leases with periodic heartbeats and explicit outcomes."""

import threading
from collections.abc import Callable

from .database import Database


def execute_owned_job[T](
    database: Database,
    name: str,
    owner: str,
    operation: Callable[[], T],
    heartbeat_seconds: int = 20,
) -> T:
    """Run an already acquired job and keep its SQLite lease alive."""
    stopped = threading.Event()
    heartbeat_errors: list[Exception] = []

    def keep_alive() -> None:
        while not stopped.wait(heartbeat_seconds):
            try:
                database.heartbeat_job(name, owner)
            except Exception as exc:
                heartbeat_errors.append(exc)
                stopped.set()
                return

    heartbeat_thread = threading.Thread(target=keep_alive, name=f"{name}-heartbeat", daemon=True)
    heartbeat_thread.start()
    try:
        result = operation()
        if heartbeat_errors:
            raise RuntimeError(
                f"Lost database lease for job {name}: {heartbeat_errors[0]}"
            ) from heartbeat_errors[0]
        status = (
            "failed"
            if isinstance(result, dict) and result.get("status") == "failed"
            else "succeeded"
        )
        database.finish_job(name, owner, status, "Job completed.")
        return result
    except Exception as exc:
        database.finish_job(name, owner, "failed", str(exc))
        raise
    finally:
        stopped.set()
        heartbeat_thread.join(timeout=heartbeat_seconds + 1)


def execute_job[T](
    database: Database,
    name: str,
    operation: Callable[[], T],
    lease_minutes: int = 30,
) -> T:
    """Acquire, execute, and release a named exclusive job."""
    owner = database.acquire_job(name, lease_minutes=lease_minutes)
    if owner is None:
        raise RuntimeError(f"Job {name} is already running.")
    return execute_owned_job(database, name, owner, operation)
