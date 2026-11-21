"""SQLite persistence for market data provenance and model operations."""

import json
import sqlite3
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from .config import load_config, resolve_path

MIGRATIONS: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS data_downloads (
        id TEXT PRIMARY KEY,
        provider TEXT NOT NULL,
        instrument TEXT NOT NULL,
        timeframe TEXT NOT NULL,
        requested_start TEXT NOT NULL,
        requested_end TEXT NOT NULL,
        downloaded_at TEXT NOT NULL,
        candle_count INTEGER NOT NULL,
        source_file TEXT NOT NULL,
        volume_semantics TEXT NOT NULL,
        quality_score REAL,
        metadata_json TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS models (
        version TEXT PRIMARY KEY,
        model_name TEXT NOT NULL,
        artifact_path TEXT NOT NULL,
        registered_at TEXT NOT NULL,
        validation_macro_f1 REAL NOT NULL,
        validation_brier REAL NOT NULL,
        metadata_json TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS system_state (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS predictions (
        id TEXT PRIMARY KEY,
        prediction_timestamp TEXT NOT NULL,
        input_candle_timestamp TEXT NOT NULL,
        target_timestamp TEXT NOT NULL,
        instrument TEXT NOT NULL,
        timeframe TEXT NOT NULL,
        model_version TEXT NOT NULL,
        predicted_class TEXT NOT NULL,
        probabilities_json TEXT NOT NULL,
        provider TEXT NOT NULL,
        data_quality TEXT NOT NULL,
        actual_return REAL,
        actual_class TEXT,
        is_correct INTEGER,
        evaluated_at TEXT
    );
    CREATE TABLE IF NOT EXISTS jobs (
        name TEXT PRIMARY KEY,
        status TEXT NOT NULL,
        owner TEXT,
        started_at TEXT,
        heartbeat_at TEXT,
        finished_at TEXT,
        message TEXT
    );
    CREATE TABLE IF NOT EXISTS job_history (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        status TEXT NOT NULL,
        started_at TEXT NOT NULL,
        finished_at TEXT,
        message TEXT
    );
    """,
)
MIGRATIONS += (
    """
    CREATE INDEX IF NOT EXISTS predictions_timestamp_idx
        ON predictions(prediction_timestamp);
    CREATE UNIQUE INDEX IF NOT EXISTS predictions_model_input_idx
        ON predictions(input_candle_timestamp, timeframe, model_version);
    CREATE INDEX IF NOT EXISTS downloads_provider_time_idx
        ON data_downloads(provider, timeframe, downloaded_at);
    """,
)


def utc_now() -> str:
    """Return a timezone-aware ISO timestamp."""
    return datetime.now(UTC).isoformat()


class Database:
    """Small SQLite repository; SQL storage stays independent from the UI."""

    def __init__(self, path: str | Path | None = None) -> None:
        if path is None:
            path = load_config()["application"]["database_path"]
        self.path = resolve_path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.migrate()

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    def migrate(self) -> None:
        """Apply idempotent schema migrations."""
        with closing(self.connect()) as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS schema_migrations "
                "(version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)"
            )
            applied = {
                row["version"]
                for row in connection.execute("SELECT version FROM schema_migrations")
            }
            for version, migration in enumerate(MIGRATIONS, start=1):
                if version in applied:
                    continue
                connection.executescript(migration)
                connection.execute(
                    "INSERT INTO schema_migrations(version, applied_at) VALUES (?, ?)",
                    (version, utc_now()),
                )

    def record_download(self, metadata: dict[str, Any]) -> str:
        """Persist provider and file provenance for a raw data batch."""
        download_id = str(uuid4())
        with closing(self.connect()) as connection:
            connection.execute(
                """INSERT INTO data_downloads (
                    id, provider, instrument, timeframe, requested_start,
                    requested_end, downloaded_at, candle_count, source_file,
                    volume_semantics, quality_score, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    download_id,
                    metadata["provider"],
                    metadata["instrument"],
                    metadata["timeframe"],
                    metadata["requested_start"],
                    metadata["requested_end"],
                    metadata["downloaded_at"],
                    metadata["candle_count"],
                    metadata["source_file"],
                    metadata["volume_semantics"],
                    metadata.get("quality_score"),
                    json.dumps(metadata, sort_keys=True),
                ),
            )
        return download_id

    def upsert_state(self, key: str, value: str) -> None:
        """Persist one system-state value."""
        with closing(self.connect()) as connection:
            connection.execute(
                """INSERT INTO system_state(key, value, updated_at) VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET value=excluded.value,
                updated_at=excluded.updated_at""",
                (key, value, utc_now()),
            )

    def set_champion(self, version: str, previous_version: str | None) -> None:
        """Atomically persist current and previous champion versions."""
        now = utc_now()
        with closing(self.connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            for key, value in (
                ("champion_version", version),
                ("champion_previous_version", previous_version or ""),
            ):
                connection.execute(
                    """INSERT INTO system_state(key, value, updated_at)
                    VALUES (?, ?, ?) ON CONFLICT(key) DO UPDATE SET
                    value=excluded.value, updated_at=excluded.updated_at""",
                    (key, value, now),
                )
            connection.commit()

    def get_state(self, key: str) -> str | None:
        """Read a system-state value."""
        with closing(self.connect()) as connection:
            row = connection.execute(
                "SELECT value FROM system_state WHERE key = ?", (key,)
            ).fetchone()
        return str(row["value"]) if row else None

    def create_prediction(self, prediction: dict[str, Any]) -> str:
        """Store an immutable forecast awaiting its outcome."""
        prediction_id = str(uuid4())
        with closing(self.connect()) as connection:
            try:
                connection.execute(
                    """INSERT INTO predictions (
                    id, prediction_timestamp, input_candle_timestamp,
                    target_timestamp, instrument, timeframe, model_version,
                    predicted_class, probabilities_json, provider, data_quality
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        prediction_id,
                        prediction["prediction_timestamp"],
                        prediction["input_candle_timestamp"],
                        prediction["target_timestamp"],
                        prediction["instrument"],
                        prediction["timeframe"],
                        prediction["model_version"],
                        prediction["prediction"],
                        json.dumps(prediction["probabilities"], sort_keys=True),
                        prediction["data_source"],
                        prediction["data_quality"],
                    ),
                )
            except sqlite3.IntegrityError:
                row = connection.execute(
                    """SELECT id FROM predictions WHERE input_candle_timestamp=?
                    AND timeframe=? AND model_version=?""",
                    (
                        prediction["input_candle_timestamp"],
                        prediction["timeframe"],
                        prediction["model_version"],
                    ),
                ).fetchone()
                if row is None:
                    raise
                return str(row["id"])
        return prediction_id

    def list_predictions(self, limit: int = 500) -> list[dict[str, Any]]:
        """Return recent forecasts with decoded probability maps."""
        with closing(self.connect()) as connection:
            rows = connection.execute(
                "SELECT * FROM predictions ORDER BY prediction_timestamp DESC LIMIT ?",
                (limit,),
            ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["probabilities"] = json.loads(item.pop("probabilities_json"))
            result.append(item)
        return result

    def acquire_job(self, name: str, lease_minutes: int = 30) -> str | None:
        """Acquire an exclusive job lease, reclaiming only expired work."""
        owner = str(uuid4())
        now = datetime.now(UTC)
        with closing(self.connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT status, heartbeat_at FROM jobs WHERE name = ?", (name,)
            ).fetchone()
            if row and row["status"] == "running" and row["heartbeat_at"]:
                heartbeat = datetime.fromisoformat(row["heartbeat_at"])
                age = (now - heartbeat).total_seconds()
                if age < lease_minutes * 60:
                    connection.rollback()
                    return None
                connection.execute(
                    """UPDATE job_history SET status='interrupted',
                    finished_at=?, message='Lease expired; job recovered after restart'
                    WHERE id=(SELECT id FROM job_history WHERE name=? AND status='running'
                        ORDER BY started_at DESC LIMIT 1)""",
                    (now.isoformat(), name),
                )
            started = now.isoformat()
            connection.execute(
                """INSERT INTO jobs(name, status, owner, started_at, heartbeat_at, message)
                VALUES (?, 'running', ?, ?, ?, 'Running')
                ON CONFLICT(name) DO UPDATE SET status='running', owner=excluded.owner,
                    started_at=excluded.started_at, heartbeat_at=excluded.heartbeat_at,
                    finished_at=NULL, message='Running'""",
                (name, owner, started, started),
            )
            connection.execute(
                """INSERT INTO job_history(id, name, status, started_at)
                VALUES (?, ?, 'running', ?)""",
                (str(uuid4()), name, started),
            )
            connection.commit()
        return owner

    def heartbeat_job(self, name: str, owner: str, message: str = "Running") -> None:
        """Refresh a job lease if this process still owns it."""
        with closing(self.connect()) as connection:
            cursor = connection.execute(
                """UPDATE jobs SET heartbeat_at=?, message=?
                WHERE name=? AND owner=? AND status='running'""",
                (utc_now(), message, name, owner),
            )
            if cursor.rowcount != 1:
                raise RuntimeError(f"Job lease lost: {name}")

    def finish_job(self, name: str, owner: str, status: str, message: str) -> None:
        """Finish a job and preserve its outcome in the history table."""
        if status not in {"succeeded", "failed", "interrupted"}:
            raise ValueError(f"Unsupported terminal job status: {status}")
        finished = utc_now()
        with closing(self.connect()) as connection:
            row = connection.execute(
                "SELECT started_at FROM jobs WHERE name=? AND owner=?",
                (name, owner),
            ).fetchone()
            if row is None:
                raise RuntimeError(f"Cannot finish unowned job: {name}")
            connection.execute(
                """UPDATE jobs SET status=?, finished_at=?, heartbeat_at=?, message=?
                WHERE name=? AND owner=?""",
                (status, finished, finished, message, name, owner),
            )
            connection.execute(
                """UPDATE job_history SET status=?, finished_at=?, message=?
                WHERE id=(SELECT id FROM job_history WHERE name=? AND status='running'
                    ORDER BY started_at DESC LIMIT 1)""",
                (status, finished, message, name),
            )

    def list_jobs(self) -> list[dict[str, Any]]:
        """Return the latest job states."""
        with closing(self.connect()) as connection:
            rows = connection.execute("SELECT * FROM jobs ORDER BY name").fetchall()
        return [dict(row) for row in rows]

    def latest_download(self, timeframe: str | None = None) -> dict[str, Any] | None:
        """Return the latest matching provenance record."""
        query = "SELECT * FROM data_downloads"
        parameters: tuple[Any, ...] = ()
        if timeframe:
            query += " WHERE timeframe=?"
            parameters = (timeframe,)
        query += " ORDER BY downloaded_at DESC LIMIT 1"
        with closing(self.connect()) as connection:
            row = connection.execute(query, parameters).fetchone()
        if row is None:
            return None
        result = dict(row)
        result["metadata"] = json.loads(result.pop("metadata_json"))
        return result

    def record_model(self, metadata: dict[str, Any]) -> None:
        """Persist a successfully written and load-verified candidate."""
        with closing(self.connect()) as connection:
            connection.execute(
                """INSERT INTO models(
                    version, model_name, artifact_path, registered_at,
                    validation_macro_f1, validation_brier, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    metadata["version"],
                    metadata["model_name"],
                    metadata["artifact_path"],
                    metadata["registered_at"],
                    metadata["validation_metrics"]["macro_f1"],
                    metadata["validation_metrics"]["brier_score"],
                    json.dumps(metadata, sort_keys=True),
                ),
            )

    def get_model(self, version: str) -> dict[str, Any] | None:
        """Read model metadata."""
        with closing(self.connect()) as connection:
            row = connection.execute("SELECT * FROM models WHERE version=?", (version,)).fetchone()
        if row is None:
            return None
        result = dict(row)
        result["metadata"] = json.loads(result.pop("metadata_json"))
        return result

    def list_models(self, limit: int = 100) -> list[dict[str, Any]]:
        """List recently registered model versions."""
        with closing(self.connect()) as connection:
            rows = connection.execute(
                "SELECT * FROM models ORDER BY registered_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["metadata"] = json.loads(item.pop("metadata_json"))
            result.append(item)
        return result

    def update_prediction_outcome(
        self,
        prediction_id: str,
        actual_return: float,
        actual_class: str,
        is_correct: bool,
    ) -> None:
        """Set a previously unknown outcome once, never revise it."""
        with closing(self.connect()) as connection:
            cursor = connection.execute(
                """UPDATE predictions SET actual_return=?, actual_class=?,
                is_correct=?, evaluated_at=? WHERE id=? AND evaluated_at IS NULL""",
                (
                    actual_return,
                    actual_class,
                    int(is_correct),
                    utc_now(),
                    prediction_id,
                ),
            )
            if cursor.rowcount != 1:
                raise ValueError(f"Prediction {prediction_id} is missing or already evaluated.")
