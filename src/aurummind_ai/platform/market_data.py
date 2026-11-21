"""Causal OHLC validation, provenance, immutable raw storage, and updates."""

import json
import logging
import math
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .config import load_config, resolve_path
from .database import Database, utc_now
from .jobs import execute_job
from .providers import TIMEFRAME_MINUTES, DukascopyProvider, MarketDataProvider

logger = logging.getLogger(__name__)
OHLCV_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


def normalize_ohlcv(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize required columns and timestamps without repairing values."""
    data = frame.copy()
    data.columns = [str(column).strip().lower() for column in data.columns]
    missing = set(OHLCV_COLUMNS) - set(data.columns)
    if missing:
        raise ValueError(f"OHLCV data is missing columns: {sorted(missing)}")
    data = data[OHLCV_COLUMNS]
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True, errors="coerce")
    for column in OHLCV_COLUMNS[1:]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    return data.reset_index(drop=True)


class MarketDataValidator:
    """Validate market candle integrity while preserving gap evidence."""

    def __init__(self, maximum_gap_fraction: float = 0.05) -> None:
        if not 0 <= maximum_gap_fraction <= 1:
            raise ValueError("maximum_gap_fraction must be between 0 and 1.")
        self.maximum_gap_fraction = maximum_gap_fraction

    def validate(
        self,
        frame: pd.DataFrame,
        timeframe: str,
        *,
        as_of: datetime | None = None,
        provider: str = "unknown",
    ) -> dict[str, Any]:
        data = normalize_ohlcv(frame)
        timeframe = timeframe.upper()
        if timeframe not in TIMEFRAME_MINUTES:
            raise ValueError(f"Unsupported timeframe: {timeframe}")
        minutes = TIMEFRAME_MINUTES[timeframe]
        now = pd.Timestamp(as_of or datetime.now(UTC))
        if now.tzinfo is None:
            raise ValueError("Validation as_of time must be timezone-aware.")
        now = now.tz_convert("UTC")
        invalid_values = data["timestamp"].isna() | data[OHLCV_COLUMNS[1:]].isna().any(axis=1)
        finite_mask = pd.Series(
            np.isfinite(data[OHLCV_COLUMNS[1:]].to_numpy(dtype=float)).all(axis=1),
            index=data.index,
        )
        invalid_values |= ~finite_mask
        prices = data[["open", "high", "low", "close"]]
        invalid_prices = (prices <= 0).any(axis=1)
        invalid_ohlc = (data["high"] < data[["open", "close", "low"]].max(axis=1)) | (
            data["low"] > data[["open", "close", "high"]].min(axis=1)
        )
        invalid_volume = data["volume"] < 0
        misaligned = data["timestamp"].notna() & (
            data["timestamp"] != data["timestamp"].dt.floor(f"{minutes}min")
        )
        order_issues = int((data["timestamp"].dropna().diff().dropna() < pd.Timedelta(0)).sum())
        incomplete = data["timestamp"].notna() & (
            data["timestamp"] + pd.Timedelta(minutes=minutes) > now
        )
        latest_time = data["timestamp"].max() if len(data) else pd.NaT
        gap_timestamps: list[str] = []
        if len(data) > 1 and data["timestamp"].notna().all():
            ordered = data["timestamp"].drop_duplicates().sort_values()
            gap_timestamps = self._find_open_session_gaps(ordered, minutes)

        invalid_row_mask = (
            invalid_values | invalid_prices | invalid_ohlc | invalid_volume | misaligned
        )
        unique_count = int(data["timestamp"].nunique())
        denominator = max(unique_count, 1)
        invalid_count = int(invalid_row_mask.sum())
        duplicate_count = int(data["timestamp"].duplicated().sum())
        gap_count = len(gap_timestamps)
        gap_fraction = gap_count / max(unique_count + gap_count, 1)
        score = max(
            0.0,
            100.0
            - 100.0 * invalid_count / max(len(data), 1)
            - min(20.0, 100.0 * gap_count / denominator),
        )
        report = {
            "report_timestamp": utc_now(),
            "provider": provider,
            "timeframe": timeframe,
            "timezone": "UTC",
            "total_candles": int(len(data)),
            "valid_candles": int(len(data) - int((invalid_row_mask | incomplete).sum())),
            "duplicate_candles": duplicate_count,
            "missing_values_or_non_finite": int(invalid_values.sum()),
            "invalid_ohlc": int(invalid_ohlc.sum()),
            "invalid_prices": int(invalid_prices.sum()),
            "invalid_volume": int(invalid_volume.sum()),
            "misaligned_timestamps": int(misaligned.sum()),
            "out_of_order_rows": order_issues,
            "incomplete_candles": int(incomplete.sum()),
            "missing_candles": gap_count,
            "missing_candle_fraction": round(gap_fraction, 6),
            "missing_candle_examples": gap_timestamps[:20],
            "latest_candle_timestamp": (None if pd.isna(latest_time) else latest_time.isoformat()),
            "quality_score": round(score, 2),
            "passed_validation": bool(
                len(data) > 0
                and invalid_count == 0
                and duplicate_count == 0
                and order_issues == 0
                and gap_fraction <= self.maximum_gap_fraction
            ),
        }
        return report

    @staticmethod
    def _find_open_session_gaps(timestamps: pd.Series, minutes: int) -> list[str]:
        """Find missing bars only during approximate New York FX open hours."""
        expected = pd.date_range(
            timestamps.iloc[0], timestamps.iloc[-1], freq=f"{minutes}min", tz="UTC"
        )
        observed = set(timestamps)
        zone = ZoneInfo("America/New_York")
        missing: list[str] = []
        for timestamp in expected:
            if timestamp in observed:
                continue
            local = timestamp.to_pydatetime().astimezone(zone)
            is_open = local.hour != 17 and (
                local.weekday() in {0, 1, 2, 3}
                or (local.weekday() == 4 and local.hour < 17)
                or (local.weekday() == 6 and local.hour >= 17)
            )
            if is_open:
                missing.append(timestamp.isoformat())
        return missing


class MarketDataService:
    """Download, validate, and persist provider candles with full provenance."""

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        provider: MarketDataProvider | None = None,
        database: Database | None = None,
    ) -> None:
        self.config = config or load_config()
        provider_name = self.config["application"]["provider"].lower()
        if provider is not None:
            self.provider = provider
        elif provider_name == "dukascopy":
            self.provider = DukascopyProvider(self.config)
        else:
            raise ValueError(
                f"Unsupported provider {provider_name!r}; configure a supported "
                "provider explicitly. No silent source fallback is allowed."
            )
        self.database = database or Database()
        self.validator = MarketDataValidator(
            float(self.config["market_data"]["maximum_gap_fraction"])
        )

    def download_history(
        self,
        start: datetime | None = None,
        end: datetime | None = None,
        timeframe: str | None = None,
    ) -> dict[str, Any]:
        target_timeframe = (timeframe or self.config["market_data"]["primary_timeframe"]).upper()
        if target_timeframe not in TIMEFRAME_MINUTES:
            raise ValueError(f"Unsupported timeframe: {target_timeframe}")
        return execute_job(
            self.database,
            f"market_data:{target_timeframe}",
            lambda: self._download_history(start, end, target_timeframe),
            lease_minutes=1,
        )

    def _download_history(
        self,
        start: datetime | None,
        end: datetime | None,
        timeframe: str,
    ) -> dict[str, Any]:
        """Download a requested historical range; caller explicitly starts it."""
        settings = self.config["market_data"]
        start = start or datetime.fromisoformat(settings["historical_start"]).replace(tzinfo=UTC)
        end = end or self.completed_end(
            timeframe,
            int(settings["completed_candle_delay_seconds"]),
        )
        candles = self.provider.download_candles(start, end, timeframe)
        return self._persist(candles, start, end, timeframe)

    def update(self, timeframe: str | None = None) -> dict[str, Any]:
        target_timeframe = (timeframe or self.config["market_data"]["primary_timeframe"]).upper()
        if target_timeframe not in TIMEFRAME_MINUTES:
            raise ValueError(f"Unsupported timeframe: {target_timeframe}")
        return execute_job(
            self.database,
            f"market_data:{target_timeframe}",
            lambda: self._update(target_timeframe),
            lease_minutes=1,
        )

    def _update(self, timeframe: str) -> dict[str, Any]:
        """Fetch new completed candles with a two-candle correction overlap."""
        settings = self.config["market_data"]
        if not self._market_session_open(datetime.now(UTC)):
            return {
                "status": "market_closed",
                "timeframe": timeframe,
                "candles_downloaded": 0,
            }
        end = self.completed_end(timeframe, int(settings["completed_candle_delay_seconds"]))
        existing = self.load_processed(timeframe)
        if existing.empty:
            raise RuntimeError(
                "No downloaded history exists. Run the explicit historical "
                "download command with confirmation before requesting updates."
            )
        overlap = timedelta(minutes=2 * TIMEFRAME_MINUTES[timeframe])
        start = (existing["timestamp"].max() - overlap).to_pydatetime()
        latest_close = existing["timestamp"].max() + pd.Timedelta(
            minutes=TIMEFRAME_MINUTES[timeframe]
        )
        if latest_close >= pd.Timestamp(end):
            return {
                "status": "up_to_date",
                "timeframe": timeframe,
                "candles_downloaded": 0,
                "last_update": self.database.get_state(f"last_update:{timeframe}"),
            }
        if start >= end:
            return {
                "status": "up_to_date",
                "timeframe": timeframe,
                "candles_downloaded": 0,
                "last_update": self.database.get_state(f"last_update:{timeframe}"),
            }
        candles = self.provider.download_candles(start, end, timeframe)
        if candles.empty:
            raise RuntimeError(
                f"Provider returned no candles for requested update "
                f"{start.isoformat()}..{end.isoformat()}."
            )
        return self._persist(candles, start, end, timeframe)

    def rebuild_processed(self, timeframe: str | None = None) -> dict[str, Any]:
        """Validate and process the latest saved raw batch without redownloading."""
        target_timeframe = (timeframe or self.config["market_data"]["primary_timeframe"]).upper()
        if target_timeframe not in TIMEFRAME_MINUTES:
            raise ValueError(f"Unsupported timeframe: {target_timeframe}")
        return execute_job(
            self.database,
            f"market_data:{target_timeframe}",
            lambda: self._rebuild_processed(target_timeframe),
            lease_minutes=1,
        )

    def _rebuild_processed(self, timeframe: str) -> dict[str, Any]:
        provenance = self.database.latest_download(timeframe)
        if provenance is None:
            raise RuntimeError(f"No saved {timeframe} provider download to rebuild.")
        provider = provenance["provider"].lower()
        if provider in {"sample", "sampledata", "synthetic", "unknown"}:
            raise ValueError("Rebuilding is blocked for synthetic/unverified data.")
        raw_root = resolve_path(self.config["market_data"]["raw_dir"]).resolve()
        raw_paths = provenance["metadata"].get("raw_files", [])
        if not raw_paths:
            raise RuntimeError("The latest download has no raw batch files recorded.")

        frames = []
        for raw_path in raw_paths:
            path = Path(raw_path).resolve()
            if not path.is_relative_to(raw_root):
                raise ValueError(f"Recorded raw file is outside the data directory: {path}")
            if not path.is_file():
                raise FileNotFoundError(f"Recorded raw batch is missing: {path}")
            frames.append(pd.read_parquet(path))

        data = normalize_ohlcv(pd.concat(frames, ignore_index=True))
        report = self.validator.validate(data, timeframe, provider=provider)
        report_path = (
            resolve_path(self.config["application"]["reports_dir"])
            / f"data_quality_report_{timeframe}.json"
        )
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        minimum_score = float(self.config["market_data"]["minimum_quality_score"])
        if not report["passed_validation"] or report["quality_score"] < minimum_score:
            raise ValueError(
                f"Saved {timeframe} raw batch failed quality gates "
                f"(score={report['quality_score']}, required={minimum_score})."
            )

        interval = pd.Timedelta(minutes=TIMEFRAME_MINUTES[timeframe])
        completed = data.loc[
            data["timestamp"].notna() & (data["timestamp"] + interval <= pd.Timestamp.now(tz="UTC"))
        ]
        clean = completed.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
        destination = (
            resolve_path(self.config["market_data"]["processed_dir"])
            / timeframe
            / f"{self.config['application']['instrument']}_{timeframe}.parquet"
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp.parquet")
        clean.to_parquet(temporary, index=False)
        temporary.replace(destination)
        self.database.upsert_state(f"last_rebuild:{timeframe}", utc_now())
        return {
            "status": "success",
            "timeframe": timeframe,
            "candles_processed": int(len(clean)),
            "processed_path": str(destination),
            "data_quality": report,
            "source": provenance["provider"],
        }

    def _persist(
        self,
        frame: pd.DataFrame,
        start: datetime,
        end: datetime,
        timeframe: str,
    ) -> dict[str, Any]:
        data = normalize_ohlcv(frame)
        now = pd.Timestamp.now(tz="UTC")
        interval = pd.Timedelta(minutes=TIMEFRAME_MINUTES[timeframe])
        completed = data.loc[
            data["timestamp"].notna() & (data["timestamp"] + interval <= now)
        ].copy()
        report = self.validator.validate(
            data, timeframe, provider=getattr(self.provider, "name", "unknown")
        )

        base = resolve_path(self.config["market_data"]["raw_dir"]) / timeframe
        batch_id = uuid4().hex
        raw_paths: list[str] = []
        if not data.empty:
            partitioned = data.assign(
                _year=data["timestamp"].dt.year,
                _month=data["timestamp"].dt.month,
            )
            for (year, month), group in partitioned.groupby(["_year", "_month"], dropna=False):
                if pd.isna(year) or pd.isna(month):
                    directory = base / "invalid_timestamp"
                else:
                    directory = base / f"year={int(year)}" / f"month={int(month):02d}"
                directory.mkdir(parents=True, exist_ok=True)
                file_path = directory / f"batch-{batch_id}.parquet"
                group.drop(columns=["_year", "_month"]).to_parquet(file_path, index=False)
                raw_paths.append(str(file_path))

        provenance = {
            "provider": self.provider.name,
            "instrument": self.config["application"]["instrument"],
            "timeframe": timeframe,
            "requested_start": start.astimezone(UTC).isoformat(),
            "requested_end": end.astimezone(UTC).isoformat(),
            "downloaded_at": utc_now(),
            "timezone": "UTC",
            "price_side": "BID",
            "volume_semantics": getattr(
                self.provider,
                "volume_semantics",
                "Provider-reported volume; semantics not independently verified.",
            ),
            "candle_count": int(len(data)),
            "raw_files": raw_paths,
            "quality_score": report["quality_score"],
            "quality_report": report,
        }
        metadata_path = base / "provenance" / f"{batch_id}.json"
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
        provenance["source_file"] = str(metadata_path)
        self.database.record_download(provenance)
        reports_dir = resolve_path(self.config["application"]["reports_dir"])
        reports_dir.mkdir(parents=True, exist_ok=True)
        (reports_dir / f"data_quality_report_{timeframe}.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )

        minimum_score = float(self.config["market_data"]["minimum_quality_score"])
        if not report["passed_validation"] or report["quality_score"] < minimum_score:
            raise ValueError(
                f"Downloaded data failed quality gates "
                f"(score={report['quality_score']}, required={minimum_score}). "
                "Raw provider files and the quality report were preserved; "
                "invalid rows were not added to processed training data."
            )

        if not completed.empty:
            clean = completed.drop_duplicates("timestamp", keep="last")
            destination = (
                resolve_path(self.config["market_data"]["processed_dir"])
                / timeframe
                / f"{self.config['application']['instrument']}_{timeframe}.parquet"
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                previous = pd.read_parquet(destination)
                clean = pd.concat([previous, clean], ignore_index=True)
            clean = clean.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
            temporary = destination.with_name(f".{destination.name}.{batch_id}.tmp.parquet")
            clean.to_parquet(temporary, index=False)
            temporary.replace(destination)
            self.database.upsert_state(f"last_update:{timeframe}", provenance["downloaded_at"])
        return {
            "status": "success",
            "timeframe": timeframe,
            "candles_downloaded": int(len(data)),
            "candles_processed": int(len(completed)),
            "total_processed": int(len(clean)) if not completed.empty else 0,
            "data_quality": report,
            "processed_path": (str(destination) if not completed.empty else None),
            "raw_paths": raw_paths,
            "source": self.provider.name,
        }

    def load_processed(self, timeframe: str | None = None) -> pd.DataFrame:
        """Read consolidated validated candles, or an empty typed frame."""
        timeframe = (timeframe or self.config["market_data"]["primary_timeframe"]).upper()
        path = (
            resolve_path(self.config["market_data"]["processed_dir"])
            / timeframe
            / f"{self.config['application']['instrument']}_{timeframe}.parquet"
        )
        if not path.exists():
            return pd.DataFrame(columns=OHLCV_COLUMNS)
        return normalize_ohlcv(pd.read_parquet(path))

    @staticmethod
    def completed_end(timeframe: str, delay_seconds: int) -> datetime:
        """Return the boundary after the most recent complete candle."""
        interval_seconds = TIMEFRAME_MINUTES[timeframe.upper()] * 60
        now = time.time() - delay_seconds
        boundary = math.floor(now / interval_seconds) * interval_seconds
        return datetime.fromtimestamp(boundary, tz=UTC)

    @staticmethod
    def _market_session_open(timestamp: datetime) -> bool:
        """Approximate FX session: Sunday 17:00 through Friday 17:00 New York."""
        local = timestamp.astimezone(ZoneInfo("America/New_York"))
        if local.weekday() == 5:
            return False
        if local.weekday() == 6:
            return local.hour >= 17
        if local.weekday() == 4:
            return local.hour < 17
        return True
