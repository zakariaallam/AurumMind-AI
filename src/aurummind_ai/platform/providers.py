"""Provider adapters for historical and incremental market data acquisition."""

import logging
import time
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol

import pandas as pd

from .config import load_config

logger = logging.getLogger(__name__)

TIMEFRAME_MINUTES = {"M5": 5, "M15": 15, "H1": 60, "H4": 240}


class MarketDataProvider(Protocol):
    """Interface implemented by market data providers."""

    name: str

    def download_candles(self, start: datetime, end: datetime, timeframe: str) -> pd.DataFrame: ...


class _TimeoutSession:
    """Create requests sessions with retries and a default per-request timeout."""

    def __init__(self, timeout_seconds: int, retries: int = 5) -> None:
        self.timeout_seconds = timeout_seconds
        self.retries = retries

    def __call__(self) -> Any:
        try:
            import requests
            from requests.adapters import HTTPAdapter
            from urllib3.util.retry import Retry
        except ImportError as exc:
            raise RuntimeError(
                "The Dukascopy adapter requires requests; install the project "
                "dependencies with `uv sync`."
            ) from exc

        timeout = self.timeout_seconds

        class TimeoutSession(requests.Session):
            def request(self, method: str, url: str | bytes, **kwargs: Any) -> requests.Response:  # type: ignore[override]
                kwargs.setdefault("timeout", timeout)
                return super().request(method, url, **kwargs)

        session = TimeoutSession()
        retry = Retry(
            total=self.retries,
            connect=self.retries,
            read=self.retries,
            backoff_factor=0.5,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset({"GET"}),
            respect_retry_after_header=True,
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        return session


class DukascopyProvider:
    """Fetch OHLC candles from Dukascopy's public historical candle feed.

    Uses the supported `dukascopy-python` client, bounded time windows, UTC
    timestamps, retrying HTTP sessions, and explicit rate limiting between
    requests. No credentials or silent alternative-provider fallback are used.
    """

    name = "dukascopy"
    instrument_code = "XAU/USD"
    volume_semantics = (
        "Dukascopy candle volume as returned by the provider; not assumed to be "
        "centralized exchange-traded volume."
    )

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or load_config()
        market = self.config["market_data"]
        self.chunk_days = int(market["request_chunk_days"])
        self.pause_seconds = float(market["request_pause_seconds"])
        self.timeout_seconds = int(market["request_timeout_seconds"])

    def download_candles(self, start: datetime, end: datetime, timeframe: str) -> pd.DataFrame:
        """Download candles over a historical range in bounded chunks."""
        timeframe = timeframe.upper()
        if timeframe not in TIMEFRAME_MINUTES:
            raise ValueError(
                f"Unsupported timeframe {timeframe}; expected one of "
                f"{', '.join(TIMEFRAME_MINUTES)}."
            )
        start = self._as_utc(start)
        end = self._as_utc(end)
        if start >= end:
            raise ValueError("Download start must be earlier than end.")

        try:
            import dukascopy_python as duka
        except ImportError as exc:
            raise RuntimeError(
                "Dukascopy downloads require dukascopy-python. Run `uv sync` "
                "and retry; no other provider will be selected automatically."
            ) from exc

        minutes = TIMEFRAME_MINUTES[timeframe]
        result: list[dict[str, Any]] = []
        cursor = start
        session_factory = _TimeoutSession(self.timeout_seconds)
        while cursor < end:
            chunk_end = min(cursor + timedelta(days=self.chunk_days), end)
            logger.info(
                "Requesting Dukascopy %s candles from %s to %s",
                timeframe,
                cursor.isoformat(),
                chunk_end.isoformat(),
            )
            try:
                candles = duka.fetch_candles(
                    instrument=self.instrument_code,
                    interval=minutes if minutes < 60 else minutes // 60,
                    time_unit=(duka.TimeUnit.MINUTE if minutes < 60 else duka.TimeUnit.HOUR),
                    offer_side=duka.OfferSide.BID,
                    start=cursor,
                    end=chunk_end,
                    version=1,
                    validate_instrument_name=False,
                    create_session=session_factory,
                )
                for candle in candles:
                    result.append(
                        {
                            "timestamp": candle.timestamp,
                            "open": candle.open,
                            "high": candle.high,
                            "low": candle.low,
                            "close": candle.close,
                            "volume": candle.volume,
                        }
                    )
            except Exception as exc:
                raise RuntimeError(
                    f"Dukascopy request failed for {timeframe} "
                    f"{cursor.isoformat()}..{chunk_end.isoformat()}: {exc}"
                ) from exc
            cursor = chunk_end
            if cursor < end and self.pause_seconds > 0:
                time.sleep(self.pause_seconds)

        frame = pd.DataFrame(
            result, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        if frame.empty:
            return frame
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        for column in ("open", "high", "low", "close", "volume"):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame = frame.reset_index(drop=True)
        interval = pd.Timedelta(minutes=minutes)
        return frame.loc[
            (frame["timestamp"] >= start) & (frame["timestamp"] + interval <= end)
        ].reset_index(drop=True)

    @staticmethod
    def _as_utc(value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("Provider date ranges must be timezone-aware.")
        return value.astimezone(UTC)
