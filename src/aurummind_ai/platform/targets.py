"""Forward-label construction, isolated from causal feature generation."""

from dataclasses import dataclass
from typing import Any

import pandas as pd

from ..features.technical_indicators import TechnicalIndicators
from .config import load_config
from .market_data import normalize_ohlcv
from .providers import TIMEFRAME_MINUTES


@dataclass(frozen=True)
class TargetFrame:
    """Future returns and labels aligned to the candle that generated features."""

    future_return: pd.Series
    neutral_threshold: pd.Series
    label: pd.Series


def create_direction_target(
    candles: pd.DataFrame,
    timeframe: str = "M15",
    config: dict[str, Any] | None = None,
    horizon_candles: int = 1,
) -> TargetFrame:
    """Label each closed candle UP/DOWN/NEUTRAL for its future horizon.

    The label row at index t uses the close at t+horizon only as the outcome.
    Rows with missing or non-contiguous future candles are unlabeled.
    """
    if horizon_candles < 1:
        raise ValueError("horizon_candles must be a positive integer.")
    timeframe = timeframe.upper()
    if timeframe not in TIMEFRAME_MINUTES:
        raise ValueError(f"Unsupported timeframe: {timeframe}")
    settings = (config or load_config())["features"]
    data = normalize_ohlcv(candles).sort_values("timestamp").reset_index(drop=True)
    if data["timestamp"].duplicated().any():
        raise ValueError("Target input contains duplicate timestamps.")

    future_close = data["close"].shift(-horizon_candles)
    future_time = data["timestamp"].shift(-horizon_candles)
    expected_delta = pd.Timedelta(minutes=TIMEFRAME_MINUTES[timeframe] * horizon_candles)
    contiguous = (future_time - data["timestamp"]) == expected_delta
    future_return = future_close / data["close"] - 1
    atr = TechnicalIndicators.atr(data, period=14)
    volatility_threshold = atr / data["close"] * float(settings["neutral_atr_multiplier"])
    cost_threshold = float(settings["estimated_round_trip_cost_bps"]) / 10_000
    threshold = volatility_threshold.clip(lower=cost_threshold)
    available = contiguous & future_return.notna() & threshold.notna()
    labels = pd.Series(pd.NA, index=data.index, dtype="string")
    labels.loc[available & (future_return > threshold)] = "UP"
    labels.loc[available & (future_return < -threshold)] = "DOWN"
    labels.loc[available & (future_return >= -threshold) & (future_return <= threshold)] = "NEUTRAL"
    return TargetFrame(
        future_return=future_return.where(available),
        neutral_threshold=threshold.where(available),
        label=labels,
    )
