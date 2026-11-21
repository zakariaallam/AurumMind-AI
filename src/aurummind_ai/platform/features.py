"""Deterministic, causal M5/M15/H1/H4 feature generation."""

import json
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd

from ..features.technical_indicators import TechnicalIndicators
from .config import load_config, resolve_path
from .market_data import normalize_ohlcv
from .providers import TIMEFRAME_MINUTES


class CausalFeaturePipeline:
    """Generate features using only the current and preceding candle data."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or load_config()
        self.settings = self.config["features"]

    @property
    def version(self) -> str:
        return str(self.settings["version"])

    def generate(self, candles: pd.DataFrame, timeframe: str = "M15") -> pd.DataFrame:
        """Create a stable feature schema; leave warm-up NaNs untouched."""
        timeframe = timeframe.upper()
        if timeframe not in TIMEFRAME_MINUTES:
            raise ValueError(f"Unsupported timeframe: {timeframe}")
        data = normalize_ohlcv(candles).sort_values("timestamp").reset_index(drop=True)
        if data.empty:
            raise ValueError("Cannot generate features without market candles.")
        if data["timestamp"].duplicated().any():
            raise ValueError("Feature input contains duplicate candle timestamps.")
        close = data["close"]
        high = data["high"]
        low = data["low"]
        open_price = data["open"]
        features = pd.DataFrame(index=data.index)

        for period in self.settings["return_periods"]:
            features[f"return_{period}"] = close.pct_change(period)

        features["candle_body_pct"] = (close - open_price) / close
        features["upper_wick_pct"] = (
            high - pd.concat([open_price, close], axis=1).max(axis=1)
        ) / close
        features["lower_wick_pct"] = (
            pd.concat([open_price, close], axis=1).min(axis=1) - low
        ) / close
        features["candle_range_pct"] = (high - low) / close
        candle_range = high - low
        features["close_location"] = np.where(candle_range > 0, (close - low) / candle_range, 0.5)

        direction = np.sign(close - open_price)
        streak = direction.groupby((direction != direction.shift()).cumsum()).cumcount() + 1
        features["bullish_streak"] = streak.where(direction > 0, 0)
        features["bearish_streak"] = streak.where(direction < 0, 0)

        for period in self.settings["rolling_periods"]:
            previous_high = high.rolling(period, min_periods=period).max().shift(1)
            previous_low = low.rolling(period, min_periods=period).min().shift(1)
            features[f"rolling_high_{period}_distance"] = previous_high / close - 1
            features[f"rolling_low_{period}_distance"] = previous_low / close - 1
            features[f"breakout_high_{period}"] = (close > previous_high).astype(float)
            features[f"breakout_low_{period}"] = (close < previous_low).astype(float)

        for period in self.settings["ema_periods"]:
            ema = TechnicalIndicators.ema(data, period=period)
            features[f"ema_{period}_distance"] = close / ema - 1
        for period in self.settings["sma_periods"]:
            sma = TechnicalIndicators.sma(data, period=period)
            features[f"sma_{period}_distance"] = close / sma - 1

        for period in self.settings["rsi_periods"]:
            features[f"rsi_{period}"] = TechnicalIndicators.rsi(data, period=period)
        macd, signal, histogram = TechnicalIndicators.macd(data)
        features["macd_pct"] = macd / close
        features["macd_signal_pct"] = signal / close
        features["macd_histogram_pct"] = histogram / close
        features["adx_14"] = TechnicalIndicators.adx(data, period=14)
        stoch_k, stoch_d = TechnicalIndicators.stochastic(data)
        features["stochastic_k"] = stoch_k
        features["stochastic_d"] = stoch_d
        features["roc_12"] = TechnicalIndicators.roc(data, period=12)

        for period in self.settings["atr_periods"]:
            atr = TechnicalIndicators.atr(data, period=period)
            features[f"atr_{period}"] = atr
            features[f"atr_{period}_pct"] = atr / close
        returns = close.pct_change()
        for period in self.settings["rolling_periods"]:
            features[f"return_volatility_{period}"] = returns.rolling(
                period, min_periods=period
            ).std()
        candles_per_day = 1440 / TIMEFRAME_MINUTES[timeframe]
        features["historical_volatility_20"] = returns.rolling(20, min_periods=20).std() * np.sqrt(
            candles_per_day
        )
        features["normalized_candle_range"] = (high - low) / data["close"].rolling(
            20, min_periods=20
        ).mean()

        self._add_time_context(data, features, timeframe)
        self._add_previous_day_levels(data, features)
        features = features.replace([np.inf, -np.inf], np.nan)
        result = pd.concat(
            [
                data[["timestamp", "open", "high", "low", "close", "volume"]],
                features,
            ],
            axis=1,
        )
        result.attrs["feature_schema_version"] = self.version
        return result

    @staticmethod
    def _add_time_context(data: pd.DataFrame, features: pd.DataFrame, timeframe: str) -> None:
        ts = data["timestamp"] + pd.Timedelta(minutes=TIMEFRAME_MINUTES[timeframe])
        features["hour_utc"] = ts.dt.hour
        features["day_of_week_utc"] = ts.dt.day_of_week
        for name, zone, start_hour, end_hour in (
            ("london_session", "Europe/London", 8, 17),
            ("new_york_session", "America/New_York", 8, 17),
            ("asian_session", "Asia/Tokyo", 9, 18),
        ):
            local = ts.dt.tz_convert(zone)
            features[name] = ((local.dt.hour >= start_hour) & (local.dt.hour < end_hour)).astype(
                int
            )
        features["london_new_york_overlap"] = (
            (features["london_session"] == 1) & (features["new_york_session"] == 1)
        ).astype(int)

    @staticmethod
    def _add_previous_day_levels(data: pd.DataFrame, features: pd.DataFrame) -> None:
        dates = data["timestamp"].dt.floor("D")
        daily = (
            data.assign(_date=dates)
            .groupby("_date")
            .agg(
                previous_day_high=("high", "max"),
                previous_day_low=("low", "min"),
                previous_day_close=("close", "last"),
            )
        )
        prior = daily.shift(1)
        for column in prior.columns:
            lookup = prior[column].to_dict()
            values = dates.map(lookup)
            features[column] = values
            features[f"distance_from_{column}"] = (data["close"] - values) / data["close"]

    def save(self, features: pd.DataFrame, timeframe: str) -> Path:
        """Save features and a versioned feature-definition report."""
        timeframe = timeframe.upper()
        if timeframe not in TIMEFRAME_MINUTES:
            raise ValueError(f"Unsupported timeframe: {timeframe}")
        directory = resolve_path(self.config["market_data"]["processed_dir"]) / timeframe
        directory.mkdir(parents=True, exist_ok=True)
        instrument = self.config["application"]["instrument"]
        destination = directory / f"{instrument}_{timeframe}_features.parquet"
        temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp.parquet")
        features.to_parquet(temporary, index=False)
        temporary.replace(destination)
        feature_names = [
            column
            for column in features
            if column not in {"timestamp", "open", "high", "low", "close", "volume"}
        ]
        report = {
            "feature_schema_version": self.version,
            "instrument": instrument,
            "timeframe": timeframe,
            "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
            "row_count": len(features),
            "feature_count": len(feature_names),
            "features": feature_names,
            "causality": (
                "Rolling indicators use current/prior closed candles. Previous-day "
                "levels use only completed earlier UTC dates. Warm-up NaNs are "
                "retained; no backward fill is performed."
            ),
        }
        destination.with_suffix(".features.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        return destination


def verify_no_future_leakage(
    candles: pd.DataFrame,
    timeframe: str = "M15",
    config: dict[str, Any] | None = None,
) -> bool:
    """Perturb the final candle and verify that earlier feature rows do not move."""
    data = normalize_ohlcv(candles).sort_values("timestamp").reset_index(drop=True)
    if len(data) < 2:
        return False
    pipeline = CausalFeaturePipeline(config)
    original = pipeline.generate(data, timeframe)
    altered_data = data.copy()
    last = altered_data.index[-1]
    changed_close = float(altered_data.at[last, "close"]) * 1.001
    altered_data.at[last, "close"] = changed_close
    altered_data.at[last, "high"] = max(float(altered_data.at[last, "high"]), changed_close)
    altered = pipeline.generate(altered_data, timeframe)
    original_features = original.iloc[:-1, 6:].to_numpy(dtype=float)
    altered_features = altered.iloc[:-1, 6:].to_numpy(dtype=float)
    return bool(np.allclose(original_features, altered_features, equal_nan=True))
