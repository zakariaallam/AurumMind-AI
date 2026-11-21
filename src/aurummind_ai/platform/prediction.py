"""Live inference and immutable forecast-outcome tracking."""

from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

from .config import load_config
from .database import Database
from .features import CausalFeaturePipeline
from .market_data import MarketDataService, MarketDataValidator
from .providers import TIMEFRAME_MINUTES
from .registry import ModelRegistry

CLASS_LABELS = {0: "DOWN", 1: "NEUTRAL", 2: "UP"}


class PredictionService:
    """Make calibrated predictions only from a valid active champion."""

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        database: Database | None = None,
        registry: ModelRegistry | None = None,
        market_data: MarketDataService | None = None,
    ) -> None:
        self.config = config or load_config()
        self.database = database or Database()
        self.registry = registry or ModelRegistry(self.config, self.database)
        self.market_data = market_data or MarketDataService(self.config, database=self.database)
        self.features = CausalFeaturePipeline(self.config)

    def predict_latest(
        self,
        instrument: str = "XAUUSD",
        timeframe: str = "M15",
    ) -> dict[str, Any]:
        """Return a structured prediction or an explicit unavailable status."""
        timeframe = timeframe.upper()
        if timeframe not in TIMEFRAME_MINUTES:
            raise ValueError(f"Unsupported timeframe: {timeframe}")
        if instrument != self.config["application"]["instrument"]:
            raise ValueError("Requested instrument is not enabled in configuration.")
        champion = self.registry.champion_version()
        if champion is None:
            return {
                "status": "unavailable",
                "reason": "MODEL NOT READY: no validated champion is registered.",
            }
        candles = self.market_data.load_processed(timeframe)
        if candles.empty:
            return {
                "status": "unavailable",
                "reason": "No validated provider candles are available.",
            }
        quality = MarketDataValidator(
            float(self.config["market_data"]["maximum_gap_fraction"])
        ).validate(
            candles,
            timeframe,
            provider=self.config["application"]["provider"],
        )
        if not quality["passed_validation"] or quality["quality_score"] < float(
            self.config["market_data"]["minimum_quality_score"]
        ):
            return {
                "status": "unavailable",
                "reason": "Latest market data failed the configured quality gates.",
                "data_quality": quality,
            }
        interval = pd.Timedelta(minutes=TIMEFRAME_MINUTES[timeframe])
        latest = candles.iloc[-1]
        close_time = latest["timestamp"] + interval
        now = pd.Timestamp.now(tz="UTC")
        allowed_age = interval * 2
        if close_time > now:
            return {
                "status": "unavailable",
                "reason": "The latest stored candle has not closed yet.",
                "latest_candle_close": close_time.isoformat(),
            }
        if now - close_time > allowed_age:
            return {
                "status": "unavailable",
                "reason": "Latest completed market candle is stale.",
                "latest_candle_close": close_time.isoformat(),
                "data_quality": quality,
            }

        model_bundle = self.registry.load(champion)
        model_metadata = model_bundle["metadata"]
        if model_metadata["timeframe"].upper() != timeframe:
            return {
                "status": "unavailable",
                "reason": (
                    f"Champion {champion} was trained for "
                    f"{model_metadata['timeframe']}, not {timeframe}."
                ),
                "model_version": champion,
            }
        feature_frame = self.features.generate(candles, timeframe)
        expected_features = model_bundle["feature_names"]
        missing = [name for name in expected_features if name not in feature_frame]
        if missing:
            return {
                "status": "unavailable",
                "reason": f"Feature schema mismatch; missing: {missing}",
                "model_version": champion,
            }
        row = feature_frame.iloc[[-1]][expected_features]
        if not hasattr(model_bundle["model"], "predict_proba"):
            return {
                "status": "unavailable",
                "reason": "Champion does not provide calibrated probabilities.",
                "model_version": champion,
            }
        probabilities_array = model_bundle["model"].predict_proba(row)[0]
        classes = model_bundle["model"].classes_
        probabilities = {
            CLASS_LABELS[int(label)]: float(probability)
            for label, probability in zip(classes, probabilities_array, strict=True)
        }
        probabilities = {
            label: probabilities.get(label, 0.0) for label in ("UP", "DOWN", "NEUTRAL")
        }
        if (
            not np.isfinite(list(probabilities.values())).all()
            or abs(sum(probabilities.values()) - 1.0) > 1e-5
        ):
            raise RuntimeError("Champion returned invalid probability values.")

        threshold = model_bundle.get("signal_threshold")
        if threshold is None:
            return {
                "status": "unavailable",
                "reason": "Champion is missing its validation-frozen signal threshold.",
                "model_version": champion,
            }
        best_label = max(probabilities, key=lambda label: probabilities[label])
        prediction = (
            best_label
            if best_label in {"UP", "DOWN"} and probabilities[best_label] >= threshold
            else "NEUTRAL"
        )
        horizon_minutes = int(model_metadata["forecast_horizon_minutes"])
        if horizon_minutes % TIMEFRAME_MINUTES[timeframe] != 0:
            raise ValueError("Forecast horizon must align with the configured candles.")
        prediction_time = datetime.now(UTC)
        target_time = close_time.to_pydatetime() + timedelta(minutes=horizon_minutes)
        provider_record = self.database.latest_download(timeframe)
        if provider_record is None:
            return {
                "status": "unavailable",
                "reason": "Data provenance record is missing.",
            }
        result = {
            "status": "available",
            "instrument": instrument,
            "timeframe": timeframe,
            "prediction": prediction,
            "probabilities": probabilities,
            "forecast_horizon_minutes": horizon_minutes,
            "model_version": champion,
            "prediction_timestamp": prediction_time.isoformat(),
            "input_candle_timestamp": latest["timestamp"].isoformat(),
            "target_timestamp": target_time.isoformat(),
            "data_source": provider_record["provider"],
            "data_quality": "valid",
            "data_quality_score": quality["quality_score"],
        }
        self.database.create_prediction(result)
        return result

    def evaluate_pending(self, timeframe: str = "M15") -> int:
        """Resolve forecasts whose full target candle has now closed."""
        candles = self.market_data.load_processed(timeframe)
        if candles.empty:
            return 0
        interval = pd.Timedelta(minutes=TIMEFRAME_MINUTES[timeframe.upper()])
        indexed = candles.set_index("timestamp")
        feature_frame = self.features.generate(candles, timeframe).set_index("timestamp")
        updates = 0
        for prediction in self.database.list_predictions(limit=5000):
            if prediction["actual_class"] is not None:
                continue
            target_close_time = pd.Timestamp(prediction["target_timestamp"])
            future_open_time = target_close_time - interval
            input_time = pd.Timestamp(prediction["input_candle_timestamp"])
            if future_open_time not in indexed.index or input_time not in indexed.index:
                continue
            initial_close = float(indexed.loc[input_time, "close"])
            future_close = float(indexed.loc[future_open_time, "close"])
            realized = future_close / initial_close - 1
            atr_pct = float(feature_frame.loc[input_time, "atr_14"] / initial_close)
            neutral_threshold = max(
                atr_pct * float(self.config["features"]["neutral_atr_multiplier"]),
                float(self.config["features"]["estimated_round_trip_cost_bps"]) / 10_000,
            )
            if realized > neutral_threshold:
                actual_class = "UP"
            elif realized < -neutral_threshold:
                actual_class = "DOWN"
            else:
                actual_class = "NEUTRAL"
            self.database.update_prediction_outcome(
                prediction_id=prediction["id"],
                actual_return=float(realized),
                actual_class=actual_class,
                is_correct=prediction["predicted_class"] == actual_class,
            )
            updates += 1
        return updates
