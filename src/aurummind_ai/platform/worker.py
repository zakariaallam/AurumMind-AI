"""Recoverable single-process data-update and retraining worker."""

import logging
import time
from datetime import UTC, datetime, timedelta
from typing import Any

from .config import load_config
from .database import Database
from .jobs import execute_owned_job
from .market_data import MarketDataService
from .prediction import PredictionService
from .registry import ModelRegistry
from .training import TrainingService

logger = logging.getLogger(__name__)


class PlatformWorker:
    """Run bounded, non-overlapping update cycles; no history bootstrap."""

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        database: Database | None = None,
    ) -> None:
        self.config = config or load_config()
        self.database = database or Database()
        self.market_data = MarketDataService(self.config, database=self.database)
        self.registry = ModelRegistry(self.config, self.database)
        self.predictor = PredictionService(
            self.config, self.database, self.registry, self.market_data
        )
        self.trainer = TrainingService(self.config, self.database, self.registry)

    def run_once(self) -> dict[str, Any]:
        """Update market state, settle forecasts, and retrain only when due."""
        owner = self.database.acquire_job("worker_cycle")
        if owner is None:
            return {"status": "already_running"}
        return execute_owned_job(
            self.database,
            "worker_cycle",
            owner,
            lambda: self._run_cycle(owner),
        )

    def _run_cycle(self, owner: str) -> dict[str, Any]:
        if self.market_data.load_processed().empty:
            return {
                "status": "history_required",
                "message": (
                    "Historical bootstrap is required. Run the explicitly "
                    "confirmed `data download` command before starting updates."
                ),
            }
        before_count = len(self.market_data.load_processed())
        update = self.market_data.update()
        after_count = len(self.market_data.load_processed())
        new_candles = max(after_count - before_count, 0)
        self.database.heartbeat_job(
            "worker_cycle", owner, "Market data updated; evaluating outcomes"
        )
        settled = self.predictor.evaluate_pending()
        prediction = self.predictor.predict_latest()
        training_result = None
        if self._retraining_due(new_candles):
            optimize = self._optimization_due()
            self.database.heartbeat_job("worker_cycle", owner, "Training chronological challengers")
            training_result = self.trainer.train_saved(optimize=optimize)
            if not training_result["models"]:
                raise RuntimeError(
                    "Scheduled training did not produce any candidates: "
                    f"{training_result['failures']}"
                )
        return {
            "status": "success",
            "data_update": update,
            "new_candles": new_candles,
            "settled_predictions": settled,
            "latest_prediction": prediction,
            "training": training_result,
        }

    def run_forever(self, stop_event: Any | None = None) -> None:
        """Poll at the configured interval until interrupted."""
        interval = int(self.config["worker"]["update_poll_seconds"])
        logger.info("Platform worker started; poll interval is %s seconds", interval)
        while stop_event is None or not stop_event.is_set():
            try:
                result = self.run_once()
                logger.info("Worker cycle status: %s", result["status"])
            except Exception:
                logger.exception("Worker cycle failed; the next cycle will retry")
            if stop_event is None:
                time.sleep(interval)
            elif stop_event.wait(interval):
                break

    def _retraining_due(self, new_candles: int) -> bool:
        threshold = int(self.config["worker"]["retrain_min_new_candles"])
        last = self.database.get_state("last_retrain_at")
        champion_missing = self.registry.champion_version() is None
        if champion_missing and not last:
            return True
        if new_candles >= threshold:
            return True
        if new_candles == 0:
            return False
        if not last:
            return True
        last_time = datetime.fromisoformat(last)
        due_after = timedelta(days=int(self.config["worker"]["retrain_interval_days"]))
        return datetime.now(UTC) - last_time >= due_after

    def _optimization_due(self) -> bool:
        last = self.database.get_state("last_optimization_at")
        if not last:
            return True
        last_time = datetime.fromisoformat(last)
        due_after = timedelta(days=int(self.config["worker"]["optimization_interval_days"]))
        return datetime.now(UTC) - last_time >= due_after
