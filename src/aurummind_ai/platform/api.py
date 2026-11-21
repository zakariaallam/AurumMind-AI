"""Local FastAPI facade for market, prediction, model, and job status."""

import hmac
import json
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any, Literal

from fastapi import FastAPI, Header, HTTPException, Query
from pydantic import BaseModel

from .config import load_config, resolve_path
from .database import Database
from .jobs import execute_owned_job
from .market_data import MarketDataService, MarketDataValidator
from .prediction import PredictionService
from .registry import ModelRegistry
from .training import TrainingService

config = load_config()
database = Database(config["application"]["database_path"])
market_data = MarketDataService(config, database=database)
registry = ModelRegistry(config, database)
prediction_service = PredictionService(config, database, registry=registry, market_data=market_data)
executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="aurummind")
app = FastAPI(
    title="AurumMind AI",
    description="Local research API for real-provider XAUUSD candles and forecasts.",
    version="0.2.0",
)


class DownloadRequest(BaseModel):
    start: datetime
    end: datetime | None = None
    timeframe: Literal["M5", "M15", "H1", "H4"] = "M15"
    confirm: bool = False


class TimeframeRequest(BaseModel):
    timeframe: Literal["M5", "M15", "H1", "H4"] = "M15"


def _authorize(authorization: str | None) -> None:
    """Protect state-changing operations with a server-side bearer token."""
    expected = os.environ.get("AURUMMIND_API_TOKEN")
    if not expected:
        raise HTTPException(
            status_code=503,
            detail="Operational API is disabled until AURUMMIND_API_TOKEN is set.",
        )
    supplied = authorization.removeprefix("Bearer ").strip() if authorization else ""
    if not hmac.compare_digest(supplied, expected):
        raise HTTPException(status_code=401, detail="Invalid or missing bearer token.")


def _submit(name: str, operation: Any) -> dict[str, str]:
    owner = database.acquire_job(name)
    if owner is None:
        raise HTTPException(status_code=409, detail=f"Job {name} is already running.")

    def run() -> None:
        try:
            execute_owned_job(database, name, owner, operation)
        except Exception as exc:
            import logging

            logging.getLogger(__name__).exception("API background job failed: %s", exc)

    executor.submit(run)
    return {"job": name, "status": "queued"}


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "champion": registry.champion_version()}


@app.get("/api/market/latest")
def market_latest(
    timeframe: Literal["M5", "M15", "H1", "H4"] = "M15",
) -> dict[str, Any]:
    candles = market_data.load_processed(timeframe)
    if candles.empty:
        raise HTTPException(status_code=404, detail="No validated market candles.")
    report = MarketDataValidator(float(config["market_data"]["maximum_gap_fraction"])).validate(
        candles, timeframe, provider=config["application"]["provider"]
    )
    return {
        "candle": candles.iloc[-1].to_dict(),
        "data_quality": report,
        "source": database.latest_download(timeframe),
    }


@app.get("/api/market/history")
def market_history(
    timeframe: Literal["M5", "M15", "H1", "H4"] = "M15",
    limit: int = Query(default=500, ge=1, le=10_000),
) -> list[dict[str, Any]]:
    candles = market_data.load_processed(timeframe)
    records = candles.tail(limit).to_dict(orient="records")
    return [{str(key): value for key, value in record.items()} for record in records]


@app.get("/api/prediction/latest")
def prediction_latest(
    timeframe: Literal["M5", "M15", "H1", "H4"] = "M15",
) -> dict[str, Any]:
    result: dict[str, Any] = prediction_service.predict_latest(timeframe=timeframe)
    if result["status"] != "available":
        raise HTTPException(status_code=503, detail=result)
    return result


@app.get("/api/prediction/history")
def prediction_history(limit: int = Query(default=500, ge=1, le=5000)) -> list[dict[str, Any]]:
    return database.list_predictions(limit)


@app.get("/api/models/champion")
def model_champion() -> dict[str, Any]:
    version = registry.champion_version()
    if version is None:
        raise HTTPException(status_code=404, detail="MODEL NOT READY")
    metadata: dict[str, Any] = registry.load(version)["metadata"]
    return metadata


@app.get("/api/models/metrics")
def model_metrics() -> dict[str, Any]:
    version = registry.champion_version()
    if version is None:
        return {"status": "unavailable", "reason": "MODEL NOT READY"}
    metadata = registry.load(version)["metadata"]
    return {
        "version": version,
        "validation": metadata["validation_metrics"],
        "walk_forward": metadata["walk_forward_metrics"],
        "holdout_test": metadata["test_metrics"],
        "economic_simulation": metadata["test_economic_metrics"],
    }


@app.get("/api/data/status")
def data_status(
    timeframe: Literal["M5", "M15", "H1", "H4"] = "M15",
) -> dict[str, Any]:
    report_path = (
        resolve_path(config["application"]["reports_dir"]) / f"data_quality_report_{timeframe}.json"
    )
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else None
    candles = market_data.load_processed(timeframe)
    return {
        "timeframe": timeframe,
        "candle_count": len(candles),
        "last_candle": (candles["timestamp"].iloc[-1].isoformat() if len(candles) else None),
        "last_update": database.get_state(f"last_update:{timeframe}"),
        "provider": config["application"]["provider"],
        "latest_download": database.latest_download(timeframe),
        "quality_report": report,
    }


@app.get("/api/jobs")
def jobs() -> list[dict[str, Any]]:
    return database.list_jobs()


@app.post("/api/data/download", status_code=202)
def data_download(
    request: DownloadRequest,
    authorization: str | None = Header(default=None),
) -> dict[str, str]:
    _authorize(authorization)
    if not request.confirm:
        raise HTTPException(
            status_code=400,
            detail="Historical download requires explicit confirm=true.",
        )
    if request.start.tzinfo is None:
        raise HTTPException(status_code=422, detail="start must include a timezone.")
    if request.end is not None and request.end.tzinfo is None:
        raise HTTPException(status_code=422, detail="end must include a timezone.")
    if request.end is not None and request.start >= request.end:
        raise HTTPException(status_code=422, detail="end must be later than start.")
    return _submit(
        "api_data_download",
        lambda: market_data.download_history(request.start, request.end, request.timeframe),
    )


@app.post("/api/data/update", status_code=202)
def data_update(
    request: TimeframeRequest,
    authorization: str | None = Header(default=None),
) -> dict[str, str]:
    _authorize(authorization)
    return _submit("api_data_update", lambda: market_data.update(request.timeframe))


@app.post("/api/training/start", status_code=202)
def training_start(
    authorization: str | None = Header(default=None),
) -> dict[str, str]:
    _authorize(authorization)
    return _submit(
        "api_training",
        lambda: TrainingService(config, database, registry).train_saved(False),
    )


@app.post("/api/training/optimize", status_code=202)
def training_optimize(
    authorization: str | None = Header(default=None),
) -> dict[str, str]:
    _authorize(authorization)
    return _submit(
        "api_optimization",
        lambda: TrainingService(config, database, registry).train_saved(True),
    )


@app.post("/api/models/evaluate")
def evaluate_model(
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    _authorize(authorization)
    version = registry.champion_version()
    if version is None:
        raise HTTPException(status_code=404, detail="MODEL NOT READY")
    metadata: dict[str, Any] = registry.load(version)["metadata"]
    return metadata
