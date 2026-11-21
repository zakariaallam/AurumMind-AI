"""Local FastAPI facade for market, prediction, model, and job status."""

import hmac
import json
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any, Literal

from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .config import load_config, resolve_path
from .database import Database
from .jobs import execute_owned_job
from .market_data import MarketDataService, MarketDataValidator
from .prediction import PredictionService
from .registry import ModelRegistry
from .training import TrainingService
from .worker import PlatformWorker
from .worker_process import start_worker, stop_worker, worker_status

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
allowed_origins = {
    origin.strip()
    for origin in os.environ.get(
        "AURUMMIND_CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173,http://localhost:4173,"
        "http://127.0.0.1:4173",
    ).split(",")
    if origin.strip()
}
app.add_middleware(
    CORSMiddleware,
    allow_origins=sorted(allowed_origins),
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
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


@app.get("/api/models")
def models() -> list[dict[str, Any]]:
    champion = registry.champion_version()
    return [
        {
            "version": row["version"],
            "model_name": row["model_name"],
            "registered_at": row["registered_at"],
            "validation_macro_f1": row["validation_macro_f1"],
            "validation_brier": row["validation_brier"],
            "is_champion": row["version"] == champion,
        }
        for row in database.list_models()
    ]


@app.get("/api/features/status")
def feature_status(
    timeframe: Literal["M5", "M15", "H1", "H4"] = "M15",
) -> dict[str, Any]:
    directory = resolve_path(config["market_data"]["processed_dir"]) / timeframe
    instrument = config["application"]["instrument"]
    feature_path = directory / f"{instrument}_{timeframe}_features.parquet"
    metadata_path = directory / f"{instrument}_{timeframe}_features.features.json"
    metadata = (
        json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata_path.is_file()
        else None
    )
    return {
        "timeframe": timeframe,
        "available": feature_path.is_file(),
        "path": str(feature_path) if feature_path.is_file() else None,
        "metadata": metadata,
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


@app.get("/api/worker")
def get_worker_status() -> dict[str, int | str | None]:
    return worker_status(database)


@app.get("/api/settings")
def settings() -> dict[str, Any]:
    """Expose operational configuration without environment secrets or local paths."""
    return {
        "application": {
            "instrument": config["application"]["instrument"],
            "provider": config["application"]["provider"],
            "timezone": config["application"]["timezone"],
        },
        "market_data": {
            key: config["market_data"][key]
            for key in (
                "primary_timeframe",
                "supported_timeframes",
                "historical_start",
                "minimum_quality_score",
                "maximum_gap_fraction",
                "request_chunk_days",
            )
        },
        "features": config["features"],
        "training": config["training"],
        "backtest": config["backtest"],
        "worker": config["worker"],
    }


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


@app.post("/api/data/validate", status_code=202)
def data_validate(
    request: TimeframeRequest,
    authorization: str | None = Header(default=None),
) -> dict[str, str]:
    _authorize(authorization)

    def validate() -> dict[str, Any]:
        candles = market_data.load_processed(request.timeframe)
        if candles.empty:
            raise RuntimeError(f"No processed {request.timeframe} candles are available.")
        provenance = database.latest_download(request.timeframe)
        if provenance is None or provenance["provider"].lower() in {
            "sample",
            "sampledata",
            "synthetic",
            "unknown",
        }:
            raise RuntimeError("Real provider provenance is required for validation.")
        report = MarketDataValidator(
            float(config["market_data"]["maximum_gap_fraction"])
        ).validate(
            candles,
            request.timeframe,
            provider=provenance["provider"],
        )
        report_path = (
            resolve_path(config["application"]["reports_dir"])
            / f"data_quality_report_{request.timeframe}.json"
        )
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        if not report["passed_validation"]:
            raise ValueError(f"Data validation failed: {report}")
        return report

    return _submit(f"api_validate_{request.timeframe}", validate)


@app.post("/api/features/generate", status_code=202)
def generate_features(
    request: TimeframeRequest,
    authorization: str | None = Header(default=None),
) -> dict[str, str]:
    _authorize(authorization)

    def generate() -> dict[str, Any]:
        provenance = database.latest_download(request.timeframe)
        if provenance is None or provenance["provider"].lower() in {
            "sample",
            "sampledata",
            "synthetic",
            "unknown",
        }:
            raise RuntimeError("Real provider provenance is required for feature generation.")
        candles = market_data.load_processed(request.timeframe)
        if candles.empty:
            raise RuntimeError(f"No processed {request.timeframe} candles are available.")
        from .features import CausalFeaturePipeline

        pipeline = CausalFeaturePipeline(config)
        features = pipeline.generate(candles, request.timeframe)
        output = pipeline.save(features, request.timeframe)
        return {
            "status": "success",
            "output": str(output),
            "rows": len(features),
            "feature_count": len(features.columns) - 6,
            "schema_version": pipeline.version,
        }

    return _submit(f"api_features_{request.timeframe}", generate)


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


@app.post("/api/models/rollback")
def rollback_model(
    authorization: str | None = Header(default=None),
) -> dict[str, str]:
    _authorize(authorization)
    return {"version": registry.rollback()}


@app.post("/api/worker/start")
def worker_start(
    authorization: str | None = Header(default=None),
) -> dict[str, int | str | None]:
    _authorize(authorization)
    return start_worker(database)


@app.post("/api/worker/stop")
def worker_stop(
    authorization: str | None = Header(default=None),
) -> dict[str, int | str | None]:
    _authorize(authorization)
    return stop_worker(database)


@app.post("/api/worker/cycle", status_code=202)
def worker_cycle(
    authorization: str | None = Header(default=None),
) -> dict[str, str]:
    _authorize(authorization)
    return _submit(
        "api_worker_cycle",
        lambda: PlatformWorker(config, database).run_once(),
    )
