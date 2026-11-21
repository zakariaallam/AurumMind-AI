"""Command-line entry points for the AurumMind operational platform."""

import argparse
import json
import logging
import os
import signal
import subprocess
import sys
from datetime import UTC, datetime
from threading import Event
from typing import Any

from .platform.config import PROJECT_ROOT, load_config, resolve_path
from .platform.database import Database
from .platform.features import CausalFeaturePipeline
from .platform.jobs import execute_job
from .platform.market_data import MarketDataService, MarketDataValidator
from .platform.prediction import PredictionService
from .platform.registry import ModelRegistry
from .platform.training import TrainingService
from .platform.worker import PlatformWorker

logger = logging.getLogger("aurummind")


def build_parser() -> argparse.ArgumentParser:
    """Create the nested operational command parser."""
    parser = argparse.ArgumentParser(prog="aurummind")
    parser.add_argument("--config", default=None, help="System YAML configuration.")
    commands = parser.add_subparsers(dest="command", required=True)

    data = commands.add_parser("data", help="Download, update, and validate data.")
    data_commands = data.add_subparsers(dest="data_command", required=True)
    download = data_commands.add_parser("download", help="Fetch historical candles.")
    download.add_argument("--start", help="UTC ISO date/time; default from config.")
    download.add_argument("--end", help="UTC ISO date/time; default latest complete.")
    download.add_argument("--timeframe", default=None)
    download.add_argument(
        "--confirm",
        action="store_true",
        help="Confirm the potentially large historical download.",
    )
    update = data_commands.add_parser("update", help="Fetch only recent candles.")
    update.add_argument("--timeframe", default=None)
    rebuild = data_commands.add_parser(
        "rebuild", help="Process an already-saved raw provider download."
    )
    rebuild.add_argument("--timeframe", default=None)
    validate = data_commands.add_parser("validate", help="Validate stored candles.")
    validate.add_argument("--timeframe", default=None)

    features = commands.add_parser("features", help="Generate causal features.")
    features_commands = features.add_subparsers(dest="features_command", required=True)
    generate = features_commands.add_parser("generate")
    generate.add_argument("--timeframe", default=None)

    train = commands.add_parser("train", help="Train candidate models.")
    train.add_argument("mode", nargs="?", choices=("baseline", "optimize"), default="baseline")
    train.add_argument(
        "--confirm",
        action="store_true",
        help="Confirm the compute-intensive Optuna search.",
    )

    commands.add_parser("evaluate", help="Show the champion holdout report.")
    predict = commands.add_parser("predict", help="Predict the next candle horizon.")
    predict.add_argument("--timeframe", default=None)

    worker = commands.add_parser("worker", help="Run automatic update/retraining.")
    worker.add_argument("--once", action="store_true", help="Run one worker cycle.")

    models = commands.add_parser("models", help="Manage model versions.")
    models.add_argument("action", choices=("list", "rollback"))

    commands.add_parser("setup", help="Show initial setup status and next actions.")
    commands.add_parser("start", help="Launch the interactive Streamlit dashboard.")
    api = commands.add_parser("api", help="Launch the local FastAPI service.")
    api.add_argument("--host", default="127.0.0.1")
    api.add_argument("--port", type=int, default=8000)
    return parser


def _parse_utc(value: str | None) -> datetime | None:
    if value is None:
        return None
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _print(value: Any) -> None:
    print(json.dumps(value, indent=2, default=str))


def _assert_provenance(database: Database, timeframe: str) -> dict[str, Any]:
    provenance = database.latest_download(timeframe)
    if provenance is None:
        raise RuntimeError(
            "No provider provenance is recorded for this timeframe. Download "
            "historical data from the configured provider first."
        )
    provider = provenance["provider"].lower()
    if provider in {"sample", "sampledata", "synthetic", "unknown"}:
        raise ValueError("This command is blocked for synthetic/unverified data.")
    return provenance


def _run(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    database = Database(config["application"]["database_path"])
    timeframe = config["market_data"]["primary_timeframe"].upper()

    if args.command == "data":
        service = MarketDataService(config, database=database)
        if args.data_command == "download":
            if not args.confirm:
                raise ValueError(
                    "Historical download not started. Review the date range and "
                    "retry with --confirm."
                )
            frame = service.download_history(
                start=_parse_utc(args.start),
                end=_parse_utc(args.end),
                timeframe=args.timeframe,
            )
            _print(frame)
        elif args.data_command == "update":
            _print(service.update(args.timeframe))
        elif args.data_command == "rebuild":
            _print(service.rebuild_processed(args.timeframe))
        else:
            target_timeframe = (args.timeframe or timeframe).upper()
            _assert_provenance(database, target_timeframe)
            candles = service.load_processed(target_timeframe)
            if candles.empty:
                raise RuntimeError(f"No processed {target_timeframe} candles exist.")
            report = MarketDataValidator(
                float(config["market_data"]["maximum_gap_fraction"])
            ).validate(
                candles,
                target_timeframe,
                provider=config["application"]["provider"],
            )
            reports_dir = resolve_path(config["application"]["reports_dir"])
            reports_dir.mkdir(parents=True, exist_ok=True)
            report_path = reports_dir / f"data_quality_report_{target_timeframe}.json"
            report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
            _print({"report_path": str(report_path), **report})
            if not report["passed_validation"]:
                return 2
        return 0

    if args.command == "features":
        target_timeframe = (args.timeframe or timeframe).upper()
        _assert_provenance(database, target_timeframe)
        candles = MarketDataService(config, database=database).load_processed(target_timeframe)
        if candles.empty:
            raise RuntimeError("No processed candles are available.")
        pipeline = CausalFeaturePipeline(config)
        features = pipeline.generate(candles, target_timeframe)
        output = pipeline.save(features, target_timeframe)
        _print(
            {
                "status": "success",
                "output": str(output),
                "feature_count": len(features.columns) - 6,
                "rows": len(features),
                "schema_version": pipeline.version,
            }
        )
        return 0

    if args.command == "train":
        optimize = args.mode == "optimize"
        if optimize and not args.confirm:
            raise ValueError(
                "Optuna optimization not started. Retry with `train optimize "
                "--confirm` to approve the configured trial budget."
            )
        result = execute_job(
            database,
            "manual_training",
            lambda: TrainingService(config, database).train_saved(optimize=optimize),
        )
        _print(result)
        return 0 if result["models"] else 2

    if args.command == "evaluate":
        registry = ModelRegistry(config, database)
        champion = registry.champion_version()
        if champion is None:
            _print({"status": "unavailable", "reason": "MODEL NOT READY"})
            return 2
        metadata = registry.load(champion)["metadata"]
        _print(
            {
                "model_version": champion,
                "validation_metrics": metadata["validation_metrics"],
                "walk_forward_metrics": metadata["walk_forward_metrics"],
                "untouched_test_metrics": metadata["test_metrics"],
                "test_economic_metrics": metadata["test_economic_metrics"],
                "disclaimer": (
                    "Holdout and backtest results are historical evaluations, "
                    "not evidence of future profitability."
                ),
            }
        )
        return 0

    if args.command == "predict":
        result = PredictionService(config, database).predict_latest(
            timeframe=(args.timeframe or timeframe).upper()
        )
        _print(result)
        return 0 if result["status"] == "available" else 2

    if args.command == "worker":
        worker_service = PlatformWorker(config, database)
        if args.once:
            _print(worker_service.run_once())
        else:
            stop_event = Event()
            signal.signal(signal.SIGTERM, lambda _signum, _frame: stop_event.set())
            signal.signal(signal.SIGINT, lambda _signum, _frame: stop_event.set())
            worker_service.run_forever(stop_event)
        return 0

    if args.command == "models":
        registry = ModelRegistry(config, database)
        if args.action == "list":
            _print(
                {
                    "champion": registry.champion_version(),
                    "candidates": database.list_models(),
                }
            )
        else:
            _print({"champion": registry.rollback()})
        return 0

    if args.command == "setup":
        _print(
            {
                "provider": config["application"]["provider"],
                "provider_client_installed": _module_available("dukascopy_python"),
                "database_path": str(database.path),
                "history_downloaded": database.latest_download(timeframe) is not None,
                "champion": ModelRegistry(config, database).champion_version(),
                "next_steps": [
                    "Confirm and run `data download --confirm`.",
                    "Validate and generate features.",
                    "Train candidates and review chronological holdout metrics.",
                    "Open the dashboard to monitor the system.",
                ],
                "historical_download_started": False,
            }
        )
        return 0

    if args.command == "start":
        app_path = PROJECT_ROOT / "app.py"
        environment = os.environ.copy()
        if args.config:
            environment["AURUMMIND_CONFIG"] = str(resolve_path(args.config))
        try:
            return subprocess.call(
                [sys.executable, "-m", "streamlit", "run", str(app_path)],
                cwd=PROJECT_ROOT,
                env=environment,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                "Streamlit is not installed. Run `uv sync` before starting the UI."
            ) from exc

    if args.command == "api":
        try:
            import uvicorn
        except ImportError as exc:
            raise RuntimeError("Install the declared API dependencies with `uv sync`.") from exc
        if args.config:
            os.environ["AURUMMIND_CONFIG"] = str(resolve_path(args.config))
        uvicorn.run(
            "aurummind_ai.platform.api:app",
            host=args.host,
            port=args.port,
            reload=False,
        )
        return 0
    return 2


def _module_available(name: str) -> bool:
    from importlib.util import find_spec

    return find_spec(name) is not None


def main(argv: list[str] | None = None) -> None:
    """Run the CLI and present operational failures as non-zero exits."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        code = _run(args)
    except Exception as exc:
        logger.exception("Command failed: %s", exc)
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    raise SystemExit(code)


if __name__ == "__main__":
    main()
