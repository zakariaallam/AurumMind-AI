"""AurumMind AI Streamlit dashboard; no market or forecast data is fabricated."""

import json
import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, date
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from aurummind_ai.platform.config import load_config, resolve_path
from aurummind_ai.platform.database import Database
from aurummind_ai.platform.features import CausalFeaturePipeline
from aurummind_ai.platform.jobs import execute_owned_job
from aurummind_ai.platform.market_data import MarketDataService, MarketDataValidator
from aurummind_ai.platform.prediction import PredictionService
from aurummind_ai.platform.registry import ModelRegistry
from aurummind_ai.platform.training import TrainingService
from aurummind_ai.platform.worker import PlatformWorker
from aurummind_ai.platform.worker_process import (
    start_worker,
    stop_worker,
    worker_status,
)

logger = logging.getLogger(__name__)
st.set_page_config(
    page_title="AurumMind AI",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

CONFIG = load_config()
DB = Database(CONFIG["application"]["database_path"])
MARKET_DATA = MarketDataService(CONFIG, database=DB)
REGISTRY = ModelRegistry(CONFIG, DB)
PREDICTOR = PredictionService(CONFIG, DB, REGISTRY, MARKET_DATA)
EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="aurummind-ui")

st.markdown(
    """
    <style>
    .stApp { background: #0b1017; color: #e8edf5; }
    [data-testid="stMetric"] { background: #111a24; border: 1px solid #263545;
      padding: 14px; border-radius: 10px; }
    div[data-testid="stAlert"] { border-radius: 10px; }
    </style>
    """,
    unsafe_allow_html=True,
)


def submit_job(name: str, operation: Callable[[], Any]) -> None:
    owner = DB.acquire_job(name)
    if owner is None:
        st.warning(f"{name} is already running.")
        return

    def run() -> None:
        try:
            execute_owned_job(DB, name, owner, operation)
        except Exception:
            logger.exception("Background job %s failed", name)

    EXECUTOR.submit(run)
    st.success(f"{name} queued. Monitor its status in the Jobs panel.")


def render_jobs() -> None:
    st.subheader("Background jobs")
    st.dataframe(pd.DataFrame(DB.list_jobs()), use_container_width=True, hide_index=True)
    if st.button("Refresh job status", key="refresh_jobs"):
        st.rerun()


def show_live_market() -> None:
    st.header("Live market")
    timeframe = st.selectbox(
        "Chart timeframe",
        CONFIG["market_data"]["supported_timeframes"],
        index=CONFIG["market_data"]["supported_timeframes"].index(
            CONFIG["market_data"]["primary_timeframe"]
        ),
    )
    candles = MARKET_DATA.load_processed(timeframe)
    latest_download = DB.latest_download(timeframe)
    champion = REGISTRY.champion_version()
    forecast_timeframe = CONFIG["market_data"]["primary_timeframe"]
    latest_prediction = PREDICTOR.predict_latest(timeframe=forecast_timeframe)

    if candles.empty:
        st.warning(
            "No real provider candles are stored. Download historical Dukascopy "
            "data from Training Center; sample CSVs are not used in this dashboard."
        )
        st.info("Model status: MODEL NOT READY")
        return

    last = candles.iloc[-1]
    latest_close = last["timestamp"] + pd.Timedelta(
        minutes={"M5": 5, "M15": 15, "H1": 60, "H4": 240}[timeframe]
    )
    freshness_seconds = (pd.Timestamp.now(tz="UTC") - latest_close).total_seconds()
    stale = freshness_seconds > (2 * {"M5": 300, "M15": 900, "H1": 3600, "H4": 14400}[timeframe])
    market, model, connection, freshness = st.columns(4)
    market.metric("Latest close", f"{last['close']:,.2f}")
    model.metric("Champion model", champion or "MODEL NOT READY")
    connection.metric(
        "Provider", latest_download["provider"] if latest_download else "Not connected"
    )
    freshness.metric("Data freshness", "STALE" if stale else "Current")
    if stale:
        st.error(f"Market data is stale. Last completed candle closed {latest_close}.")

    feature_frame = CausalFeaturePipeline(CONFIG).generate(candles, timeframe)
    chart = go.Figure()
    chart.add_trace(
        go.Candlestick(
            x=candles["timestamp"],
            open=candles["open"],
            high=candles["high"],
            low=candles["low"],
            close=candles["close"],
            name="Dukascopy candles",
        )
    )
    for period in (20, 50, 200):
        column = f"ema_{period}_distance"
        if column in feature_frame:
            chart.add_trace(
                go.Scatter(
                    x=candles["timestamp"],
                    y=candles["close"] / (1 + feature_frame[column]),
                    mode="lines",
                    name=f"EMA {period}",
                )
            )
    historical = DB.list_predictions(500)
    for label, color, symbol in (
        ("UP", "#22c55e", "triangle-up"),
        ("DOWN", "#ef4444", "triangle-down"),
        ("NEUTRAL", "#94a3b8", "circle"),
    ):
        items = [item for item in historical if item["predicted_class"] == label]
        if not items:
            continue
        times = pd.to_datetime([item["input_candle_timestamp"] for item in items], utc=True)
        points = candles.set_index("timestamp").reindex(times)
        chart.add_trace(
            go.Scatter(
                x=times,
                y=points["close"].to_numpy(),
                mode="markers",
                marker={"color": color, "symbol": symbol, "size": 9},
                name=f"Past {label} forecasts",
                customdata=[
                    [
                        item["model_version"],
                        item["target_timestamp"],
                        item["probabilities"].get(label, 0),
                    ]
                    for item in items
                ],
                hovertemplate=(
                    "%{x}<br>Close: %{y:.2f}<br>Model: %{customdata[0]}"
                    "<br>Target: %{customdata[1]}<br>Probability: "
                    "%{customdata[2]:.1%}<extra></extra>"
                ),
            )
        )
    chart.update_layout(
        template="plotly_dark",
        height=560,
        xaxis_rangeslider_visible=False,
        paper_bgcolor="#0b1017",
        plot_bgcolor="#0b1017",
        legend_orientation="h",
    )
    st.plotly_chart(chart, use_container_width=True)
    st.caption(
        "Historical prediction markers are forecasts, not confirmed future prices. "
        "Volume is provider-reported and may represent tick volume."
    )
    forecast_horizon = latest_prediction.get("forecast_horizon_minutes")
    forecast_title = (
        f"Next {forecast_horizon}-minute forecast ({forecast_timeframe})"
        if forecast_horizon is not None
        else f"Forecast ({forecast_timeframe})"
    )
    st.subheader(forecast_title)
    if latest_prediction["status"] != "available":
        st.warning(latest_prediction["reason"])
        return
    st.markdown(f"## {latest_prediction['prediction']}")
    st.write(
        f"Model `{latest_prediction['model_version']}` • "
        f"generated {latest_prediction['prediction_timestamp']} • "
        f"input candle {latest_prediction['input_candle_timestamp']} • "
        f"target {latest_prediction['target_timestamp']}"
    )
    st.progress(
        min(
            1.0,
            max(latest_prediction["probabilities"].values()),
        ),
        text="Highest calibrated class probability",
    )
    probability_columns = st.columns(3)
    for probability_column, label in zip(
        probability_columns, ("UP", "DOWN", "NEUTRAL"), strict=True
    ):
        probability_column.metric(
            f"{label} probability",
            f"{latest_prediction['probabilities'][label]:.1%}",
        )


def show_model_analytics() -> None:
    st.header("Model analytics")
    champion = REGISTRY.champion_version()
    if champion is None:
        st.warning("MODEL NOT READY. No candidate has passed promotion checks.")
    else:
        metadata = REGISTRY.load(champion)["metadata"]
        st.success(f"Active champion: {champion}")
        st.json(
            {
                "validation": metadata["validation_metrics"],
                "walk_forward": metadata["walk_forward_metrics"],
                "untouched_test": metadata["test_metrics"],
                "cost_aware_backtest": metadata["test_economic_metrics"],
                "feature_schema_version": metadata["feature_schema_version"],
                "data_range": metadata["source_data_range"],
            }
        )
        st.caption(
            "Holdout and backtest results are historical evaluations, not "
            "predictions of future performance."
        )
    models = DB.list_models()
    st.subheader("Candidate and champion history")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "version": row["version"],
                    "model": row["model_name"],
                    "registered_at": row["registered_at"],
                    "validation_macro_f1": row["validation_macro_f1"],
                    "validation_brier": row["validation_brier"],
                    "is_champion": row["version"] == champion,
                }
                for row in models
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )


def show_training_center() -> None:
    st.header("Training center and initial setup")
    has_history = not MARKET_DATA.load_processed().empty
    has_champion = REGISTRY.champion_version() is not None
    st.write(
        "Setup progress: "
        f"{'✅' if has_history else '⬜'} provider history → "
        f"{'✅' if has_history else '⬜'} validation/features → "
        f"{'✅' if has_champion else '⬜'} trained and promoted champion"
    )
    st.info(
        "Historical download and Optuna search are never started automatically. "
        "Use the explicit confirmation controls below. Synthetic sample files "
        "are not used for training or displayed as real market data."
    )
    confirm = st.checkbox(
        "I confirm a historical download from the configured provider.",
        key="confirm_history",
    )
    start_date = st.date_input(
        "Historical start date",
        value=date.fromisoformat(CONFIG["market_data"]["historical_start"]),
        key="history_start",
    )
    download_col, update_col = st.columns(2)
    if download_col.button(
        "Download historical M15 data",
        disabled=not confirm,
        type="primary",
    ):
        from datetime import datetime, time

        start = datetime.combine(start_date, time.min, tzinfo=UTC)
        submit_job(
            "ui_historical_download",
            lambda: MARKET_DATA.download_history(start=start, timeframe="M15"),
        )
    if update_col.button("Update completed candles"):
        submit_job("ui_data_update", lambda: MARKET_DATA.update("M15"))

    validate_col, feature_col, train_col, optimize_col = st.columns(4)

    def validate_data() -> None:
        data = MARKET_DATA.load_processed("M15")
        if data.empty:
            raise RuntimeError("Download real M15 history before validation.")
        report = MarketDataValidator(float(CONFIG["market_data"]["maximum_gap_fraction"])).validate(
            data, "M15", provider=CONFIG["application"]["provider"]
        )
        if not report["passed_validation"]:
            raise ValueError(f"Data validation failed: {report}")
        report_path = (
            resolve_path(CONFIG["application"]["reports_dir"]) / "data_quality_report_M15.json"
        )
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if validate_col.button("Queue validation"):
        submit_job("ui_validate", validate_data)
    if feature_col.button("Generate M15 features"):

        def generate_features() -> None:
            data = MARKET_DATA.load_processed("M15")
            if data.empty:
                raise RuntimeError("Download real M15 history before feature generation.")
            _assert_real_provenance("M15")
            pipeline = CausalFeaturePipeline(CONFIG)
            pipeline.save(pipeline.generate(data), "M15")

        submit_job("ui_feature_generation", generate_features)
    if train_col.button("Train baseline challengers"):
        submit_job(
            "ui_training",
            lambda: TrainingService(CONFIG, DB, REGISTRY).train_saved(False),
        )
    optimization_confirmed = optimize_col.checkbox(
        "Confirm configured Optuna trial budget",
        key="confirm_optuna",
    )
    if optimize_col.button("Run Optuna optimization", disabled=not optimization_confirmed):
        submit_job(
            "ui_optimization",
            lambda: TrainingService(CONFIG, DB, REGISTRY).train_saved(True),
        )
    st.subheader("Worker")
    st.write(
        "The worker checks for new completed candles, settles past forecasts, "
        "runs scheduled retraining, and performs Optuna optimization when its "
        "configured interval is due. It never bootstraps a large history download."
    )
    status = worker_status(DB)
    st.write(f"Worker status: **{status['status']}** • PID: {status['pid'] or '—'}")
    start_col, stop_col, cycle_col = st.columns(3)
    if start_col.button("Start scheduled worker", disabled=status["status"] == "running"):
        try:
            st.success(start_worker(DB))
        except Exception as exc:
            st.error(f"Could not start worker: {exc}")
    if stop_col.button("Stop scheduled worker", disabled=status["status"] != "running"):
        try:
            st.info(stop_worker(DB))
        except Exception as exc:
            st.error(f"Could not stop worker: {exc}")
    if cycle_col.button("Run one worker cycle"):
        submit_job("ui_worker_cycle", lambda: PlatformWorker(CONFIG, DB).run_once())
    render_jobs()


def _assert_real_provenance(timeframe: str) -> None:
    provenance = DB.latest_download(timeframe)
    if provenance is None or provenance["provider"].lower() in {
        "sample",
        "sampledata",
        "synthetic",
        "unknown",
    }:
        raise RuntimeError("Real provider provenance is required for this operation.")


def show_data_center() -> None:
    st.header("Data center")
    timeframe = st.selectbox(
        "Timeframe",
        CONFIG["market_data"]["supported_timeframes"],
        key="data_timeframe",
    )
    candles = MARKET_DATA.load_processed(timeframe)
    provenance = DB.latest_download(timeframe)
    report_path = (
        resolve_path(CONFIG["application"]["reports_dir"]) / f"data_quality_report_{timeframe}.json"
    )
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else None
    left, right = st.columns(2)
    left.metric("Validated candle count", len(candles))
    right.metric(
        "Latest candle",
        candles["timestamp"].iloc[-1].isoformat() if len(candles) else "Unavailable",
    )
    st.write("Provider", provenance["provider"] if provenance else "Not downloaded")
    st.write(
        "Last successful update",
        DB.get_state(f"last_update:{timeframe}") or "No update recorded",
    )
    if report:
        st.subheader("Data quality report")
        st.json(report)
    if provenance:
        st.subheader("Download provenance")
        metadata = provenance["metadata"].copy()
        metadata.pop("raw_files", None)
        st.json(metadata)


def show_prediction_history() -> None:
    st.header("Prediction history")
    records = DB.list_predictions(5000)
    if not records:
        st.info("No stored predictions. A validated champion and fresh data are required.")
        return
    table = pd.DataFrame(records)
    st.dataframe(
        table[
            [
                "prediction_timestamp",
                "input_candle_timestamp",
                "target_timestamp",
                "predicted_class",
                "probabilities",
                "actual_class",
                "actual_return",
                "is_correct",
                "model_version",
                "provider",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )
    settled = table.dropna(subset=["is_correct"])
    if not settled.empty:
        st.metric("Paper prediction hit rate", f"{settled['is_correct'].mean():.1%}")
        st.caption("This is simulated forecast tracking, not actual broker execution.")


def show_settings() -> None:
    st.header("Settings")
    st.write(
        "Runtime configuration is in `configs/system.yaml`. Values are validated "
        "when commands start; restart the worker after changing its schedule. "
        "Provider credentials are not required by the configured public Dukascopy "
        "historical feed; never put secrets in this dashboard."
    )
    st.json(CONFIG)


st.title("AurumMind AI")
st.caption("XAUUSD market research • calibrated forecasts • no order execution")
pages = st.tabs(
    [
        "Live market",
        "Model analytics",
        "Training center",
        "Data center",
        "Prediction history",
        "Settings",
    ]
)
with pages[0]:
    show_live_market()
with pages[1]:
    show_model_analytics()
with pages[2]:
    show_training_center()
with pages[3]:
    show_data_center()
with pages[4]:
    show_prediction_history()
with pages[5]:
    show_settings()
