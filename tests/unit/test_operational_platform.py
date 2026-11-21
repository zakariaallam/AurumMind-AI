"""Focused tests for the real-data forecasting platform."""

import sys
from copy import deepcopy
from datetime import UTC, datetime
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.dummy import DummyClassifier

from aurummind_ai.platform.config import DEFAULT_CONFIG, load_config
from aurummind_ai.platform.database import Database
from aurummind_ai.platform.features import (
    CausalFeaturePipeline,
    verify_no_future_leakage,
)
from aurummind_ai.platform.jobs import execute_job
from aurummind_ai.platform.market_data import (
    MarketDataService,
    MarketDataValidator,
)
from aurummind_ai.platform.prediction import PredictionService
from aurummind_ai.platform.providers import DukascopyProvider
from aurummind_ai.platform.registry import ModelRegistry
from aurummind_ai.platform.targets import create_direction_target
from aurummind_ai.platform.training import TrainingService


def _candles(periods: int = 320, start: pd.Timestamp | None = None) -> pd.DataFrame:
    if start is None:
        start = pd.Timestamp("2024-01-07 17:00", tz="UTC")
    timestamps = pd.date_range(start, periods=periods, freq="15min")
    rng = np.random.default_rng(42)
    close = 2000 + np.cumsum(rng.normal(0, 0.7, periods))
    open_price = np.r_[close[0], close[:-1]]
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": open_price,
            "high": np.maximum(open_price, close) + 0.2,
            "low": np.minimum(open_price, close) - 0.2,
            "close": close,
            "volume": rng.uniform(10, 100, periods),
        }
    )


def _config(tmp_path) -> dict:
    with DEFAULT_CONFIG.open(encoding="utf-8") as stream:
        config = deepcopy(yaml.safe_load(stream))
    config["application"]["database_path"] = str(tmp_path / "state.sqlite3")
    config["application"]["reports_dir"] = str(tmp_path / "reports")
    config["application"]["artifacts_dir"] = str(tmp_path / "models")
    config["market_data"]["raw_dir"] = str(tmp_path / "raw")
    config["market_data"]["processed_dir"] = str(tmp_path / "processed")
    return config


def test_config_can_be_selected_through_environment(tmp_path, monkeypatch):
    config_path = tmp_path / "custom-system.yaml"
    config_path.write_text("application:\n  provider: test-provider\n", encoding="utf-8")
    monkeypatch.setenv("AURUMMIND_CONFIG", str(config_path))
    assert load_config()["application"]["provider"] == "test-provider"


def test_market_validator_reports_source_and_rejects_bad_rows():
    candles = _candles(50)
    report = MarketDataValidator().validate(
        candles,
        "M15",
        as_of=(candles["timestamp"].iloc[-1] + pd.Timedelta(minutes=15)).to_pydatetime(),
        provider="test-provider",
    )
    assert report["provider"] == "test-provider"
    assert report["passed_validation"]

    candles.loc[5, "high"] = candles.loc[5, "low"] - 1
    invalid_report = MarketDataValidator().validate(candles, "M15")
    assert invalid_report["invalid_ohlc"] == 1
    assert not invalid_report["passed_validation"]


def test_dukascopy_adapter_chunks_and_normalizes_without_network(tmp_path, monkeypatch):
    config = _config(tmp_path)
    config["market_data"]["request_chunk_days"] = 1
    config["market_data"]["request_pause_seconds"] = 0
    calls = []

    def fetch_candles(**kwargs):
        calls.append(kwargs)
        return [
            SimpleNamespace(
                timestamp=kwargs["start"],
                open=2000,
                high=2001,
                low=1999,
                close=2000.5,
                volume=12,
            )
        ]

    client = SimpleNamespace(
        TimeUnit=SimpleNamespace(MINUTE="minute", HOUR="hour"),
        OfferSide=SimpleNamespace(BID="bid"),
        fetch_candles=fetch_candles,
    )
    monkeypatch.setitem(sys.modules, "dukascopy_python", client)
    provider = DukascopyProvider(config)
    candles = provider.download_candles(
        datetime(2024, 1, 1, tzinfo=UTC),
        datetime(2024, 1, 3, tzinfo=UTC),
        "M15",
    )

    assert len(calls) == 2
    assert all(call["instrument"] == "XAU/USD" for call in calls)
    assert all(call["interval"] == 15 for call in calls)
    assert all(call["time_unit"] == "minute" for call in calls)
    assert all(call["offer_side"] == "bid" for call in calls)
    assert all(call["validate_instrument_name"] is False for call in calls)
    assert pd.api.types.is_datetime64_any_dtype(candles["timestamp"])
    assert candles["timestamp"].dt.tz is not None
    assert candles["volume"].tolist() == [12, 12]


def test_targets_respect_horizon_and_gaps():
    candles = _candles(80)
    candles = candles.drop(index=40).reset_index(drop=True)
    target = create_direction_target(candles, timeframe="M15", horizon_candles=1)
    assert pd.isna(target.label.iloc[39])
    assert pd.isna(target.label.iloc[-1])
    assert target.label.dropna().isin({"UP", "DOWN", "NEUTRAL"}).all()


def test_features_are_causal_and_keep_warmup_missing_values():
    candles = _candles()
    pipeline = CausalFeaturePipeline()
    features = pipeline.generate(candles)
    assert pd.isna(features.iloc[0]["return_1"])
    assert verify_no_future_leakage(candles)


def test_database_migrations_and_job_lease_are_idempotent(tmp_path):
    database = Database(tmp_path / "state.sqlite3")
    database.migrate()
    owner = database.acquire_job("worker")
    assert owner is not None
    assert database.acquire_job("worker") is None
    database.heartbeat_job("worker", owner, "still active")
    database.finish_job("worker", owner, "succeeded", "done")
    assert database.acquire_job("worker") is not None


def test_job_runner_recovers_an_expired_lease(tmp_path):
    database = Database(tmp_path / "state.sqlite3")
    owner = database.acquire_job("expired-worker")
    assert owner is not None
    with database.connect() as connection:
        connection.execute(
            "UPDATE jobs SET heartbeat_at=? WHERE name=?",
            (
                (datetime.now(UTC) - pd.Timedelta(minutes=2)).isoformat(),
                "expired-worker",
            ),
        )

    assert (
        execute_job(database, "expired-worker", lambda: "recovered", lease_minutes=1) == "recovered"
    )
    with database.connect() as connection:
        previous = connection.execute(
            "SELECT status FROM job_history WHERE name=? ORDER BY started_at LIMIT 1",
            ("expired-worker",),
        ).fetchone()
    assert previous["status"] == "interrupted"


def test_data_and_training_services_share_operation_leases(tmp_path):
    config = _config(tmp_path)
    database = Database(config["application"]["database_path"])
    owner = database.acquire_job("market_data:M15")
    assert owner is not None
    data_service = MarketDataService(config, database=database)
    with pytest.raises(RuntimeError, match="market_data:M15 is already running"):
        data_service.download_history(timeframe="M15")
    database.finish_job("market_data:M15", owner, "succeeded", "test complete")

    owner = database.acquire_job("model_training")
    assert owner is not None
    trainer = TrainingService(config, database)
    with pytest.raises(RuntimeError, match="model_training is already running"):
        trainer.train_saved()


def test_registry_promotes_and_rolls_back(tmp_path):
    config = _config(tmp_path)
    database = Database(config["application"]["database_path"])
    registry = ModelRegistry(config, database)
    model = DummyClassifier(strategy="prior").fit(
        np.array([[0], [1], [2], [3], [4], [5]]),
        np.array([0, 1, 2, 0, 1, 2]),
    )
    metadata = {
        "timeframe": "M15",
        "validation_metrics": {"macro_f1": 0.6, "brier_score": 0.5},
    }
    first = registry.register("candidate", model, ["feature"], metadata)
    promoted, _ = registry.promote(
        first,
        {"macro_f1": 0.6, "brier_score": 0.5},
        None,
        data_quality_score=100,
        leakage_checks_passed=True,
        regime_macro_f1={"all": 0.5},
    )
    assert promoted

    second = registry.register("candidate", model, ["feature"], metadata)
    promoted, _ = registry.promote(
        second,
        {"macro_f1": 0.7, "brier_score": 0.45},
        {"macro_f1": 0.6, "brier_score": 0.5},
        data_quality_score=100,
        leakage_checks_passed=True,
        regime_macro_f1={"all": 0.6},
    )
    assert promoted
    assert registry.rollback() == first["version"]
    assert registry.champion_version() == first["version"]


def test_prediction_refuses_a_champion_from_another_timeframe(tmp_path):
    config = _config(tmp_path)
    now = pd.Timestamp.now(tz="UTC").floor("15min")
    candles = _candles(
        320,
        start=now - pd.Timedelta(minutes=15 * 320),
    )
    candles["timestamp"] = pd.date_range(
        candles["timestamp"].iloc[0], periods=len(candles), freq="15min"
    )

    class FakeRegistry:
        def champion_version(self):
            return "hourly-model"

        def load(self, version):
            return {"metadata": {"timeframe": "H1"}}

    class FakeMarketData:
        def load_processed(self, timeframe):
            return candles

    result = PredictionService(
        config,
        database=SimpleNamespace(),
        registry=FakeRegistry(),
        market_data=FakeMarketData(),
    ).predict_latest(timeframe="M15")
    assert result["status"] == "unavailable"
    assert "not M15" in result["reason"]


def test_training_runs_a_bounded_optuna_search_on_test_data(tmp_path):
    config = _config(tmp_path)
    config["features"]["minimum_training_rows"] = 400
    config["training"]["model_families"] = ["logistic_regression"]
    config["training"]["maximum_optuna_trials"] = 2
    config["training"]["walk_forward_splits"] = 2

    count = 2200
    rng = np.random.default_rng(42)
    returns = rng.normal(0, 0.001, count)
    close = 2000 * np.cumprod(1 + returns)
    open_price = np.r_[close[0], close[:-1]]
    candles = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-07 17:00", periods=count, freq="15min", tz="UTC"),
            "open": open_price,
            "high": np.maximum(open_price, close) + 0.5,
            "low": np.minimum(open_price, close) - 0.5,
            "close": close,
            "volume": rng.uniform(10, 100, count),
        }
    )
    database = Database(config["application"]["database_path"])
    result = TrainingService(config, database).train(
        candles,
        provider="synthetic-test-only",
        data_quality={"quality_score": 100},
        optimize=True,
    )

    assert result["models"]
    metadata = database.list_models()[0]["metadata"]
    assert metadata["optimization"]["trial_count"] == 2
    assert metadata["test_metrics"]["macro_f1"] >= 0


def test_failed_download_keeps_invalid_timestamp_rows_in_raw_batch(tmp_path):
    config = _config(tmp_path)
    candles = _candles(3)
    candles["timestamp"] = candles["timestamp"].astype(object)
    candles.loc[1, "timestamp"] = "not-a-timestamp"

    class FakeProvider:
        name = "test-provider"
        volume_semantics = "synthetic test volume"

        def download_candles(self, start, end, timeframe):
            return candles

    database = Database(config["application"]["database_path"])
    service = MarketDataService(config, FakeProvider(), database)
    with pytest.raises(ValueError, match="failed quality gates"):
        service.download_history(
            start=datetime(2024, 1, 7, tzinfo=UTC),
            end=datetime(2024, 1, 8, tzinfo=UTC),
            timeframe="M15",
        )

    raw_files = list((tmp_path / "raw" / "M15").rglob("batch-*.parquet"))
    assert len(raw_files) == 2
    assert sum(len(pd.read_parquet(path)) for path in raw_files) == 3
    assert (tmp_path / "raw" / "M15" / "provenance").is_dir()
