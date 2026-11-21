"""Chronological three-class training, calibration, evaluation, and promotion."""

import importlib.metadata
import json
import logging
import platform
import subprocess
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import PredefinedSplit, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .backtest import backtest_predictions
from .config import load_config, resolve_path
from .database import Database, utc_now
from .features import CausalFeaturePipeline, verify_no_future_leakage
from .jobs import execute_job
from .market_data import MarketDataService, MarketDataValidator
from .providers import TIMEFRAME_MINUTES
from .registry import ModelRegistry
from .targets import create_direction_target

logger = logging.getLogger(__name__)
CLASS_LABELS = ("DOWN", "NEUTRAL", "UP")
CLASS_TO_INDEX = {label: index for index, label in enumerate(CLASS_LABELS)}
RAW_COLUMNS = {"timestamp", "open", "high", "low", "close", "volume"}


def classification_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    signal_threshold: float | None = None,
) -> dict[str, Any]:
    """Return classification and probability-quality metrics."""
    predicted = probabilities.argmax(axis=1)
    if signal_threshold is not None:
        directional = (predicted != 1) & (probabilities.max(axis=1) < signal_threshold)
        predicted[directional] = 1
    one_hot = np.eye(len(CLASS_LABELS), dtype=float)[y_true]
    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predicted)),
        "precision_macro": float(
            precision_score(y_true, predicted, average="macro", zero_division=0)
        ),
        "recall_macro": float(recall_score(y_true, predicted, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, predicted, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, predicted, labels=[0, 1, 2]).tolist(),
        "brier_score": float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))),
        "log_loss": float(log_loss(y_true, probabilities, labels=[0, 1, 2])),
        "expected_calibration_error": _expected_calibration_error(y_true, probabilities),
    }
    try:
        metrics["roc_auc_ovr_macro"] = float(
            roc_auc_score(
                y_true,
                probabilities,
                labels=[0, 1, 2],
                multi_class="ovr",
                average="macro",
            )
        )
        metrics["pr_auc_macro"] = float(
            average_precision_score(one_hot, probabilities, average="macro")
        )
    except ValueError:
        metrics["roc_auc_ovr_macro"] = None
        metrics["pr_auc_macro"] = None
    return metrics


def _expected_calibration_error(
    y_true: np.ndarray, probabilities: np.ndarray, bins: int = 10
) -> float:
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == y_true
    edges = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    for lower, upper in zip(edges[:-1], edges[1:], strict=True):
        selected = (confidence >= lower) & (confidence < upper)
        if selected.any():
            error += selected.mean() * abs(
                float(correct[selected].mean()) - float(confidence[selected].mean())
            )
    return float(error)


class TrainingService:
    """Train candidates only from quality-gated, provenance-recorded data."""

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        database: Database | None = None,
        registry: ModelRegistry | None = None,
    ) -> None:
        self.config = config or load_config()
        self.database = database or Database()
        self.registry = registry or ModelRegistry(self.config, self.database)
        self.market_data = MarketDataService(self.config, database=self.database)
        self.features = CausalFeaturePipeline(self.config)

    def train_saved(self, optimize: bool = False) -> dict[str, Any]:
        """Train stored provider data under the shared model-training lease."""
        return execute_job(
            self.database,
            "model_training",
            lambda: self._train_saved(optimize),
        )

    def _train_saved(self, optimize: bool) -> dict[str, Any]:
        """Train on stored real candles; never use bundled synthetic fixtures."""
        timeframe = self.config["market_data"]["primary_timeframe"].upper()
        provenance = self.database.latest_download(timeframe)
        if provenance is None:
            raise RuntimeError(
                "No downloaded provider data is recorded. Run the explicit "
                "`aurummind data download --confirm` command first."
            )
        provider = provenance["provider"].lower()
        if provider in {"sample", "sampledata", "synthetic", "unknown"}:
            raise ValueError(
                "Training is blocked because source provenance does not identify "
                "a real market data provider."
            )
        candles = self.market_data.load_processed(timeframe)
        if candles.empty:
            raise RuntimeError("No processed candles are available for training.")
        data_quality = MarketDataValidator(
            float(self.config["market_data"]["maximum_gap_fraction"])
        ).validate(candles, timeframe, provider=provider)
        minimum_score = float(self.config["market_data"]["minimum_quality_score"])
        if not data_quality["passed_validation"] or data_quality["quality_score"] < minimum_score:
            raise ValueError(
                "Training rejected by data-quality gate: "
                f"score={data_quality['quality_score']}, required={minimum_score}."
            )
        result = self.train(
            candles,
            provider=provider,
            data_quality=data_quality,
            optimize=optimize,
            timeframe=timeframe,
        )
        if not result["models"]:
            result["status"] = "failed"
            result["message"] = "No model candidate completed successfully."
        self.database.upsert_state("last_retrain_at", utc_now())
        return result

    def train(
        self,
        candles: pd.DataFrame,
        *,
        provider: str,
        data_quality: dict[str, Any],
        optimize: bool = False,
        timeframe: str = "M15",
    ) -> dict[str, Any]:
        """Fit/calibrate candidates and evaluate them on untouched future data."""
        timeframe = timeframe.upper()
        if timeframe != self.config["market_data"]["primary_timeframe"].upper():
            raise ValueError(
                "Model training is limited to the configured primary timeframe "
                f"({self.config['market_data']['primary_timeframe']})."
            )
        if data_quality["quality_score"] < float(
            self.config["market_data"]["minimum_quality_score"]
        ):
            raise ValueError("Training data does not meet the minimum quality score.")
        horizon_minutes = int(self.config["training"]["forecast_horizon_minutes"])
        if horizon_minutes % TIMEFRAME_MINUTES[timeframe] != 0:
            raise ValueError(
                "Forecast horizon must be an integer number of primary-timeframe candles."
            )
        feature_frame = self.features.generate(candles, timeframe)
        leakage_checks_passed = verify_no_future_leakage(candles, timeframe, self.config)
        feature_path = self.features.save(feature_frame, timeframe)
        target = create_direction_target(
            candles,
            timeframe=timeframe,
            config=self.config,
            horizon_candles=horizon_minutes // TIMEFRAME_MINUTES[timeframe],
        )
        feature_names = [name for name in feature_frame if name not in RAW_COLUMNS]
        x_all = feature_frame[feature_names]
        y_all = target.label.map(CLASS_TO_INDEX)
        valid = y_all.notna() & x_all.notna().all(axis=1)
        x = x_all.loc[valid].reset_index(drop=True)
        y = y_all.loc[valid].astype(int).reset_index(drop=True)
        context = feature_frame.loc[valid].reset_index(drop=True)
        minimum_rows = int(self.config["features"]["minimum_training_rows"])
        if len(x) < minimum_rows:
            raise ValueError(
                f"Only {len(x)} fully labeled feature rows remain; {minimum_rows} are required."
            )
        if set(y.unique()) != {0, 1, 2}:
            raise ValueError(
                "The usable training history does not contain all three target "
                "classes (DOWN, NEUTRAL, UP). No classifier was trained."
            )

        boundaries = self._split_boundaries(len(x))
        train_end, calibration_end, validation_end = boundaries
        # Purge one row before each later window to prevent a forward label from
        # crossing a train/calibration/validation/test boundary.
        train_x, train_y = x.iloc[: train_end - 1], y.iloc[: train_end - 1]
        calibration_x = x.iloc[train_end:calibration_end]
        calibration_y = y.iloc[train_end:calibration_end]
        validation_x = x.iloc[calibration_end + 1 : validation_end]
        validation_y = y.iloc[calibration_end + 1 : validation_end]
        test_x = x.iloc[validation_end + 1 :]
        test_y = y.iloc[validation_end + 1 :]
        if (
            min(
                len(train_x),
                len(calibration_x),
                len(validation_x),
                len(test_x),
            )
            == 0
        ):
            raise ValueError("Chronological train/calibration/validation/test split is empty.")
        if any(
            set(part.unique()) != {0, 1, 2}
            for part in (train_y, calibration_y, validation_y, test_y)
        ):
            raise ValueError(
                "Every chronological split must contain UP, DOWN, and NEUTRAL "
                "labels; collect more provider data or review the target threshold."
            )

        baseline = self._majority_baseline(train_y, validation_y)
        benchmark = self._rule_baselines(
            context.iloc[calibration_end + 1 : validation_end], validation_y
        )
        baseline.update(benchmark)
        results: dict[str, Any] = {
            "status": "completed",
            "provider": provider,
            "timeframe": timeframe,
            "feature_schema_version": self.features.version,
            "feature_path": str(feature_path),
            "usable_rows": len(x),
            "data_quality": data_quality,
            "benchmarks": baseline,
            "models": [],
            "failures": [],
        }
        for model_name in self.config["training"]["model_families"]:
            started = time.monotonic()
            optimization: dict[str, Any] | None = None
            try:
                if optimize:
                    params, optimization = self._optimize(
                        model_name,
                        train_x,
                        train_y,
                        int(self.config["training"]["maximum_optuna_trials"]),
                    )
                else:
                    params = {}
                base_pipeline = self._pipeline(model_name, params)
                combined_x = pd.concat([train_x, calibration_x], ignore_index=True)
                combined_y = pd.concat([train_y, calibration_y], ignore_index=True)
                fold = np.concatenate(
                    [
                        np.full(len(train_x), -1, dtype=int),
                        np.zeros(len(calibration_x), dtype=int),
                    ]
                )
                calibrated = CalibratedClassifierCV(
                    estimator=base_pipeline,
                    method="sigmoid",
                    cv=PredefinedSplit(fold),
                )
                calibrated.fit(combined_x, combined_y)
                validation_probabilities = calibrated.predict_proba(validation_x)
                signal_threshold = self._select_signal_threshold(
                    validation_y.to_numpy(), validation_probabilities
                )
                validation_metrics = classification_metrics(
                    validation_y.to_numpy(),
                    validation_probabilities,
                    signal_threshold,
                )
                test_probabilities = calibrated.predict_proba(test_x)
                test_metrics = classification_metrics(
                    test_y.to_numpy(), test_probabilities, signal_threshold
                )
                test_class_indices = test_probabilities.argmax(axis=1)
                test_class_indices[
                    (test_class_indices != 1) & (test_probabilities.max(axis=1) < signal_threshold)
                ] = 1
                test_predictions = pd.Series(
                    [CLASS_LABELS[int(value)] for value in test_class_indices]
                )
                test_context = context.iloc[validation_end + 1 :]
                economic_metrics = backtest_predictions(
                    test_context[["timestamp", "open", "high", "low", "close", "volume"]],
                    test_predictions,
                    timeframe=timeframe,
                    config=self.config,
                )
                regimes = self._regime_metrics(calibrated, validation_x, validation_y)
                walk_forward_metrics = self._walk_forward_metrics(model_name, train_x, train_y)
                trained_at = utc_now()
                metadata = {
                    "provider": provider,
                    "instrument": self.config["application"]["instrument"],
                    "timeframe": timeframe,
                    "forecast_horizon_minutes": horizon_minutes,
                    "feature_schema_version": self.features.version,
                    "feature_names": feature_names,
                    "target_definition": (
                        "next close return; UP/DOWN outside the larger of "
                        "round-trip-cost and ATR-adjusted thresholds; otherwise "
                        "NEUTRAL; non-contiguous outcomes are excluded"
                    ),
                    "class_mapping": {"0": "DOWN", "1": "NEUTRAL", "2": "UP"},
                    "training_timestamp": trained_at,
                    "source_data_range": [
                        str(feature_frame["timestamp"].iloc[0]),
                        str(feature_frame["timestamp"].iloc[-1]),
                    ],
                    "fit_data_range": [
                        str(feature_frame["timestamp"].iloc[valid[valid].index[0]]),
                        str(feature_frame["timestamp"].iloc[valid[valid].index[train_end - 2]]),
                    ],
                    "data_quality_score": data_quality["quality_score"],
                    "random_seed": self.config["training"]["random_seed"],
                    "hyperparameters": params,
                    "signal_threshold": signal_threshold,
                    "optimization": optimization,
                    "validation_range": [
                        str(
                            feature_frame["timestamp"].iloc[valid[valid].index[calibration_end + 1]]
                        ),
                        str(
                            feature_frame["timestamp"].iloc[valid[valid].index[validation_end - 1]]
                        ),
                    ],
                    "test_range": [
                        str(
                            feature_frame["timestamp"].iloc[valid[valid].index[validation_end + 1]]
                        ),
                        str(feature_frame["timestamp"].iloc[valid[valid].index[-1]]),
                    ],
                    "validation_metrics": validation_metrics,
                    "test_metrics": test_metrics,
                    "test_economic_metrics": economic_metrics,
                    "walk_forward_metrics": walk_forward_metrics,
                    "regime_validation_macro_f1": regimes,
                    "training_duration_seconds": round(time.monotonic() - started, 3),
                    "code_version": self._code_version(),
                    "python_version": platform.python_version(),
                    "dependency_versions": self._dependency_versions(),
                    "leakage_checks_passed": leakage_checks_passed,
                    "benchmarks": baseline,
                    "calibration": {
                        "method": "sigmoid",
                        "fit_window": "chronological calibration window, not validation/test",
                        "validation_expected_calibration_error": validation_metrics[
                            "expected_calibration_error"
                        ],
                    },
                    "signal_threshold_selection": {
                        "method": "maximum validation macro-F1",
                        "threshold": signal_threshold,
                        "frozen_before_test": True,
                    },
                }
                candidate = self.registry.register(model_name, calibrated, feature_names, metadata)
                incumbent_metrics = self._incumbent_metrics(validation_x, validation_y, baseline)
                incumbent_regimes = self._incumbent_regime_metrics(validation_x, validation_y)
                promoted, message = self.registry.promote(
                    candidate,
                    validation_metrics,
                    incumbent_metrics,
                    data_quality_score=float(data_quality["quality_score"]),
                    leakage_checks_passed=leakage_checks_passed,
                    regime_macro_f1=regimes,
                    incumbent_regime_macro_f1=incumbent_regimes,
                )
                item = {
                    "version": candidate["version"],
                    "model_name": model_name,
                    "validation_metrics": validation_metrics,
                    "test_metrics": test_metrics,
                    "test_economic_metrics": economic_metrics,
                    "walk_forward_metrics": walk_forward_metrics,
                    "regime_validation_macro_f1": regimes,
                    "promoted": promoted,
                    "promotion_message": message,
                }
                results["models"].append(item)
                if optimize:
                    item["optimization"] = optimization
            except Exception as exc:
                logger.exception("Training failed for model %s", model_name)
                results["failures"].append({"model_name": model_name, "error": str(exc)})
        results["champion_version"] = self.registry.champion_version()
        if not results["models"]:
            results["status"] = "failed"
        if optimize:
            self.database.upsert_state("last_optimization_at", utc_now())
        return results

    def _split_boundaries(self, rows: int) -> tuple[int, int, int]:
        test_fraction = float(self.config["training"]["test_fraction"])
        validation_fraction = float(self.config["training"]["validation_fraction"])
        test_start = int(rows * (1 - test_fraction))
        validation_start = int(test_start - rows * validation_fraction)
        development_rows = validation_start
        train_end = int(development_rows * 0.8)
        return train_end, development_rows, test_start

    def _pipeline(self, model_name: str, params: dict[str, Any]) -> Pipeline:
        seed = int(self.config["training"]["random_seed"])
        if model_name == "logistic_regression":
            from sklearn.linear_model import LogisticRegression

            estimator = LogisticRegression(
                C=params.get("C", 1.0),
                max_iter=2000,
                random_state=seed,
            )
            scale = True
        elif model_name == "random_forest":
            from sklearn.ensemble import RandomForestClassifier

            estimator = RandomForestClassifier(
                n_estimators=params.get("n_estimators", 250),
                max_depth=params.get("max_depth", 8),
                min_samples_leaf=params.get("min_samples_leaf", 8),
                random_state=seed,
                n_jobs=-1,
            )
            scale = False
        elif model_name == "xgboost":
            try:
                from xgboost import XGBClassifier
            except ImportError as exc:
                raise RuntimeError("XGBoost is not installed.") from exc
            estimator = XGBClassifier(
                objective="multi:softprob",
                num_class=3,
                eval_metric="mlogloss",
                tree_method="hist",
                n_estimators=params.get("n_estimators", 250),
                max_depth=params.get("max_depth", 5),
                learning_rate=params.get("learning_rate", 0.05),
                subsample=params.get("subsample", 0.8),
                colsample_bytree=params.get("colsample_bytree", 0.8),
                reg_lambda=params.get("reg_lambda", 1.0),
                random_state=seed,
                n_jobs=4,
            )
            scale = False
        elif model_name == "lightgbm":
            try:
                from lightgbm import LGBMClassifier
            except ImportError as exc:
                raise RuntimeError("LightGBM is not installed.") from exc
            estimator = LGBMClassifier(
                objective="multiclass",
                num_class=3,
                n_estimators=params.get("n_estimators", 250),
                num_leaves=params.get("num_leaves", 15),
                max_depth=params.get("max_depth", 6),
                learning_rate=params.get("learning_rate", 0.05),
                min_child_samples=params.get("min_child_samples", 30),
                subsample=params.get("subsample", 0.8),
                colsample_bytree=params.get("colsample_bytree", 0.8),
                reg_alpha=params.get("reg_alpha", 0.0),
                reg_lambda=params.get("reg_lambda", 1.0),
                random_state=seed,
                n_jobs=4,
                verbosity=-1,
            )
            scale = False
        else:
            raise ValueError(f"Unknown model family: {model_name}")
        steps: list[tuple[str, Any]] = [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True))
        ]
        if scale:
            steps.append(("scaler", StandardScaler()))
        steps.append(("classifier", estimator))
        return Pipeline(steps)

    def _optimize(
        self,
        model_name: str,
        x: pd.DataFrame,
        y: pd.Series,
        trials: int,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Run Optuna on rolling chronological folds inside the training window."""
        try:
            import optuna
        except ImportError as exc:
            raise RuntimeError(
                "Optuna is not installed; install the declared project dependency."
            ) from exc
        if trials < 1:
            raise ValueError("Optuna trial budget must be positive.")
        studies_dir = resolve_path(self.config["application"]["artifacts_dir"]) / "studies"
        studies_dir.mkdir(parents=True, exist_ok=True)
        study_name = f"{model_name}-{utc_now().replace(':', '-')}-{uuid4().hex[:6]}"
        study_db = studies_dir / "optuna.sqlite3"
        study = optuna.create_study(
            study_name=study_name,
            storage=f"sqlite:///{study_db}",
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=int(self.config["training"]["random_seed"])),
        )
        folds = TimeSeriesSplit(n_splits=3, gap=1)

        def objective(trial: Any) -> float:
            params = self._suggest_params(model_name, trial)
            fold_scores = []
            for train_index, valid_index in folds.split(x):
                y_train = y.iloc[train_index]
                if set(y_train.unique()) != {0, 1, 2}:
                    raise ValueError(
                        "Training fold lacks one or more labels; collect a longer "
                        "chronological history before tuning."
                    )
                estimator = self._pipeline(model_name, params)
                estimator.fit(x.iloc[train_index], y_train)
                predicted = estimator.predict(x.iloc[valid_index])
                fold_scores.append(
                    f1_score(
                        y.iloc[valid_index],
                        predicted,
                        labels=[0, 1, 2],
                        average="macro",
                        zero_division=0,
                    )
                )
            return float(np.mean(fold_scores))

        study.optimize(objective, n_trials=trials, n_jobs=1, timeout=3600)
        best_parameters: dict[str, Any] = dict(study.best_params)
        result: dict[str, Any] = {
            "study_name": study_name,
            "optimization_metric": "mean chronological CV macro-F1",
            "best_value": study.best_value,
            "best_parameters": best_parameters,
            "trial_count": len(study.trials),
            "duration_seconds": sum(
                (trial.duration.total_seconds() if trial.duration else 0.0)
                for trial in study.trials
            ),
        }
        (studies_dir / f"{study_name}.json").write_text(
            json.dumps(
                {
                    **result,
                    "trials": [
                        {
                            "number": trial.number,
                            "value": trial.value,
                            "params": trial.params,
                            "state": trial.state.name,
                        }
                        for trial in study.trials
                    ],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return best_parameters, result

    @staticmethod
    def _suggest_params(model_name: str, trial: Any) -> dict[str, Any]:
        if model_name == "logistic_regression":
            return {"C": trial.suggest_float("C", 0.01, 10.0, log=True)}
        if model_name == "random_forest":
            return {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=100),
                "max_depth": trial.suggest_int("max_depth", 3, 16),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 3, 30),
            }
        if model_name == "xgboost":
            return {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=100),
                "max_depth": trial.suggest_int("max_depth", 2, 8),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
            }
        if model_name == "lightgbm":
            return {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=100),
                "num_leaves": trial.suggest_int("num_leaves", 7, 63, step=8),
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "min_child_samples": trial.suggest_int("min_child_samples", 10, 100, step=10),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            }
        raise ValueError(f"Unknown model family: {model_name}")

    @staticmethod
    def _majority_baseline(
        train_y: pd.Series, validation_y: pd.Series
    ) -> dict[str, dict[str, float]]:
        majority = int(train_y.mode().iloc[0])
        probabilities = np.zeros((len(validation_y), 3), dtype=float)
        probabilities[:, majority] = 1.0
        return {"majority_class": classification_metrics(validation_y.to_numpy(), probabilities)}

    def _rule_baselines(
        self, validation_frame: pd.DataFrame, validation_y: pd.Series
    ) -> dict[str, dict[str, float]]:
        actual = validation_y.to_numpy()
        result: dict[str, dict[str, float]] = {}
        for name, class_index in (
            ("always_down", 0),
            ("always_neutral", 1),
            ("always_up", 2),
        ):
            probabilities = np.zeros((len(actual), 3), dtype=float)
            probabilities[:, class_index] = 1.0
            result[name] = classification_metrics(actual, probabilities)

        if "return_1" in validation_frame:
            movement = validation_frame["return_1"].fillna(0).to_numpy()
        else:
            movement = np.zeros(len(actual))
        threshold = float(self.config["features"]["estimated_round_trip_cost_bps"]) / 10_000
        predicted = np.where(movement > threshold, 2, np.where(movement < -threshold, 0, 1))
        probabilities = np.full((len(actual), 3), 0.05, dtype=float)
        probabilities[np.arange(len(actual)), predicted] = 0.90
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        result["previous_direction_momentum"] = classification_metrics(actual, probabilities)
        return result

    @staticmethod
    def _regime_metrics(model: Any, x: pd.DataFrame, y: pd.Series) -> dict[str, float]:
        if "atr_14_pct" not in x:
            return {}
        threshold = float(x["atr_14_pct"].median())
        probabilities = model.predict_proba(x)
        predicted = probabilities.argmax(axis=1)
        result: dict[str, float] = {}
        for name, mask in (
            ("lower_volatility", x["atr_14_pct"].to_numpy() <= threshold),
            ("higher_volatility", x["atr_14_pct"].to_numpy() > threshold),
        ):
            if mask.any():
                result[name] = float(
                    f1_score(
                        y.to_numpy()[mask],
                        predicted[mask],
                        labels=[0, 1, 2],
                        average="macro",
                        zero_division=0,
                    )
                )
        return result

    def _select_signal_threshold(self, y_true: np.ndarray, probabilities: np.ndarray) -> float:
        """Select a neutral-abstention threshold using validation rows only."""
        configured = float(self.config["training"]["signal_probability_threshold"])
        candidates = sorted(set(np.round(np.append(np.arange(0.34, 0.81, 0.02), configured), 4)))
        best_threshold = configured
        best_score = -1.0
        base_prediction = probabilities.argmax(axis=1)
        confidence = probabilities.max(axis=1)
        for threshold in candidates:
            predicted = base_prediction.copy()
            predicted[(predicted != 1) & (confidence < threshold)] = 1
            score = f1_score(
                y_true,
                predicted,
                labels=[0, 1, 2],
                average="macro",
                zero_division=0,
            )
            if score > best_score:
                best_score = float(score)
                best_threshold = float(threshold)
        return best_threshold

    def _walk_forward_metrics(
        self, model_name: str, x: pd.DataFrame, y: pd.Series
    ) -> dict[str, Any]:
        """Evaluate expanding chronological windows without tuning on test data."""
        split_count = min(
            int(self.config["training"]["walk_forward_splits"]),
            max(2, len(x) // 100),
        )
        if split_count < 2:
            return {"folds": 0, "macro_f1_mean": None, "macro_f1_std": None}
        splitter = TimeSeriesSplit(n_splits=split_count, gap=1)
        fold_scores = []
        for train_indices, test_indices in splitter.split(x):
            fold_y = y.iloc[train_indices]
            if set(fold_y.unique()) != {0, 1, 2}:
                continue
            estimator = self._pipeline(model_name, {})
            estimator.fit(x.iloc[train_indices], fold_y)
            predicted = estimator.predict(x.iloc[test_indices])
            fold_scores.append(
                float(
                    f1_score(
                        y.iloc[test_indices],
                        predicted,
                        labels=[0, 1, 2],
                        average="macro",
                        zero_division=0,
                    )
                )
            )
        return {
            "folds": len(fold_scores),
            "macro_f1_by_fold": fold_scores,
            "macro_f1_mean": (float(np.mean(fold_scores)) if fold_scores else None),
            "macro_f1_std": (float(np.std(fold_scores)) if fold_scores else None),
        }

    def _incumbent_metrics(
        self,
        x: pd.DataFrame,
        y: pd.Series,
        baselines: dict[str, dict[str, float]],
    ) -> dict[str, float]:
        version = self.registry.champion_version()
        if version is None:
            return baselines["majority_class"]
        bundle = self.registry.load(version)
        if bundle["feature_names"] != list(x.columns):
            return {
                "macro_f1": 1.0,
                "brier_score": 0.0,
            }
        probabilities = bundle["model"].predict_proba(x)
        return classification_metrics(y.to_numpy(), probabilities)

    def _incumbent_regime_metrics(self, x: pd.DataFrame, y: pd.Series) -> dict[str, float] | None:
        version = self.registry.champion_version()
        if version is None:
            return None
        bundle = self.registry.load(version)
        if bundle["feature_names"] != list(x.columns):
            return None
        return self._regime_metrics(bundle["model"], x, y)

    @staticmethod
    def _code_version() -> str:
        try:
            completed = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=Path(__file__).resolve().parents[3],
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            )
            return completed.stdout.strip()
        except Exception:
            return "unavailable"

    @staticmethod
    def _dependency_versions() -> dict[str, str | None]:
        versions: dict[str, str | None] = {}
        for name in (
            "scikit-learn",
            "xgboost",
            "lightgbm",
            "optuna",
            "dukascopy-python",
        ):
            try:
                versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                versions[name] = None
        return versions
