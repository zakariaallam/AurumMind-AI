# AurumMind AI repository audit

## Initial state

The repository is a Python 3.12+ research project. It already contains:

- CSV/Parquet OHLCV loading, UTC normalization, validation, and raw Parquet
  storage in `src/aurummind_ai/ingestion/` and `validation/`.
- EDA/statistics, Plotly chart utilities, notebooks, and generated reports.
- Technical-indicator and price-action feature engineering.
- Scikit-learn Logistic Regression and Random Forest models, plus XGBoost and
  LightGBM model wrappers.
- Temporal train/test splits, walk-forward validation, regime analysis,
  cost-configurable backtesting, and explainability utilities.
- Synthetic data generators and unit tests. The checked-in sample CSVs are
  generated test fixtures, not real provider data.

The configured provider name `Dukascopy` is not a downloader: the existing
loader only reads local files. There is no provider adapter, M15 data model,
durable operational database, model registry, live inference path, API,
Streamlit application, or persistent update/retraining worker.

## Confirmed integration gaps

1. `configs/data.yaml` supports only `1h`, `4h`, and `1d`; its ingestion
   metadata model has the same timeframe restriction. The target product needs
   M15 (and optional M5/H1/H4).
2. `scripts/generate_features.py` writes timestamped Parquet files, and
   `scripts/train_models.py` reads those files. In contrast,
   `scripts/evaluate_models.py` expects a hard-coded CSV path named
   `xauusd_features.csv`, so that evaluator does not consume the normal
   feature-generator output.
3. Existing feature missing-value handling can backward-fill the beginning of
   a series. That is not suitable for a strict causal live-training pipeline.
   The new pipeline must retain warm-up NaNs and drop invalid training rows;
   it must not fit an imputer on future/test rows.
4. Existing model training covers binary classification, whereas the requested
   live task is three-class UP/DOWN/NEUTRAL and uses a 15-minute forward label.
5. Existing dependencies declare PyTorch, MLflow, Airflow, FastAPI, and
   Streamlit, but the inspected application source has no neural-network,
   tracking, scheduling, API, or dashboard implementation. Declared packages
   are not evidence that these services are operational.
6. There are no persisted training jobs, prediction outcomes, download
   provenance records, candidate/champion promotion rules, or rollback.
7. The README marks phases 1-6 complete in its overview but leaves phases 4-6
   unchecked in its roadmap.

## Provider verification

The provider adapter uses the public `dukascopy-python` package and its
documented Dukascopy candle API. The provider interface supports UTC candles,
15-minute intervals, and historical date ranges; Dukascopy's official page
describes its historical-data export for forex, commodities, and indices.
The library was present on PyPI as version 7.0.0 when this implementation was
researched (2026-11-19), and its GitHub repository had a commit dated
2026-11-17. This is a third-party client for Dukascopy's public historical
feed, not an official SDK. Dukascopy prices are provider-specific, and XAUUSD
volume is recorded as provider-reported volume, not asserted to be centralized
exchange volume. No account credentials are assumed.

The adapter records source and request provenance, uses bounded request
windows, retrying HTTP sessions, timeouts, UTC, and explicit errors. It never
falls back to synthetic data or another provider. Network access, provider
availability, and applicable data-use terms still need to be verified by the
operator at runtime.

## Implementation direction

The new live-market path is additive and isolated from the existing 1h/4h/1d
research interfaces. It supplies M15/M5/H1/H4 download and storage, data
quality reports, a causal feature/target pipeline, local SQLite metadata,
chronological model evaluation, calibration, model artifact/version
management, prediction tracking, CLI/API/dashboard entry points, and tests.
The existing research modules and sample files are preserved.

## Verification limitations

No live data download, model training, validation score, or prediction should
be reported as completed unless it was actually run against provider data.
Unit tests may use synthetic fixtures strictly for software correctness. A
real champion and its performance remain contingent on the operator's
historical download, data quality, available model dependencies/resources,
and validation results.
