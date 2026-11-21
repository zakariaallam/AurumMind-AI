# Operational XAUUSD Platform

## What this implementation does

This is a research-only forecasting workflow. It does not connect to a broker, submit trades, or provide individualized financial advice. The operational path uses a configurable M15 primary timeframe and produces a one-class forecast (`UP`, `DOWN`, or `NEUTRAL`) with calibrated class probabilities for the next configured horizon.

The existing CSV files under `data/raw/samples/` are synthetic test fixtures. They are useful for the original offline ingestion/EDA/feature examples, but the operational training service refuses to use them.

## Market-data source and provenance

- Provider: Dukascopy's [public historical data service](https://www.dukascopy.com/swiss/english/marketwatch/historical/), accessed through the third-party [`dukascopy-python` client](https://pypi.org/project/dukascopy-python/).
- Instrument: provider symbol `XAU/USD`; application instrument label `XAUUSD`.
- Price side: BID.
- Default timeframe and history start: M15 from 2020-01-01 UTC, configured in `configs/system.yaml`.
- No Dukascopy account credentials are configured or sent by this adapter. Confirm provider access, licensing/terms, and feed behavior before relying on the data.
- The client-reported volume is retained as provider volume; it is not assumed to be centralized exchange volume.
- Each download writes raw Parquet batches and a JSON provenance sidecar under `data/raw/dukascopy/XAUUSD/<timeframe>/`, a consolidated processed Parquet file under `data/processed/dukascopy/XAUUSD/<timeframe>/`, a quality report under `data/reports/`, and a provenance row in `data/state/aurummind.sqlite3`.
- Unfiltered provider candle rows in the normalized OHLCV schema, provider metadata, and the quality report are saved before invalid batches are rejected from processed training data. These Parquet batches are not byte-exact HTTP payload archives. Invalid timestamps are kept in a separate `invalid_timestamp` raw partition.
- The quality gate checks duplicate and misaligned timestamps, finite/positive prices, OHLC relationships, nonnegative volume, incomplete bars, and gaps during approximate New York FX open hours. Weekend closures are not treated as missing bars. Holiday/provider-specific closures can still be counted as gaps.

The provider interface is third-party software, not an official Dukascopy SDK. Existing local history came from a recorded Dukascopy batch; the adapter has not been exercised against a new live provider request in this run. The local M15 history has been rebuilt and passed validation, but its source remains historical market data rather than a live feed.

## Modeling workflow

1. Normalize and validate closed provider candles; persist download provenance.
2. Generate causal price-action, rolling-return, volatility, EMA/SMA, RSI, MACD, ATR, ADX, stochastic, breakout, UTC-time, session, and previous-day-level features. Warm-up NaNs are retained; features are not backward-filled.
3. Label the next configured horizon as `UP`, `DOWN`, or `NEUTRAL`. The directional threshold is the greater of an ATR-scaled move and estimated round-trip cost. Non-contiguous future bars are excluded.
4. Split data chronologically into fit, calibration, validation, and untouched test windows. The pipeline requires enough labeled rows and every class in each window. Leakage checks, walk-forward metrics, class/probability metrics, and cost-aware test simulation are recorded.
5. Fit the configured scikit-learn, Random Forest, XGBoost, and LightGBM candidates. Sigmoid probability calibration uses a separate chronological calibration window. A signal threshold is selected on validation data and then frozen for the test report.
6. Store trusted local model bundles in the versioned registry. A candidate is promoted only if it passes quality, leakage, validation, and regime checks against the configured gates. The active champion pointer can be rolled back.
7. Save predictions and evaluate them only after the target candle is present in stored data.

This is **machine learning**, not deep learning: the operational candidates are conventional classifiers and gradient-boosted trees. The current workflow does not define or train a neural network. Optuna is available for bounded hyperparameter search; an optimization run may use substantial compute.

The cost-aware backtest is a simplified historical simulation using configured spread, commission, and slippage assumptions. It is not broker-execution simulation and is not evidence of profitability. Metrics are unknown until real provider data is downloaded and training is actually run.

## Setup and commands

Use Python 3.12+ and the project environment. From the repository root:

```bash
uv venv
uv pip install -e .
```

The project manifest also includes large dependencies for other research areas (including PyTorch and Airflow). The operational workflow itself uses the provider client, pandas/NumPy, PyArrow, scikit-learn, XGBoost, LightGBM, Optuna, FastAPI, and Streamlit.

Historical download requires an explicit confirmation because the default range can be large:

```bash
uv run aurummind data download --confirm
```

Review the configured start date before running. To constrain the request:

```bash
uv run aurummind data download --start 2024-01-01T00:00:00Z --confirm
uv run aurummind data validate
uv run aurummind features generate
uv run aurummind train baseline
uv run aurummind evaluate
uv run aurummind predict
```

An Optuna search requires a second explicit confirmation:

```bash
uv run aurummind train optimize --confirm
```

If an earlier download saved raw files but did not create processed data, rebuild from that local batch without downloading again:

```bash
uv run aurummind data rebuild
uv run aurummind data validate
uv run aurummind train baseline
uv run aurummind evaluate
uv run aurummind predict
```

`data rebuild` revalidates the latest recorded raw batch and creates processed candles locally; use it when the historical download completed but was not admitted to processed storage. It does not contact the provider.

The incremental updater refuses to bootstrap missing history. For ongoing updates after processed history exists:

```bash
uv run aurummind data update
uv run aurummind worker --once
```

Run the dashboard with `uv run streamlit run app.py`. The dashboard can start/stop the verified local worker and queue data, validation, feature, and training jobs. The worker polls at the configured interval, never starts a large history download, and retrains only when its configured data/time conditions are met. It runs Optuna optimization when the configured optimization interval is due; the first eligible worker training may therefore run the configured trial budget.

## API and security

Start the API locally with:

```bash
uv run aurummind api
```

It binds to `127.0.0.1` by default. Read-only endpoints include `/api/health`, `/api/market/latest`, `/api/market/history`, `/api/prediction/latest`, `/api/prediction/history`, `/api/models/champion`, `/api/models/metrics`, `/api/data/status`, and `/api/jobs`. State-changing routes require a bearer token in `AURUMMIND_API_TOKEN`; without it, writes are disabled. Keep the token out of source control and do not expose the API publicly without an independently reviewed deployment/authentication design.

The champion is trained on the configured primary timeframe. Prediction requests for other timeframes are explicitly marked unavailable rather than applying a model to a timeframe on which it was not trained.

## Current limitations and operational cautions

- Dukascopy connectivity, actual candle coverage, volume semantics, and rate-limit behavior have not been verified by a live request here.
- This workspace currently contains processed Dukascopy M15 history, rebuilt from its saved raw batch and passing the configured quality gate. Training and champion availability depend on the local worker's current run; historical metrics are not a claim of future performance.
- Holiday gaps and provider-specific session behavior need review; the current FX-session calendar is approximate.
- Current orchestration is a local Python worker plus SQLite job leases. MLflow and Airflow are listed as dependencies but are not wired into this operational path.
- Dashboard start/stop controls verify worker processes through Linux `/proc`; those controls are Linux-specific.
- Local model artifacts use `joblib`. Treat registry artifacts as trusted executable files; never load a bundle supplied by an untrusted source.
- Back up the SQLite database, raw batches, processed data, provenance sidecars, and model registry together. Stop the worker before taking a consistent file-level backup.
- This project is research software, not a financial product. Historical metrics do not guarantee future results.
