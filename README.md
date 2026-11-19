# AurumMind AI

**Advanced AI Platform for Gold (XAUUSD) Market Research**

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

AurumMind AI is a production-grade artificial intelligence platform dedicated to Gold (XAUUSD) market research and probabilistic forecasting. The system analyzes historical market data, detects statistical patterns, studies market regimes, and develops Machine Learning and Deep Learning models to estimate possible future market movements.

**Important:** This platform is strictly educational and research-oriented. It does NOT execute financial transactions, connect to live order execution, or generate personalized real-money trading instructions.

## Project Status

**Phase 1: Data Engineering** ✅ COMPLETE

- ✅ Historical data ingestion (CSV and Parquet)
- ✅ Schema validation with Pydantic
- ✅ Comprehensive data quality validation
- ✅ UTC timestamp normalization
- ✅ Duplicate detection
- ✅ OHLC relationship validation
- ✅ Missing candle detection
- ✅ Suspicious price jump detection
- ✅ Data quality reporting
- ✅ Unit tests with pytest

**Phase 2: Exploratory Data Analysis** ✅ COMPLETE

- ✅ Statistical analysis (returns, volatility, distributions)
- ✅ Interactive Plotly visualizations
- ✅ Jupyter notebooks (data audit, exploratory analysis)
- ✅ Candle feature analysis
- ✅ Volume analysis
- ✅ Intraday pattern detection
- ✅ Outlier detection (IQR, Z-score)
- ✅ Automated EDA report generation
- ✅ Comprehensive test coverage

**Phase 3: Feature Engineering** ✅ COMPLETE

- ✅ Technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands, ATR, ADX, Stochastic, ROC, OBV)
- ✅ Price action features (returns, momentum, volatility, gaps, market structure)
- ✅ Support/resistance level detection
- ✅ 80+ engineered features
- ✅ No look-ahead bias verification
- ✅ Feature pipeline with validation
- ✅ Automated feature generation script
- ✅ Parquet storage with metadata
- ✅ Comprehensive test coverage

**Phase 4: Machine Learning Models** ✅ COMPLETE

- ✅ Baseline models (Logistic Regression, Random Forest)
- ✅ Advanced boosting models (XGBoost, LightGBM)
- ✅ Classification and regression support
- ✅ Model trainer with temporal splits
- ✅ Target variable creation (binary/multi-class direction, returns)
- ✅ Feature preparation and scaling
- ✅ Model evaluation metrics (accuracy, precision, recall, F1, ROC-AUC, directional accuracy)
- ✅ Model persistence and metadata tracking
- ✅ Comprehensive test coverage

**Phase 5: Model Evaluation & Validation** ✅ COMPLETE

- ✅ Walk-forward validation (rolling and anchored windows)
- ✅ Regime analysis (volatility and trend regimes)
- ✅ Regime-specific performance evaluation
- ✅ Backtesting engine with realistic costs
- ✅ Transaction cost and slippage simulation
- ✅ Risk-adjusted metrics (Sharpe, Sortino, Calmar)
- ✅ Performance attribution (time-based, confidence-based)
- ✅ Error pattern analysis
- ✅ Comprehensive evaluation script
- ✅ 45 new unit tests

## Technology Stack

### Core
- **Python 3.12+** - Primary programming language
- **uv** - Fast Python package manager
- **Ruff** - Linting and formatting
- **MyPy** - Static type checking
- **Pytest** - Testing framework

### Data Engineering
- **Pandas** - Data manipulation
- **NumPy** - Numerical computing
- **PyArrow** - Parquet file handling
- **Pydantic** - Data validation
- **YAML** - Configuration management

### Machine Learning (upcoming phases)
- Scikit-learn, XGBoost, LightGBM
- PyTorch for deep learning

### MLOps (upcoming phases)
- MLflow for experiment tracking
- Apache Airflow for orchestration

### Deployment (upcoming phases)
- FastAPI for REST API
- Streamlit for dashboard
- Docker & Docker Compose

## Architecture

```
aurummind-ai/
├── configs/              # YAML configuration files
├── data/
│   ├── raw/             # Raw ingested data (Parquet)
│   ├── interim/         # Intermediate processed data
│   └── processed/       # Final processed features
├── reports/             # Generated analysis reports
│   └── eda/            # EDA reports and visualizations
├── src/aurummind_ai/    # Main source code
│   ├── ingestion/       # Data loading and ingestion ✅
│   ├── validation/      # Data quality validation ✅
│   ├── market_analysis/ # Statistical analysis & visualization ✅
│   ├── features/        # Feature engineering ✅
│   ├── models/          # Machine learning models ✅
│   ├── evaluation/      # Model evaluation & validation ✅
│   └── utils/           # Utility functions ✅
├── notebooks/           # Jupyter notebooks for EDA ✅
├── tests/               # Unit and integration tests ✅
└── scripts/             # Utility scripts ✅
```

## Installation

### Prerequisites

- Python 3.12 or higher
- [uv](https://github.com/astral-sh/uv) package manager

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd aurummind-ai
```

2. Create environment and install dependencies:
```bash
uv venv
uv pip install -e .
```

3. Activate virtual environment:
```bash
# Windows
.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate
```

4. Copy environment template:
```bash
cp .env.example .env
```

## Phase 1, 2 & 3: Quick Start

### 1. Generate Sample Data

```bash
python scripts/generate_sample_data.py
```

This creates synthetic XAUUSD data for testing:
- `XAUUSD_1h_sample.csv` - 1-hour timeframe (30 days)
- `XAUUSD_4h_sample.csv` - 4-hour timeframe (90 days)
- `XAUUSD_1d_sample.csv` - Daily timeframe (1 year)

### 2. Test Ingestion Pipeline (Phase 1)

```bash
python scripts/test_ingestion.py
```

This will:
- Load sample data files
- Validate data quality
- Detect any issues (duplicates, missing values, invalid OHLC, etc.)
- Generate quality reports
- Save processed data to `data/raw/`

### 3. Generate EDA Reports (Phase 2)

```bash
python scripts/generate_eda_report.py
```

This will:
- Analyze all sample datasets
- Generate comprehensive statistics
- Create interactive visualizations
- Save reports to `reports/eda/`
- Output HTML charts

### 4. Generate Features (Phase 3)

```bash
python scripts/generate_features.py
```

This will:
- Generate 80+ features (technical indicators + price action)
- Process all timeframes (1h, 4h, 1d)
- Validate features
- Save to `data/processed/` as Parquet
- Include metadata JSON files

### 5. Explore with Jupyter (Phase 2)

```bash
jupyter notebook notebooks/
```

Open and run:
- `01_data_audit.ipynb` - Data quality audit
- `02_exploratory_analysis.ipynb` - Comprehensive EDA

### 6. Run Unit Tests

```bash
pytest tests/ -v
```

Expected output:
- Phase 1 tests: 24 tests PASSED
- Phase 2 tests: 11 tests PASSED
- Phase 3 tests: 22 tests PASSED
- **Total: 57 tests PASSED**

## Data Quality Validation

The validation system performs comprehensive checks:

### Schema Validation
- Required columns: timestamp, open, high, low, close, volume
- Timestamp normalization to UTC
- Proper data types

### Data Integrity
- ✓ Duplicate timestamp detection
- ✓ Missing value detection
- ✓ OHLC relationship validation (High ≥ Open/Close/Low, Low ≤ Open/Close/High)
- ✓ Negative or zero price detection
- ✓ Suspicious price jump detection
- ✓ Missing candle detection

### Quality Scoring
Each dataset receives a quality score (0-100) based on:
- Number and severity of issues found
- ERROR issues: -20 points (max)
- WARNING issues: -10 points (max)

## Configuration

Edit `configs/data.yaml` to customize:

```yaml
project:
  name: AurumMind AI
  instrument: XAUUSD
  data_source: Dukascopy

data:
  base_dir: data
  raw_dir: data/raw
  interim_dir: data/interim
  processed_dir: data/processed

timeframes:
  - 1h
  - 4h
  - 1d

validation:
  timezone: UTC
  allow_negative_prices: false
  require_unique_timestamps: true
```

## Using Real Market Data

To ingest your own XAUUSD data:

1. Ensure your CSV has these columns:
   - timestamp (any parseable datetime format)
   - open, high, low, close (prices)
   - volume

2. Place your file in `data/raw/`

3. Use the ingestion pipeline:

```python
from aurummind_ai.ingestion.pipeline import IngestionPipeline

pipeline = IngestionPipeline()

df, report = pipeline.ingest_file(
    file_path="data/raw/your_xauusd_data.csv",
    timeframe="1h",  # or "4h", "1d"
    source_name="YourDataProvider",
    file_format="csv",
)

print(report["stages"]["validation"]["quality_score"])
```

## Development

### Code Quality

```bash
# Format code
ruff format .

# Lint code
ruff check .

# Type checking
mypy src/
```

### Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=aurummind_ai --cov-report=html

# Run specific test file
pytest tests/unit/test_data_loader.py -v
```

## Project Roadmap

- [x] **Phase 1: Data Engineering** - Ingestion, validation, quality reporting
- [x] **Phase 2: Exploratory Data Analysis** - Statistical analysis, visualizations
- [x] **Phase 3: Feature Engineering** - Technical indicators, market structure
- [ ] **Phase 4: Machine Learning** - Baseline and advanced models
- [ ] **Phase 5: Model Evaluation** - Metrics, backtesting, validation
- [ ] **Phase 6: Explainable AI** - SHAP, feature importance
- [ ] **Phase 7: MLOps** - MLflow, model registry, versioning
- [ ] **Phase 8: Orchestration** - Airflow DAGs, scheduling
- [ ] **Phase 9: API** - FastAPI REST endpoints
- [ ] **Phase 10: Dashboard** - Streamlit interactive UI
- [ ] **Phase 11: Testing & Deployment** - Docker, CI/CD

## Contributing

This is a research and educational project. Contributions, suggestions, and feedback are welcome.

## Disclaimer

**This software is for educational and research purposes only.**

- NOT financial advice
- NOT a trading system
- NOT guaranteed to be profitable
- NO warranties of any kind

Trading financial instruments involves substantial risk of loss. Past performance does not indicate future results.

## License

MIT License - see LICENSE file for details.

## Contact

For questions or collaboration: zakariaallam536@gmail.com

---

**Built with ❤️ for AI research and financial market analysis**