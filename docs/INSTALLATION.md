# AurumMind AI - Installation Guide

## Prerequisites

- **Python 3.12 or higher** (Python 3.14 detected on your system works fine)
- **uv** package manager (already installed)
- **Git** (for version control)
- Approximately 500MB free disk space (for dependencies)

---

## Installation Steps

### 1. Navigate to Project Directory

```powershell
cd C:\Users\Youcode\Desktop\AurumMind-AI
```

### 2. Create Virtual Environment

```powershell
uv venv
```

This creates a `.venv` directory with an isolated Python environment.

### 3. Activate Virtual Environment

**Windows PowerShell:**
```powershell
.venv\Scripts\Activate.ps1
```

If you get an execution policy error, run:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then try activating again.

**Alternative (if PowerShell script execution is restricted):**
```powershell
.venv\Scripts\activate.bat
```

### 4. Install Dependencies

```powershell
uv pip install -e .
```

This installs all required packages including:
- pandas, numpy, pyarrow (data processing)
- pydantic, pyyaml (configuration)
- scikit-learn, xgboost, lightgbm (machine learning)
- torch (deep learning)
- mlflow (experiment tracking)
- fastapi, uvicorn (API)
- streamlit, plotly (dashboard)
- pytest, ruff, mypy (development tools)

**Note:** This may take 5-10 minutes depending on your internet connection.

### 5. Verify Installation

```powershell
python -c "import pandas, pytest, pydantic; print('✓ Core packages installed')"
```

Expected output:
```
✓ Core packages installed
```

### 6. Run Tests

```powershell
pytest tests/unit/ -v
```

Expected output: All tests passing (PASSED)

---

## Quick Start

### Generate Sample Data

```powershell
python scripts/generate_sample_data.py
```

This creates sample XAUUSD data files in `data/raw/samples/`

### Test Ingestion Pipeline

```powershell
python scripts/test_ingestion.py
```

This tests the complete data ingestion and validation pipeline.

---

## Alternative Installation (System Python)

If you prefer not to use a virtual environment:

```powershell
py -m pip install pandas numpy pyarrow pydantic pyyaml pytest ruff mypy
```

Install additional packages as needed for later phases.

---

## Troubleshooting

### Issue: PowerShell Script Execution Disabled

**Error:**
```
execution of scripts is disabled on this system
```

**Solution:**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Issue: uv command not found

**Solution:**
Install uv from https://github.com/astral-sh/uv or use pip:
```powershell
py -m pip install uv
```

### Issue: Python version mismatch

The project is configured for Python 3.12+. Your system has Python 3.14, which is compatible.

To verify:
```powershell
py --version
```

### Issue: Package installation fails

If `uv pip install` fails, try standard pip:
```powershell
py -m pip install -e .
```

### Issue: Import errors

Ensure virtual environment is activated:
```powershell
.venv\Scripts\activate
```

Or install in system Python:
```powershell
py -m pip install -e .
```

---

## Development Setup

### Code Quality Tools

```powershell
# Format code
ruff format .

# Lint code
ruff check .

# Type checking
mypy src/
```

### Testing

```powershell
# Run all tests
pytest

# Run with coverage
pytest --cov=aurummind_ai --cov-report=html

# Run specific test file
pytest tests/unit/test_data_loader.py -v
```

---

## Directory Structure After Installation

```
AurumMind-AI/
├── .venv/                      # Virtual environment (created)
├── data/
│   ├── raw/
│   │   └── samples/           # Sample data (after generation)
│   ├── interim/
│   └── processed/
├── src/aurummind_ai/          # Source code
├── tests/                      # Test suite
├── scripts/                    # Utility scripts
├── configs/                    # Configuration files
├── docs/                       # Documentation
├── pyproject.toml             # Project configuration
└── README.md                  # Main documentation
```

---

## Next Steps

1. ✅ Installation complete
2. ✅ Generate sample data
3. ✅ Run tests
4. ✅ Test ingestion pipeline
5. → Review Phase 1 completion document
6. → Proceed to Phase 2 (EDA)

---

## Support

For issues or questions:
- Check `docs/PHASE_1_COMPLETION.md` for implementation details
- Review `README.md` for usage examples
- Contact: zakariaallam536@gmail.com

---

Last updated: January 18, 2026
