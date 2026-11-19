"""Unit tests for DataLoader."""

import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from aurummind_ai.ingestion.data_loader import DataLoader


@pytest.fixture
def temp_data_dir():
    """Create a temporary directory for test data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_ohlcv_data():
    """Create sample OHLCV data for testing."""
    data = {
        "timestamp": [
            "2024-01-01 00:00:00",
            "2024-01-01 01:00:00",
            "2024-01-01 02:00:00",
        ],
        "open": [2050.0, 2051.5, 2052.0],
        "high": [2055.0, 2056.0, 2057.0],
        "low": [2048.0, 2049.0, 2050.0],
        "close": [2053.0, 2054.0, 2055.0],
        "volume": [1000.0, 1100.0, 1200.0],
    }
    return pd.DataFrame(data)


@pytest.fixture
def config_file(temp_data_dir):
    """Create a temporary config file."""
    config_content = f"""
project:
  name: AurumMind AI
  instrument: XAUUSD
  data_source: Test

data:
  base_dir: {temp_data_dir}
  raw_dir: {temp_data_dir}/raw
  interim_dir: {temp_data_dir}/interim
  processed_dir: {temp_data_dir}/processed

timeframes:
  - 1h
  - 4h
  - 1d

columns:
  timestamp: timestamp
  open: open
  high: high
  low: low
  close: close
  volume: volume

validation:
  timezone: UTC
  allow_negative_prices: false
  require_unique_timestamps: true
"""
    config_path = temp_data_dir / "config.yaml"
    config_path.write_text(config_content)
    return config_path


def test_data_loader_initialization(config_file):
    """Test DataLoader initialization."""
    loader = DataLoader(str(config_file))
    assert loader.config is not None
    assert loader.raw_dir.exists()
    assert loader.interim_dir.exists()


def test_load_csv_success(sample_ohlcv_data, temp_data_dir, config_file):
    """Test successful CSV loading."""
    csv_path = temp_data_dir / "test_data.csv"
    sample_ohlcv_data.to_csv(csv_path, index=False)
    
    loader = DataLoader(str(config_file))
    df, metadata = loader.load_csv(str(csv_path), "1h", "TestSource")
    
    assert len(df) == 3
    assert list(df.columns) == ["timestamp", "open", "high", "low", "close", "volume"]
    assert metadata.instrument == "XAUUSD"
    assert metadata.timeframe == "1h"
    assert metadata.source_name == "TestSource"
    assert metadata.total_records == 3


def test_load_csv_missing_file(config_file):
    """Test loading non-existent CSV file."""
    loader = DataLoader(str(config_file))
    
    with pytest.raises(FileNotFoundError):
        loader.load_csv("nonexistent.csv", "1h", "TestSource")


def test_load_csv_missing_columns(temp_data_dir, config_file):
    """Test loading CSV with missing required columns."""
    incomplete_data = pd.DataFrame({
        "timestamp": ["2024-01-01 00:00:00"],
        "open": [2050.0],
        "high": [2055.0],
    })
    
    csv_path = temp_data_dir / "incomplete.csv"
    incomplete_data.to_csv(csv_path, index=False)
    
    loader = DataLoader(str(config_file))
    
    with pytest.raises(ValueError, match="Missing required columns"):
        loader.load_csv(str(csv_path), "1h", "TestSource")


def test_timestamp_normalization(sample_ohlcv_data, temp_data_dir, config_file):
    """Test that timestamps are normalized to UTC."""
    csv_path = temp_data_dir / "test_data.csv"
    sample_ohlcv_data.to_csv(csv_path, index=False)
    
    loader = DataLoader(str(config_file))
    df, _ = loader.load_csv(str(csv_path), "1h", "TestSource")
    
    # Check that all timestamps are timezone-aware UTC
    assert df["timestamp"].dt.tz is not None
    assert all(ts.tzinfo == timezone.utc for ts in df["timestamp"])


def test_data_sorting(temp_data_dir, config_file):
    """Test that data is sorted by timestamp."""
    unsorted_data = pd.DataFrame({
        "timestamp": [
            "2024-01-01 02:00:00",
            "2024-01-01 00:00:00",
            "2024-01-01 01:00:00",
        ],
        "open": [2052.0, 2050.0, 2051.5],
        "high": [2057.0, 2055.0, 2056.0],
        "low": [2050.0, 2048.0, 2049.0],
        "close": [2055.0, 2053.0, 2054.0],
        "volume": [1200.0, 1000.0, 1100.0],
    })
    
    csv_path = temp_data_dir / "unsorted.csv"
    unsorted_data.to_csv(csv_path, index=False)
    
    loader = DataLoader(str(config_file))
    df, _ = loader.load_csv(str(csv_path), "1h", "TestSource")
    
    # Check that data is sorted
    assert df["timestamp"].is_monotonic_increasing


def test_save_raw_data(sample_ohlcv_data, temp_data_dir, config_file):
    """Test saving raw data to Parquet."""
    csv_path = temp_data_dir / "test_data.csv"
    sample_ohlcv_data.to_csv(csv_path, index=False)
    
    loader = DataLoader(str(config_file))
    df, metadata = loader.load_csv(str(csv_path), "1h", "TestSource")
    
    output_path = loader.save_raw_data(df, metadata)
    
    assert output_path.exists()
    assert output_path.suffix == ".parquet"
    
    # Check that metadata was also saved
    metadata_path = output_path.with_suffix(".json")
    assert metadata_path.exists()
    
    # Verify data can be loaded back
    loaded_df = pd.read_parquet(output_path)
    assert len(loaded_df) == len(df)
