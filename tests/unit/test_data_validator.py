"""Unit tests for DataValidator."""

from datetime import datetime, timezone

import pandas as pd
import pytest

from aurummind_ai.ingestion.schema import DataSourceMetadata
from aurummind_ai.validation.data_validator import DataValidator


@pytest.fixture
def valid_ohlcv_data():
    """Create valid OHLCV data for testing."""
    timestamps = pd.date_range("2024-01-01", periods=10, freq="1H", tz="UTC")
    data = {
        "timestamp": timestamps,
        "open": [2050.0 + i for i in range(10)],
        "high": [2055.0 + i for i in range(10)],
        "low": [2048.0 + i for i in range(10)],
        "close": [2053.0 + i for i in range(10)],
        "volume": [1000.0 + i * 10 for i in range(10)],
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_metadata():
    """Create sample metadata."""
    return DataSourceMetadata(
        source_name="TestSource",
        instrument="XAUUSD",
        timeframe="1h",
        ingestion_timestamp=datetime.now(timezone.utc),
        file_path="test.csv",
        total_records=10,
    )


def test_validator_initialization():
    """Test DataValidator initialization."""
    validator = DataValidator()
    assert validator.price_jump_threshold == 0.10
    assert validator.volume_outlier_threshold == 5.0


def test_validate_valid_data(valid_ohlcv_data, sample_metadata):
    """Test validation of valid data."""
    validator = DataValidator()
    report = validator.validate(valid_ohlcv_data, sample_metadata)
    
    assert report.total_records == 10
    assert report.quality_score > 90.0
    assert report.passed_validation is True


def test_detect_duplicates(valid_ohlcv_data, sample_metadata):
    """Test detection of duplicate timestamps."""
    # Add duplicate timestamp
    duplicate_row = valid_ohlcv_data.iloc[0:1].copy()
    df_with_duplicates = pd.concat([valid_ohlcv_data, duplicate_row], ignore_index=True)
    
    validator = DataValidator()
    report = validator.validate(df_with_duplicates, sample_metadata)
    
    assert report.duplicate_timestamps > 0
    assert report.has_errors()
    assert report.passed_validation is False


def test_detect_missing_values(valid_ohlcv_data, sample_metadata):
    """Test detection of missing values."""
    df_with_missing = valid_ohlcv_data.copy()
    df_with_missing.loc[0, "close"] = None
    df_with_missing.loc[1, "volume"] = None
    
    validator = DataValidator()
    report = validator.validate(df_with_missing, sample_metadata)
    
    assert "close" in report.missing_values
    assert "volume" in report.missing_values
    assert report.has_errors()


def test_detect_invalid_ohlc_relationships(valid_ohlcv_data, sample_metadata):
    """Test detection of invalid OHLC relationships."""
    df_invalid = valid_ohlcv_data.copy()
    
    # Make high < low (invalid)
    df_invalid.loc[0, "high"] = 2000.0
    df_invalid.loc[0, "low"] = 2100.0
    
    validator = DataValidator()
    report = validator.validate(df_invalid, sample_metadata)
    
    assert report.invalid_ohlc_relationships > 0
    assert report.has_errors()


def test_detect_negative_prices(valid_ohlcv_data, sample_metadata):
    """Test detection of negative prices."""
    df_negative = valid_ohlcv_data.copy()
    df_negative.loc[0, "close"] = -100.0
    
    validator = DataValidator()
    report = validator.validate(df_negative, sample_metadata)
    
    assert report.negative_prices > 0
    assert report.has_errors()


def test_detect_zero_prices(valid_ohlcv_data, sample_metadata):
    """Test detection of zero prices."""
    df_zero = valid_ohlcv_data.copy()
    df_zero.loc[0, "open"] = 0.0
    
    validator = DataValidator()
    report = validator.validate(df_zero, sample_metadata)
    
    assert report.zero_prices > 0
    # Zero prices should be a WARNING, not ERROR
    assert any(issue.category == "Zero Prices" for issue in report.issues)


def test_detect_price_jumps(valid_ohlcv_data, sample_metadata):
    """Test detection of suspicious price jumps."""
    df_jumps = valid_ohlcv_data.copy()
    
    # Create a 20% price jump (above default 10% threshold)
    df_jumps.loc[5, "close"] = df_jumps.loc[4, "close"] * 1.25
    
    validator = DataValidator()
    report = validator.validate(df_jumps, sample_metadata)
    
    assert report.suspicious_price_jumps > 0
    # Price jumps should be WARNING
    assert any(issue.category == "Price Jumps" for issue in report.issues)


def test_detect_missing_candles(sample_metadata):
    """Test detection of missing candles in time series."""
    # Create data with missing candles
    timestamps = pd.date_range("2024-01-01", periods=5, freq="1H", tz="UTC")
    # Skip timestamp at index 2
    timestamps_with_gap = timestamps[[0, 1, 3, 4]]
    
    data = {
        "timestamp": timestamps_with_gap,
        "open": [2050.0, 2051.0, 2053.0, 2054.0],
        "high": [2055.0, 2056.0, 2058.0, 2059.0],
        "low": [2048.0, 2049.0, 2051.0, 2052.0],
        "close": [2053.0, 2054.0, 2056.0, 2057.0],
        "volume": [1000.0, 1100.0, 1300.0, 1400.0],
    }
    df_with_gaps = pd.DataFrame(data)
    
    validator = DataValidator()
    report = validator.validate(df_with_gaps, sample_metadata)
    
    assert report.missing_candles > 0


def test_quality_score_calculation(valid_ohlcv_data, sample_metadata):
    """Test quality score calculation."""
    df_with_issues = valid_ohlcv_data.copy()
    
    # Add some issues
    df_with_issues.loc[0, "close"] = 0.0  # Zero price (WARNING)
    df_with_issues.loc[5, "close"] = df_with_issues.loc[4, "close"] * 1.25  # Price jump
    
    validator = DataValidator()
    report = validator.validate(df_with_issues, sample_metadata)
    
    # Quality score should be less than 100 but greater than 0
    assert 0.0 < report.quality_score < 100.0


def test_empty_dataset(sample_metadata):
    """Test validation of empty dataset."""
    empty_df = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    
    validator = DataValidator()
    report = validator.validate(empty_df, sample_metadata)
    
    assert report.total_records == 0
    assert report.has_errors()
    assert any(issue.category == "Empty Dataset" for issue in report.issues)


def test_report_summary(valid_ohlcv_data, sample_metadata):
    """Test generation of report summary."""
    validator = DataValidator()
    report = validator.validate(valid_ohlcv_data, sample_metadata)
    
    summary = report.get_summary()
    
    assert "XAUUSD" in summary
    assert "1h" in summary
    assert "Quality Score" in summary
    assert "PASSED" in summary or "FAILED" in summary
