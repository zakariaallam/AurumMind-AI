"""Unit tests for MarketStatistics."""

import numpy as np
import pandas as pd
import pytest

from aurummind_ai.market_analysis.statistics import MarketStatistics


@pytest.fixture
def sample_ohlcv_data():
    """Create sample OHLCV data for testing."""
    np.random.seed(42)
    
    dates = pd.date_range("2024-01-01", periods=100, freq="1H", tz="UTC")
    prices = 2050 + np.cumsum(np.random.randn(100) * 2)
    
    data = {
        "timestamp": dates,
        "open": prices + np.random.randn(100) * 0.5,
        "high": prices + np.abs(np.random.randn(100)) * 2,
        "low": prices - np.abs(np.random.randn(100)) * 2,
        "close": prices + np.random.randn(100) * 0.5,
        "volume": 1000 + np.random.randn(100) * 100,
    }
    
    df = pd.DataFrame(data)
    
    # Ensure OHLC relationships
    df["high"] = df[["open", "high", "low", "close"]].max(axis=1)
    df["low"] = df[["open", "high", "low", "close"]].min(axis=1)
    
    return df


def test_calculate_simple_returns(sample_ohlcv_data):
    """Test simple returns calculation."""
    returns = MarketStatistics.calculate_returns(
        sample_ohlcv_data, price_col="close", method="simple"
    )
    
    assert isinstance(returns, pd.Series)
    assert len(returns) == len(sample_ohlcv_data)
    assert pd.isna(returns.iloc[0])  # First return should be NaN
    assert not returns.iloc[1:].isna().all()  # Rest should have values


def test_calculate_log_returns(sample_ohlcv_data):
    """Test log returns calculation."""
    returns = MarketStatistics.calculate_returns(
        sample_ohlcv_data, price_col="close", method="log"
    )
    
    assert isinstance(returns, pd.Series)
    assert len(returns) == len(sample_ohlcv_data)
    assert pd.isna(returns.iloc[0])


def test_calculate_returns_invalid_method(sample_ohlcv_data):
    """Test returns calculation with invalid method."""
    with pytest.raises(ValueError, match="Unknown method"):
        MarketStatistics.calculate_returns(
            sample_ohlcv_data, price_col="close", method="invalid"
        )


def test_calculate_volatility(sample_ohlcv_data):
    """Test volatility calculation."""
    volatility = MarketStatistics.calculate_volatility(
        sample_ohlcv_data, price_col="close", window=20
    )
    
    assert isinstance(volatility, pd.Series)
    assert len(volatility) == len(sample_ohlcv_data)
    # First 19 values should be NaN due to window
    assert volatility.iloc[:19].isna().all()
    # Rest should have values
    assert not volatility.iloc[20:].isna().all()


def test_calculate_price_statistics(sample_ohlcv_data):
    """Test price statistics calculation."""
    stats = MarketStatistics.calculate_price_statistics(sample_ohlcv_data)
    
    assert isinstance(stats, dict)
    assert "open" in stats
    assert "high" in stats
    assert "low" in stats
    assert "close" in stats
    
    # Check close statistics
    close_stats = stats["close"]
    assert "mean" in close_stats
    assert "median" in close_stats
    assert "std" in close_stats
    assert "min" in close_stats
    assert "max" in close_stats
    assert "skew" in close_stats
    assert "kurtosis" in close_stats


def test_calculate_return_statistics(sample_ohlcv_data):
    """Test return statistics calculation."""
    stats = MarketStatistics.calculate_return_statistics(sample_ohlcv_data)
    
    assert isinstance(stats, dict)
    assert "mean" in stats
    assert "std" in stats
    assert "positive_returns_pct" in stats
    assert "negative_returns_pct" in stats
    
    # Percentages should sum to approximately 100
    total_pct = stats["positive_returns_pct"] + stats["negative_returns_pct"]
    assert 99.0 <= total_pct <= 101.0  # Allow small floating point error


def test_calculate_candle_features(sample_ohlcv_data):
    """Test candle feature calculation."""
    df_with_features = MarketStatistics.calculate_candle_features(sample_ohlcv_data)
    
    assert "body" in df_with_features.columns
    assert "body_abs" in df_with_features.columns
    assert "upper_wick" in df_with_features.columns
    assert "lower_wick" in df_with_features.columns
    assert "candle_range" in df_with_features.columns
    assert "body_ratio" in df_with_features.columns
    assert "is_bullish" in df_with_features.columns
    
    # Check candle range is positive
    assert (df_with_features["candle_range"] >= 0).all()
    
    # Check is_bullish is binary
    assert df_with_features["is_bullish"].isin([0, 1]).all()


def test_detect_outliers_iqr(sample_ohlcv_data):
    """Test outlier detection using IQR method."""
    returns = MarketStatistics.calculate_returns(sample_ohlcv_data)
    outliers = MarketStatistics.detect_outliers(
        returns.dropna(), method="iqr", threshold=1.5
    )
    
    assert isinstance(outliers, pd.Series)
    assert outliers.dtype == bool


def test_detect_outliers_zscore(sample_ohlcv_data):
    """Test outlier detection using Z-score method."""
    returns = MarketStatistics.calculate_returns(sample_ohlcv_data)
    outliers = MarketStatistics.detect_outliers(
        returns.dropna(), method="zscore", threshold=3.0
    )
    
    assert isinstance(outliers, pd.Series)
    assert outliers.dtype == bool


def test_detect_outliers_invalid_method(sample_ohlcv_data):
    """Test outlier detection with invalid method."""
    returns = MarketStatistics.calculate_returns(sample_ohlcv_data)
    
    with pytest.raises(ValueError, match="Unknown method"):
        MarketStatistics.detect_outliers(returns.dropna(), method="invalid")


def test_calculate_summary_report(sample_ohlcv_data):
    """Test comprehensive summary report generation."""
    report = MarketStatistics.calculate_summary_report(sample_ohlcv_data)
    
    assert isinstance(report, dict)
    assert "general" in report
    assert "price_statistics" in report
    assert "return_statistics" in report
    assert "volume_statistics" in report
    assert "candle_statistics" in report
    
    # Check general info
    assert report["general"]["total_records"] == len(sample_ohlcv_data)
    
    # Check candle statistics
    candle_stats = report["candle_statistics"]
    assert "avg_body" in candle_stats
    assert "bullish_pct" in candle_stats
    assert 0 <= candle_stats["bullish_pct"] <= 100
