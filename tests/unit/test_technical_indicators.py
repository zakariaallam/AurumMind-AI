"""Unit tests for TechnicalIndicators."""

import numpy as np
import pandas as pd
import pytest

from aurummind_ai.features.technical_indicators import TechnicalIndicators


@pytest.fixture
def sample_ohlcv_data():
    """Create sample OHLCV data for testing."""
    np.random.seed(42)
    
    dates = pd.date_range("2024-01-01", periods=200, freq="1h", tz="UTC")
    prices = 2050 + np.cumsum(np.random.randn(200) * 2)
    
    data = {
        "timestamp": dates,
        "open": prices + np.random.randn(200) * 0.5,
        "high": prices + np.abs(np.random.randn(200)) * 2,
        "low": prices - np.abs(np.random.randn(200)) * 2,
        "close": prices + np.random.randn(200) * 0.5,
        "volume": 1000 + np.random.randn(200) * 100,
    }
    
    df = pd.DataFrame(data)
    
    # Ensure OHLC relationships
    df["high"] = df[["open", "high", "low", "close"]].max(axis=1)
    df["low"] = df[["open", "high", "low", "close"]].min(axis=1)
    
    return df


def test_sma_calculation(sample_ohlcv_data):
    """Test Simple Moving Average calculation."""
    sma = TechnicalIndicators.sma(sample_ohlcv_data, period=20)
    
    assert isinstance(sma, pd.Series)
    assert len(sma) == len(sample_ohlcv_data)
    # First 19 should be NaN
    assert sma.iloc[:19].isna().all()
    # Rest should have values
    assert not sma.iloc[20:].isna().all()


def test_ema_calculation(sample_ohlcv_data):
    """Test Exponential Moving Average calculation."""
    ema = TechnicalIndicators.ema(sample_ohlcv_data, period=20)
    
    assert isinstance(ema, pd.Series)
    assert len(ema) == len(sample_ohlcv_data)
    assert not ema.iloc[20:].isna().all()


def test_rsi_calculation(sample_ohlcv_data):
    """Test RSI calculation."""
    rsi = TechnicalIndicators.rsi(sample_ohlcv_data, period=14)
    
    assert isinstance(rsi, pd.Series)
    assert len(rsi) == len(sample_ohlcv_data)
    # RSI should be between 0 and 100
    valid_rsi = rsi.dropna()
    assert (valid_rsi >= 0).all()
    assert (valid_rsi <= 100).all()


def test_macd_calculation(sample_ohlcv_data):
    """Test MACD calculation."""
    macd_line, signal_line, histogram = TechnicalIndicators.macd(sample_ohlcv_data)
    
    assert isinstance(macd_line, pd.Series)
    assert isinstance(signal_line, pd.Series)
    assert isinstance(histogram, pd.Series)
    assert len(macd_line) == len(sample_ohlcv_data)
    assert len(signal_line) == len(sample_ohlcv_data)
    assert len(histogram) == len(sample_ohlcv_data)


def test_bollinger_bands_calculation(sample_ohlcv_data):
    """Test Bollinger Bands calculation."""
    upper, middle, lower = TechnicalIndicators.bollinger_bands(
        sample_ohlcv_data, period=20
    )
    
    assert isinstance(upper, pd.Series)
    assert isinstance(middle, pd.Series)
    assert isinstance(lower, pd.Series)
    
    # Upper should be > middle > lower (where defined)
    valid_idx = ~upper.isna()
    assert (upper[valid_idx] >= middle[valid_idx]).all()
    assert (middle[valid_idx] >= lower[valid_idx]).all()


def test_atr_calculation(sample_ohlcv_data):
    """Test ATR calculation."""
    atr = TechnicalIndicators.atr(sample_ohlcv_data, period=14)
    
    assert isinstance(atr, pd.Series)
    assert len(atr) == len(sample_ohlcv_data)
    # ATR should be positive
    assert (atr.dropna() >= 0).all()


def test_adx_calculation(sample_ohlcv_data):
    """Test ADX calculation."""
    adx = TechnicalIndicators.adx(sample_ohlcv_data, period=14)
    
    assert isinstance(adx, pd.Series)
    assert len(adx) == len(sample_ohlcv_data)
    # ADX should be between 0 and 100
    valid_adx = adx.dropna()
    if len(valid_adx) > 0:
        assert (valid_adx >= 0).all()
        assert (valid_adx <= 100).all()


def test_stochastic_calculation(sample_ohlcv_data):
    """Test Stochastic Oscillator calculation."""
    k, d = TechnicalIndicators.stochastic(sample_ohlcv_data)
    
    assert isinstance(k, pd.Series)
    assert isinstance(d, pd.Series)
    assert len(k) == len(sample_ohlcv_data)
    assert len(d) == len(sample_ohlcv_data)
    
    # %K and %D should be between 0 and 100
    valid_k = k.dropna()
    valid_d = d.dropna()
    assert (valid_k >= 0).all()
    assert (valid_k <= 100).all()
    assert (valid_d >= 0).all()
    assert (valid_d <= 100).all()


def test_roc_calculation(sample_ohlcv_data):
    """Test Rate of Change calculation."""
    roc = TechnicalIndicators.roc(sample_ohlcv_data, period=12)
    
    assert isinstance(roc, pd.Series)
    assert len(roc) == len(sample_ohlcv_data)


def test_obv_calculation(sample_ohlcv_data):
    """Test On-Balance Volume calculation."""
    obv = TechnicalIndicators.obv(sample_ohlcv_data)
    
    assert isinstance(obv, pd.Series)
    assert len(obv) == len(sample_ohlcv_data)
    # OBV should be cumulative
    assert not obv.isna().all()


def test_calculate_all_indicators(sample_ohlcv_data):
    """Test calculating all indicators at once."""
    df_with_indicators = TechnicalIndicators.calculate_all_indicators(
        sample_ohlcv_data
    )
    
    # Should have more columns than original
    assert len(df_with_indicators.columns) > len(sample_ohlcv_data.columns)
    
    # Check for key indicators
    expected_indicators = [
        "sma_20", "ema_20", "rsi", "macd", "macd_signal",
        "bb_upper", "bb_middle", "bb_lower", "atr", "adx",
        "stoch_k", "stoch_d", "roc", "obv"
    ]
    
    for indicator in expected_indicators:
        assert indicator in df_with_indicators.columns
    
    # Original columns should be preserved
    for col in sample_ohlcv_data.columns:
        assert col in df_with_indicators.columns
