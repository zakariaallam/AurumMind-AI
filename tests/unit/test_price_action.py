"""Unit tests for PriceActionFeatures."""

import numpy as np
import pandas as pd
import pytest

from aurummind_ai.features.price_action import PriceActionFeatures


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
    df["high"] = df[["open", "high", "low", "close"]].max(axis=1)
    df["low"] = df[["open", "high", "low", "close"]].min(axis=1)
    
    return df


def test_calculate_returns(sample_ohlcv_data):
    """Test returns calculation."""
    df_returns = PriceActionFeatures.calculate_returns(sample_ohlcv_data)
    
    assert "return_1" in df_returns.columns
    assert "return_3" in df_returns.columns
    assert "return_5" in df_returns.columns
    
    # First return should be NaN
    assert pd.isna(df_returns["return_1"].iloc[0])


def test_calculate_rolling_statistics(sample_ohlcv_data):
    """Test rolling statistics calculation."""
    df_stats = PriceActionFeatures.calculate_rolling_statistics(sample_ohlcv_data)
    
    assert "close_rolling_min_5" in df_stats.columns
    assert "close_rolling_max_5" in df_stats.columns
    assert "close_rolling_std_5" in df_stats.columns
    assert "distance_from_high_5" in df_stats.columns
    assert "distance_from_low_5" in df_stats.columns


def test_calculate_price_position(sample_ohlcv_data):
    """Test price position calculation."""
    df_position = PriceActionFeatures.calculate_price_position(sample_ohlcv_data)
    
    assert "price_position_5" in df_position.columns
    assert "price_position_10" in df_position.columns
    
    # Position should be between 0 and 1
    valid_positions = df_position["price_position_5"].dropna()
    assert (valid_positions >= 0).all()
    assert (valid_positions <= 1).all()


def test_calculate_momentum_features(sample_ohlcv_data):
    """Test momentum features calculation."""
    df_momentum = PriceActionFeatures.calculate_momentum_features(sample_ohlcv_data)
    
    assert "momentum_3" in df_momentum.columns
    assert "momentum_5" in df_momentum.columns
    assert "acceleration_5" in df_momentum.columns
    assert "up_days" in df_momentum.columns
    assert "down_days" in df_momentum.columns
    assert "consecutive_up" in df_momentum.columns
    assert "consecutive_down" in df_momentum.columns
    
    # up_days and down_days should be binary
    assert df_momentum["up_days"].isin([0, 1]).all()
    assert df_momentum["down_days"].isin([0, 1]).all()


def test_calculate_volatility_features(sample_ohlcv_data):
    """Test volatility features calculation."""
    df_vol = PriceActionFeatures.calculate_volatility_features(sample_ohlcv_data)
    
    assert "volatility_5" in df_vol.columns
    assert "hl_range_pct" in df_vol.columns
    assert "avg_range_5" in df_vol.columns
    assert "range_expansion" in df_vol.columns
    
    # hl_range_pct should be positive
    assert (df_vol["hl_range_pct"] >= 0).all()


def test_calculate_gap_features(sample_ohlcv_data):
    """Test gap features calculation."""
    df_gaps = PriceActionFeatures.calculate_gap_features(sample_ohlcv_data)
    
    assert "gap" in df_gaps.columns
    assert "gap_pct" in df_gaps.columns
    assert "gap_up" in df_gaps.columns
    assert "gap_down" in df_gaps.columns
    assert "gap_filled" in df_gaps.columns
    
    # gap_up and gap_down should be binary
    assert df_gaps["gap_up"].isin([0, 1]).all()
    assert df_gaps["gap_down"].isin([0, 1]).all()


def test_calculate_higher_highs_lows(sample_ohlcv_data):
    """Test market structure features calculation."""
    df_structure = PriceActionFeatures.calculate_higher_highs_lows(sample_ohlcv_data)
    
    assert "higher_high" in df_structure.columns
    assert "higher_low" in df_structure.columns
    assert "lower_high" in df_structure.columns
    assert "lower_low" in df_structure.columns
    assert "uptrend_signal" in df_structure.columns
    assert "downtrend_signal" in df_structure.columns
    
    # All structure indicators should be binary
    for col in ["higher_high", "higher_low", "lower_high", "lower_low"]:
        assert df_structure[col].isin([0, 1]).all()


def test_calculate_support_resistance(sample_ohlcv_data):
    """Test support/resistance calculation."""
    df_levels = PriceActionFeatures.calculate_support_resistance(sample_ohlcv_data)
    
    assert "resistance_1" in df_levels.columns
    assert "support_1" in df_levels.columns
    assert "distance_to_resistance" in df_levels.columns
    assert "distance_to_support" in df_levels.columns
    assert "near_resistance" in df_levels.columns
    assert "near_support" in df_levels.columns
    
    # near_resistance and near_support should be binary
    assert df_levels["near_resistance"].isin([0, 1]).all()
    assert df_levels["near_support"].isin([0, 1]).all()


def test_calculate_all_price_action_features(sample_ohlcv_data):
    """Test calculating all price action features."""
    df_features = PriceActionFeatures.calculate_all_price_action_features(
        sample_ohlcv_data
    )
    
    # Should have many more columns than original
    assert len(df_features.columns) > len(sample_ohlcv_data.columns)
    
    # Original columns should be preserved
    for col in sample_ohlcv_data.columns:
        assert col in df_features.columns
    
    # Check for key feature categories
    assert any("return_" in col for col in df_features.columns)
    assert any("momentum_" in col for col in df_features.columns)
    assert any("volatility_" in col for col in df_features.columns)
    assert any("gap" in col for col in df_features.columns)
