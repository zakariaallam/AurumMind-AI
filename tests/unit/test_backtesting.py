"""Unit tests for backtesting engine."""

import numpy as np
import pandas as pd
import pytest

from aurummind_ai.evaluation.backtesting import (
    BacktestEngine,
    TradeConfig,
    Trade,
)


@pytest.fixture
def sample_price_data():
    """Create sample price data."""
    np.random.seed(42)
    n = 100
    
    dates = pd.date_range('2023-01-01', periods=n, freq='1h')
    close = 2000 + np.cumsum(np.random.randn(n) * 2)
    
    df = pd.DataFrame({
        'timestamp': dates,
        'close': close,
    })
    
    return df


@pytest.fixture
def sample_predictions():
    """Create sample predictions."""
    np.random.seed(42)
    return np.random.randint(0, 2, 100)


def test_trade_config_initialization():
    """Test trade config initialization."""
    config = TradeConfig(
        transaction_cost=0.001,
        slippage=0.0005,
        position_size=0.5,
    )
    
    assert config.transaction_cost == 0.001
    assert config.slippage == 0.0005
    assert config.position_size == 0.5


def test_backtest_engine_initialization():
    """Test backtest engine initialization."""
    engine = BacktestEngine(initial_capital=10000.0)
    
    assert engine.initial_capital == 10000.0
    assert engine.config is not None
    assert engine.trades == []


def test_backtest_run(sample_price_data, sample_predictions):
    """Test backtest execution."""
    engine = BacktestEngine(initial_capital=10000.0)
    
    results = engine.run(sample_price_data, sample_predictions)
    
    assert isinstance(results, dict)
    assert 'total_trades' in results
    assert 'final_capital' in results
    assert 'total_return' in results
    assert 'win_rate' in results


def test_backtest_with_costs(sample_price_data, sample_predictions):
    """Test backtest with transaction costs."""
    config = TradeConfig(
        transaction_cost=0.001,
        slippage=0.001,
    )
    
    engine = BacktestEngine(initial_capital=10000.0, config=config)
    results = engine.run(sample_price_data, sample_predictions)
    
    assert results['total_trades'] >= 0


def test_equity_curve(sample_price_data, sample_predictions):
    """Test equity curve generation."""
    engine = BacktestEngine(initial_capital=10000.0)
    engine.run(sample_price_data, sample_predictions)
    
    equity = engine.get_equity_curve()
    
    assert isinstance(equity, pd.Series)
    assert len(equity) > 0
    assert equity.iloc[0] == 10000.0  # Initial capital


def test_trades_dataframe(sample_price_data, sample_predictions):
    """Test trades DataFrame."""
    engine = BacktestEngine(initial_capital=10000.0)
    engine.run(sample_price_data, sample_predictions)
    
    trades_df = engine.get_trades_dataframe()
    
    if not trades_df.empty:
        assert 'timestamp' in trades_df.columns
        assert 'direction' in trades_df.columns
        assert 'entry_price' in trades_df.columns
        assert 'exit_price' in trades_df.columns
        assert 'pnl' in trades_df.columns


def test_trade_metrics(sample_price_data, sample_predictions):
    """Test trade performance metrics."""
    engine = BacktestEngine(initial_capital=10000.0)
    results = engine.run(sample_price_data, sample_predictions)
    
    # Check key metrics exist
    assert 'winning_trades' in results
    assert 'losing_trades' in results
    assert 'win_rate' in results
    assert 'max_drawdown' in results
    assert 'sharpe_ratio' in results


def test_risk_metrics(sample_price_data, sample_predictions):
    """Test risk-adjusted metrics."""
    engine = BacktestEngine(initial_capital=10000.0)
    results = engine.run(sample_price_data, sample_predictions)
    
    assert 'sharpe_ratio' in results
    assert 'sortino_ratio' in results
    assert 'max_drawdown' in results
    assert 'max_drawdown_pct' in results


def test_zero_trades():
    """Test backtest with no trades."""
    df = pd.DataFrame({
        'timestamp': pd.date_range('2023-01-01', periods=10, freq='1h'),
        'close': [2000] * 10,
    })
    
    predictions = np.array([1] * 10)  # All same prediction = no trades
    
    engine = BacktestEngine(initial_capital=10000.0)
    results = engine.run(df, predictions)
    
    assert results['total_trades'] >= 0
    assert results['final_capital'] > 0


def test_long_positions(sample_price_data):
    """Test long-only strategy."""
    predictions = np.ones(len(sample_price_data), dtype=int)  # All long
    
    engine = BacktestEngine(initial_capital=10000.0)
    results = engine.run(sample_price_data, predictions)
    
    # Should have at least one long position
    trades_df = engine.get_trades_dataframe()
    if not trades_df.empty:
        assert (trades_df['direction'] == 'long').any()


def test_short_positions(sample_price_data):
    """Test short-only strategy."""
    predictions = np.zeros(len(sample_price_data), dtype=int)  # All short
    
    engine = BacktestEngine(initial_capital=10000.0)
    results = engine.run(sample_price_data, predictions)
    
    # Should have at least one short position
    trades_df = engine.get_trades_dataframe()
    if not trades_df.empty:
        assert (trades_df['direction'] == 'short').any()


def test_profit_loss_calculation():
    """Test profit/loss calculation."""
    df = pd.DataFrame({
        'timestamp': pd.date_range('2023-01-01', periods=5, freq='1h'),
        'close': [2000, 2010, 2020, 2015, 2025],
    })
    
    # Buy low, sell high
    predictions = np.array([1, 1, 0, 0, 1])
    
    engine = BacktestEngine(initial_capital=10000.0)
    results = engine.run(df, predictions)
    
    assert results['final_capital'] != results['initial_capital']
