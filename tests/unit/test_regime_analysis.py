"""Unit tests for regime analysis."""

import numpy as np
import pandas as pd
import pytest

from aurummind_ai.evaluation.regime_analysis import (
    RegimeDetector,
    RegimeEvaluator,
    RegimeTransitionAnalyzer,
)


@pytest.fixture
def sample_price_data():
    """Create sample price data."""
    np.random.seed(42)
    n = 500
    
    # Simulate price data
    dates = pd.date_range('2023-01-01', periods=n, freq='1h')
    close = 2000 + np.cumsum(np.random.randn(n) * 2)
    
    df = pd.DataFrame({
        'timestamp': dates,
        'close': close,
        'open': close + np.random.randn(n) * 0.5,
        'high': close + abs(np.random.randn(n)) * 1.0,
        'low': close - abs(np.random.randn(n)) * 1.0,
        'volume': np.random.randint(1000, 10000, n),
    })
    
    return df


@pytest.fixture
def sample_predictions():
    """Create sample predictions."""
    np.random.seed(42)
    n = 500
    
    y_true = np.random.randint(0, 2, n)
    y_pred = np.random.randint(0, 2, n)
    
    return y_true, y_pred


def test_regime_detector_initialization():
    """Test regime detector initialization."""
    detector = RegimeDetector(volatility_window=20, trend_window=50)
    
    assert detector.volatility_window == 20
    assert detector.trend_window == 50


def test_volatility_regime_detection(sample_price_data):
    """Test volatility regime detection."""
    detector = RegimeDetector()
    
    regime = detector.detect_volatility_regime(sample_price_data)
    
    assert isinstance(regime, pd.Series)
    assert len(regime) == len(sample_price_data)
    assert regime.isin([0, 1, 2]).all()  # 0=low, 1=medium, 2=high


def test_trend_regime_detection(sample_price_data):
    """Test trend regime detection."""
    detector = RegimeDetector()
    
    regime = detector.detect_trend_regime(sample_price_data)
    
    assert isinstance(regime, pd.Series)
    assert len(regime) == len(sample_price_data)
    assert regime.isin([0, 1, 2]).all()  # 0=down, 1=ranging, 2=up


def test_combined_regime_detection(sample_price_data):
    """Test combined regime detection."""
    detector = RegimeDetector()
    
    regimes = detector.detect_combined_regime(sample_price_data)
    
    assert isinstance(regimes, pd.DataFrame)
    assert 'volatility_regime' in regimes.columns
    assert 'trend_regime' in regimes.columns
    assert 'combined_regime' in regimes.columns


def test_regime_evaluator(sample_predictions):
    """Test regime-based evaluation."""
    y_true, y_pred = sample_predictions
    regime = np.random.randint(0, 3, len(y_true))  # 3 regimes
    
    def eval_func(y_t, y_p):
        from sklearn.metrics import accuracy_score
        return {'accuracy': accuracy_score(y_t, y_p)}
    
    results = RegimeEvaluator.evaluate_by_regime(
        y_true, y_pred, regime, eval_func
    )
    
    assert isinstance(results, dict)
    assert len(results) > 0
    
    # Check that each regime has metrics
    for regime_id, metrics in results.items():
        assert 'accuracy' in metrics
        assert 'sample_count' in metrics


def test_regime_comparison(sample_predictions):
    """Test regime performance comparison."""
    y_true, y_pred = sample_predictions
    regime = np.random.randint(0, 3, len(y_true))
    
    def eval_func(y_t, y_p):
        from sklearn.metrics import accuracy_score, precision_score
        return {
            'accuracy': accuracy_score(y_t, y_p),
            'precision': precision_score(y_t, y_p, zero_division=0),
        }
    
    results = RegimeEvaluator.evaluate_by_regime(
        y_true, y_pred, regime, eval_func
    )
    
    comparison = RegimeEvaluator.compare_regime_performance(results, metric='accuracy')
    
    assert isinstance(comparison, pd.DataFrame)
    assert 'regime' in comparison.columns
    assert 'accuracy' in comparison.columns


def test_regime_stability():
    """Test regime stability calculation."""
    regime_results = {
        '0': {'accuracy': 0.60, 'sample_count': 100},
        '1': {'accuracy': 0.70, 'sample_count': 150},
        '2': {'accuracy': 0.65, 'sample_count': 120},
    }
    
    stability = RegimeEvaluator.calculate_regime_stability(
        regime_results, metric='accuracy'
    )
    
    assert isinstance(stability, dict)
    assert 'mean' in stability
    assert 'std' in stability
    assert 'min' in stability
    assert 'max' in stability
    assert 'range' in stability


def test_transition_detection():
    """Test regime transition detection."""
    regime = pd.Series([0, 0, 1, 1, 1, 2, 2, 0, 0])
    
    transitions = RegimeTransitionAnalyzer.detect_transitions(regime)
    
    assert isinstance(transitions, pd.Series)
    assert transitions.dtype == bool
    assert transitions.sum() > 0  # Should have some transitions


def test_transition_evaluation(sample_predictions):
    """Test evaluation at transitions."""
    y_true, y_pred = sample_predictions
    
    # Create regime with transitions
    regime = pd.Series(np.where(np.arange(len(y_true)) < len(y_true) // 2, 0, 1))
    transitions = RegimeTransitionAnalyzer.detect_transitions(regime)
    
    def eval_func(y_t, y_p):
        from sklearn.metrics import accuracy_score
        return {'accuracy': accuracy_score(y_t, y_p)}
    
    results = RegimeTransitionAnalyzer.evaluate_at_transitions(
        y_true, y_pred, transitions.values, window=5, eval_func=eval_func
    )
    
    assert isinstance(results, dict)
    assert 'num_transitions' in results
    assert 'transition_samples' in results
    assert 'stable_samples' in results
