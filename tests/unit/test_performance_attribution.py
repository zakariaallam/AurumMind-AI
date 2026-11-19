"""Unit tests for performance attribution."""

import numpy as np
import pandas as pd
import pytest

from aurummind_ai.evaluation.performance_attribution import (
    PerformanceAttribution,
    ErrorAnalysis,
)


@pytest.fixture
def sample_predictions():
    """Create sample predictions with timestamps."""
    np.random.seed(42)
    n = 200
    
    timestamps = pd.date_range('2023-01-01', periods=n, freq='1h')
    y_true = np.random.randint(0, 2, n)
    y_pred = np.random.randint(0, 2, n)
    
    return y_true, y_pred, timestamps


@pytest.fixture
def sample_probabilities():
    """Create sample probability predictions."""
    np.random.seed(42)
    n = 200
    
    # Binary classification probabilities
    probs = np.random.dirichlet([1, 1], size=n)
    
    return probs


def test_by_timeframe():
    """Test performance comparison by timeframe."""
    results = {
        '1h': {'accuracy': 0.65, 'f1': 0.63},
        '4h': {'accuracy': 0.70, 'f1': 0.68},
        '1d': {'accuracy': 0.72, 'f1': 0.71},
    }
    
    comparison = PerformanceAttribution.by_timeframe(results, metric='accuracy')
    
    assert isinstance(comparison, pd.DataFrame)
    assert 'timeframe' in comparison.columns
    assert 'accuracy' in comparison.columns
    assert len(comparison) == 3


def test_by_time_period(sample_predictions):
    """Test performance by time period."""
    y_true, y_pred, timestamps = sample_predictions
    
    def eval_func(y_t, y_p):
        from sklearn.metrics import accuracy_score
        return {'accuracy': accuracy_score(y_t, y_p)}
    
    monthly = PerformanceAttribution.by_time_period(
        y_true, y_pred, timestamps, eval_func, period='M'
    )
    
    assert isinstance(monthly, pd.DataFrame)
    assert 'period' in monthly.columns
    assert 'accuracy' in monthly.columns
    assert 'sample_count' in monthly.columns


def test_by_day_of_week(sample_predictions):
    """Test performance by day of week."""
    y_true, y_pred, timestamps = sample_predictions
    
    def eval_func(y_t, y_p):
        from sklearn.metrics import accuracy_score
        return {'accuracy': accuracy_score(y_t, y_p)}
    
    daily = PerformanceAttribution.by_day_of_week(
        y_true, y_pred, timestamps, eval_func
    )
    
    assert isinstance(daily, pd.DataFrame)
    assert 'day_of_week' in daily.columns
    assert 'accuracy' in daily.columns


def test_by_hour_of_day(sample_predictions):
    """Test performance by hour of day."""
    y_true, y_pred, timestamps = sample_predictions
    
    def eval_func(y_t, y_p):
        from sklearn.metrics import accuracy_score
        return {'accuracy': accuracy_score(y_t, y_p)}
    
    hourly = PerformanceAttribution.by_hour_of_day(
        y_true, y_pred, timestamps, eval_func
    )
    
    assert isinstance(hourly, pd.DataFrame)
    assert 'hour' in hourly.columns
    assert 'accuracy' in hourly.columns


def test_by_prediction_confidence(sample_predictions, sample_probabilities):
    """Test performance by prediction confidence."""
    y_true, y_pred, _ = sample_predictions
    y_proba = sample_probabilities
    
    def eval_func(y_t, y_p):
        from sklearn.metrics import accuracy_score
        return {'accuracy': accuracy_score(y_t, y_p)}
    
    confidence = PerformanceAttribution.by_prediction_confidence(
        y_true, y_pred, y_proba, eval_func, bins=5
    )
    
    assert isinstance(confidence, pd.DataFrame)
    assert 'confidence_bin' in confidence.columns
    assert 'accuracy' in confidence.columns
    assert 'avg_confidence' in confidence.columns


def test_by_prediction_horizon():
    """Test performance by prediction horizon."""
    results = {
        1: {'accuracy': 0.70, 'f1': 0.68},
        2: {'accuracy': 0.65, 'f1': 0.63},
        3: {'accuracy': 0.62, 'f1': 0.60},
    }
    
    comparison = PerformanceAttribution.by_prediction_horizon(
        results, metric='accuracy'
    )
    
    assert isinstance(comparison, pd.DataFrame)
    assert 'horizon' in comparison.columns
    assert 'accuracy' in comparison.columns


def test_feature_importance():
    """Test feature importance extraction."""
    from sklearn.ensemble import RandomForestClassifier
    
    # Train simple model
    X = np.random.randn(100, 10)
    y = np.random.randint(0, 2, 100)
    
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    
    feature_names = [f'feature_{i}' for i in range(10)]
    
    importance = PerformanceAttribution.by_feature_importance(
        model, feature_names, top_n=5
    )
    
    assert isinstance(importance, pd.DataFrame)
    assert 'feature' in importance.columns
    assert 'importance' in importance.columns
    assert 'rank' in importance.columns
    assert len(importance) == 5


def test_confusion_analysis():
    """Test confusion matrix analysis."""
    np.random.seed(42)
    y_true = np.random.randint(0, 3, 100)
    y_pred = np.random.randint(0, 3, 100)
    
    class_names = ['Down', 'Neutral', 'Up']
    
    analysis = ErrorAnalysis.confusion_analysis(y_true, y_pred, class_names)
    
    assert isinstance(analysis, dict)
    assert 'confusion_matrix' in analysis
    assert 'class_accuracy' in analysis
    assert 'top_misclassifications' in analysis


def test_regression_error_distribution():
    """Test regression error distribution."""
    np.random.seed(42)
    y_true = np.random.randn(100)
    y_pred = y_true + np.random.randn(100) * 0.5
    
    error_stats = ErrorAnalysis.regression_error_distribution(y_true, y_pred)
    
    assert isinstance(error_stats, dict)
    assert 'mean_error' in error_stats
    assert 'std_error' in error_stats
    assert 'mean_abs_error' in error_stats
    assert 'median_abs_error' in error_stats
    assert 'error_skewness' in error_stats


def test_outlier_prediction_detection():
    """Test outlier prediction detection."""
    np.random.seed(42)
    y_true = np.random.randn(100)
    y_pred = y_true.copy()
    
    # Add some outliers
    y_pred[0] = y_true[0] + 10
    y_pred[50] = y_true[50] - 10
    
    outlier_mask, stats = ErrorAnalysis.identify_outlier_predictions(
        y_true, y_pred, threshold=3.0
    )
    
    assert isinstance(outlier_mask, np.ndarray)
    assert isinstance(stats, dict)
    assert 'num_outliers' in stats
    assert 'pct_outliers' in stats
    assert stats['num_outliers'] > 0


def test_attribution_empty_data():
    """Test attribution with empty results."""
    results = {}
    
    comparison = PerformanceAttribution.by_timeframe(results)
    
    assert isinstance(comparison, pd.DataFrame)
    assert len(comparison) == 0
