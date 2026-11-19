"""Unit tests for walk-forward validation."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from aurummind_ai.evaluation.walk_forward import (
    WalkForwardValidator,
    AnchoredWalkForward,
)


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    np.random.seed(42)
    n = 500
    
    X = pd.DataFrame({
        'feature1': np.random.randn(n),
        'feature2': np.random.randn(n),
        'feature3': np.random.randn(n),
    })
    
    y = pd.Series(np.random.randint(0, 2, n))
    
    return X, y


def test_walk_forward_initialization():
    """Test walk-forward validator initialization."""
    validator = WalkForwardValidator(
        train_window=100,
        test_window=20,
        step_size=20,
        retrain_every=1,
    )
    
    assert validator.train_window == 100
    assert validator.test_window == 20
    assert validator.step_size == 20
    assert validator.retrain_every == 1


def test_walk_forward_validation(sample_data):
    """Test walk-forward validation."""
    X, y = sample_data
    
    validator = WalkForwardValidator(
        train_window=200,
        test_window=50,
        step_size=50,
        retrain_every=1,
    )
    
    def model_factory():
        return LogisticRegression(max_iter=100)
    
    def eval_func(y_true, y_pred):
        from sklearn.metrics import accuracy_score
        return {'accuracy': accuracy_score(y_true, y_pred)}
    
    results = validator.validate(model_factory, X, y, eval_func)
    
    assert 'num_steps' in results
    assert 'total_predictions' in results
    assert results['num_steps'] > 0
    assert results['total_predictions'] > 0


def test_walk_forward_predictions(sample_data):
    """Test prediction retrieval."""
    X, y = sample_data
    
    validator = WalkForwardValidator(
        train_window=200,
        test_window=50,
    )
    
    def model_factory():
        return LogisticRegression(max_iter=100)
    
    def eval_func(y_true, y_pred):
        from sklearn.metrics import accuracy_score
        return {'accuracy': accuracy_score(y_true, y_pred)}
    
    validator.validate(model_factory, X, y, eval_func)
    
    predictions, actuals = validator.get_predictions()
    
    assert len(predictions) > 0
    assert len(actuals) > 0
    assert len(predictions) == len(actuals)


def test_walk_forward_results_dataframe(sample_data):
    """Test results DataFrame."""
    X, y = sample_data
    
    validator = WalkForwardValidator(
        train_window=200,
        test_window=50,
    )
    
    def model_factory():
        return LogisticRegression(max_iter=100)
    
    def eval_func(y_true, y_pred):
        from sklearn.metrics import accuracy_score
        return {'accuracy': accuracy_score(y_true, y_pred)}
    
    validator.validate(model_factory, X, y, eval_func)
    
    df = validator.get_results_dataframe()
    
    assert not df.empty
    assert 'step' in df.columns
    assert 'accuracy' in df.columns
    assert 'train_size' in df.columns
    assert 'test_size' in df.columns


def test_anchored_walk_forward_initialization():
    """Test anchored walk-forward initialization."""
    validator = AnchoredWalkForward(
        min_train_size=100,
        test_window=20,
        step_size=20,
    )
    
    assert validator.min_train_size == 100
    assert validator.test_window == 20
    assert validator.step_size == 20


def test_anchored_walk_forward_validation(sample_data):
    """Test anchored walk-forward validation."""
    X, y = sample_data
    
    validator = AnchoredWalkForward(
        min_train_size=200,
        test_window=50,
        step_size=50,
    )
    
    def model_factory():
        return LogisticRegression(max_iter=100)
    
    def eval_func(y_true, y_pred):
        from sklearn.metrics import accuracy_score
        return {'accuracy': accuracy_score(y_true, y_pred)}
    
    results = validator.validate(model_factory, X, y, eval_func)
    
    assert 'num_steps' in results
    assert 'total_predictions' in results
    assert results['num_steps'] > 0


def test_anchored_training_window_growth(sample_data):
    """Test that training window grows in anchored approach."""
    X, y = sample_data
    
    validator = AnchoredWalkForward(
        min_train_size=100,
        test_window=50,
        step_size=50,
    )
    
    def model_factory():
        return LogisticRegression(max_iter=100)
    
    def eval_func(y_true, y_pred):
        from sklearn.metrics import accuracy_score
        return {'accuracy': accuracy_score(y_true, y_pred)}
    
    validator.validate(model_factory, X, y, eval_func)
    
    df = validator.get_results_dataframe()
    
    # Training size should increase over steps
    if len(df) > 1:
        assert df.iloc[1]['train_size'] >= df.iloc[0]['train_size']
