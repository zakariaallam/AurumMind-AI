"""Unit tests for explainability modules."""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from aurummind_ai.explainability import (
    SHAPExplainer,
    FeatureImportanceAnalyzer,
    PartialDependenceAnalyzer,
    IndividualExplainer,
)


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    np.random.seed(42)
    n = 200
    
    X = pd.DataFrame({
        'feature1': np.random.randn(n),
        'feature2': np.random.randn(n),
        'feature3': np.random.randn(n),
        'feature4': np.random.randn(n),
        'feature5': np.random.randn(n),
    })
    
    y = (X['feature1'] + X['feature2'] > 0).astype(int)
    
    return X, y


@pytest.fixture
def trained_model(sample_data):
    """Create trained model for testing."""
    X, y = sample_data
    model = RandomForestClassifier(n_estimators=10, random_state=42, max_depth=3)
    model.fit(X, y)
    return model


# ========== SHAP Explainer Tests ==========

def test_shap_explainer_initialization(trained_model):
    """Test SHAP explainer initialization."""
    feature_names = [f'feature{i}' for i in range(1, 6)]
    explainer = SHAPExplainer(trained_model, feature_names, model_type='tree')
    
    assert explainer.model == trained_model
    assert explainer.feature_names == feature_names
    assert explainer.model_type == 'tree'


def test_shap_create_explainer(trained_model, sample_data):
    """Test SHAP explainer creation."""
    X, _ = sample_data
    feature_names = X.columns.tolist()
    
    explainer = SHAPExplainer(trained_model, feature_names, model_type='tree')
    
    # Skip test if SHAP not installed
    if not explainer._shap_available:
        pytest.skip("SHAP not installed")
    
    explainer.create_explainer()
    assert explainer.explainer is not None


def test_shap_explain(trained_model, sample_data):
    """Test SHAP value calculation."""
    X, _ = sample_data
    feature_names = X.columns.tolist()
    
    explainer = SHAPExplainer(trained_model, feature_names, model_type='tree')
    
    if not explainer._shap_available:
        pytest.skip("SHAP not installed")
    
    explainer.create_explainer()
    shap_values = explainer.explain(X.head(50))
    
    assert shap_values is not None
    assert len(shap_values) == 50
    assert shap_values.shape[1] == len(feature_names)


def test_shap_global_importance(trained_model, sample_data):
    """Test global importance from SHAP values."""
    X, _ = sample_data
    feature_names = X.columns.tolist()
    
    explainer = SHAPExplainer(trained_model, feature_names, model_type='tree')
    
    if not explainer._shap_available:
        pytest.skip("SHAP not installed")
    
    explainer.create_explainer()
    explainer.explain(X.head(50))
    
    importance = explainer.get_global_importance(top_n=5)
    
    assert isinstance(importance, pd.DataFrame)
    assert 'feature' in importance.columns
    assert 'importance' in importance.columns
    assert len(importance) <= 5


# ========== Feature Importance Tests ==========

def test_feature_importance_initialization(trained_model):
    """Test feature importance analyzer initialization."""
    feature_names = [f'feature{i}' for i in range(1, 6)]
    analyzer = FeatureImportanceAnalyzer(trained_model, feature_names)
    
    assert analyzer.model == trained_model
    assert analyzer.feature_names == feature_names


def test_built_in_importance(trained_model, sample_data):
    """Test built-in feature importance extraction."""
    X, _ = sample_data
    feature_names = X.columns.tolist()
    
    analyzer = FeatureImportanceAnalyzer(trained_model, feature_names)
    importance = analyzer.get_built_in_importance(top_n=5)
    
    assert isinstance(importance, pd.DataFrame)
    assert 'feature' in importance.columns
    assert 'importance' in importance.columns
    assert 'rank' in importance.columns
    assert len(importance) <= 5


def test_permutation_importance(trained_model, sample_data):
    """Test permutation importance calculation."""
    X, y = sample_data
    feature_names = X.columns.tolist()
    
    analyzer = FeatureImportanceAnalyzer(trained_model, feature_names)
    importance = analyzer.get_permutation_importance(
        X.head(100), y.head(100).values, n_repeats=3, top_n=5
    )
    
    assert isinstance(importance, pd.DataFrame)
    assert 'feature' in importance.columns
    assert 'importance_mean' in importance.columns
    assert 'importance_std' in importance.columns
    assert len(importance) <= 5


def test_feature_correlations(trained_model, sample_data):
    """Test feature correlation analysis."""
    X, _ = sample_data
    feature_names = X.columns.tolist()
    
    # Add correlated feature
    X['feature6'] = X['feature1'] * 0.95 + np.random.randn(len(X)) * 0.1
    
    analyzer = FeatureImportanceAnalyzer(trained_model, feature_names + ['feature6'])
    correlations = analyzer.analyze_feature_correlations(X, threshold=0.7)
    
    assert isinstance(correlations, pd.DataFrame)
    if not correlations.empty:
        assert 'feature1' in correlations.columns
        assert 'feature2' in correlations.columns
        assert 'correlation' in correlations.columns


def test_importance_comparison(trained_model, sample_data):
    """Test importance method comparison."""
    X, y = sample_data
    feature_names = X.columns.tolist()
    
    analyzer = FeatureImportanceAnalyzer(trained_model, feature_names)
    
    # Compute multiple importance types
    analyzer.get_built_in_importance()
    analyzer.get_permutation_importance(X.head(100), y.head(100).values, n_repeats=2)
    
    comparison = analyzer.compare_methods(top_n=5)
    
    assert isinstance(comparison, pd.DataFrame)
    assert not comparison.empty


# ========== Partial Dependence Tests ==========

def test_partial_dependence_initialization(trained_model):
    """Test PD analyzer initialization."""
    feature_names = [f'feature{i}' for i in range(1, 6)]
    analyzer = PartialDependenceAnalyzer(trained_model, feature_names)
    
    assert analyzer.model == trained_model
    assert analyzer.feature_names == feature_names


def test_calculate_partial_dependence(trained_model, sample_data):
    """Test partial dependence calculation."""
    X, _ = sample_data
    feature_names = X.columns.tolist()
    
    analyzer = PartialDependenceAnalyzer(trained_model, feature_names)
    results = analyzer.calculate_partial_dependence(X, features=[0, 1], grid_resolution=10)
    
    assert isinstance(results, dict)
    assert 'average' in results
    assert 'grid_values' in results
    assert 'feature_names' in results


def test_calculate_ice(trained_model, sample_data):
    """Test ICE curve calculation."""
    X, _ = sample_data
    feature_names = X.columns.tolist()
    
    analyzer = PartialDependenceAnalyzer(trained_model, feature_names)
    results = analyzer.calculate_ice(X, feature=0, sample_indices=np.arange(20), grid_resolution=10)
    
    assert isinstance(results, dict)
    assert 'ice_curves' in results
    assert 'grid_values' in results
    assert 'feature_name' in results
    assert results['ice_curves'].shape[0] == 20


def test_feature_effect_summary(trained_model, sample_data):
    """Test feature effect summary."""
    X, _ = sample_data
    feature_names = X.columns.tolist()
    
    analyzer = PartialDependenceAnalyzer(trained_model, feature_names)
    summary = analyzer.get_feature_effect_summary(X, features=[0, 1])
    
    assert isinstance(summary, pd.DataFrame)
    assert 'feature' in summary.columns
    assert 'mean_effect' in summary.columns
    assert 'range' in summary.columns


# ========== Individual Explainer Tests ==========

def test_individual_explainer_initialization(trained_model):
    """Test individual explainer initialization."""
    feature_names = [f'feature{i}' for i in range(1, 6)]
    explainer = IndividualExplainer(trained_model, feature_names)
    
    assert explainer.model == trained_model
    assert explainer.feature_names == feature_names


def test_explain_sample(trained_model, sample_data):
    """Test single sample explanation."""
    X, _ = sample_data
    feature_names = X.columns.tolist()
    
    explainer = IndividualExplainer(trained_model, feature_names)
    explanation = explainer.explain_sample(X, sample_idx=0)
    
    assert isinstance(explanation, dict)
    assert 'sample_index' in explanation
    assert 'prediction' in explanation
    assert 'feature_values' in explanation


def test_compare_samples(trained_model, sample_data):
    """Test sample comparison."""
    X, _ = sample_data
    feature_names = X.columns.tolist()
    
    explainer = IndividualExplainer(trained_model, feature_names)
    comparison = explainer.compare_samples(X, sample_idx1=0, sample_idx2=10)
    
    assert isinstance(comparison, dict)
    assert 'sample1' in comparison
    assert 'sample2' in comparison
    assert 'prediction_difference' in comparison
    assert 'feature_differences' in comparison


def test_find_similar_samples(trained_model, sample_data):
    """Test similar sample finding."""
    X, _ = sample_data
    feature_names = X.columns.tolist()
    
    explainer = IndividualExplainer(trained_model, feature_names)
    similar = explainer.find_similar_samples(X, sample_idx=0, n_neighbors=5)
    
    assert isinstance(similar, dict)
    assert 'sample_index' in similar
    assert 'neighbors' in similar
    assert len(similar['neighbors']) == 5


def test_generate_counterfactual(trained_model, sample_data):
    """Test counterfactual generation."""
    X, _ = sample_data
    feature_names = X.columns.tolist()
    
    explainer = IndividualExplainer(trained_model, feature_names)
    
    # Get original prediction
    original_pred = trained_model.predict(X.iloc[[0]])[0]
    
    # Try to flip prediction
    desired_pred = 1 - original_pred
    
    counterfactual = explainer.generate_counterfactual(
        X,
        sample_idx=0,
        desired_prediction=desired_pred,
        max_iterations=50,
        step_size=0.1,
    )
    
    assert isinstance(counterfactual, dict)
    assert 'original_prediction' in counterfactual
    assert 'desired_prediction' in counterfactual
    assert 'final_prediction' in counterfactual
    assert 'modifications' in counterfactual
