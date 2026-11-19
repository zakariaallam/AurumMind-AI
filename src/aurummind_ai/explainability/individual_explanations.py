"""Individual prediction explanations.

This module provides tools to explain individual predictions
through various methods.
"""

import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class IndividualExplainer:
    """Explain individual predictions.
    
    Provides multiple methods for explaining why a model
    made a specific prediction for a given sample.
    """
    
    def __init__(
        self,
        model: Any,
        feature_names: list[str],
    ):
        """Initialize individual explainer.
        
        Args:
            model: Trained model
            feature_names: Names of features
        """
        self.model = model
        self.feature_names = feature_names
    
    def explain_sample(
        self,
        X: pd.DataFrame,
        sample_idx: int,
        shap_values: Optional[np.ndarray] = None,
        top_n: int = 10,
    ) -> dict[str, Any]:
        """Explain a single prediction.
        
        Args:
            X: Feature matrix
            sample_idx: Index of sample to explain
            shap_values: Pre-computed SHAP values (optional)
            top_n: Number of top features to include
            
        Returns:
            Dictionary with explanation
        """
        sample = X.iloc[sample_idx]
        
        # Get prediction
        prediction = self.model.predict(sample.values.reshape(1, -1))[0]
        
        # Get probabilities if available
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(sample.values.reshape(1, -1))[0]
        else:
            probabilities = None
        
        explanation = {
            'sample_index': sample_idx,
            'prediction': prediction,
            'probabilities': probabilities,
            'feature_values': sample.to_dict(),
        }
        
        # Add SHAP contributions if available
        if shap_values is not None:
            sample_shap = shap_values[sample_idx]
            
            # Get top contributing features
            abs_shap = np.abs(sample_shap)
            top_indices = np.argsort(abs_shap)[::-1][:top_n]
            
            top_features = []
            for idx in top_indices:
                top_features.append({
                    'feature': self.feature_names[idx],
                    'value': sample.iloc[idx],
                    'shap_contribution': sample_shap[idx],
                    'abs_contribution': abs_shap[idx],
                })
            
            explanation['top_features'] = top_features
        
        return explanation
    
    def compare_samples(
        self,
        X: pd.DataFrame,
        sample_idx1: int,
        sample_idx2: int,
        shap_values: Optional[np.ndarray] = None,
    ) -> dict[str, Any]:
        """Compare two predictions.
        
        Args:
            X: Feature matrix
            sample_idx1: Index of first sample
            sample_idx2: Index of second sample
            shap_values: Pre-computed SHAP values (optional)
            
        Returns:
            Dictionary with comparison
        """
        # Get explanations for both samples
        exp1 = self.explain_sample(X, sample_idx1, shap_values)
        exp2 = self.explain_sample(X, sample_idx2, shap_values)
        
        # Compare feature values
        feature_diffs = []
        
        for feature in self.feature_names:
            val1 = exp1['feature_values'][feature]
            val2 = exp2['feature_values'][feature]
            diff = val2 - val1
            
            feature_diff = {
                'feature': feature,
                'sample1_value': val1,
                'sample2_value': val2,
                'difference': diff,
                'abs_difference': abs(diff),
            }
            
            # Add SHAP differences if available
            if shap_values is not None:
                idx = self.feature_names.index(feature)
                shap_diff = shap_values[sample_idx2, idx] - shap_values[sample_idx1, idx]
                feature_diff['shap_difference'] = shap_diff
            
            feature_diffs.append(feature_diff)
        
        # Sort by absolute difference
        feature_diffs = sorted(feature_diffs, key=lambda x: x['abs_difference'], reverse=True)
        
        comparison = {
            'sample1': exp1,
            'sample2': exp2,
            'prediction_difference': exp2['prediction'] - exp1['prediction'],
            'feature_differences': feature_diffs,
        }
        
        return comparison
    
    def find_similar_samples(
        self,
        X: pd.DataFrame,
        sample_idx: int,
        n_neighbors: int = 5,
        metric: str = 'euclidean',
    ) -> dict[str, Any]:
        """Find similar samples to explain through analogy.
        
        Args:
            X: Feature matrix
            sample_idx: Index of sample
            n_neighbors: Number of neighbors to find
            metric: Distance metric
            
        Returns:
            Dictionary with similar samples
        """
        from sklearn.metrics.pairwise import pairwise_distances
        
        logger.info(f"Finding {n_neighbors} similar samples to index {sample_idx}")
        
        sample = X.iloc[[sample_idx]]
        
        # Calculate distances
        distances = pairwise_distances(sample, X, metric=metric)[0]
        
        # Find nearest neighbors (excluding self)
        distances[sample_idx] = np.inf
        neighbor_indices = np.argsort(distances)[:n_neighbors]
        
        # Get predictions for neighbors
        neighbors = []
        
        for neighbor_idx in neighbor_indices:
            pred = self.model.predict(X.iloc[[neighbor_idx]])[0]
            dist = distances[neighbor_idx]
            
            neighbors.append({
                'index': int(neighbor_idx),
                'distance': float(dist),
                'prediction': pred,
            })
        
        # Get prediction for target sample
        target_pred = self.model.predict(sample)[0]
        
        result = {
            'sample_index': sample_idx,
            'sample_prediction': target_pred,
            'neighbors': neighbors,
            'avg_neighbor_prediction': np.mean([n['prediction'] for n in neighbors]),
        }
        
        logger.info("✓ Similar samples found")
        
        return result
    
    def generate_counterfactual(
        self,
        X: pd.DataFrame,
        sample_idx: int,
        desired_prediction: Any,
        features_to_modify: Optional[list[str]] = None,
        max_iterations: int = 100,
        step_size: float = 0.1,
    ) -> dict[str, Any]:
        """Generate counterfactual explanation.
        
        Finds minimal feature changes needed to achieve desired prediction.
        
        Args:
            X: Feature matrix
            sample_idx: Index of sample
            desired_prediction: Target prediction value/class
            features_to_modify: Features allowed to change (None = all)
            max_iterations: Maximum optimization iterations
            step_size: Step size for feature modification
            
        Returns:
            Dictionary with counterfactual
        """
        logger.info(f"Generating counterfactual for sample {sample_idx}")
        
        sample = X.iloc[sample_idx].copy()
        original_pred = self.model.predict(sample.values.reshape(1, -1))[0]
        
        if features_to_modify is None:
            features_to_modify = self.feature_names
        
        # Track modifications
        modifications = []
        current_sample = sample.copy()
        
        for iteration in range(max_iterations):
            current_pred = self.model.predict(current_sample.values.reshape(1, -1))[0]
            
            # Check if reached desired prediction
            if np.isclose(current_pred, desired_prediction, atol=0.01):
                logger.info(f"✓ Counterfactual found in {iteration + 1} iterations")
                break
            
            # Try modifying each feature
            best_feature = None
            best_improvement = 0
            best_direction = 0
            
            for feature in features_to_modify:
                # Try increasing
                test_sample = current_sample.copy()
                test_sample[feature] += step_size
                pred_up = self.model.predict(test_sample.values.reshape(1, -1))[0]
                improvement_up = abs(desired_prediction - pred_up) - abs(desired_prediction - current_pred)
                
                # Try decreasing
                test_sample = current_sample.copy()
                test_sample[feature] -= step_size
                pred_down = self.model.predict(test_sample.values.reshape(1, -1))[0]
                improvement_down = abs(desired_prediction - pred_down) - abs(desired_prediction - current_pred)
                
                # Pick best direction
                if improvement_up < best_improvement:
                    best_improvement = improvement_up
                    best_feature = feature
                    best_direction = 1
                
                if improvement_down < best_improvement:
                    best_improvement = improvement_down
                    best_feature = feature
                    best_direction = -1
            
            # Apply best modification
            if best_feature and best_direction != 0:
                old_value = current_sample[best_feature]
                current_sample[best_feature] += best_direction * step_size
                new_value = current_sample[best_feature]
                
                modifications.append({
                    'iteration': iteration + 1,
                    'feature': best_feature,
                    'old_value': old_value,
                    'new_value': new_value,
                    'change': new_value - old_value,
                })
            else:
                # No improvement possible
                logger.warning("Counterfactual search stuck")
                break
        
        final_pred = self.model.predict(current_sample.values.reshape(1, -1))[0]
        
        result = {
            'sample_index': sample_idx,
            'original_prediction': original_pred,
            'desired_prediction': desired_prediction,
            'final_prediction': final_pred,
            'achieved': np.isclose(final_pred, desired_prediction, atol=0.1),
            'modifications': modifications,
            'num_features_modified': len(set(m['feature'] for m in modifications)),
            'original_sample': sample.to_dict(),
            'counterfactual_sample': current_sample.to_dict(),
        }
        
        return result
    
    def print_explanation(
        self,
        explanation: dict[str, Any],
    ) -> None:
        """Print explanation in readable format.
        
        Args:
            explanation: Explanation dictionary from explain_sample()
        """
        print("=" * 70)
        print(f"Explanation for Sample {explanation['sample_index']}")
        print("=" * 70)
        print()
        
        print(f"Prediction: {explanation['prediction']}")
        
        if explanation.get('probabilities') is not None:
            print(f"Probabilities: {explanation['probabilities']}")
        
        if 'top_features' in explanation:
            print("\nTop Contributing Features:")
            print()
            
            for feat in explanation['top_features']:
                direction = "↑" if feat['shap_contribution'] > 0 else "↓"
                print(f"  {feat['feature']:.<40} {feat['value']:.4f}")
                print(f"    SHAP contribution: {direction} {feat['shap_contribution']:.4f}")
                print()
        
        print("=" * 70)
    
    def print_comparison(
        self,
        comparison: dict[str, Any],
        top_n: int = 10,
    ) -> None:
        """Print comparison in readable format.
        
        Args:
            comparison: Comparison dictionary from compare_samples()
            top_n: Number of top differences to show
        """
        print("=" * 70)
        print("Sample Comparison")
        print("=" * 70)
        print()
        
        print(f"Sample 1 Index: {comparison['sample1']['sample_index']}")
        print(f"Sample 1 Prediction: {comparison['sample1']['prediction']}")
        print()
        
        print(f"Sample 2 Index: {comparison['sample2']['sample_index']}")
        print(f"Sample 2 Prediction: {comparison['sample2']['prediction']}")
        print()
        
        print(f"Prediction Difference: {comparison['prediction_difference']:.4f}")
        print()
        
        print(f"Top {top_n} Feature Differences:")
        print()
        
        for diff in comparison['feature_differences'][:top_n]:
            print(f"  {diff['feature']:.<40}")
            print(f"    Sample 1: {diff['sample1_value']:.4f}")
            print(f"    Sample 2: {diff['sample2_value']:.4f}")
            print(f"    Difference: {diff['difference']:.4f}")
            
            if 'shap_difference' in diff:
                print(f"    SHAP Difference: {diff['shap_difference']:.4f}")
            
            print()
        
        print("=" * 70)
