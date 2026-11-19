"""Model evaluation metrics and utilities.

This module provides comprehensive evaluation metrics
for classification and regression models.
"""

import logging
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    log_loss,
)

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Comprehensive model evaluation.
    
    Provides metrics for both classification and regression tasks,
    with special consideration for financial time series.
    """
    
    @staticmethod
    def evaluate_classifier(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
        average: str = 'weighted',
    ) -> dict[str, Any]:
        """Evaluate classification model.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities
            average: Averaging strategy for multi-class
            
        Returns:
            Dictionary with metrics
        """
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, average=average, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, average=average, zero_division=0)
        metrics['f1'] = f1_score(y_true, y_pred, average=average, zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        # Class-specific metrics
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        metrics['classification_report'] = report
        
        # Probability-based metrics (if available)
        if y_proba is not None:
            try:
                if y_proba.shape[1] == 2:
                    # Binary classification
                    metrics['roc_auc'] = roc_auc_score(y_true, y_proba[:, 1])
                    metrics['log_loss'] = log_loss(y_true, y_proba)
                else:
                    # Multi-class
                    metrics['roc_auc'] = roc_auc_score(
                        y_true, y_proba, multi_class='ovr', average=average
                    )
                    metrics['log_loss'] = log_loss(y_true, y_proba)
            except Exception as e:
                logger.warning(f"Could not calculate probability-based metrics: {e}")
        
        return metrics
    
    @staticmethod
    def evaluate_regressor(
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> dict[str, Any]:
        """Evaluate regression model.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary with metrics
        """
        metrics = {}
        
        # Standard regression metrics
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        metrics['mse'] = mean_squared_error(y_true, y_pred)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['r2'] = r2_score(y_true, y_pred)
        
        # MAPE (with zero handling)
        mask = y_true != 0
        if mask.sum() > 0:
            mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
            metrics['mape'] = mape
        else:
            metrics['mape'] = None
        
        # Directional accuracy (for financial forecasting)
        direction_true = np.sign(y_true)
        direction_pred = np.sign(y_pred)
        metrics['directional_accuracy'] = accuracy_score(direction_true, direction_pred)
        
        return metrics
    
    @staticmethod
    def calculate_directional_accuracy(
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> float:
        """Calculate directional accuracy.
        
        Measures if the model predicts the correct direction of change.
        
        Args:
            y_true: True values (returns)
            y_pred: Predicted values (returns)
            
        Returns:
            Directional accuracy (0-1)
        """
        direction_true = np.sign(y_true)
        direction_pred = np.sign(y_pred)
        
        return accuracy_score(direction_true, direction_pred)
    
    @staticmethod
    def print_classification_metrics(
        metrics: dict[str, Any],
        model_name: str = "Model",
    ) -> None:
        """Print classification metrics in readable format.
        
        Args:
            metrics: Metrics dictionary
            model_name: Name of the model
        """
        print("=" * 70)
        print(f"{model_name} - Classification Metrics")
        print("=" * 70)
        print()
        
        print("Overall Metrics:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-Score:  {metrics['f1']:.4f}")
        
        if 'roc_auc' in metrics and metrics['roc_auc'] is not None:
            print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
        if 'log_loss' in metrics and metrics['log_loss'] is not None:
            print(f"  Log Loss:  {metrics['log_loss']:.4f}")
        
        print()
        print("Confusion Matrix:")
        cm = np.array(metrics['confusion_matrix'])
        for row in cm:
            print("  ", row)
        
        print()
        print("Per-Class Metrics:")
        report = metrics['classification_report']
        for label, values in report.items():
            if isinstance(values, dict):
                print(f"  Class {label}:")
                print(f"    Precision: {values['precision']:.4f}")
                print(f"    Recall:    {values['recall']:.4f}")
                print(f"    F1-Score:  {values['f1-score']:.4f}")
                print(f"    Support:   {values['support']}")
        
        print("=" * 70)
    
    @staticmethod
    def print_regression_metrics(
        metrics: dict[str, Any],
        model_name: str = "Model",
    ) -> None:
        """Print regression metrics in readable format.
        
        Args:
            metrics: Metrics dictionary
            model_name: Name of the model
        """
        print("=" * 70)
        print(f"{model_name} - Regression Metrics")
        print("=" * 70)
        print()
        
        print("Error Metrics:")
        print(f"  MAE (Mean Absolute Error):       {metrics['mae']:.6f}")
        print(f"  MSE (Mean Squared Error):        {metrics['mse']:.6f}")
        print(f"  RMSE (Root Mean Squared Error):  {metrics['rmse']:.6f}")
        
        if metrics['mape'] is not None:
            print(f"  MAPE (Mean Absolute % Error):    {metrics['mape']:.2f}%")
        
        print()
        print("Quality Metrics:")
        print(f"  R² Score:               {metrics['r2']:.4f}")
        print(f"  Directional Accuracy:   {metrics['directional_accuracy']:.4f}")
        
        print("=" * 70)
    
    @staticmethod
    def compare_models(
        results: dict[str, dict[str, Any]],
        metric: str = 'accuracy',
    ) -> pd.DataFrame:
        """Compare multiple models.
        
        Args:
            results: Dictionary of {model_name: metrics}
            metric: Primary metric to sort by
            
        Returns:
            DataFrame with comparison
        """
        comparison_data = []
        
        for model_name, metrics in results.items():
            row = {'model': model_name}
            row.update(metrics)
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        
        if metric in df.columns:
            df = df.sort_values(metric, ascending=False)
        
        return df
    
    @staticmethod
    def calculate_profit_loss(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        returns: np.ndarray,
    ) -> dict[str, float]:
        """Calculate profit/loss metrics for trading strategy.
        
        Args:
            y_true: True direction labels
            y_pred: Predicted direction labels
            returns: Actual returns
            
        Returns:
            Dictionary with P/L metrics
        """
        # Strategy returns (take position based on prediction)
        # Assuming binary: 0=down, 1=up
        # If predict up (1), go long; if predict down (0), go short
        strategy_returns = np.where(y_pred == 1, returns, -returns)
        
        metrics = {
            'total_return': strategy_returns.sum(),
            'mean_return': strategy_returns.mean(),
            'std_return': strategy_returns.std(),
            'sharpe_ratio': strategy_returns.mean() / strategy_returns.std() if strategy_returns.std() > 0 else 0,
            'win_rate': (strategy_returns > 0).mean(),
            'max_drawdown': (strategy_returns.cumsum() - strategy_returns.cumsum().cummax()).min(),
        }
        
        return metrics
