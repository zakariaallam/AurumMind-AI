"""Performance attribution and breakdown analysis.

This module provides tools to break down model performance
by various factors: timeframe, market conditions, prediction horizon, etc.
"""

import logging
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class PerformanceAttribution:
    """Analyze performance attribution across multiple dimensions."""
    
    @staticmethod
    def by_timeframe(
        results: dict[str, dict[str, Any]],
        metric: str = 'accuracy',
    ) -> pd.DataFrame:
        """Compare performance across different timeframes.
        
        Args:
            results: Dictionary mapping timeframe to metrics
            metric: Primary metric to compare
            
        Returns:
            DataFrame with comparison
        """
        comparison = []
        
        for timeframe, metrics in results.items():
            row = {'timeframe': timeframe}
            
            # Extract numeric metrics
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    row[key] = value
            
            comparison.append(row)
        
        df = pd.DataFrame(comparison)
        
        if metric in df.columns:
            df = df.sort_values(metric, ascending=False)
        
        return df
    
    @staticmethod
    def by_time_period(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        timestamps: pd.DatetimeIndex,
        eval_func: callable,
        period: str = 'M',  # 'D'=daily, 'W'=weekly, 'M'=monthly, 'Q'=quarterly
    ) -> pd.DataFrame:
        """Analyze performance over time periods.
        
        Args:
            y_true: True labels/values
            y_pred: Predicted labels/values
            timestamps: Timestamps for each prediction
            eval_func: Function to compute metrics
            period: Pandas period string (D/W/M/Q/Y)
            
        Returns:
            DataFrame with metrics by period
        """
        # Create DataFrame
        df = pd.DataFrame({
            'timestamp': timestamps,
            'y_true': y_true,
            'y_pred': y_pred,
        })
        
        # Group by period
        df['period'] = df['timestamp'].dt.to_period(period)
        
        results = []
        
        for period_value, group in df.groupby('period'):
            metrics = eval_func(group['y_true'].values, group['y_pred'].values)
            metrics['period'] = str(period_value)
            metrics['sample_count'] = len(group)
            results.append(metrics)
        
        return pd.DataFrame(results)
    
    @staticmethod
    def by_day_of_week(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        timestamps: pd.DatetimeIndex,
        eval_func: callable,
    ) -> pd.DataFrame:
        """Analyze performance by day of week.
        
        Args:
            y_true: True labels/values
            y_pred: Predicted labels/values
            timestamps: Timestamps for each prediction
            eval_func: Function to compute metrics
            
        Returns:
            DataFrame with metrics by day of week
        """
        df = pd.DataFrame({
            'timestamp': timestamps,
            'y_true': y_true,
            'y_pred': y_pred,
        })
        
        df['day_of_week'] = df['timestamp'].dt.day_name()
        
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        results = []
        
        for day in day_order:
            group = df[df['day_of_week'] == day]
            
            if len(group) == 0:
                continue
            
            metrics = eval_func(group['y_true'].values, group['y_pred'].values)
            metrics['day_of_week'] = day
            metrics['sample_count'] = len(group)
            results.append(metrics)
        
        return pd.DataFrame(results)
    
    @staticmethod
    def by_hour_of_day(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        timestamps: pd.DatetimeIndex,
        eval_func: callable,
    ) -> pd.DataFrame:
        """Analyze performance by hour of day.
        
        Useful for intraday data to identify time-of-day patterns.
        
        Args:
            y_true: True labels/values
            y_pred: Predicted labels/values
            timestamps: Timestamps for each prediction
            eval_func: Function to compute metrics
            
        Returns:
            DataFrame with metrics by hour
        """
        df = pd.DataFrame({
            'timestamp': timestamps,
            'y_true': y_true,
            'y_pred': y_pred,
        })
        
        df['hour'] = df['timestamp'].dt.hour
        
        results = []
        
        for hour in range(24):
            group = df[df['hour'] == hour]
            
            if len(group) == 0:
                continue
            
            metrics = eval_func(group['y_true'].values, group['y_pred'].values)
            metrics['hour'] = hour
            metrics['sample_count'] = len(group)
            results.append(metrics)
        
        return pd.DataFrame(results)
    
    @staticmethod
    def by_prediction_confidence(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray,
        eval_func: callable,
        bins: int = 5,
    ) -> pd.DataFrame:
        """Analyze performance by prediction confidence.
        
        For classification models with probability outputs.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Prediction probabilities
            eval_func: Function to compute metrics
            bins: Number of confidence bins
            
        Returns:
            DataFrame with metrics by confidence bin
        """
        # Get confidence (max probability)
        confidence = y_proba.max(axis=1)
        
        # Create bins
        bin_edges = np.linspace(0, 1, bins + 1)
        bin_labels = [f"{bin_edges[i]:.2f}-{bin_edges[i+1]:.2f}" for i in range(bins)]
        
        df = pd.DataFrame({
            'y_true': y_true,
            'y_pred': y_pred,
            'confidence': confidence,
        })
        
        df['confidence_bin'] = pd.cut(
            df['confidence'],
            bins=bin_edges,
            labels=bin_labels,
            include_lowest=True,
        )
        
        results = []
        
        for bin_label in bin_labels:
            group = df[df['confidence_bin'] == bin_label]
            
            if len(group) == 0:
                continue
            
            metrics = eval_func(group['y_true'].values, group['y_pred'].values)
            metrics['confidence_bin'] = bin_label
            metrics['sample_count'] = len(group)
            metrics['avg_confidence'] = group['confidence'].mean()
            results.append(metrics)
        
        return pd.DataFrame(results)
    
    @staticmethod
    def by_prediction_horizon(
        results: dict[int, dict[str, Any]],
        metric: str = 'accuracy',
    ) -> pd.DataFrame:
        """Compare performance across different prediction horizons.
        
        Args:
            results: Dictionary mapping horizon to metrics
            metric: Primary metric to compare
            
        Returns:
            DataFrame with comparison
        """
        comparison = []
        
        for horizon, metrics in results.items():
            row = {'horizon': horizon}
            
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    row[key] = value
            
            comparison.append(row)
        
        df = pd.DataFrame(comparison)
        
        if 'horizon' in df.columns:
            df = df.sort_values('horizon')
        
        return df
    
    @staticmethod
    def by_feature_importance(
        model,
        feature_names: list[str],
        top_n: int = 20,
    ) -> pd.DataFrame:
        """Extract and rank feature importance.
        
        Args:
            model: Trained model with feature_importances_ attribute
            feature_names: Names of features
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature importance rankings
        """
        if not hasattr(model, 'feature_importances_'):
            logger.warning("Model does not have feature_importances_ attribute")
            return pd.DataFrame()
        
        importance = model.feature_importances_
        
        df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance,
        })
        
        df = df.sort_values('importance', ascending=False)
        df['rank'] = range(1, len(df) + 1)
        df['cumulative_importance'] = df['importance'].cumsum()
        
        return df.head(top_n)
    
    @staticmethod
    def print_attribution_summary(
        attribution_df: pd.DataFrame,
        title: str = "Performance Attribution",
    ) -> None:
        """Print attribution analysis in readable format.
        
        Args:
            attribution_df: DataFrame from attribution method
            title: Title for the report
        """
        print("=" * 70)
        print(title)
        print("=" * 70)
        print()
        
        # Display DataFrame
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        print(attribution_df.to_string(index=False))
        print()
        
        print("=" * 70)


class ErrorAnalysis:
    """Analyze model errors to identify patterns and improvement areas."""
    
    @staticmethod
    def confusion_analysis(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """Detailed confusion matrix analysis.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: Names of classes
            
        Returns:
            Dictionary with confusion analysis
        """
        from sklearn.metrics import confusion_matrix
        
        cm = confusion_matrix(y_true, y_pred)
        
        if class_names is None:
            class_names = [f"Class {i}" for i in range(len(cm))]
        
        # Per-class accuracy
        class_accuracy = cm.diagonal() / cm.sum(axis=1)
        
        # Most common misclassifications
        misclassifications = []
        
        for i in range(len(cm)):
            for j in range(len(cm)):
                if i != j and cm[i, j] > 0:
                    misclassifications.append({
                        'true_class': class_names[i],
                        'predicted_class': class_names[j],
                        'count': int(cm[i, j]),
                        'percentage': cm[i, j] / cm[i].sum() * 100,
                    })
        
        misclassifications.sort(key=lambda x: x['count'], reverse=True)
        
        return {
            'confusion_matrix': cm,
            'class_accuracy': dict(zip(class_names, class_accuracy)),
            'top_misclassifications': misclassifications[:10],
        }
    
    @staticmethod
    def regression_error_distribution(
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> dict[str, Any]:
        """Analyze distribution of regression errors.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary with error distribution statistics
        """
        errors = y_pred - y_true
        abs_errors = np.abs(errors)
        pct_errors = errors / np.where(y_true != 0, y_true, 1)  # Avoid division by zero
        
        return {
            'mean_error': errors.mean(),
            'std_error': errors.std(),
            'median_error': np.median(errors),
            'mean_abs_error': abs_errors.mean(),
            'median_abs_error': np.median(abs_errors),
            'mean_pct_error': pct_errors.mean() * 100,
            'median_pct_error': np.median(pct_errors) * 100,
            'error_skewness': pd.Series(errors).skew(),
            'error_kurtosis': pd.Series(errors).kurtosis(),
            'pct_overestimate': (errors > 0).mean() * 100,
            'pct_underestimate': (errors < 0).mean() * 100,
        }
    
    @staticmethod
    def identify_outlier_predictions(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        threshold: float = 3.0,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Identify outlier predictions using z-score.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            threshold: Z-score threshold for outliers
            
        Returns:
            Tuple of (outlier_mask, outlier_stats)
        """
        errors = y_pred - y_true
        z_scores = np.abs((errors - errors.mean()) / errors.std())
        
        outlier_mask = z_scores > threshold
        
        stats = {
            'num_outliers': outlier_mask.sum(),
            'pct_outliers': outlier_mask.mean() * 100,
            'outlier_indices': np.where(outlier_mask)[0].tolist()[:50],  # First 50
            'max_z_score': z_scores.max(),
            'outlier_errors': errors[outlier_mask].tolist()[:20],  # First 20
        }
        
        return outlier_mask, stats
    
    @staticmethod
    def print_error_analysis(
        error_stats: dict[str, Any],
        analysis_type: str = "regression",
    ) -> None:
        """Print error analysis in readable format.
        
        Args:
            error_stats: Error statistics dictionary
            analysis_type: Type of analysis (regression/classification)
        """
        print("=" * 70)
        print(f"Error Analysis ({analysis_type.title()})")
        print("=" * 70)
        print()
        
        for key, value in error_stats.items():
            if isinstance(value, (int, float)):
                if 'pct' in key or 'percentage' in key:
                    print(f"  {key.replace('_', ' ').title():.<40} {value:.2f}%")
                else:
                    print(f"  {key.replace('_', ' ').title():.<40} {value:.4f}")
        
        print()
        print("=" * 70)
