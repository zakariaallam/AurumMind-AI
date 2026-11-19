"""Market regime analysis and regime-specific evaluation.

This module provides tools to:
- Detect market regimes (volatility-based)
- Evaluate model performance across regimes
- Identify regime-specific patterns
"""

import logging
from enum import Enum
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class RegimeType(Enum):
    """Market regime types."""
    LOW_VOLATILITY = "low_volatility"
    MEDIUM_VOLATILITY = "medium_volatility"
    HIGH_VOLATILITY = "high_volatility"
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"


class RegimeDetector:
    """Detect market regimes based on volatility and trend.
    
    Implements multiple regime detection methods:
    - Volatility-based (low/medium/high)
    - Trend-based (up/down/ranging)
    - Combined regimes
    """
    
    def __init__(
        self,
        volatility_window: int = 20,
        trend_window: int = 50,
    ):
        """Initialize regime detector.
        
        Args:
            volatility_window: Window for volatility calculation
            trend_window: Window for trend detection
        """
        self.volatility_window = volatility_window
        self.trend_window = trend_window
    
    def detect_volatility_regime(
        self,
        df: pd.DataFrame,
        low_threshold: float = 0.33,
        high_threshold: float = 0.67,
    ) -> pd.Series:
        """Detect volatility regime using percentile thresholds.
        
        Args:
            df: DataFrame with 'close' column
            low_threshold: Percentile threshold for low volatility (0-1)
            high_threshold: Percentile threshold for high volatility (0-1)
            
        Returns:
            Series with regime labels (0=low, 1=medium, 2=high)
        """
        # Calculate rolling volatility (standard deviation of returns)
        returns = df['close'].pct_change()
        volatility = returns.rolling(window=self.volatility_window).std()
        
        # Calculate percentile thresholds
        low_vol = volatility.quantile(low_threshold)
        high_vol = volatility.quantile(high_threshold)
        
        # Classify regimes
        regime = pd.Series(1, index=df.index)  # Default to medium
        regime[volatility <= low_vol] = 0      # Low volatility
        regime[volatility >= high_vol] = 2     # High volatility
        
        logger.info("Detected volatility regimes:")
        logger.info(f"  Low (0):    {(regime == 0).sum()} samples ({(regime == 0).mean()*100:.1f}%)")
        logger.info(f"  Medium (1): {(regime == 1).sum()} samples ({(regime == 1).mean()*100:.1f}%)")
        logger.info(f"  High (2):   {(regime == 2).sum()} samples ({(regime == 2).mean()*100:.1f}%)")
        
        return regime
    
    def detect_trend_regime(
        self,
        df: pd.DataFrame,
        threshold: float = 0.001,
    ) -> pd.Series:
        """Detect trend regime using moving average slope.
        
        Args:
            df: DataFrame with 'close' column
            threshold: Threshold for ranging market (return magnitude)
            
        Returns:
            Series with regime labels (0=down, 1=ranging, 2=up)
        """
        # Calculate trend using moving average
        ma = df['close'].rolling(window=self.trend_window).mean()
        ma_return = ma.pct_change()
        
        # Classify regimes
        regime = pd.Series(1, index=df.index)  # Default to ranging
        regime[ma_return < -threshold] = 0     # Trending down
        regime[ma_return > threshold] = 2      # Trending up
        
        logger.info("Detected trend regimes:")
        logger.info(f"  Down (0):    {(regime == 0).sum()} samples ({(regime == 0).mean()*100:.1f}%)")
        logger.info(f"  Ranging (1): {(regime == 1).sum()} samples ({(regime == 1).mean()*100:.1f}%)")
        logger.info(f"  Up (2):      {(regime == 2).sum()} samples ({(regime == 2).mean()*100:.1f}%)")
        
        return regime
    
    def detect_combined_regime(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Detect both volatility and trend regimes.
        
        Args:
            df: DataFrame with 'close' column
            
        Returns:
            DataFrame with regime columns
        """
        regimes = pd.DataFrame(index=df.index)
        regimes['volatility_regime'] = self.detect_volatility_regime(df)
        regimes['trend_regime'] = self.detect_trend_regime(df)
        
        # Create combined label
        regimes['combined_regime'] = (
            regimes['volatility_regime'].astype(str) + '_' +
            regimes['trend_regime'].astype(str)
        )
        
        return regimes


class RegimeEvaluator:
    """Evaluate model performance across different market regimes."""
    
    @staticmethod
    def evaluate_by_regime(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        regime: np.ndarray,
        eval_func: callable,
    ) -> dict[str, dict[str, float]]:
        """Evaluate model performance for each regime.
        
        Args:
            y_true: True labels/values
            y_pred: Predicted labels/values
            regime: Regime labels
            eval_func: Function to compute metrics from (y_true, y_pred)
            
        Returns:
            Dictionary mapping regime to metrics
        """
        unique_regimes = np.unique(regime)
        results = {}
        
        for r in unique_regimes:
            mask = regime == r
            
            if mask.sum() == 0:
                continue
            
            y_true_r = y_true[mask]
            y_pred_r = y_pred[mask]
            
            metrics = eval_func(y_true_r, y_pred_r)
            metrics['sample_count'] = mask.sum()
            
            results[str(r)] = metrics
        
        return results
    
    @staticmethod
    def compare_regime_performance(
        regime_results: dict[str, dict[str, float]],
        metric: str = 'accuracy',
    ) -> pd.DataFrame:
        """Compare performance across regimes.
        
        Args:
            regime_results: Results from evaluate_by_regime
            metric: Metric to compare
            
        Returns:
            DataFrame with comparison
        """
        comparison = []
        
        for regime, metrics in regime_results.items():
            row = {
                'regime': regime,
                'sample_count': metrics.get('sample_count', 0),
            }
            
            for key, value in metrics.items():
                if key != 'sample_count' and isinstance(value, (int, float)):
                    row[key] = value
            
            comparison.append(row)
        
        df = pd.DataFrame(comparison)
        
        if metric in df.columns:
            df = df.sort_values(metric, ascending=False)
        
        return df
    
    @staticmethod
    def print_regime_analysis(
        regime_results: dict[str, dict[str, float]],
        regime_type: str = "volatility",
    ) -> None:
        """Print regime analysis in readable format.
        
        Args:
            regime_results: Results from evaluate_by_regime
            regime_type: Type of regime (for labeling)
        """
        regime_names = {
            '0': 'Low' if regime_type == 'volatility' else 'Down',
            '1': 'Medium' if regime_type == 'volatility' else 'Ranging',
            '2': 'High' if regime_type == 'volatility' else 'Up',
        }
        
        print("=" * 70)
        print(f"Performance by {regime_type.title()} Regime")
        print("=" * 70)
        print()
        
        for regime, metrics in regime_results.items():
            regime_name = regime_names.get(regime, regime)
            print(f"{regime_name} Regime ({metrics.get('sample_count', 0)} samples):")
            
            for key, value in metrics.items():
                if key != 'sample_count' and isinstance(value, (int, float)):
                    print(f"  {key:.<30} {value:.4f}")
            
            print()
        
        print("=" * 70)
    
    @staticmethod
    def calculate_regime_stability(
        regime_results: dict[str, dict[str, float]],
        metric: str = 'accuracy',
    ) -> dict[str, float]:
        """Calculate stability metrics across regimes.
        
        Args:
            regime_results: Results from evaluate_by_regime
            metric: Metric to analyze
            
        Returns:
            Dictionary with stability metrics
        """
        values = []
        
        for metrics in regime_results.values():
            if metric in metrics:
                values.append(metrics[metric])
        
        if not values:
            return {}
        
        values = np.array(values)
        
        stability = {
            'mean': values.mean(),
            'std': values.std(),
            'min': values.min(),
            'max': values.max(),
            'range': values.max() - values.min(),
            'coefficient_of_variation': values.std() / values.mean() if values.mean() > 0 else None,
        }
        
        return stability


class RegimeTransitionAnalyzer:
    """Analyze model performance during regime transitions."""
    
    @staticmethod
    def detect_transitions(
        regime: pd.Series,
    ) -> pd.Series:
        """Detect regime transitions.
        
        Args:
            regime: Series with regime labels
            
        Returns:
            Boolean series indicating transitions
        """
        transitions = regime != regime.shift(1)
        return transitions
    
    @staticmethod
    def evaluate_at_transitions(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        transitions: np.ndarray,
        window: int = 5,
        eval_func: callable = None,
    ) -> dict[str, Any]:
        """Evaluate model performance around regime transitions.
        
        Args:
            y_true: True labels/values
            y_pred: Predicted labels/values
            transitions: Boolean array indicating transitions
            window: Window size around transitions
            eval_func: Function to compute metrics
            
        Returns:
            Dictionary with transition metrics
        """
        # Get indices of transitions
        transition_indices = np.where(transitions)[0]
        
        # Collect predictions around transitions
        transition_mask = np.zeros(len(y_true), dtype=bool)
        
        for idx in transition_indices:
            start = max(0, idx - window)
            end = min(len(y_true), idx + window + 1)
            transition_mask[start:end] = True
        
        # Evaluate on transition periods
        y_true_trans = y_true[transition_mask]
        y_pred_trans = y_pred[transition_mask]
        
        # Evaluate on stable periods
        y_true_stable = y_true[~transition_mask]
        y_pred_stable = y_pred[~transition_mask]
        
        results = {
            'num_transitions': len(transition_indices),
            'transition_samples': transition_mask.sum(),
            'stable_samples': (~transition_mask).sum(),
        }
        
        if eval_func and len(y_true_trans) > 0:
            results['transition_metrics'] = eval_func(y_true_trans, y_pred_trans)
        
        if eval_func and len(y_true_stable) > 0:
            results['stable_metrics'] = eval_func(y_true_stable, y_pred_stable)
        
        return results
    
    @staticmethod
    def print_transition_analysis(
        results: dict[str, Any],
    ) -> None:
        """Print transition analysis in readable format.
        
        Args:
            results: Results from evaluate_at_transitions
        """
        print("=" * 70)
        print("Regime Transition Analysis")
        print("=" * 70)
        print()
        
        print(f"Number of transitions: {results.get('num_transitions', 0)}")
        print(f"Samples in transition periods: {results.get('transition_samples', 0)}")
        print(f"Samples in stable periods: {results.get('stable_samples', 0)}")
        print()
        
        if 'transition_metrics' in results:
            print("Performance during transitions:")
            for key, value in results['transition_metrics'].items():
                if isinstance(value, (int, float)):
                    print(f"  {key:.<30} {value:.4f}")
            print()
        
        if 'stable_metrics' in results:
            print("Performance during stable periods:")
            for key, value in results['stable_metrics'].items():
                if isinstance(value, (int, float)):
                    print(f"  {key:.<30} {value:.4f}")
            print()
        
        print("=" * 70)
