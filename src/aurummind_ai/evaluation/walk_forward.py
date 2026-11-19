"""Walk-forward validation for time series models.

This module provides walk-forward analysis to simulate
realistic model deployment with rolling retraining.
"""

import logging
from datetime import datetime
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

logger = logging.getLogger(__name__)


class WalkForwardValidator:
    """Walk-forward validation for time series models.
    
    Implements rolling window approach:
    1. Train on initial window
    2. Predict on next period
    3. Roll window forward
    4. Retrain and predict
    5. Repeat until end of data
    
    This simulates realistic deployment where models are
    periodically retrained on latest data.
    """
    
    def __init__(
        self,
        train_window: int,
        test_window: int,
        step_size: Optional[int] = None,
        retrain_every: int = 1,
    ):
        """Initialize walk-forward validator.
        
        Args:
            train_window: Number of samples in training window
            test_window: Number of samples to predict
            step_size: Step size for window roll (default: test_window)
            retrain_every: Retrain frequency (1=every step, 2=every other step, etc.)
        """
        self.train_window = train_window
        self.test_window = test_window
        self.step_size = step_size or test_window
        self.retrain_every = retrain_every
        
        self.results: list[dict] = []
        self.predictions: list[np.ndarray] = []
        self.actuals: list[np.ndarray] = []
        
    def validate(
        self,
        model_factory: Callable[[], BaseEstimator],
        X: pd.DataFrame,
        y: pd.Series,
        eval_func: Callable[[np.ndarray, np.ndarray], dict[str, float]],
    ) -> dict[str, Any]:
        """Run walk-forward validation.
        
        Args:
            model_factory: Factory function that returns fresh model instance
            X: Feature matrix
            y: Target series
            eval_func: Function to compute metrics from (y_true, y_pred)
            
        Returns:
            Dictionary with validation results
        """
        logger.info("Starting walk-forward validation")
        logger.info(f"  Train window: {self.train_window}")
        logger.info(f"  Test window: {self.test_window}")
        logger.info(f"  Step size: {self.step_size}")
        logger.info(f"  Retrain every: {self.retrain_every} steps")
        
        n = len(X)
        self.results = []
        self.predictions = []
        self.actuals = []
        
        model = None
        step = 0
        
        # Start position for first window
        train_start = 0
        train_end = self.train_window
        test_start = train_end
        test_end = min(test_start + self.test_window, n)
        
        while test_start < n:
            step += 1
            
            # Get training data
            X_train = X.iloc[train_start:train_end]
            y_train = y.iloc[train_start:train_end]
            
            # Get test data
            X_test = X.iloc[test_start:test_end]
            y_test = y.iloc[test_start:test_end]
            
            # Retrain if needed
            should_retrain = (step - 1) % self.retrain_every == 0
            
            if should_retrain or model is None:
                logger.info(f"Step {step}: Training on [{train_start}:{train_end}]")
                model = model_factory()
                
                # Handle NaN in training data
                train_mask = ~(X_train.isna().any(axis=1) | y_train.isna())
                X_train_clean = X_train[train_mask]
                y_train_clean = y_train[train_mask]
                
                if len(X_train_clean) < 10:
                    logger.warning(f"Step {step}: Insufficient training data after cleaning")
                    break
                
                model.fit(X_train_clean, y_train_clean)
            
            # Predict on test set
            logger.info(f"Step {step}: Testing on [{test_start}:{test_end}]")
            
            # Handle NaN in test data
            test_mask = ~X_test.isna().any(axis=1)
            X_test_clean = X_test[test_mask]
            y_test_clean = y_test[test_mask]
            
            if len(X_test_clean) == 0:
                logger.warning(f"Step {step}: No valid test samples")
                # Roll forward
                train_start += self.step_size
                train_end = train_start + self.train_window
                test_start = train_end
                test_end = min(test_start + self.test_window, n)
                continue
            
            y_pred = model.predict(X_test_clean)
            
            # Evaluate
            metrics = eval_func(y_test_clean.values, y_pred)
            
            # Store results
            result = {
                'step': step,
                'train_start': train_start,
                'train_end': train_end,
                'test_start': test_start,
                'test_end': test_end,
                'train_size': len(X_train_clean),
                'test_size': len(X_test_clean),
                'retrained': should_retrain,
                **metrics,
            }
            
            self.results.append(result)
            self.predictions.append(y_pred)
            self.actuals.append(y_test_clean.values)
            
            logger.info(f"  Metrics: {metrics}")
            
            # Roll window forward
            train_start += self.step_size
            train_end = train_start + self.train_window
            test_start = train_end
            test_end = min(test_start + self.test_window, n)
        
        logger.info(f"Walk-forward validation complete: {len(self.results)} steps")
        
        return self._aggregate_results()
    
    def _aggregate_results(self) -> dict[str, Any]:
        """Aggregate results across all steps.
        
        Returns:
            Dictionary with aggregated metrics
        """
        if not self.results:
            return {}
        
        results_df = pd.DataFrame(self.results)
        
        # Extract metric columns (exclude metadata)
        metric_cols = [
            col for col in results_df.columns
            if col not in ['step', 'train_start', 'train_end', 'test_start', 
                          'test_end', 'train_size', 'test_size', 'retrained']
        ]
        
        # Aggregate metrics
        aggregated = {
            'num_steps': len(self.results),
            'total_predictions': sum(r['test_size'] for r in self.results),
        }
        
        for col in metric_cols:
            values = results_df[col].dropna()
            if len(values) > 0:
                aggregated[f'{col}_mean'] = values.mean()
                aggregated[f'{col}_std'] = values.std()
                aggregated[f'{col}_min'] = values.min()
                aggregated[f'{col}_max'] = values.max()
        
        # Overall predictions
        all_predictions = np.concatenate(self.predictions)
        all_actuals = np.concatenate(self.actuals)
        
        aggregated['all_predictions'] = all_predictions
        aggregated['all_actuals'] = all_actuals
        
        # Step-by-step results
        aggregated['step_results'] = results_df
        
        return aggregated
    
    def get_predictions(self) -> tuple[np.ndarray, np.ndarray]:
        """Get all predictions and actuals.
        
        Returns:
            Tuple of (predictions, actuals)
        """
        predictions = np.concatenate(self.predictions) if self.predictions else np.array([])
        actuals = np.concatenate(self.actuals) if self.actuals else np.array([])
        return predictions, actuals
    
    def get_results_dataframe(self) -> pd.DataFrame:
        """Get results as DataFrame.
        
        Returns:
            DataFrame with step-by-step results
        """
        return pd.DataFrame(self.results)
    
    def plot_performance_over_time(self, metric: str = 'accuracy') -> None:
        """Plot metric performance over time steps.
        
        Args:
            metric: Metric to plot
        """
        try:
            import plotly.graph_objects as go
        except ImportError:
            logger.warning("plotly not available for plotting")
            return
        
        df = self.get_results_dataframe()
        
        if metric not in df.columns:
            logger.warning(f"Metric '{metric}' not in results")
            return
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df['step'],
            y=df[metric],
            mode='lines+markers',
            name=metric,
            line=dict(color='blue', width=2),
            marker=dict(size=8),
        ))
        
        # Add mean line
        mean_value = df[metric].mean()
        fig.add_hline(
            y=mean_value,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Mean: {mean_value:.4f}",
        )
        
        fig.update_layout(
            title=f"Walk-Forward Validation: {metric} Over Time",
            xaxis_title="Step",
            yaxis_title=metric.replace('_', ' ').title(),
            hovermode='x unified',
            template='plotly_white',
        )
        
        fig.show()


class AnchoredWalkForward:
    """Anchored walk-forward validation.
    
    Similar to standard walk-forward, but training window
    is anchored to the start and grows over time instead of rolling.
    
    This can be useful when you want to use all available
    historical data for each training iteration.
    """
    
    def __init__(
        self,
        min_train_size: int,
        test_window: int,
        step_size: Optional[int] = None,
    ):
        """Initialize anchored walk-forward validator.
        
        Args:
            min_train_size: Minimum training window size
            test_window: Number of samples to predict
            step_size: Step size for window roll (default: test_window)
        """
        self.min_train_size = min_train_size
        self.test_window = test_window
        self.step_size = step_size or test_window
        
        self.results: list[dict] = []
        self.predictions: list[np.ndarray] = []
        self.actuals: list[np.ndarray] = []
    
    def validate(
        self,
        model_factory: Callable[[], BaseEstimator],
        X: pd.DataFrame,
        y: pd.Series,
        eval_func: Callable[[np.ndarray, np.ndarray], dict[str, float]],
    ) -> dict[str, Any]:
        """Run anchored walk-forward validation.
        
        Args:
            model_factory: Factory function that returns fresh model instance
            X: Feature matrix
            y: Target series
            eval_func: Function to compute metrics from (y_true, y_pred)
            
        Returns:
            Dictionary with validation results
        """
        logger.info("Starting anchored walk-forward validation")
        logger.info(f"  Min train size: {self.min_train_size}")
        logger.info(f"  Test window: {self.test_window}")
        logger.info(f"  Step size: {self.step_size}")
        
        n = len(X)
        self.results = []
        self.predictions = []
        self.actuals = []
        
        step = 0
        
        # Anchored to start
        train_start = 0
        test_start = self.min_train_size
        test_end = min(test_start + self.test_window, n)
        
        while test_start < n:
            step += 1
            train_end = test_start
            
            # Get training data (from start to test_start)
            X_train = X.iloc[train_start:train_end]
            y_train = y.iloc[train_start:train_end]
            
            # Get test data
            X_test = X.iloc[test_start:test_end]
            y_test = y.iloc[test_start:test_end]
            
            logger.info(f"Step {step}: Training on [{train_start}:{train_end}], Testing on [{test_start}:{test_end}]")
            
            # Train model
            model = model_factory()
            
            # Handle NaN
            train_mask = ~(X_train.isna().any(axis=1) | y_train.isna())
            X_train_clean = X_train[train_mask]
            y_train_clean = y_train[train_mask]
            
            if len(X_train_clean) < self.min_train_size:
                logger.warning(f"Step {step}: Insufficient training data")
                break
            
            model.fit(X_train_clean, y_train_clean)
            
            # Predict
            test_mask = ~X_test.isna().any(axis=1)
            X_test_clean = X_test[test_mask]
            y_test_clean = y_test[test_mask]
            
            if len(X_test_clean) == 0:
                logger.warning(f"Step {step}: No valid test samples")
                test_start += self.step_size
                test_end = min(test_start + self.test_window, n)
                continue
            
            y_pred = model.predict(X_test_clean)
            
            # Evaluate
            metrics = eval_func(y_test_clean.values, y_pred)
            
            # Store results
            result = {
                'step': step,
                'train_start': train_start,
                'train_end': train_end,
                'test_start': test_start,
                'test_end': test_end,
                'train_size': len(X_train_clean),
                'test_size': len(X_test_clean),
                **metrics,
            }
            
            self.results.append(result)
            self.predictions.append(y_pred)
            self.actuals.append(y_test_clean.values)
            
            logger.info(f"  Metrics: {metrics}")
            
            # Roll test window forward
            test_start += self.step_size
            test_end = min(test_start + self.test_window, n)
        
        logger.info(f"Anchored walk-forward validation complete: {len(self.results)} steps")
        
        return self._aggregate_results()
    
    def _aggregate_results(self) -> dict[str, Any]:
        """Aggregate results across all steps."""
        if not self.results:
            return {}
        
        results_df = pd.DataFrame(self.results)
        
        metric_cols = [
            col for col in results_df.columns
            if col not in ['step', 'train_start', 'train_end', 'test_start', 
                          'test_end', 'train_size', 'test_size']
        ]
        
        aggregated = {
            'num_steps': len(self.results),
            'total_predictions': sum(r['test_size'] for r in self.results),
        }
        
        for col in metric_cols:
            values = results_df[col].dropna()
            if len(values) > 0:
                aggregated[f'{col}_mean'] = values.mean()
                aggregated[f'{col}_std'] = values.std()
                aggregated[f'{col}_min'] = values.min()
                aggregated[f'{col}_max'] = values.max()
        
        all_predictions = np.concatenate(self.predictions)
        all_actuals = np.concatenate(self.actuals)
        
        aggregated['all_predictions'] = all_predictions
        aggregated['all_actuals'] = all_actuals
        aggregated['step_results'] = results_df
        
        return aggregated
    
    def get_predictions(self) -> tuple[np.ndarray, np.ndarray]:
        """Get all predictions and actuals."""
        predictions = np.concatenate(self.predictions) if self.predictions else np.array([])
        actuals = np.concatenate(self.actuals) if self.actuals else np.array([])
        return predictions, actuals
    
    def get_results_dataframe(self) -> pd.DataFrame:
        """Get results as DataFrame."""
        return pd.DataFrame(self.results)
