"""Model training orchestration and management.

This module provides a unified interface for training,
evaluating, and managing machine learning models.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from .baseline import BaselineModels
from .boosting import BoostingModels

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Unified model training and management.
    
    Provides interface for:
    - Target variable creation
    - Train/validation/test splits
    - Model training
    - Model evaluation
    - Model persistence
    """
    
    def __init__(self, random_state: int = 42):
        """Initialize model trainer.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.baseline = BaselineModels(random_state=random_state)
        self.boosting = BoostingModels(random_state=random_state)
        
        self.target_column: Optional[str] = None
        self.feature_columns: list[str] = []
        self.split_indices: dict[str, tuple] = {}
    
    def create_direction_target(
        self,
        df: pd.DataFrame,
        horizon: int = 1,
        threshold: float = 0.0,
    ) -> pd.Series:
        """Create direction classification target.
        
        Classes:
        - 0: Down (return < -threshold)
        - 1: Sideways (-threshold <= return <= threshold)
        - 2: Up (return > threshold)
        
        Args:
            df: DataFrame with close price
            horizon: Periods ahead to predict
            threshold: Threshold for sideways (percentage)
            
        Returns:
            Series with class labels
        """
        future_return = df['close'].pct_change(periods=horizon).shift(-horizon)
        
        target = pd.Series(1, index=df.index)  # Default to sideways
        target[future_return < -threshold] = 0  # Down
        target[future_return > threshold] = 2   # Up
        
        logger.info(f"Created direction target (horizon={horizon}, threshold={threshold})")
        logger.info(f"  Class distribution:")
        logger.info(f"    Down (0): {(target == 0).sum()} ({(target == 0).mean()*100:.1f}%)")
        logger.info(f"    Sideways (1): {(target == 1).sum()} ({(target == 1).mean()*100:.1f}%)")
        logger.info(f"    Up (2): {(target == 2).sum()} ({(target == 2).mean()*100:.1f}%)")
        
        return target
    
    def create_binary_direction_target(
        self,
        df: pd.DataFrame,
        horizon: int = 1,
    ) -> pd.Series:
        """Create binary direction target (up vs down).
        
        Classes:
        - 0: Down (return < 0)
        - 1: Up (return >= 0)
        
        Args:
            df: DataFrame with close price
            horizon: Periods ahead to predict
            
        Returns:
            Series with binary labels
        """
        future_return = df['close'].pct_change(periods=horizon).shift(-horizon)
        target = (future_return >= 0).astype(int)
        
        logger.info(f"Created binary direction target (horizon={horizon})")
        logger.info(f"  Class distribution:")
        logger.info(f"    Down (0): {(target == 0).sum()} ({(target == 0).mean()*100:.1f}%)")
        logger.info(f"    Up (1): {(target == 1).sum()} ({(target == 1).mean()*100:.1f}%)")
        
        return target
    
    def create_return_target(
        self,
        df: pd.DataFrame,
        horizon: int = 1,
    ) -> pd.Series:
        """Create continuous return target for regression.
        
        Args:
            df: DataFrame with close price
            horizon: Periods ahead to predict
            
        Returns:
            Series with future returns
        """
        future_return = df['close'].pct_change(periods=horizon).shift(-horizon)
        
        logger.info(f"Created return target (horizon={horizon})")
        logger.info(f"  Mean return: {future_return.mean()*100:.4f}%")
        logger.info(f"  Std return: {future_return.std()*100:.4f}%")
        logger.info(f"  Min return: {future_return.min()*100:.4f}%")
        logger.info(f"  Max return: {future_return.max()*100:.4f}%")
        
        return future_return
    
    def temporal_train_test_split(
        self,
        df: pd.DataFrame,
        target: pd.Series,
        test_size: float = 0.2,
        val_size: float = 0.2,
    ) -> dict[str, tuple[pd.DataFrame, pd.Series]]:
        """Create temporal train/validation/test split.
        
        No shuffling - maintains time order.
        
        Args:
            df: Feature DataFrame
            target: Target series
            test_size: Proportion for test set
            val_size: Proportion of remaining for validation
            
        Returns:
            Dictionary with train, val, test splits
        """
        # Remove rows with NaN target
        valid_idx = ~target.isna()
        df_clean = df[valid_idx].copy()
        target_clean = target[valid_idx].copy()
        
        n = len(df_clean)
        n_test = int(n * test_size)
        n_val = int((n - n_test) * val_size)
        n_train = n - n_test - n_val
        
        # Split indices
        train_idx = slice(0, n_train)
        val_idx = slice(n_train, n_train + n_val)
        test_idx = slice(n_train + n_val, n)
        
        # Create splits
        X_train = df_clean.iloc[train_idx]
        y_train = target_clean.iloc[train_idx]
        
        X_val = df_clean.iloc[val_idx]
        y_val = target_clean.iloc[val_idx]
        
        X_test = df_clean.iloc[test_idx]
        y_test = target_clean.iloc[test_idx]
        
        # Store split info
        self.split_indices = {
            'train': (train_idx.start, train_idx.stop),
            'val': (val_idx.start, val_idx.stop),
            'test': (test_idx.start, test_idx.stop),
        }
        
        logger.info("Created temporal train/val/test split:")
        logger.info(f"  Train: {len(X_train)} samples ({len(X_train)/n*100:.1f}%)")
        logger.info(f"  Val:   {len(X_val)} samples ({len(X_val)/n*100:.1f}%)")
        logger.info(f"  Test:  {len(X_test)} samples ({len(X_test)/n*100:.1f}%)")
        
        return {
            'train': (X_train, y_train),
            'val': (X_val, y_val),
            'test': (X_test, y_test),
        }
    
    def prepare_features(
        self,
        df: pd.DataFrame,
        exclude_columns: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """Prepare feature matrix by excluding non-feature columns.
        
        Args:
            df: DataFrame with all columns
            exclude_columns: Columns to exclude (default: timestamp, OHLCV)
            
        Returns:
            Feature matrix
        """
        if exclude_columns is None:
            exclude_columns = [
                'timestamp', 'open', 'high', 'low', 'close', 'volume'
            ]
        
        feature_cols = [col for col in df.columns if col not in exclude_columns]
        X = df[feature_cols].copy()
        
        # Drop columns with all NaN
        X = X.dropna(axis=1, how='all')
        
        self.feature_columns = X.columns.tolist()
        
        logger.info(f"Prepared {len(self.feature_columns)} features")
        
        return X
    
    def create_time_series_cv(
        self,
        n_splits: int = 5,
        gap: int = 0,
    ) -> TimeSeriesSplit:
        """Create time series cross-validator.
        
        Args:
            n_splits: Number of splits
            gap: Gap between train and test
            
        Returns:
            TimeSeriesSplit object
        """
        tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap)
        logger.info(f"Created TimeSeriesSplit with {n_splits} splits")
        return tscv
    
    def save_model_metadata(
        self,
        model_name: str,
        model_type: str,
        output_dir: str,
        metrics: dict[str, Any],
        additional_info: Optional[dict] = None,
    ) -> Path:
        """Save model metadata.
        
        Args:
            model_name: Name of the model
            model_type: Type (classifier/regressor)
            output_dir: Output directory
            metrics: Performance metrics
            additional_info: Additional information
            
        Returns:
            Path to metadata file
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        metadata = {
            'model_name': model_name,
            'model_type': model_type,
            'timestamp': datetime.now().isoformat(),
            'random_state': self.random_state,
            'features': {
                'count': len(self.feature_columns),
                'names': self.feature_columns[:50],  # First 50
            },
            'target': self.target_column,
            'split_info': self.split_indices,
            'metrics': metrics,
        }
        
        if additional_info:
            metadata.update(additional_info)
        
        filename = f"{model_name}_metadata.json"
        metadata_path = output_path / filename
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"✓ Saved metadata to {metadata_path}")
        
        return metadata_path
    
    def get_baseline_models(self) -> BaselineModels:
        """Get baseline models manager.
        
        Returns:
            BaselineModels instance
        """
        return self.baseline
    
    def get_boosting_models(self) -> BoostingModels:
        """Get boosting models manager.
        
        Returns:
            BoostingModels instance
        """
        return self.boosting
