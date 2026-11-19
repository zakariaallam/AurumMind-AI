"""Gradient boosting models (XGBoost and LightGBM).

This module implements advanced gradient boosting models
for classification and regression tasks.
"""

import logging
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb

logger = logging.getLogger(__name__)


class BoostingModels:
    """Advanced gradient boosting models.
    
    Implements XGBoost and LightGBM for both classification
    and regression tasks.
    """
    
    def __init__(self, random_state: int = 42):
        """Initialize boosting models.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.scaler: Optional[StandardScaler] = None
        self.models: dict[str, Any] = {}
        self.feature_names: list[str] = []
    
    def create_xgboost_classifier(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        scale_pos_weight: float = 1.0,
    ) -> xgb.XGBClassifier:
        """Create XGBoost classifier.
        
        Args:
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth
            learning_rate: Learning rate (eta)
            subsample: Subsample ratio of training data
            colsample_bytree: Subsample ratio of columns
            scale_pos_weight: Balancing of positive/negative weights
            
        Returns:
            XGBoost classifier
        """
        model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            scale_pos_weight=scale_pos_weight,
            random_state=self.random_state,
            n_jobs=-1,
            eval_metric='logloss',
        )
        
        self.models['xgboost_classifier'] = model
        logger.info(f"✓ Created XGBoost classifier ({n_estimators} estimators)")
        
        return model
    
    def create_xgboost_regressor(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
    ) -> xgb.XGBRegressor:
        """Create XGBoost regressor.
        
        Args:
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth
            learning_rate: Learning rate (eta)
            subsample: Subsample ratio of training data
            colsample_bytree: Subsample ratio of columns
            
        Returns:
            XGBoost regressor
        """
        model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=self.random_state,
            n_jobs=-1,
            eval_metric='rmse',
        )
        
        self.models['xgboost_regressor'] = model
        logger.info(f"✓ Created XGBoost regressor ({n_estimators} estimators)")
        
        return model
    
    def create_lightgbm_classifier(
        self,
        n_estimators: int = 100,
        max_depth: int = -1,
        learning_rate: float = 0.1,
        num_leaves: int = 31,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        is_unbalance: bool = False,
    ) -> lgb.LGBMClassifier:
        """Create LightGBM classifier.
        
        Args:
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth (-1 for no limit)
            learning_rate: Learning rate
            num_leaves: Maximum number of leaves
            subsample: Subsample ratio
            colsample_bytree: Feature fraction
            is_unbalance: Handle imbalanced data
            
        Returns:
            LightGBM classifier
        """
        model = lgb.LGBMClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            is_unbalance=is_unbalance,
            random_state=self.random_state,
            n_jobs=-1,
            verbose=-1,
        )
        
        self.models['lightgbm_classifier'] = model
        logger.info(f"✓ Created LightGBM classifier ({n_estimators} estimators)")
        
        return model
    
    def create_lightgbm_regressor(
        self,
        n_estimators: int = 100,
        max_depth: int = -1,
        learning_rate: float = 0.1,
        num_leaves: int = 31,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
    ) -> lgb.LGBMRegressor:
        """Create LightGBM regressor.
        
        Args:
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth (-1 for no limit)
            learning_rate: Learning rate
            num_leaves: Maximum number of leaves
            subsample: Subsample ratio
            colsample_bytree: Feature fraction
            
        Returns:
            LightGBM regressor
        """
        model = lgb.LGBMRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=self.random_state,
            n_jobs=-1,
            verbose=-1,
        )
        
        self.models['lightgbm_regressor'] = model
        logger.info(f"✓ Created LightGBM regressor ({n_estimators} estimators)")
        
        return model
    
    def fit_scaler(self, X: pd.DataFrame) -> StandardScaler:
        """Fit standard scaler on features.
        
        Args:
            X: Feature matrix
            
        Returns:
            Fitted scaler
        """
        self.scaler = StandardScaler()
        self.scaler.fit(X)
        self.feature_names = X.columns.tolist()
        
        logger.info(f"✓ Fitted scaler on {len(self.feature_names)} features")
        
        return self.scaler
    
    def transform_features(self, X: pd.DataFrame) -> np.ndarray:
        """Transform features using fitted scaler.
        
        Note: Gradient boosting models often don't require scaling,
        but it's provided for consistency.
        
        Args:
            X: Feature matrix
            
        Returns:
            Scaled features
        """
        if self.scaler is None:
            raise ValueError("Scaler not fitted. Call fit_scaler() first.")
        
        return self.scaler.transform(X)
    
    def fit(
        self,
        model_name: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        eval_set: Optional[list[tuple]] = None,
        early_stopping_rounds: Optional[int] = None,
        verbose: bool = False,
    ) -> Any:
        """Fit a boosting model.
        
        Args:
            model_name: Name of model to fit
            X_train: Training features
            y_train: Training target
            eval_set: Optional validation set [(X_val, y_val)]
            early_stopping_rounds: Stop if no improvement
            verbose: Print training progress
            
        Returns:
            Fitted model
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found. Create it first.")
        
        model = self.models[model_name]
        
        # Store feature names
        if not self.feature_names:
            self.feature_names = X_train.columns.tolist()
        
        logger.info(f"Training {model_name}...")
        logger.info(f"  Training samples: {len(X_train)}")
        logger.info(f"  Features: {X_train.shape[1]}")
        
        # Fit with optional early stopping
        fit_params = {}
        if eval_set is not None:
            fit_params['eval_set'] = eval_set
        if early_stopping_rounds is not None:
            fit_params['early_stopping_rounds'] = early_stopping_rounds
        if verbose:
            fit_params['verbose'] = True
        
        model.fit(X_train, y_train, **fit_params)
        
        logger.info(f"✓ {model_name} trained successfully")
        
        # Log best iteration if available
        if hasattr(model, 'best_iteration_'):
            logger.info(f"  Best iteration: {model.best_iteration_}")
        
        return model
    
    def predict(self, model_name: str, X: pd.DataFrame) -> np.ndarray:
        """Make predictions.
        
        Args:
            model_name: Name of model
            X: Feature matrix
            
        Returns:
            Predictions
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found")
        
        model = self.models[model_name]
        return model.predict(X)
    
    def predict_proba(self, model_name: str, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities.
        
        Args:
            model_name: Name of classifier
            X: Feature matrix
            
        Returns:
            Class probabilities
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found")
        
        model = self.models[model_name]
        
        if not hasattr(model, 'predict_proba'):
            raise ValueError(f"Model '{model_name}' does not support probability predictions")
        
        return model.predict_proba(X)
    
    def get_feature_importance(
        self,
        model_name: str,
        importance_type: str = 'gain',
        top_n: Optional[int] = None,
    ) -> pd.DataFrame:
        """Get feature importance.
        
        Args:
            model_name: Name of model
            importance_type: 'gain', 'weight', or 'cover' (XGBoost/LightGBM)
            top_n: Return only top N features
            
        Returns:
            DataFrame with feature importance
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found")
        
        model = self.models[model_name]
        
        if not hasattr(model, 'feature_importances_'):
            raise ValueError(f"Model '{model_name}' does not have feature_importances_")
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        if top_n is not None:
            importance_df = importance_df.head(top_n)
        
        return importance_df
    
    def get_model(self, model_name: str) -> Any:
        """Get a specific model.
        
        Args:
            model_name: Name of model
            
        Returns:
            Model object
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found")
        
        return self.models[model_name]
    
    def list_models(self) -> list[str]:
        """List all created models.
        
        Returns:
            List of model names
        """
        return list(self.models.keys())
