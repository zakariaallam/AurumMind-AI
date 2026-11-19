"""Baseline machine learning models.

This module implements simple baseline models for comparison
with more advanced approaches.
"""

import logging
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class BaselineModels:
    """Baseline machine learning models for comparison.
    
    Provides simple models as benchmarks for more complex approaches.
    Includes both classification and regression variants.
    """
    
    def __init__(self, random_state: int = 42):
        """Initialize baseline models.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.scaler: Optional[StandardScaler] = None
        self.models: dict[str, Any] = {}
        self.feature_names: list[str] = []
    
    def create_logistic_regression(
        self,
        C: float = 1.0,
        max_iter: int = 1000,
        class_weight: Optional[str] = None,
    ) -> LogisticRegression:
        """Create Logistic Regression classifier.
        
        Args:
            C: Inverse of regularization strength
            max_iter: Maximum iterations
            class_weight: 'balanced' or None
            
        Returns:
            Logistic Regression model
        """
        model = LogisticRegression(
            C=C,
            max_iter=max_iter,
            class_weight=class_weight,
            random_state=self.random_state,
            n_jobs=-1,
        )
        
        self.models['logistic_regression'] = model
        logger.info("✓ Created Logistic Regression model")
        
        return model
    
    def create_random_forest_classifier(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        class_weight: Optional[str] = None,
    ) -> RandomForestClassifier:
        """Create Random Forest classifier.
        
        Args:
            n_estimators: Number of trees
            max_depth: Maximum tree depth
            min_samples_split: Minimum samples to split
            min_samples_leaf: Minimum samples per leaf
            class_weight: 'balanced' or None
            
        Returns:
            Random Forest classifier
        """
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            class_weight=class_weight,
            random_state=self.random_state,
            n_jobs=-1,
        )
        
        self.models['random_forest_classifier'] = model
        logger.info(f"✓ Created Random Forest classifier ({n_estimators} trees)")
        
        return model
    
    def create_linear_regression(self) -> LinearRegression:
        """Create Linear Regression model.
        
        Returns:
            Linear Regression model
        """
        model = LinearRegression(n_jobs=-1)
        
        self.models['linear_regression'] = model
        logger.info("✓ Created Linear Regression model")
        
        return model
    
    def create_random_forest_regressor(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
    ) -> RandomForestRegressor:
        """Create Random Forest regressor.
        
        Args:
            n_estimators: Number of trees
            max_depth: Maximum tree depth
            min_samples_split: Minimum samples to split
            min_samples_leaf: Minimum samples per leaf
            
        Returns:
            Random Forest regressor
        """
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=self.random_state,
            n_jobs=-1,
        )
        
        self.models['random_forest_regressor'] = model
        logger.info(f"✓ Created Random Forest regressor ({n_estimators} trees)")
        
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
        scale_features: bool = True,
    ) -> Any:
        """Fit a model.
        
        Args:
            model_name: Name of model to fit
            X_train: Training features
            y_train: Training target
            scale_features: Whether to scale features
            
        Returns:
            Fitted model
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found. Create it first.")
        
        model = self.models[model_name]
        
        # Scale features if requested
        if scale_features:
            if self.scaler is None:
                self.fit_scaler(X_train)
            X_train_scaled = self.transform_features(X_train)
        else:
            X_train_scaled = X_train.values
        
        # Fit model
        logger.info(f"Training {model_name}...")
        logger.info(f"  Training samples: {len(X_train)}")
        logger.info(f"  Features: {X_train.shape[1]}")
        
        model.fit(X_train_scaled, y_train)
        
        logger.info(f"✓ {model_name} trained successfully")
        
        return model
    
    def predict(
        self,
        model_name: str,
        X: pd.DataFrame,
        scale_features: bool = True,
    ) -> np.ndarray:
        """Make predictions.
        
        Args:
            model_name: Name of model
            X: Feature matrix
            scale_features: Whether to scale features
            
        Returns:
            Predictions
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found")
        
        model = self.models[model_name]
        
        # Scale features if requested
        if scale_features:
            if self.scaler is None:
                raise ValueError("Scaler not fitted")
            X_scaled = self.transform_features(X)
        else:
            X_scaled = X.values
        
        return model.predict(X_scaled)
    
    def predict_proba(
        self,
        model_name: str,
        X: pd.DataFrame,
        scale_features: bool = True,
    ) -> np.ndarray:
        """Predict class probabilities.
        
        Args:
            model_name: Name of classifier
            X: Feature matrix
            scale_features: Whether to scale features
            
        Returns:
            Class probabilities
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found")
        
        model = self.models[model_name]
        
        if not hasattr(model, 'predict_proba'):
            raise ValueError(f"Model '{model_name}' does not support probability predictions")
        
        # Scale features if requested
        if scale_features:
            if self.scaler is None:
                raise ValueError("Scaler not fitted")
            X_scaled = self.transform_features(X)
        else:
            X_scaled = X.values
        
        return model.predict_proba(X_scaled)
    
    def get_feature_importance(
        self,
        model_name: str,
        top_n: Optional[int] = None,
    ) -> pd.DataFrame:
        """Get feature importance from tree-based model.
        
        Args:
            model_name: Name of model
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
