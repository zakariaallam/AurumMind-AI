"""SHAP-based model explainability.

This module provides SHAP (SHapley Additive exPlanations) analysis
for machine learning models.
"""

import logging
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class SHAPExplainer:
    """SHAP-based model explainer.
    
    Provides global and local explanations using SHAP values.
    Supports tree-based models, linear models, and deep learning models.
    """
    
    def __init__(
        self,
        model: Any,
        feature_names: Optional[list[str]] = None,
        model_type: str = "tree",
    ):
        """Initialize SHAP explainer.
        
        Args:
            model: Trained model to explain
            feature_names: Names of features
            model_type: Type of model ('tree', 'linear', 'kernel', 'deep')
        """
        self.model = model
        self.feature_names = feature_names
        self.model_type = model_type
        
        self.explainer: Optional[Any] = None
        self.shap_values: Optional[np.ndarray] = None
        self.expected_value: Optional[float] = None
        
        # Check if shap is available
        try:
            import shap
            self.shap = shap
            self._shap_available = True
        except ImportError:
            logger.warning("SHAP not installed. Install with: pip install shap")
            self._shap_available = False
    
    def create_explainer(
        self,
        X_background: Optional[pd.DataFrame] = None,
        **kwargs,
    ) -> None:
        """Create SHAP explainer.
        
        Args:
            X_background: Background dataset for kernel/deep explainers
            **kwargs: Additional arguments for explainer
        """
        if not self._shap_available:
            raise ImportError("SHAP not installed")
        
        logger.info(f"Creating {self.model_type} SHAP explainer")
        
        if self.model_type == "tree":
            # Tree explainer for tree-based models
            self.explainer = self.shap.TreeExplainer(self.model, **kwargs)
            
        elif self.model_type == "linear":
            # Linear explainer for linear models
            if X_background is None:
                raise ValueError("X_background required for linear explainer")
            self.explainer = self.shap.LinearExplainer(
                self.model, X_background, **kwargs
            )
            
        elif self.model_type == "kernel":
            # Kernel explainer (model-agnostic)
            if X_background is None:
                raise ValueError("X_background required for kernel explainer")
            
            # Use subset of background data for efficiency
            background = self.shap.sample(X_background, min(100, len(X_background)))
            self.explainer = self.shap.KernelExplainer(
                self.model.predict_proba if hasattr(self.model, 'predict_proba') else self.model.predict,
                background,
                **kwargs
            )
            
        elif self.model_type == "deep":
            # Deep explainer for neural networks
            if X_background is None:
                raise ValueError("X_background required for deep explainer")
            self.explainer = self.shap.DeepExplainer(
                self.model, X_background, **kwargs
            )
            
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        logger.info("✓ SHAP explainer created")
    
    def explain(
        self,
        X: pd.DataFrame,
        check_additivity: bool = False,
    ) -> np.ndarray:
        """Calculate SHAP values for dataset.
        
        Args:
            X: Features to explain
            check_additivity: Check SHAP value additivity
            
        Returns:
            SHAP values array
        """
        if self.explainer is None:
            raise ValueError("Explainer not created. Call create_explainer() first")
        
        logger.info(f"Computing SHAP values for {len(X)} samples")
        
        # Compute SHAP values
        if self.model_type == "tree":
            shap_values = self.explainer.shap_values(
                X, check_additivity=check_additivity
            )
        else:
            shap_values = self.explainer.shap_values(X)
        
        # Handle multi-output models (e.g., multi-class classification)
        if isinstance(shap_values, list):
            # For binary classification, use positive class
            if len(shap_values) == 2:
                shap_values = shap_values[1]
            else:
                # For multi-class, keep all
                logger.info(f"Multi-class output: {len(shap_values)} classes")
        
        self.shap_values = shap_values
        
        # Get expected value (base value)
        if hasattr(self.explainer, 'expected_value'):
            expected = self.explainer.expected_value
            if isinstance(expected, (list, np.ndarray)):
                self.expected_value = expected[1] if len(expected) == 2 else expected[0]
            else:
                self.expected_value = expected
        
        logger.info("✓ SHAP values computed")
        
        return shap_values
    
    def get_global_importance(
        self,
        shap_values: Optional[np.ndarray] = None,
        top_n: int = 20,
    ) -> pd.DataFrame:
        """Get global feature importance from SHAP values.
        
        Args:
            shap_values: SHAP values (use cached if None)
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature importance
        """
        if shap_values is None:
            shap_values = self.shap_values
        
        if shap_values is None:
            raise ValueError("No SHAP values available")
        
        # Calculate mean absolute SHAP value per feature
        importance = np.abs(shap_values).mean(axis=0)
        
        # Create DataFrame
        df = pd.DataFrame({
            'feature': self.feature_names if self.feature_names else [f"feature_{i}" for i in range(len(importance))],
            'importance': importance,
        })
        
        df = df.sort_values('importance', ascending=False)
        df['rank'] = range(1, len(df) + 1)
        df['importance_pct'] = (df['importance'] / df['importance'].sum()) * 100
        df['cumulative_pct'] = df['importance_pct'].cumsum()
        
        return df.head(top_n)
    
    def plot_summary(
        self,
        X: pd.DataFrame,
        shap_values: Optional[np.ndarray] = None,
        plot_type: str = "dot",
        max_display: int = 20,
        show: bool = True,
        save_path: Optional[Path] = None,
    ) -> None:
        """Create SHAP summary plot.
        
        Args:
            X: Feature data
            shap_values: SHAP values (use cached if None)
            plot_type: Type of plot ('dot', 'bar', 'violin')
            max_display: Maximum features to display
            show: Whether to show plot
            save_path: Path to save plot
        """
        if not self._shap_available:
            raise ImportError("SHAP not installed")
        
        if shap_values is None:
            shap_values = self.shap_values
        
        if shap_values is None:
            raise ValueError("No SHAP values available")
        
        logger.info(f"Creating SHAP summary plot ({plot_type})")
        
        self.shap.summary_plot(
            shap_values,
            X,
            feature_names=self.feature_names,
            plot_type=plot_type,
            max_display=max_display,
            show=show,
        )
        
        if save_path:
            import matplotlib.pyplot as plt
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            logger.info(f"✓ Plot saved to {save_path}")
    
    def plot_waterfall(
        self,
        sample_idx: int,
        X: pd.DataFrame,
        shap_values: Optional[np.ndarray] = None,
        show: bool = True,
        save_path: Optional[Path] = None,
    ) -> None:
        """Create SHAP waterfall plot for single prediction.
        
        Args:
            sample_idx: Index of sample to explain
            X: Feature data
            shap_values: SHAP values (use cached if None)
            show: Whether to show plot
            save_path: Path to save plot
        """
        if not self._shap_available:
            raise ImportError("SHAP not installed")
        
        if shap_values is None:
            shap_values = self.shap_values
        
        if shap_values is None:
            raise ValueError("No SHAP values available")
        
        logger.info(f"Creating waterfall plot for sample {sample_idx}")
        
        # Create explanation object
        explanation = self.shap.Explanation(
            values=shap_values[sample_idx],
            base_values=self.expected_value,
            data=X.iloc[sample_idx].values,
            feature_names=self.feature_names,
        )
        
        self.shap.waterfall_plot(explanation, show=show)
        
        if save_path:
            import matplotlib.pyplot as plt
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            logger.info(f"✓ Plot saved to {save_path}")
    
    def plot_force(
        self,
        sample_idx: int,
        X: pd.DataFrame,
        shap_values: Optional[np.ndarray] = None,
        matplotlib: bool = True,
        show: bool = True,
        save_path: Optional[Path] = None,
    ) -> None:
        """Create SHAP force plot for single prediction.
        
        Args:
            sample_idx: Index of sample to explain
            X: Feature data
            shap_values: SHAP values (use cached if None)
            matplotlib: Use matplotlib backend
            show: Whether to show plot
            save_path: Path to save plot
        """
        if not self._shap_available:
            raise ImportError("SHAP not installed")
        
        if shap_values is None:
            shap_values = self.shap_values
        
        if shap_values is None:
            raise ValueError("No SHAP values available")
        
        logger.info(f"Creating force plot for sample {sample_idx}")
        
        plot = self.shap.force_plot(
            self.expected_value,
            shap_values[sample_idx],
            X.iloc[sample_idx],
            feature_names=self.feature_names,
            matplotlib=matplotlib,
            show=show,
        )
        
        if save_path and matplotlib:
            import matplotlib.pyplot as plt
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            logger.info(f"✓ Plot saved to {save_path}")
        
        return plot
    
    def plot_dependence(
        self,
        feature_idx: Union[int, str],
        X: pd.DataFrame,
        shap_values: Optional[np.ndarray] = None,
        interaction_index: Optional[Union[int, str]] = "auto",
        show: bool = True,
        save_path: Optional[Path] = None,
    ) -> None:
        """Create SHAP dependence plot.
        
        Args:
            feature_idx: Feature index or name
            X: Feature data
            shap_values: SHAP values (use cached if None)
            interaction_index: Feature to color by ('auto' for automatic selection)
            show: Whether to show plot
            save_path: Path to save plot
        """
        if not self._shap_available:
            raise ImportError("SHAP not installed")
        
        if shap_values is None:
            shap_values = self.shap_values
        
        if shap_values is None:
            raise ValueError("No SHAP values available")
        
        logger.info(f"Creating dependence plot for feature {feature_idx}")
        
        self.shap.dependence_plot(
            feature_idx,
            shap_values,
            X,
            feature_names=self.feature_names,
            interaction_index=interaction_index,
            show=show,
        )
        
        if save_path:
            import matplotlib.pyplot as plt
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            logger.info(f"✓ Plot saved to {save_path}")
    
    def get_top_features_for_sample(
        self,
        sample_idx: int,
        shap_values: Optional[np.ndarray] = None,
        top_n: int = 10,
    ) -> pd.DataFrame:
        """Get top contributing features for a single sample.
        
        Args:
            sample_idx: Index of sample
            shap_values: SHAP values (use cached if None)
            top_n: Number of top features
            
        Returns:
            DataFrame with top features and their contributions
        """
        if shap_values is None:
            shap_values = self.shap_values
        
        if shap_values is None:
            raise ValueError("No SHAP values available")
        
        sample_shap = shap_values[sample_idx]
        
        # Get absolute values for ranking
        abs_shap = np.abs(sample_shap)
        top_indices = np.argsort(abs_shap)[::-1][:top_n]
        
        df = pd.DataFrame({
            'feature': [self.feature_names[i] if self.feature_names else f"feature_{i}" for i in top_indices],
            'shap_value': sample_shap[top_indices],
            'abs_shap_value': abs_shap[top_indices],
            'rank': range(1, top_n + 1),
        })
        
        return df
    
    def save_shap_values(
        self,
        output_path: Union[str, Path],
    ) -> None:
        """Save SHAP values to file.
        
        Args:
            output_path: Path to save SHAP values
        """
        if self.shap_values is None:
            raise ValueError("No SHAP values to save")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save as numpy array
        np.save(output_path, self.shap_values)
        
        logger.info(f"✓ SHAP values saved to {output_path}")
    
    def load_shap_values(
        self,
        input_path: Union[str, Path],
    ) -> np.ndarray:
        """Load SHAP values from file.
        
        Args:
            input_path: Path to SHAP values file
            
        Returns:
            Loaded SHAP values
        """
        input_path = Path(input_path)
        
        if not input_path.exists():
            raise FileNotFoundError(f"SHAP values file not found: {input_path}")
        
        self.shap_values = np.load(input_path)
        
        logger.info(f"✓ SHAP values loaded from {input_path}")
        
        return self.shap_values
