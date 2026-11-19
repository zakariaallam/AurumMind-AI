"""Partial dependence analysis for model interpretation.

This module provides partial dependence plots (PDP) and
individual conditional expectation (ICE) plots.
"""

import logging
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
from sklearn.inspection import partial_dependence

logger = logging.getLogger(__name__)


class PartialDependenceAnalyzer:
    """Analyze partial dependence and ICE plots.
    
    Partial Dependence: Shows the marginal effect of features on predictions.
    ICE (Individual Conditional Expectation): Shows effect for individual samples.
    """
    
    def __init__(
        self,
        model: Any,
        feature_names: list[str],
    ):
        """Initialize partial dependence analyzer.
        
        Args:
            model: Trained model
            feature_names: Names of features
        """
        self.model = model
        self.feature_names = feature_names
    
    def calculate_partial_dependence(
        self,
        X: pd.DataFrame,
        features: Union[list[int], list[str]],
        grid_resolution: int = 50,
        percentiles: tuple[float, float] = (0.05, 0.95),
    ) -> dict[str, Any]:
        """Calculate partial dependence for features.
        
        Args:
            X: Feature matrix
            features: Features to analyze (indices or names)
            grid_resolution: Number of points in the grid
            percentiles: Range of feature values to consider
            
        Returns:
            Dictionary with PD results
        """
        logger.info(f"Computing partial dependence for {len(features)} features")
        
        # Convert feature names to indices if needed
        if isinstance(features[0], str):
            feature_indices = [self.feature_names.index(f) for f in features]
        else:
            feature_indices = features
        
        # Compute partial dependence
        pd_result = partial_dependence(
            self.model,
            X,
            features=feature_indices,
            grid_resolution=grid_resolution,
            percentiles=percentiles,
        )
        
        results = {
            'average': pd_result['average'],
            'grid_values': pd_result['grid_values'],
            'feature_names': [self.feature_names[i] for i in feature_indices],
            'feature_indices': feature_indices,
        }
        
        logger.info("✓ Partial dependence computed")
        
        return results
    
    def calculate_ice(
        self,
        X: pd.DataFrame,
        feature: Union[int, str],
        sample_indices: Optional[np.ndarray] = None,
        grid_resolution: int = 50,
    ) -> dict[str, Any]:
        """Calculate Individual Conditional Expectation curves.
        
        Args:
            X: Feature matrix
            feature: Feature to analyze (index or name)
            sample_indices: Indices of samples to include (None = all)
            grid_resolution: Number of points in the grid
            
        Returns:
            Dictionary with ICE results
        """
        # Convert feature name to index if needed
        if isinstance(feature, str):
            feature_idx = self.feature_names.index(feature)
            feature_name = feature
        else:
            feature_idx = feature
            feature_name = self.feature_names[feature_idx]
        
        logger.info(f"Computing ICE curves for feature: {feature_name}")
        
        # Select samples
        if sample_indices is None:
            sample_indices = np.arange(len(X))
        
        X_subset = X.iloc[sample_indices].copy()
        
        # Get feature range
        feature_col = X.columns[feature_idx]
        feature_min = X[feature_col].quantile(0.05)
        feature_max = X[feature_col].quantile(0.95)
        
        # Create grid
        grid_values = np.linspace(feature_min, feature_max, grid_resolution)
        
        # Calculate ICE curves
        ice_curves = []
        
        for sample_idx in range(len(X_subset)):
            curve = []
            
            for grid_val in grid_values:
                # Create modified sample
                X_modified = X_subset.iloc[[sample_idx]].copy()
                X_modified.iloc[0, feature_idx] = grid_val
                
                # Predict
                pred = self.model.predict(X_modified)[0]
                curve.append(pred)
            
            ice_curves.append(curve)
        
        results = {
            'ice_curves': np.array(ice_curves),
            'grid_values': grid_values,
            'feature_name': feature_name,
            'feature_index': feature_idx,
            'sample_indices': sample_indices,
        }
        
        logger.info(f"✓ ICE curves computed for {len(ice_curves)} samples")
        
        return results
    
    def plot_partial_dependence(
        self,
        X: pd.DataFrame,
        features: Union[list[int], list[str]],
        n_cols: int = 3,
        show: bool = True,
        save_path: Optional[Path] = None,
    ) -> None:
        """Plot partial dependence for multiple features.
        
        Args:
            X: Feature matrix
            features: Features to plot
            n_cols: Number of columns in plot grid
            show: Whether to show plot
            save_path: Path to save plot
        """
        try:
            from sklearn.inspection import PartialDependenceDisplay
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("sklearn.inspection or matplotlib not available")
            return
        
        logger.info(f"Creating partial dependence plots for {len(features)} features")
        
        # Convert feature names to indices if needed
        if isinstance(features[0], str):
            feature_indices = [self.feature_names.index(f) for f in features]
        else:
            feature_indices = features
        
        # Create plot
        fig, ax = plt.subplots(figsize=(5 * n_cols, 4 * ((len(features) + n_cols - 1) // n_cols)))
        
        display = PartialDependenceDisplay.from_estimator(
            self.model,
            X,
            features=feature_indices,
            feature_names=self.feature_names,
            n_cols=n_cols,
            grid_resolution=50,
            ax=ax if len(features) > 1 else [ax],
        )
        
        fig.suptitle('Partial Dependence Plots', fontsize=16, y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            logger.info(f"✓ Plot saved to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_ice(
        self,
        X: pd.DataFrame,
        feature: Union[int, str],
        sample_indices: Optional[np.ndarray] = None,
        centered: bool = False,
        show_pd: bool = True,
        show: bool = True,
        save_path: Optional[Path] = None,
    ) -> None:
        """Plot ICE curves for a feature.
        
        Args:
            X: Feature matrix
            feature: Feature to plot
            sample_indices: Sample indices (None = random subset)
            centered: Center ICE curves at first grid point
            show_pd: Show average partial dependence
            show: Whether to show plot
            save_path: Path to save plot
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available for plotting")
            return
        
        # Select random subset if not specified
        if sample_indices is None:
            n_samples = min(50, len(X))
            sample_indices = np.random.choice(len(X), n_samples, replace=False)
        
        # Calculate ICE
        ice_results = self.calculate_ice(X, feature, sample_indices)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ice_curves = ice_results['ice_curves']
        grid_values = ice_results['grid_values']
        
        # Center if requested
        if centered:
            ice_curves = ice_curves - ice_curves[:, 0:1]
        
        # Plot ICE curves
        for curve in ice_curves:
            ax.plot(grid_values, curve, alpha=0.3, color='steelblue', linewidth=0.5)
        
        # Plot average (PD)
        if show_pd:
            pd_avg = ice_curves.mean(axis=0)
            ax.plot(grid_values, pd_avg, color='orange', linewidth=3, label='Average (PD)')
            ax.legend()
        
        ax.set_xlabel(ice_results['feature_name'], fontsize=12)
        ax.set_ylabel('Prediction' + (' (Centered)' if centered else ''), fontsize=12)
        ax.set_title(
            f"ICE Plot for {ice_results['feature_name']}\n"
            f"({len(ice_curves)} samples)",
            fontsize=14
        )
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            logger.info(f"✓ Plot saved to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_2d_partial_dependence(
        self,
        X: pd.DataFrame,
        features: tuple[Union[int, str], Union[int, str]],
        grid_resolution: int = 30,
        show: bool = True,
        save_path: Optional[Path] = None,
    ) -> None:
        """Plot 2D partial dependence (interaction plot).
        
        Args:
            X: Feature matrix
            features: Pair of features to plot
            grid_resolution: Number of grid points
            show: Whether to show plot
            save_path: Path to save plot
        """
        try:
            from sklearn.inspection import PartialDependenceDisplay
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("sklearn.inspection or matplotlib not available")
            return
        
        logger.info(f"Creating 2D partial dependence plot for {features}")
        
        # Convert feature names to indices if needed
        if isinstance(features[0], str):
            feature_indices = [self.feature_names.index(f) for f in features]
        else:
            feature_indices = list(features)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        display = PartialDependenceDisplay.from_estimator(
            self.model,
            X,
            features=[feature_indices],
            feature_names=self.feature_names,
            grid_resolution=grid_resolution,
            ax=ax,
            kind='average',
        )
        
        feature_names = [self.feature_names[i] for i in feature_indices]
        fig.suptitle(
            f'2D Partial Dependence: {feature_names[0]} vs {feature_names[1]}',
            fontsize=14
        )
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            logger.info(f"✓ Plot saved to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def analyze_feature_interactions(
        self,
        X: pd.DataFrame,
        feature_pairs: list[tuple[Union[int, str], Union[int, str]]],
        grid_resolution: int = 20,
    ) -> pd.DataFrame:
        """Analyze feature interactions using 2D partial dependence.
        
        Args:
            X: Feature matrix
            feature_pairs: List of feature pairs to analyze
            grid_resolution: Number of grid points
            
        Returns:
            DataFrame with interaction strength metrics
        """
        logger.info(f"Analyzing {len(feature_pairs)} feature interactions")
        
        interactions = []
        
        for feat1, feat2 in feature_pairs:
            # Convert names to indices
            if isinstance(feat1, str):
                idx1 = self.feature_names.index(feat1)
                name1 = feat1
            else:
                idx1 = feat1
                name1 = self.feature_names[idx1]
            
            if isinstance(feat2, str):
                idx2 = self.feature_names.index(feat2)
                name2 = feat2
            else:
                idx2 = feat2
                name2 = self.feature_names[idx2]
            
            # Compute 2D partial dependence
            pd_result = partial_dependence(
                self.model,
                X,
                features=[(idx1, idx2)],
                grid_resolution=grid_resolution,
            )
            
            # Measure interaction strength (variance of 2D surface)
            pd_values = pd_result['average'][0]
            interaction_strength = pd_values.std()
            
            interactions.append({
                'feature1': name1,
                'feature2': name2,
                'interaction_strength': interaction_strength,
                'min_effect': pd_values.min(),
                'max_effect': pd_values.max(),
                'range': pd_values.max() - pd_values.min(),
            })
        
        df = pd.DataFrame(interactions)
        df = df.sort_values('interaction_strength', ascending=False)
        
        logger.info("✓ Feature interactions analyzed")
        
        return df
    
    def get_feature_effect_summary(
        self,
        X: pd.DataFrame,
        features: Optional[list[Union[int, str]]] = None,
    ) -> pd.DataFrame:
        """Get summary statistics of partial dependence effects.
        
        Args:
            X: Feature matrix
            features: Features to analyze (None = all)
            
        Returns:
            DataFrame with effect summaries
        """
        if features is None:
            features = list(range(len(self.feature_names)))
        
        logger.info(f"Computing effect summaries for {len(features)} features")
        
        summaries = []
        
        for feature in features:
            # Calculate partial dependence
            pd_result = self.calculate_partial_dependence(X, [feature])
            
            pd_values = pd_result['average'][0]
            feature_name = pd_result['feature_names'][0]
            
            summaries.append({
                'feature': feature_name,
                'mean_effect': pd_values.mean(),
                'std_effect': pd_values.std(),
                'min_effect': pd_values.min(),
                'max_effect': pd_values.max(),
                'range': pd_values.max() - pd_values.min(),
            })
        
        df = pd.DataFrame(summaries)
        df = df.sort_values('range', ascending=False)
        
        logger.info("✓ Effect summaries computed")
        
        return df
