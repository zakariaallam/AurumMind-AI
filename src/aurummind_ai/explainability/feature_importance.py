"""Feature importance analysis.

This module provides comprehensive feature importance analysis
using multiple methods: built-in, permutation, and drop-column.
"""

import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

logger = logging.getLogger(__name__)


class FeatureImportanceAnalyzer:
    """Analyze feature importance using multiple methods.
    
    Supports:
    - Built-in importance (for tree models)
    - Permutation importance (model-agnostic)
    - Drop-column importance
    - Correlation analysis
    """
    
    def __init__(
        self,
        model: Any,
        feature_names: list[str],
    ):
        """Initialize feature importance analyzer.
        
        Args:
            model: Trained model
            feature_names: Names of features
        """
        self.model = model
        self.feature_names = feature_names
        
        self.built_in_importance: Optional[pd.DataFrame] = None
        self.permutation_importance: Optional[pd.DataFrame] = None
        self.drop_column_importance: Optional[pd.DataFrame] = None
    
    def get_built_in_importance(
        self,
        top_n: int = 20,
    ) -> pd.DataFrame:
        """Get built-in feature importance from model.
        
        Works for tree-based models and models with feature_importances_ attribute.
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature importance
        """
        if not hasattr(self.model, 'feature_importances_'):
            logger.warning("Model does not have feature_importances_ attribute")
            return pd.DataFrame()
        
        logger.info("Extracting built-in feature importance")
        
        importance = self.model.feature_importances_
        
        df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance,
        })
        
        df = df.sort_values('importance', ascending=False)
        df['rank'] = range(1, len(df) + 1)
        df['importance_pct'] = (df['importance'] / df['importance'].sum()) * 100
        df['cumulative_pct'] = df['importance_pct'].cumsum()
        
        self.built_in_importance = df
        
        logger.info(f"✓ Extracted importance for {len(df)} features")
        
        return df.head(top_n)
    
    def get_permutation_importance(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        n_repeats: int = 10,
        random_state: int = 42,
        scoring: Optional[str] = None,
        top_n: int = 20,
    ) -> pd.DataFrame:
        """Calculate permutation importance.
        
        Model-agnostic method that measures importance by randomly
        shuffling each feature and observing performance degradation.
        
        Args:
            X: Feature matrix
            y: Target vector
            n_repeats: Number of times to permute each feature
            random_state: Random seed
            scoring: Scoring function (None = use model's default)
            top_n: Number of top features to return
            
        Returns:
            DataFrame with permutation importance
        """
        logger.info(f"Computing permutation importance (n_repeats={n_repeats})")
        
        result = permutation_importance(
            self.model,
            X,
            y,
            n_repeats=n_repeats,
            random_state=random_state,
            scoring=scoring,
        )
        
        df = pd.DataFrame({
            'feature': self.feature_names,
            'importance_mean': result.importances_mean,
            'importance_std': result.importances_std,
        })
        
        df = df.sort_values('importance_mean', ascending=False)
        df['rank'] = range(1, len(df) + 1)
        
        # Normalize to percentages
        total = df['importance_mean'].sum()
        if total > 0:
            df['importance_pct'] = (df['importance_mean'] / total) * 100
            df['cumulative_pct'] = df['importance_pct'].cumsum()
        
        self.permutation_importance = df
        
        logger.info(f"✓ Permutation importance computed for {len(df)} features")
        
        return df.head(top_n)
    
    def get_drop_column_importance(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        eval_metric: callable,
        top_n: int = 20,
    ) -> pd.DataFrame:
        """Calculate drop-column importance.
        
        Measures importance by evaluating model performance when
        each feature is removed.
        
        Args:
            X: Feature matrix
            y: Target vector
            eval_metric: Function to compute metric (y_true, y_pred) -> score
            top_n: Number of top features to return
            
        Returns:
            DataFrame with drop-column importance
        """
        logger.info("Computing drop-column importance")
        
        # Baseline score with all features
        y_pred_full = self.model.predict(X)
        baseline_score = eval_metric(y, y_pred_full)
        
        logger.info(f"Baseline score: {baseline_score:.4f}")
        
        importance_list = []
        
        for i, feature in enumerate(self.feature_names):
            # Drop one feature
            X_dropped = X.drop(columns=[feature])
            
            # Predict without this feature
            try:
                y_pred = self.model.predict(X_dropped)
                score = eval_metric(y, y_pred)
                
                # Importance = drop in performance
                importance = baseline_score - score
                
                importance_list.append({
                    'feature': feature,
                    'importance': importance,
                    'score_without': score,
                })
                
            except Exception as e:
                logger.warning(f"Error evaluating without {feature}: {e}")
                importance_list.append({
                    'feature': feature,
                    'importance': 0.0,
                    'score_without': baseline_score,
                })
        
        df = pd.DataFrame(importance_list)
        df = df.sort_values('importance', ascending=False)
        df['rank'] = range(1, len(df) + 1)
        df['baseline_score'] = baseline_score
        
        self.drop_column_importance = df
        
        logger.info(f"✓ Drop-column importance computed for {len(df)} features")
        
        return df.head(top_n)
    
    def compare_methods(
        self,
        top_n: int = 20,
    ) -> pd.DataFrame:
        """Compare feature importance across methods.
        
        Args:
            top_n: Number of top features to compare
            
        Returns:
            DataFrame comparing different importance methods
        """
        comparisons = []
        
        # Built-in importance
        if self.built_in_importance is not None:
            for _, row in self.built_in_importance.head(top_n).iterrows():
                comparisons.append({
                    'feature': row['feature'],
                    'method': 'built_in',
                    'importance': row['importance'],
                    'rank': row['rank'],
                })
        
        # Permutation importance
        if self.permutation_importance is not None:
            for _, row in self.permutation_importance.head(top_n).iterrows():
                comparisons.append({
                    'feature': row['feature'],
                    'method': 'permutation',
                    'importance': row['importance_mean'],
                    'rank': row['rank'],
                })
        
        # Drop-column importance
        if self.drop_column_importance is not None:
            for _, row in self.drop_column_importance.head(top_n).iterrows():
                comparisons.append({
                    'feature': row['feature'],
                    'method': 'drop_column',
                    'importance': row['importance'],
                    'rank': row['rank'],
                })
        
        df = pd.DataFrame(comparisons)
        
        # Pivot to wide format
        if not df.empty:
            pivot = df.pivot(index='feature', columns='method', values='importance')
            pivot['avg_importance'] = pivot.mean(axis=1)
            pivot = pivot.sort_values('avg_importance', ascending=False)
            
            return pivot.head(top_n)
        
        return pd.DataFrame()
    
    def analyze_feature_correlations(
        self,
        X: pd.DataFrame,
        threshold: float = 0.8,
    ) -> pd.DataFrame:
        """Analyze feature correlations to identify redundant features.
        
        Args:
            X: Feature matrix
            threshold: Correlation threshold for identifying pairs
            
        Returns:
            DataFrame with highly correlated feature pairs
        """
        logger.info(f"Analyzing feature correlations (threshold={threshold})")
        
        # Calculate correlation matrix
        corr_matrix = X.corr().abs()
        
        # Get upper triangle (avoid duplicates)
        upper_tri = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        # Find highly correlated pairs
        correlated_pairs = []
        
        for column in upper_tri.columns:
            high_corr = upper_tri[column][upper_tri[column] > threshold]
            
            for index, corr_value in high_corr.items():
                correlated_pairs.append({
                    'feature1': column,
                    'feature2': index,
                    'correlation': corr_value,
                })
        
        df = pd.DataFrame(correlated_pairs)
        
        if not df.empty:
            df = df.sort_values('correlation', ascending=False)
            logger.info(f"Found {len(df)} highly correlated pairs")
        else:
            logger.info("No highly correlated pairs found")
        
        return df
    
    def plot_importance_comparison(
        self,
        top_n: int = 20,
        show: bool = True,
        save_path: Optional[Path] = None,
    ) -> None:
        """Plot comparison of importance methods.
        
        Args:
            top_n: Number of top features to plot
            show: Whether to show plot
            save_path: Path to save plot
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            logger.warning("matplotlib/seaborn not available for plotting")
            return
        
        comparison = self.compare_methods(top_n=top_n)
        
        if comparison.empty:
            logger.warning("No importance data available for comparison")
            return
        
        fig, ax = plt.subplots(figsize=(12, max(6, len(comparison) * 0.3)))
        
        # Normalize importances for comparison
        comparison_norm = comparison.div(comparison.max(axis=0), axis=1)
        
        # Plot heatmap
        sns.heatmap(
            comparison_norm,
            annot=True,
            fmt='.2f',
            cmap='YlOrRd',
            ax=ax,
            cbar_kws={'label': 'Normalized Importance'},
        )
        
        ax.set_title('Feature Importance Comparison Across Methods', fontsize=14, pad=20)
        ax.set_xlabel('Method', fontsize=12)
        ax.set_ylabel('Feature', fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            logger.info(f"✓ Plot saved to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_cumulative_importance(
        self,
        method: str = 'built_in',
        show: bool = True,
        save_path: Optional[Path] = None,
    ) -> None:
        """Plot cumulative feature importance.
        
        Args:
            method: Which importance method to use
            show: Whether to show plot
            save_path: Path to save plot
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available for plotting")
            return
        
        # Get importance data
        if method == 'built_in' and self.built_in_importance is not None:
            df = self.built_in_importance
        elif method == 'permutation' and self.permutation_importance is not None:
            df = self.permutation_importance
        elif method == 'drop_column' and self.drop_column_importance is not None:
            df = self.drop_column_importance
        else:
            logger.warning(f"No {method} importance data available")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Individual importance
        ax1.bar(range(len(df)), df['importance'].values)
        ax1.set_xlabel('Feature Rank')
        ax1.set_ylabel('Importance')
        ax1.set_title(f'{method.replace("_", " ").title()} Feature Importance')
        ax1.grid(alpha=0.3)
        
        # Cumulative importance
        if 'cumulative_pct' in df.columns:
            ax2.plot(range(len(df)), df['cumulative_pct'].values, marker='o')
            ax2.axhline(y=80, color='r', linestyle='--', label='80% threshold')
            ax2.axhline(y=90, color='orange', linestyle='--', label='90% threshold')
            ax2.set_xlabel('Number of Features')
            ax2.set_ylabel('Cumulative Importance (%)')
            ax2.set_title('Cumulative Feature Importance')
            ax2.legend()
            ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            logger.info(f"✓ Plot saved to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def get_feature_selection_recommendation(
        self,
        cumulative_threshold: float = 0.95,
    ) -> dict[str, Any]:
        """Recommend features to keep based on cumulative importance.
        
        Args:
            cumulative_threshold: Cumulative importance threshold (0-1)
            
        Returns:
            Dictionary with feature selection recommendations
        """
        recommendations = {}
        
        for method_name, df in [
            ('built_in', self.built_in_importance),
            ('permutation', self.permutation_importance),
            ('drop_column', self.drop_column_importance),
        ]:
            if df is None or 'cumulative_pct' not in df.columns:
                continue
            
            # Find features that reach threshold
            threshold_pct = cumulative_threshold * 100
            selected = df[df['cumulative_pct'] <= threshold_pct]
            
            if len(selected) == 0:
                selected = df.head(1)
            
            recommendations[method_name] = {
                'num_features': len(selected),
                'features': selected['feature'].tolist(),
                'cumulative_importance': selected['cumulative_pct'].iloc[-1],
            }
        
        return recommendations
    
    def print_summary(
        self,
        top_n: int = 10,
    ) -> None:
        """Print feature importance summary.
        
        Args:
            top_n: Number of top features to display
        """
        print("=" * 70)
        print("Feature Importance Analysis Summary")
        print("=" * 70)
        print()
        
        if self.built_in_importance is not None:
            print("Built-in Importance (Top {}):".format(top_n))
            print(self.built_in_importance.head(top_n)[['feature', 'importance', 'rank', 'cumulative_pct']].to_string(index=False))
            print()
        
        if self.permutation_importance is not None:
            print("Permutation Importance (Top {}):".format(top_n))
            print(self.permutation_importance.head(top_n)[['feature', 'importance_mean', 'rank']].to_string(index=False))
            print()
        
        if self.drop_column_importance is not None:
            print("Drop-Column Importance (Top {}):".format(top_n))
            print(self.drop_column_importance.head(top_n)[['feature', 'importance', 'rank']].to_string(index=False))
            print()
        
        print("=" * 70)
