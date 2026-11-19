"""Model explainability demonstration script.

This script demonstrates Phase 6 explainability capabilities:
- SHAP values and visualizations
- Feature importance analysis
- Partial dependence plots
- Individual prediction explanations

Usage:
    python scripts/explain_model.py --timeframe 4h --model xgboost
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from aurummind_ai.explainability import (
    SHAPExplainer,
    FeatureImportanceAnalyzer,
    PartialDependenceAnalyzer,
    IndividualExplainer,
)
from aurummind_ai.models import BaselineModels, BoostingModels, ModelTrainer
from aurummind_ai.evaluation import ModelEvaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_feature_data(timeframe: str) -> pd.DataFrame:
    """Load feature-engineered data.
    
    Args:
        timeframe: Timeframe to load (1h/4h/1d)
        
    Returns:
        DataFrame with features
    """
    data_path = project_root / f"data/processed/{timeframe}/xauusd_features.csv"
    
    if not data_path.exists():
        raise FileNotFoundError(f"Feature data not found: {data_path}")
    
    logger.info(f"Loading feature data from {data_path}")
    df = pd.read_csv(data_path)
    
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    logger.info(f"Loaded {len(df)} samples with {len(df.columns)} columns")
    
    return df


def train_model(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Any:
    """Train a model.
    
    Args:
        model_name: Name of model
        X_train: Training features
        y_train: Training target
        
    Returns:
        Trained model
    """
    logger.info(f"Training {model_name} model...")
    
    # Initialize models
    baseline = BaselineModels()
    boosting = BoostingModels()
    
    # Select model
    if model_name == 'logistic':
        model = baseline.create_logistic_regression()
    elif model_name == 'random_forest':
        model = baseline.create_random_forest_classifier()
    elif model_name == 'xgboost':
        model = boosting.create_xgboost_classifier()
    elif model_name == 'lightgbm':
        model = boosting.create_lightgbm_classifier()
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Train
    model.fit(X_train, y_train)
    
    logger.info("✓ Model trained")
    
    return model


def demonstrate_shap(
    model: Any,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    feature_names: list[str],
    output_dir: Path,
) -> None:
    """Demonstrate SHAP explanations.
    
    Args:
        model: Trained model
        X_train: Training features (for background)
        X_test: Test features
        feature_names: Feature names
        output_dir: Output directory for plots
    """
    logger.info(f"\n{'='*70}")
    logger.info("SHAP Analysis")
    logger.info(f"{'='*70}\n")
    
    try:
        # Create SHAP explainer
        explainer = SHAPExplainer(model, feature_names, model_type='tree')
        
        # Use subset for efficiency
        background = X_train.sample(min(100, len(X_train)), random_state=42)
        test_subset = X_test.sample(min(100, len(X_test)), random_state=42)
        
        logger.info("Creating SHAP explainer...")
        explainer.create_explainer(background)
        
        logger.info("Computing SHAP values...")
        shap_values = explainer.explain(test_subset, check_additivity=False)
        
        # Global importance
        logger.info("\nGlobal Feature Importance (SHAP):")
        importance = explainer.get_global_importance(top_n=15)
        print(importance[['feature', 'importance', 'importance_pct', 'cumulative_pct']].to_string(index=False))
        
        # Save plots if output directory provided
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            
            logger.info("\nCreating SHAP visualizations...")
            
            # Summary plot
            explainer.plot_summary(
                test_subset,
                plot_type='dot',
                max_display=20,
                show=False,
                save_path=output_dir / 'shap_summary.png'
            )
            
            # Bar plot
            explainer.plot_summary(
                test_subset,
                plot_type='bar',
                max_display=20,
                show=False,
                save_path=output_dir / 'shap_bar.png'
            )
            
            # Individual explanations
            for i in [0, 10, 20]:
                if i < len(test_subset):
                    explainer.plot_waterfall(
                        i,
                        test_subset,
                        show=False,
                        save_path=output_dir / f'shap_waterfall_sample_{i}.png'
                    )
        
        logger.info("✓ SHAP analysis complete")
        
    except ImportError:
        logger.warning("SHAP not installed. Skipping SHAP analysis.")
    except Exception as e:
        logger.error(f"Error in SHAP analysis: {e}")


def demonstrate_feature_importance(
    model: Any,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    feature_names: list[str],
    output_dir: Path,
) -> None:
    """Demonstrate feature importance analysis.
    
    Args:
        model: Trained model
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target
        feature_names: Feature names
        output_dir: Output directory
    """
    logger.info(f"\n{'='*70}")
    logger.info("Feature Importance Analysis")
    logger.info(f"{'='*70}\n")
    
    analyzer = FeatureImportanceAnalyzer(model, feature_names)
    
    # Built-in importance
    if hasattr(model, 'feature_importances_'):
        logger.info("Built-in Importance:")
        built_in = analyzer.get_built_in_importance(top_n=15)
        print(built_in[['feature', 'importance', 'cumulative_pct']].to_string(index=False))
        print()
    
    # Permutation importance
    logger.info("Computing Permutation Importance...")
    perm_imp = analyzer.get_permutation_importance(
        X_test, y_test, n_repeats=5, top_n=15
    )
    print(perm_imp[['feature', 'importance_mean', 'importance_std']].to_string(index=False))
    print()
    
    # Compare methods
    logger.info("Method Comparison:")
    comparison = analyzer.compare_methods(top_n=10)
    if not comparison.empty:
        print(comparison.to_string())
    print()
    
    # Feature correlations
    logger.info("Highly Correlated Features:")
    correlations = analyzer.analyze_feature_correlations(X_train, threshold=0.8)
    if not correlations.empty:
        print(correlations.head(10).to_string(index=False))
    else:
        print("No highly correlated features found.")
    print()
    
    # Save plots
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        
        analyzer.plot_cumulative_importance(
            method='built_in',
            show=False,
            save_path=output_dir / 'importance_cumulative.png'
        )
        
        analyzer.plot_importance_comparison(
            top_n=15,
            show=False,
            save_path=output_dir / 'importance_comparison.png'
        )
    
    logger.info("✓ Feature importance analysis complete")


def demonstrate_partial_dependence(
    model: Any,
    X_test: pd.DataFrame,
    feature_names: list[str],
    output_dir: Path,
) -> None:
    """Demonstrate partial dependence plots.
    
    Args:
        model: Trained model
        X_test: Test features
        feature_names: Feature names
        output_dir: Output directory
    """
    logger.info(f"\n{'='*70}")
    logger.info("Partial Dependence Analysis")
    logger.info(f"{'='*70}\n")
    
    analyzer = PartialDependenceAnalyzer(model, feature_names)
    
    # Select top features (if available from model)
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
        top_indices = np.argsort(importance)[::-1][:6]
        top_features = [feature_names[i] for i in top_indices]
    else:
        # Use first 6 features
        top_features = feature_names[:6]
    
    logger.info(f"Computing PDP for top features: {top_features}")
    
    # Feature effect summary
    logger.info("\nFeature Effect Summary:")
    effect_summary = analyzer.get_feature_effect_summary(X_test, features=top_features)
    print(effect_summary.to_string(index=False))
    print()
    
    # Save plots
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Individual PDPs
        analyzer.plot_partial_dependence(
            X_test,
            features=top_features,
            n_cols=3,
            show=False,
            save_path=output_dir / 'pdp_top_features.png'
        )
        
        # ICE plots for top feature
        if top_features:
            analyzer.plot_ice(
                X_test,
                feature=top_features[0],
                sample_indices=np.random.choice(len(X_test), min(50, len(X_test)), replace=False),
                centered=False,
                show_pd=True,
                show=False,
                save_path=output_dir / f'ice_{top_features[0]}.png'
            )
        
        # 2D PDP for top 2 features (interaction)
        if len(top_features) >= 2:
            analyzer.plot_2d_partial_dependence(
                X_test,
                features=(top_features[0], top_features[1]),
                grid_resolution=20,
                show=False,
                save_path=output_dir / 'pdp_2d_interaction.png'
            )
    
    logger.info("✓ Partial dependence analysis complete")


def demonstrate_individual_explanations(
    model: Any,
    X_test: pd.DataFrame,
    feature_names: list[str],
    shap_values: Optional[np.ndarray] = None,
) -> None:
    """Demonstrate individual prediction explanations.
    
    Args:
        model: Trained model
        X_test: Test features
        feature_names: Feature names
        shap_values: Pre-computed SHAP values (optional)
    """
    logger.info(f"\n{'='*70}")
    logger.info("Individual Prediction Explanations")
    logger.info(f"{'='*70}\n")
    
    explainer = IndividualExplainer(model, feature_names)
    
    # Explain a sample
    sample_idx = 0
    logger.info(f"Explaining sample {sample_idx}:")
    explanation = explainer.explain_sample(X_test, sample_idx, shap_values, top_n=10)
    explainer.print_explanation(explanation)
    print()
    
    # Compare two samples
    if len(X_test) > 10:
        logger.info("Comparing two samples:")
        comparison = explainer.compare_samples(X_test, 0, 10, shap_values)
        explainer.print_comparison(comparison, top_n=5)
        print()
    
    # Find similar samples
    logger.info("Finding similar samples:")
    similar = explainer.find_similar_samples(X_test, sample_idx, n_neighbors=5)
    print(f"Sample {sample_idx} prediction: {similar['sample_prediction']}")
    print(f"Average neighbor prediction: {similar['avg_neighbor_prediction']:.4f}")
    print("\nNeighbors:")
    for neighbor in similar['neighbors']:
        print(f"  Index {neighbor['index']}: "
              f"distance={neighbor['distance']:.4f}, "
              f"prediction={neighbor['prediction']}")
    print()
    
    logger.info("✓ Individual explanations complete")


def main():
    """Main explainability script."""
    parser = argparse.ArgumentParser(description='Model explainability demonstration')
    parser.add_argument(
        '--timeframe',
        type=str,
        default='4h',
        choices=['1h', '4h', '1d'],
        help='Timeframe to use'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='xgboost',
        choices=['logistic', 'random_forest', 'xgboost', 'lightgbm'],
        help='Model to explain'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='reports/explainability',
        help='Output directory for visualizations'
    )
    
    args = parser.parse_args()
    
    try:
        # Load data
        logger.info(f"Loading {args.timeframe} data...")
        df = load_feature_data(args.timeframe)
        
        # Prepare features and target
        trainer = ModelTrainer()
        
        logger.info("Creating target variable...")
        target = trainer.create_binary_direction_target(df, horizon=1)
        
        logger.info("Preparing features...")
        X = trainer.prepare_features(df)
        
        # Remove NaN
        valid_idx = ~(target.isna() | X.isna().any(axis=1))
        X_clean = X[valid_idx]
        y_clean = target[valid_idx]
        
        logger.info(f"Clean data: {len(X_clean)} samples, {len(X_clean.columns)} features")
        
        # Train/test split
        splits = trainer.temporal_train_test_split(
            X_clean, y_clean, test_size=0.2, val_size=0.2
        )
        
        X_train, y_train = splits['train']
        X_test, y_test = splits['test']
        
        feature_names = X_train.columns.tolist()
        
        # Train model
        model = train_model(args.model, X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        metrics = ModelEvaluator.evaluate_classifier(y_test.values, y_pred)
        logger.info(f"\nModel Performance: Accuracy={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}\n")
        
        # Setup output directory
        output_dir = project_root / args.output_dir / args.model
        
        # Run explainability analyses
        demonstrate_shap(model, X_train, X_test, feature_names, output_dir)
        
        demonstrate_feature_importance(
            model, X_train, y_train.values, X_test, y_test.values, 
            feature_names, output_dir
        )
        
        demonstrate_partial_dependence(model, X_test, feature_names, output_dir)
        
        # Get SHAP values for individual explanations (if available)
        shap_values = None
        try:
            from aurummind_ai.explainability import SHAPExplainer
            explainer = SHAPExplainer(model, feature_names, model_type='tree')
            explainer.create_explainer()
            test_subset = X_test.sample(min(100, len(X_test)), random_state=42)
            shap_values = explainer.explain(test_subset, check_additivity=False)
        except:
            pass
        
        demonstrate_individual_explanations(
            model, X_test, feature_names, shap_values
        )
        
        logger.info(f"\n{'='*70}")
        logger.info("✓ Phase 6 Explainability Analysis Complete!")
        logger.info(f"{'='*70}\n")
        logger.info("Outputs saved to:")
        logger.info(f"  {output_dir}")
        
    except Exception as e:
        logger.error(f"Error during explainability analysis: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
