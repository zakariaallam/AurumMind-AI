"""Comprehensive model evaluation script.

This script demonstrates Phase 5 capabilities:
- Walk-forward validation
- Regime-specific evaluation
- Backtesting with realistic costs
- Performance attribution

Usage:
    python scripts/evaluate_models.py --timeframe 4h --model xgboost
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

from aurummind_ai.evaluation import (
    ModelEvaluator,
    WalkForwardValidator,
    RegimeDetector,
    RegimeEvaluator,
    BacktestEngine,
    TradeConfig,
    PerformanceAttribution,
    ErrorAnalysis,
)
from aurummind_ai.models import BaselineModels, BoostingModels, ModelTrainer

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
    
    # Parse timestamp
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    logger.info(f"Loaded {len(df)} samples with {len(df.columns)} columns")
    
    return df


def evaluate_classification_model(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    """Evaluate a classification model.
    
    Args:
        model_name: Name of model to evaluate
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target
        
    Returns:
        Dictionary with evaluation results
    """
    logger.info(f"\n{'='*70}")
    logger.info(f"Evaluating {model_name}")
    logger.info(f"{'='*70}\n")
    
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
    logger.info("Training model...")
    model.fit(X_train, y_train)
    
    # Predict
    logger.info("Generating predictions...")
    y_pred = model.predict(X_test)
    
    try:
        y_proba = model.predict_proba(X_test)
    except:
        y_proba = None
    
    # Basic evaluation
    logger.info("\nBasic Evaluation:")
    metrics = ModelEvaluator.evaluate_classifier(y_test.values, y_pred, y_proba)
    ModelEvaluator.print_classification_metrics(metrics, model_name)
    
    return {
        'model': model,
        'y_pred': y_pred,
        'y_proba': y_proba,
        'metrics': metrics,
    }


def run_walk_forward_validation(
    model_factory: callable,
    X: pd.DataFrame,
    y: pd.Series,
    train_window: int = 1000,
    test_window: int = 200,
) -> dict:
    """Run walk-forward validation.
    
    Args:
        model_factory: Function that returns fresh model instance
        X: Feature matrix
        y: Target series
        train_window: Training window size
        test_window: Test window size
        
    Returns:
        Validation results
    """
    logger.info(f"\n{'='*70}")
    logger.info("Walk-Forward Validation")
    logger.info(f"{'='*70}\n")
    
    # Define evaluation function
    def eval_func(y_true, y_pred):
        return ModelEvaluator.evaluate_classifier(y_true, y_pred, average='weighted')
    
    # Run validation
    validator = WalkForwardValidator(
        train_window=train_window,
        test_window=test_window,
        retrain_every=1,
    )
    
    results = validator.validate(model_factory, X, y, eval_func)
    
    # Print results
    logger.info("\nWalk-Forward Results:")
    logger.info(f"  Number of steps: {results['num_steps']}")
    logger.info(f"  Total predictions: {results['total_predictions']}")
    
    for metric in ['accuracy', 'precision', 'recall', 'f1']:
        if f'{metric}_mean' in results:
            mean_val = results[f'{metric}_mean']
            std_val = results[f'{metric}_std']
            logger.info(f"  {metric.title()}: {mean_val:.4f} ± {std_val:.4f}")
    
    return results


def run_regime_analysis(
    df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict:
    """Run regime-specific evaluation.
    
    Args:
        df: DataFrame with price data
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Regime analysis results
    """
    logger.info(f"\n{'='*70}")
    logger.info("Regime Analysis")
    logger.info(f"{'='*70}\n")
    
    # Detect regimes
    detector = RegimeDetector(volatility_window=20, trend_window=50)
    
    logger.info("\nDetecting volatility regimes...")
    vol_regime = detector.detect_volatility_regime(df)
    
    logger.info("\nDetecting trend regimes...")
    trend_regime = detector.detect_trend_regime(df)
    
    # Align with predictions
    n = min(len(y_true), len(vol_regime))
    y_true = y_true[:n]
    y_pred = y_pred[:n]
    vol_regime = vol_regime.values[:n]
    trend_regime = trend_regime.values[:n]
    
    # Evaluate by regime
    def eval_func(y_t, y_p):
        return ModelEvaluator.evaluate_classifier(y_t, y_p, average='weighted')
    
    logger.info("\nEvaluating by volatility regime...")
    vol_results = RegimeEvaluator.evaluate_by_regime(
        y_true, y_pred, vol_regime, eval_func
    )
    RegimeEvaluator.print_regime_analysis(vol_results, 'volatility')
    
    logger.info("\nEvaluating by trend regime...")
    trend_results = RegimeEvaluator.evaluate_by_regime(
        y_true, y_pred, trend_regime, eval_func
    )
    RegimeEvaluator.print_regime_analysis(trend_results, 'trend')
    
    return {
        'volatility_regimes': vol_results,
        'trend_regimes': trend_results,
    }


def run_backtesting(
    df: pd.DataFrame,
    predictions: np.ndarray,
    initial_capital: float = 10000.0,
) -> dict:
    """Run backtesting with realistic costs.
    
    Args:
        df: DataFrame with price data
        predictions: Model predictions
        initial_capital: Starting capital
        
    Returns:
        Backtest results
    """
    logger.info(f"\n{'='*70}")
    logger.info("Backtesting")
    logger.info(f"{'='*70}\n")
    
    # Configure backtest
    config = TradeConfig(
        transaction_cost=0.0001,  # 0.01% per trade
        slippage=0.0001,          # 0.01% slippage
        position_size=1.0,         # Full position
    )
    
    # Run backtest
    engine = BacktestEngine(initial_capital=initial_capital, config=config)
    
    # Align data
    n = min(len(df), len(predictions))
    df_test = df.iloc[:n].copy()
    predictions_test = predictions[:n]
    
    results = engine.run(df_test, predictions_test)
    
    # Print results
    BacktestEngine.print_backtest_results(results)
    
    # Get trades
    trades_df = engine.get_trades_dataframe()
    if not trades_df.empty:
        logger.info(f"\nFirst 10 trades:")
        print(trades_df.head(10).to_string())
    
    return results


def run_performance_attribution(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    timestamps: pd.DatetimeIndex,
) -> dict:
    """Run performance attribution analysis.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        timestamps: Timestamps for predictions
        
    Returns:
        Attribution results
    """
    logger.info(f"\n{'='*70}")
    logger.info("Performance Attribution")
    logger.info(f"{'='*70}\n")
    
    def eval_func(y_t, y_p):
        return ModelEvaluator.evaluate_classifier(y_t, y_p, average='weighted')
    
    # By month
    logger.info("\nPerformance by Month:")
    monthly = PerformanceAttribution.by_time_period(
        y_true, y_pred, timestamps, eval_func, period='M'
    )
    print(monthly[['period', 'accuracy', 'f1', 'sample_count']].to_string(index=False))
    
    # By day of week
    logger.info("\n\nPerformance by Day of Week:")
    daily = PerformanceAttribution.by_day_of_week(
        y_true, y_pred, timestamps, eval_func
    )
    print(daily[['day_of_week', 'accuracy', 'f1', 'sample_count']].to_string(index=False))
    
    return {
        'monthly': monthly,
        'daily': daily,
    }


def run_error_analysis(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict:
    """Run error analysis.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Error analysis results
    """
    logger.info(f"\n{'='*70}")
    logger.info("Error Analysis")
    logger.info(f"{'='*70}\n")
    
    # Confusion analysis
    confusion = ErrorAnalysis.confusion_analysis(y_true, y_pred)
    
    logger.info("Per-class Accuracy:")
    for class_name, accuracy in confusion['class_accuracy'].items():
        logger.info(f"  {class_name}: {accuracy:.4f}")
    
    logger.info("\nTop Misclassifications:")
    for misc in confusion['top_misclassifications'][:5]:
        logger.info(
            f"  {misc['true_class']} -> {misc['predicted_class']}: "
            f"{misc['count']} ({misc['percentage']:.2f}%)"
        )
    
    return confusion


def main():
    """Main evaluation script."""
    parser = argparse.ArgumentParser(description='Comprehensive model evaluation')
    parser.add_argument(
        '--timeframe',
        type=str,
        default='4h',
        choices=['1h', '4h', '1d'],
        help='Timeframe to evaluate'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='xgboost',
        choices=['logistic', 'random_forest', 'xgboost', 'lightgbm'],
        help='Model to evaluate'
    )
    parser.add_argument(
        '--train-window',
        type=int,
        default=1000,
        help='Training window for walk-forward validation'
    )
    parser.add_argument(
        '--test-window',
        type=int,
        default=200,
        help='Test window for walk-forward validation'
    )
    parser.add_argument(
        '--skip-walk-forward',
        action='store_true',
        help='Skip walk-forward validation (slow)'
    )
    
    args = parser.parse_args()
    
    try:
        # Load data
        logger.info(f"Loading {args.timeframe} data...")
        df = load_feature_data(args.timeframe)
        
        # Prepare features and target
        trainer = ModelTrainer()
        
        # Create target
        logger.info("Creating target variable...")
        target = trainer.create_binary_direction_target(df, horizon=1)
        
        # Prepare features
        logger.info("Preparing features...")
        X = trainer.prepare_features(df)
        
        # Remove NaN
        valid_idx = ~(target.isna() | X.isna().any(axis=1))
        X_clean = X[valid_idx]
        y_clean = target[valid_idx]
        df_clean = df[valid_idx]
        
        logger.info(f"Clean data: {len(X_clean)} samples, {len(X_clean.columns)} features")
        
        # Train/test split (temporal)
        logger.info("Creating train/test split...")
        splits = trainer.temporal_train_test_split(
            X_clean, y_clean, test_size=0.2, val_size=0.2
        )
        
        X_train, y_train = splits['train']
        X_test, y_test = splits['test']
        
        # Get corresponding price data for test set
        test_start_idx = len(X_train) + len(splits['val'][0])
        df_test = df_clean.iloc[test_start_idx:].reset_index(drop=True)
        
        # 1. Basic evaluation
        eval_results = evaluate_classification_model(
            args.model, X_train, y_train, X_test, y_test
        )
        
        # 2. Regime analysis
        regime_results = run_regime_analysis(
            df_test, y_test.values, eval_results['y_pred']
        )
        
        # 3. Backtesting
        backtest_results = run_backtesting(
            df_test, eval_results['y_pred'], initial_capital=10000.0
        )
        
        # 4. Performance attribution
        if 'timestamp' in df_test.columns:
            timestamps = pd.to_datetime(df_test['timestamp'])
            attribution_results = run_performance_attribution(
                y_test.values, eval_results['y_pred'], timestamps
            )
        
        # 5. Error analysis
        error_results = run_error_analysis(
            y_test.values, eval_results['y_pred']
        )
        
        # 6. Walk-forward validation (optional, can be slow)
        if not args.skip_walk_forward:
            # Create model factory
            if args.model == 'xgboost':
                model_factory = lambda: BoostingModels().create_xgboost_classifier()
            elif args.model == 'lightgbm':
                model_factory = lambda: BoostingModels().create_lightgbm_classifier()
            elif args.model == 'random_forest':
                model_factory = lambda: BaselineModels().create_random_forest_classifier()
            else:
                model_factory = lambda: BaselineModels().create_logistic_regression()
            
            wf_results = run_walk_forward_validation(
                model_factory, X_clean, y_clean,
                train_window=args.train_window,
                test_window=args.test_window,
            )
        
        logger.info(f"\n{'='*70}")
        logger.info("✓ Phase 5 Evaluation Complete!")
        logger.info(f"{'='*70}\n")
        logger.info("All evaluation methods executed successfully:")
        logger.info("  ✓ Basic model evaluation")
        logger.info("  ✓ Regime-specific analysis")
        logger.info("  ✓ Backtesting with realistic costs")
        logger.info("  ✓ Performance attribution")
        logger.info("  ✓ Error analysis")
        if not args.skip_walk_forward:
            logger.info("  ✓ Walk-forward validation")
        
    except Exception as e:
        logger.error(f"Error during evaluation: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
