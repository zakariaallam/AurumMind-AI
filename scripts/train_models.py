"""Train machine learning models on engineered features.

This script trains baseline and advanced models for
market direction prediction and return forecasting.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
import yaml

from aurummind_ai.models import ModelTrainer
from aurummind_ai.evaluation import ModelEvaluator
from aurummind_ai.utils.logging_config import setup_logging

setup_logging(level="INFO")


def load_config(config_path: str = "configs/models.yaml") -> dict:
    """Load model configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def train_classification_models(
    features_path: str,
    config: dict,
    timeframe: str,
) -> None:
    """Train classification models.
    
    Args:
        features_path: Path to features Parquet file
        config: Model configuration
        timeframe: Timeframe identifier
    """
    print("\n" + "=" * 70)
    print(f"Training Classification Models - {timeframe.upper()}")
    print("=" * 70)
    print()
    
    # Load features
    print("Loading features...")
    df = pd.read_parquet(features_path)
    print(f"✓ Loaded {len(df)} records with {len(df.columns)} columns")
    print()
    
    # Initialize trainer
    trainer = ModelTrainer(random_state=42)
    evaluator = ModelEvaluator()
    
    # Create target variable
    print("Creating target variable...")
    target_config = config['target']
    
    if target_config['type'] == 'binary_direction':
        target = trainer.create_binary_direction_target(
            df, horizon=target_config['horizon']
        )
    elif target_config['type'] == 'multi_direction':
        target = trainer.create_direction_target(
            df, 
            horizon=target_config['horizon'],
            threshold=target_config['threshold']
        )
    else:
        raise ValueError(f"Unknown target type: {target_config['type']}")
    
    print()
    
    # Prepare features
    print("Preparing features...")
    X = trainer.prepare_features(df, exclude_columns=config['features']['exclude_columns'])
    print(f"✓ Using {len(trainer.feature_columns)} features")
    print()
    
    # Create train/val/test splits
    print("Creating temporal train/val/test splits...")
    splits = trainer.temporal_train_test_split(
        X, target,
        test_size=config['split']['test_size'],
        val_size=config['split']['val_size']
    )
    
    X_train, y_train = splits['train']
    X_val, y_val = splits['val']
    X_test, y_test = splits['test']
    print()
    
    # Store results
    results = {}
    
    # Train baseline models
    print("=" * 70)
    print("BASELINE MODELS")
    print("=" * 70)
    print()
    
    baseline = trainer.get_baseline_models()
    
    # 1. Logistic Regression
    print("1. Training Logistic Regression...")
    baseline.create_logistic_regression(**config['baseline']['logistic_regression'])
    baseline.fit('logistic_regression', X_train, y_train, scale_features=True)
    
    y_pred = baseline.predict('logistic_regression', X_test, scale_features=True)
    y_proba = baseline.predict_proba('logistic_regression', X_test, scale_features=True)
    
    metrics = evaluator.evaluate_classifier(y_test, y_pred, y_proba)
    results['logistic_regression'] = metrics
    
    print(f"  Test Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Test F1-Score: {metrics['f1']:.4f}")
    print()
    
    # 2. Random Forest
    print("2. Training Random Forest...")
    baseline.create_random_forest_classifier(**config['baseline']['random_forest_classifier'])
    baseline.fit('random_forest_classifier', X_train, y_train, scale_features=False)
    
    y_pred = baseline.predict('random_forest_classifier', X_test, scale_features=False)
    y_proba = baseline.predict_proba('random_forest_classifier', X_test, scale_features=False)
    
    metrics = evaluator.evaluate_classifier(y_test, y_pred, y_proba)
    results['random_forest'] = metrics
    
    print(f"  Test Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Test F1-Score: {metrics['f1']:.4f}")
    
    # Feature importance
    importance = baseline.get_feature_importance('random_forest_classifier', top_n=10)
    print("\n  Top 10 Features:")
    for idx, row in importance.iterrows():
        print(f"    {row['feature']}: {row['importance']:.4f}")
    print()
    
    # Train advanced models
    print("=" * 70)
    print("ADVANCED MODELS (BOOSTING)")
    print("=" * 70)
    print()
    
    boosting = trainer.get_boosting_models()
    
    # 3. XGBoost
    print("3. Training XGBoost...")
    xgb_config = config['boosting']['xgboost_classifier']
    early_stopping = xgb_config.pop('early_stopping_rounds', None)
    
    boosting.create_xgboost_classifier(**xgb_config)
    boosting.fit(
        'xgboost_classifier', X_train, y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=early_stopping,
        verbose=False
    )
    
    y_pred = boosting.predict('xgboost_classifier', X_test)
    y_proba = boosting.predict_proba('xgboost_classifier', X_test)
    
    metrics = evaluator.evaluate_classifier(y_test, y_pred, y_proba)
    results['xgboost'] = metrics
    
    print(f"  Test Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Test F1-Score: {metrics['f1']:.4f}")
    
    # Feature importance
    importance = boosting.get_feature_importance('xgboost_classifier', top_n=10)
    print("\n  Top 10 Features:")
    for idx, row in importance.iterrows():
        print(f"    {row['feature']}: {row['importance']:.4f}")
    print()
    
    # 4. LightGBM
    print("4. Training LightGBM...")
    lgb_config = config['boosting']['lightgbm_classifier']
    early_stopping = lgb_config.pop('early_stopping_rounds', None)
    
    boosting.create_lightgbm_classifier(**lgb_config)
    boosting.fit(
        'lightgbm_classifier', X_train, y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=early_stopping,
        verbose=False
    )
    
    y_pred = boosting.predict('lightgbm_classifier', X_test)
    y_proba = boosting.predict_proba('lightgbm_classifier', X_test)
    
    metrics = evaluator.evaluate_classifier(y_test, y_pred, y_proba)
    results['lightgbm'] = metrics
    
    print(f"  Test Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Test F1-Score: {metrics['f1']:.4f}")
    
    # Feature importance
    importance = boosting.get_feature_importance('lightgbm_classifier', top_n=10)
    print("\n  Top 10 Features:")
    for idx, row in importance.iterrows():
        print(f"    {row['feature']}: {row['importance']:.4f}")
    print()
    
    # Compare all models
    print("=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)
    print()
    
    comparison = evaluator.compare_models(results, metric='accuracy')
    print(comparison[['model', 'accuracy', 'precision', 'recall', 'f1', 'roc_auc']])
    print()
    
    # Best model
    best_model = comparison.iloc[0]['model']
    best_accuracy = comparison.iloc[0]['accuracy']
    print(f"Best Model: {best_model} (Accuracy: {best_accuracy:.4f})")
    print()
    
    print("=" * 70)
    print(f"✓ Training complete for {timeframe}")
    print("=" * 70)
    print()


def main():
    """Main entry point."""
    print("\n")
    print("=" * 70)
    print("AurumMind AI - Model Training")
    print("=" * 70)
    print()
    
    # Load configuration
    config = load_config()
    print(f"✓ Loaded configuration")
    print(f"  Task: {config['target']['task']}")
    print(f"  Target: {config['target']['type']}")
    print(f"  Horizon: {config['target']['horizon']}")
    print()
    
    # Define features files
    features_files = [
        {
            'path': 'data/processed/1h',
            'timeframe': '1h',
        },
        # Uncomment to train on other timeframes
        # {
        #     'path': 'data/processed/4h',
        #     'timeframe': '4h',
        # },
        # {
        #     'path': 'data/processed/1d',
        #     'timeframe': '1d',
        # },
    ]
    
    # Train models for each timeframe
    for file_config in features_files:
        features_dir = Path(file_config['path'])
        
        # Find the most recent features file
        features_files_list = list(features_dir.glob('features_*.parquet'))
        
        if not features_files_list:
            print(f"⚠ No features found in {features_dir}")
            print(f"  Run: python scripts/generate_features.py")
            print()
            continue
        
        # Use most recent file
        features_path = sorted(features_files_list)[-1]
        
        train_classification_models(
            features_path=str(features_path),
            config=config,
            timeframe=file_config['timeframe']
        )
    
    print()
    print("=" * 70)
    print("All model training complete!")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()
