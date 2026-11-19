"""Generate features for XAUUSD datasets.

This script processes market data and generates comprehensive
technical indicators and price action features.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd

from aurummind_ai.features import FeaturePipeline
from aurummind_ai.utils.logging_config import setup_logging

setup_logging(level="INFO")


def generate_features_for_file(
    input_path: str,
    output_dir: str,
    timeframe: str,
) -> None:
    """Generate features for a single data file.
    
    Args:
        input_path: Path to input CSV file
        output_dir: Directory to save features
        timeframe: Timeframe identifier (1h, 4h, 1d)
    """
    print("=" * 70)
    print(f"Processing: {input_path}")
    print("=" * 70)
    print()
    
    # Check if file exists
    if not Path(input_path).exists():
        print(f"✗ File not found: {input_path}")
        print()
        return
    
    # Load data
    print("Loading data...")
    df = pd.read_csv(input_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    print(f"✓ Loaded {len(df)} records")
    print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print()
    
    # Initialize pipeline
    pipeline = FeaturePipeline()
    
    # Generate features
    print("Generating features...")
    print()
    df_features = pipeline.generate_features(df)
    print()
    
    # Display feature info
    feature_info = pipeline.get_feature_info()
    print("Feature Summary:")
    print(f"  Total features: {feature_info['total_features']}")
    print(f"  Technical indicators: {feature_info['technical_indicators']}")
    print(f"  Price action features: {feature_info['price_action_features']}")
    print()
    
    # Validate features
    print("Validating features...")
    validation = pipeline.validate_features(df_features)
    print(f"  Total records: {validation['total_records']}")
    print(f"  Total columns: {validation['total_columns']}")
    print(f"  Missing values: {validation['missing_values']}")
    print(f"  Infinite values: {validation['infinite_values']}")
    
    if validation['issues']:
        print(f"  Issues found: {len(validation['issues'])}")
        for issue in validation['issues']:
            print(f"    - {issue}")
    else:
        print(f"  ✓ Validation passed")
    print()
    
    # Save features
    print("Saving features...")
    output_path = pipeline.save_features(df_features, output_dir, timeframe)
    print()
    
    print("=" * 70)
    print(f"✓ Feature generation complete for {timeframe}")
    print(f"  Output: {output_path}")
    print("=" * 70)
    print()


def main():
    """Main entry point."""
    print("\n")
    print("=" * 70)
    print("AurumMind AI - Feature Generation")
    print("=" * 70)
    print()
    
    # Define files to process
    files = [
        {
            "input": "data/raw/samples/XAUUSD_1h_sample.csv",
            "output": "data/processed/1h",
            "timeframe": "1h",
        },
        {
            "input": "data/raw/samples/XAUUSD_4h_sample.csv",
            "output": "data/processed/4h",
            "timeframe": "4h",
        },
        {
            "input": "data/raw/samples/XAUUSD_1d_sample.csv",
            "output": "data/processed/1d",
            "timeframe": "1d",
        },
    ]
    
    # Process each file
    for file_config in files:
        generate_features_for_file(
            input_path=file_config["input"],
            output_dir=file_config["output"],
            timeframe=file_config["timeframe"],
        )
    
    print()
    print("=" * 70)
    print("All feature generation complete!")
    print("=" * 70)
    print()
    print("Generated files:")
    print("  - data/processed/1h/features_XAUUSD_1h_*.parquet")
    print("  - data/processed/4h/features_XAUUSD_4h_*.parquet")
    print("  - data/processed/1d/features_XAUUSD_1d_*.parquet")
    print()
    print("Each file includes:")
    print("  - All original OHLCV data")
    print("  - Technical indicators (SMA, EMA, RSI, MACD, BB, ATR, ADX, etc.)")
    print("  - Price action features (returns, momentum, volatility, structure)")
    print("  - Metadata JSON file with feature descriptions")
    print()


if __name__ == "__main__":
    main()
