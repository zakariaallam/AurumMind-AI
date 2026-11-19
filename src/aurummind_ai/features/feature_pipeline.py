"""Feature engineering pipeline for AurumMind AI.

This module orchestrates the feature generation process,
ensuring proper ordering and preventing data leakage.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from .price_action import PriceActionFeatures
from .technical_indicators import TechnicalIndicators

logger = logging.getLogger(__name__)


class FeaturePipeline:
    """Complete feature engineering pipeline.
    
    Orchestrates technical indicators and price action features,
    handles missing values, and saves engineered features.
    """
    
    def __init__(self, config: Optional[dict[str, Any]] = None):
        """Initialize feature pipeline.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or self._default_config()
        self.feature_names: list[str] = []
        self.original_columns: list[str] = []
    
    @staticmethod
    def _default_config() -> dict[str, Any]:
        """Get default configuration."""
        return {
            "technical_indicators": {
                "sma_periods": [20, 50, 200],
                "ema_periods": [9, 20, 50, 200],
                "rsi_period": 14,
                "macd_params": (12, 26, 9),
                "bb_period": 20,
                "atr_period": 14,
                "adx_period": 14,
                "stoch_params": (14, 3),
                "roc_period": 12,
                "include_obv": True,
            },
            "price_action": {
                "return_periods": [1, 3, 5, 10, 20],
                "rolling_windows": [5, 10, 20],
                "momentum_periods": [3, 5, 10, 20],
                "lookback_structure": 5,
                "support_resistance_window": 20,
            },
            "handle_missing": "forward_fill",  # forward_fill, drop, or interpolate
            "min_required_rows": 200,  # Minimum rows after feature generation
        }
    
    def generate_features(
        self,
        df: pd.DataFrame,
        include_technical: bool = True,
        include_price_action: bool = True,
    ) -> pd.DataFrame:
        """Generate all features for the dataset.
        
        Args:
            df: DataFrame with OHLCV data
            include_technical: Whether to include technical indicators
            include_price_action: Whether to include price action features
            
        Returns:
            DataFrame with all generated features
        """
        logger.info("=" * 70)
        logger.info("Feature Engineering Pipeline")
        logger.info("=" * 70)
        logger.info(f"Input shape: {df.shape}")
        logger.info(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        
        self.original_columns = df.columns.tolist()
        df_features = df.copy()
        
        # Generate technical indicators
        if include_technical:
            logger.info("\n1. Generating technical indicators...")
            df_features = TechnicalIndicators.calculate_all_indicators(
                df_features,
                **self.config["technical_indicators"]
            )
        
        # Generate price action features
        if include_price_action:
            logger.info("\n2. Generating price action features...")
            df_features = PriceActionFeatures.calculate_all_price_action_features(
                df_features
            )
        
        # Store feature names (excluding original columns)
        self.feature_names = [
            col for col in df_features.columns 
            if col not in self.original_columns
        ]
        
        logger.info(f"\n✓ Generated {len(self.feature_names)} features")
        logger.info(f"Output shape: {df_features.shape}")
        
        # Handle missing values
        df_features = self._handle_missing_values(df_features)
        
        logger.info("=" * 70)
        
        return df_features
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in features.
        
        Args:
            df: DataFrame with features
            
        Returns:
            DataFrame with handled missing values
        """
        method = self.config["handle_missing"]
        
        missing_before = df.isnull().sum().sum()
        
        if missing_before == 0:
            logger.info("\n✓ No missing values detected")
            return df
        
        logger.info(f"\n3. Handling missing values ({missing_before} total)...")
        
        if method == "forward_fill":
            df = df.fillna(method="ffill")
            remaining = df.isnull().sum().sum()
            if remaining > 0:
                # Fill remaining NaNs at the start with backward fill
                df = df.fillna(method="bfill")
            logger.info(f"   Method: Forward fill")
        
        elif method == "drop":
            df = df.dropna()
            logger.info(f"   Method: Drop rows with missing values")
            logger.info(f"   Rows remaining: {len(df)}")
        
        elif method == "interpolate":
            # Only interpolate numeric columns
            numeric_cols = df.select_dtypes(include=['number']).columns
            df[numeric_cols] = df[numeric_cols].interpolate(method='linear')
            df = df.fillna(method="bfill")  # Handle any remaining
            logger.info(f"   Method: Linear interpolation")
        
        missing_after = df.isnull().sum().sum()
        logger.info(f"   Remaining missing values: {missing_after}")
        
        # Check minimum required rows
        if len(df) < self.config["min_required_rows"]:
            logger.warning(
                f"⚠ Only {len(df)} rows remaining after feature generation. "
                f"Minimum required: {self.config['min_required_rows']}"
            )
        
        return df
    
    def get_feature_info(self) -> dict[str, Any]:
        """Get information about generated features.
        
        Returns:
            Dictionary with feature information
        """
        if not self.feature_names:
            return {"message": "No features generated yet"}
        
        # Categorize features
        technical_features = [f for f in self.feature_names if any(
            indicator in f for indicator in 
            ["sma", "ema", "rsi", "macd", "bb_", "atr", "adx", "stoch", "roc", "obv"]
        )]
        
        price_action_features = [f for f in self.feature_names if f not in technical_features]
        
        return {
            "total_features": len(self.feature_names),
            "technical_indicators": len(technical_features),
            "price_action_features": len(price_action_features),
            "feature_categories": {
                "technical": technical_features[:10],  # First 10 examples
                "price_action": price_action_features[:10],
            },
        }
    
    def save_features(
        self,
        df: pd.DataFrame,
        output_dir: str,
        timeframe: str,
        include_metadata: bool = True,
    ) -> Path:
        """Save engineered features to disk.
        
        Args:
            df: DataFrame with features
            output_dir: Directory to save features
            timeframe: Timeframe identifier (1h, 4h, 1d)
            include_metadata: Whether to save metadata file
            
        Returns:
            Path to saved features file
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"features_XAUUSD_{timeframe}_{timestamp}.parquet"
        features_path = output_path / filename
        
        # Save features as Parquet
        df.to_parquet(features_path, index=False, engine="pyarrow")
        logger.info(f"✓ Saved features to {features_path}")
        
        # Save metadata
        if include_metadata:
            metadata = {
                "timestamp": timestamp,
                "timeframe": timeframe,
                "total_records": len(df),
                "total_features": len(self.feature_names),
                "date_range": {
                    "start": str(df["timestamp"].min()),
                    "end": str(df["timestamp"].max()),
                },
                "feature_info": self.get_feature_info(),
                "config": self.config,
                "original_columns": self.original_columns,
                "feature_names": self.feature_names,
            }
            
            metadata_path = features_path.with_suffix(".json")
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2, default=str)
            
            logger.info(f"✓ Saved metadata to {metadata_path}")
        
        return features_path
    
    def load_features(self, features_path: str) -> tuple[pd.DataFrame, dict]:
        """Load previously saved features.
        
        Args:
            features_path: Path to features Parquet file
            
        Returns:
            Tuple of (DataFrame, metadata dict)
        """
        features_path = Path(features_path)
        
        # Load features
        df = pd.read_parquet(features_path)
        logger.info(f"✓ Loaded features from {features_path}")
        logger.info(f"  Shape: {df.shape}")
        
        # Load metadata if available
        metadata_path = features_path.with_suffix(".json")
        metadata = {}
        
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            logger.info(f"✓ Loaded metadata from {metadata_path}")
        
        return df, metadata
    
    def validate_features(self, df: pd.DataFrame) -> dict[str, Any]:
        """Validate generated features.
        
        Args:
            df: DataFrame with features
            
        Returns:
            Dictionary with validation results
        """
        validation = {
            "total_records": len(df),
            "total_columns": len(df.columns),
            "feature_count": len(self.feature_names),
            "missing_values": df.isnull().sum().sum(),
            "infinite_values": np.isinf(df.select_dtypes(include=['number'])).sum().sum(),
            "date_range": {
                "start": str(df["timestamp"].min()),
                "end": str(df["timestamp"].max()),
            },
            "issues": [],
        }
        
        # Check for issues
        if validation["missing_values"] > 0:
            validation["issues"].append(
                f"Found {validation['missing_values']} missing values"
            )
        
        if validation["infinite_values"] > 0:
            validation["issues"].append(
                f"Found {validation['infinite_values']} infinite values"
            )
        
        if len(df) < self.config["min_required_rows"]:
            validation["issues"].append(
                f"Only {len(df)} rows, minimum required: {self.config['min_required_rows']}"
            )
        
        validation["passed"] = len(validation["issues"]) == 0
        
        return validation


def main() -> None:
    """Main entry point for testing feature pipeline."""
    from aurummind_ai.utils.logging_config import setup_logging
    
    setup_logging(level="INFO")
    
    logger.info("Feature Pipeline - Test Run")
    logger.info("=" * 70)
    
    # Example usage
    print("\nFeature pipeline ready for use.")
    print("\nExample usage:")
    print("""
    from aurummind_ai.features import FeaturePipeline
    import pandas as pd
    
    # Load data
    df = pd.read_csv('data/raw/samples/XAUUSD_1h_sample.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Initialize pipeline
    pipeline = FeaturePipeline()
    
    # Generate features
    df_features = pipeline.generate_features(df)
    
    # Save features
    pipeline.save_features(df_features, 'data/processed', '1h')
    
    # Get feature info
    info = pipeline.get_feature_info()
    print(info)
    """)


if __name__ == "__main__":
    main()
