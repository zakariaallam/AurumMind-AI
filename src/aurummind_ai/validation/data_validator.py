"""Data validation functionality for OHLCV data."""

import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ..ingestion.schema import DataSourceMetadata
from .quality_report import DataQualityReport

logger = logging.getLogger(__name__)


class DataValidator:
    """Comprehensive validation of OHLCV market data.
    
    Performs validation checks including:
    - Schema validation
    - Duplicate timestamps
    - Missing values
    - Invalid OHLC relationships
    - Suspicious prices and outliers
    - Missing candles detection
    - Data quality scoring
    """
    
    def __init__(
        self,
        price_jump_threshold: float = 0.10,
        volume_outlier_threshold: float = 5.0,
    ):
        """Initialize DataValidator.
        
        Args:
            price_jump_threshold: Maximum allowed price change ratio (default 10%)
            volume_outlier_threshold: Z-score threshold for volume outliers
        """
        self.price_jump_threshold = price_jump_threshold
        self.volume_outlier_threshold = volume_outlier_threshold
    
    def validate(
        self,
        df: pd.DataFrame,
        metadata: DataSourceMetadata,
    ) -> DataQualityReport:
        """Perform comprehensive validation on OHLCV data.
        
        Args:
            df: DataFrame containing OHLCV data
            metadata: Metadata about the data source
            
        Returns:
            DataQualityReport with all validation results
        """
        logger.info(f"Starting validation for {metadata.instrument} {metadata.timeframe}")
        
        report = DataQualityReport(
            report_timestamp=datetime.now(timezone.utc),
            instrument=metadata.instrument,
            timeframe=metadata.timeframe,
            data_source=metadata.source_name,
            total_records=len(df),
        )
        
        if len(df) == 0:
            report.add_issue("ERROR", "Empty Dataset", "No records found in dataset")
            report.calculate_quality_score()
            return report
        
        # Set date range
        report.date_range_start = df["timestamp"].min()
        report.date_range_end = df["timestamp"].max()
        
        # Run all validation checks
        self._check_duplicates(df, report)
        self._check_missing_values(df, report)
        self._check_ohlc_relationships(df, report)
        self._check_negative_or_zero_prices(df, report)
        self._check_price_jumps(df, report)
        self._check_missing_candles(df, metadata.timeframe, report)
        self._calculate_statistics(df, report)
        
        # Calculate overall quality score
        report.calculate_quality_score()
        
        # Determine if validation passed
        report.passed_validation = not report.has_errors()
        
        logger.info(f"Validation complete. Quality score: {report.quality_score:.2f}")
        
        return report
    
    def _check_duplicates(self, df: pd.DataFrame, report: DataQualityReport) -> None:
        """Check for duplicate timestamps."""
        duplicates = df["timestamp"].duplicated()
        duplicate_count = duplicates.sum()
        
        if duplicate_count > 0:
            report.duplicate_timestamps = duplicate_count
            examples = df[duplicates]["timestamp"].head(5).tolist()
            report.add_issue(
                severity="ERROR",
                category="Duplicate Timestamps",
                message=f"Found {duplicate_count} duplicate timestamps",
                count=duplicate_count,
                examples=[str(ts) for ts in examples],
            )
    
    def _check_missing_values(self, df: pd.DataFrame, report: DataQualityReport) -> None:
        """Check for missing values in any column."""
        missing = df.isnull().sum()
        
        for column, count in missing.items():
            if count > 0:
                report.missing_values[column] = int(count)
                report.add_issue(
                    severity="ERROR",
                    category="Missing Values",
                    message=f"Column '{column}' has {count} missing values",
                    count=int(count),
                )
    
    def _check_ohlc_relationships(
        self, df: pd.DataFrame, report: DataQualityReport
    ) -> None:
        """Validate OHLC relationships: High >= Open, Close, Low and Low <= Open, Close."""
        # High should be >= Open, Close, and Low
        invalid_high = (
            (df["high"] < df["open"])
            | (df["high"] < df["close"])
            | (df["high"] < df["low"])
        )
        
        # Low should be <= Open, Close, and High
        invalid_low = (
            (df["low"] > df["open"])
            | (df["low"] > df["close"])
            | (df["low"] > df["high"])
        )
        
        invalid_ohlc = invalid_high | invalid_low
        invalid_count = invalid_ohlc.sum()
        
        if invalid_count > 0:
            report.invalid_ohlc_relationships = invalid_count
            
            # Get examples
            examples = df[invalid_ohlc][["timestamp", "open", "high", "low", "close"]].head(
                3
            )
            example_list = examples.to_dict(orient="records")
            
            report.add_issue(
                severity="ERROR",
                category="Invalid OHLC",
                message=f"Found {invalid_count} candles with invalid OHLC relationships",
                count=invalid_count,
                examples=example_list,
            )
    
    def _check_negative_or_zero_prices(
        self, df: pd.DataFrame, report: DataQualityReport
    ) -> None:
        """Check for negative or zero prices."""
        price_columns = ["open", "high", "low", "close"]
        
        for col in price_columns:
            negative = df[col] < 0
            zero = df[col] == 0
            
            negative_count = negative.sum()
            zero_count = zero.sum()
            
            if negative_count > 0:
                report.negative_prices += negative_count
                report.add_issue(
                    severity="ERROR",
                    category="Negative Prices",
                    message=f"Column '{col}' has {negative_count} negative values",
                    count=negative_count,
                )
            
            if zero_count > 0:
                report.zero_prices += zero_count
                report.add_issue(
                    severity="WARNING",
                    category="Zero Prices",
                    message=f"Column '{col}' has {zero_count} zero values",
                    count=zero_count,
                )
    
    def _check_price_jumps(self, df: pd.DataFrame, report: DataQualityReport) -> None:
        """Detect suspicious price jumps between consecutive candles."""
        if len(df) < 2:
            return
        
        # Calculate price change ratio
        price_change = df["close"].pct_change().abs()
        
        # Find jumps exceeding threshold
        suspicious = price_change > self.price_jump_threshold
        suspicious_count = suspicious.sum()
        
        if suspicious_count > 0:
            report.suspicious_price_jumps = suspicious_count
            
            # Get examples with context
            suspicious_indices = df[suspicious].index.tolist()[:3]
            examples = []
            
            for idx in suspicious_indices:
                if idx > 0:
                    prev_close = df.loc[idx - 1, "close"]
                    curr_close = df.loc[idx, "close"]
                    change_pct = ((curr_close - prev_close) / prev_close) * 100
                    
                    examples.append({
                        "timestamp": str(df.loc[idx, "timestamp"]),
                        "prev_close": float(prev_close),
                        "curr_close": float(curr_close),
                        "change_pct": float(change_pct),
                    })
            
            report.add_issue(
                severity="WARNING",
                category="Price Jumps",
                message=(
                    f"Found {suspicious_count} suspicious price jumps "
                    f"exceeding {self.price_jump_threshold * 100:.1f}%"
                ),
                count=suspicious_count,
                examples=examples,
            )
    
    def _check_missing_candles(
        self, df: pd.DataFrame, timeframe: str, report: DataQualityReport
    ) -> None:
        """Detect missing candles based on expected timeframe frequency."""
        if len(df) < 2:
            return
        
        # Map timeframe to pandas frequency
        freq_map = {
            "1h": "1H",
            "4h": "4H",
            "1d": "1D",
        }
        
        freq = freq_map.get(timeframe)
        if not freq:
            logger.warning(f"Unknown timeframe: {timeframe}. Skipping missing candles check.")
            return
        
        # Create expected date range
        expected_range = pd.date_range(
            start=df["timestamp"].min(),
            end=df["timestamp"].max(),
            freq=freq,
        )
        
        # Find missing timestamps
        actual_timestamps = set(df["timestamp"])
        expected_timestamps = set(expected_range)
        missing = expected_timestamps - actual_timestamps
        
        missing_count = len(missing)
        
        if missing_count > 0:
            report.missing_candles = missing_count
            
            # Get some examples
            missing_sorted = sorted(missing)[:5]
            
            report.add_issue(
                severity="WARNING",
                category="Missing Candles",
                message=(
                    f"Found {missing_count} missing candles in the time series. "
                    "This may indicate market gaps or data collection issues."
                ),
                count=missing_count,
                examples=[str(ts) for ts in missing_sorted],
            )
    
    def _calculate_statistics(
        self, df: pd.DataFrame, report: DataQualityReport
    ) -> None:
        """Calculate descriptive statistics for prices and volume."""
        price_columns = ["open", "high", "low", "close"]
        
        for col in price_columns:
            report.price_statistics[f"{col}_mean"] = float(df[col].mean())
            report.price_statistics[f"{col}_median"] = float(df[col].median())
            report.price_statistics[f"{col}_std"] = float(df[col].std())
            report.price_statistics[f"{col}_min"] = float(df[col].min())
            report.price_statistics[f"{col}_max"] = float(df[col].max())
        
        # Volume statistics
        report.volume_statistics["mean"] = float(df["volume"].mean())
        report.volume_statistics["median"] = float(df["volume"].median())
        report.volume_statistics["std"] = float(df["volume"].std())
        report.volume_statistics["min"] = float(df["volume"].min())
        report.volume_statistics["max"] = float(df["volume"].max())
        
        # Check for zero volume
        zero_volume = (df["volume"] == 0).sum()
        if zero_volume > 0:
            report.add_issue(
                severity="INFO",
                category="Volume",
                message=(
                    f"Found {zero_volume} candles with zero volume. "
                    "This may be normal for spot markets using tick volume."
                ),
                count=zero_volume,
            )
    
    def save_report(self, report: DataQualityReport, output_path: str | Path) -> None:
        """Save data quality report to JSON file.
        
        Args:
            report: Data quality report to save
            output_path: Path where to save the report
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w") as f:
            f.write(report.model_dump_json(indent=2))
        
        logger.info(f"Saved quality report to {output_path}")
