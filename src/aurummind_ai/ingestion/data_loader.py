"""Data loading functionality for CSV and Parquet files."""

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import pandas as pd
import pyarrow.parquet as pq
import yaml

from .schema import DataSourceMetadata

logger = logging.getLogger(__name__)


class DataLoader:
    """Load and normalize OHLCV data from various file formats.
    
    Supports CSV and Parquet formats with automatic schema validation
    and timestamp normalization to UTC.
    """
    
    def __init__(self, config_path: str = "configs/data.yaml"):
        """Initialize DataLoader with configuration.
        
        Args:
            config_path: Path to the data configuration YAML file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.raw_dir = Path(self.config["data"]["raw_dir"])
        self.interim_dir = Path(self.config["data"]["interim_dir"])
        
        # Create directories if they don't exist
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.interim_dir.mkdir(parents=True, exist_ok=True)
    
    def _load_config(self) -> dict:
        """Load configuration from YAML file."""
        with open(self.config_path, "r") as f:
            return yaml.safe_load(f)
    
    def load_csv(
        self,
        file_path: str,
        timeframe: Literal["1h", "4h", "1d"],
        source_name: str = "Unknown",
    ) -> tuple[pd.DataFrame, DataSourceMetadata]:
        """Load OHLCV data from CSV file.
        
        Args:
            file_path: Path to the CSV file
            timeframe: Timeframe of the data (1h, 4h, or 1d)
            source_name: Name of the data provider
            
        Returns:
            Tuple of (DataFrame, metadata)
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If required columns are missing
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        logger.info(f"Loading CSV file: {file_path}")
        
        # Load CSV with automatic type inference
        df = pd.read_csv(file_path)
        
        # Normalize column names to lowercase
        df.columns = df.columns.str.lower().str.strip()
        
        # Validate required columns
        required_columns = ["timestamp", "open", "high", "low", "close", "volume"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Keep only required columns
        df = df[required_columns].copy()
        
        # Normalize timestamp to UTC
        df = self._normalize_timestamps(df)
        
        # Sort by timestamp
        df = df.sort_values("timestamp").reset_index(drop=True)
        
        # Create metadata
        metadata = DataSourceMetadata(
            source_name=source_name,
            instrument=self.config["project"]["instrument"],
            timeframe=timeframe,
            ingestion_timestamp=datetime.now(timezone.utc),
            file_path=str(file_path),
            total_records=len(df),
        )
        
        logger.info(f"Loaded {len(df)} records from {file_path}")
        
        return df, metadata
    
    def load_parquet(
        self,
        file_path: str,
        timeframe: Literal["1h", "4h", "1d"],
        source_name: str = "Unknown",
    ) -> tuple[pd.DataFrame, DataSourceMetadata]:
        """Load OHLCV data from Parquet file.
        
        Args:
            file_path: Path to the Parquet file
            timeframe: Timeframe of the data (1h, 4h, or 1d)
            source_name: Name of the data provider
            
        Returns:
            Tuple of (DataFrame, metadata)
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If required columns are missing
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        logger.info(f"Loading Parquet file: {file_path}")
        
        # Load Parquet file
        table = pq.read_table(file_path)
        df = table.to_pandas()
        
        # Normalize column names
        df.columns = df.columns.str.lower().str.strip()
        
        # Validate required columns
        required_columns = ["timestamp", "open", "high", "low", "close", "volume"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Keep only required columns
        df = df[required_columns].copy()
        
        # Normalize timestamp to UTC
        df = self._normalize_timestamps(df)
        
        # Sort by timestamp
        df = df.sort_values("timestamp").reset_index(drop=True)
        
        # Create metadata
        metadata = DataSourceMetadata(
            source_name=source_name,
            instrument=self.config["project"]["instrument"],
            timeframe=timeframe,
            ingestion_timestamp=datetime.now(timezone.utc),
            file_path=str(file_path),
            total_records=len(df),
        )
        
        logger.info(f"Loaded {len(df)} records from {file_path}")
        
        return df, metadata
    
    def _normalize_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize timestamps to UTC timezone-aware format.
        
        Args:
            df: DataFrame with timestamp column
            
        Returns:
            DataFrame with normalized timestamps
        """
        # Convert timestamp to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        
        # If timezone-naive, assume UTC
        if df["timestamp"].dt.tz is None:
            df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
        else:
            # Convert to UTC if in different timezone
            df["timestamp"] = df["timestamp"].dt.tz_convert("UTC")
        
        return df
    
    def save_raw_data(
        self, df: pd.DataFrame, metadata: DataSourceMetadata
    ) -> Path:
        """Save raw data to the raw directory as Parquet.
        
        Args:
            df: DataFrame to save
            metadata: Metadata about the data
            
        Returns:
            Path to the saved file
        """
        instrument = metadata.instrument
        timeframe = metadata.timeframe
        timestamp = metadata.ingestion_timestamp.strftime("%Y%m%d_%H%M%S")
        
        filename = f"{instrument}_{timeframe}_{timestamp}_raw.parquet"
        output_path = self.raw_dir / filename
        
        # Save as Parquet
        df.to_parquet(output_path, index=False, engine="pyarrow")
        
        # Save metadata as JSON
        metadata_path = output_path.with_suffix(".json")
        with open(metadata_path, "w") as f:
            f.write(metadata.model_dump_json(indent=2))
        
        logger.info(f"Saved raw data to {output_path}")
        
        return output_path
