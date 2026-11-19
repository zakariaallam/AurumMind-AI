"""OHLCV data schema definitions using Pydantic."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class OHLCVSchema(BaseModel):
    """Schema for OHLCV (Open, High, Low, Close, Volume) market data.
    
    Attributes:
        timestamp: UTC timestamp of the candle
        open: Opening price
        high: Highest price during the period
        low: Lowest price during the period
        close: Closing price
        volume: Trading volume (may represent tick volume for spot markets)
    """
    
    timestamp: datetime = Field(..., description="UTC timestamp")
    open: float = Field(..., gt=0, description="Opening price")
    high: float = Field(..., gt=0, description="Highest price")
    low: float = Field(..., gt=0, description="Lowest price")
    close: float = Field(..., gt=0, description="Closing price")
    volume: float = Field(..., ge=0, description="Trading volume")
    
    @field_validator("timestamp")
    @classmethod
    def validate_timestamp(cls, v: datetime) -> datetime:
        """Ensure timestamp is timezone-aware UTC."""
        if v.tzinfo is None:
            raise ValueError("Timestamp must be timezone-aware (UTC)")
        return v
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "timestamp": "2024-01-01T00:00:00+00:00",
                    "open": 2063.45,
                    "high": 2065.20,
                    "low": 2062.10,
                    "close": 2064.80,
                    "volume": 1523.5,
                }
            ]
        }
    }


class DataSourceMetadata(BaseModel):
    """Metadata about the data source.
    
    Attributes:
        source_name: Name of the data provider
        instrument: Trading instrument symbol
        timeframe: Candle timeframe
        ingestion_timestamp: When the data was ingested
        file_path: Original file path
        total_records: Number of records ingested
    """
    
    source_name: str
    instrument: str
    timeframe: Literal["1h", "4h", "1d"]
    ingestion_timestamp: datetime
    file_path: str
    total_records: int
