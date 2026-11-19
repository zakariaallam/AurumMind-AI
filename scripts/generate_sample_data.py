"""Generate sample XAUUSD data for testing the ingestion pipeline.

This script creates synthetic but realistic OHLCV data for Gold (XAUUSD).
DO NOT use this data for real trading or research.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path


def generate_ohlcv_candles(
    start_date: str,
    num_candles: int,
    timeframe_hours: int,
    initial_price: float = 2050.0,
    volatility: float = 0.002,
) -> pd.DataFrame:
    """Generate synthetic OHLCV data.
    
    Args:
        start_date: Starting date (YYYY-MM-DD)
        num_candles: Number of candles to generate
        timeframe_hours: Hours per candle (1, 4, or 24)
        initial_price: Starting price
        volatility: Price volatility (standard deviation)
        
    Returns:
        DataFrame with OHLCV data
    """
    np.random.seed(42)  # For reproducibility
    
    # Generate timestamps
    start = pd.to_datetime(start_date, utc=True)
    timestamps = [start + timedelta(hours=i * timeframe_hours) for i in range(num_candles)]
    
    # Generate price series with random walk
    returns = np.random.normal(0, volatility, num_candles)
    prices = initial_price * np.exp(np.cumsum(returns))
    
    data = []
    
    for i, (ts, price) in enumerate(zip(timestamps, prices)):
        # Generate OHLC with realistic intra-candle variation
        candle_volatility = abs(np.random.normal(0, volatility * 0.5))
        
        open_price = price if i == 0 else data[-1]["close"]
        close_price = price
        
        # High and Low around open and close
        high_price = max(open_price, close_price) * (1 + candle_volatility)
        low_price = min(open_price, close_price) * (1 - candle_volatility)
        
        # Volume with random variation
        base_volume = 1000
        volume = base_volume + np.random.normal(0, base_volume * 0.3)
        volume = max(0, volume)  # Ensure non-negative
        
        data.append({
            "timestamp": ts,
            "open": round(open_price, 2),
            "high": round(high_price, 2),
            "low": round(low_price, 2),
            "close": round(close_price, 2),
            "volume": round(volume, 1),
        })
    
    return pd.DataFrame(data)


def main():
    """Generate sample datasets for different timeframes."""
    output_dir = Path("data/raw/samples")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Generating sample XAUUSD data...")
    print("=" * 60)
    
    # Generate 1H data (30 days)
    df_1h = generate_ohlcv_candles(
        start_date="2024-01-01",
        num_candles=720,  # 30 days * 24 hours
        timeframe_hours=1,
        initial_price=2050.0,
        volatility=0.002,
    )
    
    path_1h = output_dir / "XAUUSD_1h_sample.csv"
    df_1h.to_csv(path_1h, index=False)
    print(f"✓ Generated 1H data: {path_1h}")
    print(f"  Records: {len(df_1h)}")
    print(f"  Date range: {df_1h['timestamp'].min()} to {df_1h['timestamp'].max()}")
    print()
    
    # Generate 4H data (90 days)
    df_4h = generate_ohlcv_candles(
        start_date="2024-01-01",
        num_candles=540,  # 90 days * 6 candles/day
        timeframe_hours=4,
        initial_price=2050.0,
        volatility=0.003,
    )
    
    path_4h = output_dir / "XAUUSD_4h_sample.csv"
    df_4h.to_csv(path_4h, index=False)
    print(f"✓ Generated 4H data: {path_4h}")
    print(f"  Records: {len(df_4h)}")
    print(f"  Date range: {df_4h['timestamp'].min()} to {df_4h['timestamp'].max()}")
    print()
    
    # Generate 1D data (1 year)
    df_1d = generate_ohlcv_candles(
        start_date="2024-01-01",
        num_candles=365,
        timeframe_hours=24,
        initial_price=2050.0,
        volatility=0.005,
    )
    
    path_1d = output_dir / "XAUUSD_1d_sample.csv"
    df_1d.to_csv(path_1d, index=False)
    print(f"✓ Generated 1D data: {path_1d}")
    print(f"  Records: {len(df_1d)}")
    print(f"  Date range: {df_1d['timestamp'].min()} to {df_1d['timestamp'].max()}")
    print()
    
    # Generate a dataset with some quality issues for testing
    df_issues = df_1h.copy().iloc[:100]
    
    # Add duplicate timestamp
    df_issues = pd.concat([df_issues, df_issues.iloc[50:51]], ignore_index=True)
    
    # Add missing value
    df_issues.loc[10, "volume"] = None
    
    # Add invalid OHLC (high < low)
    df_issues.loc[20, "high"] = 2000.0
    df_issues.loc[20, "low"] = 2100.0
    
    path_issues = output_dir / "XAUUSD_1h_with_issues.csv"
    df_issues.to_csv(path_issues, index=False)
    print(f"✓ Generated data with issues: {path_issues}")
    print(f"  (For testing validation logic)")
    print()
    
    print("=" * 60)
    print("Sample data generation complete!")
    print()
    print("To test the ingestion pipeline, run:")
    print("  python scripts/test_ingestion.py")


if __name__ == "__main__":
    main()
