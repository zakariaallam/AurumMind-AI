"""Statistical analysis functions for market data."""

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class MarketStatistics:
    """Comprehensive statistical analysis for OHLCV market data.
    
    Provides methods for calculating returns, volatility, and other
    statistical measures relevant to financial time series.
    """
    
    @staticmethod
    def calculate_returns(
        df: pd.DataFrame,
        price_col: str = "close",
        method: str = "simple",
    ) -> pd.Series:
        """Calculate price returns.
        
        Args:
            df: DataFrame with price data
            price_col: Column name for price (default: close)
            method: 'simple' or 'log' returns
            
        Returns:
            Series of returns
        """
        if method == "simple":
            returns = df[price_col].pct_change()
        elif method == "log":
            returns = np.log(df[price_col] / df[price_col].shift(1))
        else:
            raise ValueError(f"Unknown method: {method}. Use 'simple' or 'log'")
        
        return returns
    
    @staticmethod
    def calculate_volatility(
        df: pd.DataFrame,
        price_col: str = "close",
        window: int = 20,
        annualize: bool = False,
    ) -> pd.Series:
        """Calculate rolling volatility.
        
        Args:
            df: DataFrame with price data
            price_col: Column name for price
            window: Rolling window size
            annualize: If True, annualize volatility (assumes 252 trading days)
            
        Returns:
            Series of rolling volatility
        """
        returns = MarketStatistics.calculate_returns(df, price_col, method="log")
        volatility = returns.rolling(window=window).std()
        
        if annualize:
            # Annualize assuming 252 trading days and 24 hours/day for forex
            periods_per_year = 252 * 24  # For hourly data
            volatility = volatility * np.sqrt(periods_per_year)
        
        return volatility
    
    @staticmethod
    def calculate_price_statistics(df: pd.DataFrame) -> dict[str, Any]:
        """Calculate comprehensive price statistics.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Dictionary with price statistics
        """
        stats = {}
        
        for col in ["open", "high", "low", "close"]:
            if col in df.columns:
                stats[col] = {
                    "mean": float(df[col].mean()),
                    "median": float(df[col].median()),
                    "std": float(df[col].std()),
                    "min": float(df[col].min()),
                    "max": float(df[col].max()),
                    "q25": float(df[col].quantile(0.25)),
                    "q75": float(df[col].quantile(0.75)),
                    "skew": float(df[col].skew()),
                    "kurtosis": float(df[col].kurtosis()),
                }
        
        return stats
    
    @staticmethod
    def calculate_return_statistics(
        df: pd.DataFrame,
        price_col: str = "close",
    ) -> dict[str, float]:
        """Calculate return distribution statistics.
        
        Args:
            df: DataFrame with price data
            price_col: Column name for price
            
        Returns:
            Dictionary with return statistics
        """
        returns = MarketStatistics.calculate_returns(df, price_col, method="simple")
        returns = returns.dropna()
        
        stats = {
            "mean": float(returns.mean()),
            "median": float(returns.median()),
            "std": float(returns.std()),
            "min": float(returns.min()),
            "max": float(returns.max()),
            "skew": float(returns.skew()),
            "kurtosis": float(returns.kurtosis()),
            "positive_returns_pct": float((returns > 0).sum() / len(returns) * 100),
            "negative_returns_pct": float((returns < 0).sum() / len(returns) * 100),
        }
        
        return stats
    
    @staticmethod
    def calculate_candle_features(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate candle-based features.
        
        Args:
            df: DataFrame with OHLC data
            
        Returns:
            DataFrame with added candle features
        """
        df = df.copy()
        
        # Candle body
        df["body"] = df["close"] - df["open"]
        df["body_abs"] = df["body"].abs()
        
        # Candle wicks
        df["upper_wick"] = df["high"] - df[["open", "close"]].max(axis=1)
        df["lower_wick"] = df[["open", "close"]].min(axis=1) - df["low"]
        
        # Candle range
        df["candle_range"] = df["high"] - df["low"]
        
        # Body ratio (body size relative to range)
        df["body_ratio"] = df["body_abs"] / df["candle_range"].replace(0, np.nan)
        
        # Candle direction
        df["is_bullish"] = (df["close"] > df["open"]).astype(int)
        
        return df
    
    @staticmethod
    def detect_outliers(
        series: pd.Series,
        method: str = "iqr",
        threshold: float = 1.5,
    ) -> pd.Series:
        """Detect outliers in a series.
        
        Args:
            series: Data series
            method: 'iqr' or 'zscore'
            threshold: Threshold for outlier detection
            
        Returns:
            Boolean series indicating outliers
        """
        if method == "iqr":
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr
            outliers = (series < lower_bound) | (series > upper_bound)
            
        elif method == "zscore":
            z_scores = np.abs((series - series.mean()) / series.std())
            outliers = z_scores > threshold
            
        else:
            raise ValueError(f"Unknown method: {method}. Use 'iqr' or 'zscore'")
        
        return outliers
    
    @staticmethod
    def calculate_summary_report(df: pd.DataFrame) -> dict[str, Any]:
        """Generate comprehensive summary report.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Dictionary with comprehensive statistics
        """
        report = {
            "general": {
                "total_records": len(df),
                "date_range": {
                    "start": str(df["timestamp"].min()) if "timestamp" in df.columns else None,
                    "end": str(df["timestamp"].max()) if "timestamp" in df.columns else None,
                    "duration_days": (
                        (df["timestamp"].max() - df["timestamp"].min()).days
                        if "timestamp" in df.columns
                        else None
                    ),
                },
            },
            "price_statistics": MarketStatistics.calculate_price_statistics(df),
            "return_statistics": MarketStatistics.calculate_return_statistics(df),
        }
        
        # Volume statistics
        if "volume" in df.columns:
            report["volume_statistics"] = {
                "mean": float(df["volume"].mean()),
                "median": float(df["volume"].median()),
                "std": float(df["volume"].std()),
                "min": float(df["volume"].min()),
                "max": float(df["volume"].max()),
                "zero_volume_pct": float((df["volume"] == 0).sum() / len(df) * 100),
            }
        
        # Candle analysis
        df_with_features = MarketStatistics.calculate_candle_features(df)
        report["candle_statistics"] = {
            "avg_body": float(df_with_features["body_abs"].mean()),
            "avg_upper_wick": float(df_with_features["upper_wick"].mean()),
            "avg_lower_wick": float(df_with_features["lower_wick"].mean()),
            "avg_range": float(df_with_features["candle_range"].mean()),
            "bullish_pct": float(df_with_features["is_bullish"].mean() * 100),
        }
        
        return report
