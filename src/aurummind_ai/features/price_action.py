"""Price action features for market analysis.

This module implements features based on price movement patterns,
market structure, and historical price relationships.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class PriceActionFeatures:
    """Price action and market structure feature engineering.
    
    All features are calculated using only historical data to prevent
    look-ahead bias.
    """
    
    @staticmethod
    def calculate_returns(df: pd.DataFrame, periods: Optional[list[int]] = None) -> pd.DataFrame:
        """Calculate price returns for multiple periods.
        
        Args:
            df: DataFrame with close price
            periods: List of periods for returns (default: [1, 3, 5, 10, 20])
            
        Returns:
            DataFrame with added return columns
        """
        if periods is None:
            periods = [1, 3, 5, 10, 20]
        
        df_returns = df.copy()
        
        for period in periods:
            df_returns[f"return_{period}"] = df["close"].pct_change(periods=period)
        
        return df_returns
    
    @staticmethod
    def calculate_rolling_statistics(
        df: pd.DataFrame,
        column: str = "close",
        windows: Optional[list[int]] = None,
    ) -> pd.DataFrame:
        """Calculate rolling statistical features.
        
        Args:
            df: DataFrame with price data
            column: Column to calculate statistics on
            windows: List of window sizes (default: [5, 10, 20])
            
        Returns:
            DataFrame with added statistical columns
        """
        if windows is None:
            windows = [5, 10, 20]
        
        df_stats = df.copy()
        
        for window in windows:
            # Rolling min and max
            df_stats[f"{column}_rolling_min_{window}"] = (
                df[column].rolling(window=window, min_periods=window).min()
            )
            df_stats[f"{column}_rolling_max_{window}"] = (
                df[column].rolling(window=window, min_periods=window).max()
            )
            
            # Rolling std
            df_stats[f"{column}_rolling_std_{window}"] = (
                df[column].rolling(window=window, min_periods=window).std()
            )
            
            # Distance from rolling high/low
            df_stats[f"distance_from_high_{window}"] = (
                df[column] - df_stats[f"{column}_rolling_max_{window}"]
            ) / df_stats[f"{column}_rolling_max_{window}"]
            
            df_stats[f"distance_from_low_{window}"] = (
                df[column] - df_stats[f"{column}_rolling_min_{window}"]
            ) / df_stats[f"{column}_rolling_min_{window}"]
        
        return df_stats
    
    @staticmethod
    def calculate_price_position(df: pd.DataFrame, windows: Optional[list[int]] = None) -> pd.DataFrame:
        """Calculate where price is positioned within recent range.
        
        Position = (Close - Low) / (High - Low)
        Value of 0 means at low, 1 means at high, 0.5 means middle.
        
        Args:
            df: DataFrame with OHLC data
            windows: List of window sizes (default: [5, 10, 20])
            
        Returns:
            DataFrame with position features
        """
        if windows is None:
            windows = [5, 10, 20]
        
        df_position = df.copy()
        
        for window in windows:
            rolling_high = df["high"].rolling(window=window, min_periods=window).max()
            rolling_low = df["low"].rolling(window=window, min_periods=window).min()
            
            range_size = rolling_high - rolling_low
            df_position[f"price_position_{window}"] = np.where(
                range_size > 0,
                (df["close"] - rolling_low) / range_size,
                0.5  # Middle position if no range
            )
        
        return df_position
    
    @staticmethod
    def calculate_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate momentum-based features.
        
        Args:
            df: DataFrame with close price
            
        Returns:
            DataFrame with momentum features
        """
        df_momentum = df.copy()
        
        # Rate of change over multiple periods
        for period in [3, 5, 10, 20]:
            df_momentum[f"momentum_{period}"] = df["close"].diff(periods=period)
        
        # Acceleration (change in momentum)
        df_momentum["acceleration_5"] = df_momentum["momentum_5"].diff()
        
        # Consecutive up/down days
        returns = df["close"].pct_change()
        df_momentum["up_days"] = (returns > 0).astype(int)
        df_momentum["down_days"] = (returns < 0).astype(int)
        
        # Cumulative consecutive days
        df_momentum["consecutive_up"] = (
            df_momentum["up_days"]
            .groupby((df_momentum["up_days"] != df_momentum["up_days"].shift()).cumsum())
            .cumsum()
        )
        df_momentum["consecutive_down"] = (
            df_momentum["down_days"]
            .groupby((df_momentum["down_days"] != df_momentum["down_days"].shift()).cumsum())
            .cumsum()
        )
        
        return df_momentum
    
    @staticmethod
    def calculate_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volatility-based features.
        
        Args:
            df: DataFrame with OHLC data
            
        Returns:
            DataFrame with volatility features
        """
        df_vol = df.copy()
        
        # Historical volatility (rolling std of returns)
        returns = df["close"].pct_change()
        
        for window in [5, 10, 20]:
            df_vol[f"volatility_{window}"] = (
                returns.rolling(window=window, min_periods=window).std()
            )
        
        # High-Low range as percentage of close
        df_vol["hl_range_pct"] = (df["high"] - df["low"]) / df["close"]
        
        # Average range
        for window in [5, 10, 20]:
            df_vol[f"avg_range_{window}"] = (
                df_vol["hl_range_pct"].rolling(window=window, min_periods=window).mean()
            )
        
        # Range expansion/contraction
        df_vol["range_expansion"] = (
            df_vol["hl_range_pct"] / df_vol["avg_range_5"]
        )
        
        return df_vol
    
    @staticmethod
    def calculate_gap_features(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate gap-related features.
        
        Args:
            df: DataFrame with OHLC data
            
        Returns:
            DataFrame with gap features
        """
        df_gaps = df.copy()
        
        # Gap size (open - previous close)
        df_gaps["gap"] = df["open"] - df["close"].shift(1)
        df_gaps["gap_pct"] = df_gaps["gap"] / df["close"].shift(1)
        
        # Gap direction
        df_gaps["gap_up"] = (df_gaps["gap"] > 0).astype(int)
        df_gaps["gap_down"] = (df_gaps["gap"] < 0).astype(int)
        
        # Gap filled (if low crosses previous close on gap up, or high crosses on gap down)
        prev_close = df["close"].shift(1)
        df_gaps["gap_filled"] = (
            ((df_gaps["gap_up"] == 1) & (df["low"] <= prev_close)) |
            ((df_gaps["gap_down"] == 1) & (df["high"] >= prev_close))
        ).astype(int)
        
        return df_gaps
    
    @staticmethod
    def calculate_higher_highs_lows(df: pd.DataFrame, lookback: int = 5) -> pd.DataFrame:
        """Detect higher highs, higher lows, lower highs, lower lows.
        
        Args:
            df: DataFrame with OHLC data
            lookback: Number of periods to look back
            
        Returns:
            DataFrame with market structure features
        """
        df_structure = df.copy()
        
        # Higher high: current high > previous high
        df_structure["higher_high"] = (
            df["high"] > df["high"].shift(lookback)
        ).astype(int)
        
        # Higher low: current low > previous low
        df_structure["higher_low"] = (
            df["low"] > df["low"].shift(lookback)
        ).astype(int)
        
        # Lower high: current high < previous high
        df_structure["lower_high"] = (
            df["high"] < df["high"].shift(lookback)
        ).astype(int)
        
        # Lower low: current low < previous low
        df_structure["lower_low"] = (
            df["low"] < df["low"].shift(lookback)
        ).astype(int)
        
        # Trend indication
        df_structure["uptrend_signal"] = (
            df_structure["higher_high"] & df_structure["higher_low"]
        ).astype(int)
        
        df_structure["downtrend_signal"] = (
            df_structure["lower_high"] & df_structure["lower_low"]
        ).astype(int)
        
        return df_structure
    
    @staticmethod
    def calculate_support_resistance(
        df: pd.DataFrame,
        window: int = 20,
        num_levels: int = 3,
    ) -> pd.DataFrame:
        """Estimate support and resistance levels.
        
        Args:
            df: DataFrame with OHLC data
            window: Window for identifying levels
            num_levels: Number of support/resistance levels
            
        Returns:
            DataFrame with estimated levels
        """
        df_levels = df.copy()
        
        # Resistance: recent highs
        rolling_high = df["high"].rolling(window=window, min_periods=window).max()
        df_levels["resistance_1"] = rolling_high
        
        # Support: recent lows
        rolling_low = df["low"].rolling(window=window, min_periods=window).min()
        df_levels["support_1"] = rolling_low
        
        # Distance to support/resistance
        df_levels["distance_to_resistance"] = (
            (df_levels["resistance_1"] - df["close"]) / df["close"]
        )
        df_levels["distance_to_support"] = (
            (df["close"] - df_levels["support_1"]) / df["close"]
        )
        
        # Potential breakout
        df_levels["near_resistance"] = (
            df_levels["distance_to_resistance"] < 0.01
        ).astype(int)
        df_levels["near_support"] = (
            df_levels["distance_to_support"] < 0.01
        ).astype(int)
        
        return df_levels
    
    @staticmethod
    def calculate_all_price_action_features(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all price action features.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with all price action features
        """
        logger.info("Calculating price action features...")
        
        df_features = df.copy()
        
        # Returns
        df_features = PriceActionFeatures.calculate_returns(df_features)
        
        # Rolling statistics
        df_features = PriceActionFeatures.calculate_rolling_statistics(df_features)
        
        # Price position
        df_features = PriceActionFeatures.calculate_price_position(df_features)
        
        # Momentum
        df_features = PriceActionFeatures.calculate_momentum_features(df_features)
        
        # Volatility
        df_features = PriceActionFeatures.calculate_volatility_features(df_features)
        
        # Gaps
        df_features = PriceActionFeatures.calculate_gap_features(df_features)
        
        # Market structure
        df_features = PriceActionFeatures.calculate_higher_highs_lows(df_features)
        
        # Support/Resistance
        df_features = PriceActionFeatures.calculate_support_resistance(df_features)
        
        num_new_features = len(df_features.columns) - len(df.columns)
        logger.info(f"✓ Calculated {num_new_features} price action features")
        
        return df_features
