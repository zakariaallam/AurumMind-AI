"""Technical indicators for financial time series.

This module implements common technical analysis indicators
used in market analysis and algorithmic trading.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class TechnicalIndicators:
    """Collection of technical indicators for market analysis.
    
    All indicators are calculated using only historical data available
    at each timestamp to prevent look-ahead bias.
    """
    
    @staticmethod
    def sma(df: pd.DataFrame, column: str = "close", period: int = 20) -> pd.Series:
        """Calculate Simple Moving Average (SMA).
        
        Args:
            df: DataFrame with price data
            column: Column name to calculate SMA on
            period: Number of periods for moving average
            
        Returns:
            Series with SMA values
        """
        return df[column].rolling(window=period, min_periods=period).mean()
    
    @staticmethod
    def ema(df: pd.DataFrame, column: str = "close", period: int = 20) -> pd.Series:
        """Calculate Exponential Moving Average (EMA).
        
        Args:
            df: DataFrame with price data
            column: Column name to calculate EMA on
            period: Number of periods for EMA
            
        Returns:
            Series with EMA values
        """
        return df[column].ewm(span=period, adjust=False, min_periods=period).mean()
    
    @staticmethod
    def rsi(df: pd.DataFrame, column: str = "close", period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index (RSI).
        
        RSI = 100 - (100 / (1 + RS))
        where RS = Average Gain / Average Loss
        
        Args:
            df: DataFrame with price data
            column: Column name to calculate RSI on
            period: Number of periods for RSI calculation
            
        Returns:
            Series with RSI values (0-100)
        """
        delta = df[column].diff()
        
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        
        avg_gain = gain.ewm(span=period, adjust=False, min_periods=period).mean()
        avg_loss = loss.ewm(span=period, adjust=False, min_periods=period).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    @staticmethod
    def macd(
        df: pd.DataFrame,
        column: str = "close",
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD (Moving Average Convergence Divergence).
        
        MACD Line = EMA(fast) - EMA(slow)
        Signal Line = EMA(MACD, signal_period)
        Histogram = MACD Line - Signal Line
        
        Args:
            df: DataFrame with price data
            column: Column name to calculate MACD on
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal line EMA period
            
        Returns:
            Tuple of (MACD line, Signal line, Histogram)
        """
        ema_fast = df[column].ewm(span=fast_period, adjust=False).mean()
        ema_slow = df[column].ewm(span=slow_period, adjust=False).mean()
        
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    @staticmethod
    def bollinger_bands(
        df: pd.DataFrame,
        column: str = "close",
        period: int = 20,
        num_std: float = 2.0,
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands.
        
        Middle Band = SMA(period)
        Upper Band = Middle Band + (std * num_std)
        Lower Band = Middle Band - (std * num_std)
        
        Args:
            df: DataFrame with price data
            column: Column name to calculate bands on
            period: Number of periods for SMA and std
            num_std: Number of standard deviations
            
        Returns:
            Tuple of (upper band, middle band, lower band)
        """
        middle_band = df[column].rolling(window=period, min_periods=period).mean()
        std = df[column].rolling(window=period, min_periods=period).std()
        
        upper_band = middle_band + (std * num_std)
        lower_band = middle_band - (std * num_std)
        
        return upper_band, middle_band, lower_band
    
    @staticmethod
    def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range (ATR).
        
        True Range = max(high - low, abs(high - prev_close), abs(low - prev_close))
        ATR = EMA of True Range
        
        Args:
            df: DataFrame with OHLC data
            period: Number of periods for ATR
            
        Returns:
            Series with ATR values
        """
        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift()).abs()
        low_close = (df["low"] - df["close"].shift()).abs()
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.ewm(span=period, adjust=False, min_periods=period).mean()
        
        return atr
    
    @staticmethod
    def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average Directional Index (ADX).
        
        Measures trend strength (0-100).
        Values above 25 indicate a strong trend.
        
        Args:
            df: DataFrame with OHLC data
            period: Number of periods for ADX
            
        Returns:
            Series with ADX values
        """
        # Calculate directional movement
        high_diff = df["high"].diff()
        low_diff = -df["low"].diff()
        
        pos_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0.0)
        neg_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0.0)
        
        # Calculate ATR
        atr = TechnicalIndicators.atr(df, period)
        
        # Calculate directional indicators
        pos_di = 100 * pos_dm.ewm(span=period, adjust=False).mean() / atr
        neg_di = 100 * neg_dm.ewm(span=period, adjust=False).mean() / atr
        
        # Calculate DX and ADX
        dx = 100 * (pos_di - neg_di).abs() / (pos_di + neg_di)
        adx = dx.ewm(span=period, adjust=False, min_periods=period).mean()
        
        return adx
    
    @staticmethod
    def stochastic(
        df: pd.DataFrame,
        k_period: int = 14,
        d_period: int = 3,
    ) -> tuple[pd.Series, pd.Series]:
        """Calculate Stochastic Oscillator.
        
        %K = (Close - Lowest Low) / (Highest High - Lowest Low) * 100
        %D = SMA of %K
        
        Args:
            df: DataFrame with OHLC data
            k_period: Period for %K calculation
            d_period: Period for %D smoothing
            
        Returns:
            Tuple of (%K, %D)
        """
        lowest_low = df["low"].rolling(window=k_period, min_periods=k_period).min()
        highest_high = df["high"].rolling(window=k_period, min_periods=k_period).max()
        
        k = 100 * (df["close"] - lowest_low) / (highest_high - lowest_low)
        d = k.rolling(window=d_period, min_periods=d_period).mean()
        
        return k, d
    
    @staticmethod
    def roc(df: pd.DataFrame, column: str = "close", period: int = 12) -> pd.Series:
        """Calculate Rate of Change (ROC).
        
        ROC = (Price - Price_n_periods_ago) / Price_n_periods_ago * 100
        
        Args:
            df: DataFrame with price data
            column: Column name to calculate ROC on
            period: Number of periods to look back
            
        Returns:
            Series with ROC values (percentage)
        """
        roc = ((df[column] - df[column].shift(period)) / df[column].shift(period)) * 100
        return roc
    
    @staticmethod
    def obv(df: pd.DataFrame) -> pd.Series:
        """Calculate On-Balance Volume (OBV).
        
        OBV increases by volume on up days, decreases on down days.
        
        Args:
            df: DataFrame with close price and volume
            
        Returns:
            Series with OBV values
        """
        direction = np.sign(df["close"].diff())
        obv = (direction * df["volume"]).cumsum()
        return obv
    
    @staticmethod
    def calculate_all_indicators(
        df: pd.DataFrame,
        sma_periods: Optional[list[int]] = None,
        ema_periods: Optional[list[int]] = None,
        rsi_period: int = 14,
        macd_params: Optional[tuple[int, int, int]] = None,
        bb_period: int = 20,
        atr_period: int = 14,
        adx_period: int = 14,
        stoch_params: Optional[tuple[int, int]] = None,
        roc_period: int = 12,
        include_obv: bool = True,
    ) -> pd.DataFrame:
        """Calculate all technical indicators and add to DataFrame.
        
        Args:
            df: DataFrame with OHLCV data
            sma_periods: List of SMA periods (default: [20, 50, 200])
            ema_periods: List of EMA periods (default: [9, 20, 50, 200])
            rsi_period: RSI period
            macd_params: MACD parameters (fast, slow, signal)
            bb_period: Bollinger Bands period
            atr_period: ATR period
            adx_period: ADX period
            stoch_params: Stochastic parameters (k_period, d_period)
            roc_period: Rate of Change period
            include_obv: Whether to include OBV
            
        Returns:
            DataFrame with all indicators added
        """
        df_with_indicators = df.copy()
        
        # Default periods
        if sma_periods is None:
            sma_periods = [20, 50, 200]
        if ema_periods is None:
            ema_periods = [9, 20, 50, 200]
        if macd_params is None:
            macd_params = (12, 26, 9)
        if stoch_params is None:
            stoch_params = (14, 3)
        
        logger.info("Calculating technical indicators...")
        
        # Moving Averages
        for period in sma_periods:
            df_with_indicators[f"sma_{period}"] = TechnicalIndicators.sma(df, period=period)
        
        for period in ema_periods:
            df_with_indicators[f"ema_{period}"] = TechnicalIndicators.ema(df, period=period)
        
        # RSI
        df_with_indicators["rsi"] = TechnicalIndicators.rsi(df, period=rsi_period)
        
        # MACD
        macd_line, signal_line, histogram = TechnicalIndicators.macd(
            df, fast_period=macd_params[0], slow_period=macd_params[1], 
            signal_period=macd_params[2]
        )
        df_with_indicators["macd"] = macd_line
        df_with_indicators["macd_signal"] = signal_line
        df_with_indicators["macd_histogram"] = histogram
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = TechnicalIndicators.bollinger_bands(
            df, period=bb_period
        )
        df_with_indicators["bb_upper"] = bb_upper
        df_with_indicators["bb_middle"] = bb_middle
        df_with_indicators["bb_lower"] = bb_lower
        
        # ATR
        df_with_indicators["atr"] = TechnicalIndicators.atr(df, period=atr_period)
        
        # ADX
        df_with_indicators["adx"] = TechnicalIndicators.adx(df, period=adx_period)
        
        # Stochastic
        stoch_k, stoch_d = TechnicalIndicators.stochastic(
            df, k_period=stoch_params[0], d_period=stoch_params[1]
        )
        df_with_indicators["stoch_k"] = stoch_k
        df_with_indicators["stoch_d"] = stoch_d
        
        # Rate of Change
        df_with_indicators["roc"] = TechnicalIndicators.roc(df, period=roc_period)
        
        # On-Balance Volume
        if include_obv and "volume" in df.columns:
            df_with_indicators["obv"] = TechnicalIndicators.obv(df)
        
        logger.info(f"✓ Calculated {len(df_with_indicators.columns) - len(df.columns)} indicators")
        
        return df_with_indicators
