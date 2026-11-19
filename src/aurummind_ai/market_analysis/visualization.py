"""Visualization functions for market data using Plotly."""

import logging
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)


class MarketVisualizer:
    """Professional market data visualization using Plotly.
    
    Provides methods for creating interactive charts including candlestick
    charts, volume plots, and statistical visualizations.
    """
    
    @staticmethod
    def plot_candlestick(
        df: pd.DataFrame,
        title: str = "XAUUSD Candlestick Chart",
        show_volume: bool = True,
        height: int = 800,
    ) -> go.Figure:
        """Create interactive candlestick chart.
        
        Args:
            df: DataFrame with OHLCV data
            title: Chart title
            show_volume: Whether to show volume subplot
            height: Chart height in pixels
            
        Returns:
            Plotly Figure object
        """
        if show_volume:
            fig = make_subplots(
                rows=2,
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                row_heights=[0.7, 0.3],
                subplot_titles=(title, "Volume"),
            )
            
            # Candlestick
            fig.add_trace(
                go.Candlestick(
                    x=df["timestamp"],
                    open=df["open"],
                    high=df["high"],
                    low=df["low"],
                    close=df["close"],
                    name="XAUUSD",
                ),
                row=1,
                col=1,
            )
            
            # Volume
            colors = [
                "green" if close >= open_price else "red"
                for close, open_price in zip(df["close"], df["open"])
            ]
            
            fig.add_trace(
                go.Bar(
                    x=df["timestamp"],
                    y=df["volume"],
                    name="Volume",
                    marker_color=colors,
                    showlegend=False,
                ),
                row=2,
                col=1,
            )
            
        else:
            fig = go.Figure(
                data=[
                    go.Candlestick(
                        x=df["timestamp"],
                        open=df["open"],
                        high=df["high"],
                        low=df["low"],
                        close=df["close"],
                        name="XAUUSD",
                    )
                ]
            )
            fig.update_layout(title=title)
        
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            height=height,
            xaxis_rangeslider_visible=False,
            template="plotly_white",
        )
        
        return fig
    
    @staticmethod
    def plot_price_distribution(
        df: pd.DataFrame,
        price_col: str = "close",
        title: str = "Price Distribution",
    ) -> go.Figure:
        """Create price distribution histogram.
        
        Args:
            df: DataFrame with price data
            price_col: Column name for price
            title: Chart title
            
        Returns:
            Plotly Figure object
        """
        fig = go.Figure(
            data=[
                go.Histogram(
                    x=df[price_col],
                    nbinsx=50,
                    name=price_col.capitalize(),
                    marker_color="steelblue",
                )
            ]
        )
        
        fig.update_layout(
            title=title,
            xaxis_title=f"{price_col.capitalize()} Price (USD)",
            yaxis_title="Frequency",
            template="plotly_white",
            showlegend=False,
        )
        
        return fig
    
    @staticmethod
    def plot_returns_distribution(
        df: pd.DataFrame,
        price_col: str = "close",
        title: str = "Returns Distribution",
    ) -> go.Figure:
        """Create returns distribution histogram.
        
        Args:
            df: DataFrame with price data
            price_col: Column name for price
            title: Chart title
            
        Returns:
            Plotly Figure object
        """
        returns = df[price_col].pct_change().dropna()
        
        fig = go.Figure(
            data=[
                go.Histogram(
                    x=returns * 100,  # Convert to percentage
                    nbinsx=50,
                    name="Returns",
                    marker_color="steelblue",
                )
            ]
        )
        
        fig.update_layout(
            title=title,
            xaxis_title="Returns (%)",
            yaxis_title="Frequency",
            template="plotly_white",
            showlegend=False,
        )
        
        # Add vertical line at zero
        fig.add_vline(x=0, line_dash="dash", line_color="red", opacity=0.5)
        
        return fig
    
    @staticmethod
    def plot_volatility(
        df: pd.DataFrame,
        price_col: str = "close",
        window: int = 20,
        title: str = "Rolling Volatility",
    ) -> go.Figure:
        """Plot rolling volatility over time.
        
        Args:
            df: DataFrame with price data
            price_col: Column name for price
            window: Rolling window size
            title: Chart title
            
        Returns:
            Plotly Figure object
        """
        from .statistics import MarketStatistics
        
        volatility = MarketStatistics.calculate_volatility(df, price_col, window)
        
        fig = go.Figure(
            data=[
                go.Scatter(
                    x=df["timestamp"],
                    y=volatility * 100,  # Convert to percentage
                    mode="lines",
                    name=f"{window}-period Volatility",
                    line=dict(color="steelblue", width=2),
                )
            ]
        )
        
        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Volatility (%)",
            template="plotly_white",
            hovermode="x unified",
        )
        
        return fig
    
    @staticmethod
    def plot_price_with_ma(
        df: pd.DataFrame,
        ma_periods: list[int] = [20, 50, 200],
        title: str = "Price with Moving Averages",
    ) -> go.Figure:
        """Plot closing price with moving averages.
        
        Args:
            df: DataFrame with price data
            ma_periods: List of MA periods to plot
            title: Chart title
            
        Returns:
            Plotly Figure object
        """
        fig = go.Figure()
        
        # Close price
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=df["close"],
                mode="lines",
                name="Close",
                line=dict(color="black", width=1),
            )
        )
        
        # Moving averages
        colors = ["blue", "orange", "green", "red", "purple"]
        for i, period in enumerate(ma_periods):
            ma = df["close"].rolling(window=period).mean()
            fig.add_trace(
                go.Scatter(
                    x=df["timestamp"],
                    y=ma,
                    mode="lines",
                    name=f"MA{period}",
                    line=dict(color=colors[i % len(colors)], width=1.5),
                )
            )
        
        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            template="plotly_white",
            hovermode="x unified",
            height=600,
        )
        
        return fig
    
    @staticmethod
    def plot_ohlc_comparison(df: pd.DataFrame, title: str = "OHLC Comparison") -> go.Figure:
        """Plot comparison of OHLC values.
        
        Args:
            df: DataFrame with OHLC data
            title: Chart title
            
        Returns:
            Plotly Figure object
        """
        fig = go.Figure()
        
        for col, color in [
            ("open", "blue"),
            ("high", "green"),
            ("low", "red"),
            ("close", "black"),
        ]:
            fig.add_trace(
                go.Scatter(
                    x=df["timestamp"],
                    y=df[col],
                    mode="lines",
                    name=col.capitalize(),
                    line=dict(color=color, width=1),
                )
            )
        
        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            template="plotly_white",
            hovermode="x unified",
            height=600,
        )
        
        return fig
    
    @staticmethod
    def plot_volume_analysis(
        df: pd.DataFrame,
        title: str = "Volume Analysis",
    ) -> go.Figure:
        """Create volume analysis with statistics.
        
        Args:
            df: DataFrame with volume data
            title: Chart title
            
        Returns:
            Plotly Figure object
        """
        fig = make_subplots(
            rows=2,
            cols=1,
            subplot_titles=("Volume Over Time", "Volume Distribution"),
            vertical_spacing=0.15,
        )
        
        # Volume over time
        fig.add_trace(
            go.Bar(
                x=df["timestamp"],
                y=df["volume"],
                name="Volume",
                marker_color="steelblue",
            ),
            row=1,
            col=1,
        )
        
        # Volume distribution
        fig.add_trace(
            go.Histogram(
                x=df["volume"],
                nbinsx=50,
                name="Distribution",
                marker_color="steelblue",
                showlegend=False,
            ),
            row=2,
            col=1,
        )
        
        fig.update_xaxes(title_text="Date", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=1, col=1)
        fig.update_xaxes(title_text="Volume", row=2, col=1)
        fig.update_yaxes(title_text="Frequency", row=2, col=1)
        
        fig.update_layout(
            title_text=title,
            template="plotly_white",
            height=800,
        )
        
        return fig
    
    @staticmethod
    def plot_intraday_patterns(
        df: pd.DataFrame,
        title: str = "Intraday Patterns",
    ) -> go.Figure:
        """Analyze and plot intraday patterns.
        
        Args:
            df: DataFrame with timestamp and price data
            title: Chart title
            
        Returns:
            Plotly Figure object
        """
        df = df.copy()
        df["hour"] = pd.to_datetime(df["timestamp"]).dt.hour
        
        # Calculate average returns by hour
        df["returns"] = df["close"].pct_change()
        hourly_returns = df.groupby("hour")["returns"].mean() * 100
        
        fig = go.Figure(
            data=[
                go.Bar(
                    x=hourly_returns.index,
                    y=hourly_returns.values,
                    marker_color=[
                        "green" if x > 0 else "red" for x in hourly_returns.values
                    ],
                )
            ]
        )
        
        fig.update_layout(
            title=title,
            xaxis_title="Hour of Day (UTC)",
            yaxis_title="Average Returns (%)",
            template="plotly_white",
            showlegend=False,
        )
        
        return fig
