"""Transparent single-horizon cost-aware backtesting."""

from typing import Any

import numpy as np
import pandas as pd

from .config import load_config
from .market_data import normalize_ohlcv
from .providers import TIMEFRAME_MINUTES


def backtest_predictions(
    candles: pd.DataFrame,
    predictions: pd.Series,
    timeframe: str = "M15",
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Simulate one-horizon positions with visible spread, fees, and slippage."""
    settings = (config or load_config())["backtest"]
    timeframe = timeframe.upper()
    if timeframe not in TIMEFRAME_MINUTES:
        raise ValueError(f"Unsupported timeframe: {timeframe}")
    data = normalize_ohlcv(candles).sort_values("timestamp").reset_index(drop=True)
    if len(data) != len(predictions):
        raise ValueError("Predictions must align one-to-one with candle rows.")
    forward_return = data["close"].shift(-1) / data["close"] - 1
    next_timestamp = data["timestamp"].shift(-1)
    contiguous = (next_timestamp - data["timestamp"]) == pd.Timedelta(
        minutes=TIMEFRAME_MINUTES[timeframe]
    )
    position = predictions.reset_index(drop=True).map({"UP": 1.0, "DOWN": -1.0, "NEUTRAL": 0.0})
    if position.isna().any():
        raise ValueError("Prediction series must contain only UP/DOWN/NEUTRAL.")
    position_size = float(settings.get("position_size_fraction", 1.0))
    if not 0 < position_size <= float(settings.get("maximum_position_fraction", 1.0)):
        raise ValueError("Position size must be positive and within its configured cap.")
    costs_bps = (
        float(settings["spread_bps"])
        + float(settings["commission_bps"])
        + float(settings["slippage_bps"])
    )
    active = (position != 0) & contiguous & forward_return.notna()
    strategy_returns = pd.Series(0.0, index=data.index)
    strategy_returns.loc[active] = position_size * (
        position.loc[active] * forward_return.loc[active] - costs_bps / 10_000
    )
    equity = pd.concat(
        [pd.Series([1.0]), (1 + strategy_returns).cumprod().reset_index(drop=True)],
        ignore_index=True,
    )
    running_peak = equity.cummax()
    drawdown = equity / running_peak - 1
    trade_returns = strategy_returns.loc[active]
    wins = trade_returns[trade_returns > 0]
    losses = trade_returns[trade_returns < 0]
    gross_profit = float(wins.sum())
    gross_loss = float(-losses.sum())
    candles_per_year = 252 * 24 * 60 / TIMEFRAME_MINUTES[timeframe]
    sharpe = (
        float(strategy_returns.mean() / strategy_returns.std(ddof=1) * np.sqrt(candles_per_year))
        if strategy_returns.std(ddof=1) > 0
        else 0.0
    )
    return {
        "simulation_only": True,
        "cost_assumptions_bps": {
            "spread": float(settings["spread_bps"]),
            "commission": float(settings["commission_bps"]),
            "slippage": float(settings["slippage_bps"]),
        },
        "number_of_trades": int(active.sum()),
        "position_size_fraction": position_size,
        "net_return": float(equity.iloc[-1] - 1) if len(equity) else 0.0,
        "profit_factor": (gross_profit / gross_loss if gross_loss > 0 else None),
        "win_rate": float((trade_returns > 0).mean()) if len(trade_returns) else 0.0,
        "maximum_drawdown": float(drawdown.min()) if len(drawdown) else 0.0,
        "sharpe_ratio": sharpe,
        "average_win": float(wins.mean()) if len(wins) else 0.0,
        "average_loss": float(losses.mean()) if len(losses) else 0.0,
        "expectancy": float(trade_returns.mean()) if len(trade_returns) else 0.0,
        "backtest_disclaimer": (
            "Historical simulation only. Results are not broker-execution results "
            "and do not imply future performance."
        ),
    }
