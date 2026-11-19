"""Backtesting framework for trading strategies.

This module provides realistic backtesting with:
- Transaction costs
- Slippage simulation
- Position sizing
- Risk management
- Performance metrics
"""

import logging
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class TradeConfig:
    """Configuration for trade execution."""
    transaction_cost: float = 0.0001  # 0.01% per trade
    slippage: float = 0.0001          # 0.01% slippage
    position_size: float = 1.0        # Position size (fraction of capital)
    max_position: float = 1.0         # Maximum position size
    min_position: float = 0.0         # Minimum position size


@dataclass
class Trade:
    """Represents a single trade."""
    timestamp: pd.Timestamp
    direction: str  # 'long' or 'short'
    entry_price: float
    exit_price: Optional[float] = None
    exit_timestamp: Optional[pd.Timestamp] = None
    position_size: float = 1.0
    pnl: Optional[float] = None
    return_pct: Optional[float] = None


class BacktestEngine:
    """Backtesting engine for trading strategies.
    
    Simulates realistic trading with costs and constraints.
    """
    
    def __init__(
        self,
        initial_capital: float = 10000.0,
        config: Optional[TradeConfig] = None,
    ):
        """Initialize backtest engine.
        
        Args:
            initial_capital: Starting capital
            config: Trade configuration
        """
        self.initial_capital = initial_capital
        self.config = config or TradeConfig()
        
        self.trades: list[Trade] = []
        self.equity_curve: list[float] = []
        self.positions: list[float] = []
        
    def run(
        self,
        df: pd.DataFrame,
        predictions: np.ndarray,
        actual_returns: Optional[np.ndarray] = None,
    ) -> dict[str, Any]:
        """Run backtest on predictions.
        
        Args:
            df: DataFrame with price data (must have 'close' and 'timestamp')
            predictions: Predicted signals (0=short, 1=long or regression values)
            actual_returns: Actual returns (if None, computed from close prices)
            
        Returns:
            Dictionary with backtest results
        """
        logger.info("Starting backtest")
        logger.info(f"  Initial capital: ${self.initial_capital:,.2f}")
        logger.info(f"  Transaction cost: {self.config.transaction_cost*100:.4f}%")
        logger.info(f"  Slippage: {self.config.slippage*100:.4f}%")
        
        # Align data
        n = min(len(df), len(predictions))
        df = df.iloc[:n].copy()
        predictions = predictions[:n]
        
        # Calculate returns if not provided
        if actual_returns is None:
            actual_returns = df['close'].pct_change().fillna(0).values
        else:
            actual_returns = actual_returns[:n]
        
        # Initialize
        capital = self.initial_capital
        position = 0  # 0 = no position, 1 = long, -1 = short
        
        self.trades = []
        self.equity_curve = [capital]
        self.positions = [position]
        
        current_trade: Optional[Trade] = None
        
        for i in range(len(df)):
            timestamp = df.iloc[i]['timestamp'] if 'timestamp' in df.columns else df.index[i]
            price = df.iloc[i]['close']
            actual_return = actual_returns[i]
            
            # Determine signal (binary classification or regression threshold)
            if predictions[i] in [0, 1]:
                # Binary: 1=long, 0=short
                signal = 1 if predictions[i] == 1 else -1
            else:
                # Regression: positive=long, negative=short
                signal = 1 if predictions[i] > 0 else -1
            
            # Check for position change
            if signal != position:
                # Close existing position
                if current_trade is not None:
                    exit_price = price * (1 - self.config.slippage * signal)
                    current_trade.exit_price = exit_price
                    current_trade.exit_timestamp = timestamp
                    
                    # Calculate P&L
                    if current_trade.direction == 'long':
                        trade_return = (exit_price - current_trade.entry_price) / current_trade.entry_price
                    else:  # short
                        trade_return = (current_trade.entry_price - exit_price) / current_trade.entry_price
                    
                    # Apply costs
                    trade_return -= 2 * self.config.transaction_cost  # Entry + exit
                    
                    current_trade.return_pct = trade_return
                    current_trade.pnl = capital * current_trade.position_size * trade_return
                    
                    capital += current_trade.pnl
                    
                    self.trades.append(current_trade)
                    current_trade = None
                
                # Open new position
                if signal != 0:
                    entry_price = price * (1 + self.config.slippage * signal)
                    direction = 'long' if signal == 1 else 'short'
                    
                    current_trade = Trade(
                        timestamp=timestamp,
                        direction=direction,
                        entry_price=entry_price,
                        position_size=self.config.position_size,
                    )
                
                position = signal
            
            # Update equity curve based on current position
            if position != 0 and actual_return != 0:
                position_return = actual_return if position == 1 else -actual_return
                capital *= (1 + position_return * self.config.position_size)
            
            self.equity_curve.append(capital)
            self.positions.append(position)
        
        # Close final position if open
        if current_trade is not None:
            exit_price = df.iloc[-1]['close']
            current_trade.exit_price = exit_price
            current_trade.exit_timestamp = df.iloc[-1]['timestamp'] if 'timestamp' in df.columns else df.index[-1]
            
            if current_trade.direction == 'long':
                trade_return = (exit_price - current_trade.entry_price) / current_trade.entry_price
            else:
                trade_return = (current_trade.entry_price - exit_price) / current_trade.entry_price
            
            trade_return -= 2 * self.config.transaction_cost
            current_trade.return_pct = trade_return
            current_trade.pnl = capital * current_trade.position_size * trade_return
            
            capital += current_trade.pnl
            self.trades.append(current_trade)
        
        logger.info(f"Backtest complete: {len(self.trades)} trades")
        
        return self._calculate_metrics()
    
    def _calculate_metrics(self) -> dict[str, Any]:
        """Calculate backtest performance metrics.
        
        Returns:
            Dictionary with metrics
        """
        if not self.trades:
            return {
                'total_trades': 0,
                'final_capital': self.initial_capital,
                'total_return': 0.0,
            }
        
        # Extract trade data
        trade_returns = [t.return_pct for t in self.trades if t.return_pct is not None]
        trade_pnls = [t.pnl for t in self.trades if t.pnl is not None]
        
        # Calculate equity curve metrics
        equity = np.array(self.equity_curve)
        returns = np.diff(equity) / equity[:-1]
        returns = returns[returns != 0]  # Remove zeros
        
        # Drawdown
        cumulative = equity
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Win rate
        winning_trades = sum(1 for r in trade_returns if r > 0)
        losing_trades = sum(1 for r in trade_returns if r < 0)
        
        # Profit factor
        gross_profit = sum(p for p in trade_pnls if p > 0) if trade_pnls else 0
        gross_loss = abs(sum(p for p in trade_pnls if p < 0)) if trade_pnls else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else None
        
        # Risk metrics
        total_return = (equity[-1] - equity[0]) / equity[0]
        
        metrics = {
            'total_trades': len(self.trades),
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': winning_trades / len(self.trades) if self.trades else 0,
            
            'initial_capital': self.initial_capital,
            'final_capital': equity[-1],
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            
            'avg_trade_return': np.mean(trade_returns) if trade_returns else 0,
            'std_trade_return': np.std(trade_returns) if trade_returns else 0,
            'best_trade': max(trade_returns) if trade_returns else 0,
            'worst_trade': min(trade_returns) if trade_returns else 0,
            
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown * 100,
            
            'profit_factor': profit_factor,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss,
        }
        
        # Sharpe ratio (annualized, assuming daily data)
        if len(returns) > 0 and returns.std() > 0:
            metrics['sharpe_ratio'] = returns.mean() / returns.std() * np.sqrt(252)
        else:
            metrics['sharpe_ratio'] = 0
        
        # Sortino ratio (annualized)
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0:
            downside_std = downside_returns.std()
            if downside_std > 0:
                metrics['sortino_ratio'] = returns.mean() / downside_std * np.sqrt(252)
            else:
                metrics['sortino_ratio'] = 0
        else:
            metrics['sortino_ratio'] = 0
        
        # Calmar ratio (return / max drawdown)
        if abs(max_drawdown) > 0:
            metrics['calmar_ratio'] = total_return / abs(max_drawdown)
        else:
            metrics['calmar_ratio'] = None
        
        return metrics
    
    def get_equity_curve(self) -> pd.Series:
        """Get equity curve as pandas Series.
        
        Returns:
            Series with equity over time
        """
        return pd.Series(self.equity_curve)
    
    def get_trades_dataframe(self) -> pd.DataFrame:
        """Get trades as pandas DataFrame.
        
        Returns:
            DataFrame with trade details
        """
        if not self.trades:
            return pd.DataFrame()
        
        trades_data = []
        for trade in self.trades:
            trades_data.append({
                'timestamp': trade.timestamp,
                'direction': trade.direction,
                'entry_price': trade.entry_price,
                'exit_price': trade.exit_price,
                'exit_timestamp': trade.exit_timestamp,
                'position_size': trade.position_size,
                'pnl': trade.pnl,
                'return_pct': trade.return_pct,
            })
        
        return pd.DataFrame(trades_data)
    
    @staticmethod
    def print_backtest_results(
        metrics: dict[str, Any],
    ) -> None:
        """Print backtest results in readable format.
        
        Args:
            metrics: Metrics dictionary from run()
        """
        print("=" * 70)
        print("Backtest Results")
        print("=" * 70)
        print()
        
        print("Trading Activity:")
        print(f"  Total trades:      {metrics['total_trades']}")
        print(f"  Winning trades:    {metrics['winning_trades']}")
        print(f"  Losing trades:     {metrics['losing_trades']}")
        print(f"  Win rate:          {metrics['win_rate']*100:.2f}%")
        print()
        
        print("Returns:")
        print(f"  Initial capital:   ${metrics['initial_capital']:,.2f}")
        print(f"  Final capital:     ${metrics['final_capital']:,.2f}")
        print(f"  Total return:      {metrics['total_return_pct']:.2f}%")
        print(f"  Avg trade return:  {metrics['avg_trade_return']*100:.2f}%")
        print(f"  Best trade:        {metrics['best_trade']*100:.2f}%")
        print(f"  Worst trade:       {metrics['worst_trade']*100:.2f}%")
        print()
        
        print("Risk Metrics:")
        print(f"  Max drawdown:      {metrics['max_drawdown_pct']:.2f}%")
        print(f"  Sharpe ratio:      {metrics['sharpe_ratio']:.2f}")
        print(f"  Sortino ratio:     {metrics['sortino_ratio']:.2f}")
        
        if metrics['calmar_ratio'] is not None:
            print(f"  Calmar ratio:      {metrics['calmar_ratio']:.2f}")
        
        if metrics['profit_factor'] is not None:
            print(f"  Profit factor:     {metrics['profit_factor']:.2f}")
        
        print()
        
        print("Profit & Loss:")
        print(f"  Gross profit:      ${metrics['gross_profit']:,.2f}")
        print(f"  Gross loss:        ${metrics['gross_loss']:,.2f}")
        print()
        
        print("=" * 70)


class WalkForwardBacktest:
    """Combined walk-forward validation with backtesting.
    
    Performs rolling retraining and realistic backtesting
    in a single integrated workflow.
    """
    
    def __init__(
        self,
        train_window: int,
        test_window: int,
        backtest_config: Optional[TradeConfig] = None,
    ):
        """Initialize walk-forward backtest.
        
        Args:
            train_window: Training window size
            test_window: Test window size
            backtest_config: Backtest configuration
        """
        self.train_window = train_window
        self.test_window = test_window
        self.backtest_config = backtest_config or TradeConfig()
        
        self.results: list[dict] = []
    
    def run(
        self,
        model_factory: callable,
        X: pd.DataFrame,
        y: pd.Series,
        price_df: pd.DataFrame,
    ) -> dict[str, Any]:
        """Run walk-forward backtest.
        
        Args:
            model_factory: Function that returns fresh model instance
            X: Feature matrix
            y: Target series
            price_df: DataFrame with price data for backtesting
            
        Returns:
            Aggregated backtest results
        """
        logger.info("Starting walk-forward backtest")
        
        n = len(X)
        train_start = 0
        train_end = self.train_window
        test_start = train_end
        test_end = min(test_start + self.test_window, n)
        
        self.results = []
        all_trades = []
        
        while test_start < n:
            # Get data
            X_train = X.iloc[train_start:train_end]
            y_train = y.iloc[train_start:train_end]
            X_test = X.iloc[test_start:test_end]
            y_test = y.iloc[test_start:test_end]
            price_test = price_df.iloc[test_start:test_end]
            
            # Train model
            model = model_factory()
            model.fit(X_train, y_train)
            
            # Predict
            y_pred = model.predict(X_test)
            
            # Backtest this window
            engine = BacktestEngine(config=self.backtest_config)
            metrics = engine.run(price_test, y_pred)
            
            self.results.append(metrics)
            all_trades.extend(engine.trades)
            
            # Roll forward
            train_start += self.test_window
            train_end = train_start + self.train_window
            test_start = train_end
            test_end = min(test_start + self.test_window, n)
        
        # Aggregate results
        aggregated = self._aggregate_results()
        aggregated['all_trades'] = all_trades
        
        return aggregated
    
    def _aggregate_results(self) -> dict[str, Any]:
        """Aggregate results across windows."""
        if not self.results:
            return {}
        
        # Average metrics across windows
        aggregated = {
            'num_windows': len(self.results),
        }
        
        numeric_metrics = [
            'total_trades', 'win_rate', 'total_return', 'avg_trade_return',
            'max_drawdown', 'sharpe_ratio', 'sortino_ratio'
        ]
        
        for metric in numeric_metrics:
            values = [r[metric] for r in self.results if metric in r]
            if values:
                aggregated[f'{metric}_mean'] = np.mean(values)
                aggregated[f'{metric}_std'] = np.std(values)
        
        return aggregated
