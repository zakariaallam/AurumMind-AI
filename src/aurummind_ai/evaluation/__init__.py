"""Model evaluation module for AurumMind AI."""

from .metrics import ModelEvaluator
from .walk_forward import WalkForwardValidator, AnchoredWalkForward
from .regime_analysis import (
    RegimeDetector,
    RegimeEvaluator,
    RegimeTransitionAnalyzer,
    RegimeType,
)
from .backtesting import BacktestEngine, TradeConfig, WalkForwardBacktest
from .performance_attribution import PerformanceAttribution, ErrorAnalysis

__all__ = [
    # Base evaluation
    "ModelEvaluator",
    
    # Walk-forward validation
    "WalkForwardValidator",
    "AnchoredWalkForward",
    
    # Regime analysis
    "RegimeDetector",
    "RegimeEvaluator",
    "RegimeTransitionAnalyzer",
    "RegimeType",
    
    # Backtesting
    "BacktestEngine",
    "TradeConfig",
    "WalkForwardBacktest",
    
    # Performance attribution
    "PerformanceAttribution",
    "ErrorAnalysis",
]
