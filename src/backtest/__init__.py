"""Backtesting framework for strategy evaluation."""

from .engine import BacktestEngine, BacktestConfig, BacktestResult
from .data import HistoricalDataLoader, OHLCV
from .metrics import BacktestMetrics, calculate_sharpe, calculate_max_drawdown
from .comparison import StrategyComparison, StrategyComparisonResult
from .optimizer import ParameterOptimizer, OptimizationResult
from .monte_carlo import MonteCarloAnalyzer, MonteCarloResult
from .walk_forward import WalkForwardAnalyzer, WalkForwardConfig, WalkForwardResult

__all__ = [
    # Core
    "BacktestEngine",
    "BacktestConfig",
    "BacktestResult",
    "HistoricalDataLoader",
    "OHLCV",
    "BacktestMetrics",
    "calculate_sharpe",
    "calculate_max_drawdown",
    # Comparison
    "StrategyComparison",
    "StrategyComparisonResult",
    # Optimization
    "ParameterOptimizer",
    "OptimizationResult",
    # Monte Carlo
    "MonteCarloAnalyzer",
    "MonteCarloResult",
    # Walk-Forward
    "WalkForwardAnalyzer",
    "WalkForwardConfig",
    "WalkForwardResult",
]
