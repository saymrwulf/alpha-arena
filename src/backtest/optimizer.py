"""Parameter optimization for backtesting strategies."""

import asyncio
import itertools
import logging
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any, Callable

from .data import HistoricalMarket
from .engine import BacktestConfig, BacktestEngine, BacktestResult, BacktestStrategy

logger = logging.getLogger(__name__)


@dataclass
class ParameterSet:
    """A specific combination of parameters."""

    params: dict[str, Any]
    result: BacktestResult | None = None
    objective_value: Decimal = Decimal("0")


@dataclass
class OptimizationResult:
    """Results from parameter optimization."""

    optimization_date: datetime
    param_grid: dict[str, list[Any]]
    objective: str
    total_combinations: int
    completed_combinations: int
    best_params: dict[str, Any]
    best_objective_value: Decimal
    best_result: BacktestResult | None
    all_results: list[ParameterSet]
    sensitivity: dict[str, dict[Any, Decimal]]  # param -> value -> avg objective

    def summary(self) -> str:
        """Generate optimization summary."""
        lines = [
            "",
            "=" * 70,
            "  PARAMETER OPTIMIZATION RESULTS",
            "=" * 70,
            f"  Date: {self.optimization_date.strftime('%Y-%m-%d %H:%M')}",
            f"  Objective: {self.objective}",
            f"  Combinations tested: {self.completed_combinations}/{self.total_combinations}",
            "",
            "  BEST PARAMETERS",
            "-" * 70,
        ]

        for param, value in self.best_params.items():
            lines.append(f"    {param}: {value}")

        lines.append("")
        lines.append(f"  Best {self.objective}: {self.best_objective_value:.4f}")

        if self.best_result:
            lines.append("")
            lines.append("  BEST RESULT METRICS")
            lines.append("-" * 70)
            m = self.best_result.metrics
            lines.append(f"    Total Return: {m.total_return_pct:+.2f}%")
            lines.append(f"    Sharpe Ratio: {m.sharpe_ratio:.2f}")
            lines.append(f"    Max Drawdown: {m.max_drawdown_pct:.2f}%")
            lines.append(f"    Win Rate: {m.win_rate:.1f}%")
            lines.append(f"    Total Trades: {m.total_trades}")

        lines.append("")
        lines.append("  PARAMETER SENSITIVITY")
        lines.append("-" * 70)

        for param, values in self.sensitivity.items():
            lines.append(f"    {param}:")
            for value, avg_obj in sorted(values.items(), key=lambda x: x[1], reverse=True):
                lines.append(f"      {value}: {avg_obj:.4f}")

        lines.append("=" * 70)
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "optimization_date": self.optimization_date.isoformat(),
            "objective": self.objective,
            "total_combinations": self.total_combinations,
            "completed_combinations": self.completed_combinations,
            "best_params": {k: str(v) for k, v in self.best_params.items()},
            "best_objective_value": float(self.best_objective_value),
            "sensitivity": {
                param: {str(k): float(v) for k, v in values.items()}
                for param, values in self.sensitivity.items()
            },
        }


class ParameterOptimizer:
    """
    Grid search optimizer for backtest strategies.

    Usage:
        optimizer = ParameterOptimizer()

        def strategy_factory(buy_threshold, sell_threshold):
            return MyStrategy(buy_threshold=buy_threshold, sell_threshold=sell_threshold)

        result = await optimizer.grid_search(
            strategy_factory=strategy_factory,
            param_grid={
                "buy_threshold": [0.35, 0.40, 0.45],
                "sell_threshold": [0.55, 0.60, 0.65],
            },
            markets=historical_data,
            objective="sharpe_ratio",
        )
        print(result.summary())
    """

    # Objectives where higher is better
    HIGHER_BETTER = {
        "sharpe_ratio", "sortino_ratio", "calmar_ratio",
        "total_return_pct", "win_rate", "profit_factor"
    }

    # Objectives where lower is better
    LOWER_BETTER = {"max_drawdown_pct"}

    def __init__(self, config: BacktestConfig | None = None):
        self.config = config or BacktestConfig()
        self.engine = BacktestEngine(self.config)

    async def grid_search(
        self,
        strategy_factory: Callable[..., BacktestStrategy],
        param_grid: dict[str, list[Any]],
        markets: list[HistoricalMarket],
        objective: str = "sharpe_ratio",
        parallel_workers: int = 4,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> OptimizationResult:
        """
        Perform grid search optimization.

        Args:
            strategy_factory: Function that creates strategy from params
            param_grid: Dictionary of param_name -> list of values to try
            markets: Historical market data
            objective: Metric to optimize (sharpe_ratio, total_return_pct, etc.)
            parallel_workers: Number of concurrent backtests
            progress_callback: Optional callback(completed, total)

        Returns:
            OptimizationResult with best parameters and sensitivity analysis
        """
        optimization_date = datetime.utcnow()

        # Generate all parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        combinations = list(itertools.product(*param_values))
        total_combinations = len(combinations)

        logger.info(f"Starting grid search: {total_combinations} combinations")

        # Create parameter sets
        param_sets = []
        for combo in combinations:
            params = dict(zip(param_names, combo))
            param_sets.append(ParameterSet(params=params))

        # Run backtests
        completed = 0
        semaphore = asyncio.Semaphore(parallel_workers)

        async def run_one(ps: ParameterSet) -> None:
            nonlocal completed
            async with semaphore:
                try:
                    strategy = strategy_factory(**ps.params)
                    result = await self.engine.run(strategy, markets)
                    ps.result = result
                    ps.objective_value = self._get_objective_value(result, objective)
                except Exception as e:
                    logger.error(f"Backtest failed for {ps.params}: {e}")
                    ps.objective_value = Decimal("-999999") if objective in self.HIGHER_BETTER else Decimal("999999")

                completed += 1
                if progress_callback:
                    progress_callback(completed, total_combinations)

        # Run all backtests
        tasks = [run_one(ps) for ps in param_sets]
        await asyncio.gather(*tasks)

        # Find best result
        higher_better = objective in self.HIGHER_BETTER
        if higher_better:
            best_ps = max(param_sets, key=lambda ps: ps.objective_value)
        else:
            best_ps = min(param_sets, key=lambda ps: ps.objective_value)

        # Calculate sensitivity
        sensitivity = self._calculate_sensitivity(param_sets, param_grid, objective)

        return OptimizationResult(
            optimization_date=optimization_date,
            param_grid=param_grid,
            objective=objective,
            total_combinations=total_combinations,
            completed_combinations=completed,
            best_params=best_ps.params,
            best_objective_value=best_ps.objective_value,
            best_result=best_ps.result,
            all_results=param_sets,
            sensitivity=sensitivity,
        )

    async def random_search(
        self,
        strategy_factory: Callable[..., BacktestStrategy],
        param_ranges: dict[str, tuple[Any, Any]],
        markets: list[HistoricalMarket],
        objective: str = "sharpe_ratio",
        n_iterations: int = 50,
        parallel_workers: int = 4,
    ) -> OptimizationResult:
        """
        Perform random search optimization.

        Useful for high-dimensional parameter spaces.

        Args:
            strategy_factory: Function that creates strategy from params
            param_ranges: Dictionary of param_name -> (min_value, max_value)
            markets: Historical market data
            objective: Metric to optimize
            n_iterations: Number of random samples to try
            parallel_workers: Number of concurrent backtests

        Returns:
            OptimizationResult with best parameters found
        """
        import random

        optimization_date = datetime.utcnow()

        # Generate random parameter combinations
        param_sets = []
        param_grid: dict[str, list[Any]] = {k: [] for k in param_ranges}

        for _ in range(n_iterations):
            params = {}
            for param_name, (min_val, max_val) in param_ranges.items():
                if isinstance(min_val, Decimal):
                    value = Decimal(str(random.uniform(float(min_val), float(max_val))))
                    value = value.quantize(Decimal("0.01"))
                elif isinstance(min_val, int):
                    value = random.randint(min_val, max_val)
                else:
                    value = random.uniform(min_val, max_val)
                params[param_name] = value
                param_grid[param_name].append(value)

            param_sets.append(ParameterSet(params=params))

        logger.info(f"Starting random search: {n_iterations} iterations")

        # Run backtests
        completed = 0
        semaphore = asyncio.Semaphore(parallel_workers)

        async def run_one(ps: ParameterSet) -> None:
            nonlocal completed
            async with semaphore:
                try:
                    strategy = strategy_factory(**ps.params)
                    result = await self.engine.run(strategy, markets)
                    ps.result = result
                    ps.objective_value = self._get_objective_value(result, objective)
                except Exception as e:
                    logger.error(f"Backtest failed for {ps.params}: {e}")
                    ps.objective_value = Decimal("-999999") if objective in self.HIGHER_BETTER else Decimal("999999")

                completed += 1

        tasks = [run_one(ps) for ps in param_sets]
        await asyncio.gather(*tasks)

        # Find best
        higher_better = objective in self.HIGHER_BETTER
        if higher_better:
            best_ps = max(param_sets, key=lambda ps: ps.objective_value)
        else:
            best_ps = min(param_sets, key=lambda ps: ps.objective_value)

        sensitivity = self._calculate_sensitivity(param_sets, param_grid, objective)

        return OptimizationResult(
            optimization_date=optimization_date,
            param_grid=param_grid,
            objective=objective,
            total_combinations=n_iterations,
            completed_combinations=completed,
            best_params=best_ps.params,
            best_objective_value=best_ps.objective_value,
            best_result=best_ps.result,
            all_results=param_sets,
            sensitivity=sensitivity,
        )

    def _get_objective_value(self, result: BacktestResult, objective: str) -> Decimal:
        """Extract objective value from backtest result."""
        metrics = result.metrics
        value = getattr(metrics, objective, Decimal("0"))
        return Decimal(str(value))

    def _calculate_sensitivity(
        self,
        param_sets: list[ParameterSet],
        param_grid: dict[str, list[Any]],
        objective: str,
    ) -> dict[str, dict[Any, Decimal]]:
        """Calculate parameter sensitivity - average objective by parameter value."""
        sensitivity = {}

        for param_name in param_grid:
            sensitivity[param_name] = {}
            value_sums: dict[Any, list[Decimal]] = {}

            for ps in param_sets:
                value = ps.params.get(param_name)
                if value not in value_sums:
                    value_sums[value] = []
                value_sums[value].append(ps.objective_value)

            for value, objectives in value_sums.items():
                avg = sum(objectives) / len(objectives)
                sensitivity[param_name][value] = Decimal(str(round(float(avg), 4)))

        return sensitivity
