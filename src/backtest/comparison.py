"""Multi-strategy comparison and analysis framework."""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any, Callable

from .data import HistoricalMarket
from .engine import BacktestConfig, BacktestEngine, BacktestResult, BacktestStrategy
from .metrics import BacktestMetrics

logger = logging.getLogger(__name__)


@dataclass
class StrategyResult:
    """Result for a single strategy in comparison."""

    name: str
    result: BacktestResult
    rank: int = 0


@dataclass
class ComparisonMetrics:
    """Key metrics for strategy comparison."""

    total_return_pct: Decimal
    sharpe_ratio: Decimal
    sortino_ratio: Decimal
    max_drawdown_pct: Decimal
    win_rate: Decimal
    profit_factor: Decimal
    total_trades: int
    calmar_ratio: Decimal


@dataclass
class StrategyComparisonResult:
    """Results from comparing multiple strategies."""

    comparison_date: datetime
    strategy_names: list[str]
    results: dict[str, StrategyResult]
    metrics_table: dict[str, ComparisonMetrics]
    rankings: dict[str, dict[str, int]]  # metric -> strategy -> rank
    best_by_metric: dict[str, str]  # metric -> best strategy name
    correlation_matrix: dict[str, dict[str, Decimal]] | None = None

    def summary(self) -> str:
        """Generate comparison summary."""
        lines = [
            "",
            "=" * 70,
            "  STRATEGY COMPARISON RESULTS",
            "=" * 70,
            f"  Date: {self.comparison_date.strftime('%Y-%m-%d %H:%M')}",
            f"  Strategies Compared: {len(self.strategy_names)}",
            "",
            "  RANKINGS (1 = best)",
            "-" * 70,
        ]

        # Build ranking table
        metrics = ["sharpe_ratio", "total_return_pct", "win_rate", "max_drawdown_pct"]
        header = f"  {'Strategy':<20}"
        for m in metrics:
            header += f" {m[:12]:>12}"
        lines.append(header)
        lines.append("-" * 70)

        for name in self.strategy_names:
            row = f"  {name:<20}"
            for m in metrics:
                rank = self.rankings.get(m, {}).get(name, "-")
                row += f" {rank:>12}"
            lines.append(row)

        lines.append("-" * 70)
        lines.append("")
        lines.append("  BEST BY METRIC")

        for metric, best_name in self.best_by_metric.items():
            value = getattr(self.metrics_table[best_name], metric, "N/A")
            lines.append(f"    {metric}: {best_name} ({value})")

        lines.append("=" * 70)
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "comparison_date": self.comparison_date.isoformat(),
            "strategy_names": self.strategy_names,
            "metrics_table": {
                name: {
                    "total_return_pct": float(m.total_return_pct),
                    "sharpe_ratio": float(m.sharpe_ratio),
                    "sortino_ratio": float(m.sortino_ratio),
                    "max_drawdown_pct": float(m.max_drawdown_pct),
                    "win_rate": float(m.win_rate),
                    "profit_factor": float(m.profit_factor),
                    "total_trades": m.total_trades,
                    "calmar_ratio": float(m.calmar_ratio),
                }
                for name, m in self.metrics_table.items()
            },
            "rankings": self.rankings,
            "best_by_metric": self.best_by_metric,
        }


class StrategyComparison:
    """
    Compare multiple strategies on the same historical data.

    Usage:
        comparison = StrategyComparison()
        result = await comparison.compare(
            strategies=[
                ("momentum", momentum_strategy),
                ("mean_reversion", mean_reversion_strategy),
            ],
            markets=historical_markets,
        )
        print(result.summary())
    """

    def __init__(self, config: BacktestConfig | None = None):
        self.config = config or BacktestConfig()
        self.engine = BacktestEngine(self.config)

    async def compare(
        self,
        strategies: list[tuple[str, BacktestStrategy]],
        markets: list[HistoricalMarket],
        parallel: bool = True,
        objective: str = "sharpe_ratio",
    ) -> StrategyComparisonResult:
        """
        Compare multiple strategies on the same data.

        Args:
            strategies: List of (name, strategy) tuples
            markets: Historical market data
            parallel: Run strategies in parallel if True
            objective: Primary metric for ranking

        Returns:
            StrategyComparisonResult with rankings and metrics
        """
        logger.info(f"Comparing {len(strategies)} strategies...")
        comparison_date = datetime.utcnow()

        # Run all strategies
        if parallel:
            results = await self._run_parallel(strategies, markets)
        else:
            results = await self._run_sequential(strategies, markets)

        # Extract metrics for comparison
        metrics_table = {}
        for name, result in results.items():
            m = result.metrics
            metrics_table[name] = ComparisonMetrics(
                total_return_pct=m.total_return_pct,
                sharpe_ratio=m.sharpe_ratio,
                sortino_ratio=m.sortino_ratio,
                max_drawdown_pct=m.max_drawdown_pct,
                win_rate=m.win_rate,
                profit_factor=m.profit_factor,
                total_trades=m.total_trades,
                calmar_ratio=m.calmar_ratio,
            )

        # Calculate rankings
        rankings = self._calculate_rankings(metrics_table)

        # Find best by each metric
        best_by_metric = {}
        for metric in ["sharpe_ratio", "total_return_pct", "sortino_ratio",
                       "win_rate", "profit_factor", "calmar_ratio"]:
            best_by_metric[metric] = self._find_best(metrics_table, metric, higher_better=True)

        # Lower is better for max_drawdown
        best_by_metric["max_drawdown_pct"] = self._find_best(
            metrics_table, "max_drawdown_pct", higher_better=False
        )

        # Build strategy results with ranks
        strategy_results = {}
        for name, result in results.items():
            strategy_results[name] = StrategyResult(
                name=name,
                result=result,
                rank=rankings.get(objective, {}).get(name, 0),
            )

        # Calculate correlation matrix if we have equity curves
        correlation_matrix = self._calculate_correlations(results)

        return StrategyComparisonResult(
            comparison_date=comparison_date,
            strategy_names=list(results.keys()),
            results=strategy_results,
            metrics_table=metrics_table,
            rankings=rankings,
            best_by_metric=best_by_metric,
            correlation_matrix=correlation_matrix,
        )

    async def _run_parallel(
        self,
        strategies: list[tuple[str, BacktestStrategy]],
        markets: list[HistoricalMarket],
    ) -> dict[str, BacktestResult]:
        """Run strategies in parallel."""
        async def run_one(name: str, strategy: BacktestStrategy) -> tuple[str, BacktestResult]:
            logger.info(f"Running backtest for strategy: {name}")
            result = await self.engine.run(strategy, markets)
            return name, result

        tasks = [run_one(name, strategy) for name, strategy in strategies]
        completed = await asyncio.gather(*tasks)

        return {name: result for name, result in completed}

    async def _run_sequential(
        self,
        strategies: list[tuple[str, BacktestStrategy]],
        markets: list[HistoricalMarket],
    ) -> dict[str, BacktestResult]:
        """Run strategies sequentially."""
        results = {}
        for name, strategy in strategies:
            logger.info(f"Running backtest for strategy: {name}")
            results[name] = await self.engine.run(strategy, markets)
        return results

    def _calculate_rankings(
        self,
        metrics_table: dict[str, ComparisonMetrics],
    ) -> dict[str, dict[str, int]]:
        """Calculate rankings for each metric."""
        rankings = {}

        # Metrics where higher is better
        higher_better = [
            "sharpe_ratio", "sortino_ratio", "total_return_pct",
            "win_rate", "profit_factor", "calmar_ratio"
        ]

        # Metrics where lower is better
        lower_better = ["max_drawdown_pct"]

        for metric in higher_better + lower_better:
            values = []
            for name, m in metrics_table.items():
                value = getattr(m, metric, Decimal("0"))
                values.append((name, value))

            # Sort by value
            reverse = metric in higher_better
            values.sort(key=lambda x: x[1], reverse=reverse)

            # Assign ranks
            rankings[metric] = {}
            for rank, (name, _) in enumerate(values, 1):
                rankings[metric][name] = rank

        return rankings

    def _find_best(
        self,
        metrics_table: dict[str, ComparisonMetrics],
        metric: str,
        higher_better: bool,
    ) -> str:
        """Find best strategy for a metric."""
        best_name = None
        best_value = None

        for name, m in metrics_table.items():
            value = getattr(m, metric, Decimal("0"))
            if best_value is None:
                best_name = name
                best_value = value
            elif higher_better and value > best_value:
                best_name = name
                best_value = value
            elif not higher_better and value < best_value:
                best_name = name
                best_value = value

        return best_name or ""

    def _calculate_correlations(
        self,
        results: dict[str, BacktestResult],
    ) -> dict[str, dict[str, Decimal]] | None:
        """Calculate correlation matrix of daily returns."""
        # Extract daily returns from equity curves
        returns_by_strategy: dict[str, list[Decimal]] = {}

        for name, result in results.items():
            if len(result.equity_curve) < 2:
                continue

            daily_returns = []
            prev_value = result.equity_curve[0][1]
            for _, value in result.equity_curve[1:]:
                if prev_value > 0:
                    ret = (value - prev_value) / prev_value
                    daily_returns.append(ret)
                prev_value = value

            returns_by_strategy[name] = daily_returns

        if len(returns_by_strategy) < 2:
            return None

        # Calculate pairwise correlations
        correlation_matrix = {}
        names = list(returns_by_strategy.keys())

        for name1 in names:
            correlation_matrix[name1] = {}
            for name2 in names:
                if name1 == name2:
                    correlation_matrix[name1][name2] = Decimal("1.0")
                else:
                    corr = self._pearson_correlation(
                        returns_by_strategy[name1],
                        returns_by_strategy[name2],
                    )
                    correlation_matrix[name1][name2] = corr

        return correlation_matrix

    def _pearson_correlation(
        self,
        x: list[Decimal],
        y: list[Decimal],
    ) -> Decimal:
        """Calculate Pearson correlation coefficient."""
        n = min(len(x), len(y))
        if n < 2:
            return Decimal("0")

        x = x[:n]
        y = y[:n]

        mean_x = sum(x) / n
        mean_y = sum(y) / n

        numerator = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
        var_x = sum((xi - mean_x) ** 2 for xi in x)
        var_y = sum((yi - mean_y) ** 2 for yi in y)

        if var_x == 0 or var_y == 0:
            return Decimal("0")

        denominator = (var_x * var_y) ** Decimal("0.5")
        correlation = numerator / denominator

        return Decimal(str(round(float(correlation), 4)))
