"""Monte Carlo simulation for backtest analysis."""

import random
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any

from .metrics import TradeResult


@dataclass
class MonteCarloPath:
    """A single simulated equity path."""

    final_value: Decimal
    max_drawdown: Decimal
    total_return_pct: Decimal
    equity_curve: list[Decimal] = field(default_factory=list)


@dataclass
class ConfidenceInterval:
    """Confidence interval for a metric."""

    lower: Decimal
    median: Decimal
    upper: Decimal
    confidence_level: int  # e.g., 95


@dataclass
class MonteCarloResult:
    """Results from Monte Carlo simulation."""

    simulation_date: datetime
    num_simulations: int
    starting_capital: Decimal
    num_trades: int

    # Distribution statistics
    final_value_mean: Decimal
    final_value_std: Decimal
    final_value_ci: ConfidenceInterval

    total_return_mean: Decimal
    total_return_ci: ConfidenceInterval

    max_drawdown_mean: Decimal
    max_drawdown_ci: ConfidenceInterval

    # Risk metrics
    probability_of_profit: Decimal
    probability_of_ruin: Decimal  # Probability of losing > X%
    ruin_threshold: Decimal

    # Best/worst cases
    best_case_return: Decimal
    worst_case_return: Decimal

    # All simulated paths (for plotting)
    paths: list[MonteCarloPath] = field(default_factory=list)

    def summary(self) -> str:
        """Generate Monte Carlo summary."""
        lines = [
            "",
            "=" * 70,
            "  MONTE CARLO SIMULATION RESULTS",
            "=" * 70,
            f"  Date: {self.simulation_date.strftime('%Y-%m-%d %H:%M')}",
            f"  Simulations: {self.num_simulations:,}",
            f"  Trades per path: {self.num_trades}",
            f"  Starting Capital: ${self.starting_capital:,.2f}",
            "",
            "  FINAL VALUE DISTRIBUTION",
            "-" * 70,
            f"    Mean: ${self.final_value_mean:,.2f}",
            f"    Std Dev: ${self.final_value_std:,.2f}",
            f"    {self.final_value_ci.confidence_level}% CI: "
            f"[${self.final_value_ci.lower:,.2f}, ${self.final_value_ci.upper:,.2f}]",
            "",
            "  RETURN DISTRIBUTION",
            "-" * 70,
            f"    Mean Return: {self.total_return_mean:+.2f}%",
            f"    {self.total_return_ci.confidence_level}% CI: "
            f"[{self.total_return_ci.lower:+.2f}%, {self.total_return_ci.upper:+.2f}%]",
            f"    Best Case: {self.best_case_return:+.2f}%",
            f"    Worst Case: {self.worst_case_return:+.2f}%",
            "",
            "  RISK METRICS",
            "-" * 70,
            f"    Probability of Profit: {self.probability_of_profit:.1f}%",
            f"    Probability of Ruin (>{self.ruin_threshold:.0f}% loss): {self.probability_of_ruin:.1f}%",
            f"    Expected Max Drawdown: {self.max_drawdown_mean:.2f}%",
            f"    {self.max_drawdown_ci.confidence_level}% Drawdown CI: "
            f"[{self.max_drawdown_ci.lower:.2f}%, {self.max_drawdown_ci.upper:.2f}%]",
            "=" * 70,
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "simulation_date": self.simulation_date.isoformat(),
            "num_simulations": self.num_simulations,
            "starting_capital": float(self.starting_capital),
            "num_trades": self.num_trades,
            "final_value": {
                "mean": float(self.final_value_mean),
                "std": float(self.final_value_std),
                "ci_lower": float(self.final_value_ci.lower),
                "ci_upper": float(self.final_value_ci.upper),
            },
            "total_return": {
                "mean": float(self.total_return_mean),
                "ci_lower": float(self.total_return_ci.lower),
                "ci_upper": float(self.total_return_ci.upper),
            },
            "max_drawdown": {
                "mean": float(self.max_drawdown_mean),
                "ci_lower": float(self.max_drawdown_ci.lower),
                "ci_upper": float(self.max_drawdown_ci.upper),
            },
            "probability_of_profit": float(self.probability_of_profit),
            "probability_of_ruin": float(self.probability_of_ruin),
            "best_case_return": float(self.best_case_return),
            "worst_case_return": float(self.worst_case_return),
        }


class MonteCarloAnalyzer:
    """
    Monte Carlo simulation for strategy analysis.

    Resamples historical trade outcomes to generate synthetic
    equity paths and estimate confidence intervals.

    Usage:
        analyzer = MonteCarloAnalyzer()
        result = analyzer.simulate(
            closed_trades=backtest_result.trades,
            starting_capital=Decimal("10000"),
            num_simulations=10000,
        )
        print(result.summary())
    """

    def __init__(self, random_seed: int | None = None):
        """Initialize with optional random seed for reproducibility."""
        if random_seed is not None:
            random.seed(random_seed)

    def simulate(
        self,
        closed_trades: list[TradeResult],
        starting_capital: Decimal,
        num_simulations: int = 10000,
        confidence_level: int = 95,
        ruin_threshold: Decimal = Decimal("50"),
        store_paths: bool = False,
    ) -> MonteCarloResult:
        """
        Run Monte Carlo simulation on trade results.

        Args:
            closed_trades: List of historical trade results
            starting_capital: Starting portfolio value
            num_simulations: Number of simulation paths
            confidence_level: Confidence level for intervals (e.g., 95)
            ruin_threshold: Loss percentage considered "ruin"
            store_paths: Whether to store all paths (memory intensive)

        Returns:
            MonteCarloResult with distributions and confidence intervals
        """
        if not closed_trades:
            return self._empty_result(starting_capital, confidence_level, ruin_threshold)

        simulation_date = datetime.utcnow()
        num_trades = len(closed_trades)

        # Extract trade returns (as decimals, e.g., 0.05 for 5% gain)
        trade_returns = []
        for trade in closed_trades:
            if trade.entry_value > 0:
                ret = (trade.pnl / trade.entry_value)
                trade_returns.append(ret)

        if not trade_returns:
            return self._empty_result(starting_capital, confidence_level, ruin_threshold)

        # Run simulations
        final_values = []
        total_returns = []
        max_drawdowns = []
        paths = []

        for _ in range(num_simulations):
            path = self._simulate_path(
                trade_returns,
                starting_capital,
                num_trades,
            )

            final_values.append(path.final_value)
            total_returns.append(path.total_return_pct)
            max_drawdowns.append(path.max_drawdown)

            if store_paths:
                paths.append(path)

        # Calculate statistics
        final_value_mean = self._mean(final_values)
        final_value_std = self._std(final_values, final_value_mean)
        final_value_ci = self._confidence_interval(
            final_values, confidence_level
        )

        total_return_mean = self._mean(total_returns)
        total_return_ci = self._confidence_interval(
            total_returns, confidence_level
        )

        max_drawdown_mean = self._mean(max_drawdowns)
        max_drawdown_ci = self._confidence_interval(
            max_drawdowns, confidence_level
        )

        # Risk metrics
        profitable_count = sum(1 for r in total_returns if r > 0)
        probability_of_profit = Decimal(str(profitable_count / num_simulations * 100))

        ruined_count = sum(1 for r in total_returns if r < -ruin_threshold)
        probability_of_ruin = Decimal(str(ruined_count / num_simulations * 100))

        return MonteCarloResult(
            simulation_date=simulation_date,
            num_simulations=num_simulations,
            starting_capital=starting_capital,
            num_trades=num_trades,
            final_value_mean=final_value_mean,
            final_value_std=final_value_std,
            final_value_ci=final_value_ci,
            total_return_mean=total_return_mean,
            total_return_ci=total_return_ci,
            max_drawdown_mean=max_drawdown_mean,
            max_drawdown_ci=max_drawdown_ci,
            probability_of_profit=probability_of_profit,
            probability_of_ruin=probability_of_ruin,
            ruin_threshold=ruin_threshold,
            best_case_return=max(total_returns),
            worst_case_return=min(total_returns),
            paths=paths if store_paths else [],
        )

    def _simulate_path(
        self,
        trade_returns: list[Decimal],
        starting_capital: Decimal,
        num_trades: int,
    ) -> MonteCarloPath:
        """Simulate a single equity path by resampling trades."""
        equity = starting_capital
        equity_curve = [equity]
        peak = equity

        # Resample trades with replacement
        sampled_returns = random.choices(trade_returns, k=num_trades)

        max_drawdown = Decimal("0")

        for ret in sampled_returns:
            # Apply return
            equity = equity * (1 + ret)
            equity_curve.append(equity)

            # Track drawdown
            if equity > peak:
                peak = equity
            drawdown = (peak - equity) / peak * 100 if peak > 0 else Decimal("0")
            if drawdown > max_drawdown:
                max_drawdown = drawdown

        final_value = equity
        total_return = (final_value - starting_capital) / starting_capital * 100

        return MonteCarloPath(
            final_value=final_value,
            max_drawdown=max_drawdown,
            total_return_pct=total_return,
            equity_curve=equity_curve,
        )

    def _mean(self, values: list[Decimal]) -> Decimal:
        """Calculate mean of values."""
        if not values:
            return Decimal("0")
        return sum(values) / len(values)

    def _std(self, values: list[Decimal], mean: Decimal) -> Decimal:
        """Calculate standard deviation."""
        if len(values) < 2:
            return Decimal("0")
        variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
        return Decimal(str(float(variance) ** 0.5))

    def _confidence_interval(
        self,
        values: list[Decimal],
        confidence_level: int,
    ) -> ConfidenceInterval:
        """Calculate confidence interval."""
        if not values:
            return ConfidenceInterval(
                lower=Decimal("0"),
                median=Decimal("0"),
                upper=Decimal("0"),
                confidence_level=confidence_level,
            )

        sorted_values = sorted(values)
        n = len(sorted_values)

        # Calculate percentiles
        lower_pct = (100 - confidence_level) / 2
        upper_pct = 100 - lower_pct

        lower_idx = int(n * lower_pct / 100)
        upper_idx = int(n * upper_pct / 100) - 1
        median_idx = n // 2

        return ConfidenceInterval(
            lower=sorted_values[max(0, lower_idx)],
            median=sorted_values[median_idx],
            upper=sorted_values[min(n - 1, upper_idx)],
            confidence_level=confidence_level,
        )

    def _empty_result(
        self,
        starting_capital: Decimal,
        confidence_level: int,
        ruin_threshold: Decimal,
    ) -> MonteCarloResult:
        """Return empty result when no trades available."""
        empty_ci = ConfidenceInterval(
            lower=Decimal("0"),
            median=Decimal("0"),
            upper=Decimal("0"),
            confidence_level=confidence_level,
        )
        return MonteCarloResult(
            simulation_date=datetime.utcnow(),
            num_simulations=0,
            starting_capital=starting_capital,
            num_trades=0,
            final_value_mean=starting_capital,
            final_value_std=Decimal("0"),
            final_value_ci=empty_ci,
            total_return_mean=Decimal("0"),
            total_return_ci=empty_ci,
            max_drawdown_mean=Decimal("0"),
            max_drawdown_ci=empty_ci,
            probability_of_profit=Decimal("0"),
            probability_of_ruin=Decimal("0"),
            ruin_threshold=ruin_threshold,
            best_case_return=Decimal("0"),
            worst_case_return=Decimal("0"),
            paths=[],
        )
