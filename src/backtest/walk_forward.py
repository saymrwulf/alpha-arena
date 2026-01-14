"""Walk-forward analysis for robust strategy validation."""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Callable

from .data import HistoricalMarket, MarketSnapshot
from .engine import BacktestConfig, BacktestEngine, BacktestResult, BacktestStrategy
from .metrics import BacktestMetrics

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardWindow:
    """A single train/test window in walk-forward analysis."""

    window_number: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    train_result: BacktestResult | None = None
    test_result: BacktestResult | None = None
    optimized_params: dict[str, Any] | None = None


@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward analysis."""

    # Window sizing (in days)
    train_days: int = 180  # 6 months training
    test_days: int = 60  # 2 months testing

    # Window mode
    anchored: bool = False  # If True, training always starts from beginning

    # Step size (days between windows)
    step_days: int | None = None  # Defaults to test_days (rolling)

    # Minimum data requirements
    min_train_snapshots: int = 100
    min_test_snapshots: int = 20

    # Whether to require training before testing
    require_train: bool = True


@dataclass
class WalkForwardResult:
    """Results from walk-forward analysis."""

    analysis_date: datetime
    config: WalkForwardConfig
    total_windows: int
    completed_windows: int

    # Aggregated test results
    test_metrics: BacktestMetrics
    total_test_return_pct: Decimal
    avg_window_return_pct: Decimal
    window_win_rate: Decimal  # % of windows profitable
    consistency_score: Decimal  # Std dev of window returns (lower = better)

    # Individual windows
    windows: list[WalkForwardWindow]

    # Combined equity curve from all test periods
    combined_equity_curve: list[tuple[datetime, Decimal]]

    # Degradation analysis
    train_vs_test_ratio: Decimal  # Avg test return / avg train return
    overfitting_score: Decimal  # 0-100, higher = more overfit

    def summary(self) -> str:
        """Generate walk-forward summary."""
        lines = [
            "",
            "=" * 70,
            "  WALK-FORWARD ANALYSIS RESULTS",
            "=" * 70,
            f"  Date: {self.analysis_date.strftime('%Y-%m-%d %H:%M')}",
            f"  Windows Completed: {self.completed_windows}/{self.total_windows}",
            f"  Train Period: {self.config.train_days} days",
            f"  Test Period: {self.config.test_days} days",
            f"  Mode: {'Anchored' if self.config.anchored else 'Rolling'}",
            "",
            "  AGGREGATE TEST PERFORMANCE",
            "-" * 70,
            f"    Total Return: {self.total_test_return_pct:+.2f}%",
            f"    Avg Window Return: {self.avg_window_return_pct:+.2f}%",
            f"    Window Win Rate: {self.window_win_rate:.1f}%",
            f"    Consistency Score: {self.consistency_score:.2f} (lower = better)",
            "",
            "  ROBUSTNESS METRICS",
            "-" * 70,
            f"    Train/Test Ratio: {self.train_vs_test_ratio:.2f}",
            f"    Overfitting Score: {self.overfitting_score:.1f}/100",
            "",
            "  TEST METRICS",
            "-" * 70,
            f"    Sharpe Ratio: {self.test_metrics.sharpe_ratio:.2f}",
            f"    Sortino Ratio: {self.test_metrics.sortino_ratio:.2f}",
            f"    Max Drawdown: {self.test_metrics.max_drawdown_pct:.2f}%",
            f"    Win Rate: {self.test_metrics.win_rate:.1f}%",
            f"    Total Trades: {self.test_metrics.total_trades}",
            "",
            "  WINDOW BREAKDOWN",
            "-" * 70,
        ]

        for window in self.windows:
            if window.test_result:
                ret = window.test_result.metrics.total_return_pct
                trades = window.test_result.metrics.total_trades
                lines.append(
                    f"    Window {window.window_number}: "
                    f"{window.test_start.date()} to {window.test_end.date()} | "
                    f"Return: {ret:+.2f}% | Trades: {trades}"
                )

        lines.append("=" * 70)
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "analysis_date": self.analysis_date.isoformat(),
            "total_windows": self.total_windows,
            "completed_windows": self.completed_windows,
            "total_test_return_pct": float(self.total_test_return_pct),
            "avg_window_return_pct": float(self.avg_window_return_pct),
            "window_win_rate": float(self.window_win_rate),
            "consistency_score": float(self.consistency_score),
            "train_vs_test_ratio": float(self.train_vs_test_ratio),
            "overfitting_score": float(self.overfitting_score),
            "test_metrics": {
                "sharpe_ratio": float(self.test_metrics.sharpe_ratio),
                "sortino_ratio": float(self.test_metrics.sortino_ratio),
                "max_drawdown_pct": float(self.test_metrics.max_drawdown_pct),
                "win_rate": float(self.test_metrics.win_rate),
                "total_trades": self.test_metrics.total_trades,
            },
            "windows": [
                {
                    "window_number": w.window_number,
                    "train_start": w.train_start.isoformat(),
                    "train_end": w.train_end.isoformat(),
                    "test_start": w.test_start.isoformat(),
                    "test_end": w.test_end.isoformat(),
                    "test_return": float(w.test_result.metrics.total_return_pct)
                    if w.test_result
                    else None,
                }
                for w in self.windows
            ],
        }


class WalkForwardAnalyzer:
    """
    Walk-forward analysis for strategy validation.

    Splits historical data into sequential train/test windows to validate
    strategy robustness and detect overfitting.

    Usage:
        analyzer = WalkForwardAnalyzer()
        result = await analyzer.analyze(
            strategy_factory=lambda: MyStrategy(),
            markets=historical_markets,
            config=WalkForwardConfig(train_days=180, test_days=60),
        )
        print(result.summary())
    """

    def __init__(self, backtest_config: BacktestConfig | None = None):
        self.backtest_config = backtest_config or BacktestConfig()
        self.engine = BacktestEngine(self.backtest_config)

    async def analyze(
        self,
        strategy_factory: Callable[[], BacktestStrategy],
        markets: list[HistoricalMarket],
        config: WalkForwardConfig | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> WalkForwardResult:
        """
        Run walk-forward analysis.

        Args:
            strategy_factory: Function that creates a fresh strategy instance
            markets: Historical market data
            config: Walk-forward configuration
            progress_callback: Optional callback(completed, total)

        Returns:
            WalkForwardResult with aggregated metrics and window breakdown
        """
        config = config or WalkForwardConfig()
        analysis_date = datetime.utcnow()

        # Find data date range
        all_timestamps = self._extract_timestamps(markets)
        if not all_timestamps:
            return self._empty_result(config, analysis_date)

        data_start = min(all_timestamps)
        data_end = max(all_timestamps)

        # Generate windows
        windows = self._generate_windows(data_start, data_end, config)

        if not windows:
            logger.warning("No valid windows could be generated")
            return self._empty_result(config, analysis_date)

        total_windows = len(windows)
        logger.info(f"Starting walk-forward analysis with {total_windows} windows")

        # Run each window
        all_test_trades = []
        all_test_equity = []
        train_returns = []
        test_returns = []

        for i, window in enumerate(windows):
            # Filter markets for this window
            train_markets = self._filter_markets_by_date(
                markets, window.train_start, window.train_end
            )
            test_markets = self._filter_markets_by_date(
                markets, window.test_start, window.test_end
            )

            # Check minimum data requirements
            train_snapshots = sum(len(m.snapshots) for m in train_markets)
            test_snapshots = sum(len(m.snapshots) for m in test_markets)

            if train_snapshots < config.min_train_snapshots and config.require_train:
                logger.debug(f"Window {i+1}: Insufficient training data ({train_snapshots} snapshots)")
                continue

            if test_snapshots < config.min_test_snapshots:
                logger.debug(f"Window {i+1}: Insufficient test data ({test_snapshots} snapshots)")
                continue

            # Run training (for metrics comparison)
            if train_markets:
                strategy = strategy_factory()
                try:
                    window.train_result = await self.engine.run(strategy, train_markets)
                    train_returns.append(window.train_result.metrics.total_return_pct)
                except Exception as e:
                    logger.error(f"Window {i+1} training failed: {e}")

            # Run test
            if test_markets:
                strategy = strategy_factory()
                try:
                    window.test_result = await self.engine.run(strategy, test_markets)
                    test_returns.append(window.test_result.metrics.total_return_pct)

                    # Aggregate test results
                    all_test_trades.extend(window.test_result.trades)
                    all_test_equity.extend(window.test_result.equity_curve)
                except Exception as e:
                    logger.error(f"Window {i+1} testing failed: {e}")

            if progress_callback:
                progress_callback(i + 1, total_windows)

        # Calculate aggregate metrics
        completed_windows = len([w for w in windows if w.test_result is not None])

        if not test_returns:
            return self._empty_result(config, analysis_date, windows)

        # Sort combined equity curve by timestamp
        all_test_equity.sort(key=lambda x: x[0])

        # Calculate test metrics from all trades
        equity_values = [e[1] for e in all_test_equity] if all_test_equity else []
        test_metrics = BacktestMetrics.from_trades(
            all_test_trades,
            equity_values,
            self.backtest_config.starting_capital,
        )

        # Calculate statistics
        total_test_return = sum(test_returns)
        avg_window_return = total_test_return / len(test_returns)
        profitable_windows = sum(1 for r in test_returns if r > 0)
        window_win_rate = Decimal(str(profitable_windows / len(test_returns) * 100))

        # Consistency score (std dev of returns, lower is better)
        mean_return = avg_window_return
        variance = sum((r - mean_return) ** 2 for r in test_returns) / len(test_returns)
        consistency_score = Decimal(str(float(variance) ** 0.5))

        # Train vs test comparison (detect overfitting)
        avg_train = sum(train_returns) / len(train_returns) if train_returns else Decimal("1")
        avg_test = avg_window_return

        # Avoid division by zero
        if avg_train != 0:
            train_vs_test_ratio = avg_test / avg_train
        else:
            train_vs_test_ratio = Decimal("1")

        # Overfitting score: 0-100
        # High score = test performance much worse than train
        # Score = 100 * (1 - test/train) capped at 0-100
        if avg_train > 0 and avg_test < avg_train:
            overfit_raw = (1 - float(avg_test / avg_train)) * 100
            overfitting_score = Decimal(str(min(100, max(0, overfit_raw))))
        elif avg_train < 0 and avg_test > avg_train:
            # Both negative but test less negative = good
            overfitting_score = Decimal("0")
        else:
            overfitting_score = Decimal("0")

        return WalkForwardResult(
            analysis_date=analysis_date,
            config=config,
            total_windows=total_windows,
            completed_windows=completed_windows,
            test_metrics=test_metrics,
            total_test_return_pct=total_test_return,
            avg_window_return_pct=avg_window_return,
            window_win_rate=window_win_rate,
            consistency_score=consistency_score,
            windows=windows,
            combined_equity_curve=all_test_equity,
            train_vs_test_ratio=train_vs_test_ratio,
            overfitting_score=overfitting_score,
        )

    async def analyze_with_optimization(
        self,
        strategy_factory: Callable[..., BacktestStrategy],
        param_grid: dict[str, list[Any]],
        markets: list[HistoricalMarket],
        config: WalkForwardConfig | None = None,
        objective: str = "sharpe_ratio",
    ) -> WalkForwardResult:
        """
        Walk-forward with parameter optimization in each training window.

        Re-optimizes strategy parameters in each training period and
        applies best parameters to the following test period.

        Args:
            strategy_factory: Function that creates strategy from params
            param_grid: Parameter grid for optimization
            markets: Historical market data
            config: Walk-forward configuration
            objective: Optimization objective

        Returns:
            WalkForwardResult with optimized parameters per window
        """
        from .optimizer import ParameterOptimizer

        config = config or WalkForwardConfig()
        analysis_date = datetime.utcnow()

        # Find data date range
        all_timestamps = self._extract_timestamps(markets)
        if not all_timestamps:
            return self._empty_result(config, analysis_date)

        data_start = min(all_timestamps)
        data_end = max(all_timestamps)

        # Generate windows
        windows = self._generate_windows(data_start, data_end, config)

        if not windows:
            return self._empty_result(config, analysis_date)

        optimizer = ParameterOptimizer(self.backtest_config)
        total_windows = len(windows)

        all_test_trades = []
        all_test_equity = []
        train_returns = []
        test_returns = []

        for i, window in enumerate(windows):
            train_markets = self._filter_markets_by_date(
                markets, window.train_start, window.train_end
            )
            test_markets = self._filter_markets_by_date(
                markets, window.test_start, window.test_end
            )

            if not train_markets or not test_markets:
                continue

            # Optimize on training data
            try:
                opt_result = await optimizer.grid_search(
                    strategy_factory=strategy_factory,
                    param_grid=param_grid,
                    markets=train_markets,
                    objective=objective,
                )
                window.optimized_params = opt_result.best_params
                window.train_result = opt_result.best_result

                if window.train_result:
                    train_returns.append(window.train_result.metrics.total_return_pct)

            except Exception as e:
                logger.error(f"Window {i+1} optimization failed: {e}")
                continue

            # Test with optimized params
            try:
                best_strategy = strategy_factory(**window.optimized_params)
                window.test_result = await self.engine.run(best_strategy, test_markets)
                test_returns.append(window.test_result.metrics.total_return_pct)

                all_test_trades.extend(window.test_result.trades)
                all_test_equity.extend(window.test_result.equity_curve)

            except Exception as e:
                logger.error(f"Window {i+1} testing failed: {e}")

        # Calculate aggregates (same as analyze method)
        completed_windows = len([w for w in windows if w.test_result is not None])

        if not test_returns:
            return self._empty_result(config, analysis_date, windows)

        all_test_equity.sort(key=lambda x: x[0])
        equity_values = [e[1] for e in all_test_equity] if all_test_equity else []
        test_metrics = BacktestMetrics.from_trades(
            all_test_trades,
            equity_values,
            self.backtest_config.starting_capital,
        )

        total_test_return = sum(test_returns)
        avg_window_return = total_test_return / len(test_returns)
        profitable_windows = sum(1 for r in test_returns if r > 0)
        window_win_rate = Decimal(str(profitable_windows / len(test_returns) * 100))

        mean_return = avg_window_return
        variance = sum((r - mean_return) ** 2 for r in test_returns) / len(test_returns)
        consistency_score = Decimal(str(float(variance) ** 0.5))

        avg_train = sum(train_returns) / len(train_returns) if train_returns else Decimal("1")
        avg_test = avg_window_return
        train_vs_test_ratio = avg_test / avg_train if avg_train != 0 else Decimal("1")

        if avg_train > 0 and avg_test < avg_train:
            overfit_raw = (1 - float(avg_test / avg_train)) * 100
            overfitting_score = Decimal(str(min(100, max(0, overfit_raw))))
        else:
            overfitting_score = Decimal("0")

        return WalkForwardResult(
            analysis_date=analysis_date,
            config=config,
            total_windows=total_windows,
            completed_windows=completed_windows,
            test_metrics=test_metrics,
            total_test_return_pct=total_test_return,
            avg_window_return_pct=avg_window_return,
            window_win_rate=window_win_rate,
            consistency_score=consistency_score,
            windows=windows,
            combined_equity_curve=all_test_equity,
            train_vs_test_ratio=train_vs_test_ratio,
            overfitting_score=overfitting_score,
        )

    def _extract_timestamps(self, markets: list[HistoricalMarket]) -> list[datetime]:
        """Extract all timestamps from markets."""
        timestamps = []
        for market in markets:
            for snapshot in market.snapshots:
                timestamps.append(snapshot.timestamp)
        return timestamps

    def _generate_windows(
        self,
        data_start: datetime,
        data_end: datetime,
        config: WalkForwardConfig,
    ) -> list[WalkForwardWindow]:
        """Generate train/test windows based on configuration."""
        windows = []
        window_num = 1

        train_delta = timedelta(days=config.train_days)
        test_delta = timedelta(days=config.test_days)
        step_delta = timedelta(days=config.step_days or config.test_days)

        if config.anchored:
            # Anchored: training always starts from data_start
            current_train_start = data_start
            current_test_start = data_start + train_delta

            while current_test_start + test_delta <= data_end:
                windows.append(
                    WalkForwardWindow(
                        window_number=window_num,
                        train_start=current_train_start,
                        train_end=current_test_start,
                        test_start=current_test_start,
                        test_end=current_test_start + test_delta,
                    )
                )
                window_num += 1
                current_test_start += step_delta
        else:
            # Rolling: both train and test windows move forward
            current_train_start = data_start

            while current_train_start + train_delta + test_delta <= data_end:
                train_end = current_train_start + train_delta
                test_start = train_end
                test_end = test_start + test_delta

                windows.append(
                    WalkForwardWindow(
                        window_number=window_num,
                        train_start=current_train_start,
                        train_end=train_end,
                        test_start=test_start,
                        test_end=test_end,
                    )
                )
                window_num += 1
                current_train_start += step_delta

        return windows

    def _filter_markets_by_date(
        self,
        markets: list[HistoricalMarket],
        start_date: datetime,
        end_date: datetime,
    ) -> list[HistoricalMarket]:
        """Filter markets and their snapshots to a date range."""
        filtered = []

        for market in markets:
            # Filter snapshots within date range
            filtered_snapshots = [
                s for s in market.snapshots
                if start_date <= s.timestamp <= end_date
            ]

            if filtered_snapshots:
                # Create new market with filtered snapshots
                filtered_market = HistoricalMarket(
                    market_id=market.market_id,
                    question=market.question,
                    start_date=max(market.start_date, start_date),
                    end_date=min(market.end_date or end_date, end_date),
                    resolution_date=market.resolution_date,
                    outcome=market.outcome,
                    snapshots=filtered_snapshots,
                )
                filtered.append(filtered_market)

        return filtered

    def _empty_result(
        self,
        config: WalkForwardConfig,
        analysis_date: datetime,
        windows: list[WalkForwardWindow] | None = None,
    ) -> WalkForwardResult:
        """Return empty result when analysis cannot be performed."""
        return WalkForwardResult(
            analysis_date=analysis_date,
            config=config,
            total_windows=len(windows) if windows else 0,
            completed_windows=0,
            test_metrics=BacktestMetrics(
                starting_capital=self.backtest_config.starting_capital,
                ending_capital=self.backtest_config.starting_capital,
            ),
            total_test_return_pct=Decimal("0"),
            avg_window_return_pct=Decimal("0"),
            window_win_rate=Decimal("0"),
            consistency_score=Decimal("0"),
            windows=windows or [],
            combined_equity_curve=[],
            train_vs_test_ratio=Decimal("1"),
            overfitting_score=Decimal("0"),
        )
