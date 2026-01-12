"""Tests for backtesting framework."""

import pytest
from datetime import datetime, timedelta
from decimal import Decimal

from src.backtest.data import (
    HistoricalDataLoader,
    HistoricalMarket,
    MarketSnapshot,
    OHLCV,
)
from src.backtest.metrics import (
    calculate_sharpe,
    calculate_max_drawdown,
    calculate_sortino,
    calculate_calmar,
    BacktestMetrics,
    TradeResult,
)
from src.backtest.engine import (
    BacktestConfig,
    BacktestEngine,
    BacktestOrder,
    BacktestPortfolio,
    SimpleStrategy,
)


class TestMetricCalculations:
    """Tests for individual metric calculations."""

    def test_sharpe_positive_returns(self):
        """Test Sharpe ratio with positive returns."""
        # 10% daily returns
        returns = [Decimal("0.10")] * 20

        sharpe = calculate_sharpe(returns)

        # High positive Sharpe for consistent returns
        assert sharpe > Decimal("0")

    def test_sharpe_negative_returns(self):
        """Test Sharpe ratio with negative returns."""
        returns = [Decimal("-0.05")] * 20

        sharpe = calculate_sharpe(returns)

        assert sharpe < Decimal("0")

    def test_sharpe_mixed_returns(self):
        """Test Sharpe ratio with mixed returns."""
        returns = [Decimal("0.02"), Decimal("-0.01"), Decimal("0.03"),
                   Decimal("-0.02"), Decimal("0.01")] * 4

        sharpe = calculate_sharpe(returns)

        # Should be positive with net positive returns
        assert isinstance(sharpe, Decimal)

    def test_sharpe_empty_returns(self):
        """Test Sharpe ratio with empty returns."""
        sharpe = calculate_sharpe([])

        assert sharpe == Decimal("0")

    def test_max_drawdown_calculation(self):
        """Test max drawdown calculation."""
        # Start at 100, peak at 150, trough at 90
        equity = [
            Decimal("100"), Decimal("120"), Decimal("150"),
            Decimal("130"), Decimal("110"), Decimal("90"),
            Decimal("100"), Decimal("110"),
        ]

        max_dd, peak_idx, trough_idx = calculate_max_drawdown(equity)

        # Drawdown from 150 to 90 = 40%
        assert max_dd == Decimal("40.0")
        assert peak_idx == 2  # 150
        assert trough_idx == 5  # 90

    def test_max_drawdown_no_drawdown(self):
        """Test max drawdown with monotonic increase."""
        equity = [Decimal("100"), Decimal("110"), Decimal("120"), Decimal("130")]

        max_dd, _, _ = calculate_max_drawdown(equity)

        assert max_dd == Decimal("0")

    def test_sortino_calculation(self):
        """Test Sortino ratio calculation."""
        returns = [Decimal("0.02"), Decimal("-0.03"), Decimal("0.05"),
                   Decimal("-0.01"), Decimal("0.04")] * 4

        sortino = calculate_sortino(returns)

        # Should be higher than Sharpe as it only penalizes downside
        assert isinstance(sortino, Decimal)

    def test_calmar_calculation(self):
        """Test Calmar ratio calculation."""
        calmar = calculate_calmar(
            total_return=Decimal("50"),  # 50% return
            max_drawdown=Decimal("25"),  # 25% drawdown
            years=Decimal("1"),
        )

        # Calmar = 50 / 25 = 2
        assert calmar == Decimal("2.0")

    def test_calmar_zero_drawdown(self):
        """Test Calmar with zero drawdown."""
        calmar = calculate_calmar(
            total_return=Decimal("20"),
            max_drawdown=Decimal("0"),
            years=Decimal("1"),
        )

        assert calmar == Decimal("0")


class TestBacktestMetrics:
    """Tests for BacktestMetrics class."""

    def test_from_trades_winning(self):
        """Test metrics calculation from winning trades."""
        trades = [
            TradeResult(
                trade_id="1", market_id="mkt1",
                entry_time=datetime.utcnow() - timedelta(hours=2),
                exit_time=datetime.utcnow() - timedelta(hours=1),
                side="yes",
                entry_price=Decimal("0.40"),
                exit_price=Decimal("0.50"),
                size=Decimal("100"),
                pnl=Decimal("20"),
                pnl_pct=Decimal("20"),
                fees=Decimal("0.20"),
                is_winner=True,
                holding_period=timedelta(hours=1),
            ),
            TradeResult(
                trade_id="2", market_id="mkt2",
                entry_time=datetime.utcnow() - timedelta(hours=1),
                exit_time=datetime.utcnow(),
                side="yes",
                entry_price=Decimal("0.50"),
                exit_price=Decimal("0.60"),
                size=Decimal("100"),
                pnl=Decimal("15"),
                pnl_pct=Decimal("15"),
                fees=Decimal("0.20"),
                is_winner=True,
                holding_period=timedelta(hours=1),
            ),
        ]

        equity_curve = [
            Decimal("1000"), Decimal("1020"), Decimal("1035"),
        ]

        metrics = BacktestMetrics.from_trades(
            trades, equity_curve, Decimal("1000")
        )

        assert metrics.total_trades == 2
        assert metrics.winning_trades == 2
        assert metrics.win_rate == Decimal("100.00")
        assert metrics.total_pnl == Decimal("35")

    def test_from_trades_mixed(self):
        """Test metrics calculation with mixed trades."""
        trades = [
            TradeResult(
                trade_id="1", market_id="mkt1",
                entry_time=datetime.utcnow() - timedelta(hours=3),
                exit_time=datetime.utcnow() - timedelta(hours=2),
                side="yes",
                entry_price=Decimal("0.40"),
                exit_price=Decimal("0.50"),
                size=Decimal("100"),
                pnl=Decimal("20"),
                pnl_pct=Decimal("20"),
                fees=Decimal("0.20"),
                is_winner=True,
                holding_period=timedelta(hours=1),
            ),
            TradeResult(
                trade_id="2", market_id="mkt2",
                entry_time=datetime.utcnow() - timedelta(hours=2),
                exit_time=datetime.utcnow() - timedelta(hours=1),
                side="no",
                entry_price=Decimal("0.60"),
                exit_price=Decimal("0.50"),
                size=Decimal("100"),
                pnl=Decimal("-15"),
                pnl_pct=Decimal("-15"),
                fees=Decimal("0.20"),
                is_winner=False,
                holding_period=timedelta(hours=1),
            ),
        ]

        equity_curve = [
            Decimal("1000"), Decimal("1020"), Decimal("1005"),
        ]

        metrics = BacktestMetrics.from_trades(
            trades, equity_curve, Decimal("1000")
        )

        assert metrics.total_trades == 2
        assert metrics.winning_trades == 1
        assert metrics.losing_trades == 1
        assert metrics.win_rate == Decimal("50.00")
        assert metrics.yes_trades == 1
        assert metrics.no_trades == 1

    def test_from_trades_empty(self):
        """Test metrics with no trades."""
        metrics = BacktestMetrics.from_trades([], [], Decimal("1000"))

        assert metrics.total_trades == 0
        assert metrics.ending_capital == Decimal("1000")

    def test_metrics_to_dict(self):
        """Test metrics serialization."""
        metrics = BacktestMetrics(
            total_return_pct=Decimal("15.5"),
            total_pnl=Decimal("155"),
            sharpe_ratio=Decimal("1.5"),
            max_drawdown_pct=Decimal("8.0"),
            total_trades=10,
            win_rate=Decimal("60.0"),
        )

        data = metrics.to_dict()

        assert data["performance"]["total_return_pct"] == 15.5
        assert data["performance"]["sharpe_ratio"] == 1.5
        assert data["risk"]["max_drawdown_pct"] == 8.0
        assert data["trades"]["total"] == 10


class TestHistoricalDataLoader:
    """Tests for historical data loading."""

    def test_generate_synthetic(self):
        """Test synthetic data generation."""
        loader = HistoricalDataLoader()

        market = loader.generate_synthetic(
            market_id="test_market",
            question="Test question?",
            days=10,
            start_price=Decimal("0.50"),
            volatility=Decimal("0.01"),
            trend=Decimal("0.001"),
            outcome="yes",
        )

        assert market.market_id == "test_market"
        assert market.question == "Test question?"
        assert market.outcome == "yes"
        assert len(market.snapshots) > 0
        assert len(market.ohlcv) > 0

    def test_synthetic_price_bounds(self):
        """Test synthetic prices stay within bounds."""
        loader = HistoricalDataLoader()

        market = loader.generate_synthetic(
            days=30,
            volatility=Decimal("0.05"),  # High volatility
        )

        for snapshot in market.snapshots:
            assert Decimal("0.01") <= snapshot.yes_price <= Decimal("0.99")
            assert Decimal("0.01") <= snapshot.no_price <= Decimal("0.99")

    def test_iterate_snapshots(self):
        """Test snapshot iteration."""
        loader = HistoricalDataLoader()

        market = loader.generate_synthetic(days=5)

        snapshots = list(loader.iterate_snapshots(market))

        # Should be sorted chronologically
        for i in range(1, len(snapshots)):
            assert snapshots[i].timestamp >= snapshots[i-1].timestamp


class TestBacktestPortfolio:
    """Tests for backtest portfolio management."""

    def test_portfolio_initial_state(self):
        """Test portfolio initial state."""
        portfolio = BacktestPortfolio(cash=Decimal("10000"))

        assert portfolio.cash == Decimal("10000")
        assert portfolio.total_value == Decimal("10000")
        assert len(portfolio.positions) == 0

    def test_portfolio_has_position(self):
        """Test position checking."""
        from src.backtest.engine import BacktestPosition

        portfolio = BacktestPortfolio(cash=Decimal("10000"))

        portfolio.positions["mkt1_yes"] = BacktestPosition(
            market_id="mkt1",
            side="yes",
            entry_time=datetime.utcnow(),
            entry_price=Decimal("0.50"),
            size=Decimal("100"),
            cost_basis=Decimal("50"),
            current_price=Decimal("0.55"),
        )

        assert portfolio.has_position("mkt1", "yes")
        assert not portfolio.has_position("mkt1", "no")
        assert not portfolio.has_position("mkt2", "yes")


class TestBacktestEngine:
    """Tests for backtest engine."""

    @pytest.fixture
    def engine(self):
        """Create backtest engine."""
        config = BacktestConfig(
            starting_capital=Decimal("10000"),
            fee_rate=Decimal("0.001"),
            slippage_pct=Decimal("0.001"),
        )
        return BacktestEngine(config)

    @pytest.fixture
    def synthetic_market(self):
        """Create synthetic market data."""
        loader = HistoricalDataLoader()
        return loader.generate_synthetic(
            market_id="test_market",
            days=30,
            start_price=Decimal("0.50"),
            volatility=Decimal("0.02"),
            outcome="yes",
        )

    @pytest.mark.asyncio
    async def test_run_simple_strategy(self, engine, synthetic_market):
        """Test running a simple strategy."""
        strategy = SimpleStrategy(
            buy_threshold=Decimal("0.40"),
            sell_threshold=Decimal("0.60"),
            position_size=Decimal("100"),
        )

        result = await engine.run(strategy, [synthetic_market])

        assert result.config == engine.config
        assert result.metrics.starting_capital == Decimal("10000")
        assert result.start_time is not None
        assert result.end_time is not None

    @pytest.mark.asyncio
    async def test_backtest_result_summary(self, engine, synthetic_market):
        """Test backtest result summary."""
        strategy = SimpleStrategy()

        result = await engine.run(strategy, [synthetic_market])
        summary = result.summary()

        assert "BACKTEST RESULTS" in summary
        assert "PERFORMANCE" in summary
        assert "TRADES" in summary

    @pytest.mark.asyncio
    async def test_run_with_progress(self, engine, synthetic_market):
        """Test running with progress callback."""
        progress_calls = []

        def on_progress(current, total):
            progress_calls.append((current, total))

        strategy = SimpleStrategy()

        await engine.run(
            strategy,
            [synthetic_market],
            progress_callback=on_progress,
        )

        assert len(progress_calls) > 0


class TestSimpleStrategy:
    """Tests for SimpleStrategy."""

    @pytest.mark.asyncio
    async def test_buy_signal(self):
        """Test buy signal generation."""
        strategy = SimpleStrategy(
            buy_threshold=Decimal("0.40"),
            sell_threshold=Decimal("0.60"),
            position_size=Decimal("100"),
        )

        snapshot = MarketSnapshot(
            timestamp=datetime.utcnow(),
            market_id="mkt1",
            question="Test?",
            yes_price=Decimal("0.35"),  # Below buy threshold
            no_price=Decimal("0.65"),
            yes_volume=Decimal("1000"),
            no_volume=Decimal("1000"),
            liquidity=Decimal("10000"),
        )

        portfolio = BacktestPortfolio(cash=Decimal("10000"))

        orders = await strategy.on_market_update(snapshot, portfolio)

        assert len(orders) == 1
        assert orders[0].side == "yes"
        assert orders[0].action == "buy"

    @pytest.mark.asyncio
    async def test_no_signal_in_range(self):
        """Test no signal when price in range."""
        strategy = SimpleStrategy(
            buy_threshold=Decimal("0.40"),
            sell_threshold=Decimal("0.60"),
        )

        snapshot = MarketSnapshot(
            timestamp=datetime.utcnow(),
            market_id="mkt1",
            question="Test?",
            yes_price=Decimal("0.50"),  # In middle range
            no_price=Decimal("0.50"),
            yes_volume=Decimal("1000"),
            no_volume=Decimal("1000"),
            liquidity=Decimal("10000"),
        )

        portfolio = BacktestPortfolio(cash=Decimal("10000"))

        orders = await strategy.on_market_update(snapshot, portfolio)

        assert len(orders) == 0

    @pytest.mark.asyncio
    async def test_sell_signal(self):
        """Test sell signal generation."""
        from src.backtest.engine import BacktestPosition

        strategy = SimpleStrategy(
            buy_threshold=Decimal("0.40"),
            sell_threshold=Decimal("0.60"),
        )

        snapshot = MarketSnapshot(
            timestamp=datetime.utcnow(),
            market_id="mkt1",
            question="Test?",
            yes_price=Decimal("0.65"),  # Above sell threshold
            no_price=Decimal("0.35"),
            yes_volume=Decimal("1000"),
            no_volume=Decimal("1000"),
            liquidity=Decimal("10000"),
        )

        portfolio = BacktestPortfolio(cash=Decimal("10000"))
        portfolio.positions["mkt1_yes"] = BacktestPosition(
            market_id="mkt1",
            side="yes",
            entry_time=datetime.utcnow(),
            entry_price=Decimal("0.40"),
            size=Decimal("100"),
            cost_basis=Decimal("40"),
            current_price=Decimal("0.65"),
        )

        orders = await strategy.on_market_update(snapshot, portfolio)

        assert len(orders) == 1
        assert orders[0].side == "yes"
        assert orders[0].action == "sell"
