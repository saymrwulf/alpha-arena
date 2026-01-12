"""Backtesting engine for strategy evaluation."""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Callable, Protocol
import uuid

from .data import HistoricalDataLoader, HistoricalMarket, MarketSnapshot
from .metrics import BacktestMetrics, TradeResult


class BacktestStrategy(Protocol):
    """Protocol for backtest-compatible strategies."""

    async def on_market_update(
        self,
        snapshot: MarketSnapshot,
        portfolio: "BacktestPortfolio",
    ) -> list["BacktestOrder"]:
        """Process market update and return orders."""
        ...

    async def on_trade_executed(
        self,
        trade: "BacktestTrade",
        portfolio: "BacktestPortfolio",
    ) -> None:
        """Called when a trade is executed."""
        ...


@dataclass
class BacktestConfig:
    """Configuration for backtest run."""
    starting_capital: Decimal = Decimal("10000")
    fee_rate: Decimal = Decimal("0.001")  # 0.1% per trade
    slippage_pct: Decimal = Decimal("0.001")  # 0.1% slippage
    max_position_pct: Decimal = Decimal("0.10")  # Max 10% per position
    allow_shorting: bool = False
    trade_on_close: bool = True  # Trade at close price vs open


@dataclass
class BacktestOrder:
    """Order in backtest."""
    order_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    market_id: str = ""
    side: str = "yes"  # "yes" or "no"
    action: str = "buy"  # "buy" or "sell"
    size: Decimal = Decimal("0")  # in USDC
    limit_price: Decimal | None = None  # None = market order


@dataclass
class BacktestTrade:
    """Executed trade in backtest."""
    trade_id: str
    order_id: str
    market_id: str
    timestamp: datetime
    side: str
    action: str
    size: Decimal
    price: Decimal
    fees: Decimal
    slippage: Decimal


@dataclass
class BacktestPosition:
    """Open position in backtest."""
    market_id: str
    side: str
    entry_time: datetime
    entry_price: Decimal
    size: Decimal  # shares
    cost_basis: Decimal  # USDC spent
    current_price: Decimal = Decimal("0")

    @property
    def market_value(self) -> Decimal:
        return self.size * self.current_price

    @property
    def unrealized_pnl(self) -> Decimal:
        return self.market_value - self.cost_basis


@dataclass
class BacktestPortfolio:
    """Portfolio state during backtest."""
    cash: Decimal
    positions: dict[str, BacktestPosition] = field(default_factory=dict)
    trades: list[BacktestTrade] = field(default_factory=list)
    closed_trades: list[TradeResult] = field(default_factory=list)

    @property
    def total_value(self) -> Decimal:
        positions_value = sum(p.market_value for p in self.positions.values())
        return self.cash + positions_value

    def get_position(self, market_id: str, side: str) -> BacktestPosition | None:
        key = f"{market_id}_{side}"
        return self.positions.get(key)

    def has_position(self, market_id: str, side: str) -> bool:
        return f"{market_id}_{side}" in self.positions


@dataclass
class BacktestResult:
    """Complete backtest results."""
    config: BacktestConfig
    metrics: BacktestMetrics
    trades: list[TradeResult]
    equity_curve: list[tuple[datetime, Decimal]]
    markets_traded: list[str]
    start_time: datetime
    end_time: datetime
    duration: timedelta

    def summary(self) -> str:
        """Generate summary string."""
        return f"""
╔══════════════════════════════════════════════════════════════╗
║                      BACKTEST RESULTS                         ║
╠══════════════════════════════════════════════════════════════╣
║  Period: {self.start_time.date()} to {self.end_time.date()}
║  Duration: {self.duration.days} days
║
║  PERFORMANCE
║  ├─ Total Return: {self.metrics.total_return_pct:+.2f}%
║  ├─ Total PnL: ${self.metrics.total_pnl:,.2f}
║  ├─ Sharpe Ratio: {self.metrics.sharpe_ratio:.2f}
║  ├─ Sortino Ratio: {self.metrics.sortino_ratio:.2f}
║  └─ Max Drawdown: {self.metrics.max_drawdown_pct:.2f}%
║
║  TRADES
║  ├─ Total Trades: {self.metrics.total_trades}
║  ├─ Win Rate: {self.metrics.win_rate:.1f}%
║  ├─ Profit Factor: {self.metrics.profit_factor:.2f}
║  ├─ Avg Win: ${self.metrics.avg_win:,.2f}
║  └─ Avg Loss: ${self.metrics.avg_loss:,.2f}
║
║  POSITIONS
║  ├─ Avg Size: ${self.metrics.avg_position_size:,.2f}
║  └─ Avg Hold: {self.metrics.avg_holding_period.total_seconds() / 3600:.1f} hours
╚══════════════════════════════════════════════════════════════╝
"""


class BacktestEngine:
    """
    Event-driven backtesting engine.

    Simulates trading on historical market data with realistic
    fees, slippage, and position constraints.
    """

    def __init__(self, config: BacktestConfig | None = None):
        self.config = config or BacktestConfig()
        self.data_loader = HistoricalDataLoader()

    async def run(
        self,
        strategy: BacktestStrategy,
        markets: list[HistoricalMarket],
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> BacktestResult:
        """
        Run backtest on historical data.

        Args:
            strategy: Strategy to backtest
            markets: Historical market data
            progress_callback: Optional callback for progress updates

        Returns:
            Complete backtest results
        """
        start_time = datetime.utcnow()

        # Initialize portfolio
        portfolio = BacktestPortfolio(cash=self.config.starting_capital)
        equity_curve: list[tuple[datetime, Decimal]] = []

        # Merge all snapshots and sort by timestamp
        all_snapshots: list[tuple[MarketSnapshot, HistoricalMarket]] = []
        for market in markets:
            for snapshot in market.snapshots:
                all_snapshots.append((snapshot, market))

        all_snapshots.sort(key=lambda x: x[0].timestamp)

        if not all_snapshots:
            return BacktestResult(
                config=self.config,
                metrics=BacktestMetrics(
                    starting_capital=self.config.starting_capital,
                    ending_capital=self.config.starting_capital,
                ),
                trades=[],
                equity_curve=[],
                markets_traded=[],
                start_time=start_time,
                end_time=datetime.utcnow(),
                duration=timedelta(0),
            )

        total_snapshots = len(all_snapshots)
        data_start = all_snapshots[0][0].timestamp
        data_end = all_snapshots[-1][0].timestamp

        # Process each snapshot
        for i, (snapshot, market) in enumerate(all_snapshots):
            # Update position prices
            for pos in portfolio.positions.values():
                if pos.market_id == snapshot.market_id:
                    pos.current_price = snapshot.yes_price if pos.side == "yes" else snapshot.no_price

            # Check for market resolution
            if market.outcome and snapshot.timestamp >= (market.resolution_date or datetime.max):
                self._settle_position(portfolio, market)

            # Get strategy orders
            orders = await strategy.on_market_update(snapshot, portfolio)

            # Execute orders
            for order in orders:
                trade = self._execute_order(order, snapshot, portfolio)
                if trade:
                    await strategy.on_trade_executed(trade, portfolio)

            # Record equity
            equity_curve.append((snapshot.timestamp, portfolio.total_value))

            # Progress callback
            if progress_callback and i % 100 == 0:
                progress_callback(i, total_snapshots)

        # Settle any remaining positions at last price
        for market in markets:
            if market.outcome:
                self._settle_position(portfolio, market)

        # Calculate metrics
        equity_values = [e[1] for e in equity_curve]
        metrics = BacktestMetrics.from_trades(
            portfolio.closed_trades,
            equity_values,
            self.config.starting_capital,
        )

        return BacktestResult(
            config=self.config,
            metrics=metrics,
            trades=portfolio.closed_trades,
            equity_curve=equity_curve,
            markets_traded=list(set(m.market_id for m in markets)),
            start_time=data_start,
            end_time=data_end,
            duration=data_end - data_start,
        )

    def _execute_order(
        self,
        order: BacktestOrder,
        snapshot: MarketSnapshot,
        portfolio: BacktestPortfolio,
    ) -> BacktestTrade | None:
        """Execute an order with fees and slippage."""
        # Get execution price
        if order.side == "yes":
            base_price = snapshot.yes_price
        else:
            base_price = snapshot.no_price

        # Apply slippage
        if order.action == "buy":
            slippage = base_price * self.config.slippage_pct
            exec_price = base_price + slippage
        else:
            slippage = base_price * self.config.slippage_pct
            exec_price = base_price - slippage

        # Check limit price
        if order.limit_price:
            if order.action == "buy" and exec_price > order.limit_price:
                return None  # Price too high
            if order.action == "sell" and exec_price < order.limit_price:
                return None  # Price too low

        # Calculate fees
        fees = order.size * self.config.fee_rate

        # Validate and execute
        if order.action == "buy":
            # Check cash
            total_cost = order.size + fees
            if portfolio.cash < total_cost:
                return None

            # Check position limits
            max_position = portfolio.total_value * self.config.max_position_pct
            existing = portfolio.get_position(order.market_id, order.side)
            current_exposure = existing.cost_basis if existing else Decimal("0")

            if current_exposure + order.size > max_position:
                order.size = max(Decimal("0"), max_position - current_exposure)
                if order.size <= 0:
                    return None

            # Calculate shares
            shares = order.size / exec_price

            # Update portfolio
            portfolio.cash -= (order.size + fees)

            key = f"{order.market_id}_{order.side}"
            if key in portfolio.positions:
                # Add to existing position
                pos = portfolio.positions[key]
                total_cost_basis = pos.cost_basis + order.size
                total_shares = pos.size + shares
                pos.entry_price = total_cost_basis / total_shares if total_shares > 0 else exec_price
                pos.size = total_shares
                pos.cost_basis = total_cost_basis
            else:
                # New position
                portfolio.positions[key] = BacktestPosition(
                    market_id=order.market_id,
                    side=order.side,
                    entry_time=snapshot.timestamp,
                    entry_price=exec_price,
                    size=shares,
                    cost_basis=order.size,
                    current_price=exec_price,
                )

        else:  # sell
            key = f"{order.market_id}_{order.side}"
            pos = portfolio.positions.get(key)

            if not pos:
                return None  # No position to sell

            # Calculate shares to sell
            shares_to_sell = order.size / exec_price
            shares_to_sell = min(shares_to_sell, pos.size)

            if shares_to_sell <= 0:
                return None

            # Calculate proceeds
            proceeds = shares_to_sell * exec_price - fees

            # Update portfolio
            portfolio.cash += proceeds

            # Record closed trade
            pnl = proceeds - (pos.cost_basis * shares_to_sell / pos.size)
            pnl_pct = (pnl / (pos.cost_basis * shares_to_sell / pos.size) * 100) if pos.cost_basis > 0 else Decimal("0")

            portfolio.closed_trades.append(TradeResult(
                trade_id=str(uuid.uuid4())[:8],
                market_id=order.market_id,
                entry_time=pos.entry_time,
                exit_time=snapshot.timestamp,
                side=order.side,
                entry_price=pos.entry_price,
                exit_price=exec_price,
                size=pos.cost_basis * shares_to_sell / pos.size,
                pnl=pnl,
                pnl_pct=pnl_pct,
                fees=fees,
                is_winner=pnl > 0,
                holding_period=snapshot.timestamp - pos.entry_time,
            ))

            # Update or remove position
            pos.size -= shares_to_sell
            pos.cost_basis -= pos.cost_basis * shares_to_sell / (pos.size + shares_to_sell)

            if pos.size <= Decimal("0.0001"):
                del portfolio.positions[key]

        # Create trade record
        trade = BacktestTrade(
            trade_id=str(uuid.uuid4())[:8],
            order_id=order.order_id,
            market_id=order.market_id,
            timestamp=snapshot.timestamp,
            side=order.side,
            action=order.action,
            size=order.size,
            price=exec_price,
            fees=fees,
            slippage=slippage,
        )

        portfolio.trades.append(trade)
        return trade

    def _settle_position(
        self,
        portfolio: BacktestPortfolio,
        market: HistoricalMarket,
    ) -> None:
        """Settle position when market resolves."""
        for side in ["yes", "no"]:
            key = f"{market.market_id}_{side}"
            pos = portfolio.positions.get(key)

            if not pos:
                continue

            # Determine settlement price
            if market.outcome == side:
                settlement_price = Decimal("1")
            else:
                settlement_price = Decimal("0")

            # Calculate proceeds
            proceeds = pos.size * settlement_price

            # Record closed trade
            pnl = proceeds - pos.cost_basis
            pnl_pct = (pnl / pos.cost_basis * 100) if pos.cost_basis > 0 else Decimal("0")

            portfolio.closed_trades.append(TradeResult(
                trade_id=str(uuid.uuid4())[:8],
                market_id=market.market_id,
                entry_time=pos.entry_time,
                exit_time=market.resolution_date or datetime.utcnow(),
                side=side,
                entry_price=pos.entry_price,
                exit_price=settlement_price,
                size=pos.cost_basis,
                pnl=pnl,
                pnl_pct=pnl_pct,
                fees=Decimal("0"),  # No fees on settlement
                is_winner=pnl > 0,
                holding_period=(market.resolution_date or datetime.utcnow()) - pos.entry_time,
            ))

            # Update portfolio
            portfolio.cash += proceeds
            del portfolio.positions[key]

    async def run_walk_forward(
        self,
        strategy_factory: Callable[[], BacktestStrategy],
        markets: list[HistoricalMarket],
        train_periods: int = 3,
        test_periods: int = 1,
    ) -> list[BacktestResult]:
        """
        Run walk-forward optimization.

        Splits data into train/test periods for more robust evaluation.
        """
        results = []

        # Sort markets by start date
        markets.sort(key=lambda m: m.start_date)

        total_periods = train_periods + test_periods
        if len(markets) < total_periods:
            # Not enough data, run single backtest
            strategy = strategy_factory()
            result = await self.run(strategy, markets)
            return [result]

        # Split into periods
        period_size = len(markets) // total_periods

        for i in range(total_periods - test_periods):
            # Train period
            train_start = i * period_size
            train_end = train_start + train_periods * period_size
            train_markets = markets[train_start:train_end]

            # Test period
            test_start = train_end
            test_end = test_start + test_periods * period_size
            test_markets = markets[test_start:test_end]

            if not test_markets:
                continue

            # Create fresh strategy
            strategy = strategy_factory()

            # Run on test period
            result = await self.run(strategy, test_markets)
            results.append(result)

        return results


class SimpleStrategy:
    """
    Simple mean-reversion strategy for testing.

    Buys when price < threshold, sells when price > threshold.
    """

    def __init__(
        self,
        buy_threshold: Decimal = Decimal("0.4"),
        sell_threshold: Decimal = Decimal("0.6"),
        position_size: Decimal = Decimal("100"),
    ):
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.position_size = position_size

    async def on_market_update(
        self,
        snapshot: MarketSnapshot,
        portfolio: BacktestPortfolio,
    ) -> list[BacktestOrder]:
        """Generate orders based on price thresholds."""
        orders = []

        # Check YES side
        if snapshot.yes_price < self.buy_threshold:
            if not portfolio.has_position(snapshot.market_id, "yes"):
                orders.append(BacktestOrder(
                    market_id=snapshot.market_id,
                    side="yes",
                    action="buy",
                    size=self.position_size,
                ))
        elif snapshot.yes_price > self.sell_threshold:
            if portfolio.has_position(snapshot.market_id, "yes"):
                orders.append(BacktestOrder(
                    market_id=snapshot.market_id,
                    side="yes",
                    action="sell",
                    size=self.position_size,
                ))

        return orders

    async def on_trade_executed(
        self,
        trade: BacktestTrade,
        portfolio: BacktestPortfolio,
    ) -> None:
        """Handle trade execution."""
        pass  # No additional logic needed
