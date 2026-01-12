"""Main trading loop - observe -> decide -> execute -> log."""

import asyncio
import signal
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any, Callable

from ..agent.base import Agent, AgentConfig, AgentState
from ..agent.llm import LLMAgent
from ..broker.base import Broker, Position
from ..broker.polymarket import PolymarketBroker
from ..data.market import MarketData, MarketInfo
from ..data.sentiment import SentimentData, SentimentProvider
from ..metrics.logger import DecisionLog, MetricsLogger, OrderLog, PnLSnapshot
from ..risk.controls import RiskConfig, RiskManager
from ..strategy.base import SignalType, TradeSignal


@dataclass
class RunnerConfig:
    """Configuration for the trading runner."""

    # Loop settings
    loop_interval_seconds: int = 60
    max_iterations: int | None = None

    # Data refresh
    market_refresh_seconds: int = 30
    sentiment_refresh_seconds: int = 300

    # Market filters
    market_categories: list[str] = field(default_factory=list)
    min_liquidity: Decimal = Decimal("1000")
    min_volume_24h: Decimal = Decimal("500")

    # Paths
    jsonl_path: str = "logs/decisions.jsonl"
    sqlite_path: str = "logs/metrics.db"


class TradingRunner:
    """
    Main trading loop orchestrator.

    Coordinates: Broker, Agent, Data, Risk, Metrics
    Loop: observe -> decide -> execute -> log
    """

    def __init__(
        self,
        broker: Broker,
        agent: Agent,
        market_data: MarketData,
        sentiment_provider: SentimentProvider | None,
        risk_manager: RiskManager,
        metrics_logger: MetricsLogger,
        config: RunnerConfig,
    ):
        self.broker = broker
        self.agent = agent
        self.market_data = market_data
        self.sentiment_provider = sentiment_provider
        self.risk_manager = risk_manager
        self.metrics = metrics_logger
        self.config = config

        self._running = False
        self._iteration = 0
        self._markets_cache: list[MarketInfo] = []
        self._markets_updated: datetime | None = None
        self._sentiment_cache: dict[str, SentimentData] = {}
        self._sentiment_updated: datetime | None = None
        self._shutdown_handlers: list[Callable] = []

    async def start(self) -> None:
        """Start the trading loop."""
        self._running = True
        self._setup_signal_handlers()

        print(f"Starting trading loop (interval: {self.config.loop_interval_seconds}s)")

        try:
            while self._running:
                if self.config.max_iterations and self._iteration >= self.config.max_iterations:
                    print(f"Reached max iterations ({self.config.max_iterations})")
                    break

                await self._run_cycle()
                self._iteration += 1

                if self._running:
                    await asyncio.sleep(self.config.loop_interval_seconds)

        except asyncio.CancelledError:
            print("Trading loop cancelled")
        finally:
            await self._cleanup()

    async def stop(self) -> None:
        """Stop the trading loop gracefully."""
        print("Stopping trading loop...")
        self._running = False

    async def _run_cycle(self) -> None:
        """Run one trading cycle."""
        cycle_start = datetime.utcnow()
        print(f"\n=== Cycle {self._iteration + 1} at {cycle_start.isoformat()} ===")

        try:
            # 1. OBSERVE
            self.agent.state = AgentState.OBSERVING
            await self._refresh_data()

            balance = await self.broker.get_balance()
            positions = await self.broker.get_positions()

            print(f"Balance: ${balance:.2f} | Positions: {len(positions)} | Markets: {len(self._markets_cache)}")

            # 2. DECIDE
            self.agent.state = AgentState.DECIDING
            observation, decision, _ = await self.agent.run_cycle(
                self.broker,
                self._markets_cache,
                self._sentiment_cache,
            )

            print(f"Generated {len(decision.signals)} signals (latency: {decision.latency_ms}ms)")

            # 3. RISK CHECK & EXECUTE
            self.agent.state = AgentState.EXECUTING
            executed_orders = []

            for signal in decision.signals:
                if signal.signal_type == SignalType.HOLD:
                    continue

                # Risk check
                check = self.risk_manager.check_signal(signal, balance, positions)

                if not check.passed:
                    print(f"  Signal blocked: {check.messages}")

                    # Try to adjust
                    adjusted = self.risk_manager.adjust_signal_for_risk(
                        signal, balance, positions
                    )
                    if adjusted:
                        print(f"  Adjusted size: {signal.size} -> {adjusted.size}")
                        signal = adjusted
                    else:
                        continue

                # Execute
                if signal.side and signal.size and signal.price:
                    try:
                        order = await self.broker.place_order(
                            market_id=signal.market_id,
                            token_id=signal.token_id,
                            side=signal.side,
                            size=signal.size,
                            price=signal.price,
                        )
                        executed_orders.append(order)
                        self.risk_manager.record_order(order)

                        print(f"  Order placed: {order.id} {signal.side.value} {signal.size} @ {signal.price}")

                        # Log order
                        await self.metrics.log_order(OrderLog(
                            timestamp=datetime.utcnow(),
                            order_id=order.id,
                            market_id=order.market_id,
                            token_id=order.token_id,
                            side=order.side.value,
                            size=order.size,
                            price=order.price,
                            status=order.status.value,
                        ))

                    except Exception as e:
                        print(f"  Order failed: {e}")

            # 4. LOG
            self.agent.state = AgentState.LOGGING

            # Log decision
            await self.metrics.log_decision(DecisionLog(
                timestamp=cycle_start,
                iteration=self._iteration + 1,
                balance=balance,
                positions_count=len(positions),
                markets_analyzed=len(self._markets_cache),
                signals_generated=len(decision.signals),
                signals_executed=len(executed_orders),
                model_used=decision.model_used,
                latency_ms=decision.latency_ms,
                tokens_used=decision.tokens_used,
                reasoning=decision.reasoning,
                signals=[s.to_dict() for s in decision.signals],
            ))

            # Update and log PnL
            await self._log_pnl_snapshot(balance, positions)

            # Check for exit signals on existing positions
            await self._check_exit_conditions(positions, balance)

            self.agent.state = AgentState.IDLE
            print(f"Cycle complete ({(datetime.utcnow() - cycle_start).total_seconds():.1f}s)")

        except Exception as e:
            self.agent.state = AgentState.ERROR
            print(f"Cycle error: {e}")
            import traceback
            traceback.print_exc()

    async def _refresh_data(self) -> None:
        """Refresh market and sentiment data if stale."""
        now = datetime.utcnow()

        # Refresh markets
        if (
            not self._markets_updated or
            (now - self._markets_updated).total_seconds() > self.config.market_refresh_seconds
        ):
            self._markets_cache = await self.market_data.get_markets(
                active_only=True,
                categories=self.config.market_categories or None,
                min_liquidity=self.config.min_liquidity,
                min_volume_24h=self.config.min_volume_24h,
            )
            self._markets_updated = now

        # Refresh sentiment
        if self.sentiment_provider and (
            not self._sentiment_updated or
            (now - self._sentiment_updated).total_seconds() > self.config.sentiment_refresh_seconds
        ):
            self._sentiment_cache = {}
            for market in self._markets_cache[:10]:  # Top 10 markets
                try:
                    # Extract keywords from question
                    keywords = market.question.split()[:5]
                    sentiment = await self.sentiment_provider.get_sentiment(
                        market.question[:50],
                        keywords,
                    )
                    self._sentiment_cache[market.condition_id] = sentiment
                except Exception:
                    continue
            self._sentiment_updated = now

    async def _log_pnl_snapshot(
        self,
        balance: Decimal,
        positions: list[Position],
    ) -> None:
        """Calculate and log PnL snapshot."""
        unrealized = sum(p.unrealized_pnl for p in positions)
        realized = sum(p.realized_pnl for p in positions)
        positions_value = sum(p.size * p.current_price for p in positions)
        total_equity = balance + positions_value

        drawdown = self.metrics.calculate_drawdown(total_equity)

        await self.metrics.log_pnl_snapshot(PnLSnapshot(
            timestamp=datetime.utcnow(),
            balance=balance,
            unrealized_pnl=unrealized,
            realized_pnl=realized,
            total_equity=total_equity,
            positions_value=positions_value,
            drawdown_pct=drawdown,
            high_water_mark=self.metrics._high_water_mark,
        ))

        # Record realized PnL for risk tracking
        self.risk_manager.record_pnl(realized)

    async def _check_exit_conditions(
        self,
        positions: list[Position],
        balance: Decimal,
    ) -> None:
        """Check if any positions should be exited."""
        # This is a simplified check - in production you'd integrate
        # with the strategy's should_exit method

        for position in positions:
            # Check for large drawdown on position
            if position.unrealized_pnl < 0:
                loss_pct = abs(position.unrealized_pnl) / (position.size * position.avg_entry_price)
                if loss_pct > Decimal("0.15"):  # 15% loss
                    print(f"  Warning: Position {position.market_id[:8]}... down {loss_pct:.1%}")

    def _setup_signal_handlers(self) -> None:
        """Setup graceful shutdown on SIGINT/SIGTERM."""
        loop = asyncio.get_event_loop()

        def handle_signal():
            print("\nReceived shutdown signal")
            asyncio.create_task(self.stop())

        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, handle_signal)

    async def _cleanup(self) -> None:
        """Cleanup on shutdown."""
        print("Cleaning up...")
        for handler in self._shutdown_handlers:
            try:
                handler()
            except Exception:
                pass

    def add_shutdown_handler(self, handler: Callable) -> None:
        """Add a cleanup handler for shutdown."""
        self._shutdown_handlers.append(handler)

    @property
    def is_running(self) -> bool:
        """Check if runner is active."""
        return self._running

    @property
    def iteration(self) -> int:
        """Current iteration count."""
        return self._iteration

    def get_status(self) -> dict[str, Any]:
        """Get current runner status."""
        return {
            "running": self._running,
            "iteration": self._iteration,
            "agent_state": self.agent.state.value,
            "markets_cached": len(self._markets_cache),
            "sentiment_cached": len(self._sentiment_cache),
            "risk_status": self.risk_manager.get_status(),
        }
