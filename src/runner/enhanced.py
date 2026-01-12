"""
Enhanced trading runner with multi-agent coordination.

This is the world-class runner that orchestrates:
- Multi-agent decision making (Research, Risk, Execution, Reflection)
- Technical indicators (EMA, RSI, MACD, volume, ATR)
- Cross-platform arbitrage detection
- Memory system (short-term, long-term, episodic)
- Kelly Criterion position sizing
"""

import asyncio
import signal
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable

from ..agents.coordinator import AgentCoordinator, CoordinatorDecision
from ..arbitrage.detector import ArbitrageDetector, ArbitrageOpportunity
from ..arbitrage.platforms import Platform, PolymarketPlatform, KalshiPlatform
from ..broker.base import Broker, Order, Position
from ..core.config import (
    AgentConfig,
    MemoryConfig,
    IndicatorConfig,
    LLMProviderConfig,
)
from ..core.types import MarketState, Signal, OrderBook
from ..data.market import MarketData, MarketInfo
from ..indicators.calculator import IndicatorCalculator
from ..indicators.types import IndicatorResult
from ..llm.registry import ProviderRegistry
from ..memory.manager import MemoryManager
from ..memory.episodic import Episode
from ..metrics.logger import MetricsLogger, DecisionLog, OrderLog, PnLSnapshot


@dataclass
class EnhancedRunnerConfig:
    """Configuration for the enhanced trading runner."""
    # Loop settings
    loop_interval_seconds: int = 60
    max_iterations: int | None = None

    # Data refresh intervals
    market_refresh_seconds: int = 30
    indicator_update_seconds: int = 60
    arbitrage_scan_seconds: int = 120
    memory_context_seconds: int = 300

    # Market filters
    market_categories: list[str] = field(default_factory=list)
    min_liquidity: Decimal = Decimal("5000")
    min_volume_24h: Decimal = Decimal("1000")
    max_markets_per_cycle: int = 20

    # Feature toggles
    enable_indicators: bool = True
    enable_arbitrage: bool = True
    enable_reflection: bool = True
    enable_memory: bool = True

    # Paths
    memory_db_path: str = "data/memory.db"
    metrics_jsonl_path: str = "logs/decisions.jsonl"
    metrics_sqlite_path: str = "logs/metrics.db"


class EnhancedTradingRunner:
    """
    World-class trading runner with multi-agent coordination.

    This runner orchestrates all components for sophisticated
    prediction market trading:

    1. Data Layer:
       - Real-time market data from Polymarket
       - Technical indicators (EMA, RSI, MACD)
       - Cross-platform arbitrage scanning

    2. Intelligence Layer:
       - Multi-agent decision system
       - Memory-augmented context
       - Reflection and learning loops

    3. Execution Layer:
       - Risk-adjusted position sizing (Kelly)
       - Order execution with slippage management
       - Position monitoring

    4. Logging Layer:
       - Full decision audit trail
       - PnL tracking
       - Performance metrics
    """

    def __init__(
        self,
        broker: Broker,
        market_data: MarketData,
        coordinator: AgentCoordinator,
        memory: MemoryManager,
        indicators: IndicatorCalculator,
        metrics_logger: MetricsLogger,
        config: EnhancedRunnerConfig,
        arbitrage_platforms: list[Platform] | None = None,
    ):
        self.broker = broker
        self.market_data = market_data
        self.coordinator = coordinator
        self.memory = memory
        self.indicators = indicators
        self.metrics = metrics_logger
        self.config = config

        # Arbitrage
        if config.enable_arbitrage and arbitrage_platforms:
            self.arbitrage = ArbitrageDetector(arbitrage_platforms)
        else:
            self.arbitrage = None

        # State
        self._running = False
        self._iteration = 0
        self._markets_cache: list[MarketInfo] = []
        self._markets_updated: datetime | None = None
        self._indicator_cache: dict[str, IndicatorResult] = {}
        self._arbitrage_cache: list[ArbitrageOpportunity] = []
        self._active_episodes: dict[str, Episode] = {}
        self._shutdown_handlers: list[Callable] = []

    async def start(self) -> None:
        """Start the enhanced trading loop."""
        self._running = True
        self._setup_signal_handlers()

        print("=" * 60)
        print("  ENHANCED TRADING RUNNER")
        print("=" * 60)
        print(f"  Loop interval: {self.config.loop_interval_seconds}s")
        print(f"  Features enabled:")
        print(f"    - Multi-agent: Yes")
        print(f"    - Indicators: {self.config.enable_indicators}")
        print(f"    - Arbitrage: {self.config.enable_arbitrage}")
        print(f"    - Memory: {self.config.enable_memory}")
        print(f"    - Reflection: {self.config.enable_reflection}")
        print("=" * 60)

        # Initialize memory
        if self.config.enable_memory:
            await self.memory.initialize()

        # Initialize arbitrage platforms
        if self.arbitrage:
            for platform in self.arbitrage.platforms.values():
                await platform.connect()

        # Initialize coordinator
        await self.coordinator.initialize()

        try:
            while self._running:
                if self.config.max_iterations and self._iteration >= self.config.max_iterations:
                    print(f"\nReached max iterations ({self.config.max_iterations})")
                    break

                await self._run_cycle()
                self._iteration += 1

                if self._running:
                    await asyncio.sleep(self.config.loop_interval_seconds)

        except asyncio.CancelledError:
            print("\nTrading loop cancelled")
        finally:
            await self._cleanup()

    async def stop(self) -> None:
        """Stop the trading loop gracefully."""
        print("\nStopping enhanced trading loop...")
        self._running = False

    async def _run_cycle(self) -> None:
        """Run one enhanced trading cycle."""
        cycle_start = datetime.utcnow()
        print(f"\n{'='*60}")
        print(f"  CYCLE {self._iteration + 1} | {cycle_start.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}")

        try:
            # Phase 1: DATA GATHERING
            print("\n[1] DATA GATHERING")
            await self._refresh_market_data()
            balance = await self.broker.get_balance()
            positions = await self.broker.get_positions()

            print(f"    Balance: ${balance:,.2f}")
            print(f"    Open positions: {len(positions)}")
            print(f"    Markets loaded: {len(self._markets_cache)}")

            # Convert to MarketState for agents
            market_states = await self._build_market_states()

            # Update indicators
            if self.config.enable_indicators:
                await self._update_indicators()
                print(f"    Indicators updated: {len(self._indicator_cache)}")

            # Scan for arbitrage
            if self.arbitrage:
                await self._scan_arbitrage()
                print(f"    Arbitrage opps: {len(self._arbitrage_cache)}")

            # Phase 2: MULTI-AGENT DECISION
            print("\n[2] MULTI-AGENT ANALYSIS")
            decision = await self.coordinator.decide(
                markets=market_states,
                portfolio_value=balance + sum(p.size * p.current_price for p in positions),
                positions=positions,
            )

            print(f"    Signals generated: {len(decision.signals)}")
            print(f"    Consensus confidence: {decision.consensus_confidence:.2%}")
            print(f"    Debate rounds: {decision.debate_rounds}")
            print(f"    Total tokens: {decision.total_tokens:,}")
            print(f"    Latency: {decision.total_latency_ms}ms")

            # Phase 3: EXECUTION
            print("\n[3] EXECUTION")
            executed_orders = await self._execute_signals(decision, balance, positions)
            print(f"    Orders executed: {len(executed_orders)}")

            # Phase 4: LOGGING & LEARNING
            print("\n[4] LOGGING & LEARNING")
            await self._log_cycle(cycle_start, balance, positions, decision, executed_orders)

            # Run reflection on completed trades
            if self.config.enable_reflection:
                await self._run_reflection()

            # Store decision in memory
            if self.config.enable_memory:
                await self._store_decision_memory(decision, market_states)

            cycle_duration = (datetime.utcnow() - cycle_start).total_seconds()
            print(f"\n    Cycle complete: {cycle_duration:.1f}s")

        except Exception as e:
            print(f"\n[ERROR] Cycle failed: {e}")
            import traceback
            traceback.print_exc()

    async def _refresh_market_data(self) -> None:
        """Refresh market data from broker."""
        now = datetime.utcnow()

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

            # Limit markets per cycle
            self._markets_cache = self._markets_cache[:self.config.max_markets_per_cycle]
            self._markets_updated = now

    async def _build_market_states(self) -> list[MarketState]:
        """Convert MarketInfo to MarketState for agents."""
        states = []

        for market in self._markets_cache:
            # Get indicators if available
            indicator = self._indicator_cache.get(market.condition_id)

            # Build simplified order book
            order_book = OrderBook(
                best_bid=market.best_bid,
                best_ask=market.best_ask,
                bid_depth=market.liquidity / 2,  # Simplified
                ask_depth=market.liquidity / 2,
                spread=market.best_ask - market.best_bid if market.best_ask and market.best_bid else Decimal("0.01"),
            )

            states.append(MarketState(
                market_id=market.condition_id,
                question=market.question,
                yes_token_id=market.yes_token_id,
                no_token_id=market.no_token_id,
                yes_price=market.yes_price,
                no_price=market.no_price,
                volume_24h=market.volume_24h,
                liquidity=market.liquidity,
                order_book=order_book,
                indicators=indicator,
                end_date=market.end_date,
            ))

        return states

    async def _update_indicators(self) -> None:
        """Update technical indicators for all markets."""
        for market in self._markets_cache:
            # Update indicator calculator with latest price
            self.indicators.update(
                token_id=market.condition_id,
                price=market.yes_price,
                volume=market.volume_24h,
                high=market.yes_price,  # Simplified
                low=market.yes_price,
            )

            # Calculate indicators
            result = self.indicators.calculate(market.condition_id)
            self._indicator_cache[market.condition_id] = result

    async def _scan_arbitrage(self) -> None:
        """Scan for cross-platform arbitrage opportunities."""
        if not self.arbitrage:
            return

        await self.arbitrage.refresh_markets()
        self._arbitrage_cache = await self.arbitrage.detect_all()

        # Log any profitable opportunities
        for opp in self._arbitrage_cache:
            if opp.is_profitable():
                print(f"    [ARB] {opp.type}: {opp.net_profit_pct:.2f}% net profit")

    async def _execute_signals(
        self,
        decision: CoordinatorDecision,
        balance: Decimal,
        positions: list[Position],
    ) -> list[Order]:
        """Execute trading signals with risk checks."""
        executed = []

        for signal in decision.signals:
            # Skip if confidence too low
            if signal.confidence.overall < Decimal("0.6"):
                print(f"    Skipped {signal.market_id[:8]}: low confidence ({signal.confidence.overall:.1%})")
                continue

            # Check position limits
            existing_exposure = sum(
                p.size * p.current_price
                for p in positions
                if p.market_id == signal.market_id
            )

            max_exposure = balance * Decimal("0.1")  # 10% max per market
            if existing_exposure >= max_exposure:
                print(f"    Skipped {signal.market_id[:8]}: max exposure reached")
                continue

            # Adjust size if needed
            size = min(signal.size_recommendation, max_exposure - existing_exposure)
            if size < Decimal("10"):  # Min $10 order
                continue

            try:
                # Create episode for tracking
                episode = Episode(
                    episode_id=signal.id,
                    market_id=signal.market_id,
                    entry_signal=signal,
                )
                self._active_episodes[signal.market_id] = episode

                # Place order
                order = await self.broker.place_order(
                    market_id=signal.market_id,
                    token_id=signal.token_id,
                    side="buy",
                    size=size,
                    price=signal.target_price,
                )

                executed.append(order)

                print(f"    Executed: {signal.market_id[:8]} BUY {size:.2f} @ {signal.target_price:.3f}")

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
                print(f"    Order failed {signal.market_id[:8]}: {e}")

        return executed

    async def _log_cycle(
        self,
        cycle_start: datetime,
        balance: Decimal,
        positions: list[Position],
        decision: CoordinatorDecision,
        executed_orders: list[Order],
    ) -> None:
        """Log cycle results."""
        # Decision log
        await self.metrics.log_decision(DecisionLog(
            timestamp=cycle_start,
            iteration=self._iteration + 1,
            balance=balance,
            positions_count=len(positions),
            markets_analyzed=len(self._markets_cache),
            signals_generated=len(decision.signals),
            signals_executed=len(executed_orders),
            model_used="multi-agent",
            latency_ms=decision.total_latency_ms,
            tokens_used=decision.total_tokens,
            reasoning=str(decision.execution_plan),
            signals=[s.to_dict() for s in decision.signals],
        ))

        # PnL snapshot
        unrealized = sum(p.unrealized_pnl for p in positions)
        realized = sum(p.realized_pnl for p in positions)
        positions_value = sum(p.size * p.current_price for p in positions)
        total_equity = balance + positions_value

        await self.metrics.log_pnl_snapshot(PnLSnapshot(
            timestamp=datetime.utcnow(),
            balance=balance,
            unrealized_pnl=unrealized,
            realized_pnl=realized,
            total_equity=total_equity,
            positions_value=positions_value,
            drawdown_pct=self.metrics.calculate_drawdown(total_equity),
            high_water_mark=self.metrics._high_water_mark,
        ))

    async def _run_reflection(self) -> None:
        """Run reflection on closed trades."""
        await self.coordinator.reflect_on_trades()

    async def _store_decision_memory(
        self,
        decision: CoordinatorDecision,
        markets: list[MarketState],
    ) -> None:
        """Store decision context in memory."""
        # Short-term: recent decision
        self.memory.short_term.add({
            "type": "decision",
            "timestamp": datetime.utcnow().isoformat(),
            "signals_count": len(decision.signals),
            "consensus": float(decision.consensus_confidence),
            "markets": [m.market_id for m in markets[:5]],
        })

        # Long-term: significant decisions
        if decision.signals and decision.consensus_confidence > Decimal("0.7"):
            for signal in decision.signals:
                await self.memory.long_term.store(
                    content=f"Signal for {signal.market_id}: {signal.signal_type.value} "
                           f"at {signal.target_price} with {signal.confidence.overall:.1%} confidence. "
                           f"Edge: {signal.edge.expected_value:.2%} EV, Kelly: {signal.edge.kelly_fraction:.2%}",
                    metadata={
                        "type": "trading_signal",
                        "market_id": signal.market_id,
                        "signal_type": signal.signal_type.value,
                        "confidence": float(signal.confidence.overall),
                    },
                )

    def _setup_signal_handlers(self) -> None:
        """Setup graceful shutdown."""
        loop = asyncio.get_event_loop()

        def handle_signal():
            asyncio.create_task(self.stop())

        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, handle_signal)

    async def _cleanup(self) -> None:
        """Cleanup on shutdown."""
        print("\nCleaning up...")

        # Disconnect arbitrage platforms
        if self.arbitrage:
            for platform in self.arbitrage.platforms.values():
                await platform.disconnect()

        # Run handlers
        for handler in self._shutdown_handlers:
            try:
                handler()
            except Exception:
                pass

        print("Cleanup complete")

    def add_shutdown_handler(self, handler: Callable) -> None:
        """Add cleanup handler."""
        self._shutdown_handlers.append(handler)

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def iteration(self) -> int:
        return self._iteration

    def get_status(self) -> dict[str, Any]:
        """Get current runner status."""
        return {
            "running": self._running,
            "iteration": self._iteration,
            "markets_cached": len(self._markets_cache),
            "indicators_cached": len(self._indicator_cache),
            "arbitrage_opportunities": len(self._arbitrage_cache),
            "active_episodes": len(self._active_episodes),
            "memory_stats": {
                "short_term": self.memory.short_term.size,
            },
        }


async def create_enhanced_runner(
    broker: Broker,
    market_data: MarketData,
    llm_config: dict[str, LLMProviderConfig],
    config: EnhancedRunnerConfig | None = None,
) -> EnhancedTradingRunner:
    """
    Factory function to create an enhanced runner with all components.

    This sets up:
    - Provider registry with all configured LLMs
    - Memory system
    - Multi-agent coordinator
    - Technical indicators
    - Metrics logger
    """
    from ..core.config import AgentConfig, MemoryConfig, IndicatorConfig

    config = config or EnhancedRunnerConfig()

    # Setup provider registry
    registry = ProviderRegistry()
    for provider_name, provider_config in llm_config.items():
        await registry.register_from_config(provider_config)

    # Setup memory
    memory_config = MemoryConfig(
        db_path=config.memory_db_path,
        short_term_max_items=100,
        short_term_ttl_seconds=3600,
    )
    memory = MemoryManager(memory_config)

    # Setup coordinator
    agent_config = AgentConfig(
        research_agent_model="claude-sonnet-4-20250514",
        risk_agent_model="claude-sonnet-4-20250514",
        execution_agent_model="claude-haiku-3-5-20241022",
        reflection_agent_model="claude-sonnet-4-20250514",
        enable_multi_agent_debate=True,
        debate_rounds=2,
        enable_reflection=config.enable_reflection,
    )
    coordinator = AgentCoordinator(registry, memory, agent_config)

    # Setup indicators
    indicator_config = IndicatorConfig()
    indicators = IndicatorCalculator(indicator_config)

    # Setup metrics
    metrics = MetricsLogger(
        jsonl_path=config.metrics_jsonl_path,
        sqlite_path=config.metrics_sqlite_path,
    )
    await metrics.initialize()

    # Setup arbitrage platforms
    arbitrage_platforms = None
    if config.enable_arbitrage:
        arbitrage_platforms = [
            PolymarketPlatform(),
            # KalshiPlatform(),  # Add with API keys
        ]

    return EnhancedTradingRunner(
        broker=broker,
        market_data=market_data,
        coordinator=coordinator,
        memory=memory,
        indicators=indicators,
        metrics_logger=metrics,
        config=config,
        arbitrage_platforms=arbitrage_platforms,
    )
