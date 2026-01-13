"""
Integration Tests for Alpha Arena Trading System.

These tests verify that all components work together correctly:
- Multi-agent coordination (Research, Risk, Execution, Reflection)
- Debate system integration
- Signal aggregation
- Full trading loop
"""

import asyncio
import json
import pytest
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from src.core.types import (
    MarketState, OrderBook, PriceLevel, Signal, SignalType,
    Edge, Confidence
)
from src.core.config import AgentConfig
from src.agents.coordinator import AgentCoordinator, CoordinatorDecision
from src.agents.base import AgentRole, AgentResponse
from src.agents.debate import (
    DebateOrchestrator, DebatePersona, DebatePosition,
    DebateResult, ConfidenceCalibrator
)
from src.signals.aggregator import SignalAggregator, SignalSource, AggregatedSignal
from src.signals.events import EventCalendar, MarketEvent, EventType, EventImpact

pytestmark = [pytest.mark.integration]


# ============================================================================
# Mock LLM Provider
# ============================================================================

class MockLLMResponse:
    """Mock LLM response object matching LLMResponse interface."""
    def __init__(self, content: str, tokens_used: int = 100):
        self.content = content
        self.model = "mock-model"
        self.provider = "mock"
        self.tokens_input = tokens_used // 2
        self.tokens_output = tokens_used // 2
        self.tokens_used = tokens_used
        self.latency_ms = 50
        self.cost_estimate = Decimal("0.001")
        self.finish_reason = "stop"
        self.timestamp = datetime.utcnow()
        self.metadata = {}

    @property
    def total_tokens(self) -> int:
        return self.tokens_input + self.tokens_output


class MockLLMProvider:
    """Mock LLM provider that returns structured responses for agents."""

    def __init__(self, response_map: dict[str, str] | None = None):
        self.response_map = response_map or {}
        self.call_history: list[dict] = []
        self.default_model = "mock-model"
        self.call_count = 0

    async def complete(
        self,
        messages: list | None = None,
        system: str | None = None,
        json_mode: bool = False,
        **kwargs
    ) -> MockLLMResponse:
        """Return mock response based on system prompt."""
        self.call_count += 1

        # Extract prompt from messages if provided
        prompt = ""
        if messages:
            prompt = str(messages[0].content if hasattr(messages[0], 'content') else messages[0])[:200]

        self.call_history.append({
            "prompt": prompt,
            "system": system[:100] if system else None,
        })

        # Determine response based on system prompt
        content = self._get_response_content(system)
        return MockLLMResponse(content=content, tokens_used=100)

    def _get_response_content(self, system_prompt: str | None) -> str:
        """Get response content based on system prompt."""
        if system_prompt:
            system_lower = system_prompt.lower()
            if "research" in system_lower:
                return self._research_response()
            elif "risk" in system_lower:
                return self._risk_response()
            elif "execution" in system_lower:
                return self._execution_response()
            elif "reflection" in system_lower:
                return self._reflection_response()
            elif "debate" in system_lower:
                return self._debate_response()

        # Default response
        return json.dumps({"response": "mock response"})

    def _research_response(self) -> str:
        """Mock research agent response."""
        return json.dumps({
            "opportunities": [
                {
                    "market_id": "test_market_001",
                    "question": "Will test event occur?",
                    "direction": "YES",
                    "current_price": 0.45,
                    "fair_value_estimate": 0.65,
                    "confidence": 0.75,
                    "edge_percentage": 20,
                    "edge_source": "fundamental_analysis",
                    "reasoning": "Strong fundamentals suggest YES outcome",
                    "catalyst": "Upcoming announcement",
                    "timeframe": "1-2 weeks",
                },
            ],
            "market_analysis": "Market showing bullish signals",
            "risk_factors": ["Market volatility", "Low liquidity"],
        })

    def _risk_response(self) -> str:
        """Mock risk agent response."""
        return json.dumps({
            "approved_trades": [
                {
                    "market_id": "test_market_001",
                    "approved": True,
                    "recommended_size_usdc": 25,
                    "max_size_usdc": 50,
                    "kelly_fraction": 0.15,
                    "risk_score": 0.3,
                    "reasoning": "Acceptable risk-reward ratio",
                },
            ],
            "portfolio_risk": {
                "current_exposure": 0.25,
                "max_recommended": 0.5,
                "correlation_risk": "low",
            },
            "warnings": [],
        })

    def _execution_response(self) -> str:
        """Mock execution agent response."""
        return json.dumps({
            "execution_plan": [
                {
                    "market_id": "test_market_001",
                    "strategy": "limit_order",
                    "entry_price": 0.44,
                    "slippage_estimate": 0.01,
                    "time_in_force": "GTC",
                    "split_orders": False,
                    "urgency": "normal",
                },
            ],
            "market_conditions": "favorable",
            "recommended_timing": "immediate",
        })

    def _reflection_response(self) -> str:
        """Mock reflection agent response."""
        return json.dumps({
            "lessons_learned": [
                "Entry timing was optimal",
                "Position sizing was appropriate",
            ],
            "strategy_adjustments": [],
            "confidence_calibration": 0.82,
        })

    def _debate_response(self) -> str:
        """Mock debate response."""
        return json.dumps({
            "position": "bullish",
            "confidence": 0.72,
            "arguments": [
                "Strong market momentum",
                "Favorable risk-reward",
            ],
            "counterarguments_addressed": [
                "Volatility risk is manageable",
            ],
        })


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_llm_provider():
    """Create mock LLM provider."""
    return MockLLMProvider()


@pytest.fixture
def mock_provider_registry(mock_llm_provider):
    """Create mock provider registry."""
    registry = MagicMock()
    registry.get_for_model.return_value = (mock_llm_provider, "mock-model")
    registry.get_active.return_value = mock_llm_provider
    return registry


@pytest.fixture
def mock_memory_manager():
    """Create mock memory manager."""
    memory = MagicMock()
    memory.get_context_for_decision = AsyncMock(return_value="Historical context: Previous trades were profitable")
    memory.get_performance_context = AsyncMock(return_value="Win rate: 65%, Sharpe: 1.2")
    memory.get_episodes_for_reflection = AsyncMock(return_value=[])
    memory.store_decision = AsyncMock()
    memory.update_calibration = AsyncMock()
    return memory


@pytest.fixture
def agent_config():
    """Create agent configuration."""
    return AgentConfig(
        research_agent_model="mock-model",
        risk_agent_model="mock-model",
        execution_agent_model="mock-model",
        reflection_agent_model="mock-model",
        enable_multi_agent_debate=True,
        debate_rounds=2,
        enable_reflection=True,
    )


@pytest.fixture
def sample_markets():
    """Create sample market states for testing."""
    return [
        MarketState(
            market_id="test_market_001",
            condition_id="cond_001",
            question="Will test event occur by end of month?",
            category="crypto",
            yes_price=Decimal("0.45"),
            no_price=Decimal("0.55"),
            yes_token_id="yes_token_001",
            no_token_id="no_token_001",
            volume_24h=Decimal("50000"),
            liquidity=Decimal("25000"),
            end_date=datetime.utcnow() + timedelta(days=30),
            yes_book=OrderBook(
                token_id="yes_token_001",
                bids=[PriceLevel(price=Decimal("0.44"), size=Decimal("1000"))],
                asks=[PriceLevel(price=Decimal("0.46"), size=Decimal("1000"))],
            ),
            no_book=OrderBook(
                token_id="no_token_001",
                bids=[PriceLevel(price=Decimal("0.54"), size=Decimal("1000"))],
                asks=[PriceLevel(price=Decimal("0.56"), size=Decimal("1000"))],
            ),
        ),
        MarketState(
            market_id="test_market_002",
            condition_id="cond_002",
            question="Will secondary event happen?",
            category="politics",
            yes_price=Decimal("0.60"),
            no_price=Decimal("0.40"),
            yes_token_id="yes_token_002",
            no_token_id="no_token_002",
            volume_24h=Decimal("30000"),
            liquidity=Decimal("15000"),
            end_date=datetime.utcnow() + timedelta(days=14),
            yes_book=OrderBook(
                token_id="yes_token_002",
                bids=[PriceLevel(price=Decimal("0.59"), size=Decimal("500"))],
                asks=[PriceLevel(price=Decimal("0.61"), size=Decimal("500"))],
            ),
            no_book=OrderBook(
                token_id="no_token_002",
                bids=[PriceLevel(price=Decimal("0.39"), size=Decimal("500"))],
                asks=[PriceLevel(price=Decimal("0.41"), size=Decimal("500"))],
            ),
        ),
    ]


# ============================================================================
# Integration Tests: Agent Coordinator
# ============================================================================

class TestAgentCoordinatorIntegration:
    """Test full agent coordination flow."""

    @pytest.mark.asyncio
    async def test_full_decision_cycle(
        self,
        mock_provider_registry,
        mock_memory_manager,
        agent_config,
        sample_markets,
    ):
        """
        Test complete decision cycle:
        Research -> Risk -> (Debate) -> Execution -> Final Decision
        """
        coordinator = AgentCoordinator(
            provider_registry=mock_provider_registry,
            memory=mock_memory_manager,
            config=agent_config,
        )

        await coordinator.initialize()

        decision = await coordinator.decide(
            markets=sample_markets,
            portfolio_value=Decimal("10000"),
            positions=[],
        )

        # Verify decision structure
        assert isinstance(decision, CoordinatorDecision)
        assert decision.total_latency_ms >= 0
        assert decision.total_tokens >= 0

        # Verify all agents participated
        assert AgentRole.RESEARCH in decision.agent_responses
        assert AgentRole.RISK in decision.agent_responses
        assert AgentRole.EXECUTION in decision.agent_responses

        # Verify signals were generated
        assert isinstance(decision.signals, list)

        # Verify consensus confidence is reasonable
        assert Decimal("0") <= decision.consensus_confidence <= Decimal("1")

    @pytest.mark.asyncio
    async def test_decision_with_no_opportunities(
        self,
        mock_memory_manager,
        agent_config,
        sample_markets,
    ):
        """Test decision when research finds no opportunities."""
        # Create provider that returns empty opportunities
        provider = MockLLMProvider()
        provider._research_response = lambda: json.dumps({
            "opportunities": [],
            "market_analysis": "No compelling opportunities found",
        })

        registry = MagicMock()
        registry.get_for_model.return_value = (provider, "mock-model")

        coordinator = AgentCoordinator(
            provider_registry=registry,
            memory=mock_memory_manager,
            config=agent_config,
        )

        await coordinator.initialize()

        decision = await coordinator.decide(
            markets=sample_markets,
            portfolio_value=Decimal("10000"),
            positions=[],
        )

        # Should return early with no signals
        assert decision.signals == []
        assert decision.consensus_confidence == Decimal("0")

    @pytest.mark.asyncio
    async def test_decision_with_existing_positions(
        self,
        mock_provider_registry,
        mock_memory_manager,
        agent_config,
        sample_markets,
    ):
        """Test decision when portfolio has existing positions."""
        from src.broker.base import Position

        existing_positions = [
            Position(
                market_id="existing_001",
                token_id="token_001",
                outcome="YES",
                size=Decimal("100"),
                avg_entry_price=Decimal("0.50"),
                current_price=Decimal("0.55"),
                unrealized_pnl=Decimal("5"),
            ),
        ]

        coordinator = AgentCoordinator(
            provider_registry=mock_provider_registry,
            memory=mock_memory_manager,
            config=agent_config,
        )

        await coordinator.initialize()

        decision = await coordinator.decide(
            markets=sample_markets,
            portfolio_value=Decimal("10000"),
            positions=existing_positions,
        )

        # Decision should still work with existing positions
        assert isinstance(decision, CoordinatorDecision)
        assert AgentRole.RESEARCH in decision.agent_responses


# ============================================================================
# Integration Tests: Debate System
# ============================================================================

class TestDebateSystemIntegration:
    """Test debate system integration with trading decisions."""

    @pytest.mark.asyncio
    async def test_debate_with_multiple_personas(self, mock_llm_provider, sample_markets):
        """Test debate orchestrator with multiple personas."""
        # Create providers dict for all personas
        providers = {
            DebatePersona.OPTIMIST: mock_llm_provider,
            DebatePersona.PESSIMIST: mock_llm_provider,
            DebatePersona.FUNDAMENTALIST: mock_llm_provider,
            DebatePersona.DEVILS_ADVOCATE: mock_llm_provider,
        }

        orchestrator = DebateOrchestrator(
            providers=providers,
            max_rounds=2,
        )

        # Use correct method signature: debate() not run_debate()
        result = await orchestrator.debate(
            market=sample_markets[0],
            context="Market context for debate",
        )

        assert isinstance(result, DebateResult)
        assert result.total_rounds >= 0  # Correct attribute name

    def test_confidence_calibrator_tracking(self):
        """Test confidence calibrator tracks predictions correctly."""
        calibrator = ConfidenceCalibrator()

        # Add predictions using correct signature
        calibrator.record_prediction(
            market_id="market_001",
            persona=DebatePersona.OPTIMIST,
            predicted_prob=Decimal("0.75"),
            confidence=Decimal("0.8"),
            timestamp=datetime.utcnow(),
        )
        calibrator.record_prediction(
            market_id="market_002",
            persona=DebatePersona.PESSIMIST,
            predicted_prob=Decimal("0.40"),
            confidence=Decimal("0.7"),
            timestamp=datetime.utcnow(),
        )

        # Record outcomes (bool: True=YES resolved, False=NO resolved)
        calibrator.record_outcome("market_001", outcome=True)  # YES
        calibrator.record_outcome("market_002", outcome=False)  # NO

        # Check calibration report
        report = calibrator.get_calibration_report()
        assert "total_predictions" in report
        assert report["total_predictions"] == 2


# ============================================================================
# Integration Tests: Signal Aggregation
# ============================================================================

class TestSignalAggregationIntegration:
    """Test signal aggregation with multiple signal sources."""

    @pytest.mark.asyncio
    async def test_signal_aggregation_flow(self, sample_markets):
        """Test signal aggregation from multiple sources."""
        # Create aggregator with mock dependencies
        aggregator = SignalAggregator(
            event_calendar=None,  # Will use default
            news_provider=None,   # Will use default
        )

        market = sample_markets[0]

        # Aggregate signals for market using correct signature
        result = await aggregator.aggregate_signals(
            market_id=market.market_id,
            market_question=market.question,
            current_price=market.yes_price,
            fetch_news=False,  # Don't try to fetch real news in test
            check_events=True,
        )

        # Result should have aggregated signal info
        assert result is not None
        assert isinstance(result, AggregatedSignal)


# ============================================================================
# Integration Tests: Event Calendar
# ============================================================================

class TestEventCalendarIntegration:
    """Test event calendar integration."""

    def test_event_calendar_add_and_get(self):
        """Test adding and retrieving events."""
        calendar = EventCalendar()

        # Add event with correct signature
        event = MarketEvent(
            event_type=EventType.EARNINGS,
            title="Q4 Earnings Release",
            description="Company quarterly earnings announcement",
            timestamp=datetime.utcnow() + timedelta(hours=2),
            impact=EventImpact.HIGH,
            related_markets=["test_market_001"],
            related_keywords=["earnings", "revenue"],
        )
        calendar.add_event(event)

        # Get upcoming events
        upcoming = calendar.get_upcoming_events(hours_ahead=24)
        assert len(upcoming) >= 1

    def test_event_calendar_market_impact(self):
        """Test getting events for a market."""
        calendar = EventCalendar()

        # Add high-impact event (using correct EventType enum value)
        event = MarketEvent(
            event_type=EventType.FED_MEETING,
            title="Fed Rate Decision",
            description="Federal Reserve interest rate announcement",
            timestamp=datetime.utcnow() + timedelta(hours=1),
            impact=EventImpact.HIGH,
            related_keywords=["fed", "interest", "rate"],
        )
        calendar.add_event(event)

        # Get events for a market - should find related event by keyword
        events = calendar.get_events_for_market(
            market_question="Will the Fed raise rates?"
        )
        # Should find the Fed event
        assert isinstance(events, list)
        assert len(events) >= 1


# ============================================================================
# Integration Tests: Full Trading Loop
# ============================================================================

class TestFullTradingLoopIntegration:
    """Test the complete trading loop integration."""

    @pytest.mark.asyncio
    async def test_end_to_end_trading_cycle(
        self,
        mock_provider_registry,
        mock_memory_manager,
        agent_config,
        sample_markets,
        mock_broker,
    ):
        """
        Test complete trading cycle:
        1. Agent coordination produces signals
        2. Signals are validated
        3. Orders would be placed (mocked)
        4. Position tracking updates
        """
        # Step 1: Initialize coordinator
        coordinator = AgentCoordinator(
            provider_registry=mock_provider_registry,
            memory=mock_memory_manager,
            config=agent_config,
        )
        await coordinator.initialize()

        # Step 2: Get trading decision
        decision = await coordinator.decide(
            markets=sample_markets,
            portfolio_value=Decimal("10000"),
            positions=[],
        )

        # Step 3: Validate signals
        for signal in decision.signals:
            assert signal.market_id in [m.market_id for m in sample_markets]
            assert signal.size_recommendation > Decimal("0")
            assert signal.confidence.overall >= Decimal("0")

        # Step 4: Execute trades (mocked)
        for signal in decision.signals:
            if signal.signal_type == SignalType.BUY:
                order = await mock_broker.place_order(
                    market_id=signal.market_id,
                    token_id=signal.token_id,
                    side="buy",
                    size=signal.size_recommendation,
                    price=signal.target_price,
                )
                assert order["status"] == "filled"

        # Step 5: Verify broker state
        assert len(mock_broker.orders) == len(decision.signals)

    @pytest.mark.asyncio
    async def test_trading_loop_with_risk_rejection(
        self,
        mock_memory_manager,
        agent_config,
        sample_markets,
    ):
        """Test trading loop when risk agent rejects trades."""
        # Provider that returns no approved trades
        provider = MockLLMProvider()
        provider._risk_response = lambda: json.dumps({
            "approved_trades": [
                {
                    "market_id": "test_market_001",
                    "approved": False,
                    "reason": "Risk too high",
                    "risk_score": 0.85,
                },
            ],
            "warnings": ["Portfolio exposure exceeded"],
        })

        registry = MagicMock()
        registry.get_for_model.return_value = (provider, "mock-model")

        coordinator = AgentCoordinator(
            provider_registry=registry,
            memory=mock_memory_manager,
            config=agent_config,
        )
        await coordinator.initialize()

        decision = await coordinator.decide(
            markets=sample_markets,
            portfolio_value=Decimal("10000"),
            positions=[],
        )

        # No signals should be generated if risk rejects
        assert decision.signals == []

    @pytest.mark.asyncio
    async def test_trading_loop_with_debate(
        self,
        mock_provider_registry,
        mock_memory_manager,
        sample_markets,
    ):
        """Test trading loop with debate enabled."""
        config = AgentConfig(
            research_agent_model="mock-model",
            risk_agent_model="mock-model",
            execution_agent_model="mock-model",
            reflection_agent_model="mock-model",
            enable_multi_agent_debate=True,
            debate_rounds=2,
            enable_reflection=True,
        )

        coordinator = AgentCoordinator(
            provider_registry=mock_provider_registry,
            memory=mock_memory_manager,
            config=config,
        )
        await coordinator.initialize()

        decision = await coordinator.decide(
            markets=sample_markets,
            portfolio_value=Decimal("10000"),
            positions=[],
        )

        # With debate enabled, should track rounds
        assert isinstance(decision.debate_rounds, int)


# ============================================================================
# Integration Tests: Error Handling
# ============================================================================

class TestErrorHandlingIntegration:
    """Test error handling across integrated components."""

    @pytest.mark.asyncio
    async def test_llm_failure_raises_exception(
        self,
        mock_memory_manager,
        agent_config,
        sample_markets,
    ):
        """Test system handles LLM failures by raising exception."""
        # Provider that raises exception
        provider = MockLLMProvider()
        provider.complete = AsyncMock(side_effect=Exception("LLM API Error"))

        registry = MagicMock()
        registry.get_for_model.return_value = (provider, "mock-model")

        coordinator = AgentCoordinator(
            provider_registry=registry,
            memory=mock_memory_manager,
            config=agent_config,
        )
        await coordinator.initialize()

        # Should raise exception on LLM failure
        with pytest.raises(Exception):
            await coordinator.decide(
                markets=sample_markets,
                portfolio_value=Decimal("10000"),
                positions=[],
            )


# ============================================================================
# Integration Tests: Performance
# ============================================================================

class TestPerformanceIntegration:
    """Test performance-related integration scenarios."""

    @pytest.mark.asyncio
    async def test_decision_latency_tracking(
        self,
        mock_provider_registry,
        mock_memory_manager,
        agent_config,
        sample_markets,
    ):
        """Test that latency is properly tracked."""
        coordinator = AgentCoordinator(
            provider_registry=mock_provider_registry,
            memory=mock_memory_manager,
            config=agent_config,
        )
        await coordinator.initialize()

        decision = await coordinator.decide(
            markets=sample_markets,
            portfolio_value=Decimal("10000"),
            positions=[],
        )

        # Latency should be tracked
        assert decision.total_latency_ms >= 0

    @pytest.mark.asyncio
    async def test_token_usage_tracking(
        self,
        mock_provider_registry,
        mock_memory_manager,
        agent_config,
        sample_markets,
    ):
        """Test that token usage is properly tracked."""
        coordinator = AgentCoordinator(
            provider_registry=mock_provider_registry,
            memory=mock_memory_manager,
            config=agent_config,
        )
        await coordinator.initialize()

        decision = await coordinator.decide(
            markets=sample_markets,
            portfolio_value=Decimal("10000"),
            positions=[],
        )

        # Token usage should be tracked
        assert decision.total_tokens >= 0
