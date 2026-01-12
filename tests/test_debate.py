"""Tests for the multi-agent debate system."""

import asyncio
import pytest
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

from src.agents.debate import (
    DebatePersona,
    DebatePosition,
    DebateRound,
    DebateResult,
    DebateAgent,
    DebateOrchestrator,
    ConfidenceCalibrator,
    get_confidence_calibrator,
)
from src.core.types import MarketState, OrderBook, PriceLevel
from src.llm.base import LLMResponse


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def mock_market():
    """Create a mock market state for testing."""
    return MarketState(
        market_id="test_market_123",
        condition_id="cond_123",
        question="Will Bitcoin exceed $100,000 by end of 2025?",
        category="crypto",
        end_date=datetime(2025, 12, 31),
        yes_token_id="yes_123",
        no_token_id="no_123",
        yes_price=Decimal("0.45"),
        no_price=Decimal("0.55"),
        volume_24h=Decimal("50000"),
        liquidity=Decimal("100000"),
        yes_book=OrderBook(
            token_id="yes_123",
            bids=[PriceLevel(price=Decimal("0.44"), size=Decimal("1000"))],
            asks=[PriceLevel(price=Decimal("0.46"), size=Decimal("1000"))],
        ),
        no_book=OrderBook(
            token_id="no_123",
            bids=[PriceLevel(price=Decimal("0.54"), size=Decimal("1000"))],
            asks=[PriceLevel(price=Decimal("0.56"), size=Decimal("1000"))],
        ),
        indicators={"rsi": 55, "macd_signal": 0.02},
        sentiment={"score": 0.3, "label": "bullish"},
    )


@pytest.fixture
def mock_llm_provider():
    """Create a mock LLM provider."""
    provider = MagicMock()
    provider.default_model = "test-model"

    async def mock_complete(*args, **kwargs):
        return LLMResponse(
            content='{"direction": "yes", "fair_value": 0.55, "confidence": 0.7, "reasoning": "Test reasoning", "key_arguments": ["arg1", "arg2"], "counter_arguments": ["counter1"], "risk_factors": ["risk1"]}',
            model="test-model",
            provider="test",
            tokens_input=100,
            tokens_output=50,
            latency_ms=100,
            cost_estimate=Decimal("0.001"),
            finish_reason="stop",
        )

    provider.complete = mock_complete
    return provider


@pytest.fixture
def mock_providers(mock_llm_provider):
    """Create mock providers for all personas."""
    return {
        DebatePersona.OPTIMIST: mock_llm_provider,
        DebatePersona.PESSIMIST: mock_llm_provider,
        DebatePersona.FUNDAMENTALIST: mock_llm_provider,
        DebatePersona.TECHNICIAN: mock_llm_provider,
        DebatePersona.DEVILS_ADVOCATE: mock_llm_provider,
    }


# =============================================================================
# DebatePersona Tests
# =============================================================================

class TestDebatePersona:
    """Tests for DebatePersona enum."""

    def test_all_personas_exist(self):
        assert DebatePersona.OPTIMIST.value == "optimist"
        assert DebatePersona.PESSIMIST.value == "pessimist"
        assert DebatePersona.FUNDAMENTALIST.value == "fundamentalist"
        assert DebatePersona.TECHNICIAN.value == "technician"
        assert DebatePersona.DEVILS_ADVOCATE.value == "devils_advocate"
        assert DebatePersona.NEUTRAL.value == "neutral"

    def test_persona_count(self):
        assert len(DebatePersona) == 6


# =============================================================================
# DebatePosition Tests
# =============================================================================

class TestDebatePosition:
    """Tests for DebatePosition dataclass."""

    @pytest.fixture
    def position(self):
        return DebatePosition(
            persona=DebatePersona.OPTIMIST,
            model="test-model",
            direction="yes",
            fair_value=Decimal("0.65"),
            confidence=Decimal("0.8"),
            reasoning="Strong bullish case based on momentum",
            key_arguments=["arg1", "arg2"],
            counter_arguments=["counter1"],
            risk_factors=["risk1", "risk2"],
        )

    def test_position_creation(self, position):
        assert position.persona == DebatePersona.OPTIMIST
        assert position.direction == "yes"
        assert position.fair_value == Decimal("0.65")
        assert position.confidence == Decimal("0.8")

    def test_position_to_dict(self, position):
        d = position.to_dict()
        assert d["persona"] == "optimist"
        assert d["direction"] == "yes"
        assert d["fair_value"] == 0.65
        assert d["confidence"] == 0.8
        assert "timestamp" in d
        assert len(d["key_arguments"]) == 2


# =============================================================================
# DebateRound Tests
# =============================================================================

class TestDebateRound:
    """Tests for DebateRound dataclass."""

    @pytest.fixture
    def debate_round(self):
        positions = [
            DebatePosition(
                persona=DebatePersona.OPTIMIST,
                model="test",
                direction="yes",
                fair_value=Decimal("0.6"),
                confidence=Decimal("0.7"),
                reasoning="Bullish",
                key_arguments=["arg1"],
                counter_arguments=[],
                risk_factors=[],
            ),
            DebatePosition(
                persona=DebatePersona.PESSIMIST,
                model="test",
                direction="no",
                fair_value=Decimal("0.4"),
                confidence=Decimal("0.6"),
                reasoning="Bearish",
                key_arguments=["arg2"],
                counter_arguments=[],
                risk_factors=[],
            ),
        ]
        return DebateRound(
            round_number=1,
            positions=positions,
            rebuttals={DebatePersona.OPTIMIST: "My rebuttal"},
            consensus_direction="yes",
            consensus_confidence=Decimal("0.55"),
            disagreement_level=Decimal("0.4"),
        )

    def test_round_creation(self, debate_round):
        assert debate_round.round_number == 1
        assert len(debate_round.positions) == 2
        assert debate_round.consensus_direction == "yes"

    def test_round_to_dict(self, debate_round):
        d = debate_round.to_dict()
        assert d["round_number"] == 1
        assert len(d["positions"]) == 2
        assert "optimist" in d["rebuttals"]
        assert d["consensus_confidence"] == 0.55


# =============================================================================
# DebateResult Tests
# =============================================================================

class TestDebateResult:
    """Tests for DebateResult dataclass."""

    @pytest.fixture
    def debate_result(self):
        return DebateResult(
            market_id="test_123",
            market_question="Will X happen?",
            rounds=[],
            final_direction="yes",
            final_confidence=Decimal("0.75"),
            final_fair_value=Decimal("0.6"),
            consensus_strength=Decimal("0.8"),
            key_bull_arguments=["bull1", "bull2"],
            key_bear_arguments=["bear1"],
            unresolved_concerns=["concern1"],
            recommendation="BUY YES",
            total_rounds=2,
            total_tokens=1000,
            total_latency_ms=500,
        )

    def test_result_creation(self, debate_result):
        assert debate_result.market_id == "test_123"
        assert debate_result.final_direction == "yes"
        assert debate_result.final_confidence == Decimal("0.75")

    def test_result_to_dict(self, debate_result):
        d = debate_result.to_dict()
        assert d["market_id"] == "test_123"
        assert d["final_direction"] == "yes"
        assert d["recommendation"] == "BUY YES"
        assert d["total_rounds"] == 2


# =============================================================================
# DebateAgent Tests
# =============================================================================

class TestDebateAgent:
    """Tests for DebateAgent."""

    @pytest.fixture
    def agent(self, mock_llm_provider):
        return DebateAgent(
            persona=DebatePersona.FUNDAMENTALIST,
            provider=mock_llm_provider,
            weight=Decimal("1.0"),
        )

    def test_agent_creation(self, agent):
        assert agent.persona == DebatePersona.FUNDAMENTALIST
        assert agent.weight == Decimal("1.0")

    def test_persona_prompts_exist(self):
        for persona in DebatePersona:
            assert persona in DebateAgent.PERSONA_PROMPTS

    def test_system_prompt_contains_persona(self, agent):
        prompt = agent.system_prompt
        assert "FUNDAMENTALIST" in prompt
        assert "JSON" in prompt

    @pytest.mark.asyncio
    async def test_analyze(self, agent, mock_market):
        position = await agent.analyze(mock_market, "Test context")

        assert isinstance(position, DebatePosition)
        assert position.persona == DebatePersona.FUNDAMENTALIST
        assert position.direction == "yes"
        assert position.fair_value == Decimal("0.55")
        assert position.confidence == Decimal("0.7")

    @pytest.mark.asyncio
    async def test_analyze_with_previous_positions(self, agent, mock_market):
        prev_position = DebatePosition(
            persona=DebatePersona.OPTIMIST,
            model="test",
            direction="yes",
            fair_value=Decimal("0.7"),
            confidence=Decimal("0.8"),
            reasoning="Previous position",
            key_arguments=["prev_arg"],
            counter_arguments=[],
            risk_factors=[],
        )

        position = await agent.analyze(mock_market, "Context", [prev_position])
        assert isinstance(position, DebatePosition)

    @pytest.mark.asyncio
    async def test_rebut(self, agent, mock_market):
        own_position = DebatePosition(
            persona=DebatePersona.FUNDAMENTALIST,
            model="test",
            direction="yes",
            fair_value=Decimal("0.55"),
            confidence=Decimal("0.7"),
            reasoning="My position",
            key_arguments=["arg1"],
            counter_arguments=[],
            risk_factors=[],
        )
        other_position = DebatePosition(
            persona=DebatePersona.PESSIMIST,
            model="test",
            direction="no",
            fair_value=Decimal("0.4"),
            confidence=Decimal("0.6"),
            reasoning="Their position",
            key_arguments=["counter_arg"],
            counter_arguments=[],
            risk_factors=[],
        )

        rebuttal = await agent.rebut(mock_market, own_position, [other_position])
        assert isinstance(rebuttal, str)

    def test_parse_response_valid_json(self, agent):
        content = '{"direction": "no", "fair_value": 0.3, "confidence": 0.6, "reasoning": "Test", "key_arguments": [], "counter_arguments": [], "risk_factors": []}'
        position = agent._parse_response(content)

        assert position.direction == "no"
        assert position.fair_value == Decimal("0.3")
        assert position.confidence == Decimal("0.6")

    def test_parse_response_markdown_json(self, agent):
        content = '```json\n{"direction": "yes", "fair_value": 0.5, "confidence": 0.5, "reasoning": "Test", "key_arguments": [], "counter_arguments": [], "risk_factors": []}\n```'
        position = agent._parse_response(content)

        assert position.direction == "yes"

    def test_parse_response_invalid_json(self, agent):
        content = "This is not JSON at all"
        position = agent._parse_response(content)

        assert position.direction == "abstain"
        assert position.confidence == Decimal("0.3")


# =============================================================================
# DebateOrchestrator Tests
# =============================================================================

class TestDebateOrchestrator:
    """Tests for DebateOrchestrator."""

    @pytest.fixture
    def orchestrator(self, mock_providers):
        return DebateOrchestrator(
            providers=mock_providers,
            max_rounds=2,
            consensus_threshold=Decimal("0.7"),
        )

    def test_orchestrator_creation(self, orchestrator):
        assert orchestrator.max_rounds == 2
        assert orchestrator.consensus_threshold == Decimal("0.7")
        assert len(orchestrator.agents) == 5

    def test_default_weights(self, orchestrator):
        assert DebatePersona.FUNDAMENTALIST in orchestrator.weights
        assert orchestrator.weights[DebatePersona.FUNDAMENTALIST] == Decimal("0.25")
        assert orchestrator.weights[DebatePersona.DEVILS_ADVOCATE] == Decimal("0.10")

    def test_calculate_consensus_all_yes(self, orchestrator):
        positions = [
            DebatePosition(
                persona=DebatePersona.OPTIMIST,
                model="test",
                direction="yes",
                fair_value=Decimal("0.6"),
                confidence=Decimal("0.8"),
                reasoning="",
                key_arguments=[],
                counter_arguments=[],
                risk_factors=[],
            ),
            DebatePosition(
                persona=DebatePersona.FUNDAMENTALIST,
                model="test",
                direction="yes",
                fair_value=Decimal("0.65"),
                confidence=Decimal("0.9"),
                reasoning="",
                key_arguments=[],
                counter_arguments=[],
                risk_factors=[],
            ),
        ]

        direction, confidence, disagreement = orchestrator._calculate_consensus(positions)

        assert direction == "yes"
        assert confidence > Decimal("0.5")
        assert disagreement < Decimal("0.5")

    def test_calculate_consensus_mixed(self, orchestrator):
        positions = [
            DebatePosition(
                persona=DebatePersona.OPTIMIST,
                model="test",
                direction="yes",
                fair_value=Decimal("0.7"),
                confidence=Decimal("0.8"),
                reasoning="",
                key_arguments=[],
                counter_arguments=[],
                risk_factors=[],
            ),
            DebatePosition(
                persona=DebatePersona.PESSIMIST,
                model="test",
                direction="no",
                fair_value=Decimal("0.3"),
                confidence=Decimal("0.8"),
                reasoning="",
                key_arguments=[],
                counter_arguments=[],
                risk_factors=[],
            ),
        ]

        direction, confidence, disagreement = orchestrator._calculate_consensus(positions)

        # With equal weights and confidence, should have high disagreement
        assert disagreement > Decimal("0.3")

    def test_calculate_consensus_empty(self, orchestrator):
        direction, confidence, disagreement = orchestrator._calculate_consensus([])

        assert direction == "abstain"
        assert confidence == Decimal("0")
        assert disagreement == Decimal("1")

    def test_synthesize_result_empty_rounds(self, orchestrator, mock_market):
        result = orchestrator._synthesize_result(
            market=mock_market,
            rounds=[],
            total_tokens=0,
            total_latency_ms=0,
        )

        assert result.final_direction == "abstain"
        assert result.final_confidence == Decimal("0")
        assert "Insufficient" in result.recommendation

    @pytest.mark.asyncio
    async def test_debate_flow(self, orchestrator, mock_market):
        result = await orchestrator.debate(
            market=mock_market,
            context="Test context",
        )

        assert isinstance(result, DebateResult)
        assert result.market_id == mock_market.market_id
        assert result.total_rounds >= 1
        assert len(result.rounds) >= 1

    @pytest.mark.asyncio
    async def test_debate_with_specific_personas(self, orchestrator, mock_market):
        result = await orchestrator.debate(
            market=mock_market,
            context="Test context",
            required_personas=[DebatePersona.OPTIMIST, DebatePersona.PESSIMIST],
        )

        assert isinstance(result, DebateResult)

    @pytest.mark.asyncio
    async def test_debate_no_agents_raises(self, mock_market):
        orchestrator = DebateOrchestrator(providers={}, max_rounds=1)

        with pytest.raises(ValueError, match="No debate agents available"):
            await orchestrator.debate(mock_market)


# =============================================================================
# ConfidenceCalibrator Tests
# =============================================================================

class TestConfidenceCalibrator:
    """Tests for ConfidenceCalibrator."""

    @pytest.fixture
    def calibrator(self):
        return ConfidenceCalibrator()

    def test_calibrator_init(self, calibrator):
        assert calibrator._predictions == []
        assert calibrator._calibration_scores == {}

    def test_record_prediction(self, calibrator):
        calibrator.record_prediction(
            market_id="test_123",
            persona=DebatePersona.OPTIMIST,
            predicted_prob=Decimal("0.7"),
            confidence=Decimal("0.8"),
            timestamp=datetime.utcnow(),
        )

        assert len(calibrator._predictions) == 1
        assert calibrator._predictions[0]["market_id"] == "test_123"
        assert calibrator._predictions[0]["outcome"] is None

    def test_record_outcome(self, calibrator):
        calibrator.record_prediction(
            market_id="test_123",
            persona=DebatePersona.OPTIMIST,
            predicted_prob=Decimal("0.7"),
            confidence=Decimal("0.8"),
            timestamp=datetime.utcnow(),
        )

        calibrator.record_outcome("test_123", True)

        assert calibrator._predictions[0]["outcome"] is True
        assert DebatePersona.OPTIMIST in calibrator._calibration_scores

    def test_calibration_after_multiple_outcomes(self, calibrator):
        # Record multiple predictions
        for i in range(5):
            calibrator.record_prediction(
                market_id=f"market_{i}",
                persona=DebatePersona.FUNDAMENTALIST,
                predicted_prob=Decimal("0.7"),
                confidence=Decimal("0.8"),
                timestamp=datetime.utcnow(),
            )
            # Alternate outcomes
            calibrator.record_outcome(f"market_{i}", i % 2 == 0)

        scores = calibrator._calibration_scores[DebatePersona.FUNDAMENTALIST]
        assert scores["total_predictions"] == 5
        assert scores["brier_score"] > Decimal("0")

    def test_get_calibration_adjustment_no_data(self, calibrator):
        adjustment = calibrator.get_calibration_adjustment(DebatePersona.OPTIMIST)
        assert adjustment == Decimal("1.0")

    def test_get_calibration_adjustment_insufficient_data(self, calibrator):
        # Add less than 10 predictions
        for i in range(5):
            calibrator.record_prediction(
                market_id=f"market_{i}",
                persona=DebatePersona.OPTIMIST,
                predicted_prob=Decimal("0.9"),
                confidence=Decimal("0.9"),
                timestamp=datetime.utcnow(),
            )
            calibrator.record_outcome(f"market_{i}", False)

        adjustment = calibrator.get_calibration_adjustment(DebatePersona.OPTIMIST)
        assert adjustment == Decimal("1.0")  # Not enough data

    def test_overconfidence_adjustment(self, calibrator):
        # Simulate overconfident predictions (high confidence, wrong outcomes)
        for i in range(15):
            calibrator.record_prediction(
                market_id=f"market_{i}",
                persona=DebatePersona.OPTIMIST,
                predicted_prob=Decimal("0.9"),
                confidence=Decimal("0.9"),
                timestamp=datetime.utcnow(),
            )
            # Most outcomes are wrong
            calibrator.record_outcome(f"market_{i}", i < 3)

        adjustment = calibrator.get_calibration_adjustment(DebatePersona.OPTIMIST)
        assert adjustment < Decimal("1.0")  # Should reduce confidence

    def test_get_calibration_report(self, calibrator):
        calibrator.record_prediction(
            market_id="test_1",
            persona=DebatePersona.FUNDAMENTALIST,
            predicted_prob=Decimal("0.6"),
            confidence=Decimal("0.7"),
            timestamp=datetime.utcnow(),
        )
        calibrator.record_outcome("test_1", True)

        report = calibrator.get_calibration_report()

        assert report["total_predictions"] == 1
        assert report["pending_predictions"] == 0
        assert "fundamentalist" in report["by_persona"]


class TestGetConfidenceCalibrator:
    """Tests for get_confidence_calibrator singleton."""

    def test_returns_same_instance(self):
        cal1 = get_confidence_calibrator()
        cal2 = get_confidence_calibrator()
        assert cal1 is cal2


# =============================================================================
# Integration Tests
# =============================================================================

class TestDebateIntegration:
    """Integration tests for the complete debate system."""

    @pytest.mark.asyncio
    async def test_full_debate_flow(self, mock_market, mock_providers):
        """Test a complete debate from start to finish."""
        orchestrator = DebateOrchestrator(
            providers=mock_providers,
            max_rounds=2,
            consensus_threshold=Decimal("0.6"),
        )

        result = await orchestrator.debate(
            market=mock_market,
            context="Bitcoin has been showing strong momentum. Recent ETF approvals have increased institutional interest.",
        )

        # Verify result structure
        assert result.market_id == mock_market.market_id
        assert result.final_direction in ["yes", "no", "abstain"]
        assert Decimal("0") <= result.final_confidence <= Decimal("1")
        assert Decimal("0") <= result.final_fair_value <= Decimal("1")
        assert len(result.rounds) >= 1
        assert result.total_tokens > 0

    @pytest.mark.asyncio
    async def test_debate_with_calibration(self, mock_market, mock_providers):
        """Test debate with confidence calibration tracking."""
        orchestrator = DebateOrchestrator(
            providers=mock_providers,
            max_rounds=1,
        )
        calibrator = ConfidenceCalibrator()

        # Run debate
        result = await orchestrator.debate(mock_market, "Test")

        # Record predictions from each position
        for round in result.rounds:
            for position in round.positions:
                calibrator.record_prediction(
                    market_id=result.market_id,
                    persona=position.persona,
                    predicted_prob=position.fair_value,
                    confidence=position.confidence,
                    timestamp=position.timestamp,
                )

        # Simulate outcome
        calibrator.record_outcome(result.market_id, True)

        # Check report
        report = calibrator.get_calibration_report()
        assert report["total_predictions"] > 0

    @pytest.mark.asyncio
    async def test_devils_advocate_opposes_consensus(self, mock_market):
        """Test that devil's advocate argues against consensus."""
        # Create providers with different responses
        bullish_provider = MagicMock()
        bullish_provider.default_model = "bullish-model"

        async def bullish_complete(*args, **kwargs):
            return LLMResponse(
                content='{"direction": "yes", "fair_value": 0.7, "confidence": 0.8, "reasoning": "Bullish!", "key_arguments": ["bull"], "counter_arguments": [], "risk_factors": []}',
                model="bullish",
                provider="test",
                tokens_input=100,
                tokens_output=50,
                latency_ms=100,
                cost_estimate=Decimal("0.001"),
                finish_reason="stop",
            )

        bullish_provider.complete = bullish_complete

        # Devil's advocate should oppose (mock returns YES but in real usage would return NO)
        da_provider = MagicMock()
        da_provider.default_model = "da-model"

        async def da_complete(*args, **kwargs):
            return LLMResponse(
                content='{"direction": "no", "fair_value": 0.35, "confidence": 0.6, "reasoning": "Contrarian view", "key_arguments": ["risk"], "counter_arguments": [], "risk_factors": []}',
                model="da",
                provider="test",
                tokens_input=100,
                tokens_output=50,
                latency_ms=100,
                cost_estimate=Decimal("0.001"),
                finish_reason="stop",
            )

        da_provider.complete = da_complete

        orchestrator = DebateOrchestrator(
            providers={
                DebatePersona.OPTIMIST: bullish_provider,
                DebatePersona.DEVILS_ADVOCATE: da_provider,
            },
            max_rounds=1,
        )

        result = await orchestrator.debate(mock_market, "Test")

        # Check that we have mixed positions
        directions = set()
        for round in result.rounds:
            for pos in round.positions:
                directions.add(pos.direction)

        # Should have both yes and no
        assert "yes" in directions
        assert "no" in directions
