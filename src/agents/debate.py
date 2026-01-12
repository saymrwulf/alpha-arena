"""Multi-Agent Debate System for improved trading decisions."""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Optional

from ..core.types import MarketState
from ..llm.base import LLMProvider, Message, Role
from .base import AgentResponse, AgentRole

logger = logging.getLogger(__name__)


class DebatePersona(str, Enum):
    """Debate agent personas with different analytical biases."""
    OPTIMIST = "optimist"  # Bullish bias, looks for opportunity
    PESSIMIST = "pessimist"  # Bearish bias, focuses on risks
    FUNDAMENTALIST = "fundamentalist"  # Focus on underlying value and news
    TECHNICIAN = "technician"  # Focus on price action and patterns
    DEVILS_ADVOCATE = "devils_advocate"  # Argues against consensus
    NEUTRAL = "neutral"  # Balanced analysis


@dataclass
class DebatePosition:
    """A position taken by a debate agent."""
    persona: DebatePersona
    model: str
    direction: str  # "yes", "no", or "abstain"
    fair_value: Decimal
    confidence: Decimal
    reasoning: str
    key_arguments: list[str]
    counter_arguments: list[str]  # Arguments against their own position
    risk_factors: list[str]
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "persona": self.persona.value,
            "model": self.model,
            "direction": self.direction,
            "fair_value": float(self.fair_value),
            "confidence": float(self.confidence),
            "reasoning": self.reasoning,
            "key_arguments": self.key_arguments,
            "counter_arguments": self.counter_arguments,
            "risk_factors": self.risk_factors,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class DebateRound:
    """A single round of debate."""
    round_number: int
    positions: list[DebatePosition]
    rebuttals: dict[DebatePersona, str]  # Responses to other positions
    consensus_direction: Optional[str]
    consensus_confidence: Decimal
    disagreement_level: Decimal  # 0 = full agreement, 1 = complete disagreement

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "round_number": self.round_number,
            "positions": [p.to_dict() for p in self.positions],
            "rebuttals": {k.value: v for k, v in self.rebuttals.items()},
            "consensus_direction": self.consensus_direction,
            "consensus_confidence": float(self.consensus_confidence),
            "disagreement_level": float(self.disagreement_level),
        }


@dataclass
class DebateResult:
    """Final result of a multi-agent debate."""
    market_id: str
    market_question: str
    rounds: list[DebateRound]
    final_direction: str  # "yes", "no", or "abstain"
    final_confidence: Decimal
    final_fair_value: Decimal
    consensus_strength: Decimal  # How strong is the agreement
    key_bull_arguments: list[str]
    key_bear_arguments: list[str]
    unresolved_concerns: list[str]
    recommendation: str
    total_rounds: int
    total_tokens: int
    total_latency_ms: int
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "market_id": self.market_id,
            "market_question": self.market_question,
            "rounds": [r.to_dict() for r in self.rounds],
            "final_direction": self.final_direction,
            "final_confidence": float(self.final_confidence),
            "final_fair_value": float(self.final_fair_value),
            "consensus_strength": float(self.consensus_strength),
            "key_bull_arguments": self.key_bull_arguments,
            "key_bear_arguments": self.key_bear_arguments,
            "unresolved_concerns": self.unresolved_concerns,
            "recommendation": self.recommendation,
            "total_rounds": self.total_rounds,
            "total_tokens": self.total_tokens,
            "total_latency_ms": self.total_latency_ms,
            "timestamp": self.timestamp.isoformat(),
        }


class DebateAgent:
    """An agent with a specific persona for debates."""

    PERSONA_PROMPTS = {
        DebatePersona.OPTIMIST: """You are an OPTIMIST analyst in a trading debate.
Your role is to identify opportunities and potential upside.
You tend to see the glass as half full, but must still be intellectually honest.
Focus on: catalysts, growth potential, positive news, momentum, market inefficiencies.
You should NOT be blindly bullish - acknowledge real risks but emphasize opportunities.""",

        DebatePersona.PESSIMIST: """You are a PESSIMIST analyst in a trading debate.
Your role is to identify risks and potential downside.
You tend to see the glass as half empty, but must still be intellectually honest.
Focus on: risk factors, overvaluation, negative catalysts, bear cases, what could go wrong.
You should NOT be blindly bearish - acknowledge opportunities but emphasize risks.""",

        DebatePersona.FUNDAMENTALIST: """You are a FUNDAMENTALIST analyst in a trading debate.
Your role is to analyze underlying value and real-world factors.
Focus on: news events, probability calculations, base rates, historical precedents, logic.
Ignore price action - focus only on what the TRUE probability should be based on facts.
You believe markets can be significantly mispriced when emotions run high.""",

        DebatePersona.TECHNICIAN: """You are a TECHNICAL analyst in a trading debate.
Your role is to analyze price action, momentum, and market structure.
Focus on: order book imbalances, volume patterns, price momentum, liquidity, spread.
Ignore news and fundamentals - focus only on what the MARKET is telling you through price.
You believe price action reveals true sentiment and information.""",

        DebatePersona.DEVILS_ADVOCATE: """You are the DEVIL'S ADVOCATE in a trading debate.
Your role is to argue AGAINST whatever the emerging consensus is.
If others are bullish, find reasons to be bearish. If others are bearish, find reasons to be bullish.
Challenge assumptions, poke holes in arguments, and stress-test the thesis.
Your goal is to ensure the group doesn't fall into groupthink.
Be intellectually rigorous and find the strongest counterarguments.""",

        DebatePersona.NEUTRAL: """You are a NEUTRAL analyst in a trading debate.
Your role is to provide balanced, unbiased analysis.
Weigh both bull and bear cases equally and reach an objective conclusion.
Focus on: probability estimation, evidence quality, argument strength.
Your goal is to find the truth, not to take a side.""",
    }

    def __init__(
        self,
        persona: DebatePersona,
        provider: LLMProvider,
        weight: Decimal = Decimal("1.0"),
    ):
        self.persona = persona
        self.provider = provider
        self.weight = weight  # Weight in consensus voting
        self._tokens_used = 0
        self._latency_ms = 0

    @property
    def system_prompt(self) -> str:
        """Get persona-specific system prompt."""
        base_prompt = self.PERSONA_PROMPTS.get(
            self.persona,
            self.PERSONA_PROMPTS[DebatePersona.NEUTRAL]
        )
        return f"""{base_prompt}

OUTPUT FORMAT (JSON):
{{
    "direction": "yes" | "no" | "abstain",
    "fair_value": 0.XX,
    "confidence": 0.XX,
    "reasoning": "Your detailed analysis...",
    "key_arguments": ["argument1", "argument2", "argument3"],
    "counter_arguments": ["What could prove me wrong..."],
    "risk_factors": ["risk1", "risk2"]
}}

Be specific, data-driven, and intellectually honest.
If you're uncertain, say so and explain why."""

    async def analyze(
        self,
        market: MarketState,
        context: str,
        other_positions: list[DebatePosition] | None = None,
    ) -> DebatePosition:
        """Analyze a market and take a position."""
        prompt = self._build_prompt(market, context, other_positions)

        messages = [Message(role=Role.USER, content=prompt)]

        start_time = datetime.utcnow()
        response = await self.provider.complete(
            messages=messages,
            system=self.system_prompt,
            json_mode=True,
            max_tokens=1500,
        )
        self._latency_ms += response.latency_ms
        self._tokens_used += response.total_tokens

        # Parse response
        position = self._parse_response(response.content)
        return position

    async def rebut(
        self,
        market: MarketState,
        own_position: DebatePosition,
        other_positions: list[DebatePosition],
    ) -> str:
        """Respond to other agents' positions."""
        positions_text = self._format_positions(other_positions)

        prompt = f"""You previously analyzed this market:

{market.question}

Your position: {own_position.direction.upper()} at {own_position.fair_value:.1%} (confidence: {own_position.confidence:.0%})
Your reasoning: {own_position.reasoning}

Other analysts have taken these positions:
{positions_text}

Respond to their arguments:
1. Which counterarguments are valid and should be considered?
2. Which arguments strengthen or weaken your position?
3. Should you adjust your view? If so, how?

Provide a brief, focused rebuttal (2-3 paragraphs)."""

        messages = [Message(role=Role.USER, content=prompt)]
        response = await self.provider.complete(
            messages=messages,
            system=self.system_prompt,
            max_tokens=800,
        )
        self._latency_ms += response.latency_ms
        self._tokens_used += response.total_tokens

        return response.content

    def _build_prompt(
        self,
        market: MarketState,
        context: str,
        other_positions: list[DebatePosition] | None,
    ) -> str:
        """Build analysis prompt."""
        base = f"""Analyze this prediction market:

MARKET: {market.question}
Category: {market.category}
End Date: {market.end_date}

CURRENT PRICING:
- YES: ${market.yes_price:.4f} (implied: {market.implied_probability:.1%})
- NO: ${market.no_price:.4f}
- 24h Volume: ${market.volume_24h:,.0f}
- Liquidity: ${market.liquidity:,.0f}

ORDER BOOK:
- YES Spread: {market.yes_book.spread_bps:.0f}bps
- Bid Depth: ${sum(level.size for level in market.yes_book.bids):,.0f}
- Ask Depth: ${sum(level.size for level in market.yes_book.asks):,.0f}

TECHNICAL INDICATORS:
{json.dumps(market.indicators, indent=2, default=str) if market.indicators else 'N/A'}

SENTIMENT:
{json.dumps(market.sentiment, indent=2, default=str) if market.sentiment else 'N/A'}

CONTEXT:
{context}
"""

        if other_positions:
            base += f"\n\nOTHER ANALYST POSITIONS:\n{self._format_positions(other_positions)}"
            base += "\n\nConsider their arguments in your analysis."

        base += "\n\nProvide your analysis as JSON."
        return base

    def _format_positions(self, positions: list[DebatePosition]) -> str:
        """Format other positions for display."""
        lines = []
        for p in positions:
            lines.append(f"""
[{p.persona.value.upper()}] - {p.direction.upper()} @ {p.fair_value:.1%} (conf: {p.confidence:.0%})
{p.reasoning[:200]}...
Key arguments: {', '.join(p.key_arguments[:2])}
""")
        return "\n".join(lines)

    def _parse_response(self, content: str) -> DebatePosition:
        """Parse LLM response into DebatePosition."""
        try:
            # Handle markdown code blocks
            if "```" in content:
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            data = json.loads(content)
        except json.JSONDecodeError:
            # Fallback parsing
            data = {
                "direction": "abstain",
                "fair_value": 0.5,
                "confidence": 0.3,
                "reasoning": content[:500],
                "key_arguments": [],
                "counter_arguments": [],
                "risk_factors": [],
            }

        return DebatePosition(
            persona=self.persona,
            model=self.provider.default_model,
            direction=data.get("direction", "abstain"),
            fair_value=Decimal(str(data.get("fair_value", 0.5))),
            confidence=Decimal(str(data.get("confidence", 0.5))),
            reasoning=data.get("reasoning", ""),
            key_arguments=data.get("key_arguments", []),
            counter_arguments=data.get("counter_arguments", []),
            risk_factors=data.get("risk_factors", []),
        )


class DebateOrchestrator:
    """
    Orchestrates multi-agent debates for trading decisions.

    Runs structured debates between agents with different perspectives,
    then synthesizes a consensus recommendation.
    """

    # Default weights for each persona in consensus voting
    DEFAULT_WEIGHTS = {
        DebatePersona.OPTIMIST: Decimal("0.15"),
        DebatePersona.PESSIMIST: Decimal("0.15"),
        DebatePersona.FUNDAMENTALIST: Decimal("0.25"),
        DebatePersona.TECHNICIAN: Decimal("0.20"),
        DebatePersona.DEVILS_ADVOCATE: Decimal("0.10"),
        DebatePersona.NEUTRAL: Decimal("0.15"),
    }

    def __init__(
        self,
        providers: dict[DebatePersona, LLMProvider],
        max_rounds: int = 3,
        consensus_threshold: Decimal = Decimal("0.7"),
        weights: dict[DebatePersona, Decimal] | None = None,
    ):
        """
        Initialize the debate orchestrator.

        Args:
            providers: Map of persona to LLM provider
            max_rounds: Maximum debate rounds before forcing consensus
            consensus_threshold: Agreement level needed to end early
            weights: Custom weights for each persona in voting
        """
        self.max_rounds = max_rounds
        self.consensus_threshold = consensus_threshold
        self.weights = weights or self.DEFAULT_WEIGHTS

        # Create debate agents
        self.agents: dict[DebatePersona, DebateAgent] = {}
        for persona, provider in providers.items():
            weight = self.weights.get(persona, Decimal("1.0"))
            self.agents[persona] = DebateAgent(persona, provider, weight)

    async def debate(
        self,
        market: MarketState,
        context: str = "",
        required_personas: list[DebatePersona] | None = None,
    ) -> DebateResult:
        """
        Run a full multi-agent debate on a market.

        Args:
            market: The market to analyze
            context: Additional context (memory, news, etc.)
            required_personas: Specific personas to include (default: all available)

        Returns:
            DebateResult with consensus recommendation
        """
        start_time = datetime.utcnow()

        # Select agents to participate
        if required_personas:
            active_agents = {
                p: a for p, a in self.agents.items()
                if p in required_personas
            }
        else:
            active_agents = self.agents

        if not active_agents:
            raise ValueError("No debate agents available")

        rounds: list[DebateRound] = []
        total_tokens = 0
        total_latency = 0

        # Run debate rounds
        for round_num in range(1, self.max_rounds + 1):
            logger.info(f"Starting debate round {round_num} for {market.market_id}")

            # Get previous positions for context
            prev_positions = rounds[-1].positions if rounds else None

            # Phase 1: Each agent analyzes and takes a position
            positions = await self._run_analysis_phase(
                active_agents, market, context, prev_positions
            )

            # Track tokens
            for agent in active_agents.values():
                total_tokens += agent._tokens_used
                total_latency += agent._latency_ms

            # Calculate consensus
            consensus_dir, consensus_conf, disagreement = self._calculate_consensus(positions)

            # Phase 2: Agents rebut each other (if disagreement exists)
            rebuttals: dict[DebatePersona, str] = {}
            if disagreement > Decimal("0.3") and round_num < self.max_rounds:
                rebuttals = await self._run_rebuttal_phase(
                    active_agents, market, positions
                )
                for agent in active_agents.values():
                    total_tokens += agent._tokens_used
                    total_latency += agent._latency_ms

            round_result = DebateRound(
                round_number=round_num,
                positions=positions,
                rebuttals=rebuttals,
                consensus_direction=consensus_dir,
                consensus_confidence=consensus_conf,
                disagreement_level=disagreement,
            )
            rounds.append(round_result)

            # Check if consensus reached
            if consensus_conf >= self.consensus_threshold:
                logger.info(f"Consensus reached in round {round_num}")
                break

        # Synthesize final result
        final_result = self._synthesize_result(
            market=market,
            rounds=rounds,
            total_tokens=total_tokens,
            total_latency_ms=total_latency,
        )

        return final_result

    async def _run_analysis_phase(
        self,
        agents: dict[DebatePersona, DebateAgent],
        market: MarketState,
        context: str,
        prev_positions: list[DebatePosition] | None,
    ) -> list[DebatePosition]:
        """Run parallel analysis from all agents."""
        tasks = []
        for persona, agent in agents.items():
            # Reset counters
            agent._tokens_used = 0
            agent._latency_ms = 0

            # Pass previous positions for context (except own)
            other_positions = None
            if prev_positions:
                other_positions = [p for p in prev_positions if p.persona != persona]

            tasks.append(agent.analyze(market, context, other_positions))

        positions = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions
        valid_positions = []
        for pos in positions:
            if isinstance(pos, DebatePosition):
                valid_positions.append(pos)
            elif isinstance(pos, Exception):
                logger.error(f"Debate agent failed: {pos}")

        return valid_positions

    async def _run_rebuttal_phase(
        self,
        agents: dict[DebatePersona, DebateAgent],
        market: MarketState,
        positions: list[DebatePosition],
    ) -> dict[DebatePersona, str]:
        """Run parallel rebuttals from all agents."""
        tasks = {}

        for persona, agent in agents.items():
            # Find this agent's position
            own_position = next(
                (p for p in positions if p.persona == persona),
                None
            )
            if not own_position:
                continue

            # Other positions
            other_positions = [p for p in positions if p.persona != persona]

            tasks[persona] = agent.rebut(market, own_position, other_positions)

        # Run in parallel
        results = {}
        for persona, task in tasks.items():
            try:
                results[persona] = await task
            except Exception as e:
                logger.error(f"Rebuttal failed for {persona}: {e}")
                results[persona] = ""

        return results

    def _calculate_consensus(
        self,
        positions: list[DebatePosition],
    ) -> tuple[str, Decimal, Decimal]:
        """
        Calculate weighted consensus from positions.

        Returns (direction, confidence, disagreement_level).
        """
        if not positions:
            return "abstain", Decimal("0"), Decimal("1")

        # Weighted voting
        yes_weight = Decimal("0")
        no_weight = Decimal("0")
        abstain_weight = Decimal("0")
        total_weight = Decimal("0")

        fair_values = []
        weights_list = []

        for pos in positions:
            weight = self.weights.get(pos.persona, Decimal("1.0"))
            total_weight += weight

            if pos.direction == "yes":
                yes_weight += weight * pos.confidence
            elif pos.direction == "no":
                no_weight += weight * pos.confidence
            else:
                abstain_weight += weight

            fair_values.append(pos.fair_value)
            weights_list.append(weight)

        # Determine consensus direction
        if yes_weight > no_weight and yes_weight > abstain_weight:
            direction = "yes"
            confidence = yes_weight / total_weight
        elif no_weight > yes_weight and no_weight > abstain_weight:
            direction = "no"
            confidence = no_weight / total_weight
        else:
            direction = "abstain"
            confidence = Decimal("0.3")

        # Calculate disagreement
        # Disagreement = 1 - (winner_margin / total)
        max_weight = max(yes_weight, no_weight, abstain_weight)
        second_weight = sorted([yes_weight, no_weight, abstain_weight])[-2]
        if total_weight > 0:
            margin = (max_weight - second_weight) / total_weight
            disagreement = Decimal("1") - margin
        else:
            disagreement = Decimal("1")

        return direction, confidence, disagreement

    def _synthesize_result(
        self,
        market: MarketState,
        rounds: list[DebateRound],
        total_tokens: int,
        total_latency_ms: int,
    ) -> DebateResult:
        """Synthesize final debate result."""
        if not rounds:
            return DebateResult(
                market_id=market.market_id,
                market_question=market.question,
                rounds=[],
                final_direction="abstain",
                final_confidence=Decimal("0"),
                final_fair_value=Decimal("0.5"),
                consensus_strength=Decimal("0"),
                key_bull_arguments=[],
                key_bear_arguments=[],
                unresolved_concerns=[],
                recommendation="Insufficient debate data",
                total_rounds=0,
                total_tokens=total_tokens,
                total_latency_ms=total_latency_ms,
            )

        # Use final round for consensus
        final_round = rounds[-1]
        positions = final_round.positions

        # Calculate weighted fair value
        total_weight = Decimal("0")
        weighted_fv = Decimal("0")
        for pos in positions:
            weight = self.weights.get(pos.persona, Decimal("1.0"))
            weighted_fv += pos.fair_value * weight * pos.confidence
            total_weight += weight * pos.confidence

        final_fair_value = weighted_fv / total_weight if total_weight > 0 else Decimal("0.5")

        # Collect arguments
        bull_args = []
        bear_args = []
        concerns = []

        for pos in positions:
            if pos.direction == "yes":
                bull_args.extend(pos.key_arguments[:2])
            elif pos.direction == "no":
                bear_args.extend(pos.key_arguments[:2])
            concerns.extend(pos.risk_factors[:1])

        # Deduplicate
        bull_args = list(dict.fromkeys(bull_args))[:5]
        bear_args = list(dict.fromkeys(bear_args))[:5]
        concerns = list(dict.fromkeys(concerns))[:5]

        # Generate recommendation
        consensus_strength = Decimal("1") - final_round.disagreement_level
        if final_round.consensus_confidence >= Decimal("0.7") and consensus_strength >= Decimal("0.6"):
            if final_round.consensus_direction == "yes":
                recommendation = f"BUY YES at current price. Strong consensus ({final_round.consensus_confidence:.0%}) supports YES outcome."
            elif final_round.consensus_direction == "no":
                recommendation = f"BUY NO at current price. Strong consensus ({final_round.consensus_confidence:.0%}) supports NO outcome."
            else:
                recommendation = "ABSTAIN. No clear edge identified."
        elif final_round.consensus_confidence >= Decimal("0.5"):
            recommendation = f"CONSIDER {final_round.consensus_direction.upper()} with reduced size. Moderate consensus ({final_round.consensus_confidence:.0%}) with some disagreement."
        else:
            recommendation = "ABSTAIN. Insufficient consensus among analysts. Wait for more clarity."

        return DebateResult(
            market_id=market.market_id,
            market_question=market.question,
            rounds=rounds,
            final_direction=final_round.consensus_direction or "abstain",
            final_confidence=final_round.consensus_confidence,
            final_fair_value=final_fair_value,
            consensus_strength=consensus_strength,
            key_bull_arguments=bull_args,
            key_bear_arguments=bear_args,
            unresolved_concerns=concerns,
            recommendation=recommendation,
            total_rounds=len(rounds),
            total_tokens=total_tokens,
            total_latency_ms=total_latency_ms,
        )


class ConfidenceCalibrator:
    """
    Tracks and calibrates agent confidence over time.

    Compares predicted probabilities to actual outcomes
    to improve confidence accuracy.
    """

    def __init__(self):
        self._predictions: list[dict] = []  # Historical predictions
        self._calibration_scores: dict[DebatePersona, dict] = {}  # Per-persona calibration

    def record_prediction(
        self,
        market_id: str,
        persona: DebatePersona,
        predicted_prob: Decimal,
        confidence: Decimal,
        timestamp: datetime,
    ) -> None:
        """Record a prediction for later calibration."""
        self._predictions.append({
            "market_id": market_id,
            "persona": persona,
            "predicted_prob": predicted_prob,
            "confidence": confidence,
            "timestamp": timestamp,
            "outcome": None,  # Filled in later
        })

    def record_outcome(
        self,
        market_id: str,
        outcome: bool,  # True = YES resolved, False = NO resolved
    ) -> None:
        """Record the actual outcome for calibration."""
        for pred in self._predictions:
            if pred["market_id"] == market_id and pred["outcome"] is None:
                pred["outcome"] = outcome
                self._update_calibration(pred)

    def _update_calibration(self, prediction: dict) -> None:
        """Update calibration scores based on resolved prediction."""
        persona = prediction["persona"]
        if persona not in self._calibration_scores:
            self._calibration_scores[persona] = {
                "total_predictions": 0,
                "calibration_error": Decimal("0"),
                "brier_score": Decimal("0"),
                "overconfidence_count": 0,
                "underconfidence_count": 0,
            }

        scores = self._calibration_scores[persona]
        scores["total_predictions"] += 1

        predicted = prediction["predicted_prob"]
        actual = Decimal("1") if prediction["outcome"] else Decimal("0")

        # Brier score component
        brier = (predicted - actual) ** 2
        n = scores["total_predictions"]
        scores["brier_score"] = (
            scores["brier_score"] * Decimal(str((n - 1) / n)) +
            brier / Decimal(str(n))
        )

        # Calibration error
        error = abs(predicted - actual)
        scores["calibration_error"] = (
            scores["calibration_error"] * Decimal(str((n - 1) / n)) +
            error / Decimal(str(n))
        )

        # Over/under confidence tracking
        confidence = prediction["confidence"]
        if confidence > Decimal("0.7") and error > Decimal("0.3"):
            scores["overconfidence_count"] += 1
        elif confidence < Decimal("0.4") and error < Decimal("0.2"):
            scores["underconfidence_count"] += 1

    def get_calibration_adjustment(
        self,
        persona: DebatePersona,
    ) -> Decimal:
        """
        Get confidence adjustment factor for a persona.

        Returns a multiplier for confidence (e.g., 0.9 = reduce confidence 10%).
        """
        if persona not in self._calibration_scores:
            return Decimal("1.0")  # No data, no adjustment

        scores = self._calibration_scores[persona]
        if scores["total_predictions"] < 10:
            return Decimal("1.0")  # Not enough data

        # Adjust based on overconfidence/underconfidence
        over = scores["overconfidence_count"]
        under = scores["underconfidence_count"]
        total = scores["total_predictions"]

        if over > under and over / total > 0.2:
            # Consistently overconfident - reduce confidence
            return Decimal("0.85")
        elif under > over and under / total > 0.2:
            # Consistently underconfident - increase confidence
            return Decimal("1.15")

        return Decimal("1.0")

    def get_calibration_report(self) -> dict:
        """Get full calibration report."""
        return {
            "total_predictions": len([p for p in self._predictions if p["outcome"] is not None]),
            "pending_predictions": len([p for p in self._predictions if p["outcome"] is None]),
            "by_persona": {
                persona.value: {
                    "total": scores["total_predictions"],
                    "brier_score": float(scores["brier_score"]),
                    "calibration_error": float(scores["calibration_error"]),
                    "overconfident": scores["overconfidence_count"],
                    "underconfident": scores["underconfidence_count"],
                    "adjustment": float(self.get_calibration_adjustment(persona)),
                }
                for persona, scores in self._calibration_scores.items()
            },
        }


# Global calibrator instance
_calibrator: Optional[ConfidenceCalibrator] = None


def get_confidence_calibrator() -> ConfidenceCalibrator:
    """Get or create the global confidence calibrator."""
    global _calibrator
    if _calibrator is None:
        _calibrator = ConfidenceCalibrator()
    return _calibrator
