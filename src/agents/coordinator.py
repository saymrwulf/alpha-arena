"""Agent Coordinator - Orchestrates multi-agent trading decisions."""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any

from ..core.types import MarketState, Signal, SignalType, Edge, Confidence
from ..core.config import AgentConfig
from ..llm.registry import ProviderRegistry
from ..memory.manager import MemoryManager
from .base import AgentRole, AgentResponse
from .research import ResearchAgent
from .risk import RiskAgent
from .execution import ExecutionAgent
from .reflection import ReflectionAgent


@dataclass
class CoordinatorDecision:
    """Final decision from the agent coordinator."""
    signals: list[Signal]
    consensus_confidence: Decimal
    debate_rounds: int
    agent_responses: dict[AgentRole, AgentResponse]
    execution_plan: dict[str, Any]
    total_latency_ms: int
    total_tokens: int
    timestamp: datetime = field(default_factory=datetime.utcnow)


class AgentCoordinator:
    """
    Coordinates multiple specialized agents for trading decisions.

    Flow:
    1. Research Agent analyzes markets → identifies opportunities
    2. Risk Agent validates → sizes positions
    3. (Optional) Debate round for high-stakes decisions
    4. Execution Agent plans → optimizes entry
    5. Final consensus → generate signals
    6. (Post-trade) Reflection Agent learns → updates memory

    This is the "brain" of the trading system.
    """

    def __init__(
        self,
        provider_registry: ProviderRegistry,
        memory: MemoryManager,
        config: AgentConfig,
    ):
        self.registry = provider_registry
        self.memory = memory
        self.config = config
        self._agents: dict[AgentRole, Any] = {}

    async def initialize(self) -> None:
        """Initialize all agents with their assigned models."""
        # Research Agent
        provider_info = self.registry.get_for_model(self.config.research_agent_model)
        if provider_info:
            provider, _ = provider_info
            self._agents[AgentRole.RESEARCH] = ResearchAgent(provider, self.memory)

        # Risk Agent
        provider_info = self.registry.get_for_model(self.config.risk_agent_model)
        if provider_info:
            provider, _ = provider_info
            self._agents[AgentRole.RISK] = RiskAgent(provider, self.memory)

        # Execution Agent
        provider_info = self.registry.get_for_model(self.config.execution_agent_model)
        if provider_info:
            provider, _ = provider_info
            self._agents[AgentRole.EXECUTION] = ExecutionAgent(provider, self.memory)

        # Reflection Agent
        provider_info = self.registry.get_for_model(self.config.reflection_agent_model)
        if provider_info:
            provider, _ = provider_info
            self._agents[AgentRole.REFLECTION] = ReflectionAgent(provider, self.memory)

    async def decide(
        self,
        markets: list[MarketState],
        portfolio_value: Decimal,
        positions: list,
    ) -> CoordinatorDecision:
        """
        Run the full multi-agent decision process.

        Returns final trading signals with execution plan.
        """
        start_time = datetime.utcnow()
        responses: dict[AgentRole, AgentResponse] = {}
        total_tokens = 0

        # Get memory context (parallel)
        context, performance_context = await asyncio.gather(
            self.memory.get_context_for_decision(""),
            self.memory.get_performance_context(),
        )
        full_context = f"{context}\n\n{performance_context}"

        # Phase 1: Research Agent analyzes markets
        if AgentRole.RESEARCH in self._agents:
            research_response = await self._agents[AgentRole.RESEARCH].analyze(
                markets=markets,
                context=full_context,
            )
            responses[AgentRole.RESEARCH] = research_response
            total_tokens += research_response.tokens_used

            # If no opportunities found, return early
            if not research_response.content.get("opportunities"):
                return CoordinatorDecision(
                    signals=[],
                    consensus_confidence=Decimal("0"),
                    debate_rounds=0,
                    agent_responses=responses,
                    execution_plan={},
                    total_latency_ms=int((datetime.utcnow() - start_time).total_seconds() * 1000),
                    total_tokens=total_tokens,
                )

        # Phase 2: Risk Agent validates and sizes
        if AgentRole.RISK in self._agents:
            portfolio_context = f"""
{full_context}

PORTFOLIO:
- Value: ${portfolio_value:,.2f}
- Open Positions: {len(positions)}
- Current Exposure: ${sum(getattr(p, 'market_value', 0) for p in positions):,.2f}
"""
            risk_response = await self._agents[AgentRole.RISK].analyze(
                markets=markets,
                context=portfolio_context,
                other_responses=[responses.get(AgentRole.RESEARCH)],
            )
            responses[AgentRole.RISK] = risk_response
            total_tokens += risk_response.tokens_used

        # Phase 3: Multi-agent debate (if enabled and stakes are high)
        debate_rounds = 0
        if self.config.enable_multi_agent_debate:
            opportunities = responses.get(AgentRole.RESEARCH, AgentResponse(
                agent_role=AgentRole.RESEARCH,
                agent_model="",
                content={},
                confidence=Decimal("0"),
                reasoning="",
                recommendations=[],
            )).content.get("opportunities", [])

            # Debate for high-value opportunities
            high_value_ops = [o for o in opportunities if o.get("confidence", 0) >= 0.7]
            if high_value_ops:
                debate_rounds = await self._run_debate(
                    markets, responses, full_context
                )

        # Phase 4: Execution Agent plans trades
        if AgentRole.EXECUTION in self._agents:
            exec_response = await self._agents[AgentRole.EXECUTION].analyze(
                markets=markets,
                context=full_context,
                other_responses=[
                    responses.get(AgentRole.RESEARCH),
                    responses.get(AgentRole.RISK),
                ],
            )
            responses[AgentRole.EXECUTION] = exec_response
            total_tokens += exec_response.tokens_used

        # Phase 5: Build final signals
        signals = self._build_signals(markets, responses)

        # Calculate consensus confidence
        confidences = [r.confidence for r in responses.values()]
        consensus = sum(confidences) / len(confidences) if confidences else Decimal("0")

        return CoordinatorDecision(
            signals=signals,
            consensus_confidence=consensus,
            debate_rounds=debate_rounds,
            agent_responses=responses,
            execution_plan=responses.get(AgentRole.EXECUTION, AgentResponse(
                agent_role=AgentRole.EXECUTION,
                agent_model="",
                content={},
                confidence=Decimal("0"),
                reasoning="",
                recommendations=[],
            )).content,
            total_latency_ms=int((datetime.utcnow() - start_time).total_seconds() * 1000),
            total_tokens=total_tokens,
        )

    async def _run_debate(
        self,
        markets: list[MarketState],
        responses: dict[AgentRole, AgentResponse],
        context: str,
    ) -> int:
        """
        Run debate rounds between agents.

        Returns number of debate rounds completed.
        """
        rounds = 0

        for _ in range(self.config.debate_rounds):
            # Research and Risk agents debate
            research_resp = responses.get(AgentRole.RESEARCH)
            risk_resp = responses.get(AgentRole.RISK)

            if not research_resp or not risk_resp:
                break

            # Check if they disagree significantly
            research_conf = research_resp.confidence
            risk_conf = risk_resp.confidence

            if abs(research_conf - risk_conf) < Decimal("0.2"):
                # Close enough consensus
                break

            # Research responds to Risk's concerns
            if AgentRole.RESEARCH in self._agents:
                research_rebuttal = await self._agents[AgentRole.RESEARCH].debate(
                    topic="Trading opportunities",
                    position=research_resp.reasoning,
                    counterargument=risk_resp.reasoning + "\n" + "\n".join(risk_resp.warnings),
                )

                # Update response with refined analysis
                research_resp.reasoning = research_rebuttal

            # Risk responds to refined research
            if AgentRole.RISK in self._agents:
                risk_rebuttal = await self._agents[AgentRole.RISK].debate(
                    topic="Risk assessment",
                    position=risk_resp.reasoning,
                    counterargument=research_rebuttal if 'research_rebuttal' in dir() else "",
                )

                risk_resp.reasoning = risk_rebuttal

            rounds += 1

        return rounds

    def _build_signals(
        self,
        markets: list[MarketState],
        responses: dict[AgentRole, AgentResponse],
    ) -> list[Signal]:
        """Build final trading signals from agent responses."""
        signals = []

        research_resp = responses.get(AgentRole.RESEARCH)
        risk_resp = responses.get(AgentRole.RISK)
        exec_resp = responses.get(AgentRole.EXECUTION)

        if not research_resp or not risk_resp:
            return signals

        # Get approved trades
        approved = {}
        for trade in risk_resp.content.get("approved_trades", []):
            if trade.get("approved"):
                approved[trade.get("market_id")] = trade

        # Get execution plans
        exec_plans = {}
        if exec_resp:
            for plan in exec_resp.content.get("execution_plan", []):
                exec_plans[plan.get("market_id")] = plan

        # Build signals for approved opportunities
        for opp in research_resp.content.get("opportunities", []):
            market_id = opp.get("market_id", "")
            if market_id not in approved:
                continue

            risk_data = approved[market_id]
            exec_data = exec_plans.get(market_id, {})

            # Find market
            market = next((m for m in markets if m.market_id == market_id), None)
            if not market:
                continue

            # Determine signal type
            direction = opp.get("direction", "")
            if "yes" in direction.lower():
                signal_type = SignalType.BUY
                token_id = market.yes_token_id
                target_price = Decimal(str(opp.get("current_price", 0)))
            else:
                signal_type = SignalType.BUY
                token_id = market.no_token_id
                target_price = Decimal(str(opp.get("current_price", 0)))

            # Build edge
            edge = Edge.calculate(
                win_prob=Decimal(str(opp.get("fair_value_estimate", 0.5))),
                win_amount=Decimal("1") - target_price,
                loss_amount=target_price,
                source=opp.get("edge_source", "research"),
            )

            # Build confidence
            confidence = Confidence(
                overall=Decimal(str(opp.get("confidence", 0))),
                model_confidence=Decimal(str(research_resp.confidence)),
                edge_confidence=Decimal(str(opp.get("edge_percentage", 0) / 100)),
                consensus_confidence=Decimal(str(
                    (research_resp.confidence + risk_resp.confidence) / 2
                )),
            )

            signals.append(Signal(
                id=f"sig_{market_id[:8]}_{datetime.utcnow().timestamp():.0f}",
                market_id=market_id,
                token_id=token_id,
                signal_type=signal_type,
                side="buy",
                target_price=target_price,
                size_recommendation=Decimal(str(risk_data.get("recommended_size_usdc", 0))),
                edge=edge,
                confidence=confidence,
                reasoning=opp.get("reasoning", ""),
                sources=[AgentRole.RESEARCH.value, AgentRole.RISK.value],
                metadata={
                    "risk_data": risk_data,
                    "exec_data": exec_data,
                    "opp_data": opp,
                },
            ))

        return signals

    async def reflect_on_trades(self) -> None:
        """Run reflection on recent closed trades."""
        if AgentRole.REFLECTION not in self._agents:
            return

        # Get episodes needing reflection
        episodes = await self.memory.get_episodes_for_reflection()

        for episode in episodes:
            await self._agents[AgentRole.REFLECTION].reflect_on_episode(
                episode=episode,
                memory=self.memory,
            )

    async def get_agent_status(self) -> dict[str, Any]:
        """Get status of all agents."""
        return {
            "agents_active": list(self._agents.keys()),
            "models": {
                role.value: agent.provider.default_model
                for role, agent in self._agents.items()
            },
            "config": {
                "debate_enabled": self.config.enable_multi_agent_debate,
                "debate_rounds": self.config.debate_rounds,
                "reflection_enabled": self.config.enable_reflection,
            },
        }
