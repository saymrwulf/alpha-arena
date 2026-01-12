"""Base agent interface for multi-agent trading system."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any

from ..core.types import MarketState, Signal
from ..llm.base import LLMProvider, LLMResponse, Message, Role
from ..memory.manager import MemoryManager


class AgentRole(str, Enum):
    """Agent roles in the trading system."""
    RESEARCH = "research"  # Market analysis and opportunity identification
    RISK = "risk"  # Risk assessment and position sizing
    EXECUTION = "execution"  # Trade execution optimization
    REFLECTION = "reflection"  # Post-trade analysis and learning
    SENTIMENT = "sentiment"  # Social sentiment analysis


@dataclass
class AgentResponse:
    """Response from an agent."""
    agent_role: AgentRole
    agent_model: str
    content: dict[str, Any]
    confidence: Decimal
    reasoning: str
    recommendations: list[str]
    warnings: list[str] = field(default_factory=list)
    latency_ms: int = 0
    tokens_used: int = 0
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_message(self) -> str:
        """Convert to string for inter-agent communication."""
        lines = [
            f"[{self.agent_role.value.upper()} AGENT - {self.agent_model}]",
            f"Confidence: {self.confidence:.0%}",
            "",
            "Analysis:",
            self.reasoning,
            "",
            "Recommendations:",
        ]
        for rec in self.recommendations:
            lines.append(f"  • {rec}")

        if self.warnings:
            lines.append("")
            lines.append("Warnings:")
            for warn in self.warnings:
                lines.append(f"  ⚠ {warn}")

        return "\n".join(lines)


class BaseAgent(ABC):
    """
    Base class for all trading agents.

    Each agent has a specific role and can use any LLM provider.
    Agents communicate through structured messages and can debate.
    """

    role: AgentRole = AgentRole.RESEARCH
    name: str = "base"

    def __init__(
        self,
        provider: LLMProvider,
        memory: MemoryManager | None = None,
    ):
        self.provider = provider
        self.memory = memory
        self._response_count = 0

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """System prompt defining the agent's role and behavior."""
        pass

    @abstractmethod
    async def analyze(
        self,
        markets: list[MarketState],
        context: str,
        other_responses: list[AgentResponse] | None = None,
    ) -> AgentResponse:
        """
        Analyze markets and produce response.

        Args:
            markets: Current market states
            context: Memory context and additional info
            other_responses: Responses from other agents (for debate)
        """
        pass

    async def _call_llm(
        self,
        prompt: str,
        json_mode: bool = False,
    ) -> LLMResponse:
        """Make an LLM call with the agent's system prompt."""
        messages = [Message(role=Role.USER, content=prompt)]
        return await self.provider.complete(
            messages=messages,
            system=self.system_prompt,
            json_mode=json_mode,
        )

    async def debate(
        self,
        topic: str,
        position: str,
        counterargument: str,
    ) -> str:
        """
        Engage in debate with another agent's position.

        Returns refined analysis considering the counterargument.
        """
        prompt = f"""Topic: {topic}

Your previous position:
{position}

Counterargument from another agent:
{counterargument}

Carefully consider this counterargument. You may:
1. Strengthen your position with additional evidence
2. Acknowledge valid points and refine your view
3. Change your position if the counterargument is compelling

Provide your refined analysis."""

        response = await self._call_llm(prompt)
        return response.content

    def _build_market_context(self, markets: list[MarketState]) -> str:
        """Build market context string for prompts."""
        lines = []
        for m in markets[:20]:  # Limit to top 20
            lines.append(f"""
Market: {m.question[:80]}
  ID: {m.condition_id[:16]}...
  YES: ${m.yes_price:.3f} | NO: ${m.no_price:.3f}
  Volume 24h: ${m.volume_24h:,.0f} | Liquidity: ${m.liquidity:,.0f}
  Spread: {m.yes_book.spread_bps:.0f}bps (YES), {m.no_book.spread_bps:.0f}bps (NO)
  Indicators: {self._format_indicators(m.indicators)}
  Sentiment: {self._format_sentiment(m.sentiment)}
""")
        return "\n".join(lines)

    def _format_indicators(self, indicators: dict) -> str:
        """Format technical indicators for display."""
        if not indicators:
            return "N/A"
        parts = []
        if "rsi" in indicators:
            parts.append(f"RSI={indicators['rsi']:.0f}")
        if "macd_signal" in indicators:
            parts.append(f"MACD={'↑' if indicators['macd_signal'] > 0 else '↓'}")
        if "ema_trend" in indicators:
            parts.append(f"Trend={indicators['ema_trend']}")
        return ", ".join(parts) if parts else "N/A"

    def _format_sentiment(self, sentiment: dict) -> str:
        """Format sentiment data for display."""
        if not sentiment:
            return "N/A"
        score = sentiment.get("score", 0)
        label = sentiment.get("label", "neutral")
        return f"{label} ({score:+.2f})"
