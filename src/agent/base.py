"""Base agent interface."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any

from ..broker.base import Broker, Order, Position
from ..data.market import MarketInfo
from ..data.sentiment import SentimentData
from ..strategy.base import TradeSignal


class AgentState(str, Enum):
    """Agent lifecycle states."""

    IDLE = "idle"
    OBSERVING = "observing"
    DECIDING = "deciding"
    EXECUTING = "executing"
    LOGGING = "logging"
    ERROR = "error"
    STOPPED = "stopped"


@dataclass
class AgentConfig:
    """Agent configuration."""

    name: str = "default"
    loop_interval_seconds: int = 60
    max_iterations: int | None = None
    llm_provider: str = "anthropic"
    llm_model: str = "claude-sonnet-4-20250514"
    temperature: float = 0.3
    max_tokens: int = 4096


@dataclass
class AgentObservation:
    """What the agent observes each cycle."""

    timestamp: datetime
    balance: Decimal
    positions: list[Position]
    markets: list[MarketInfo]
    sentiment: dict[str, SentimentData]
    open_orders: list[Order]
    recent_fills: list[Any] = field(default_factory=list)


@dataclass
class AgentDecision:
    """Agent's decision for the current cycle."""

    timestamp: datetime
    signals: list[TradeSignal]
    reasoning: str
    model_used: str
    latency_ms: int
    tokens_used: int = 0


class Agent(ABC):
    """Abstract base agent for trading."""

    def __init__(self, config: AgentConfig):
        self.config = config
        self.state = AgentState.IDLE
        self._iteration = 0

    @abstractmethod
    async def observe(self, broker: Broker, markets: list[MarketInfo], sentiment: dict[str, SentimentData]) -> AgentObservation:
        """Gather current market state."""
        pass

    @abstractmethod
    async def decide(self, observation: AgentObservation) -> AgentDecision:
        """Make trading decisions based on observation."""
        pass

    @abstractmethod
    async def execute(self, decision: AgentDecision, broker: Broker) -> list[Order]:
        """Execute trading decisions."""
        pass

    async def run_cycle(
        self,
        broker: Broker,
        markets: list[MarketInfo],
        sentiment: dict[str, SentimentData],
    ) -> tuple[AgentObservation, AgentDecision, list[Order]]:
        """Run one observe -> decide -> execute cycle."""
        self._iteration += 1

        self.state = AgentState.OBSERVING
        observation = await self.observe(broker, markets, sentiment)

        self.state = AgentState.DECIDING
        decision = await self.decide(observation)

        self.state = AgentState.EXECUTING
        orders = await self.execute(decision, broker)

        self.state = AgentState.IDLE
        return observation, decision, orders

    @property
    def iteration(self) -> int:
        """Current iteration count."""
        return self._iteration
