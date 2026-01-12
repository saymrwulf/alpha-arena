"""Base strategy interface and types."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any

from ..broker.base import OrderSide, Position
from ..data.market import MarketInfo
from ..data.sentiment import SentimentData


class SignalType(str, Enum):
    """Type of trading signal."""

    ENTER_LONG = "enter_long"  # Buy YES tokens
    ENTER_SHORT = "enter_short"  # Buy NO tokens (effectively short YES)
    EXIT = "exit"  # Close position
    HOLD = "hold"  # Do nothing


@dataclass
class ExitPlan:
    """Exit strategy for a position."""

    profit_target_price: Decimal | None = None
    stop_loss_price: Decimal | None = None
    max_hold_until: datetime | None = None
    invalidation_conditions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "profit_target_price": str(self.profit_target_price) if self.profit_target_price else None,
            "stop_loss_price": str(self.stop_loss_price) if self.stop_loss_price else None,
            "max_hold_until": self.max_hold_until.isoformat() if self.max_hold_until else None,
            "invalidation_conditions": self.invalidation_conditions,
        }


@dataclass
class TradeSignal:
    """Structured output from strategy decision."""

    market_id: str
    token_id: str
    signal_type: SignalType
    side: OrderSide | None = None
    size: Decimal | None = None
    price: Decimal | None = None
    confidence: Decimal = Decimal("0")  # 0 to 1
    expected_edge: Decimal = Decimal("0")  # Expected profit margin
    exit_plan: ExitPlan | None = None
    reasoning: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "market_id": self.market_id,
            "token_id": self.token_id,
            "signal_type": self.signal_type.value,
            "side": self.side.value if self.side else None,
            "size": str(self.size) if self.size else None,
            "price": str(self.price) if self.price else None,
            "confidence": str(self.confidence),
            "expected_edge": str(self.expected_edge),
            "exit_plan": self.exit_plan.to_dict() if self.exit_plan else None,
            "reasoning": self.reasoning,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class StrategyContext:
    """Context provided to strategy for decision-making."""

    markets: list[MarketInfo]
    positions: list[Position]
    balance: Decimal
    sentiment: dict[str, SentimentData]  # market_id -> sentiment
    recent_fills: list[Any] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class Strategy(ABC):
    """Abstract base class for trading strategies."""

    name: str = "base"
    description: str = "Base strategy"

    @abstractmethod
    async def analyze(self, ctx: StrategyContext) -> list[TradeSignal]:
        """
        Analyze markets and return trading signals.

        Returns a list of signals - can be empty if no opportunities found.
        Signals should be sorted by confidence/priority.
        """
        pass

    @abstractmethod
    async def should_exit(
        self,
        position: Position,
        market: MarketInfo,
        ctx: StrategyContext,
    ) -> TradeSignal | None:
        """
        Check if a position should be exited.

        Returns an EXIT signal if position should be closed, None otherwise.
        """
        pass

    def validate_signal(self, signal: TradeSignal) -> bool:
        """Validate a signal before execution."""
        if signal.signal_type == SignalType.HOLD:
            return True

        if signal.signal_type in (SignalType.ENTER_LONG, SignalType.ENTER_SHORT):
            if not signal.side or not signal.size or not signal.price:
                return False
            if signal.size <= 0 or signal.price <= 0:
                return False
            if signal.confidence < 0 or signal.confidence > 1:
                return False

        return True
