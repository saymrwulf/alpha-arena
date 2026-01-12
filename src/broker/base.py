"""Abstract broker interface - can be swapped for different APIs."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderStatus(str, Enum):
    PENDING = "pending"
    OPEN = "open"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class Order:
    """Represents a trading order."""

    id: str
    market_id: str
    token_id: str
    side: OrderSide
    size: Decimal
    price: Decimal
    status: OrderStatus = OrderStatus.PENDING
    filled_size: Decimal = Decimal("0")
    avg_fill_price: Decimal | None = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Fill:
    """Represents a trade fill."""

    order_id: str
    market_id: str
    token_id: str
    side: OrderSide
    size: Decimal
    price: Decimal
    fee: Decimal
    timestamp: datetime


@dataclass
class Position:
    """Represents an open position."""

    market_id: str
    token_id: str
    outcome: str
    size: Decimal
    avg_entry_price: Decimal
    current_price: Decimal
    unrealized_pnl: Decimal
    realized_pnl: Decimal = Decimal("0")


class Broker(ABC):
    """Abstract broker interface for market execution."""

    @abstractmethod
    async def connect(self) -> None:
        """Connect to the broker API."""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from the broker API."""
        pass

    @abstractmethod
    async def get_balance(self) -> Decimal:
        """Get current USDC balance."""
        pass

    @abstractmethod
    async def get_positions(self) -> list[Position]:
        """Get all open positions."""
        pass

    @abstractmethod
    async def get_position(self, market_id: str, token_id: str) -> Position | None:
        """Get a specific position."""
        pass

    @abstractmethod
    async def place_order(
        self,
        market_id: str,
        token_id: str,
        side: OrderSide,
        size: Decimal,
        price: Decimal,
    ) -> Order:
        """Place a limit order."""
        pass

    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order."""
        pass

    @abstractmethod
    async def get_order(self, order_id: str) -> Order | None:
        """Get order status."""
        pass

    @abstractmethod
    async def get_open_orders(self) -> list[Order]:
        """Get all open orders."""
        pass

    @abstractmethod
    async def get_fills(self, since: datetime | None = None) -> list[Fill]:
        """Get recent fills."""
        pass
