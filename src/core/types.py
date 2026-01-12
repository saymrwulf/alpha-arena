"""Core type definitions for the trading harness."""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Literal


class SignalType(str, Enum):
    """Trading signal types."""
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    HOLD = "hold"
    SELL = "sell"
    STRONG_SELL = "strong_sell"
    ARBITRAGE = "arbitrage"


@dataclass(frozen=True)
class Confidence:
    """Confidence level with components."""
    overall: Decimal  # 0-1
    model_confidence: Decimal  # LLM's self-reported confidence
    edge_confidence: Decimal  # Statistical edge confidence
    consensus_confidence: Decimal  # Multi-agent agreement

    def __post_init__(self):
        for name in ['overall', 'model_confidence', 'edge_confidence', 'consensus_confidence']:
            val = getattr(self, name)
            if not (Decimal('0') <= val <= Decimal('1')):
                object.__setattr__(self, name, max(Decimal('0'), min(Decimal('1'), val)))


@dataclass(frozen=True)
class Edge:
    """Estimated edge for a trade."""
    expected_value: Decimal  # E[return]
    win_probability: Decimal  # P(win)
    payoff_ratio: Decimal  # Win amount / Loss amount
    kelly_fraction: Decimal  # Optimal bet size (full Kelly)
    half_kelly: Decimal  # Conservative bet size
    edge_source: str  # Where the edge comes from

    @classmethod
    def calculate(
        cls,
        win_prob: Decimal,
        win_amount: Decimal,
        loss_amount: Decimal,
        source: str = "model",
    ) -> "Edge":
        """Calculate edge metrics from probabilities."""
        lose_prob = Decimal('1') - win_prob

        # Expected value
        ev = win_prob * win_amount - lose_prob * loss_amount

        # Payoff ratio (b in Kelly formula)
        payoff_ratio = win_amount / loss_amount if loss_amount > 0 else Decimal('0')

        # Kelly fraction: f* = (bp - q) / b
        # where b = payoff ratio, p = win prob, q = lose prob
        if payoff_ratio > 0:
            kelly = (payoff_ratio * win_prob - lose_prob) / payoff_ratio
            kelly = max(Decimal('0'), kelly)  # Never negative
        else:
            kelly = Decimal('0')

        return cls(
            expected_value=ev.quantize(Decimal('0.0001')),
            win_probability=win_prob,
            payoff_ratio=payoff_ratio.quantize(Decimal('0.01')),
            kelly_fraction=kelly.quantize(Decimal('0.0001')),
            half_kelly=(kelly / 2).quantize(Decimal('0.0001')),
            edge_source=source,
        )


@dataclass
class PriceLevel:
    """Single price level in order book."""
    price: Decimal
    size: Decimal
    order_count: int = 1


@dataclass
class OrderBook:
    """Order book snapshot."""
    token_id: str
    bids: list[PriceLevel]  # Sorted high to low
    asks: list[PriceLevel]  # Sorted low to high
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @property
    def best_bid(self) -> Decimal | None:
        return self.bids[0].price if self.bids else None

    @property
    def best_ask(self) -> Decimal | None:
        return self.asks[0].price if self.asks else None

    @property
    def mid_price(self) -> Decimal | None:
        if self.best_bid and self.best_ask:
            return (self.best_bid + self.best_ask) / 2
        return None

    @property
    def spread(self) -> Decimal | None:
        if self.best_bid and self.best_ask:
            return self.best_ask - self.best_bid
        return None

    @property
    def spread_bps(self) -> Decimal | None:
        """Spread in basis points."""
        if self.mid_price and self.spread:
            return (self.spread / self.mid_price) * 10000
        return None

    def depth_at_price(self, side: Literal["bid", "ask"], price: Decimal) -> Decimal:
        """Get cumulative depth up to a price level."""
        levels = self.bids if side == "bid" else self.asks
        total = Decimal('0')
        for level in levels:
            if side == "bid" and level.price >= price:
                total += level.size
            elif side == "ask" and level.price <= price:
                total += level.size
        return total


@dataclass
class Position:
    """Current position in a market."""
    market_id: str
    token_id: str
    outcome: str
    size: Decimal
    avg_entry_price: Decimal
    current_price: Decimal
    entry_time: datetime
    unrealized_pnl: Decimal = Decimal('0')
    realized_pnl: Decimal = Decimal('0')
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def market_value(self) -> Decimal:
        return self.size * self.current_price

    @property
    def cost_basis(self) -> Decimal:
        return self.size * self.avg_entry_price

    @property
    def return_pct(self) -> Decimal:
        if self.cost_basis > 0:
            return (self.unrealized_pnl / self.cost_basis) * 100
        return Decimal('0')

    @property
    def hold_duration_hours(self) -> float:
        return (datetime.utcnow() - self.entry_time).total_seconds() / 3600


@dataclass
class Trade:
    """Executed trade record."""
    id: str
    market_id: str
    token_id: str
    side: Literal["buy", "sell"]
    size: Decimal
    price: Decimal
    fee: Decimal
    timestamp: datetime
    signal_id: str | None = None
    edge: Edge | None = None
    confidence: Confidence | None = None
    pnl: Decimal | None = None  # Set when position closed
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Signal:
    """Trading signal from analysis."""
    id: str
    market_id: str
    token_id: str
    signal_type: SignalType
    side: Literal["buy", "sell"] | None
    target_price: Decimal | None
    size_recommendation: Decimal | None  # From Kelly
    edge: Edge | None
    confidence: Confidence
    reasoning: str
    sources: list[str]  # Which agents/models contributed
    timestamp: datetime = field(default_factory=datetime.utcnow)
    expires_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def is_actionable(self, min_confidence: Decimal = Decimal('0.6')) -> bool:
        """Check if signal should be acted upon."""
        if self.signal_type == SignalType.HOLD:
            return False
        if self.confidence.overall < min_confidence:
            return False
        if self.edge and self.edge.expected_value <= 0:
            return False
        if self.expires_at and datetime.utcnow() > self.expires_at:
            return False
        return True


@dataclass
class MarketState:
    """Complete market state snapshot."""
    market_id: str
    condition_id: str
    question: str
    category: str
    end_date: datetime | None

    # Tokens
    yes_token_id: str
    no_token_id: str
    yes_price: Decimal
    no_price: Decimal

    # Order books
    yes_book: OrderBook | None = None
    no_book: OrderBook | None = None

    # Market metrics
    volume_24h: Decimal = Decimal('0')
    liquidity: Decimal = Decimal('0')
    open_interest: Decimal = Decimal('0')

    # Computed
    implied_probability: Decimal = Decimal('0')  # From yes_price

    # Technical indicators (populated by indicator module)
    indicators: dict[str, Any] = field(default_factory=dict)

    # Sentiment (populated by sentiment module)
    sentiment: dict[str, Any] = field(default_factory=dict)

    # Arbitrage opportunities
    arbitrage: dict[str, Any] = field(default_factory=dict)

    timestamp: datetime = field(default_factory=datetime.utcnow)

    @property
    def spread_yes(self) -> Decimal | None:
        return self.yes_book.spread if self.yes_book else None

    @property
    def spread_no(self) -> Decimal | None:
        return self.no_book.spread if self.no_book else None
