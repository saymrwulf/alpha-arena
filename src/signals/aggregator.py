"""Signal aggregation and scoring for trading decisions."""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Optional

from .events import EventCalendar, MarketEvent, EventImpact, get_event_calendar
from .news import NewsProvider, NewsItem, NewsSentiment, get_news_provider

logger = logging.getLogger(__name__)


class SignalSource(Enum):
    """Sources of trading signals."""
    TECHNICAL = "technical"
    SENTIMENT = "sentiment"
    NEWS = "news"
    EVENT = "event"
    ORDERBOOK = "orderbook"
    LLM_ANALYSIS = "llm_analysis"
    HISTORICAL = "historical"


class SignalDirection(Enum):
    """Trading signal direction."""
    STRONG_YES = "strong_yes"
    YES = "yes"
    NEUTRAL = "neutral"
    NO = "no"
    STRONG_NO = "strong_no"


@dataclass
class SignalComponent:
    """A single signal component from one source."""
    source: SignalSource
    direction: SignalDirection
    strength: Decimal  # 0.0 to 1.0
    confidence: Decimal  # 0.0 to 1.0
    reasoning: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: dict = field(default_factory=dict)

    @property
    def weighted_score(self) -> Decimal:
        """
        Calculate weighted score.

        Score ranges from -1.0 (strong no) to 1.0 (strong yes).
        """
        direction_scores = {
            SignalDirection.STRONG_YES: Decimal("1.0"),
            SignalDirection.YES: Decimal("0.5"),
            SignalDirection.NEUTRAL: Decimal("0.0"),
            SignalDirection.NO: Decimal("-0.5"),
            SignalDirection.STRONG_NO: Decimal("-1.0"),
        }
        base_score = direction_scores[self.direction]
        return base_score * self.strength * self.confidence

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "source": self.source.value,
            "direction": self.direction.value,
            "strength": float(self.strength),
            "confidence": float(self.confidence),
            "weighted_score": float(self.weighted_score),
            "reasoning": self.reasoning,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class AggregatedSignal:
    """Aggregated signal from multiple sources."""
    market_id: str
    market_question: str
    timestamp: datetime

    # Component signals
    components: list[SignalComponent]

    # Aggregated results
    direction: SignalDirection
    final_score: Decimal  # -1.0 to 1.0
    confidence: Decimal  # 0.0 to 1.0
    edge_estimate: Decimal  # Expected edge in probability

    # Trading recommendation
    recommended_side: Optional[str] = None  # "YES" or "NO"
    recommended_size_pct: Decimal = Decimal("0")  # % of max position
    hold_reason: Optional[str] = None

    # Risk factors
    volatility_adjustment: Decimal = Decimal("1.0")
    upcoming_events: list[dict] = field(default_factory=list)
    conflicting_signals: bool = False

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "market_id": self.market_id,
            "market_question": self.market_question,
            "timestamp": self.timestamp.isoformat(),
            "direction": self.direction.value,
            "final_score": float(self.final_score),
            "confidence": float(self.confidence),
            "edge_estimate": float(self.edge_estimate),
            "recommended_side": self.recommended_side,
            "recommended_size_pct": float(self.recommended_size_pct),
            "hold_reason": self.hold_reason,
            "volatility_adjustment": float(self.volatility_adjustment),
            "conflicting_signals": self.conflicting_signals,
            "components": [c.to_dict() for c in self.components],
            "upcoming_events": self.upcoming_events,
        }


class SignalAggregator:
    """
    Aggregates signals from multiple sources into a unified trading signal.

    Combines:
    - Technical indicators (price momentum, volume)
    - News sentiment
    - Event calendar impact
    - Order book analysis
    - LLM analysis
    - Historical patterns
    """

    # Source weights for aggregation
    DEFAULT_WEIGHTS = {
        SignalSource.LLM_ANALYSIS: Decimal("0.30"),
        SignalSource.NEWS: Decimal("0.20"),
        SignalSource.SENTIMENT: Decimal("0.15"),
        SignalSource.EVENT: Decimal("0.15"),
        SignalSource.TECHNICAL: Decimal("0.10"),
        SignalSource.ORDERBOOK: Decimal("0.10"),
    }

    def __init__(
        self,
        event_calendar: Optional[EventCalendar] = None,
        news_provider: Optional[NewsProvider] = None,
        weights: Optional[dict[SignalSource, Decimal]] = None,
        min_confidence: Decimal = Decimal("0.5"),
        min_edge: Decimal = Decimal("0.03"),
    ):
        self._event_calendar = event_calendar or get_event_calendar()
        self._news_provider = news_provider or get_news_provider()
        self._weights = weights or self.DEFAULT_WEIGHTS
        self._min_confidence = min_confidence
        self._min_edge = min_edge

    async def aggregate_signals(
        self,
        market_id: str,
        market_question: str,
        current_price: Decimal,
        components: Optional[list[SignalComponent]] = None,
        fetch_news: bool = True,
        check_events: bool = True,
    ) -> AggregatedSignal:
        """
        Aggregate all available signals for a market.

        Args:
            market_id: The market's unique identifier
            market_question: The market question text
            current_price: Current YES price (0.0 to 1.0)
            components: Pre-computed signal components
            fetch_news: Whether to fetch fresh news
            check_events: Whether to check event calendar

        Returns:
            AggregatedSignal with trading recommendation
        """
        all_components = list(components) if components else []

        # Add event-based signal
        if check_events:
            event_signal = self._get_event_signal(market_question)
            if event_signal:
                all_components.append(event_signal)

        # Add news-based signal
        if fetch_news:
            news_signal = await self._get_news_signal(market_question)
            if news_signal:
                all_components.append(news_signal)

        # Aggregate the signals
        final_score, confidence, conflicting = self._aggregate_components(all_components)

        # Determine direction
        direction = self._score_to_direction(final_score)

        # Calculate edge estimate
        fair_value = self._estimate_fair_value(final_score, current_price)
        edge = fair_value - current_price if final_score > 0 else current_price - (Decimal("1") - fair_value)

        # Get event adjustments
        event_adjustment = self._event_calendar.get_event_signal_adjustment(market_question)
        volatility_adj = event_adjustment.get("volatility_multiplier", Decimal("1.0"))
        upcoming_events = event_adjustment.get("events", [])

        # Determine trading recommendation
        recommended_side, size_pct, hold_reason = self._get_recommendation(
            direction=direction,
            final_score=final_score,
            confidence=confidence,
            edge=edge,
            volatility_adj=volatility_adj,
            conflicting=conflicting,
        )

        return AggregatedSignal(
            market_id=market_id,
            market_question=market_question,
            timestamp=datetime.utcnow(),
            components=all_components,
            direction=direction,
            final_score=final_score,
            confidence=confidence,
            edge_estimate=edge,
            recommended_side=recommended_side,
            recommended_size_pct=size_pct,
            hold_reason=hold_reason,
            volatility_adjustment=volatility_adj,
            upcoming_events=upcoming_events,
            conflicting_signals=conflicting,
        )

    def _aggregate_components(
        self,
        components: list[SignalComponent],
    ) -> tuple[Decimal, Decimal, bool]:
        """
        Aggregate component signals into final score and confidence.

        Returns (final_score, confidence, has_conflicts).
        """
        if not components:
            return Decimal("0"), Decimal("0"), False

        # Group by source and take most recent
        by_source: dict[SignalSource, SignalComponent] = {}
        for comp in sorted(components, key=lambda c: c.timestamp):
            by_source[comp.source] = comp

        # Calculate weighted average
        total_weight = Decimal("0")
        weighted_sum = Decimal("0")
        confidence_sum = Decimal("0")

        for source, component in by_source.items():
            weight = self._weights.get(source, Decimal("0.1"))
            weighted_sum += component.weighted_score * weight
            confidence_sum += component.confidence * weight
            total_weight += weight

        if total_weight == 0:
            return Decimal("0"), Decimal("0"), False

        final_score = weighted_sum / total_weight
        avg_confidence = confidence_sum / total_weight

        # Check for conflicting signals
        directions = [c.direction for c in by_source.values()]
        bullish = sum(1 for d in directions if d in [SignalDirection.YES, SignalDirection.STRONG_YES])
        bearish = sum(1 for d in directions if d in [SignalDirection.NO, SignalDirection.STRONG_NO])
        conflicting = bullish > 0 and bearish > 0 and abs(bullish - bearish) <= 1

        # Reduce confidence if signals conflict
        if conflicting:
            avg_confidence *= Decimal("0.7")

        return final_score, avg_confidence, conflicting

    def _score_to_direction(self, score: Decimal) -> SignalDirection:
        """Convert numeric score to direction enum."""
        if score > Decimal("0.6"):
            return SignalDirection.STRONG_YES
        elif score > Decimal("0.2"):
            return SignalDirection.YES
        elif score < Decimal("-0.6"):
            return SignalDirection.STRONG_NO
        elif score < Decimal("-0.2"):
            return SignalDirection.NO
        else:
            return SignalDirection.NEUTRAL

    def _estimate_fair_value(
        self,
        score: Decimal,
        current_price: Decimal,
    ) -> Decimal:
        """
        Estimate fair value based on signal score.

        Uses the score as a bias adjustment from market price.
        """
        # Score of 1.0 suggests we're 90% confident in YES
        # Score of -1.0 suggests we're 90% confident in NO
        # Score of 0 means fair value equals market price

        max_adjustment = Decimal("0.15")  # Maximum 15% adjustment from market
        adjustment = score * max_adjustment

        fair_value = current_price + adjustment
        return max(Decimal("0.01"), min(Decimal("0.99"), fair_value))

    def _get_recommendation(
        self,
        direction: SignalDirection,
        final_score: Decimal,
        confidence: Decimal,
        edge: Decimal,
        volatility_adj: Decimal,
        conflicting: bool,
    ) -> tuple[Optional[str], Decimal, Optional[str]]:
        """
        Generate trading recommendation.

        Returns (side, size_pct, hold_reason).
        """
        # Check minimum thresholds
        if confidence < self._min_confidence:
            return None, Decimal("0"), f"Low confidence ({float(confidence):.2f} < {float(self._min_confidence):.2f})"

        if abs(edge) < self._min_edge:
            return None, Decimal("0"), f"Insufficient edge ({float(edge):.1%} < {float(self._min_edge):.1%})"

        if direction == SignalDirection.NEUTRAL:
            return None, Decimal("0"), "Neutral signal - no clear direction"

        if conflicting:
            return None, Decimal("0"), "Conflicting signals - waiting for clarity"

        # Determine side
        if direction in [SignalDirection.STRONG_YES, SignalDirection.YES]:
            side = "YES"
        else:
            side = "NO"

        # Calculate position size (Kelly-inspired)
        # Size based on edge and confidence, scaled by volatility adjustment
        raw_size = abs(edge) * confidence * Decimal("2")  # Scale up
        size_pct = raw_size * volatility_adj

        # Cap at 100%
        size_pct = min(Decimal("1.0"), size_pct)

        # Scale based on signal strength
        if direction in [SignalDirection.STRONG_YES, SignalDirection.STRONG_NO]:
            size_pct *= Decimal("1.2")
        elif direction in [SignalDirection.YES, SignalDirection.NO]:
            size_pct *= Decimal("0.8")

        size_pct = min(Decimal("1.0"), size_pct)

        return side, size_pct, None

    def _get_event_signal(self, market_question: str) -> Optional[SignalComponent]:
        """Generate signal based on upcoming events."""
        events = self._event_calendar.get_events_for_market(market_question, hours_ahead=72)

        if not events:
            return None

        # Find most impactful upcoming event
        critical_events = [e for e in events if e.impact in [EventImpact.HIGH, EventImpact.CRITICAL]]

        if not critical_events:
            return None

        event = critical_events[0]

        # Determine signal based on pre-event bias and timing
        hours_until = event.hours_until

        if hours_until < 2:
            # Very close to event - high uncertainty
            direction = SignalDirection.NEUTRAL
            strength = Decimal("0.2")
            reasoning = f"Event imminent: {event.title}. High uncertainty."
        elif event.pre_event_bias == "bullish":
            direction = SignalDirection.YES
            strength = Decimal("0.6")
            reasoning = f"Bullish bias expected for {event.title}"
        elif event.pre_event_bias == "bearish":
            direction = SignalDirection.NO
            strength = Decimal("0.6")
            reasoning = f"Bearish bias expected for {event.title}"
        else:
            direction = SignalDirection.NEUTRAL
            strength = Decimal("0.3")
            reasoning = f"Upcoming event: {event.title} in {hours_until:.1f}h"

        return SignalComponent(
            source=SignalSource.EVENT,
            direction=direction,
            strength=strength,
            confidence=event.confidence,
            reasoning=reasoning,
            metadata={"event": event.to_dict()},
        )

    async def _get_news_signal(self, market_question: str) -> Optional[SignalComponent]:
        """Generate signal based on recent news."""
        try:
            await self._news_provider.connect()
            news_items = await self._news_provider.get_news_for_market(
                market_question,
                hours_back=24,
                max_items=10,
            )
        except Exception as e:
            logger.warning(f"Failed to fetch news: {e}")
            return None

        if not news_items:
            return None

        # Aggregate news sentiment
        total_sentiment = Decimal("0")
        total_weight = Decimal("0")

        for item in news_items:
            weight = item.impact_score * (Decimal("1") - Decimal(str(item.age_hours / 48)))
            total_sentiment += item.weighted_sentiment * weight
            total_weight += weight

        if total_weight == 0:
            return None

        avg_sentiment = total_sentiment / total_weight

        # Convert to direction
        if avg_sentiment > Decimal("0.3"):
            direction = SignalDirection.STRONG_YES if avg_sentiment > Decimal("0.6") else SignalDirection.YES
        elif avg_sentiment < Decimal("-0.3"):
            direction = SignalDirection.STRONG_NO if avg_sentiment < Decimal("-0.6") else SignalDirection.NO
        else:
            direction = SignalDirection.NEUTRAL

        # Confidence based on news volume and consistency
        confidence = min(Decimal("0.8"), Decimal(str(len(news_items) * 0.1 + 0.3)))

        breaking_news = [n for n in news_items if n.is_breaking]
        reasoning = f"Based on {len(news_items)} news items"
        if breaking_news:
            reasoning += f" ({len(breaking_news)} breaking)"

        return SignalComponent(
            source=SignalSource.NEWS,
            direction=direction,
            strength=min(Decimal("1.0"), abs(avg_sentiment)),
            confidence=confidence,
            reasoning=reasoning,
            metadata={
                "news_count": len(news_items),
                "avg_sentiment": float(avg_sentiment),
                "top_headlines": [n.title for n in news_items[:3]],
            },
        )

    def create_technical_signal(
        self,
        price_momentum: Decimal,
        volume_trend: Decimal,
        rsi: Optional[Decimal] = None,
    ) -> SignalComponent:
        """
        Create a technical analysis signal.

        Args:
            price_momentum: Price change rate (-1 to 1)
            volume_trend: Volume change rate (-1 to 1)
            rsi: RSI indicator (0-100)
        """
        # Combine momentum and volume
        combined = price_momentum * Decimal("0.6") + volume_trend * Decimal("0.4")

        # Adjust for RSI extremes
        if rsi is not None:
            if rsi > Decimal("70"):
                combined -= Decimal("0.2")  # Overbought
            elif rsi < Decimal("30"):
                combined += Decimal("0.2")  # Oversold

        # Determine direction
        if combined > Decimal("0.3"):
            direction = SignalDirection.STRONG_YES if combined > Decimal("0.6") else SignalDirection.YES
        elif combined < Decimal("-0.3"):
            direction = SignalDirection.STRONG_NO if combined < Decimal("-0.6") else SignalDirection.NO
        else:
            direction = SignalDirection.NEUTRAL

        reasoning = f"Momentum: {float(price_momentum):.2f}, Volume: {float(volume_trend):.2f}"
        if rsi:
            reasoning += f", RSI: {float(rsi):.0f}"

        return SignalComponent(
            source=SignalSource.TECHNICAL,
            direction=direction,
            strength=min(Decimal("1.0"), abs(combined)),
            confidence=Decimal("0.6"),  # Technical signals have moderate confidence
            reasoning=reasoning,
            metadata={
                "price_momentum": float(price_momentum),
                "volume_trend": float(volume_trend),
                "rsi": float(rsi) if rsi else None,
            },
        )

    def create_orderbook_signal(
        self,
        bid_depth: Decimal,
        ask_depth: Decimal,
        spread_pct: Decimal,
    ) -> SignalComponent:
        """
        Create an order book analysis signal.

        Args:
            bid_depth: Total bid volume
            ask_depth: Total ask volume
            spread_pct: Bid-ask spread as percentage
        """
        if bid_depth + ask_depth == 0:
            return SignalComponent(
                source=SignalSource.ORDERBOOK,
                direction=SignalDirection.NEUTRAL,
                strength=Decimal("0"),
                confidence=Decimal("0.2"),
                reasoning="No orderbook depth",
            )

        # Calculate imbalance
        imbalance = (bid_depth - ask_depth) / (bid_depth + ask_depth)

        # Higher spread = lower confidence
        spread_factor = max(Decimal("0.3"), Decimal("1") - spread_pct * Decimal("5"))

        if imbalance > Decimal("0.2"):
            direction = SignalDirection.YES if imbalance < Decimal("0.5") else SignalDirection.STRONG_YES
        elif imbalance < Decimal("-0.2"):
            direction = SignalDirection.NO if imbalance > Decimal("-0.5") else SignalDirection.STRONG_NO
        else:
            direction = SignalDirection.NEUTRAL

        return SignalComponent(
            source=SignalSource.ORDERBOOK,
            direction=direction,
            strength=min(Decimal("1.0"), abs(imbalance)),
            confidence=Decimal("0.5") * spread_factor,
            reasoning=f"Book imbalance: {float(imbalance):.2f}, spread: {float(spread_pct):.2%}",
            metadata={
                "bid_depth": float(bid_depth),
                "ask_depth": float(ask_depth),
                "imbalance": float(imbalance),
                "spread_pct": float(spread_pct),
            },
        )

    def create_llm_signal(
        self,
        direction: SignalDirection,
        confidence: Decimal,
        reasoning: str,
        model: str,
    ) -> SignalComponent:
        """
        Create signal from LLM analysis.

        Args:
            direction: LLM's determined direction
            confidence: LLM's confidence level
            reasoning: LLM's explanation
            model: Model used for analysis
        """
        return SignalComponent(
            source=SignalSource.LLM_ANALYSIS,
            direction=direction,
            strength=confidence,  # LLM strength equals confidence
            confidence=confidence,
            reasoning=reasoning,
            metadata={"model": model},
        )


# Global aggregator instance
_aggregator: Optional[SignalAggregator] = None


def get_signal_aggregator() -> SignalAggregator:
    """Get or create the global signal aggregator."""
    global _aggregator
    if _aggregator is None:
        _aggregator = SignalAggregator()
    return _aggregator
