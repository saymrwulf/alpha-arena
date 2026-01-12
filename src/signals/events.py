"""Event detection and calendar for market-moving events."""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Optional

import httpx

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Types of market-moving events."""
    # Economic
    FED_MEETING = "fed_meeting"
    FED_SPEECH = "fed_speech"
    FOMC_MINUTES = "fomc_minutes"
    CPI_RELEASE = "cpi_release"
    JOBS_REPORT = "jobs_report"
    GDP_RELEASE = "gdp_release"

    # Political
    ELECTION = "election"
    DEBATE = "debate"
    COURT_RULING = "court_ruling"
    LEGISLATION_VOTE = "legislation_vote"
    EXECUTIVE_ACTION = "executive_action"

    # Corporate
    EARNINGS = "earnings"
    PRODUCT_LAUNCH = "product_launch"
    MERGER_ANNOUNCEMENT = "merger_announcement"

    # Crypto
    HALVING = "halving"
    PROTOCOL_UPGRADE = "protocol_upgrade"
    ETF_DECISION = "etf_decision"

    # Sports
    GAME = "game"
    CHAMPIONSHIP = "championship"
    DRAFT = "draft"

    # Other
    CUSTOM = "custom"


class EventImpact(Enum):
    """Expected market impact of an event."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class MarketEvent:
    """A market-moving event."""
    event_type: EventType
    title: str
    description: str
    timestamp: datetime
    impact: EventImpact

    # Related markets (Polymarket condition IDs or keywords)
    related_markets: list[str] = field(default_factory=list)
    related_keywords: list[str] = field(default_factory=list)

    # Trading implications
    expected_volatility: Decimal = Decimal("0.1")  # 10% expected move
    pre_event_bias: Optional[str] = None  # "bullish", "bearish", or None

    # Source and confidence
    source: str = "manual"
    confidence: Decimal = Decimal("0.8")

    @property
    def is_upcoming(self) -> bool:
        """Check if event is in the future."""
        return self.timestamp > datetime.utcnow()

    @property
    def time_until(self) -> timedelta:
        """Time until event occurs."""
        return self.timestamp - datetime.utcnow()

    @property
    def hours_until(self) -> float:
        """Hours until event (negative if past)."""
        return self.time_until.total_seconds() / 3600

    def affects_market(self, market_question: str) -> bool:
        """Check if this event likely affects a market."""
        question_lower = market_question.lower()

        # Check keywords
        for keyword in self.related_keywords:
            if keyword.lower() in question_lower:
                return True

        # Check by event type
        type_keywords = {
            EventType.FED_MEETING: ["fed", "interest rate", "fomc", "powell"],
            EventType.ELECTION: ["election", "vote", "president", "congress", "senate"],
            EventType.CPI_RELEASE: ["inflation", "cpi", "price"],
            EventType.JOBS_REPORT: ["jobs", "unemployment", "employment", "labor"],
            EventType.EARNINGS: ["earnings", "revenue", "profit"],
            EventType.COURT_RULING: ["court", "ruling", "supreme", "judge", "legal"],
            EventType.HALVING: ["bitcoin", "halving", "btc"],
            EventType.ETF_DECISION: ["etf", "sec", "bitcoin", "crypto"],
        }

        keywords = type_keywords.get(self.event_type, [])
        return any(kw in question_lower for kw in keywords)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "event_type": self.event_type.value,
            "title": self.title,
            "description": self.description,
            "timestamp": self.timestamp.isoformat(),
            "impact": self.impact.value,
            "hours_until": round(self.hours_until, 1),
            "related_markets": self.related_markets,
            "related_keywords": self.related_keywords,
            "expected_volatility": float(self.expected_volatility),
            "pre_event_bias": self.pre_event_bias,
            "confidence": float(self.confidence),
        }


class EventCalendar:
    """
    Calendar of upcoming market-moving events.

    Provides event detection for:
    - Fed meetings and speeches
    - Economic data releases
    - Elections and political events
    - Earnings and corporate events
    - Crypto-specific events
    """

    def __init__(self):
        self._events: list[MarketEvent] = []
        self._last_refresh: Optional[datetime] = None
        self._http_client: Optional[httpx.AsyncClient] = None

        # Initialize with known static events
        self._add_known_events()

    def _add_known_events(self):
        """Add known upcoming events (can be updated periodically)."""
        now = datetime.utcnow()
        year = now.year

        # 2025 FOMC Meeting dates (approximate - verify against actual schedule)
        fomc_dates = [
            datetime(year, 1, 29),
            datetime(year, 3, 19),
            datetime(year, 5, 7),
            datetime(year, 6, 18),
            datetime(year, 7, 30),
            datetime(year, 9, 17),
            datetime(year, 11, 5),
            datetime(year, 12, 17),
        ]

        for date in fomc_dates:
            if date > now:
                self._events.append(MarketEvent(
                    event_type=EventType.FED_MEETING,
                    title=f"FOMC Meeting - {date.strftime('%B %Y')}",
                    description="Federal Reserve interest rate decision",
                    timestamp=date.replace(hour=18, minute=0),  # 2 PM EST
                    impact=EventImpact.CRITICAL,
                    related_keywords=["fed", "interest rate", "fomc", "powell", "inflation"],
                    expected_volatility=Decimal("0.15"),
                ))

        # Monthly jobs report (first Friday of each month)
        for month in range(1, 13):
            # Find first Friday
            first_day = datetime(year, month, 1)
            days_until_friday = (4 - first_day.weekday()) % 7
            jobs_date = first_day + timedelta(days=days_until_friday)

            if jobs_date > now:
                self._events.append(MarketEvent(
                    event_type=EventType.JOBS_REPORT,
                    title=f"Jobs Report - {jobs_date.strftime('%B %Y')}",
                    description="Non-farm payrolls and unemployment rate",
                    timestamp=jobs_date.replace(hour=12, minute=30),  # 8:30 AM EST
                    impact=EventImpact.HIGH,
                    related_keywords=["jobs", "unemployment", "employment", "labor", "economy"],
                    expected_volatility=Decimal("0.08"),
                ))

        # CPI releases (mid-month)
        for month in range(1, 13):
            cpi_date = datetime(year, month, 13)  # Approximate
            if cpi_date > now:
                self._events.append(MarketEvent(
                    event_type=EventType.CPI_RELEASE,
                    title=f"CPI Release - {cpi_date.strftime('%B %Y')}",
                    description="Consumer Price Index inflation data",
                    timestamp=cpi_date.replace(hour=12, minute=30),
                    impact=EventImpact.HIGH,
                    related_keywords=["inflation", "cpi", "prices", "fed"],
                    expected_volatility=Decimal("0.10"),
                ))

    async def refresh_from_api(self) -> None:
        """
        Refresh events from external calendar APIs.

        Could integrate with:
        - Federal Reserve calendar
        - Economic calendar APIs
        - News APIs for event detection
        """
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(timeout=30.0)

        # TODO: Integrate with actual calendar APIs
        # For now, rely on static events + manual additions

        self._last_refresh = datetime.utcnow()

    def add_event(self, event: MarketEvent) -> None:
        """Add a custom event."""
        self._events.append(event)
        # Keep sorted by timestamp
        self._events.sort(key=lambda e: e.timestamp)

    def get_upcoming_events(
        self,
        hours_ahead: int = 72,
        min_impact: EventImpact = EventImpact.MEDIUM,
    ) -> list[MarketEvent]:
        """Get events within the specified time window."""
        now = datetime.utcnow()
        cutoff = now + timedelta(hours=hours_ahead)

        impact_order = [EventImpact.LOW, EventImpact.MEDIUM, EventImpact.HIGH, EventImpact.CRITICAL]
        min_impact_idx = impact_order.index(min_impact)

        return [
            event for event in self._events
            if now < event.timestamp <= cutoff
            and impact_order.index(event.impact) >= min_impact_idx
        ]

    def get_events_for_market(
        self,
        market_question: str,
        hours_ahead: int = 168,  # 1 week
    ) -> list[MarketEvent]:
        """Get events that may affect a specific market."""
        upcoming = self.get_upcoming_events(hours_ahead=hours_ahead, min_impact=EventImpact.LOW)
        return [event for event in upcoming if event.affects_market(market_question)]

    def get_next_critical_event(self) -> Optional[MarketEvent]:
        """Get the next critical impact event."""
        for event in self._events:
            if event.is_upcoming and event.impact == EventImpact.CRITICAL:
                return event
        return None

    def should_reduce_exposure(
        self,
        market_question: str,
        hours_threshold: int = 24,
    ) -> tuple[bool, Optional[MarketEvent]]:
        """
        Check if we should reduce exposure due to upcoming event.

        Returns (should_reduce, triggering_event).
        """
        events = self.get_events_for_market(market_question, hours_ahead=hours_threshold)

        for event in events:
            if event.impact in [EventImpact.HIGH, EventImpact.CRITICAL]:
                return True, event

        return False, None

    def get_event_signal_adjustment(
        self,
        market_question: str,
    ) -> dict:
        """
        Get signal adjustments based on upcoming events.

        Returns dict with:
        - volatility_multiplier: Adjust position size
        - confidence_adjustment: Adjust confidence
        - pre_event_bias: Expected direction
        - reason: Explanation
        """
        events = self.get_events_for_market(market_question, hours_ahead=48)

        if not events:
            return {
                "volatility_multiplier": Decimal("1.0"),
                "confidence_adjustment": Decimal("0"),
                "pre_event_bias": None,
                "reason": "No significant events detected",
                "events": [],
            }

        # Find most impactful event
        impact_order = [EventImpact.LOW, EventImpact.MEDIUM, EventImpact.HIGH, EventImpact.CRITICAL]
        most_impactful = max(events, key=lambda e: impact_order.index(e.impact))

        # Calculate adjustments
        hours_until = most_impactful.hours_until

        if hours_until < 2:
            # Very close to event - reduce exposure significantly
            vol_mult = Decimal("0.25")
            conf_adj = Decimal("-0.3")
        elif hours_until < 6:
            vol_mult = Decimal("0.5")
            conf_adj = Decimal("-0.2")
        elif hours_until < 24:
            vol_mult = Decimal("0.75")
            conf_adj = Decimal("-0.1")
        else:
            vol_mult = Decimal("1.0")
            conf_adj = Decimal("0")

        return {
            "volatility_multiplier": vol_mult,
            "confidence_adjustment": conf_adj,
            "pre_event_bias": most_impactful.pre_event_bias,
            "reason": f"Upcoming: {most_impactful.title} in {hours_until:.1f}h",
            "events": [e.to_dict() for e in events],
        }

    async def close(self) -> None:
        """Cleanup resources."""
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None


# Global calendar instance
_calendar: Optional[EventCalendar] = None


def get_event_calendar() -> EventCalendar:
    """Get or create the global event calendar."""
    global _calendar
    if _calendar is None:
        _calendar = EventCalendar()
    return _calendar
