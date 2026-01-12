"""Signal aggregation and scoring module."""

from .aggregator import (
    SignalAggregator,
    AggregatedSignal,
    SignalComponent,
    SignalSource,
    SignalDirection,
    get_signal_aggregator,
)
from .events import (
    EventCalendar,
    MarketEvent,
    EventType,
    EventImpact,
    get_event_calendar,
)
from .news import (
    NewsProvider,
    NewsItem,
    NewsSentiment,
    NewsSource,
    get_news_provider,
)

__all__ = [
    # Aggregator
    "SignalAggregator",
    "AggregatedSignal",
    "SignalComponent",
    "SignalSource",
    "SignalDirection",
    "get_signal_aggregator",
    # Events
    "EventCalendar",
    "MarketEvent",
    "EventType",
    "EventImpact",
    "get_event_calendar",
    # News
    "NewsProvider",
    "NewsItem",
    "NewsSentiment",
    "NewsSource",
    "get_news_provider",
]
