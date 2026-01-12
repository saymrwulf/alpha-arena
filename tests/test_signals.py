"""Tests for the signals module."""

import asyncio
import pytest
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

from src.signals.events import (
    EventCalendar,
    MarketEvent,
    EventType,
    EventImpact,
    get_event_calendar,
)
from src.signals.news import (
    NewsProvider,
    NewsItem,
    NewsSentiment,
    NewsSource,
    get_news_provider,
)
from src.signals.aggregator import (
    SignalAggregator,
    AggregatedSignal,
    SignalComponent,
    SignalSource,
    SignalDirection,
    get_signal_aggregator,
)


# =============================================================================
# Event Calendar Tests
# =============================================================================

class TestEventType:
    """Tests for EventType enum."""

    def test_all_event_types_exist(self):
        assert EventType.FED_MEETING.value == "fed_meeting"
        assert EventType.ELECTION.value == "election"
        assert EventType.CPI_RELEASE.value == "cpi_release"
        assert EventType.HALVING.value == "halving"
        assert EventType.GAME.value == "game"


class TestEventImpact:
    """Tests for EventImpact enum."""

    def test_impact_levels(self):
        assert EventImpact.LOW.value == "low"
        assert EventImpact.MEDIUM.value == "medium"
        assert EventImpact.HIGH.value == "high"
        assert EventImpact.CRITICAL.value == "critical"


class TestMarketEvent:
    """Tests for MarketEvent dataclass."""

    @pytest.fixture
    def future_event(self):
        return MarketEvent(
            event_type=EventType.FED_MEETING,
            title="FOMC Meeting",
            description="Interest rate decision",
            timestamp=datetime.utcnow() + timedelta(hours=24),
            impact=EventImpact.CRITICAL,
            related_keywords=["fed", "interest rate"],
        )

    @pytest.fixture
    def past_event(self):
        return MarketEvent(
            event_type=EventType.CPI_RELEASE,
            title="CPI Release",
            description="Inflation data",
            timestamp=datetime.utcnow() - timedelta(hours=2),
            impact=EventImpact.HIGH,
        )

    def test_is_upcoming_future(self, future_event):
        assert future_event.is_upcoming is True

    def test_is_upcoming_past(self, past_event):
        assert past_event.is_upcoming is False

    def test_hours_until_positive(self, future_event):
        assert future_event.hours_until > 0
        assert 23 < future_event.hours_until < 25

    def test_hours_until_negative(self, past_event):
        assert past_event.hours_until < 0

    def test_affects_market_keyword_match(self, future_event):
        assert future_event.affects_market("Will the Fed raise interest rates?")
        assert future_event.affects_market("FOMC meeting impact")

    def test_affects_market_no_match(self, future_event):
        assert not future_event.affects_market("Bitcoin price prediction")

    def test_affects_market_type_keywords(self):
        event = MarketEvent(
            event_type=EventType.ELECTION,
            title="US Presidential Election",
            description="",
            timestamp=datetime.utcnow() + timedelta(days=30),
            impact=EventImpact.CRITICAL,
        )
        assert event.affects_market("Who will win the election?")
        assert event.affects_market("Senate vote prediction")
        assert not event.affects_market("Bitcoin ETF approval")

    def test_to_dict(self, future_event):
        d = future_event.to_dict()
        assert d["event_type"] == "fed_meeting"
        assert d["title"] == "FOMC Meeting"
        assert d["impact"] == "critical"
        assert "hours_until" in d


class TestEventCalendar:
    """Tests for EventCalendar."""

    @pytest.fixture
    def calendar(self):
        return EventCalendar()

    def test_calendar_has_known_events(self, calendar):
        # Should have FOMC, jobs, CPI events pre-populated
        all_events = calendar._events
        assert len(all_events) > 0

        event_types = {e.event_type for e in all_events}
        assert EventType.FED_MEETING in event_types or EventType.JOBS_REPORT in event_types

    def test_add_event(self, calendar):
        custom_event = MarketEvent(
            event_type=EventType.CUSTOM,
            title="Custom Event",
            description="Test",
            timestamp=datetime.utcnow() + timedelta(hours=12),
            impact=EventImpact.MEDIUM,
        )
        initial_count = len(calendar._events)
        calendar.add_event(custom_event)
        assert len(calendar._events) == initial_count + 1

    def test_events_sorted_by_timestamp(self, calendar):
        calendar.add_event(MarketEvent(
            event_type=EventType.CUSTOM,
            title="Event A",
            description="",
            timestamp=datetime.utcnow() + timedelta(hours=48),
            impact=EventImpact.LOW,
        ))
        calendar.add_event(MarketEvent(
            event_type=EventType.CUSTOM,
            title="Event B",
            description="",
            timestamp=datetime.utcnow() + timedelta(hours=1),
            impact=EventImpact.LOW,
        ))
        # Should be sorted
        for i in range(len(calendar._events) - 1):
            assert calendar._events[i].timestamp <= calendar._events[i + 1].timestamp

    def test_get_upcoming_events(self, calendar):
        events = calendar.get_upcoming_events(hours_ahead=720, min_impact=EventImpact.LOW)
        for event in events:
            assert event.is_upcoming
            assert event.hours_until <= 720

    def test_get_upcoming_events_impact_filter(self, calendar):
        # Add events with different impacts
        calendar.add_event(MarketEvent(
            event_type=EventType.CUSTOM,
            title="Low Impact",
            description="",
            timestamp=datetime.utcnow() + timedelta(hours=6),
            impact=EventImpact.LOW,
        ))
        calendar.add_event(MarketEvent(
            event_type=EventType.CUSTOM,
            title="Critical Impact",
            description="",
            timestamp=datetime.utcnow() + timedelta(hours=6),
            impact=EventImpact.CRITICAL,
        ))

        high_events = calendar.get_upcoming_events(hours_ahead=24, min_impact=EventImpact.HIGH)
        for event in high_events:
            assert event.impact in [EventImpact.HIGH, EventImpact.CRITICAL]

    def test_get_events_for_market(self, calendar):
        events = calendar.get_events_for_market(
            "Will inflation exceed 3% this month?",
            hours_ahead=720,
        )
        # Should find CPI-related events
        if events:
            for event in events:
                assert event.affects_market("Will inflation exceed 3% this month?")

    def test_should_reduce_exposure_no_events(self, calendar):
        should_reduce, event = calendar.should_reduce_exposure(
            "Random unrelated market question xyz123",
            hours_threshold=24,
        )
        assert should_reduce is False
        assert event is None

    def test_get_event_signal_adjustment_no_events(self, calendar):
        adjustment = calendar.get_event_signal_adjustment("Random xyz market")
        assert adjustment["volatility_multiplier"] == Decimal("1.0")
        assert adjustment["confidence_adjustment"] == Decimal("0")
        assert adjustment["reason"] == "No significant events detected"

    def test_get_next_critical_event(self, calendar):
        # Add a critical event
        calendar.add_event(MarketEvent(
            event_type=EventType.FED_MEETING,
            title="Next FOMC",
            description="",
            timestamp=datetime.utcnow() + timedelta(hours=12),
            impact=EventImpact.CRITICAL,
        ))
        critical = calendar.get_next_critical_event()
        assert critical is not None
        assert critical.impact == EventImpact.CRITICAL


class TestGetEventCalendar:
    """Tests for get_event_calendar singleton."""

    def test_returns_same_instance(self):
        cal1 = get_event_calendar()
        cal2 = get_event_calendar()
        assert cal1 is cal2


# =============================================================================
# News Provider Tests
# =============================================================================

class TestNewsSentiment:
    """Tests for NewsSentiment enum."""

    def test_sentiment_values(self):
        assert NewsSentiment.VERY_BEARISH.value == "very_bearish"
        assert NewsSentiment.NEUTRAL.value == "neutral"
        assert NewsSentiment.VERY_BULLISH.value == "very_bullish"


class TestNewsSource:
    """Tests for NewsSource enum."""

    def test_source_values(self):
        assert NewsSource.TWITTER.value == "twitter"
        assert NewsSource.NEWS_API.value == "news_api"
        assert NewsSource.RSS.value == "rss"


class TestNewsItem:
    """Tests for NewsItem dataclass."""

    @pytest.fixture
    def fresh_news(self):
        return NewsItem(
            title="Breaking: Fed announces rate hike",
            content="The Federal Reserve announced a 25bps rate increase...",
            source=NewsSource.NEWS_API,
            url="https://example.com/article",
            published_at=datetime.utcnow() - timedelta(hours=1),
            sentiment=NewsSentiment.BEARISH,
            sentiment_score=Decimal("-0.4"),
            confidence=Decimal("0.8"),
            is_breaking=True,
        )

    @pytest.fixture
    def old_news(self):
        return NewsItem(
            title="Old market update",
            content="...",
            source=NewsSource.RSS,
            url=None,
            published_at=datetime.utcnow() - timedelta(hours=12),
        )

    def test_age_hours(self, fresh_news, old_news):
        assert 0.5 < fresh_news.age_hours < 1.5
        assert 11 < old_news.age_hours < 13

    def test_is_fresh(self, fresh_news, old_news):
        assert fresh_news.is_fresh is True
        assert old_news.is_fresh is False

    def test_weighted_sentiment(self, fresh_news):
        # weighted = score * confidence * freshness
        ws = fresh_news.weighted_sentiment
        # Should be negative (bearish)
        assert ws < 0
        # Should be dampened by freshness weight
        assert ws > Decimal("-0.4")

    def test_to_dict(self, fresh_news):
        d = fresh_news.to_dict()
        assert d["title"] == "Breaking: Fed announces rate hike"
        assert d["source"] == "news_api"
        assert d["is_breaking"] is True
        assert "age_hours" in d


class TestNewsProvider:
    """Tests for NewsProvider."""

    @pytest.fixture
    def provider(self):
        return NewsProvider()

    def test_provider_init(self, provider):
        assert provider._http_client is None
        assert provider._cache == {}

    @pytest.mark.asyncio
    async def test_connect_creates_client(self, provider):
        await provider.connect()
        assert provider._http_client is not None
        await provider.disconnect()

    @pytest.mark.asyncio
    async def test_disconnect_cleans_up(self, provider):
        await provider.connect()
        await provider.disconnect()
        assert provider._http_client is None

    def test_extract_keywords(self, provider):
        keywords = provider._extract_keywords(
            "Will Bitcoin exceed $100,000 by December 2024?"
        )
        assert "Bitcoin" in keywords or "bitcoin" in [k.lower() for k in keywords]
        assert "December" in keywords or "december" in [k.lower() for k in keywords]
        # Stop words should be filtered
        assert "will" not in [k.lower() for k in keywords]
        assert "by" not in [k.lower() for k in keywords]

    def test_clean_html(self, provider):
        html = "<p>Hello &amp; <b>world</b>!</p>"
        cleaned = provider._clean_html(html)
        assert cleaned == "Hello & world!"

    def test_clean_html_cdata(self, provider):
        cdata = "<![CDATA[Test content here]]>"
        cleaned = provider._clean_html(cdata)
        assert cleaned == "Test content here"

    def test_deduplicate(self, provider):
        items = [
            NewsItem(
                title="Breaking: Fed raises rates",
                content="",
                source=NewsSource.NEWS_API,
                url=None,
                published_at=datetime.utcnow(),
            ),
            NewsItem(
                title="Breaking: Fed raises rates!",  # Similar
                content="",
                source=NewsSource.RSS,
                url=None,
                published_at=datetime.utcnow(),
            ),
            NewsItem(
                title="Markets respond to inflation data",  # Different
                content="",
                source=NewsSource.NEWS_API,
                url=None,
                published_at=datetime.utcnow(),
            ),
        ]
        unique = provider._deduplicate(items)
        assert len(unique) == 2

    def test_simple_sentiment_bullish(self, provider):
        sentiment, score, conf = provider._simple_sentiment(
            "Markets surge on strong earnings, profits beat expectations"
        )
        assert sentiment in [NewsSentiment.BULLISH, NewsSentiment.VERY_BULLISH]
        assert score > 0

    def test_simple_sentiment_bearish(self, provider):
        sentiment, score, conf = provider._simple_sentiment(
            "Markets crash amid fears of recession, losses mount"
        )
        assert sentiment in [NewsSentiment.BEARISH, NewsSentiment.VERY_BEARISH]
        assert score < 0

    def test_simple_sentiment_neutral(self, provider):
        sentiment, score, conf = provider._simple_sentiment(
            "The company announced quarterly results"
        )
        assert sentiment == NewsSentiment.NEUTRAL
        assert score == Decimal("0.0")

    def test_score_relevance(self, provider):
        item = NewsItem(
            title="Fed meeting impacts markets",
            content="Federal Reserve decision affects rates",
            source=NewsSource.NEWS_API,
            url=None,
            published_at=datetime.utcnow(),
            is_breaking=True,
        )
        score = provider._score_relevance(
            item,
            "Will the Fed raise rates?",
            ["Fed", "rates", "meeting"],
        )
        # Breaking news + keyword matches + fresh = high score
        assert score >= Decimal("0.5")

    @pytest.mark.asyncio
    async def test_get_news_for_market_empty_cache(self, provider):
        # Mock HTTP client to avoid actual requests
        with patch.object(provider, "_fetch_from_rss", return_value=[]):
            await provider.connect()
            news = await provider.get_news_for_market(
                "Test market question",
                hours_back=24,
            )
            # Empty result is fine for test
            assert isinstance(news, list)
            await provider.disconnect()


class TestGetNewsProvider:
    """Tests for get_news_provider singleton."""

    def test_returns_same_instance(self):
        p1 = get_news_provider()
        p2 = get_news_provider()
        assert p1 is p2


# =============================================================================
# Signal Aggregator Tests
# =============================================================================

class TestSignalSource:
    """Tests for SignalSource enum."""

    def test_source_values(self):
        assert SignalSource.TECHNICAL.value == "technical"
        assert SignalSource.NEWS.value == "news"
        assert SignalSource.LLM_ANALYSIS.value == "llm_analysis"


class TestSignalDirection:
    """Tests for SignalDirection enum."""

    def test_direction_values(self):
        assert SignalDirection.STRONG_YES.value == "strong_yes"
        assert SignalDirection.NEUTRAL.value == "neutral"
        assert SignalDirection.STRONG_NO.value == "strong_no"


class TestSignalComponent:
    """Tests for SignalComponent dataclass."""

    @pytest.fixture
    def bullish_signal(self):
        return SignalComponent(
            source=SignalSource.NEWS,
            direction=SignalDirection.YES,
            strength=Decimal("0.8"),
            confidence=Decimal("0.7"),
            reasoning="Positive news sentiment",
        )

    @pytest.fixture
    def bearish_signal(self):
        return SignalComponent(
            source=SignalSource.TECHNICAL,
            direction=SignalDirection.NO,
            strength=Decimal("0.6"),
            confidence=Decimal("0.5"),
            reasoning="Negative momentum",
        )

    def test_weighted_score_bullish(self, bullish_signal):
        # direction(0.5) * strength(0.8) * confidence(0.7) = 0.28
        ws = bullish_signal.weighted_score
        assert ws > 0
        assert abs(ws - Decimal("0.28")) < Decimal("0.01")

    def test_weighted_score_bearish(self, bearish_signal):
        # direction(-0.5) * strength(0.6) * confidence(0.5) = -0.15
        ws = bearish_signal.weighted_score
        assert ws < 0
        assert abs(ws - Decimal("-0.15")) < Decimal("0.01")

    def test_weighted_score_neutral(self):
        signal = SignalComponent(
            source=SignalSource.EVENT,
            direction=SignalDirection.NEUTRAL,
            strength=Decimal("0.9"),
            confidence=Decimal("0.9"),
            reasoning="No clear direction",
        )
        assert signal.weighted_score == Decimal("0")

    def test_to_dict(self, bullish_signal):
        d = bullish_signal.to_dict()
        assert d["source"] == "news"
        assert d["direction"] == "yes"
        assert d["strength"] == 0.8
        assert "weighted_score" in d


class TestAggregatedSignal:
    """Tests for AggregatedSignal dataclass."""

    @pytest.fixture
    def aggregated_signal(self):
        return AggregatedSignal(
            market_id="market123",
            market_question="Will X happen?",
            timestamp=datetime.utcnow(),
            components=[],
            direction=SignalDirection.YES,
            final_score=Decimal("0.5"),
            confidence=Decimal("0.7"),
            edge_estimate=Decimal("0.05"),
            recommended_side="YES",
            recommended_size_pct=Decimal("0.3"),
        )

    def test_to_dict(self, aggregated_signal):
        d = aggregated_signal.to_dict()
        assert d["market_id"] == "market123"
        assert d["direction"] == "yes"
        assert d["final_score"] == 0.5
        assert d["recommended_side"] == "YES"


class TestSignalAggregator:
    """Tests for SignalAggregator."""

    @pytest.fixture
    def aggregator(self):
        return SignalAggregator(
            min_confidence=Decimal("0.5"),
            min_edge=Decimal("0.03"),
        )

    def test_default_weights(self, aggregator):
        assert SignalSource.LLM_ANALYSIS in aggregator._weights
        assert SignalSource.NEWS in aggregator._weights
        assert aggregator._weights[SignalSource.LLM_ANALYSIS] == Decimal("0.30")

    def test_score_to_direction(self, aggregator):
        assert aggregator._score_to_direction(Decimal("0.8")) == SignalDirection.STRONG_YES
        assert aggregator._score_to_direction(Decimal("0.3")) == SignalDirection.YES
        assert aggregator._score_to_direction(Decimal("0.0")) == SignalDirection.NEUTRAL
        assert aggregator._score_to_direction(Decimal("-0.3")) == SignalDirection.NO
        assert aggregator._score_to_direction(Decimal("-0.8")) == SignalDirection.STRONG_NO

    def test_estimate_fair_value(self, aggregator):
        # Bullish signal should increase fair value
        fv = aggregator._estimate_fair_value(Decimal("0.5"), Decimal("0.5"))
        assert fv > Decimal("0.5")

        # Bearish signal should decrease fair value
        fv = aggregator._estimate_fair_value(Decimal("-0.5"), Decimal("0.5"))
        assert fv < Decimal("0.5")

        # Neutral should keep near market
        fv = aggregator._estimate_fair_value(Decimal("0.0"), Decimal("0.5"))
        assert fv == Decimal("0.5")

    def test_aggregate_components_empty(self, aggregator):
        score, conf, conflict = aggregator._aggregate_components([])
        assert score == Decimal("0")
        assert conf == Decimal("0")
        assert conflict is False

    def test_aggregate_components_single(self, aggregator):
        components = [
            SignalComponent(
                source=SignalSource.NEWS,
                direction=SignalDirection.YES,
                strength=Decimal("0.8"),
                confidence=Decimal("0.7"),
                reasoning="Test",
            )
        ]
        score, conf, conflict = aggregator._aggregate_components(components)
        assert score > 0
        assert conf > 0
        assert conflict is False

    def test_aggregate_components_conflicting(self, aggregator):
        components = [
            SignalComponent(
                source=SignalSource.NEWS,
                direction=SignalDirection.YES,
                strength=Decimal("0.8"),
                confidence=Decimal("0.7"),
                reasoning="Bullish news",
            ),
            SignalComponent(
                source=SignalSource.TECHNICAL,
                direction=SignalDirection.NO,
                strength=Decimal("0.8"),
                confidence=Decimal("0.7"),
                reasoning="Bearish technicals",
            ),
        ]
        score, conf, conflict = aggregator._aggregate_components(components)
        assert conflict is True
        # Confidence should be reduced due to conflict
        assert conf < Decimal("0.7")

    def test_get_recommendation_low_confidence(self, aggregator):
        side, size, reason = aggregator._get_recommendation(
            direction=SignalDirection.YES,
            final_score=Decimal("0.5"),
            confidence=Decimal("0.3"),  # Below threshold
            edge=Decimal("0.05"),
            volatility_adj=Decimal("1.0"),
            conflicting=False,
        )
        assert side is None
        assert size == Decimal("0")
        assert "confidence" in reason.lower()

    def test_get_recommendation_low_edge(self, aggregator):
        side, size, reason = aggregator._get_recommendation(
            direction=SignalDirection.YES,
            final_score=Decimal("0.5"),
            confidence=Decimal("0.7"),
            edge=Decimal("0.01"),  # Below threshold
            volatility_adj=Decimal("1.0"),
            conflicting=False,
        )
        assert side is None
        assert "edge" in reason.lower()

    def test_get_recommendation_conflicting(self, aggregator):
        side, size, reason = aggregator._get_recommendation(
            direction=SignalDirection.YES,
            final_score=Decimal("0.5"),
            confidence=Decimal("0.7"),
            edge=Decimal("0.05"),
            volatility_adj=Decimal("1.0"),
            conflicting=True,
        )
        assert side is None
        assert "conflict" in reason.lower()

    def test_get_recommendation_valid_yes(self, aggregator):
        side, size, reason = aggregator._get_recommendation(
            direction=SignalDirection.YES,
            final_score=Decimal("0.5"),
            confidence=Decimal("0.7"),
            edge=Decimal("0.05"),
            volatility_adj=Decimal("1.0"),
            conflicting=False,
        )
        assert side == "YES"
        assert size > Decimal("0")
        assert reason is None

    def test_get_recommendation_valid_no(self, aggregator):
        side, size, reason = aggregator._get_recommendation(
            direction=SignalDirection.NO,
            final_score=Decimal("-0.5"),
            confidence=Decimal("0.7"),
            edge=Decimal("-0.05"),
            volatility_adj=Decimal("1.0"),
            conflicting=False,
        )
        assert side == "NO"
        assert size > Decimal("0")

    def test_create_technical_signal(self, aggregator):
        signal = aggregator.create_technical_signal(
            price_momentum=Decimal("0.5"),
            volume_trend=Decimal("0.3"),
            rsi=Decimal("65"),
        )
        assert signal.source == SignalSource.TECHNICAL
        assert signal.direction in [SignalDirection.YES, SignalDirection.STRONG_YES]

    def test_create_technical_signal_overbought(self, aggregator):
        signal = aggregator.create_technical_signal(
            price_momentum=Decimal("0.3"),
            volume_trend=Decimal("0.2"),
            rsi=Decimal("75"),  # Overbought
        )
        # RSI overbought should dampen bullish signal
        assert signal.direction != SignalDirection.STRONG_YES

    def test_create_orderbook_signal_imbalance(self, aggregator):
        signal = aggregator.create_orderbook_signal(
            bid_depth=Decimal("10000"),
            ask_depth=Decimal("5000"),
            spread_pct=Decimal("0.02"),
        )
        assert signal.source == SignalSource.ORDERBOOK
        # More bids than asks = bullish
        assert signal.direction in [SignalDirection.YES, SignalDirection.STRONG_YES]

    def test_create_orderbook_signal_no_depth(self, aggregator):
        signal = aggregator.create_orderbook_signal(
            bid_depth=Decimal("0"),
            ask_depth=Decimal("0"),
            spread_pct=Decimal("0.1"),
        )
        assert signal.direction == SignalDirection.NEUTRAL
        assert signal.strength == Decimal("0")

    def test_create_llm_signal(self, aggregator):
        signal = aggregator.create_llm_signal(
            direction=SignalDirection.YES,
            confidence=Decimal("0.8"),
            reasoning="Based on fundamental analysis...",
            model="claude-sonnet-4-20250514",
        )
        assert signal.source == SignalSource.LLM_ANALYSIS
        assert signal.metadata["model"] == "claude-sonnet-4-20250514"

    @pytest.mark.asyncio
    async def test_aggregate_signals(self, aggregator):
        # Mock the news provider to avoid HTTP calls
        with patch.object(aggregator._news_provider, "connect", new_callable=AsyncMock):
            with patch.object(aggregator._news_provider, "get_news_for_market", return_value=[]):
                components = [
                    SignalComponent(
                        source=SignalSource.LLM_ANALYSIS,
                        direction=SignalDirection.YES,
                        strength=Decimal("0.7"),
                        confidence=Decimal("0.8"),
                        reasoning="LLM thinks YES",
                    ),
                ]

                result = await aggregator.aggregate_signals(
                    market_id="test123",
                    market_question="Will X happen?",
                    current_price=Decimal("0.5"),
                    components=components,
                    fetch_news=True,
                    check_events=True,
                )

                assert isinstance(result, AggregatedSignal)
                assert result.market_id == "test123"
                assert len(result.components) >= 1


class TestGetSignalAggregator:
    """Tests for get_signal_aggregator singleton."""

    def test_returns_same_instance(self):
        agg1 = get_signal_aggregator()
        agg2 = get_signal_aggregator()
        assert agg1 is agg2


# =============================================================================
# Integration Tests
# =============================================================================

class TestSignalsIntegration:
    """Integration tests for the complete signals system."""

    @pytest.mark.asyncio
    async def test_end_to_end_signal_aggregation(self):
        """Test complete signal aggregation flow."""
        aggregator = SignalAggregator()

        # Create strong signal components for clear direction
        tech_signal = aggregator.create_technical_signal(
            price_momentum=Decimal("0.7"),
            volume_trend=Decimal("0.6"),
            rsi=Decimal("45"),  # Not overbought
        )

        orderbook_signal = aggregator.create_orderbook_signal(
            bid_depth=Decimal("15000"),
            ask_depth=Decimal("5000"),  # Strong imbalance
            spread_pct=Decimal("0.01"),
        )

        llm_signal = aggregator.create_llm_signal(
            direction=SignalDirection.STRONG_YES,
            confidence=Decimal("0.85"),
            reasoning="Analysis strongly supports YES outcome",
            model="test-model",
        )

        # Mock news to avoid HTTP
        with patch.object(aggregator._news_provider, "connect", new_callable=AsyncMock):
            with patch.object(aggregator._news_provider, "get_news_for_market", return_value=[]):
                result = await aggregator.aggregate_signals(
                    market_id="integration_test",
                    market_question="Will the market move up?",
                    current_price=Decimal("0.45"),
                    components=[tech_signal, orderbook_signal, llm_signal],
                    fetch_news=True,
                    check_events=True,
                )

        # Verify result structure
        assert result.market_id == "integration_test"
        assert len(result.components) >= 3
        assert result.final_score > Decimal("0")  # Should be positive with bullish signals
        assert result.confidence > Decimal("0")

        # Should recommend YES side with strong bullish signals
        if not result.conflicting_signals and result.direction != SignalDirection.NEUTRAL:
            assert result.recommended_side == "YES"

    @pytest.mark.asyncio
    async def test_event_calendar_integration(self):
        """Test event calendar affects signal adjustment."""
        calendar = EventCalendar()

        # Add an imminent high-impact event
        calendar.add_event(MarketEvent(
            event_type=EventType.FED_MEETING,
            title="Urgent Fed Meeting",
            description="",
            timestamp=datetime.utcnow() + timedelta(hours=1),
            impact=EventImpact.CRITICAL,
            related_keywords=["fed", "rates"],
        ))

        adjustment = calendar.get_event_signal_adjustment("Fed interest rates decision")

        # Should reduce exposure near critical event
        assert adjustment["volatility_multiplier"] < Decimal("1.0")
        assert adjustment["confidence_adjustment"] < Decimal("0")
