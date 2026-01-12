"""Tests for core types and Edge/Kelly calculations."""

import pytest
from datetime import datetime
from decimal import Decimal

from src.core.types import (
    Edge,
    Confidence,
    OrderBook,
    PriceLevel,
    Position,
    Trade,
    Signal,
    SignalType,
    MarketState,
)


class TestEdge:
    """Tests for Edge calculation including Kelly Criterion."""

    def test_edge_positive_ev(self):
        """Test edge calculation with positive expected value."""
        # 60% win probability, 1:1 payout
        edge = Edge.calculate(
            win_prob=Decimal("0.60"),
            win_amount=Decimal("1.0"),
            loss_amount=Decimal("1.0"),
            source="test",
        )

        assert edge.expected_value > 0
        assert edge.win_probability == Decimal("0.60")
        assert edge.payoff_ratio == Decimal("1.0")
        # Kelly = (1*0.6 - 0.4) / 1 = 0.2
        assert edge.kelly_fraction == Decimal("0.2")
        assert edge.half_kelly == Decimal("0.1")

    def test_edge_negative_ev(self):
        """Test edge calculation with negative expected value."""
        # 40% win probability, 1:1 payout
        edge = Edge.calculate(
            win_prob=Decimal("0.40"),
            win_amount=Decimal("1.0"),
            loss_amount=Decimal("1.0"),
            source="test",
        )

        assert edge.expected_value < 0
        # Kelly should be 0 for negative EV
        assert edge.kelly_fraction == Decimal("0")

    def test_edge_high_payout(self):
        """Test edge with high payout ratio."""
        # 30% win prob but 3:1 payout
        edge = Edge.calculate(
            win_prob=Decimal("0.30"),
            win_amount=Decimal("3.0"),
            loss_amount=Decimal("1.0"),
            source="test",
        )

        # EV = 0.3 * 3 - 0.7 * 1 = 0.9 - 0.7 = 0.2
        assert edge.expected_value == Decimal("0.2")
        assert edge.payoff_ratio == Decimal("3.0")
        # Kelly = (3*0.3 - 0.7) / 3 = 0.2 / 3 ≈ 0.067
        assert edge.kelly_fraction > Decimal("0.06")
        assert edge.kelly_fraction < Decimal("0.07")

    def test_edge_zero_loss(self):
        """Test edge with zero loss amount (shouldn't happen but handle gracefully)."""
        edge = Edge.calculate(
            win_prob=Decimal("0.50"),
            win_amount=Decimal("1.0"),
            loss_amount=Decimal("0"),
            source="test",
        )

        assert edge.payoff_ratio == Decimal("0")
        assert edge.kelly_fraction == Decimal("0")

    def test_edge_certainty(self):
        """Test edge with near-certain outcome."""
        edge = Edge.calculate(
            win_prob=Decimal("0.99"),
            win_amount=Decimal("1.0"),
            loss_amount=Decimal("1.0"),
            source="test",
        )

        assert edge.expected_value > Decimal("0.9")
        assert edge.kelly_fraction > Decimal("0.95")

    def test_edge_typical_prediction_market(self):
        """Test typical prediction market scenario."""
        # Buy YES at 0.40, if correct get $1
        # Win amount = 1 - 0.40 = 0.60
        # Loss amount = 0.40
        edge = Edge.calculate(
            win_prob=Decimal("0.55"),  # Our estimate
            win_amount=Decimal("0.60"),
            loss_amount=Decimal("0.40"),
            source="model",
        )

        # EV = 0.55 * 0.60 - 0.45 * 0.40 = 0.33 - 0.18 = 0.15
        assert edge.expected_value > Decimal("0.14")
        assert edge.expected_value < Decimal("0.16")

    def test_edge_positive_ev_check(self):
        """Test checking if edge is positive."""
        positive = Edge.calculate(
            Decimal("0.60"), Decimal("1.0"), Decimal("1.0"), "test"
        )
        assert positive.expected_value > 0

        negative = Edge.calculate(
            Decimal("0.40"), Decimal("1.0"), Decimal("1.0"), "test"
        )
        assert negative.expected_value < 0


class TestConfidence:
    """Tests for Confidence dataclass."""

    def test_confidence_creation(self):
        """Test confidence creation."""
        conf = Confidence(
            overall=Decimal("0.75"),
            model_confidence=Decimal("0.80"),
            edge_confidence=Decimal("0.70"),
            consensus_confidence=Decimal("0.75"),
        )

        assert conf.overall == Decimal("0.75")
        assert conf.model_confidence == Decimal("0.80")

    def test_confidence_clamping(self):
        """Test confidence values are clamped to 0-1."""
        conf = Confidence(
            overall=Decimal("1.5"),
            model_confidence=Decimal("-0.2"),
            edge_confidence=Decimal("0.5"),
            consensus_confidence=Decimal("0.5"),
        )

        # Values should be clamped to valid range
        assert conf.overall == Decimal("1")
        assert conf.model_confidence == Decimal("0")


class TestOrderBook:
    """Tests for OrderBook dataclass."""

    def test_order_book_creation(self):
        """Test order book creation."""
        bids = [
            PriceLevel(price=Decimal("0.45"), size=Decimal("1000")),
            PriceLevel(price=Decimal("0.44"), size=Decimal("2000")),
        ]
        asks = [
            PriceLevel(price=Decimal("0.47"), size=Decimal("1500")),
            PriceLevel(price=Decimal("0.48"), size=Decimal("2500")),
        ]

        ob = OrderBook(
            token_id="tok_yes",
            bids=bids,
            asks=asks,
        )

        assert ob.best_bid == Decimal("0.45")
        assert ob.best_ask == Decimal("0.47")
        assert ob.spread == Decimal("0.02")

    def test_order_book_mid_price(self):
        """Test mid price calculation."""
        bids = [PriceLevel(price=Decimal("0.40"), size=Decimal("1000"))]
        asks = [PriceLevel(price=Decimal("0.50"), size=Decimal("1000"))]

        ob = OrderBook(token_id="tok", bids=bids, asks=asks)

        assert ob.mid_price == Decimal("0.45")

    def test_order_book_empty(self):
        """Test order book with no levels."""
        ob = OrderBook(token_id="tok", bids=[], asks=[])

        assert ob.best_bid is None
        assert ob.best_ask is None
        assert ob.mid_price is None
        assert ob.spread is None


class TestPosition:
    """Tests for Position dataclass."""

    def test_position_creation(self):
        """Test position creation."""
        pos = Position(
            market_id="mkt_abc",
            token_id="tok_yes",
            outcome="yes",
            size=Decimal("100"),
            avg_entry_price=Decimal("0.45"),
            current_price=Decimal("0.55"),
            entry_time=datetime.utcnow(),
        )

        assert pos.size == Decimal("100")
        assert pos.avg_entry_price == Decimal("0.45")
        assert pos.current_price == Decimal("0.55")

    def test_position_market_value(self):
        """Test position market value calculation."""
        pos = Position(
            market_id="mkt_abc",
            token_id="tok_yes",
            outcome="yes",
            size=Decimal("100"),
            avg_entry_price=Decimal("0.40"),
            current_price=Decimal("0.50"),
            entry_time=datetime.utcnow(),
        )

        assert pos.market_value == Decimal("50")  # 100 * 0.50
        assert pos.cost_basis == Decimal("40")  # 100 * 0.40


class TestTrade:
    """Tests for Trade dataclass."""

    def test_trade_creation(self):
        """Test trade creation."""
        trade = Trade(
            id="trade_001",
            market_id="mkt_abc",
            token_id="tok_yes",
            side="buy",
            size=Decimal("50"),
            price=Decimal("0.45"),
            fee=Decimal("0.50"),
            timestamp=datetime.utcnow(),
        )

        assert trade.side == "buy"
        assert trade.size == Decimal("50")

    def test_trade_with_signal(self):
        """Test trade linked to signal."""
        trade = Trade(
            id="trade_002",
            market_id="mkt_abc",
            token_id="tok_yes",
            side="buy",
            size=Decimal("100"),
            price=Decimal("0.50"),
            fee=Decimal("0.50"),
            timestamp=datetime.utcnow(),
            signal_id="sig_001",
        )

        assert trade.signal_id == "sig_001"


class TestSignal:
    """Tests for Signal dataclass."""

    def test_signal_creation(self):
        """Test signal creation."""
        edge = Edge.calculate(
            Decimal("0.60"), Decimal("1.0"), Decimal("1.0"), "test"
        )
        conf = Confidence(
            overall=Decimal("0.75"),
            model_confidence=Decimal("0.80"),
            edge_confidence=Decimal("0.70"),
            consensus_confidence=Decimal("0.75"),
        )

        signal = Signal(
            id="sig_001",
            market_id="mkt_abc",
            token_id="tok_yes",
            signal_type=SignalType.BUY,
            side="buy",
            target_price=Decimal("0.45"),
            size_recommendation=Decimal("100"),
            edge=edge,
            confidence=conf,
            reasoning="Test signal",
            sources=["research", "risk"],
        )

        assert signal.signal_type == SignalType.BUY
        assert signal.target_price == Decimal("0.45")

    def test_signal_is_actionable(self):
        """Test signal actionability."""
        edge = Edge.calculate(
            Decimal("0.60"), Decimal("1.0"), Decimal("1.0"), "test"
        )
        conf = Confidence(
            overall=Decimal("0.75"),
            model_confidence=Decimal("0.80"),
            edge_confidence=Decimal("0.70"),
            consensus_confidence=Decimal("0.75"),
        )

        signal = Signal(
            id="sig_001",
            market_id="mkt_abc",
            token_id="tok_yes",
            signal_type=SignalType.BUY,
            side="buy",
            target_price=Decimal("0.45"),
            size_recommendation=Decimal("100"),
            edge=edge,
            confidence=conf,
            reasoning="Test signal",
            sources=["research"],
        )

        assert signal.is_actionable()

    def test_signal_not_actionable_low_confidence(self):
        """Test signal not actionable with low confidence."""
        edge = Edge.calculate(
            Decimal("0.60"), Decimal("1.0"), Decimal("1.0"), "test"
        )
        conf = Confidence(
            overall=Decimal("0.40"),  # Low confidence
            model_confidence=Decimal("0.40"),
            edge_confidence=Decimal("0.40"),
            consensus_confidence=Decimal("0.40"),
        )

        signal = Signal(
            id="sig_001",
            market_id="mkt_abc",
            token_id="tok_yes",
            signal_type=SignalType.BUY,
            side="buy",
            target_price=Decimal("0.45"),
            size_recommendation=Decimal("100"),
            edge=edge,
            confidence=conf,
            reasoning="Test signal",
            sources=["research"],
        )

        assert not signal.is_actionable()

    def test_signal_hold_not_actionable(self):
        """Test HOLD signal is not actionable."""
        conf = Confidence(
            overall=Decimal("0.90"),
            model_confidence=Decimal("0.90"),
            edge_confidence=Decimal("0.90"),
            consensus_confidence=Decimal("0.90"),
        )

        signal = Signal(
            id="sig_001",
            market_id="mkt_abc",
            token_id="tok_yes",
            signal_type=SignalType.HOLD,
            side=None,
            target_price=None,
            size_recommendation=None,
            edge=None,
            confidence=conf,
            reasoning="No action needed",
            sources=["research"],
        )

        assert not signal.is_actionable()


class TestMarketState:
    """Tests for MarketState dataclass."""

    def test_market_state_creation(self):
        """Test market state creation."""
        state = MarketState(
            market_id="mkt_abc",
            condition_id="cond_123",
            question="Will it rain tomorrow?",
            category="weather",
            end_date=datetime(2024, 12, 31),
            yes_token_id="tok_yes",
            no_token_id="tok_no",
            yes_price=Decimal("0.45"),
            no_price=Decimal("0.55"),
            volume_24h=Decimal("10000"),
            liquidity=Decimal("50000"),
        )

        assert state.yes_price + state.no_price == Decimal("1.0")
        assert state.volume_24h == Decimal("10000")

    def test_market_state_with_order_books(self):
        """Test market state with order books."""
        yes_bids = [PriceLevel(price=Decimal("0.44"), size=Decimal("1000"))]
        yes_asks = [PriceLevel(price=Decimal("0.46"), size=Decimal("1000"))]
        yes_book = OrderBook(token_id="tok_yes", bids=yes_bids, asks=yes_asks)

        state = MarketState(
            market_id="mkt_abc",
            condition_id="cond_123",
            question="Test question",
            category="test",
            end_date=datetime(2024, 12, 31),
            yes_token_id="tok_yes",
            no_token_id="tok_no",
            yes_price=Decimal("0.45"),
            no_price=Decimal("0.55"),
            yes_book=yes_book,
        )

        assert state.spread_yes == Decimal("0.02")
        assert state.spread_no is None  # No no_book set

    def test_market_state_with_indicators(self):
        """Test market state with indicators."""
        state = MarketState(
            market_id="mkt_abc",
            condition_id="cond_123",
            question="Test question",
            category="test",
            end_date=datetime(2024, 12, 31),
            yes_token_id="tok_yes",
            no_token_id="tok_no",
            yes_price=Decimal("0.45"),
            no_price=Decimal("0.55"),
            indicators={
                "ema_9": 0.44,
                "rsi": 55,
                "trend": "up",
            },
        )

        assert state.indicators["ema_9"] == 0.44
        assert state.indicators["trend"] == "up"
