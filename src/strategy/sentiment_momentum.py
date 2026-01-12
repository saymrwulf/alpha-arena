"""Sentiment-based momentum strategy."""

from datetime import datetime, timedelta
from decimal import Decimal

from ..broker.base import OrderSide, Position
from ..data.market import MarketInfo
from ..data.sentiment import SentimentData, SentimentLevel
from .base import (
    ExitPlan,
    SignalType,
    Strategy,
    StrategyContext,
    TradeSignal,
)


class SentimentMomentumStrategy(Strategy):
    """
    Strategy based on social sentiment momentum.

    Buy YES when sentiment is positive and rising.
    Buy NO when sentiment is negative and falling.
    Exit on sentiment reversal or profit target/stop loss.
    """

    name = "sentiment_momentum"
    description = "Trade based on social media sentiment momentum"

    def __init__(
        self,
        min_confidence: Decimal = Decimal("0.6"),
        min_edge: Decimal = Decimal("0.05"),
        profit_target_pct: Decimal = Decimal("0.15"),
        stop_loss_pct: Decimal = Decimal("0.10"),
        max_hold_hours: int = 72,
        min_sentiment_volume: int = 10,
    ):
        self.min_confidence = min_confidence
        self.min_edge = min_edge
        self.profit_target_pct = profit_target_pct
        self.stop_loss_pct = stop_loss_pct
        self.max_hold_hours = max_hold_hours
        self.min_sentiment_volume = min_sentiment_volume

    async def analyze(self, ctx: StrategyContext) -> list[TradeSignal]:
        """Analyze markets for sentiment-based opportunities."""
        signals = []

        for market in ctx.markets:
            if market.resolved or not market.active:
                continue

            # Skip if already have position in this market
            if any(p.market_id == market.condition_id for p in ctx.positions):
                continue

            # Get sentiment for this market
            sentiment = ctx.sentiment.get(market.condition_id)
            if not sentiment or sentiment.volume < self.min_sentiment_volume:
                continue

            signal = self._analyze_market(market, sentiment, ctx.balance)
            if signal and signal.confidence >= self.min_confidence:
                signals.append(signal)

        # Sort by confidence descending
        signals.sort(key=lambda s: s.confidence, reverse=True)
        return signals

    def _analyze_market(
        self,
        market: MarketInfo,
        sentiment: SentimentData,
        balance: Decimal,
    ) -> TradeSignal | None:
        """Analyze a single market for trading opportunity."""
        yes_token = market.yes_token
        no_token = market.no_token

        if not yes_token or not no_token:
            return None

        # Determine direction based on sentiment
        if sentiment.level in (SentimentLevel.POSITIVE, SentimentLevel.VERY_POSITIVE):
            # Bullish sentiment -> buy YES
            token = yes_token
            side = OrderSide.BUY
            signal_type = SignalType.ENTER_LONG

            # Calculate edge: if sentiment suggests YES, current price should be undervalued
            fair_value = self._sentiment_to_fair_value(sentiment.score)
            edge = fair_value - token.price

        elif sentiment.level in (SentimentLevel.NEGATIVE, SentimentLevel.VERY_NEGATIVE):
            # Bearish sentiment -> buy NO (short YES)
            token = no_token
            side = OrderSide.BUY
            signal_type = SignalType.ENTER_SHORT

            # For NO tokens, inverse the fair value
            fair_value = Decimal("1") - self._sentiment_to_fair_value(sentiment.score)
            edge = fair_value - token.price

        else:
            # Neutral sentiment - no trade
            return None

        # Check if edge is sufficient
        if edge < self.min_edge:
            return None

        # Calculate position size (simple: fixed % of balance)
        size = min(balance * Decimal("0.1"), Decimal("25"))  # Max $25 per trade

        # Calculate confidence based on sentiment strength and volume
        base_confidence = abs(sentiment.score)
        volume_factor = min(Decimal(str(sentiment.volume)) / Decimal("100"), Decimal("1"))
        confidence = base_confidence * Decimal("0.7") + volume_factor * Decimal("0.3")

        # Create exit plan
        exit_plan = ExitPlan(
            profit_target_price=token.price + (token.price * self.profit_target_pct),
            stop_loss_price=token.price - (token.price * self.stop_loss_pct),
            max_hold_until=datetime.utcnow() + timedelta(hours=self.max_hold_hours),
            invalidation_conditions=[
                f"Sentiment reverses to {SentimentLevel.NEUTRAL.value}",
                "Market resolved",
            ],
        )

        return TradeSignal(
            market_id=market.condition_id,
            token_id=token.token_id,
            signal_type=signal_type,
            side=side,
            size=size,
            price=token.price,
            confidence=confidence.quantize(Decimal("0.01")),
            expected_edge=edge.quantize(Decimal("0.01")),
            exit_plan=exit_plan,
            reasoning=self._generate_reasoning(market, sentiment, token, edge),
        )

    def _sentiment_to_fair_value(self, score: Decimal) -> Decimal:
        """Convert sentiment score to estimated fair probability."""
        # Map [-1, 1] to [0.2, 0.8] as fair value range
        # Extreme sentiments don't mean 0% or 100% probability
        normalized = (score + Decimal("1")) / Decimal("2")  # [0, 1]
        fair_value = Decimal("0.2") + normalized * Decimal("0.6")  # [0.2, 0.8]
        return fair_value.quantize(Decimal("0.01"))

    def _generate_reasoning(
        self,
        market: MarketInfo,
        sentiment: SentimentData,
        token,
        edge: Decimal,
    ) -> str:
        """Generate human-readable reasoning for the trade."""
        direction = "bullish" if sentiment.score > 0 else "bearish"
        return (
            f"Sentiment analysis shows {direction} sentiment "
            f"(score: {sentiment.score}, volume: {sentiment.volume} posts). "
            f"Current price {token.price} appears undervalued with "
            f"expected edge of {edge:.1%}. "
            f"Market: {market.question[:100]}"
        )

    async def should_exit(
        self,
        position: Position,
        market: MarketInfo,
        ctx: StrategyContext,
    ) -> TradeSignal | None:
        """Check if position should be exited."""
        # Check profit target
        if position.unrealized_pnl >= position.avg_entry_price * self.profit_target_pct * position.size:
            return TradeSignal(
                market_id=position.market_id,
                token_id=position.token_id,
                signal_type=SignalType.EXIT,
                side=OrderSide.SELL,
                size=position.size,
                price=position.current_price,
                confidence=Decimal("0.9"),
                reasoning="Profit target reached",
            )

        # Check stop loss
        if position.unrealized_pnl <= -(position.avg_entry_price * self.stop_loss_pct * position.size):
            return TradeSignal(
                market_id=position.market_id,
                token_id=position.token_id,
                signal_type=SignalType.EXIT,
                side=OrderSide.SELL,
                size=position.size,
                price=position.current_price,
                confidence=Decimal("0.95"),
                reasoning="Stop loss triggered",
            )

        # Check sentiment reversal
        sentiment = ctx.sentiment.get(position.market_id)
        if sentiment:
            # If we're long YES and sentiment turned negative, exit
            # If we're long NO and sentiment turned positive, exit
            is_yes_position = position.outcome.lower() in ("yes", "true", "1")

            if is_yes_position and sentiment.level in (SentimentLevel.NEGATIVE, SentimentLevel.VERY_NEGATIVE):
                return TradeSignal(
                    market_id=position.market_id,
                    token_id=position.token_id,
                    signal_type=SignalType.EXIT,
                    side=OrderSide.SELL,
                    size=position.size,
                    price=position.current_price,
                    confidence=Decimal("0.7"),
                    reasoning=f"Sentiment reversal to {sentiment.level.value}",
                )

            if not is_yes_position and sentiment.level in (SentimentLevel.POSITIVE, SentimentLevel.VERY_POSITIVE):
                return TradeSignal(
                    market_id=position.market_id,
                    token_id=position.token_id,
                    signal_type=SignalType.EXIT,
                    side=OrderSide.SELL,
                    size=position.size,
                    price=position.current_price,
                    confidence=Decimal("0.7"),
                    reasoning=f"Sentiment reversal to {sentiment.level.value}",
                )

        # Check if market resolved
        if market.resolved:
            return TradeSignal(
                market_id=position.market_id,
                token_id=position.token_id,
                signal_type=SignalType.EXIT,
                side=OrderSide.SELL,
                size=position.size,
                price=position.current_price,
                confidence=Decimal("1.0"),
                reasoning="Market resolved",
            )

        return None
