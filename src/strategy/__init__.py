"""Strategy plugin interface for trading decisions."""

from .base import (
    Strategy,
    TradeSignal,
    SignalType,
    ExitPlan,
    StrategyContext,
)
from .sentiment_momentum import SentimentMomentumStrategy

__all__ = [
    "Strategy",
    "TradeSignal",
    "SignalType",
    "ExitPlan",
    "StrategyContext",
    "SentimentMomentumStrategy",
]
