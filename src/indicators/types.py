"""Types for technical indicators."""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any


class TrendDirection(str, Enum):
    """Trend direction."""
    STRONG_UP = "strong_up"
    UP = "up"
    NEUTRAL = "neutral"
    DOWN = "down"
    STRONG_DOWN = "strong_down"


@dataclass
class IndicatorResult:
    """Result of indicator calculation."""
    # Moving Averages
    ema_9: Decimal | None = None
    ema_21: Decimal | None = None
    ema_50: Decimal | None = None
    ema_trend: TrendDirection = TrendDirection.NEUTRAL

    # RSI
    rsi: Decimal | None = None
    rsi_signal: str = "neutral"  # "oversold", "neutral", "overbought"

    # MACD
    macd_line: Decimal | None = None
    macd_signal: Decimal | None = None
    macd_histogram: Decimal | None = None
    macd_crossover: str = "none"  # "bullish", "bearish", "none"

    # Volume
    volume_ma: Decimal | None = None
    volume_ratio: Decimal | None = None  # Current vs MA
    volume_trend: str = "normal"  # "low", "normal", "high", "spike"

    # Volatility
    atr: Decimal | None = None
    volatility_percentile: Decimal | None = None

    # Support/Resistance
    support_levels: list[Decimal] = field(default_factory=list)
    resistance_levels: list[Decimal] = field(default_factory=list)

    # Overall
    overall_signal: str = "neutral"  # "strong_buy", "buy", "neutral", "sell", "strong_sell"
    signal_strength: Decimal = Decimal("0")  # -1 to 1

    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "ema_9": float(self.ema_9) if self.ema_9 else None,
            "ema_21": float(self.ema_21) if self.ema_21 else None,
            "ema_50": float(self.ema_50) if self.ema_50 else None,
            "ema_trend": self.ema_trend.value,
            "rsi": float(self.rsi) if self.rsi else None,
            "rsi_signal": self.rsi_signal,
            "macd_line": float(self.macd_line) if self.macd_line else None,
            "macd_signal": float(self.macd_signal) if self.macd_signal else None,
            "macd_histogram": float(self.macd_histogram) if self.macd_histogram else None,
            "macd_crossover": self.macd_crossover,
            "volume_ratio": float(self.volume_ratio) if self.volume_ratio else None,
            "volume_trend": self.volume_trend,
            "atr": float(self.atr) if self.atr else None,
            "overall_signal": self.overall_signal,
            "signal_strength": float(self.signal_strength),
        }
