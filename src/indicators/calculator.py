"""Technical indicator calculator."""

from collections import deque
from decimal import Decimal
from typing import Any

from ..core.config import IndicatorConfig
from .types import IndicatorResult, TrendDirection


class IndicatorCalculator:
    """
    Calculates technical indicators for prediction markets.

    Maintains price history and computes:
    - EMAs (9, 21, 50)
    - RSI (14)
    - MACD (12, 26, 9)
    - Volume analysis
    - ATR for volatility
    """

    def __init__(self, config: IndicatorConfig):
        self.config = config
        # Price/volume history per token
        self._history: dict[str, deque[dict[str, Decimal]]] = {}
        self._max_history = 200  # Keep enough for longest indicator

    def update(
        self,
        token_id: str,
        price: Decimal,
        volume: Decimal,
        high: Decimal | None = None,
        low: Decimal | None = None,
    ) -> None:
        """Update price history with new data point."""
        if token_id not in self._history:
            self._history[token_id] = deque(maxlen=self._max_history)

        self._history[token_id].append({
            "price": price,
            "volume": volume,
            "high": high or price,
            "low": low or price,
        })

    def calculate(self, token_id: str) -> IndicatorResult:
        """Calculate all indicators for a token."""
        history = self._history.get(token_id)
        if not history or len(history) < 2:
            return IndicatorResult()

        prices = [h["price"] for h in history]
        volumes = [h["volume"] for h in history]
        highs = [h["high"] for h in history]
        lows = [h["low"] for h in history]

        result = IndicatorResult()

        # EMAs
        if len(prices) >= 9:
            result.ema_9 = self._ema(prices, 9)
        if len(prices) >= 21:
            result.ema_21 = self._ema(prices, 21)
        if len(prices) >= 50:
            result.ema_50 = self._ema(prices, 50)

        # EMA Trend
        result.ema_trend = self._determine_ema_trend(result)

        # RSI
        if len(prices) >= self.config.rsi_period + 1:
            result.rsi = self._rsi(prices, self.config.rsi_period)
            if result.rsi:
                if result.rsi < self.config.rsi_oversold:
                    result.rsi_signal = "oversold"
                elif result.rsi > self.config.rsi_overbought:
                    result.rsi_signal = "overbought"
                else:
                    result.rsi_signal = "neutral"

        # MACD
        if len(prices) >= self.config.macd_slow + self.config.macd_signal:
            macd_result = self._macd(
                prices,
                self.config.macd_fast,
                self.config.macd_slow,
                self.config.macd_signal,
            )
            result.macd_line = macd_result["macd"]
            result.macd_signal = macd_result["signal"]
            result.macd_histogram = macd_result["histogram"]
            result.macd_crossover = macd_result["crossover"]

        # Volume
        if len(volumes) >= self.config.volume_ma_period:
            result.volume_ma = self._sma(volumes, self.config.volume_ma_period)
            current_vol = volumes[-1]
            if result.volume_ma and result.volume_ma > 0:
                result.volume_ratio = current_vol / result.volume_ma
                if result.volume_ratio < Decimal("0.5"):
                    result.volume_trend = "low"
                elif result.volume_ratio > Decimal("2.0"):
                    result.volume_trend = "spike"
                elif result.volume_ratio > Decimal("1.5"):
                    result.volume_trend = "high"
                else:
                    result.volume_trend = "normal"

        # ATR
        if len(prices) >= self.config.atr_period:
            result.atr = self._atr(highs, lows, prices, self.config.atr_period)

        # Support/Resistance (simplified - recent highs/lows)
        if len(prices) >= 20:
            result.support_levels = self._find_support_levels(lows[-20:])
            result.resistance_levels = self._find_resistance_levels(highs[-20:])

        # Overall signal
        result.overall_signal, result.signal_strength = self._compute_overall_signal(result)

        return result

    def _ema(self, prices: list[Decimal], period: int) -> Decimal:
        """Calculate Exponential Moving Average."""
        if len(prices) < period:
            return Decimal("0")

        multiplier = Decimal("2") / (Decimal(str(period)) + Decimal("1"))
        ema = sum(prices[:period]) / Decimal(str(period))  # SMA for first value

        for price in prices[period:]:
            ema = (price - ema) * multiplier + ema

        return ema.quantize(Decimal("0.0001"))

    def _sma(self, values: list[Decimal], period: int) -> Decimal:
        """Calculate Simple Moving Average."""
        if len(values) < period:
            return Decimal("0")
        return (sum(values[-period:]) / Decimal(str(period))).quantize(Decimal("0.0001"))

    def _rsi(self, prices: list[Decimal], period: int) -> Decimal:
        """Calculate Relative Strength Index."""
        if len(prices) < period + 1:
            return Decimal("50")

        gains = []
        losses = []

        for i in range(1, len(prices)):
            change = prices[i] - prices[i - 1]
            if change > 0:
                gains.append(change)
                losses.append(Decimal("0"))
            else:
                gains.append(Decimal("0"))
                losses.append(abs(change))

        # Initial averages
        avg_gain = sum(gains[:period]) / Decimal(str(period))
        avg_loss = sum(losses[:period]) / Decimal(str(period))

        # Smoothed averages
        for i in range(period, len(gains)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / Decimal(str(period))
            avg_loss = (avg_loss * (period - 1) + losses[i]) / Decimal(str(period))

        if avg_loss == 0:
            return Decimal("100")

        rs = avg_gain / avg_loss
        rsi = Decimal("100") - (Decimal("100") / (Decimal("1") + rs))

        return rsi.quantize(Decimal("0.01"))

    def _macd(
        self,
        prices: list[Decimal],
        fast: int,
        slow: int,
        signal: int,
    ) -> dict[str, Any]:
        """Calculate MACD."""
        if len(prices) < slow + signal:
            return {
                "macd": None,
                "signal": None,
                "histogram": None,
                "crossover": "none",
            }

        fast_ema = self._ema(prices, fast)
        slow_ema = self._ema(prices, slow)
        macd_line = fast_ema - slow_ema

        # Calculate MACD line history for signal line
        macd_history = []
        for i in range(slow, len(prices)):
            subset = prices[:i + 1]
            f = self._ema(subset, fast)
            s = self._ema(subset, slow)
            macd_history.append(f - s)

        signal_line = self._ema(macd_history, signal) if len(macd_history) >= signal else Decimal("0")
        histogram = macd_line - signal_line

        # Detect crossover
        crossover = "none"
        if len(macd_history) >= 2:
            prev_hist = macd_history[-2] - signal_line
            if prev_hist < 0 and histogram > 0:
                crossover = "bullish"
            elif prev_hist > 0 and histogram < 0:
                crossover = "bearish"

        return {
            "macd": macd_line.quantize(Decimal("0.0001")),
            "signal": signal_line.quantize(Decimal("0.0001")),
            "histogram": histogram.quantize(Decimal("0.0001")),
            "crossover": crossover,
        }

    def _atr(
        self,
        highs: list[Decimal],
        lows: list[Decimal],
        closes: list[Decimal],
        period: int,
    ) -> Decimal:
        """Calculate Average True Range."""
        if len(closes) < period + 1:
            return Decimal("0")

        true_ranges = []
        for i in range(1, len(closes)):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i] - closes[i - 1]),
            )
            true_ranges.append(tr)

        return self._sma(true_ranges[-period:], period)

    def _determine_ema_trend(self, result: IndicatorResult) -> TrendDirection:
        """Determine trend from EMA alignment."""
        if not result.ema_9 or not result.ema_21:
            return TrendDirection.NEUTRAL

        if result.ema_50:
            if result.ema_9 > result.ema_21 > result.ema_50:
                return TrendDirection.STRONG_UP
            if result.ema_9 < result.ema_21 < result.ema_50:
                return TrendDirection.STRONG_DOWN

        if result.ema_9 > result.ema_21:
            return TrendDirection.UP
        if result.ema_9 < result.ema_21:
            return TrendDirection.DOWN

        return TrendDirection.NEUTRAL

    def _find_support_levels(self, lows: list[Decimal]) -> list[Decimal]:
        """Find recent support levels."""
        if len(lows) < 5:
            return []

        # Simple: find local minima
        supports = []
        for i in range(2, len(lows) - 2):
            if lows[i] <= min(lows[i - 2:i]) and lows[i] <= min(lows[i + 1:i + 3]):
                supports.append(lows[i])

        return sorted(set(supports))[:3]

    def _find_resistance_levels(self, highs: list[Decimal]) -> list[Decimal]:
        """Find recent resistance levels."""
        if len(highs) < 5:
            return []

        # Simple: find local maxima
        resistances = []
        for i in range(2, len(highs) - 2):
            if highs[i] >= max(highs[i - 2:i]) and highs[i] >= max(highs[i + 1:i + 3]):
                resistances.append(highs[i])

        return sorted(set(resistances), reverse=True)[:3]

    def _compute_overall_signal(
        self,
        result: IndicatorResult,
    ) -> tuple[str, Decimal]:
        """Compute overall signal from all indicators."""
        score = Decimal("0")
        signals = 0

        # EMA trend
        if result.ema_trend == TrendDirection.STRONG_UP:
            score += Decimal("1.0")
            signals += 1
        elif result.ema_trend == TrendDirection.UP:
            score += Decimal("0.5")
            signals += 1
        elif result.ema_trend == TrendDirection.DOWN:
            score -= Decimal("0.5")
            signals += 1
        elif result.ema_trend == TrendDirection.STRONG_DOWN:
            score -= Decimal("1.0")
            signals += 1

        # RSI
        if result.rsi_signal == "oversold":
            score += Decimal("0.5")  # Potential bounce
            signals += 1
        elif result.rsi_signal == "overbought":
            score -= Decimal("0.5")  # Potential pullback
            signals += 1

        # MACD
        if result.macd_crossover == "bullish":
            score += Decimal("0.75")
            signals += 1
        elif result.macd_crossover == "bearish":
            score -= Decimal("0.75")
            signals += 1
        elif result.macd_histogram and result.macd_histogram > 0:
            score += Decimal("0.25")
            signals += 1
        elif result.macd_histogram and result.macd_histogram < 0:
            score -= Decimal("0.25")
            signals += 1

        # Normalize
        if signals > 0:
            strength = (score / Decimal(str(signals))).quantize(Decimal("0.01"))
        else:
            strength = Decimal("0")

        # Determine signal
        if strength > Decimal("0.5"):
            signal = "strong_buy"
        elif strength > Decimal("0.2"):
            signal = "buy"
        elif strength < Decimal("-0.5"):
            signal = "strong_sell"
        elif strength < Decimal("-0.2"):
            signal = "sell"
        else:
            signal = "neutral"

        return signal, strength

    def clear_history(self, token_id: str | None = None) -> None:
        """Clear price history."""
        if token_id:
            self._history.pop(token_id, None)
        else:
            self._history.clear()
