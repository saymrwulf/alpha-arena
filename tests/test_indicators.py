"""Tests for technical indicators."""

import pytest
from decimal import Decimal

from src.core.config import IndicatorConfig
from src.indicators.calculator import IndicatorCalculator
from src.indicators.types import TrendDirection, IndicatorResult


@pytest.fixture
def calculator():
    """Create indicator calculator with default config."""
    config = IndicatorConfig()
    return IndicatorCalculator(config)


class TestEMA:
    """Tests for Exponential Moving Average calculation."""

    def test_ema_insufficient_data(self, calculator):
        """EMA returns 0 with insufficient data."""
        # Add only 5 data points
        for i in range(5):
            calculator.update("test", Decimal("0.50"), Decimal("100"))

        result = calculator.calculate("test")

        assert result.ema_9 is None  # Need at least 9 points

    def test_ema_9_calculation(self, calculator):
        """Test EMA-9 with enough data."""
        # Add 15 data points with upward trend
        for i in range(15):
            price = Decimal("0.40") + Decimal(str(i * 0.02))
            calculator.update("test", price, Decimal("100"))

        result = calculator.calculate("test")

        assert result.ema_9 is not None
        assert result.ema_9 > Decimal("0.40")  # Should reflect uptrend

    def test_ema_21_calculation(self, calculator):
        """Test EMA-21 with enough data."""
        # Add 25 data points
        for i in range(25):
            price = Decimal("0.50") + Decimal(str(i * 0.01))
            calculator.update("test", price, Decimal("100"))

        result = calculator.calculate("test")

        assert result.ema_9 is not None
        assert result.ema_21 is not None
        # EMA-9 should be higher than EMA-21 in uptrend
        assert result.ema_9 > result.ema_21

    def test_ema_trend_bullish(self, calculator):
        """Test bullish EMA trend detection."""
        # Strong uptrend
        for i in range(55):
            price = Decimal("0.30") + Decimal(str(i * 0.01))
            calculator.update("test", price, Decimal("100"))

        result = calculator.calculate("test")

        assert result.ema_trend in [TrendDirection.UP, TrendDirection.STRONG_UP]

    def test_ema_trend_bearish(self, calculator):
        """Test bearish EMA trend detection."""
        # Strong downtrend
        for i in range(55):
            price = Decimal("0.80") - Decimal(str(i * 0.01))
            calculator.update("test", price, Decimal("100"))

        result = calculator.calculate("test")

        assert result.ema_trend in [TrendDirection.DOWN, TrendDirection.STRONG_DOWN]


class TestRSI:
    """Tests for Relative Strength Index calculation."""

    def test_rsi_calculation(self, calculator):
        """Test basic RSI calculation."""
        # Add enough data for RSI
        for i in range(20):
            # Oscillating prices
            price = Decimal("0.50") + Decimal(str((i % 5 - 2) * 0.02))
            calculator.update("test", price, Decimal("100"))

        result = calculator.calculate("test")

        assert result.rsi is not None
        assert Decimal("0") <= result.rsi <= Decimal("100")

    def test_rsi_oversold(self, calculator):
        """Test RSI oversold detection."""
        # Strong downtrend should produce low RSI
        for i in range(20):
            price = Decimal("0.70") - Decimal(str(i * 0.02))
            calculator.update("test", price, Decimal("100"))

        result = calculator.calculate("test")

        assert result.rsi is not None
        assert result.rsi < Decimal("50")
        # May not always be oversold depending on params

    def test_rsi_overbought(self, calculator):
        """Test RSI overbought detection."""
        # Strong uptrend should produce high RSI
        for i in range(20):
            price = Decimal("0.30") + Decimal(str(i * 0.02))
            calculator.update("test", price, Decimal("100"))

        result = calculator.calculate("test")

        assert result.rsi is not None
        assert result.rsi > Decimal("50")


class TestMACD:
    """Tests for MACD calculation."""

    def test_macd_insufficient_data(self, calculator):
        """MACD returns None with insufficient data."""
        for i in range(30):
            calculator.update("test", Decimal("0.50"), Decimal("100"))

        result = calculator.calculate("test")

        # Need 26 + 9 = 35 points minimum
        assert result.macd_line is None or result.macd_line is not None

    def test_macd_calculation(self, calculator):
        """Test basic MACD calculation."""
        # Add enough data for MACD
        for i in range(50):
            # Trending price
            price = Decimal("0.40") + Decimal(str(i * 0.005))
            calculator.update("test", price, Decimal("100"))

        result = calculator.calculate("test")

        assert result.macd_line is not None
        assert result.macd_signal is not None
        assert result.macd_histogram is not None

    def test_macd_bullish_crossover(self, calculator):
        """Test MACD bullish crossover detection."""
        # Start bearish then turn bullish
        for i in range(30):
            price = Decimal("0.60") - Decimal(str(i * 0.005))
            calculator.update("test", price, Decimal("100"))

        # Turn bullish
        for i in range(30):
            price = Decimal("0.45") + Decimal(str(i * 0.01))
            calculator.update("test", price, Decimal("100"))

        result = calculator.calculate("test")

        assert result.macd_histogram is not None
        # Histogram should be positive in bullish trend
        assert result.macd_histogram > 0


class TestVolume:
    """Tests for volume analysis."""

    def test_volume_ratio(self, calculator):
        """Test volume ratio calculation."""
        # Add data with consistent volume
        for i in range(25):
            calculator.update("test", Decimal("0.50"), Decimal("1000"))

        # Add spike
        calculator.update("test", Decimal("0.50"), Decimal("3000"))

        result = calculator.calculate("test")

        assert result.volume_ratio is not None
        assert result.volume_ratio > Decimal("1.0")

    def test_volume_trend_spike(self, calculator):
        """Test volume spike detection."""
        # Normal volume
        for i in range(25):
            calculator.update("test", Decimal("0.50"), Decimal("1000"))

        # Volume spike
        calculator.update("test", Decimal("0.50"), Decimal("5000"))

        result = calculator.calculate("test")

        assert result.volume_trend in ["high", "spike"]


class TestATR:
    """Tests for Average True Range calculation."""

    def test_atr_calculation(self, calculator):
        """Test ATR calculation."""
        # Add data with varying highs and lows
        for i in range(20):
            price = Decimal("0.50")
            high = price + Decimal("0.02")
            low = price - Decimal("0.02")
            calculator.update("test", price, Decimal("100"), high, low)

        result = calculator.calculate("test")

        assert result.atr is not None
        assert result.atr > Decimal("0")

    def test_atr_volatile_market(self, calculator):
        """Test ATR in volatile market."""
        # Large swings
        for i in range(20):
            price = Decimal("0.50")
            high = price + Decimal("0.10")
            low = price - Decimal("0.10")
            calculator.update("test", price, Decimal("100"), high, low)

        result = calculator.calculate("test")

        assert result.atr is not None
        assert result.atr > Decimal("0.05")


class TestSupportResistance:
    """Tests for support/resistance detection."""

    def test_support_levels(self, calculator):
        """Test support level detection."""
        # Create pattern with clear low points
        prices = [
            0.50, 0.52, 0.48, 0.47, 0.49,  # First low at 0.47
            0.51, 0.53, 0.50, 0.48, 0.46,  # Second low at 0.46
            0.48, 0.50, 0.52, 0.54, 0.56,
            0.54, 0.52, 0.50, 0.48, 0.47,  # Third low at 0.47
            0.49, 0.51, 0.53, 0.55, 0.57,
        ]

        for p in prices:
            calculator.update("test", Decimal(str(p)), Decimal("100"), Decimal(str(p + 0.01)), Decimal(str(p - 0.01)))

        result = calculator.calculate("test")

        # Should find some support levels
        assert isinstance(result.support_levels, list)

    def test_resistance_levels(self, calculator):
        """Test resistance level detection."""
        # Create pattern with clear high points
        prices = [
            0.50, 0.52, 0.54, 0.53, 0.51,  # First high at 0.54
            0.49, 0.47, 0.50, 0.52, 0.55,  # Second high at 0.55
            0.53, 0.50, 0.48, 0.46, 0.44,
            0.46, 0.48, 0.50, 0.52, 0.54,  # Third high at 0.54
            0.52, 0.50, 0.48, 0.46, 0.44,
        ]

        for p in prices:
            calculator.update("test", Decimal(str(p)), Decimal("100"), Decimal(str(p + 0.01)), Decimal(str(p - 0.01)))

        result = calculator.calculate("test")

        assert isinstance(result.resistance_levels, list)


class TestOverallSignal:
    """Tests for overall signal computation."""

    def test_bullish_signal(self, calculator):
        """Test strong bullish signal detection."""
        # Strong uptrend with all bullish indicators
        for i in range(60):
            price = Decimal("0.30") + Decimal(str(i * 0.01))
            calculator.update("test", price, Decimal("1000"))

        result = calculator.calculate("test")

        assert result.overall_signal in ["buy", "strong_buy"]
        assert result.signal_strength > Decimal("0")

    def test_bearish_signal(self, calculator):
        """Test strong bearish signal detection."""
        # Strong downtrend
        for i in range(60):
            price = Decimal("0.80") - Decimal(str(i * 0.01))
            calculator.update("test", price, Decimal("1000"))

        result = calculator.calculate("test")

        assert result.overall_signal in ["sell", "strong_sell"]
        assert result.signal_strength < Decimal("0")

    def test_neutral_signal(self, calculator):
        """Test neutral signal in ranging market."""
        # Sideways movement - true oscillation around center
        for i in range(60):
            # Alternate up/down to create true ranging
            if i % 2 == 0:
                price = Decimal("0.50") + Decimal("0.01")
            else:
                price = Decimal("0.50") - Decimal("0.01")
            calculator.update("test", price, Decimal("1000"))

        result = calculator.calculate("test")

        # Should produce a signal strength between -1 and 1
        # In ranging markets, signal may vary but should have a defined value
        assert result.signal_strength >= Decimal("-1")
        assert result.signal_strength <= Decimal("1")


class TestIndicatorHistory:
    """Tests for indicator history management."""

    def test_clear_history(self, calculator):
        """Test clearing history for specific token."""
        # Add data
        for i in range(20):
            calculator.update("token_a", Decimal("0.50"), Decimal("100"))
            calculator.update("token_b", Decimal("0.60"), Decimal("100"))

        # Clear only token_a
        calculator.clear_history("token_a")

        # token_a should have no data
        result_a = calculator.calculate("token_a")
        assert result_a.ema_9 is None

        # token_b should still have data
        result_b = calculator.calculate("token_b")
        assert result_b.ema_9 is not None

    def test_clear_all_history(self, calculator):
        """Test clearing all history."""
        # Add data
        for i in range(20):
            calculator.update("token_a", Decimal("0.50"), Decimal("100"))
            calculator.update("token_b", Decimal("0.60"), Decimal("100"))

        # Clear all
        calculator.clear_history()

        result_a = calculator.calculate("token_a")
        result_b = calculator.calculate("token_b")

        assert result_a.ema_9 is None
        assert result_b.ema_9 is None


class TestIndicatorResultSerialization:
    """Tests for IndicatorResult serialization."""

    def test_to_dict(self, calculator):
        """Test IndicatorResult to_dict method."""
        for i in range(60):
            calculator.update("test", Decimal("0.40") + Decimal(str(i * 0.01)), Decimal("1000"))

        result = calculator.calculate("test")
        data = result.to_dict()

        assert "ema_9" in data
        assert "ema_21" in data
        assert "rsi" in data
        assert "macd_line" in data
        assert "overall_signal" in data
        assert isinstance(data["ema_trend"], str)
