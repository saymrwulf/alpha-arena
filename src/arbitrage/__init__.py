"""Cross-platform arbitrage detection."""

from .detector import ArbitrageDetector, ArbitrageOpportunity
from .platforms import Platform, PolymarketPlatform, KalshiPlatform

__all__ = [
    "ArbitrageDetector",
    "ArbitrageOpportunity",
    "Platform",
    "PolymarketPlatform",
    "KalshiPlatform",
]
