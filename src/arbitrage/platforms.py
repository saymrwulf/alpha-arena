"""Platform interfaces for arbitrage detection."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

import httpx


@dataclass
class PlatformMarket:
    """Market data from a platform."""
    platform: str
    market_id: str
    question: str
    yes_price: Decimal
    no_price: Decimal
    yes_ask: Decimal  # Best ask for YES
    no_ask: Decimal  # Best ask for NO
    liquidity: Decimal
    fee_rate: Decimal  # Platform fee as decimal


class Platform(ABC):
    """Abstract platform interface."""

    name: str = "base"
    fee_rate: Decimal = Decimal("0")

    @abstractmethod
    async def connect(self) -> None:
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        pass

    @abstractmethod
    async def get_markets(self) -> list[PlatformMarket]:
        pass

    @abstractmethod
    async def get_market(self, market_id: str) -> PlatformMarket | None:
        pass


class PolymarketPlatform(Platform):
    """Polymarket platform interface."""

    name = "polymarket"
    fee_rate = Decimal("0.00")  # Polymarket has no trading fees

    def __init__(self):
        self._http: httpx.AsyncClient | None = None

    async def connect(self) -> None:
        self._http = httpx.AsyncClient(timeout=30.0)

    async def disconnect(self) -> None:
        if self._http:
            await self._http.aclose()
            self._http = None

    async def get_markets(self) -> list[PlatformMarket]:
        if not self._http:
            return []

        resp = await self._http.get(
            "https://gamma-api.polymarket.com/markets",
            params={"active": "true", "closed": "false", "limit": 100},
        )
        resp.raise_for_status()
        data = resp.json()

        markets = []
        for m in data:
            try:
                markets.append(self._parse_market(m))
            except (KeyError, ValueError):
                continue

        return markets

    async def get_market(self, market_id: str) -> PlatformMarket | None:
        if not self._http:
            return None

        resp = await self._http.get(
            f"https://gamma-api.polymarket.com/markets/{market_id}"
        )
        if resp.status_code == 404:
            return None
        resp.raise_for_status()

        return self._parse_market(resp.json())

    def _parse_market(self, data: dict[str, Any]) -> PlatformMarket:
        import json

        outcome_prices = data.get("outcomePrices", [])
        if isinstance(outcome_prices, str):
            outcome_prices = json.loads(outcome_prices)

        yes_price = Decimal(str(outcome_prices[0])) if outcome_prices else Decimal("0.5")
        no_price = Decimal(str(outcome_prices[1])) if len(outcome_prices) > 1 else Decimal("1") - yes_price

        return PlatformMarket(
            platform=self.name,
            market_id=data.get("conditionId", ""),
            question=data.get("question", ""),
            yes_price=yes_price,
            no_price=no_price,
            yes_ask=yes_price + Decimal("0.01"),  # Approximate spread
            no_ask=no_price + Decimal("0.01"),
            liquidity=Decimal(str(data.get("liquidity", 0))),
            fee_rate=self.fee_rate,
        )


class KalshiPlatform(Platform):
    """Kalshi platform interface."""

    name = "kalshi"
    fee_rate = Decimal("0.01")  # 1% fee
    BASE_URL = "https://trading-api.kalshi.com/trade-api/v2"

    def __init__(self, api_key: str = "", api_secret: str = ""):
        self.api_key = api_key
        self.api_secret = api_secret
        self._http: httpx.AsyncClient | None = None

    async def connect(self) -> None:
        headers = {}
        if self.api_key:
            # Kalshi uses different auth - simplified here
            headers["Authorization"] = f"Bearer {self.api_key}"

        self._http = httpx.AsyncClient(
            timeout=30.0,
            headers=headers,
        )

    async def disconnect(self) -> None:
        if self._http:
            await self._http.aclose()
            self._http = None

    async def get_markets(self) -> list[PlatformMarket]:
        if not self._http:
            return []

        try:
            resp = await self._http.get(
                f"{self.BASE_URL}/markets",
                params={"status": "open", "limit": 100},
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            return []

        markets = []
        for m in data.get("markets", []):
            try:
                markets.append(self._parse_market(m))
            except (KeyError, ValueError):
                continue

        return markets

    async def get_market(self, market_id: str) -> PlatformMarket | None:
        if not self._http:
            return None

        try:
            resp = await self._http.get(f"{self.BASE_URL}/markets/{market_id}")
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            return self._parse_market(resp.json().get("market", {}))
        except Exception:
            return None

    def _parse_market(self, data: dict[str, Any]) -> PlatformMarket:
        yes_bid = Decimal(str(data.get("yes_bid", 50))) / 100
        yes_ask = Decimal(str(data.get("yes_ask", 50))) / 100
        no_bid = Decimal(str(data.get("no_bid", 50))) / 100
        no_ask = Decimal(str(data.get("no_ask", 50))) / 100

        return PlatformMarket(
            platform=self.name,
            market_id=data.get("ticker", ""),
            question=data.get("title", ""),
            yes_price=(yes_bid + yes_ask) / 2,
            no_price=(no_bid + no_ask) / 2,
            yes_ask=yes_ask,
            no_ask=no_ask,
            liquidity=Decimal(str(data.get("volume", 0))),
            fee_rate=self.fee_rate,
        )
