"""Polymarket CLOB API broker implementation."""

import asyncio
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Any, Callable, TypeVar

import httpx
from py_clob_client.client import ClobClient
from py_clob_client.clob_types import OrderArgs, OrderType
from py_clob_client.order_builder.constants import BUY, SELL

from .base import Broker, Fill, Order, OrderSide, OrderStatus, Position

logger = logging.getLogger(__name__)
T = TypeVar("T")


@dataclass
class CacheEntry:
    """Cached data with expiration."""

    data: Any
    expires_at: float


class BrokerCache:
    """Simple TTL cache for broker data."""

    def __init__(self):
        self._cache: dict[str, CacheEntry] = {}

    def get(self, key: str) -> Any | None:
        """Get cached value if not expired."""
        entry = self._cache.get(key)
        if entry is None:
            return None
        if time.time() > entry.expires_at:
            del self._cache[key]
            return None
        return entry.data

    def set(self, key: str, data: Any, ttl_seconds: int) -> None:
        """Cache data with TTL."""
        self._cache[key] = CacheEntry(
            data=data,
            expires_at=time.time() + ttl_seconds,
        )

    def invalidate(self, key: str) -> None:
        """Remove a cached entry."""
        self._cache.pop(key, None)

    def clear(self) -> None:
        """Clear all cached data."""
        self._cache.clear()


class PolymarketBroker(Broker):
    """Broker implementation for Polymarket CLOB API."""

    CLOB_HOST = "https://clob.polymarket.com"
    GAMMA_HOST = "https://gamma-api.polymarket.com"

    # Cache TTLs (seconds)
    MARKETS_CACHE_TTL = 300  # 5 minutes - market list rarely changes
    MARKET_CACHE_TTL = 120  # 2 minutes - individual market details
    ORDERBOOK_CACHE_TTL = 30  # 30 seconds - orderbook updates frequently

    # Timeout for CLOB client operations (seconds)
    CLOB_TIMEOUT = 30.0

    def __init__(
        self,
        private_key: str,
        api_key: str | None = None,
        api_secret: str | None = None,
        api_passphrase: str | None = None,
        chain_id: int = 137,  # Polygon mainnet
    ):
        self.private_key = private_key
        self.api_key = api_key
        self.api_secret = api_secret
        self.api_passphrase = api_passphrase
        self.chain_id = chain_id
        self._client: ClobClient | None = None
        self._http: httpx.AsyncClient | None = None
        self._cache = BrokerCache()

    async def connect(self) -> None:
        """Connect to Polymarket CLOB API."""
        creds = None
        if self.api_key and self.api_secret and self.api_passphrase:
            creds = {
                "apiKey": self.api_key,
                "secret": self.api_secret,
                "passphrase": self.api_passphrase,
            }

        self._client = ClobClient(
            host=self.CLOB_HOST,
            chain_id=self.chain_id,
            key=self.private_key,
            creds=creds,
        )

        self._http = httpx.AsyncClient(timeout=30.0)

        # Derive API credentials if not provided
        if not creds:
            await self._derive_api_credentials()

    async def _derive_api_credentials(self) -> None:
        """Derive API credentials from private key."""
        if self._client is None:
            raise RuntimeError("Client not connected")

        logger.info("Deriving API credentials from private key...")
        creds = await self._run_sync_with_timeout(
            self._client.derive_api_key,
            "derive_api_key",
            timeout=60.0,  # Key derivation can be slow
        )

        # Set credentials on client
        self._client.set_api_creds(creds)
        logger.info("API credentials derived successfully")

    async def disconnect(self) -> None:
        """Disconnect from the API."""
        if self._http:
            await self._http.aclose()
            self._http = None
        self._client = None

    async def _run_sync_with_timeout(
        self,
        func: Callable[[], T],
        operation_name: str,
        timeout: float | None = None,
    ) -> T:
        """
        Run a synchronous function in executor with timeout protection.

        Args:
            func: Synchronous function to run
            operation_name: Name for logging
            timeout: Timeout in seconds (default: CLOB_TIMEOUT)

        Raises:
            asyncio.TimeoutError: If operation exceeds timeout
        """
        timeout = timeout or self.CLOB_TIMEOUT
        loop = asyncio.get_event_loop()

        try:
            result = await asyncio.wait_for(
                loop.run_in_executor(None, func),
                timeout=timeout,
            )
            return result
        except asyncio.TimeoutError:
            logger.error(f"CLOB operation '{operation_name}' timed out after {timeout}s")
            raise asyncio.TimeoutError(f"CLOB {operation_name} timed out")
        except Exception as e:
            logger.error(f"CLOB operation '{operation_name}' failed: {e}")
            raise

    async def get_balance(self) -> Decimal:
        """Get USDC balance on Polygon."""
        if self._client is None or self._http is None:
            raise RuntimeError("Broker not connected")

        balance_info = await self._run_sync_with_timeout(
            self._client.get_balance_allowance,
            "get_balance",
        )

        # Balance is returned in USDC (6 decimals)
        balance = Decimal(str(balance_info.get("balance", "0")))
        return balance / Decimal("1000000")  # Convert from wei to USDC

    async def get_positions(self) -> list[Position]:
        """Get all open positions."""
        if self._client is None:
            raise RuntimeError("Broker not connected")

        positions_data = await self._run_sync_with_timeout(
            self._client.get_positions,
            "get_positions",
        )

        positions = []
        for pos in positions_data:
            size = Decimal(str(pos.get("size", "0")))
            if size > 0:
                avg_price = Decimal(str(pos.get("avgPrice", "0")))
                current_price = Decimal(str(pos.get("curPrice", avg_price)))
                unrealized = size * (current_price - avg_price)

                positions.append(Position(
                    market_id=pos.get("conditionId", ""),
                    token_id=pos.get("tokenId", ""),
                    outcome=pos.get("outcome", ""),
                    size=size,
                    avg_entry_price=avg_price,
                    current_price=current_price,
                    unrealized_pnl=unrealized,
                    realized_pnl=Decimal(str(pos.get("realizedPnl", "0"))),
                ))

        return positions

    async def get_position(self, market_id: str, token_id: str) -> Position | None:
        """Get a specific position."""
        positions = await self.get_positions()
        for pos in positions:
            if pos.market_id == market_id and pos.token_id == token_id:
                return pos
        return None

    async def place_order(
        self,
        market_id: str,
        token_id: str,
        side: OrderSide,
        size: Decimal,
        price: Decimal,
    ) -> Order:
        """Place a limit order on Polymarket."""
        if self._client is None:
            raise RuntimeError("Broker not connected")

        clob_side = BUY if side == OrderSide.BUY else SELL

        order_args = OrderArgs(
            token_id=token_id,
            price=float(price),
            size=float(size),
            side=clob_side,
        )

        logger.info(f"Placing order: {token_id} {side.value} {size} @ {price}")

        signed_order = await self._run_sync_with_timeout(
            lambda: self._client.create_order(order_args),
            "create_order",
        )

        response = await self._run_sync_with_timeout(
            lambda: self._client.post_order(signed_order, OrderType.GTC),
            "post_order",
        )

        order_id = response.get("orderID", "")
        status = self._map_order_status(response.get("status", ""))

        logger.info(f"Order placed: {order_id} status={status.value}")

        return Order(
            id=order_id,
            market_id=market_id,
            token_id=token_id,
            side=side,
            size=size,
            price=price,
            status=status,
            metadata={"response": response},
        )

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order."""
        if self._client is None:
            raise RuntimeError("Broker not connected")

        logger.info(f"Cancelling order: {order_id}")

        result = await self._run_sync_with_timeout(
            lambda: self._client.cancel(order_id),
            f"cancel_order_{order_id}",
        )

        success = result.get("success", False)
        logger.info(f"Cancel order {order_id}: {'success' if success else 'failed'}")
        return success

    async def get_order(self, order_id: str) -> Order | None:
        """Get order status."""
        if self._client is None:
            raise RuntimeError("Broker not connected")

        order_data = await self._run_sync_with_timeout(
            lambda: self._client.get_order(order_id),
            f"get_order_{order_id}",
        )

        if not order_data:
            return None

        side = OrderSide.BUY if order_data.get("side") == "BUY" else OrderSide.SELL
        status = self._map_order_status(order_data.get("status", ""))

        return Order(
            id=order_id,
            market_id=order_data.get("conditionId", ""),
            token_id=order_data.get("tokenId", ""),
            side=side,
            size=Decimal(str(order_data.get("originalSize", "0"))),
            price=Decimal(str(order_data.get("price", "0"))),
            status=status,
            filled_size=Decimal(str(order_data.get("sizeFilled", "0"))),
            avg_fill_price=Decimal(str(order_data.get("avgFillPrice", "0"))) if order_data.get("avgFillPrice") else None,
        )

    async def get_open_orders(self) -> list[Order]:
        """Get all open orders."""
        if self._client is None:
            raise RuntimeError("Broker not connected")

        orders_data = await self._run_sync_with_timeout(
            self._client.get_orders,
            "get_open_orders",
        )

        orders = []
        for od in orders_data:
            side = OrderSide.BUY if od.get("side") == "BUY" else OrderSide.SELL
            status = self._map_order_status(od.get("status", ""))

            orders.append(Order(
                id=od.get("id", ""),
                market_id=od.get("conditionId", ""),
                token_id=od.get("tokenId", ""),
                side=side,
                size=Decimal(str(od.get("originalSize", "0"))),
                price=Decimal(str(od.get("price", "0"))),
                status=status,
                filled_size=Decimal(str(od.get("sizeFilled", "0"))),
            ))

        return orders

    async def get_fills(self, since: datetime | None = None) -> list[Fill]:
        """Get recent trade fills."""
        if self._client is None:
            raise RuntimeError("Broker not connected")

        trades = await self._run_sync_with_timeout(
            self._client.get_trades,
            "get_fills",
        )

        fills = []
        for trade in trades:
            trade_time = datetime.fromisoformat(
                trade.get("timestamp", "").replace("Z", "+00:00")
            )
            if since and trade_time < since:
                continue

            side = OrderSide.BUY if trade.get("side") == "BUY" else OrderSide.SELL

            fills.append(Fill(
                order_id=trade.get("orderId", ""),
                market_id=trade.get("conditionId", ""),
                token_id=trade.get("tokenId", ""),
                side=side,
                size=Decimal(str(trade.get("size", "0"))),
                price=Decimal(str(trade.get("price", "0"))),
                fee=Decimal(str(trade.get("fee", "0"))),
                timestamp=trade_time,
            ))

        return fills

    async def get_markets(self, active_only: bool = True) -> list[dict[str, Any]]:
        """Fetch available markets from Gamma API (cached)."""
        if self._http is None:
            raise RuntimeError("Broker not connected")

        cache_key = f"markets:{active_only}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        params = {"active": str(active_only).lower()}
        resp = await self._http.get(f"{self.GAMMA_HOST}/markets", params=params)
        resp.raise_for_status()

        markets = resp.json()
        self._cache.set(cache_key, markets, self.MARKETS_CACHE_TTL)
        return markets

    async def get_market(self, condition_id: str) -> dict[str, Any] | None:
        """Get market details by condition ID (cached)."""
        if self._http is None:
            raise RuntimeError("Broker not connected")

        cache_key = f"market:{condition_id}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        resp = await self._http.get(f"{self.GAMMA_HOST}/markets/{condition_id}")
        if resp.status_code == 404:
            return None
        resp.raise_for_status()

        market = resp.json()
        self._cache.set(cache_key, market, self.MARKET_CACHE_TTL)
        return market

    async def get_orderbook(self, token_id: str) -> dict[str, Any]:
        """Get orderbook for a token (cached)."""
        if self._client is None:
            raise RuntimeError("Broker not connected")

        cache_key = f"orderbook:{token_id}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        book = await self._run_sync_with_timeout(
            lambda: self._client.get_order_book(token_id),
            f"get_orderbook_{token_id[:8]}",
        )

        self._cache.set(cache_key, book, self.ORDERBOOK_CACHE_TTL)
        return book

    async def get_midpoint(self, token_id: str) -> Decimal:
        """Get midpoint price for a token."""
        if self._client is None:
            raise RuntimeError("Broker not connected")

        midpoint = await self._run_sync_with_timeout(
            lambda: self._client.get_midpoint(token_id),
            f"get_midpoint_{token_id[:8]}",
        )

        return Decimal(str(midpoint))

    def _map_order_status(self, status: str) -> OrderStatus:
        """Map Polymarket status to internal OrderStatus."""
        status_map = {
            "live": OrderStatus.OPEN,
            "matched": OrderStatus.FILLED,
            "cancelled": OrderStatus.CANCELLED,
            "partial-fill": OrderStatus.PARTIALLY_FILLED,
        }
        return status_map.get(status.lower(), OrderStatus.PENDING)
