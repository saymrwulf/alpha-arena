"""Polymarket CLOB API broker implementation."""

import asyncio
from datetime import datetime
from decimal import Decimal
from typing import Any

import httpx
from py_clob_client.client import ClobClient
from py_clob_client.clob_types import OrderArgs, OrderType
from py_clob_client.order_builder.constants import BUY, SELL

from .base import Broker, Fill, Order, OrderSide, OrderStatus, Position


class PolymarketBroker(Broker):
    """Broker implementation for Polymarket CLOB API."""

    CLOB_HOST = "https://clob.polymarket.com"
    GAMMA_HOST = "https://gamma-api.polymarket.com"

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

        # Run sync operation in thread pool
        loop = asyncio.get_event_loop()
        creds = await loop.run_in_executor(
            None, self._client.derive_api_key
        )

        # Set credentials on client
        self._client.set_api_creds(creds)

    async def disconnect(self) -> None:
        """Disconnect from the API."""
        if self._http:
            await self._http.aclose()
            self._http = None
        self._client = None

    async def get_balance(self) -> Decimal:
        """Get USDC balance on Polygon."""
        if self._client is None or self._http is None:
            raise RuntimeError("Broker not connected")

        # Get balance from CLOB API
        loop = asyncio.get_event_loop()
        balance_info = await loop.run_in_executor(
            None, self._client.get_balance_allowance
        )

        # Balance is returned in USDC (6 decimals)
        balance = Decimal(str(balance_info.get("balance", "0")))
        return balance / Decimal("1000000")  # Convert from wei to USDC

    async def get_positions(self) -> list[Position]:
        """Get all open positions."""
        if self._client is None:
            raise RuntimeError("Broker not connected")

        loop = asyncio.get_event_loop()
        positions_data = await loop.run_in_executor(
            None, self._client.get_positions
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

        loop = asyncio.get_event_loop()
        signed_order = await loop.run_in_executor(
            None,
            lambda: self._client.create_order(order_args)
        )

        response = await loop.run_in_executor(
            None,
            lambda: self._client.post_order(signed_order, OrderType.GTC)
        )

        order_id = response.get("orderID", "")
        status = self._map_order_status(response.get("status", ""))

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

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: self._client.cancel(order_id)
        )

        return result.get("success", False)

    async def get_order(self, order_id: str) -> Order | None:
        """Get order status."""
        if self._client is None:
            raise RuntimeError("Broker not connected")

        loop = asyncio.get_event_loop()
        order_data = await loop.run_in_executor(
            None,
            lambda: self._client.get_order(order_id)
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

        loop = asyncio.get_event_loop()
        orders_data = await loop.run_in_executor(
            None,
            self._client.get_orders
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

        loop = asyncio.get_event_loop()
        trades = await loop.run_in_executor(
            None,
            self._client.get_trades
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
        """Fetch available markets from Gamma API."""
        if self._http is None:
            raise RuntimeError("Broker not connected")

        params = {"active": str(active_only).lower()}
        resp = await self._http.get(f"{self.GAMMA_HOST}/markets", params=params)
        resp.raise_for_status()

        return resp.json()

    async def get_market(self, condition_id: str) -> dict[str, Any] | None:
        """Get market details by condition ID."""
        if self._http is None:
            raise RuntimeError("Broker not connected")

        resp = await self._http.get(f"{self.GAMMA_HOST}/markets/{condition_id}")
        if resp.status_code == 404:
            return None
        resp.raise_for_status()

        return resp.json()

    async def get_orderbook(self, token_id: str) -> dict[str, Any]:
        """Get orderbook for a token."""
        if self._client is None:
            raise RuntimeError("Broker not connected")

        loop = asyncio.get_event_loop()
        book = await loop.run_in_executor(
            None,
            lambda: self._client.get_order_book(token_id)
        )

        return book

    async def get_midpoint(self, token_id: str) -> Decimal:
        """Get midpoint price for a token."""
        if self._client is None:
            raise RuntimeError("Broker not connected")

        loop = asyncio.get_event_loop()
        midpoint = await loop.run_in_executor(
            None,
            lambda: self._client.get_midpoint(token_id)
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
