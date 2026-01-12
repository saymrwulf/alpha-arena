"""Wallet analysis - fetch and parse Polymarket wallet history."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Optional
import asyncio
import httpx


class TransactionType(Enum):
    """Types of wallet transactions."""
    BUY = "buy"
    SELL = "sell"
    DEPOSIT = "deposit"
    WITHDRAWAL = "withdrawal"
    CLAIM = "claim"  # Market resolution claim
    TRANSFER = "transfer"


@dataclass
class WalletTransaction:
    """A single wallet transaction."""
    tx_hash: str
    timestamp: datetime
    tx_type: TransactionType
    market_id: Optional[str]
    market_question: Optional[str]
    outcome: Optional[str]  # YES or NO
    amount: Decimal  # USDC amount
    shares: Optional[Decimal]  # Number of shares
    price: Optional[Decimal]  # Price per share
    fee: Decimal = Decimal("0")
    block_number: int = 0

    @property
    def net_amount(self) -> Decimal:
        """Amount after fees."""
        if self.tx_type == TransactionType.BUY:
            return -(self.amount + self.fee)
        elif self.tx_type == TransactionType.SELL:
            return self.amount - self.fee
        elif self.tx_type == TransactionType.CLAIM:
            return self.amount
        return self.amount


@dataclass
class WalletPosition:
    """Current or historical position in a market."""
    market_id: str
    market_question: str
    outcome: str
    shares: Decimal
    avg_entry_price: Decimal
    current_price: Optional[Decimal]
    realized_pnl: Decimal = Decimal("0")
    unrealized_pnl: Decimal = Decimal("0")
    total_invested: Decimal = Decimal("0")
    total_returned: Decimal = Decimal("0")
    entry_time: Optional[datetime] = None
    exit_time: Optional[datetime] = None
    is_closed: bool = False
    is_resolved: bool = False

    @property
    def total_pnl(self) -> Decimal:
        return self.realized_pnl + self.unrealized_pnl

    @property
    def pnl_percent(self) -> Decimal:
        if self.total_invested == 0:
            return Decimal("0")
        return (self.total_pnl / self.total_invested) * 100


@dataclass
class WalletSummary:
    """Summary of wallet activity."""
    address: str
    total_transactions: int
    total_volume: Decimal
    total_realized_pnl: Decimal
    total_fees_paid: Decimal
    first_transaction: Optional[datetime]
    last_transaction: Optional[datetime]
    unique_markets: int
    open_positions: int
    closed_positions: int
    win_count: int
    loss_count: int
    avg_position_size: Decimal
    largest_position: Decimal
    transactions: list[WalletTransaction] = field(default_factory=list)
    positions: list[WalletPosition] = field(default_factory=list)

    @property
    def win_rate(self) -> Decimal:
        total = self.win_count + self.loss_count
        if total == 0:
            return Decimal("0")
        return Decimal(str(self.win_count / total * 100))


class WalletAnalyzer:
    """Analyze Polymarket wallet history and performance."""

    # Polymarket API endpoints
    GAMMA_API = "https://gamma-api.polymarket.com"
    CLOB_API = "https://clob.polymarket.com"

    def __init__(self, timeout: float = 30.0):
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    async def close(self) -> None:
        """Close HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    async def fetch_wallet_history(
        self,
        address: str,
        limit: int = 500,
        offset: int = 0
    ) -> list[WalletTransaction]:
        """
        Fetch transaction history for a wallet address.

        Args:
            address: Polygon wallet address
            limit: Maximum transactions to fetch
            offset: Pagination offset

        Returns:
            List of wallet transactions
        """
        client = await self._get_client()
        transactions: list[WalletTransaction] = []

        # Fetch from Polymarket activity API
        try:
            # Get user activity from CLOB
            activity_url = f"{self.CLOB_API}/activity"
            params = {
                "user": address.lower(),
                "limit": min(limit, 100),
                "offset": offset
            }

            response = await client.get(activity_url, params=params)

            if response.status_code == 200:
                data = response.json()
                for item in data.get("data", []):
                    tx = self._parse_activity_item(item)
                    if tx:
                        transactions.append(tx)

            # Also fetch from gamma API for market info
            gamma_url = f"{self.GAMMA_API}/trades"
            params = {
                "user": address.lower(),
                "limit": min(limit, 100)
            }

            response = await client.get(gamma_url, params=params)

            if response.status_code == 200:
                data = response.json()
                for item in data:
                    tx = self._parse_gamma_trade(item)
                    if tx and tx.tx_hash not in {t.tx_hash for t in transactions}:
                        transactions.append(tx)

        except httpx.RequestError as e:
            # Log error but continue with what we have
            print(f"Warning: API request failed: {e}")

        # Sort by timestamp
        transactions.sort(key=lambda t: t.timestamp, reverse=True)

        return transactions[:limit]

    def _parse_activity_item(self, item: dict) -> Optional[WalletTransaction]:
        """Parse activity API response item."""
        try:
            tx_type_map = {
                "buy": TransactionType.BUY,
                "sell": TransactionType.SELL,
                "deposit": TransactionType.DEPOSIT,
                "withdrawal": TransactionType.WITHDRAWAL,
            }

            action = item.get("action", "").lower()
            tx_type = tx_type_map.get(action)
            if not tx_type:
                return None

            timestamp_str = item.get("timestamp") or item.get("created_at")
            if timestamp_str:
                timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
            else:
                timestamp = datetime.now(timezone.utc)

            return WalletTransaction(
                tx_hash=item.get("transaction_hash", item.get("id", "")),
                timestamp=timestamp,
                tx_type=tx_type,
                market_id=item.get("market_id") or item.get("condition_id"),
                market_question=item.get("question") or item.get("title"),
                outcome=item.get("outcome") or item.get("side"),
                amount=Decimal(str(item.get("amount", 0))),
                shares=Decimal(str(item.get("size", 0))) if item.get("size") else None,
                price=Decimal(str(item.get("price", 0))) if item.get("price") else None,
                fee=Decimal(str(item.get("fee", 0))),
                block_number=item.get("block_number", 0)
            )
        except (ValueError, KeyError):
            return None

    def _parse_gamma_trade(self, item: dict) -> Optional[WalletTransaction]:
        """Parse gamma API trade item."""
        try:
            side = item.get("side", "").upper()
            tx_type = TransactionType.BUY if side == "BUY" else TransactionType.SELL

            timestamp_str = item.get("timestamp") or item.get("created_at")
            if timestamp_str:
                if isinstance(timestamp_str, (int, float)):
                    timestamp = datetime.fromtimestamp(timestamp_str, tz=timezone.utc)
                else:
                    timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
            else:
                timestamp = datetime.now(timezone.utc)

            price = Decimal(str(item.get("price", 0)))
            size = Decimal(str(item.get("size", 0)))
            amount = price * size

            return WalletTransaction(
                tx_hash=item.get("id", ""),
                timestamp=timestamp,
                tx_type=tx_type,
                market_id=item.get("market") or item.get("condition_id"),
                market_question=item.get("question"),
                outcome=item.get("outcome") or item.get("asset_id"),
                amount=amount,
                shares=size,
                price=price,
                fee=Decimal(str(item.get("fee", 0)))
            )
        except (ValueError, KeyError):
            return None

    async def fetch_positions(
        self,
        address: str,
        include_closed: bool = True
    ) -> list[WalletPosition]:
        """
        Fetch current and historical positions.

        Args:
            address: Wallet address
            include_closed: Include closed/resolved positions

        Returns:
            List of positions
        """
        client = await self._get_client()
        positions: list[WalletPosition] = []

        try:
            # Fetch from CLOB positions endpoint
            url = f"{self.CLOB_API}/positions"
            params = {"user": address.lower()}

            response = await client.get(url, params=params)

            if response.status_code == 200:
                data = response.json()
                for item in data:
                    pos = self._parse_position(item)
                    if pos:
                        if include_closed or not pos.is_closed:
                            positions.append(pos)

        except httpx.RequestError as e:
            print(f"Warning: Position fetch failed: {e}")

        return positions

    def _parse_position(self, item: dict) -> Optional[WalletPosition]:
        """Parse position from API response."""
        try:
            shares = Decimal(str(item.get("size", 0)))
            avg_price = Decimal(str(item.get("avg_price", 0)))
            current_price = Decimal(str(item.get("current_price", 0))) if item.get("current_price") else None

            position = WalletPosition(
                market_id=item.get("market") or item.get("condition_id", ""),
                market_question=item.get("question", "Unknown Market"),
                outcome=item.get("outcome", "YES"),
                shares=shares,
                avg_entry_price=avg_price,
                current_price=current_price,
                total_invested=shares * avg_price,
                is_closed=shares == 0,
                is_resolved=item.get("resolved", False)
            )

            if current_price and shares > 0:
                position.unrealized_pnl = shares * (current_price - avg_price)

            return position
        except (ValueError, KeyError):
            return None

    def build_positions_from_history(
        self,
        transactions: list[WalletTransaction]
    ) -> list[WalletPosition]:
        """
        Reconstruct positions from transaction history.

        More reliable than API positions for historical analysis.
        """
        # Group transactions by market and outcome
        market_txs: dict[tuple[str, str], list[WalletTransaction]] = {}

        for tx in transactions:
            if tx.market_id and tx.outcome:
                key = (tx.market_id, tx.outcome)
                if key not in market_txs:
                    market_txs[key] = []
                market_txs[key].append(tx)

        positions: list[WalletPosition] = []

        for (market_id, outcome), txs in market_txs.items():
            # Sort by timestamp
            txs.sort(key=lambda t: t.timestamp)

            total_shares = Decimal("0")
            total_cost = Decimal("0")
            total_invested = Decimal("0")
            total_returned = Decimal("0")
            realized_pnl = Decimal("0")
            entry_time = None
            exit_time = None
            market_question = ""

            for tx in txs:
                if tx.market_question:
                    market_question = tx.market_question

                if tx.tx_type == TransactionType.BUY:
                    if entry_time is None:
                        entry_time = tx.timestamp

                    shares = tx.shares or Decimal("0")
                    cost = tx.amount

                    total_shares += shares
                    total_cost += cost
                    total_invested += cost

                elif tx.tx_type == TransactionType.SELL:
                    shares = tx.shares or Decimal("0")
                    proceeds = tx.amount

                    if total_shares > 0:
                        avg_cost = total_cost / total_shares
                        cost_basis = shares * avg_cost
                        realized_pnl += proceeds - cost_basis
                        total_cost -= cost_basis

                    total_shares -= shares
                    total_returned += proceeds
                    exit_time = tx.timestamp

                elif tx.tx_type == TransactionType.CLAIM:
                    # Market resolved
                    realized_pnl += tx.amount - total_cost
                    total_returned += tx.amount
                    total_shares = Decimal("0")
                    total_cost = Decimal("0")
                    exit_time = tx.timestamp

            avg_entry = total_invested / (total_shares + sum(
                tx.shares or 0 for tx in txs if tx.tx_type == TransactionType.SELL
            )) if total_invested > 0 else Decimal("0")

            position = WalletPosition(
                market_id=market_id,
                market_question=market_question,
                outcome=outcome,
                shares=max(total_shares, Decimal("0")),
                avg_entry_price=avg_entry,
                current_price=None,
                realized_pnl=realized_pnl,
                total_invested=total_invested,
                total_returned=total_returned,
                entry_time=entry_time,
                exit_time=exit_time,
                is_closed=total_shares <= 0
            )

            positions.append(position)

        return positions

    async def analyze_wallet(self, address: str) -> WalletSummary:
        """
        Comprehensive wallet analysis.

        Args:
            address: Wallet address to analyze

        Returns:
            Complete wallet summary
        """
        # Fetch all transactions
        transactions = await self.fetch_wallet_history(address, limit=1000)

        # Build positions from history
        positions = self.build_positions_from_history(transactions)

        # Calculate summary statistics
        total_volume = sum(tx.amount for tx in transactions if tx.tx_type in [TransactionType.BUY, TransactionType.SELL])
        total_fees = sum(tx.fee for tx in transactions)
        total_realized_pnl = sum(p.realized_pnl for p in positions)

        open_positions = [p for p in positions if not p.is_closed]
        closed_positions = [p for p in positions if p.is_closed]

        win_count = sum(1 for p in closed_positions if p.realized_pnl > 0)
        loss_count = sum(1 for p in closed_positions if p.realized_pnl < 0)

        unique_markets = len(set(tx.market_id for tx in transactions if tx.market_id))

        position_sizes = [p.total_invested for p in positions if p.total_invested > 0]
        avg_position_size = sum(position_sizes) / len(position_sizes) if position_sizes else Decimal("0")
        largest_position = max(position_sizes) if position_sizes else Decimal("0")

        return WalletSummary(
            address=address,
            total_transactions=len(transactions),
            total_volume=total_volume,
            total_realized_pnl=total_realized_pnl,
            total_fees_paid=total_fees,
            first_transaction=min((tx.timestamp for tx in transactions), default=None),
            last_transaction=max((tx.timestamp for tx in transactions), default=None),
            unique_markets=unique_markets,
            open_positions=len(open_positions),
            closed_positions=len(closed_positions),
            win_count=win_count,
            loss_count=loss_count,
            avg_position_size=avg_position_size,
            largest_position=largest_position,
            transactions=transactions,
            positions=positions
        )

    async def compare_wallets(
        self,
        addresses: list[str]
    ) -> dict[str, WalletSummary]:
        """
        Compare multiple wallets.

        Args:
            addresses: List of wallet addresses

        Returns:
            Dict mapping address to summary
        """
        results = {}
        tasks = [self.analyze_wallet(addr) for addr in addresses]
        summaries = await asyncio.gather(*tasks, return_exceptions=True)

        for addr, summary in zip(addresses, summaries):
            if isinstance(summary, WalletSummary):
                results[addr] = summary
            else:
                print(f"Warning: Failed to analyze {addr}: {summary}")

        return results
