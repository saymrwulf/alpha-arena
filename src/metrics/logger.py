"""Logging and metrics storage with JSONL and SQLite."""

import asyncio
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import aiosqlite
import orjson


@dataclass
class DecisionLog:
    """Log entry for agent decisions."""

    timestamp: datetime
    iteration: int
    balance: Decimal
    positions_count: int
    markets_analyzed: int
    signals_generated: int
    signals_executed: int
    model_used: str
    latency_ms: int
    tokens_used: int
    reasoning: str
    signals: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class OrderLog:
    """Log entry for orders."""

    timestamp: datetime
    order_id: str
    market_id: str
    token_id: str
    side: str
    size: Decimal
    price: Decimal
    status: str
    fill_size: Decimal | None = None
    fill_price: Decimal | None = None
    fee: Decimal | None = None


@dataclass
class PnLSnapshot:
    """Point-in-time PnL snapshot."""

    timestamp: datetime
    balance: Decimal
    unrealized_pnl: Decimal
    realized_pnl: Decimal
    total_equity: Decimal
    positions_value: Decimal
    drawdown_pct: Decimal
    high_water_mark: Decimal


class MetricsLogger:
    """
    Dual logging to JSONL (decisions) and SQLite (aggregates).

    JSONL: Append-only log of all decisions and reasoning.
    SQLite: Queryable metrics for analysis.
    """

    def __init__(
        self,
        jsonl_path: str = "logs/decisions.jsonl",
        sqlite_path: str = "logs/metrics.db",
    ):
        self.jsonl_path = Path(jsonl_path)
        self.sqlite_path = Path(sqlite_path)
        self._db: aiosqlite.Connection | None = None
        self._high_water_mark: Decimal = Decimal("0")

    async def connect(self) -> None:
        """Initialize logging backends."""
        # Create directories
        self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize SQLite
        self._db = await aiosqlite.connect(str(self.sqlite_path))
        await self._init_db()

    async def disconnect(self) -> None:
        """Close logging backends."""
        if self._db:
            await self._db.close()
            self._db = None

    async def _init_db(self) -> None:
        """Initialize SQLite schema."""
        if self._db is None:
            return

        await self._db.executescript("""
            CREATE TABLE IF NOT EXISTS decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                iteration INTEGER,
                balance REAL,
                positions_count INTEGER,
                markets_analyzed INTEGER,
                signals_generated INTEGER,
                signals_executed INTEGER,
                model_used TEXT,
                latency_ms INTEGER,
                tokens_used INTEGER
            );

            CREATE TABLE IF NOT EXISTS orders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                order_id TEXT UNIQUE,
                market_id TEXT,
                token_id TEXT,
                side TEXT,
                size REAL,
                price REAL,
                status TEXT,
                fill_size REAL,
                fill_price REAL,
                fee REAL
            );

            CREATE TABLE IF NOT EXISTS pnl_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                balance REAL,
                unrealized_pnl REAL,
                realized_pnl REAL,
                total_equity REAL,
                positions_value REAL,
                drawdown_pct REAL,
                high_water_mark REAL
            );

            CREATE INDEX IF NOT EXISTS idx_decisions_timestamp ON decisions(timestamp);
            CREATE INDEX IF NOT EXISTS idx_orders_timestamp ON orders(timestamp);
            CREATE INDEX IF NOT EXISTS idx_pnl_timestamp ON pnl_snapshots(timestamp);
        """)
        await self._db.commit()

    async def log_decision(self, log: DecisionLog) -> None:
        """Log a decision to both JSONL and SQLite."""
        # JSONL append
        log_dict = {
            "type": "decision",
            "timestamp": log.timestamp.isoformat(),
            "iteration": log.iteration,
            "balance": str(log.balance),
            "positions_count": log.positions_count,
            "markets_analyzed": log.markets_analyzed,
            "signals_generated": log.signals_generated,
            "signals_executed": log.signals_executed,
            "model_used": log.model_used,
            "latency_ms": log.latency_ms,
            "tokens_used": log.tokens_used,
            "reasoning": log.reasoning,
            "signals": log.signals,
        }

        await self._append_jsonl(log_dict)

        # SQLite insert
        if self._db:
            await self._db.execute("""
                INSERT INTO decisions (
                    timestamp, iteration, balance, positions_count,
                    markets_analyzed, signals_generated, signals_executed,
                    model_used, latency_ms, tokens_used
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                log.timestamp.isoformat(),
                log.iteration,
                float(log.balance),
                log.positions_count,
                log.markets_analyzed,
                log.signals_generated,
                log.signals_executed,
                log.model_used,
                log.latency_ms,
                log.tokens_used,
            ))
            await self._db.commit()

    async def log_order(self, log: OrderLog) -> None:
        """Log an order."""
        log_dict = {
            "type": "order",
            "timestamp": log.timestamp.isoformat(),
            "order_id": log.order_id,
            "market_id": log.market_id,
            "token_id": log.token_id,
            "side": log.side,
            "size": str(log.size),
            "price": str(log.price),
            "status": log.status,
            "fill_size": str(log.fill_size) if log.fill_size else None,
            "fill_price": str(log.fill_price) if log.fill_price else None,
            "fee": str(log.fee) if log.fee else None,
        }

        await self._append_jsonl(log_dict)

        if self._db:
            await self._db.execute("""
                INSERT OR REPLACE INTO orders (
                    timestamp, order_id, market_id, token_id, side,
                    size, price, status, fill_size, fill_price, fee
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                log.timestamp.isoformat(),
                log.order_id,
                log.market_id,
                log.token_id,
                log.side,
                float(log.size),
                float(log.price),
                log.status,
                float(log.fill_size) if log.fill_size else None,
                float(log.fill_price) if log.fill_price else None,
                float(log.fee) if log.fee else None,
            ))
            await self._db.commit()

    async def log_pnl_snapshot(self, snapshot: PnLSnapshot) -> None:
        """Log a PnL snapshot."""
        # Update high water mark
        if snapshot.total_equity > self._high_water_mark:
            self._high_water_mark = snapshot.total_equity

        snapshot_dict = {
            "type": "pnl_snapshot",
            "timestamp": snapshot.timestamp.isoformat(),
            "balance": str(snapshot.balance),
            "unrealized_pnl": str(snapshot.unrealized_pnl),
            "realized_pnl": str(snapshot.realized_pnl),
            "total_equity": str(snapshot.total_equity),
            "positions_value": str(snapshot.positions_value),
            "drawdown_pct": str(snapshot.drawdown_pct),
            "high_water_mark": str(snapshot.high_water_mark),
        }

        await self._append_jsonl(snapshot_dict)

        if self._db:
            await self._db.execute("""
                INSERT INTO pnl_snapshots (
                    timestamp, balance, unrealized_pnl, realized_pnl,
                    total_equity, positions_value, drawdown_pct, high_water_mark
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                snapshot.timestamp.isoformat(),
                float(snapshot.balance),
                float(snapshot.unrealized_pnl),
                float(snapshot.realized_pnl),
                float(snapshot.total_equity),
                float(snapshot.positions_value),
                float(snapshot.drawdown_pct),
                float(snapshot.high_water_mark),
            ))
            await self._db.commit()

    async def _append_jsonl(self, data: dict[str, Any]) -> None:
        """Append a line to JSONL file."""
        line = orjson.dumps(data).decode() + "\n"

        # Use asyncio for file I/O
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: self.jsonl_path.open("a").write(line)
        )

    async def get_recent_decisions(self, limit: int = 100) -> list[dict[str, Any]]:
        """Get recent decisions from SQLite."""
        if self._db is None:
            return []

        async with self._db.execute("""
            SELECT * FROM decisions ORDER BY timestamp DESC LIMIT ?
        """, (limit,)) as cursor:
            rows = await cursor.fetchall()
            columns = [d[0] for d in cursor.description]
            return [dict(zip(columns, row)) for row in rows]

    async def get_pnl_history(
        self,
        since: datetime | None = None,
    ) -> list[dict[str, Any]]:
        """Get PnL history."""
        if self._db is None:
            return []

        if since:
            async with self._db.execute("""
                SELECT * FROM pnl_snapshots
                WHERE timestamp >= ?
                ORDER BY timestamp ASC
            """, (since.isoformat(),)) as cursor:
                rows = await cursor.fetchall()
        else:
            async with self._db.execute("""
                SELECT * FROM pnl_snapshots ORDER BY timestamp ASC
            """) as cursor:
                rows = await cursor.fetchall()

        columns = [d[0] for d in cursor.description]
        return [dict(zip(columns, row)) for row in rows]

    async def get_statistics(self) -> dict[str, Any]:
        """Get aggregate statistics."""
        if self._db is None:
            return {}

        stats = {}

        # Decision stats
        async with self._db.execute("""
            SELECT
                COUNT(*) as total_decisions,
                AVG(latency_ms) as avg_latency,
                SUM(signals_executed) as total_trades,
                SUM(tokens_used) as total_tokens
            FROM decisions
        """) as cursor:
            row = await cursor.fetchone()
            if row:
                stats["total_decisions"] = row[0]
                stats["avg_latency_ms"] = row[1]
                stats["total_trades"] = row[2]
                stats["total_tokens"] = row[3]

        # PnL stats
        async with self._db.execute("""
            SELECT
                MIN(total_equity) as min_equity,
                MAX(total_equity) as max_equity,
                MIN(drawdown_pct) as max_drawdown
            FROM pnl_snapshots
        """) as cursor:
            row = await cursor.fetchone()
            if row:
                stats["min_equity"] = row[0]
                stats["max_equity"] = row[1]
                stats["max_drawdown_pct"] = row[2]

        # Latest PnL
        async with self._db.execute("""
            SELECT total_equity, realized_pnl, unrealized_pnl
            FROM pnl_snapshots
            ORDER BY timestamp DESC
            LIMIT 1
        """) as cursor:
            row = await cursor.fetchone()
            if row:
                stats["current_equity"] = row[0]
                stats["realized_pnl"] = row[1]
                stats["unrealized_pnl"] = row[2]

        return stats

    def calculate_drawdown(
        self,
        current_equity: Decimal,
    ) -> Decimal:
        """Calculate current drawdown from high water mark."""
        if self._high_water_mark <= 0:
            return Decimal("0")

        drawdown = (self._high_water_mark - current_equity) / self._high_water_mark
        return drawdown.quantize(Decimal("0.0001"))
