"""Episodic memory - complete trading episodes for reflection."""

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import aiosqlite

from .base import MemoryEntry, MemoryType


@dataclass
class Episode:
    """A complete trading episode from signal to outcome."""
    id: str
    market_id: str
    market_question: str

    # Timeline
    signal_time: datetime
    entry_time: datetime | None = None
    exit_time: datetime | None = None

    # Decision
    signal_type: str = ""
    signal_confidence: float = 0.0
    signal_reasoning: str = ""
    agents_involved: list[str] = field(default_factory=list)

    # Execution
    entry_price: Decimal | None = None
    exit_price: Decimal | None = None
    size: Decimal | None = None

    # Outcome
    pnl: Decimal | None = None
    pnl_percent: Decimal | None = None
    outcome: str = ""  # "win", "loss", "breakeven"
    hold_duration_hours: float | None = None

    # Analysis
    what_worked: list[str] = field(default_factory=list)
    what_failed: list[str] = field(default_factory=list)
    lessons: list[str] = field(default_factory=list)
    counterfactual: str = ""  # What could have been done differently

    # Market context at time of trade
    market_context: dict[str, Any] = field(default_factory=dict)

    # Status
    status: str = "open"  # "open", "closed", "reflected"

    def to_memory_entry(self) -> MemoryEntry:
        """Convert to MemoryEntry for storage."""
        importance = 0.5
        if self.pnl:
            # Higher importance for bigger wins/losses
            importance = min(0.9, 0.5 + abs(float(self.pnl)) / 100)

        return MemoryEntry(
            id=self.id,
            type=MemoryType.OUTCOME,
            content={
                "market_id": self.market_id,
                "market_question": self.market_question,
                "signal_type": self.signal_type,
                "signal_confidence": self.signal_confidence,
                "signal_reasoning": self.signal_reasoning,
                "entry_price": str(self.entry_price) if self.entry_price else None,
                "exit_price": str(self.exit_price) if self.exit_price else None,
                "size": str(self.size) if self.size else None,
                "outcome": self.outcome,
                "hold_duration_hours": self.hold_duration_hours,
                "what_worked": self.what_worked,
                "what_failed": self.what_failed,
                "lessons": self.lessons,
                "counterfactual": self.counterfactual,
            },
            timestamp=self.signal_time,
            importance=importance,
            pnl=self.pnl,
            success=self.outcome == "win",
            tags=[self.outcome, self.signal_type],
        )


class EpisodicMemory:
    """
    Episodic memory for complete trading episodes.

    Tracks the full lifecycle of trades for:
    - Post-trade reflection
    - Pattern extraction
    - Learning from outcomes
    """

    def __init__(self, db_path: str = "logs/episodes.db"):
        self.db_path = Path(db_path)
        self._db: aiosqlite.Connection | None = None

    async def connect(self) -> None:
        """Initialize database."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(str(self.db_path))

        await self._db.executescript("""
            CREATE TABLE IF NOT EXISTS episodes (
                id TEXT PRIMARY KEY,
                market_id TEXT,
                market_question TEXT,
                signal_time TEXT,
                entry_time TEXT,
                exit_time TEXT,
                signal_type TEXT,
                signal_confidence REAL,
                signal_reasoning TEXT,
                agents_involved TEXT,
                entry_price REAL,
                exit_price REAL,
                size REAL,
                pnl REAL,
                pnl_percent REAL,
                outcome TEXT,
                hold_duration_hours REAL,
                what_worked TEXT,
                what_failed TEXT,
                lessons TEXT,
                counterfactual TEXT,
                market_context TEXT,
                status TEXT DEFAULT 'open'
            );

            CREATE INDEX IF NOT EXISTS idx_episodes_status ON episodes(status);
            CREATE INDEX IF NOT EXISTS idx_episodes_outcome ON episodes(outcome);
            CREATE INDEX IF NOT EXISTS idx_episodes_market ON episodes(market_id);
        """)
        await self._db.commit()

    async def disconnect(self) -> None:
        """Close database."""
        if self._db:
            await self._db.close()
            self._db = None

    async def start_episode(
        self,
        market_id: str,
        market_question: str,
        signal_type: str,
        signal_confidence: float,
        signal_reasoning: str,
        agents_involved: list[str],
        market_context: dict[str, Any],
    ) -> str:
        """Start a new trading episode."""
        if self._db is None:
            raise RuntimeError("Memory not connected")

        episode_id = str(uuid.uuid4())

        await self._db.execute("""
            INSERT INTO episodes
            (id, market_id, market_question, signal_time, signal_type,
             signal_confidence, signal_reasoning, agents_involved, market_context, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'open')
        """, (
            episode_id,
            market_id,
            market_question,
            datetime.utcnow().isoformat(),
            signal_type,
            signal_confidence,
            signal_reasoning,
            json.dumps(agents_involved),
            json.dumps(market_context),
        ))
        await self._db.commit()

        return episode_id

    async def record_entry(
        self,
        episode_id: str,
        entry_price: Decimal,
        size: Decimal,
    ) -> None:
        """Record trade entry."""
        if self._db is None:
            raise RuntimeError("Memory not connected")

        await self._db.execute("""
            UPDATE episodes
            SET entry_time = ?, entry_price = ?, size = ?
            WHERE id = ?
        """, (
            datetime.utcnow().isoformat(),
            float(entry_price),
            float(size),
            episode_id,
        ))
        await self._db.commit()

    async def close_episode(
        self,
        episode_id: str,
        exit_price: Decimal,
        pnl: Decimal,
    ) -> Episode:
        """Close an episode with exit details."""
        if self._db is None:
            raise RuntimeError("Memory not connected")

        # Get episode
        async with self._db.execute(
            "SELECT * FROM episodes WHERE id = ?",
            (episode_id,)
        ) as cursor:
            row = await cursor.fetchone()

        if not row:
            raise ValueError(f"Episode not found: {episode_id}")

        episode = self._row_to_episode(row)

        # Calculate outcome
        outcome = "win" if pnl > 0 else "loss" if pnl < 0 else "breakeven"
        pnl_percent = None
        if episode.entry_price and episode.size:
            cost = episode.entry_price * episode.size
            if cost > 0:
                pnl_percent = (pnl / cost) * 100

        hold_duration = None
        if episode.entry_time:
            hold_duration = (datetime.utcnow() - episode.entry_time).total_seconds() / 3600

        await self._db.execute("""
            UPDATE episodes
            SET exit_time = ?, exit_price = ?, pnl = ?, pnl_percent = ?,
                outcome = ?, hold_duration_hours = ?, status = 'closed'
            WHERE id = ?
        """, (
            datetime.utcnow().isoformat(),
            float(exit_price),
            float(pnl),
            float(pnl_percent) if pnl_percent else None,
            outcome,
            hold_duration,
            episode_id,
        ))
        await self._db.commit()

        # Return updated episode
        async with self._db.execute(
            "SELECT * FROM episodes WHERE id = ?",
            (episode_id,)
        ) as cursor:
            row = await cursor.fetchone()

        return self._row_to_episode(row)

    async def add_reflection(
        self,
        episode_id: str,
        what_worked: list[str],
        what_failed: list[str],
        lessons: list[str],
        counterfactual: str,
    ) -> None:
        """Add reflection analysis to episode."""
        if self._db is None:
            raise RuntimeError("Memory not connected")

        await self._db.execute("""
            UPDATE episodes
            SET what_worked = ?, what_failed = ?, lessons = ?,
                counterfactual = ?, status = 'reflected'
            WHERE id = ?
        """, (
            json.dumps(what_worked),
            json.dumps(what_failed),
            json.dumps(lessons),
            counterfactual,
            episode_id,
        ))
        await self._db.commit()

    async def get_open_episodes(self) -> list[Episode]:
        """Get all open episodes."""
        if self._db is None:
            return []

        async with self._db.execute(
            "SELECT * FROM episodes WHERE status = 'open'"
        ) as cursor:
            rows = await cursor.fetchall()

        return [self._row_to_episode(row) for row in rows]

    async def get_unreflected_episodes(self, limit: int = 10) -> list[Episode]:
        """Get closed episodes awaiting reflection."""
        if self._db is None:
            return []

        async with self._db.execute("""
            SELECT * FROM episodes
            WHERE status = 'closed'
            ORDER BY exit_time DESC
            LIMIT ?
        """, (limit,)) as cursor:
            rows = await cursor.fetchall()

        return [self._row_to_episode(row) for row in rows]

    async def get_recent_episodes(
        self,
        limit: int = 20,
        outcome: str | None = None,
    ) -> list[Episode]:
        """Get recent episodes."""
        if self._db is None:
            return []

        if outcome:
            async with self._db.execute("""
                SELECT * FROM episodes
                WHERE outcome = ?
                ORDER BY signal_time DESC
                LIMIT ?
            """, (outcome, limit)) as cursor:
                rows = await cursor.fetchall()
        else:
            async with self._db.execute("""
                SELECT * FROM episodes
                ORDER BY signal_time DESC
                LIMIT ?
            """, (limit,)) as cursor:
                rows = await cursor.fetchall()

        return [self._row_to_episode(row) for row in rows]

    async def get_stats(self) -> dict[str, Any]:
        """Get episode statistics."""
        if self._db is None:
            return {}

        stats = {}

        async with self._db.execute("SELECT COUNT(*) FROM episodes") as cursor:
            stats["total"] = (await cursor.fetchone())[0]

        async with self._db.execute("""
            SELECT outcome, COUNT(*), AVG(pnl), SUM(pnl)
            FROM episodes WHERE outcome IS NOT NULL
            GROUP BY outcome
        """) as cursor:
            for row in await cursor.fetchall():
                outcome, count, avg_pnl, total_pnl = row
                stats[f"{outcome}_count"] = count
                stats[f"{outcome}_avg_pnl"] = avg_pnl
                stats[f"{outcome}_total_pnl"] = total_pnl

        async with self._db.execute("""
            SELECT AVG(hold_duration_hours) FROM episodes
            WHERE hold_duration_hours IS NOT NULL
        """) as cursor:
            stats["avg_hold_hours"] = (await cursor.fetchone())[0]

        return stats

    def _row_to_episode(self, row: tuple) -> Episode:
        """Convert database row to Episode."""
        return Episode(
            id=row[0],
            market_id=row[1],
            market_question=row[2],
            signal_time=datetime.fromisoformat(row[3]) if row[3] else datetime.utcnow(),
            entry_time=datetime.fromisoformat(row[4]) if row[4] else None,
            exit_time=datetime.fromisoformat(row[5]) if row[5] else None,
            signal_type=row[6] or "",
            signal_confidence=row[7] or 0.0,
            signal_reasoning=row[8] or "",
            agents_involved=json.loads(row[9]) if row[9] else [],
            entry_price=Decimal(str(row[10])) if row[10] else None,
            exit_price=Decimal(str(row[11])) if row[11] else None,
            size=Decimal(str(row[12])) if row[12] else None,
            pnl=Decimal(str(row[13])) if row[13] else None,
            pnl_percent=Decimal(str(row[14])) if row[14] else None,
            outcome=row[15] or "",
            hold_duration_hours=row[16],
            what_worked=json.loads(row[17]) if row[17] else [],
            what_failed=json.loads(row[18]) if row[18] else [],
            lessons=json.loads(row[19]) if row[19] else [],
            counterfactual=row[20] or "",
            market_context=json.loads(row[21]) if row[21] else {},
            status=row[22] or "open",
        )
