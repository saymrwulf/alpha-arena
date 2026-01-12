"""Long-term memory - patterns, lessons, and historical data."""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import aiosqlite

from .base import Memory, MemoryEntry, MemoryType


class LongTermMemory(Memory):
    """
    Long-term memory using SQLite with vector similarity.

    Stores:
    - Learned patterns
    - Trading lessons
    - Historical outcomes
    - Semantic embeddings for retrieval
    """

    def __init__(
        self,
        db_path: str = "logs/memory.db",
        capacity: int = 10000,
    ):
        self.db_path = Path(db_path)
        self.capacity = capacity
        self._db: aiosqlite.Connection | None = None

    async def connect(self) -> None:
        """Initialize database."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(str(self.db_path))

        await self._db.executescript("""
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                type TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                importance REAL DEFAULT 0.5,
                embedding TEXT,
                tags TEXT,
                related_ids TEXT,
                pnl REAL,
                success INTEGER,
                access_count INTEGER DEFAULT 0,
                last_accessed TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(type);
            CREATE INDEX IF NOT EXISTS idx_memories_timestamp ON memories(timestamp);
            CREATE INDEX IF NOT EXISTS idx_memories_importance ON memories(importance);
        """)
        await self._db.commit()

    async def disconnect(self) -> None:
        """Close database."""
        if self._db:
            await self._db.close()
            self._db = None

    async def store(self, entry: MemoryEntry) -> None:
        """Store entry in long-term memory."""
        if self._db is None:
            raise RuntimeError("Memory not connected")

        # Check capacity and evict if needed
        await self._evict_if_needed()

        await self._db.execute("""
            INSERT OR REPLACE INTO memories
            (id, type, content, timestamp, importance, embedding, tags, related_ids, pnl, success)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            entry.id,
            entry.type.value,
            json.dumps(entry.content),
            entry.timestamp.isoformat(),
            entry.importance,
            json.dumps(entry.embedding) if entry.embedding else None,
            json.dumps(entry.tags),
            json.dumps(entry.related_ids),
            float(entry.pnl) if entry.pnl else None,
            1 if entry.success else 0 if entry.success is not None else None,
        ))
        await self._db.commit()

    async def retrieve(
        self,
        query: str | None = None,
        type_filter: MemoryType | None = None,
        limit: int = 10,
        min_importance: float = 0.0,
    ) -> list[MemoryEntry]:
        """Retrieve memories."""
        if self._db is None:
            raise RuntimeError("Memory not connected")

        sql = "SELECT * FROM memories WHERE importance >= ?"
        params: list[Any] = [min_importance]

        if type_filter:
            sql += " AND type = ?"
            params.append(type_filter.value)

        if query:
            # Simple text search in content
            sql += " AND content LIKE ?"
            params.append(f"%{query}%")

        sql += " ORDER BY importance DESC, timestamp DESC LIMIT ?"
        params.append(limit)

        async with self._db.execute(sql, params) as cursor:
            rows = await cursor.fetchall()

        # Update access counts
        ids = [row[0] for row in rows]
        if ids:
            placeholders = ",".join("?" * len(ids))
            await self._db.execute(f"""
                UPDATE memories
                SET access_count = access_count + 1, last_accessed = ?
                WHERE id IN ({placeholders})
            """, [datetime.utcnow().isoformat()] + ids)
            await self._db.commit()

        return [self._row_to_entry(row) for row in rows]

    async def search_similar(
        self,
        embedding: list[float],
        limit: int = 5,
    ) -> list[MemoryEntry]:
        """Find semantically similar memories using cosine similarity."""
        if self._db is None:
            raise RuntimeError("Memory not connected")

        # Get all entries with embeddings
        async with self._db.execute(
            "SELECT * FROM memories WHERE embedding IS NOT NULL"
        ) as cursor:
            rows = await cursor.fetchall()

        if not rows:
            return []

        # Calculate similarities
        scored = []
        for row in rows:
            stored_embedding = json.loads(row[5]) if row[5] else None
            if stored_embedding:
                similarity = self._cosine_similarity(embedding, stored_embedding)
                scored.append((similarity, row))

        # Sort by similarity and return top results
        scored.sort(key=lambda x: x[0], reverse=True)
        return [self._row_to_entry(row) for _, row in scored[:limit]]

    async def clear(self) -> None:
        """Clear all memories."""
        if self._db is None:
            raise RuntimeError("Memory not connected")

        await self._db.execute("DELETE FROM memories")
        await self._db.commit()

    async def get_stats(self) -> dict[str, Any]:
        """Get memory statistics."""
        if self._db is None:
            return {"type": "long_term", "connected": False}

        async with self._db.execute("SELECT COUNT(*) FROM memories") as cursor:
            total = (await cursor.fetchone())[0]

        async with self._db.execute("""
            SELECT type, COUNT(*) FROM memories GROUP BY type
        """) as cursor:
            by_type = dict(await cursor.fetchall())

        async with self._db.execute("""
            SELECT AVG(importance), AVG(access_count) FROM memories
        """) as cursor:
            row = await cursor.fetchone()
            avg_importance, avg_access = row

        return {
            "type": "long_term",
            "total_entries": total,
            "capacity": self.capacity,
            "utilization": total / self.capacity,
            "by_type": by_type,
            "avg_importance": avg_importance or 0,
            "avg_access_count": avg_access or 0,
        }

    async def get_patterns(self, min_confidence: float = 0.6) -> list[MemoryEntry]:
        """Get learned patterns above confidence threshold."""
        return await self.retrieve(
            type_filter=MemoryType.PATTERN,
            min_importance=min_confidence,
            limit=50,
        )

    async def get_lessons(self, limit: int = 20) -> list[MemoryEntry]:
        """Get extracted lessons."""
        return await self.retrieve(
            type_filter=MemoryType.LESSON,
            limit=limit,
        )

    async def get_successful_trades(self, limit: int = 20) -> list[MemoryEntry]:
        """Get successful trade memories."""
        if self._db is None:
            return []

        async with self._db.execute("""
            SELECT * FROM memories
            WHERE type = ? AND success = 1
            ORDER BY pnl DESC
            LIMIT ?
        """, (MemoryType.OUTCOME.value, limit)) as cursor:
            rows = await cursor.fetchall()

        return [self._row_to_entry(row) for row in rows]

    async def _evict_if_needed(self) -> None:
        """Evict low-importance entries if at capacity."""
        if self._db is None:
            return

        async with self._db.execute("SELECT COUNT(*) FROM memories") as cursor:
            count = (await cursor.fetchone())[0]

        if count >= self.capacity:
            # Delete 10% lowest importance entries
            evict_count = int(self.capacity * 0.1)
            await self._db.execute("""
                DELETE FROM memories
                WHERE id IN (
                    SELECT id FROM memories
                    ORDER BY importance ASC, access_count ASC
                    LIMIT ?
                )
            """, (evict_count,))
            await self._db.commit()

    def _row_to_entry(self, row: tuple) -> MemoryEntry:
        """Convert database row to MemoryEntry."""
        return MemoryEntry(
            id=row[0],
            type=MemoryType(row[1]),
            content=json.loads(row[2]),
            timestamp=datetime.fromisoformat(row[3]),
            importance=row[4],
            embedding=json.loads(row[5]) if row[5] else None,
            tags=json.loads(row[6]) if row[6] else [],
            related_ids=json.loads(row[7]) if row[7] else [],
            pnl=row[8],
            success=bool(row[9]) if row[9] is not None else None,
        )

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if len(a) != len(b):
            return 0.0

        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)
