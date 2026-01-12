"""Short-term memory - recent observations and context."""

from collections import deque
from datetime import datetime, timedelta
from typing import Any

from .base import Memory, MemoryEntry, MemoryType


class ShortTermMemory(Memory):
    """
    Short-term memory for recent observations.

    - Fixed-size sliding window
    - Fast access to recent context
    - Automatic expiration
    """

    def __init__(
        self,
        capacity: int = 50,
        ttl_minutes: int = 60,
    ):
        self.capacity = capacity
        self.ttl = timedelta(minutes=ttl_minutes)
        self._entries: deque[MemoryEntry] = deque(maxlen=capacity)

    async def store(self, entry: MemoryEntry) -> None:
        """Store entry in short-term memory."""
        self._entries.append(entry)
        self._cleanup_expired()

    async def retrieve(
        self,
        query: str | None = None,
        type_filter: MemoryType | None = None,
        limit: int = 10,
        min_importance: float = 0.0,
    ) -> list[MemoryEntry]:
        """Retrieve recent memories."""
        self._cleanup_expired()

        results = []
        for entry in reversed(self._entries):
            if type_filter and entry.type != type_filter:
                continue
            if entry.importance < min_importance:
                continue
            results.append(entry)
            if len(results) >= limit:
                break

        return results

    async def search_similar(
        self,
        embedding: list[float],
        limit: int = 5,
    ) -> list[MemoryEntry]:
        """Short-term doesn't support semantic search."""
        return []

    async def clear(self) -> None:
        """Clear all short-term memories."""
        self._entries.clear()

    async def get_stats(self) -> dict[str, Any]:
        """Get memory stats."""
        self._cleanup_expired()
        return {
            "type": "short_term",
            "capacity": self.capacity,
            "current_size": len(self._entries),
            "utilization": len(self._entries) / self.capacity,
        }

    def _cleanup_expired(self) -> None:
        """Remove expired entries."""
        cutoff = datetime.utcnow() - self.ttl
        while self._entries and self._entries[0].timestamp < cutoff:
            self._entries.popleft()

    async def get_recent_context(self, n: int = 10) -> str:
        """Get recent context as text for LLM."""
        entries = await self.retrieve(limit=n)
        return "\n\n".join(e.to_text() for e in entries)

    async def get_recent_trades(self, n: int = 5) -> list[MemoryEntry]:
        """Get recent trade memories."""
        return await self.retrieve(type_filter=MemoryType.TRADE, limit=n)

    async def get_recent_decisions(self, n: int = 5) -> list[MemoryEntry]:
        """Get recent decision memories."""
        return await self.retrieve(type_filter=MemoryType.DECISION, limit=n)
