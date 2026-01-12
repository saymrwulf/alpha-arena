"""Base memory interface and types."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any


class MemoryType(str, Enum):
    """Types of memory entries."""
    OBSERVATION = "observation"  # Market state snapshots
    DECISION = "decision"  # Trading decisions made
    TRADE = "trade"  # Executed trades
    OUTCOME = "outcome"  # Trade results
    REFLECTION = "reflection"  # Post-trade analysis
    PATTERN = "pattern"  # Learned patterns
    LESSON = "lesson"  # Extracted lessons


@dataclass
class MemoryEntry:
    """Single memory entry."""
    id: str
    type: MemoryType
    content: dict[str, Any]
    timestamp: datetime
    importance: float = 0.5  # 0-1, used for retrieval priority
    embedding: list[float] | None = None  # For semantic search
    tags: list[str] = field(default_factory=list)
    related_ids: list[str] = field(default_factory=list)

    # For outcome tracking
    pnl: Decimal | None = None
    success: bool | None = None

    def to_text(self) -> str:
        """Convert to text for LLM context."""
        lines = [f"[{self.type.value}] {self.timestamp.isoformat()}"]

        if self.type == MemoryType.TRADE:
            c = self.content
            lines.append(f"  {c.get('side', '?')} {c.get('size', '?')} @ {c.get('price', '?')}")
            lines.append(f"  Market: {c.get('market_question', '?')[:50]}")
            if self.pnl is not None:
                lines.append(f"  PnL: ${self.pnl}")

        elif self.type == MemoryType.REFLECTION:
            lines.append(f"  {self.content.get('summary', '')}")
            if lessons := self.content.get('lessons', []):
                lines.append(f"  Lessons: {', '.join(lessons[:3])}")

        elif self.type == MemoryType.PATTERN:
            lines.append(f"  Pattern: {self.content.get('description', '')}")
            lines.append(f"  Confidence: {self.content.get('confidence', 0):.0%}")

        else:
            # Generic format
            for k, v in list(self.content.items())[:5]:
                lines.append(f"  {k}: {str(v)[:100]}")

        return "\n".join(lines)


class Memory(ABC):
    """Abstract memory interface."""

    @abstractmethod
    async def store(self, entry: MemoryEntry) -> None:
        """Store a memory entry."""
        pass

    @abstractmethod
    async def retrieve(
        self,
        query: str | None = None,
        type_filter: MemoryType | None = None,
        limit: int = 10,
        min_importance: float = 0.0,
    ) -> list[MemoryEntry]:
        """Retrieve relevant memories."""
        pass

    @abstractmethod
    async def search_similar(
        self,
        embedding: list[float],
        limit: int = 5,
    ) -> list[MemoryEntry]:
        """Find semantically similar memories."""
        pass

    @abstractmethod
    async def clear(self) -> None:
        """Clear all memories."""
        pass

    @abstractmethod
    async def get_stats(self) -> dict[str, Any]:
        """Get memory statistics."""
        pass
