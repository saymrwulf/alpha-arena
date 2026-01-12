"""Tests for memory system."""

import pytest
import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
import tempfile
import uuid

from src.memory.short_term import ShortTermMemory
from src.memory.long_term import LongTermMemory
from src.memory.base import MemoryEntry, MemoryType


class TestShortTermMemory:
    """Tests for short-term memory."""

    @pytest.fixture
    def memory(self):
        """Create short-term memory."""
        return ShortTermMemory(capacity=10, ttl_minutes=60)

    @pytest.mark.asyncio
    async def test_store_and_retrieve(self, memory):
        """Test storing and retrieving entries."""
        entry = MemoryEntry(
            id=str(uuid.uuid4()),
            type=MemoryType.OBSERVATION,
            content={"price": 0.50, "market": "test"},
            timestamp=datetime.utcnow(),
            importance=0.5,
        )

        await memory.store(entry)

        results = await memory.retrieve(limit=10)
        assert len(results) == 1
        assert results[0].content["price"] == 0.50

    @pytest.mark.asyncio
    async def test_capacity_limit(self, memory):
        """Test capacity limit enforcement."""
        for i in range(15):
            entry = MemoryEntry(
                id=str(uuid.uuid4()),
                type=MemoryType.OBSERVATION,
                content={"index": i},
                timestamp=datetime.utcnow(),
                importance=0.5,
            )
            await memory.store(entry)

        results = await memory.retrieve(limit=20)
        # Should only have 10 (capacity limit)
        assert len(results) <= 10

    @pytest.mark.asyncio
    async def test_type_filter(self, memory):
        """Test filtering by type."""
        # Store different types
        entry1 = MemoryEntry(
            id=str(uuid.uuid4()),
            type=MemoryType.TRADE,
            content={"trade": "buy"},
            timestamp=datetime.utcnow(),
            importance=0.5,
        )
        entry2 = MemoryEntry(
            id=str(uuid.uuid4()),
            type=MemoryType.OBSERVATION,
            content={"price": 0.50},
            timestamp=datetime.utcnow(),
            importance=0.5,
        )

        await memory.store(entry1)
        await memory.store(entry2)

        trades = await memory.retrieve(type_filter=MemoryType.TRADE, limit=10)
        assert len(trades) == 1
        assert trades[0].type == MemoryType.TRADE

    @pytest.mark.asyncio
    async def test_clear(self, memory):
        """Test clearing memory."""
        entry = MemoryEntry(
            id=str(uuid.uuid4()),
            type=MemoryType.OBSERVATION,
            content={"test": 1},
            timestamp=datetime.utcnow(),
            importance=0.5,
        )
        await memory.store(entry)

        await memory.clear()

        results = await memory.retrieve(limit=10)
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_get_stats(self, memory):
        """Test getting stats."""
        entry = MemoryEntry(
            id=str(uuid.uuid4()),
            type=MemoryType.OBSERVATION,
            content={"test": 1},
            timestamp=datetime.utcnow(),
            importance=0.5,
        )
        await memory.store(entry)

        stats = await memory.get_stats()

        assert stats["type"] == "short_term"
        assert stats["capacity"] == 10
        assert stats["current_size"] == 1


class TestLongTermMemory:
    """Tests for long-term memory (SQLite-based)."""

    @pytest.fixture
    def memory(self):
        """Create long-term memory with temp database."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        memory = LongTermMemory(db_path=db_path)
        yield memory

        # Cleanup
        Path(db_path).unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_store_and_retrieve(self, memory):
        """Test storing and retrieving memories."""
        await memory.connect()

        entry = MemoryEntry(
            id=str(uuid.uuid4()),
            type=MemoryType.OBSERVATION,
            content={"price": 50000, "asset": "BTC"},
            timestamp=datetime.utcnow(),
            importance=0.7,
        )

        await memory.store(entry)

        results = await memory.retrieve(limit=5)
        assert len(results) == 1
        assert results[0].content["asset"] == "BTC"

        await memory.disconnect()

    @pytest.mark.asyncio
    async def test_text_search(self, memory):
        """Test text search in content."""
        await memory.connect()

        entry1 = MemoryEntry(
            id=str(uuid.uuid4()),
            type=MemoryType.OBSERVATION,
            content={"note": "Bitcoin price increased"},
            timestamp=datetime.utcnow(),
            importance=0.5,
        )
        entry2 = MemoryEntry(
            id=str(uuid.uuid4()),
            type=MemoryType.OBSERVATION,
            content={"note": "Ethereum rally"},
            timestamp=datetime.utcnow(),
            importance=0.5,
        )

        await memory.store(entry1)
        await memory.store(entry2)

        # Search for Bitcoin
        results = await memory.retrieve(query="Bitcoin", limit=10)
        assert len(results) == 1
        assert "Bitcoin" in results[0].content["note"]

        await memory.disconnect()

    @pytest.mark.asyncio
    async def test_clear(self, memory):
        """Test clearing memories."""
        await memory.connect()

        entry = MemoryEntry(
            id=str(uuid.uuid4()),
            type=MemoryType.OBSERVATION,
            content={"test": 1},
            timestamp=datetime.utcnow(),
            importance=0.5,
        )
        await memory.store(entry)

        await memory.clear()

        results = await memory.retrieve(limit=10)
        assert len(results) == 0

        await memory.disconnect()

    @pytest.mark.asyncio
    async def test_get_stats(self, memory):
        """Test getting stats."""
        await memory.connect()

        entry = MemoryEntry(
            id=str(uuid.uuid4()),
            type=MemoryType.OBSERVATION,
            content={"test": 1},
            timestamp=datetime.utcnow(),
            importance=0.5,
        )
        await memory.store(entry)

        stats = await memory.get_stats()

        assert stats["type"] == "long_term"
        assert stats["total_entries"] == 1

        await memory.disconnect()


class TestMemoryEntry:
    """Tests for MemoryEntry dataclass."""

    def test_entry_creation(self):
        """Test creating a memory entry."""
        entry = MemoryEntry(
            id="test_001",
            type=MemoryType.OBSERVATION,
            content={"price": 0.50, "volume": 1000},
            timestamp=datetime.utcnow(),
            importance=0.7,
        )

        assert entry.id == "test_001"
        assert entry.type == MemoryType.OBSERVATION
        assert entry.content["price"] == 0.50
        assert entry.importance == 0.7

    def test_entry_with_tags(self):
        """Test entry with tags."""
        entry = MemoryEntry(
            id="test_002",
            type=MemoryType.LESSON,
            content={"lesson": "Don't chase momentum"},
            timestamp=datetime.utcnow(),
            importance=0.9,
            tags=["trading", "risk"],
        )

        assert "trading" in entry.tags
        assert "risk" in entry.tags

    def test_entry_to_text(self):
        """Test entry text serialization."""
        entry = MemoryEntry(
            id="test_003",
            type=MemoryType.DECISION,
            content={"action": "buy", "market": "ABC"},
            timestamp=datetime.utcnow(),
            importance=0.8,
        )

        text = entry.to_text()
        assert isinstance(text, str)
        # Should contain relevant info
        assert "DECISION" in text or "decision" in text.lower()


class TestMemoryType:
    """Tests for MemoryType enum."""

    def test_memory_types(self):
        """Test memory type values."""
        assert MemoryType.OBSERVATION.value == "observation"
        assert MemoryType.DECISION.value == "decision"
        assert MemoryType.TRADE.value == "trade"
        assert MemoryType.OUTCOME.value == "outcome"
        assert MemoryType.PATTERN.value == "pattern"
        assert MemoryType.LESSON.value == "lesson"
