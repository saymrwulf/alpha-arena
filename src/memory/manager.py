"""Memory manager - coordinates all memory subsystems."""

import uuid
from datetime import datetime
from decimal import Decimal
from typing import Any

from ..core.config import MemoryConfig
from .base import MemoryEntry, MemoryType
from .short_term import ShortTermMemory
from .long_term import LongTermMemory
from .episodic import EpisodicMemory, Episode


class MemoryManager:
    """
    Unified memory manager coordinating:
    - Short-term: Recent context
    - Long-term: Patterns and lessons
    - Episodic: Complete trade lifecycles
    """

    def __init__(self, config: MemoryConfig):
        self.config = config
        self.short_term = ShortTermMemory(
            capacity=config.short_term_window,
        )
        self.long_term = LongTermMemory(
            capacity=config.long_term_capacity,
        )
        self.episodic = EpisodicMemory()

    async def connect(self) -> None:
        """Initialize all memory systems."""
        await self.long_term.connect()
        await self.episodic.connect()

    async def disconnect(self) -> None:
        """Cleanup all memory systems."""
        await self.long_term.disconnect()
        await self.episodic.disconnect()

    # ============ Observation Memory ============

    async def remember_observation(
        self,
        market_id: str,
        observation: dict[str, Any],
        importance: float = 0.3,
    ) -> str:
        """Store a market observation."""
        entry = MemoryEntry(
            id=str(uuid.uuid4()),
            type=MemoryType.OBSERVATION,
            content={
                "market_id": market_id,
                **observation,
            },
            timestamp=datetime.utcnow(),
            importance=importance,
        )

        await self.short_term.store(entry)
        return entry.id

    # ============ Decision Memory ============

    async def remember_decision(
        self,
        market_id: str,
        signal_type: str,
        reasoning: str,
        confidence: float,
        agents: list[str],
    ) -> str:
        """Store a trading decision."""
        entry = MemoryEntry(
            id=str(uuid.uuid4()),
            type=MemoryType.DECISION,
            content={
                "market_id": market_id,
                "signal_type": signal_type,
                "reasoning": reasoning,
                "confidence": confidence,
                "agents": agents,
            },
            timestamp=datetime.utcnow(),
            importance=confidence,
            tags=[signal_type],
        )

        await self.short_term.store(entry)
        return entry.id

    # ============ Trade Memory ============

    async def remember_trade(
        self,
        trade_id: str,
        market_id: str,
        market_question: str,
        side: str,
        size: Decimal,
        price: Decimal,
        signal_id: str | None = None,
    ) -> str:
        """Store an executed trade."""
        entry = MemoryEntry(
            id=trade_id,
            type=MemoryType.TRADE,
            content={
                "market_id": market_id,
                "market_question": market_question,
                "side": side,
                "size": str(size),
                "price": str(price),
                "signal_id": signal_id,
            },
            timestamp=datetime.utcnow(),
            importance=0.6,
            tags=[side],
        )

        await self.short_term.store(entry)
        await self.long_term.store(entry)
        return entry.id

    # ============ Episode Management ============

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
        """Start tracking a new trading episode."""
        return await self.episodic.start_episode(
            market_id=market_id,
            market_question=market_question,
            signal_type=signal_type,
            signal_confidence=signal_confidence,
            signal_reasoning=signal_reasoning,
            agents_involved=agents_involved,
            market_context=market_context,
        )

    async def record_episode_entry(
        self,
        episode_id: str,
        entry_price: Decimal,
        size: Decimal,
    ) -> None:
        """Record trade entry for episode."""
        await self.episodic.record_entry(episode_id, entry_price, size)

    async def close_episode(
        self,
        episode_id: str,
        exit_price: Decimal,
        pnl: Decimal,
    ) -> Episode:
        """Close episode and return for reflection."""
        episode = await self.episodic.close_episode(episode_id, exit_price, pnl)

        # Also store as outcome in long-term memory
        await self.long_term.store(episode.to_memory_entry())

        return episode

    # ============ Pattern & Lesson Storage ============

    async def store_pattern(
        self,
        description: str,
        conditions: dict[str, Any],
        outcome: str,
        confidence: float,
        sample_size: int,
    ) -> str:
        """Store a learned pattern."""
        entry = MemoryEntry(
            id=str(uuid.uuid4()),
            type=MemoryType.PATTERN,
            content={
                "description": description,
                "conditions": conditions,
                "outcome": outcome,
                "confidence": confidence,
                "sample_size": sample_size,
            },
            timestamp=datetime.utcnow(),
            importance=confidence,
            tags=["pattern", outcome],
        )

        await self.long_term.store(entry)
        return entry.id

    async def store_lesson(
        self,
        lesson: str,
        source_episodes: list[str],
        category: str,
        importance: float = 0.7,
    ) -> str:
        """Store an extracted lesson."""
        entry = MemoryEntry(
            id=str(uuid.uuid4()),
            type=MemoryType.LESSON,
            content={
                "lesson": lesson,
                "category": category,
            },
            timestamp=datetime.utcnow(),
            importance=importance,
            tags=["lesson", category],
            related_ids=source_episodes,
        )

        await self.long_term.store(entry)
        return entry.id

    async def store_reflection(
        self,
        episode_id: str,
        what_worked: list[str],
        what_failed: list[str],
        lessons: list[str],
        counterfactual: str,
    ) -> None:
        """Store reflection for an episode."""
        await self.episodic.add_reflection(
            episode_id=episode_id,
            what_worked=what_worked,
            what_failed=what_failed,
            lessons=lessons,
            counterfactual=counterfactual,
        )

        # Also store as reflection entry
        entry = MemoryEntry(
            id=str(uuid.uuid4()),
            type=MemoryType.REFLECTION,
            content={
                "episode_id": episode_id,
                "what_worked": what_worked,
                "what_failed": what_failed,
                "lessons": lessons,
                "counterfactual": counterfactual,
                "summary": f"Reflection on episode. Lessons: {'; '.join(lessons[:2])}",
            },
            timestamp=datetime.utcnow(),
            importance=0.8,
            tags=["reflection"],
            related_ids=[episode_id],
        )

        await self.long_term.store(entry)

    # ============ Retrieval ============

    async def get_context_for_decision(self, market_id: str) -> str:
        """Get relevant context for making a decision."""
        parts = []

        # Recent observations
        recent = await self.short_term.get_recent_context(n=5)
        if recent:
            parts.append("=== Recent Activity ===")
            parts.append(recent)

        # Relevant patterns
        patterns = await self.long_term.get_patterns(min_confidence=0.6)
        if patterns:
            parts.append("\n=== Known Patterns ===")
            for p in patterns[:5]:
                parts.append(p.to_text())

        # Lessons learned
        lessons = await self.long_term.get_lessons(limit=5)
        if lessons:
            parts.append("\n=== Lessons Learned ===")
            for lesson in lessons:
                parts.append(f"- {lesson.content.get('lesson', '')}")

        # Recent similar outcomes
        similar_trades = await self.long_term.retrieve(
            query=market_id,
            type_filter=MemoryType.OUTCOME,
            limit=3,
        )
        if similar_trades:
            parts.append("\n=== Similar Past Trades ===")
            for trade in similar_trades:
                parts.append(trade.to_text())

        return "\n".join(parts)

    async def get_episodes_for_reflection(self) -> list[Episode]:
        """Get episodes that need reflection."""
        return await self.episodic.get_unreflected_episodes(limit=5)

    async def get_performance_context(self) -> str:
        """Get performance context for agent awareness."""
        stats = await self.episodic.get_stats()

        total = stats.get("total", 0)
        wins = stats.get("win_count", 0)
        losses = stats.get("loss_count", 0)
        total_pnl = stats.get("win_total_pnl", 0) + stats.get("loss_total_pnl", 0)
        win_rate = wins / total if total > 0 else 0

        return f"""=== Trading Performance ===
Total Trades: {total}
Win Rate: {win_rate:.1%}
Wins: {wins}, Losses: {losses}
Total PnL: ${total_pnl:.2f}
Avg Hold Time: {stats.get('avg_hold_hours', 0):.1f} hours
"""

    async def get_stats(self) -> dict[str, Any]:
        """Get combined memory stats."""
        return {
            "short_term": await self.short_term.get_stats(),
            "long_term": await self.long_term.get_stats(),
            "episodic": await self.episodic.get_stats(),
        }
