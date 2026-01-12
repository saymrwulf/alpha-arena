"""Reflection Agent - Post-trade analysis and learning."""

import json
from decimal import Decimal
from typing import Any

from ..memory.episodic import Episode
from ..memory.manager import MemoryManager
from .base import BaseAgent, AgentRole, AgentResponse


class ReflectionAgent(BaseAgent):
    """
    Reflection Agent: Analyzes past trades and extracts lessons.

    Responsibilities:
    - Analyze completed trade episodes
    - Identify what worked and what didn't
    - Extract actionable lessons
    - Update long-term memory with patterns
    - Suggest strategy adjustments
    """

    role = AgentRole.REFLECTION
    name = "reflection"

    @property
    def system_prompt(self) -> str:
        return """You are a Reflection Agent in an autonomous trading system for prediction markets.

Your role is CRITICAL for long-term improvement: Learn from past trades.

REFLECTION FRAMEWORK:
1. OUTCOME ANALYSIS
   - Was the trade profitable? By how much?
   - Did it match our expectations?
   - What was the actual vs predicted edge?

2. PROCESS ANALYSIS
   - Was the original thesis correct?
   - Was the timing right?
   - Was position sizing appropriate?
   - Were risk controls followed?

3. COUNTERFACTUAL ANALYSIS
   - What would have happened with different sizing?
   - What if we entered/exited at different times?
   - What signals did we miss?

4. LESSON EXTRACTION
   - What patterns led to success?
   - What patterns led to failure?
   - What should we do differently next time?

OUTPUT FORMAT (JSON):
{
    "episode_analysis": {
        "episode_id": "...",
        "outcome_assessment": "success|partial_success|failure|unlucky|lucky",
        "pnl_vs_expected": "better|as_expected|worse",
        "thesis_accuracy": "correct|partially_correct|incorrect",
        "timing_assessment": "optimal|acceptable|poor",
        "sizing_assessment": "appropriate|too_small|too_large"
    },
    "what_worked": [
        "Specific thing that worked well..."
    ],
    "what_failed": [
        "Specific thing that didn't work..."
    ],
    "counterfactual": "What could have been done differently...",
    "lessons": [
        {
            "lesson": "Concise, actionable lesson",
            "category": "entry|exit|sizing|timing|selection",
            "importance": "high|medium|low",
            "applies_to": "all_trades|specific_market_type|specific_condition"
        }
    ],
    "pattern_detected": {
        "description": "Pattern observed...",
        "conditions": {"key": "value"},
        "outcome": "positive|negative",
        "confidence": 0.XX,
        "sample_size": X
    },
    "strategy_adjustments": [
        "Specific adjustment to make..."
    ]
}

BE BRUTALLY HONEST. Accurate self-assessment is essential for improvement.
Distinguish between skill and luck."""

    async def analyze(
        self,
        markets: list,  # Not used in reflection
        context: str,
        other_responses: list[AgentResponse] | None = None,
    ) -> AgentResponse:
        """
        Reflect on recent trading episodes.

        This agent is typically called with episode data in context.
        """
        prompt = f"""Analyze this trading episode and extract lessons:

{context}

Provide thorough reflection including:
1. Honest assessment of outcome (skill vs luck)
2. What worked and what didn't
3. Counterfactual analysis
4. Actionable lessons for future trades
5. Any patterns detected

Return your analysis as JSON."""

        response = await self._call_llm(prompt, json_mode=True)
        analysis = self._parse_reflection(response.content)

        lessons = analysis.get("lessons", [])
        recommendations = []
        for lesson in lessons:
            if lesson.get("importance") == "high":
                recommendations.append(f"LESSON: {lesson.get('lesson', '')}")

        for adjustment in analysis.get("strategy_adjustments", []):
            recommendations.append(f"ADJUST: {adjustment}")

        warnings = analysis.get("what_failed", [])

        return AgentResponse(
            agent_role=self.role,
            agent_model=self.provider.default_model,
            content=analysis,
            confidence=Decimal("0.7"),
            reasoning=analysis.get("counterfactual", ""),
            recommendations=recommendations,
            warnings=warnings[:3],
            latency_ms=response.latency_ms,
            tokens_used=response.total_tokens,
        )

    def _parse_reflection(self, content: str) -> dict[str, Any]:
        """Parse reflection response."""
        try:
            if "```" in content:
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            return json.loads(content)
        except json.JSONDecodeError:
            return {
                "lessons": [],
                "what_worked": [],
                "what_failed": [],
                "counterfactual": content[:500],
            }

    async def reflect_on_episode(
        self,
        episode: Episode,
        memory: MemoryManager,
    ) -> dict[str, Any]:
        """
        Perform full reflection on a completed episode.

        Updates memory with lessons and patterns.
        """
        # Build episode context
        context = f"""
EPISODE DETAILS:
- Market: {episode.market_question}
- Signal Type: {episode.signal_type}
- Signal Confidence: {episode.signal_confidence:.0%}
- Signal Reasoning: {episode.signal_reasoning}

EXECUTION:
- Entry Price: ${episode.entry_price}
- Exit Price: ${episode.exit_price}
- Size: ${episode.size}
- Hold Duration: {episode.hold_duration_hours:.1f} hours

OUTCOME:
- PnL: ${episode.pnl} ({episode.pnl_percent:+.1f}%)
- Result: {episode.outcome.upper()}

MARKET CONTEXT AT ENTRY:
{json.dumps(episode.market_context, indent=2, default=str)}

AGENTS INVOLVED: {', '.join(episode.agents_involved)}
"""

        # Get memory context
        memory_context = await memory.get_performance_context()
        context += f"\n{memory_context}"

        # Perform reflection
        response = await self.analyze([], context)
        analysis = response.content

        # Store reflection in memory
        await memory.store_reflection(
            episode_id=episode.id,
            what_worked=analysis.get("what_worked", []),
            what_failed=analysis.get("what_failed", []),
            lessons=[l.get("lesson", "") for l in analysis.get("lessons", [])],
            counterfactual=analysis.get("counterfactual", ""),
        )

        # Store high-importance lessons
        for lesson in analysis.get("lessons", []):
            if lesson.get("importance") in ("high", "medium"):
                await memory.store_lesson(
                    lesson=lesson.get("lesson", ""),
                    source_episodes=[episode.id],
                    category=lesson.get("category", "general"),
                    importance=0.9 if lesson.get("importance") == "high" else 0.7,
                )

        # Store detected pattern
        if pattern := analysis.get("pattern_detected"):
            if pattern.get("confidence", 0) >= 0.6:
                await memory.store_pattern(
                    description=pattern.get("description", ""),
                    conditions=pattern.get("conditions", {}),
                    outcome=pattern.get("outcome", ""),
                    confidence=pattern.get("confidence", 0),
                    sample_size=pattern.get("sample_size", 1),
                )

        return analysis

    async def batch_reflect(
        self,
        episodes: list[Episode],
        memory: MemoryManager,
    ) -> dict[str, Any]:
        """
        Reflect on multiple episodes to find aggregate patterns.
        """
        if not episodes:
            return {"patterns": [], "lessons": []}

        # Build batch context
        episode_summaries = []
        for ep in episodes:
            episode_summaries.append({
                "id": ep.id,
                "market": ep.market_question[:50],
                "signal_type": ep.signal_type,
                "confidence": ep.signal_confidence,
                "pnl": float(ep.pnl) if ep.pnl else 0,
                "outcome": ep.outcome,
                "hold_hours": ep.hold_duration_hours,
            })

        prompt = f"""Analyze these {len(episodes)} trading episodes for aggregate patterns:

EPISODES:
{json.dumps(episode_summaries, indent=2)}

PERFORMANCE CONTEXT:
{await memory.get_performance_context()}

Identify:
1. Common patterns in winning vs losing trades
2. Systematic errors being made
3. Conditions that predict success
4. Strategy-level adjustments needed

Return analysis as JSON with:
{{
    "aggregate_patterns": [...],
    "systematic_errors": [...],
    "success_predictors": [...],
    "strategy_recommendations": [...]
}}"""

        response = await self._call_llm(prompt, json_mode=True)
        return self._parse_reflection(response.content)
