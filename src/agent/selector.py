"""Agent/LLM selection framework for comparing different configurations."""

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any

from .base import AgentConfig, AgentDecision, AgentObservation
from .llm import LLMAgent


@dataclass
class ModelMetrics:
    """Metrics for a single model evaluation."""

    model: str
    provider: str
    latency_ms: int
    tokens_used: int
    cost_estimate: Decimal
    signal_count: int
    avg_confidence: Decimal
    reasoning_quality: int  # 1-5 scale
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ModelComparison:
    """Comparison results across multiple models."""

    observation: AgentObservation
    results: dict[str, tuple[AgentDecision, ModelMetrics]]
    recommended_model: str
    recommendation_reasoning: str


# Approximate costs per 1K tokens (as of late 2024)
TOKEN_COSTS = {
    "anthropic": {
        "claude-sonnet-4-20250514": {"input": Decimal("0.003"), "output": Decimal("0.015")},
        "claude-3-5-haiku-20241022": {"input": Decimal("0.001"), "output": Decimal("0.005")},
    },
    "openai": {
        "gpt-4o": {"input": Decimal("0.005"), "output": Decimal("0.015")},
        "gpt-4o-mini": {"input": Decimal("0.00015"), "output": Decimal("0.0006")},
    },
}


class AgentSelector:
    """
    Framework for testing and comparing different LLM agents.

    Supports evaluation modes:
    - Online vs Offline: Does the model need live data or can it work with curated snapshots?
    - Fast vs Slow: Latency comparison for time-sensitive decisions
    - Cost vs Quality: Token usage and cost estimation
    """

    def __init__(self, api_keys: dict[str, str]):
        """
        Initialize with API keys for each provider.

        api_keys: {"anthropic": "sk-...", "openai": "sk-..."}
        """
        self.api_keys = api_keys
        self._agents: dict[str, LLMAgent] = {}
        self._metrics_history: list[ModelMetrics] = []

    async def connect(self) -> None:
        """Initialize all agents."""
        for agent in self._agents.values():
            await agent.connect()

    async def disconnect(self) -> None:
        """Cleanup all agents."""
        for agent in self._agents.values():
            await agent.disconnect()
        self._agents.clear()

    def add_model(
        self,
        name: str,
        provider: str,
        model: str,
        temperature: float = 0.3,
    ) -> None:
        """Add a model configuration to test."""
        config = AgentConfig(
            name=name,
            llm_provider=provider,
            llm_model=model,
            temperature=temperature,
        )

        api_key = self.api_keys.get(provider)
        if not api_key:
            raise ValueError(f"No API key for provider: {provider}")

        self._agents[name] = LLMAgent(config, api_key)

    async def compare_models(
        self,
        observation: AgentObservation,
        models: list[str] | None = None,
    ) -> ModelComparison:
        """
        Run the same observation through multiple models and compare results.

        If models is None, runs all configured models.
        """
        models_to_test = models or list(self._agents.keys())
        results: dict[str, tuple[AgentDecision, ModelMetrics]] = {}

        # Run models concurrently
        tasks = []
        for model_name in models_to_test:
            if model_name in self._agents:
                tasks.append(self._evaluate_model(model_name, observation))

        completed = await asyncio.gather(*tasks, return_exceptions=True)

        for model_name, result in zip(models_to_test, completed):
            if isinstance(result, Exception):
                print(f"Model {model_name} failed: {result}")
                continue
            results[model_name] = result

        # Determine recommendation
        recommended, reasoning = self._select_best_model(results)

        return ModelComparison(
            observation=observation,
            results=results,
            recommended_model=recommended,
            recommendation_reasoning=reasoning,
        )

    async def _evaluate_model(
        self,
        model_name: str,
        observation: AgentObservation,
    ) -> tuple[AgentDecision, ModelMetrics]:
        """Evaluate a single model on the observation."""
        agent = self._agents[model_name]

        start_time = time.time()
        decision = await agent.decide(observation)
        total_latency = int((time.time() - start_time) * 1000)

        # Calculate metrics
        config = agent.config
        cost = self._estimate_cost(
            config.llm_provider,
            config.llm_model,
            decision.tokens_used,
        )

        avg_confidence = Decimal("0")
        if decision.signals:
            avg_confidence = sum(s.confidence for s in decision.signals) / len(decision.signals)

        # Simple reasoning quality heuristic
        reasoning_quality = self._score_reasoning(decision.reasoning)

        metrics = ModelMetrics(
            model=config.llm_model,
            provider=config.llm_provider,
            latency_ms=total_latency,
            tokens_used=decision.tokens_used,
            cost_estimate=cost,
            signal_count=len(decision.signals),
            avg_confidence=avg_confidence,
            reasoning_quality=reasoning_quality,
        )

        self._metrics_history.append(metrics)
        return decision, metrics

    def _estimate_cost(self, provider: str, model: str, tokens: int) -> Decimal:
        """Estimate cost for token usage."""
        costs = TOKEN_COSTS.get(provider, {}).get(model)
        if not costs:
            return Decimal("0")

        # Rough estimate: assume 70% input, 30% output tokens
        input_tokens = int(tokens * 0.7)
        output_tokens = int(tokens * 0.3)

        cost = (
            (Decimal(str(input_tokens)) / 1000) * costs["input"] +
            (Decimal(str(output_tokens)) / 1000) * costs["output"]
        )
        return cost.quantize(Decimal("0.0001"))

    def _score_reasoning(self, reasoning: str) -> int:
        """Score reasoning quality 1-5."""
        if not reasoning:
            return 1

        score = 2  # Base score for having reasoning

        # Check for key elements
        if len(reasoning) > 100:
            score += 1
        if any(word in reasoning.lower() for word in ["because", "therefore", "since", "due to"]):
            score += 1
        if any(word in reasoning.lower() for word in ["risk", "confidence", "edge", "probability"]):
            score += 1

        return min(score, 5)

    def _select_best_model(
        self,
        results: dict[str, tuple[AgentDecision, ModelMetrics]],
    ) -> tuple[str, str]:
        """Select the best model based on metrics."""
        if not results:
            return "", "No models evaluated"

        # Scoring weights
        weights = {
            "latency": -0.2,  # Lower is better
            "cost": -0.3,  # Lower is better
            "confidence": 0.3,  # Higher is better
            "reasoning": 0.2,  # Higher is better
        }

        scores: dict[str, float] = {}
        reasons: dict[str, list[str]] = {}

        # Normalize metrics
        all_latencies = [m.latency_ms for _, m in results.values()]
        all_costs = [float(m.cost_estimate) for _, m in results.values()]
        max_latency = max(all_latencies) or 1
        max_cost = max(all_costs) or 0.01

        for model_name, (decision, metrics) in results.items():
            reasons[model_name] = []

            # Normalize to 0-1 scale
            norm_latency = metrics.latency_ms / max_latency
            norm_cost = float(metrics.cost_estimate) / max_cost
            norm_confidence = float(metrics.avg_confidence)
            norm_reasoning = metrics.reasoning_quality / 5

            score = (
                weights["latency"] * norm_latency +
                weights["cost"] * norm_cost +
                weights["confidence"] * norm_confidence +
                weights["reasoning"] * norm_reasoning
            )

            scores[model_name] = score

            # Build reasoning
            if norm_latency < 0.5:
                reasons[model_name].append("fast response")
            if norm_cost < 0.5:
                reasons[model_name].append("cost-effective")
            if norm_confidence > 0.6:
                reasons[model_name].append("high confidence signals")
            if norm_reasoning > 0.6:
                reasons[model_name].append("quality reasoning")

        # Select best
        best_model = max(scores, key=lambda k: scores[k])
        reasoning = f"Selected {best_model}: " + ", ".join(reasons[best_model]) if reasons[best_model] else f"Best overall score"

        return best_model, reasoning

    def get_historical_metrics(self) -> list[ModelMetrics]:
        """Get all historical metrics."""
        return self._metrics_history.copy()

    def get_model_summary(self) -> dict[str, dict[str, Any]]:
        """Get summary statistics per model."""
        from collections import defaultdict

        stats: dict[str, dict[str, list]] = defaultdict(lambda: {
            "latencies": [],
            "costs": [],
            "confidences": [],
            "reasoning_scores": [],
        })

        for m in self._metrics_history:
            key = f"{m.provider}:{m.model}"
            stats[key]["latencies"].append(m.latency_ms)
            stats[key]["costs"].append(float(m.cost_estimate))
            stats[key]["confidences"].append(float(m.avg_confidence))
            stats[key]["reasoning_scores"].append(m.reasoning_quality)

        summary = {}
        for key, data in stats.items():
            summary[key] = {
                "avg_latency_ms": sum(data["latencies"]) / len(data["latencies"]),
                "avg_cost": sum(data["costs"]) / len(data["costs"]),
                "avg_confidence": sum(data["confidences"]) / len(data["confidences"]),
                "avg_reasoning_score": sum(data["reasoning_scores"]) / len(data["reasoning_scores"]),
                "sample_count": len(data["latencies"]),
            }

        return summary


def recommended_config() -> AgentConfig:
    """
    Return the recommended agent configuration.

    Based on tradeoffs:
    - Claude Sonnet: Best balance of quality/cost for trading decisions
    - Temperature 0.3: Low randomness for consistent decisions
    - Reasonable latency for 60s loop interval
    """
    return AgentConfig(
        name="default",
        loop_interval_seconds=60,
        llm_provider="anthropic",
        llm_model="claude-sonnet-4-20250514",
        temperature=0.3,
        max_tokens=4096,
    )
