"""Base LLM provider interface."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, AsyncIterator


class Role(str, Enum):
    """Message roles."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass
class Message:
    """Chat message."""
    role: Role
    content: str
    name: str | None = None  # For multi-agent identification
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMResponse:
    """Response from LLM call."""
    content: str
    model: str
    provider: str
    tokens_input: int
    tokens_output: int
    latency_ms: int
    cost_estimate: Decimal
    finish_reason: str = "stop"
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.tokens_input + self.tokens_output


# Token costs per 1M tokens (as of Jan 2025)
TOKEN_COSTS: dict[str, dict[str, dict[str, Decimal]]] = {
    "anthropic": {
        "claude-sonnet-4-20250514": {"input": Decimal("3.00"), "output": Decimal("15.00")},
        "claude-3-5-haiku-20241022": {"input": Decimal("0.80"), "output": Decimal("4.00")},
        "claude-opus-4-20250514": {"input": Decimal("15.00"), "output": Decimal("75.00")},
    },
    "openai": {
        "gpt-4o": {"input": Decimal("2.50"), "output": Decimal("10.00")},
        "gpt-4o-mini": {"input": Decimal("0.15"), "output": Decimal("0.60")},
        "o1-preview": {"input": Decimal("15.00"), "output": Decimal("60.00")},
        "o1-mini": {"input": Decimal("3.00"), "output": Decimal("12.00")},
    },
    "xai": {
        "grok-3": {"input": Decimal("3.00"), "output": Decimal("15.00")},
        "grok-3-mini": {"input": Decimal("0.30"), "output": Decimal("0.50")},
    },
    "local": {
        # Local models are free
        "deepseek-r1:70b": {"input": Decimal("0"), "output": Decimal("0")},
        "qwen2.5:72b": {"input": Decimal("0"), "output": Decimal("0")},
        "llama3.3:70b": {"input": Decimal("0"), "output": Decimal("0")},
    },
}


def estimate_cost(
    provider: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
) -> Decimal:
    """Estimate cost for a request."""
    costs = TOKEN_COSTS.get(provider, {}).get(model)
    if not costs:
        return Decimal("0")

    input_cost = (Decimal(str(input_tokens)) / Decimal("1000000")) * costs["input"]
    output_cost = (Decimal(str(output_tokens)) / Decimal("1000000")) * costs["output"]
    return (input_cost + output_cost).quantize(Decimal("0.000001"))


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    provider_name: str = "base"

    def __init__(
        self,
        api_key: str,
        default_model: str,
        base_url: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ):
        self.api_key = api_key
        self.default_model = default_model
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client: Any = None

    @abstractmethod
    async def connect(self) -> None:
        """Initialize the client."""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Cleanup the client."""
        pass

    @abstractmethod
    async def complete(
        self,
        messages: list[Message],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        system: str | None = None,
        json_mode: bool = False,
    ) -> LLMResponse:
        """Generate a completion."""
        pass

    @abstractmethod
    async def stream(
        self,
        messages: list[Message],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        system: str | None = None,
    ) -> AsyncIterator[str]:
        """Stream a completion."""
        pass

    def _format_messages(self, messages: list[Message]) -> list[dict[str, str]]:
        """Format messages for API call."""
        return [
            {"role": msg.role.value, "content": msg.content}
            for msg in messages
        ]
