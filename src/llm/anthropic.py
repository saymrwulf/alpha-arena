"""Anthropic (Claude) LLM provider."""

import time
from typing import AsyncIterator

import anthropic

from .base import LLMProvider, LLMResponse, Message, Role, estimate_cost


class AnthropicProvider(LLMProvider):
    """Anthropic Claude provider."""

    provider_name = "anthropic"

    async def connect(self) -> None:
        """Initialize Anthropic client."""
        self._client = anthropic.AsyncAnthropic(api_key=self.api_key)

    async def disconnect(self) -> None:
        """Cleanup client."""
        self._client = None

    async def complete(
        self,
        messages: list[Message],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        system: str | None = None,
        json_mode: bool = False,
    ) -> LLMResponse:
        """Generate completion using Claude."""
        if self._client is None:
            raise RuntimeError("Provider not connected")

        model = model or self.default_model
        temperature = temperature if temperature is not None else self.temperature
        max_tokens = max_tokens or self.max_tokens

        # Separate system message
        api_messages = []
        system_content = system or ""

        for msg in messages:
            if msg.role == Role.SYSTEM:
                system_content = msg.content + "\n" + system_content
            else:
                api_messages.append({
                    "role": msg.role.value,
                    "content": msg.content,
                })

        start_time = time.time()

        response = await self._client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_content.strip() if system_content else None,
            messages=api_messages,
        )

        latency_ms = int((time.time() - start_time) * 1000)

        return LLMResponse(
            content=response.content[0].text,
            model=response.model,
            provider=self.provider_name,
            tokens_input=response.usage.input_tokens,
            tokens_output=response.usage.output_tokens,
            latency_ms=latency_ms,
            cost_estimate=estimate_cost(
                self.provider_name,
                model,
                response.usage.input_tokens,
                response.usage.output_tokens,
            ),
            finish_reason=response.stop_reason or "stop",
        )

    async def stream(
        self,
        messages: list[Message],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        system: str | None = None,
    ) -> AsyncIterator[str]:
        """Stream completion from Claude."""
        if self._client is None:
            raise RuntimeError("Provider not connected")

        model = model or self.default_model
        temperature = temperature if temperature is not None else self.temperature
        max_tokens = max_tokens or self.max_tokens

        api_messages = []
        system_content = system or ""

        for msg in messages:
            if msg.role == Role.SYSTEM:
                system_content = msg.content + "\n" + system_content
            else:
                api_messages.append({
                    "role": msg.role.value,
                    "content": msg.content,
                })

        async with self._client.messages.stream(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_content.strip() if system_content else None,
            messages=api_messages,
        ) as stream:
            async for text in stream.text_stream:
                yield text
