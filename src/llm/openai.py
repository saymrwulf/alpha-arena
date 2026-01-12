"""OpenAI LLM provider."""

import time
from typing import AsyncIterator

import openai

from .base import LLMProvider, LLMResponse, Message, Role, estimate_cost


class OpenAIProvider(LLMProvider):
    """OpenAI GPT provider."""

    provider_name = "openai"

    async def connect(self) -> None:
        """Initialize OpenAI client."""
        kwargs = {"api_key": self.api_key}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        self._client = openai.AsyncOpenAI(**kwargs)

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
        """Generate completion using GPT."""
        if self._client is None:
            raise RuntimeError("Provider not connected")

        model = model or self.default_model
        temperature = temperature if temperature is not None else self.temperature
        max_tokens = max_tokens or self.max_tokens

        # Build messages with system
        api_messages = []
        if system:
            api_messages.append({"role": "system", "content": system})

        for msg in messages:
            api_messages.append({
                "role": msg.role.value,
                "content": msg.content,
            })

        start_time = time.time()

        kwargs = {
            "model": model,
            "messages": api_messages,
            "max_tokens": max_tokens,
        }

        # o1 models don't support temperature
        if not model.startswith("o1"):
            kwargs["temperature"] = temperature

        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        response = await self._client.chat.completions.create(**kwargs)

        latency_ms = int((time.time() - start_time) * 1000)

        usage = response.usage
        input_tokens = usage.prompt_tokens if usage else 0
        output_tokens = usage.completion_tokens if usage else 0

        return LLMResponse(
            content=response.choices[0].message.content or "",
            model=response.model,
            provider=self.provider_name,
            tokens_input=input_tokens,
            tokens_output=output_tokens,
            latency_ms=latency_ms,
            cost_estimate=estimate_cost(
                self.provider_name,
                model,
                input_tokens,
                output_tokens,
            ),
            finish_reason=response.choices[0].finish_reason or "stop",
        )

    async def stream(
        self,
        messages: list[Message],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        system: str | None = None,
    ) -> AsyncIterator[str]:
        """Stream completion from GPT."""
        if self._client is None:
            raise RuntimeError("Provider not connected")

        model = model or self.default_model
        temperature = temperature if temperature is not None else self.temperature
        max_tokens = max_tokens or self.max_tokens

        api_messages = []
        if system:
            api_messages.append({"role": "system", "content": system})

        for msg in messages:
            api_messages.append({
                "role": msg.role.value,
                "content": msg.content,
            })

        kwargs = {
            "model": model,
            "messages": api_messages,
            "max_tokens": max_tokens,
            "stream": True,
        }

        if not model.startswith("o1"):
            kwargs["temperature"] = temperature

        stream = await self._client.chat.completions.create(**kwargs)

        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
