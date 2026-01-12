"""xAI (Grok) LLM provider - best for real-time X/Twitter sentiment."""

import time
from typing import AsyncIterator

import openai

from .base import LLMProvider, LLMResponse, Message, Role, estimate_cost


class XAIProvider(LLMProvider):
    """xAI Grok provider - has native X/Twitter knowledge."""

    provider_name = "xai"
    XAI_BASE_URL = "https://api.x.ai/v1"

    async def connect(self) -> None:
        """Initialize xAI client (OpenAI-compatible API)."""
        self._client = openai.AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url or self.XAI_BASE_URL,
        )

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
        """Generate completion using Grok."""
        if self._client is None:
            raise RuntimeError("Provider not connected")

        model = model or self.default_model
        temperature = temperature if temperature is not None else self.temperature
        max_tokens = max_tokens or self.max_tokens

        # Build messages
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
            "temperature": temperature,
        }

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
        """Stream completion from Grok."""
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

        stream = await self._client.chat.completions.create(
            model=model,
            messages=api_messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
        )

        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    async def get_realtime_sentiment(
        self,
        topic: str,
        context: str = "",
    ) -> dict:
        """
        Leverage Grok's native X knowledge for real-time sentiment.

        Grok has direct access to X/Twitter data, making it uniquely suited
        for sentiment analysis without external API calls.
        """
        system = """You are analyzing real-time sentiment on X (Twitter) about a topic.
You have access to current X discussions and trends.

Provide your analysis as JSON:
{
    "sentiment_score": float (-1 to 1),
    "sentiment_label": "very_negative|negative|neutral|positive|very_positive",
    "confidence": float (0 to 1),
    "volume": "low|medium|high|viral",
    "trending": boolean,
    "key_narratives": ["narrative1", "narrative2"],
    "influential_voices": ["@user1", "@user2"],
    "recent_shift": "becoming_more_positive|stable|becoming_more_negative",
    "summary": "Brief summary of current sentiment"
}"""

        prompt = f"Analyze current X/Twitter sentiment about: {topic}"
        if context:
            prompt += f"\n\nContext: {context}"

        response = await self.complete(
            messages=[Message(role=Role.USER, content=prompt)],
            system=system,
            json_mode=True,
        )

        import json
        try:
            return json.loads(response.content)
        except json.JSONDecodeError:
            return {
                "sentiment_score": 0,
                "sentiment_label": "neutral",
                "confidence": 0,
                "error": "Failed to parse response",
            }
