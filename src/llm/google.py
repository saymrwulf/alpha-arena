"""Google Gemini LLM provider."""

import time
from typing import AsyncIterator

import google.generativeai as genai
from google.generativeai.types import GenerationConfig

from .base import LLMProvider, LLMResponse, Message, Role, estimate_cost


class GoogleProvider(LLMProvider):
    """Google Gemini provider."""

    provider_name = "google"

    def __init__(
        self,
        api_key: str,
        default_model: str = "gemini-2.0-flash",
        base_url: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ):
        super().__init__(api_key, default_model, base_url, temperature, max_tokens)
        self._model = None

    async def connect(self) -> None:
        """Initialize Gemini client."""
        genai.configure(api_key=self.api_key)
        self._client = genai

    async def disconnect(self) -> None:
        """Cleanup client."""
        self._client = None
        self._model = None

    def _get_model(self, model_name: str):
        """Get or create a GenerativeModel instance."""
        return genai.GenerativeModel(model_name)

    async def complete(
        self,
        messages: list[Message],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        system: str | None = None,
        json_mode: bool = False,
    ) -> LLMResponse:
        """Generate completion using Gemini."""
        if self._client is None:
            raise RuntimeError("Provider not connected")

        model_name = model or self.default_model
        temperature = temperature if temperature is not None else self.temperature
        max_tokens = max_tokens or self.max_tokens

        # Build content for Gemini
        gemini_model = self._get_model(model_name)

        # Gemini uses a different message format
        # System instruction is passed separately
        system_instruction = system or ""

        # Convert messages to Gemini format
        contents = []
        for msg in messages:
            if msg.role == Role.SYSTEM:
                system_instruction = msg.content + "\n" + system_instruction
            elif msg.role == Role.USER:
                contents.append({"role": "user", "parts": [msg.content]})
            elif msg.role == Role.ASSISTANT:
                contents.append({"role": "model", "parts": [msg.content]})

        # Configure generation
        generation_config = GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

        if json_mode:
            generation_config.response_mime_type = "application/json"

        start_time = time.time()

        # Create model with system instruction if provided
        if system_instruction.strip():
            gemini_model = genai.GenerativeModel(
                model_name,
                system_instruction=system_instruction.strip(),
            )

        response = await gemini_model.generate_content_async(
            contents,
            generation_config=generation_config,
        )

        latency_ms = int((time.time() - start_time) * 1000)

        # Extract token counts from usage metadata
        input_tokens = 0
        output_tokens = 0
        if hasattr(response, 'usage_metadata'):
            input_tokens = getattr(response.usage_metadata, 'prompt_token_count', 0)
            output_tokens = getattr(response.usage_metadata, 'candidates_token_count', 0)

        return LLMResponse(
            content=response.text,
            model=model_name,
            provider=self.provider_name,
            tokens_input=input_tokens,
            tokens_output=output_tokens,
            latency_ms=latency_ms,
            cost_estimate=estimate_cost(
                self.provider_name,
                model_name,
                input_tokens,
                output_tokens,
            ),
            finish_reason=response.candidates[0].finish_reason.name if response.candidates else "stop",
        )

    async def stream(
        self,
        messages: list[Message],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        system: str | None = None,
    ) -> AsyncIterator[str]:
        """Stream completion from Gemini."""
        if self._client is None:
            raise RuntimeError("Provider not connected")

        model_name = model or self.default_model
        temperature = temperature if temperature is not None else self.temperature
        max_tokens = max_tokens or self.max_tokens

        gemini_model = self._get_model(model_name)

        # Build system instruction
        system_instruction = system or ""

        # Convert messages
        contents = []
        for msg in messages:
            if msg.role == Role.SYSTEM:
                system_instruction = msg.content + "\n" + system_instruction
            elif msg.role == Role.USER:
                contents.append({"role": "user", "parts": [msg.content]})
            elif msg.role == Role.ASSISTANT:
                contents.append({"role": "model", "parts": [msg.content]})

        generation_config = GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

        # Create model with system instruction
        if system_instruction.strip():
            gemini_model = genai.GenerativeModel(
                model_name,
                system_instruction=system_instruction.strip(),
            )

        response = await gemini_model.generate_content_async(
            contents,
            generation_config=generation_config,
            stream=True,
        )

        async for chunk in response:
            if chunk.text:
                yield chunk.text
