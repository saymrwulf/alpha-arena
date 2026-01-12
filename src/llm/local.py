"""Local LLM provider via Ollama, LM Studio, or vLLM."""

import time
from typing import AsyncIterator

import httpx
import openai

from .base import LLMProvider, LLMResponse, Message, Role, estimate_cost


class LocalProvider(LLMProvider):
    """
    Local model provider supporting multiple backends:
    - Ollama (default port 11434)
    - LM Studio (default port 1234)
    - vLLM (custom port)
    - Any OpenAI-compatible local server
    """

    provider_name = "local"

    # Known local server configurations
    BACKENDS = {
        "ollama": {"port": 11434, "health_path": "/api/tags"},
        "lmstudio": {"port": 1234, "health_path": "/v1/models"},
        "vllm": {"port": 8000, "health_path": "/v1/models"},
    }

    def __init__(
        self,
        api_key: str = "not-needed",
        default_model: str = "deepseek-r1:70b",
        base_url: str = "http://localhost:11434/v1",
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ):
        super().__init__(api_key, default_model, base_url, temperature, max_tokens)
        self._backend: str = "ollama"
        self._available_models: list[str] = []

    def _detect_backend(self) -> str:
        """Detect which backend is being used based on URL."""
        if not self.base_url:
            return "ollama"

        url_lower = self.base_url.lower()
        if "11434" in url_lower:
            return "ollama"
        elif "1234" in url_lower:
            return "lmstudio"
        elif "8000" in url_lower:
            return "vllm"
        else:
            # Assume OpenAI-compatible server
            return "openai_compatible"

    async def connect(self) -> None:
        """Initialize local client."""
        # Detect backend from URL
        self._backend = self._detect_backend()

        # Use OpenAI-compatible client (works with all backends)
        self._client = openai.AsyncOpenAI(
            api_key=self.api_key or "not-needed",
            base_url=self.base_url,
        )

        # Verify connection and get available models
        await self._check_connection()

    async def _check_connection(self) -> None:
        """Verify the local server is running and list models."""
        async with httpx.AsyncClient() as client:
            if self._backend == "ollama":
                # Ollama-specific health check
                try:
                    health_url = self.base_url.replace("/v1", "") + "/api/tags"
                    resp = await client.get(health_url, timeout=5.0)
                    if resp.status_code != 200:
                        raise ConnectionError("Ollama not responding")
                    # Get available models
                    data = resp.json()
                    self._available_models = [m["name"] for m in data.get("models", [])]
                except httpx.ConnectError:
                    raise ConnectionError(
                        f"Cannot connect to Ollama at {self.base_url}. "
                        "Make sure Ollama is running: ollama serve"
                    )
            elif self._backend == "lmstudio":
                # LM Studio uses OpenAI-compatible /v1/models endpoint
                try:
                    models_url = self.base_url.rstrip("/")
                    if not models_url.endswith("/models"):
                        models_url = models_url.rstrip("/v1") + "/v1/models"
                    resp = await client.get(models_url, timeout=5.0)
                    if resp.status_code != 200:
                        raise ConnectionError("LM Studio not responding")
                    # Get available models
                    data = resp.json()
                    self._available_models = [m["id"] for m in data.get("data", [])]
                except httpx.ConnectError:
                    raise ConnectionError(
                        f"Cannot connect to LM Studio at {self.base_url}. "
                        "Make sure LM Studio is running with 'Start Server' enabled."
                    )
            else:
                # Generic OpenAI-compatible check
                try:
                    models_url = self.base_url.rstrip("/") + "/models"
                    resp = await client.get(models_url, timeout=5.0)
                    if resp.status_code == 200:
                        data = resp.json()
                        self._available_models = [m["id"] for m in data.get("data", [])]
                except httpx.ConnectError:
                    raise ConnectionError(
                        f"Cannot connect to local server at {self.base_url}"
                    )

    async def disconnect(self) -> None:
        """Cleanup client."""
        self._client = None

    async def list_models(self) -> list[str]:
        """List available local models."""
        async with httpx.AsyncClient() as client:
            if self._backend == "ollama":
                resp = await client.get(
                    self.base_url.replace("/v1", "") + "/api/tags",
                    timeout=10.0,
                )
                data = resp.json()
                return [m["name"] for m in data.get("models", [])]
            else:
                # OpenAI-compatible /v1/models endpoint (LM Studio, vLLM, etc.)
                models_url = self.base_url.rstrip("/")
                if not models_url.endswith("/models"):
                    models_url = models_url.rstrip("/v1") + "/v1/models"
                resp = await client.get(models_url, timeout=10.0)
                if resp.status_code == 200:
                    data = resp.json()
                    return [m["id"] for m in data.get("data", [])]
        return self._available_models

    @property
    def available_models(self) -> list[str]:
        """Return cached list of available models."""
        return self._available_models

    @property
    def backend(self) -> str:
        """Return detected backend type."""
        return self._backend

    async def pull_model(self, model: str) -> None:
        """Pull a model (Ollama only)."""
        if self._backend != "ollama":
            raise NotImplementedError("Model pulling only supported for Ollama")

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                self.base_url.replace("/v1", "") + "/api/pull",
                json={"name": model},
                timeout=600.0,  # Models can take a while to download
            )
            resp.raise_for_status()

    async def complete(
        self,
        messages: list[Message],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        system: str | None = None,
        json_mode: bool = False,
    ) -> LLMResponse:
        """Generate completion using local model."""
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
            model=model,
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
            metadata={"backend": self._backend},
        )

    async def stream(
        self,
        messages: list[Message],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        system: str | None = None,
    ) -> AsyncIterator[str]:
        """Stream completion from local model."""
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
