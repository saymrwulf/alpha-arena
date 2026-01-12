"""LLM provider registry for plug-and-play model switching."""

import asyncio
import logging
import os
import time
from typing import Type

from ..core.config import Config, LLMProviderConfig
from .base import LLMProvider, LLMResponse, Message
from .anthropic import AnthropicProvider
from .openai import OpenAIProvider
from .xai import XAIProvider
from .google import GoogleProvider
from .local import LocalProvider

logger = logging.getLogger(__name__)


PROVIDER_CLASSES: dict[str, Type[LLMProvider]] = {
    "anthropic": AnthropicProvider,
    "openai": OpenAIProvider,
    "xai": XAIProvider,
    "google": GoogleProvider,
    "local": LocalProvider,
}

# Default fallback order - local last as the ultimate fallback
DEFAULT_FALLBACK_ORDER = ["anthropic", "openai", "google", "xai", "local"]


class ProviderHealth:
    """Track health status of a provider."""

    def __init__(self, name: str):
        self.name = name
        self.is_healthy = True
        self.last_check: float = 0
        self.last_success: float = 0
        self.consecutive_failures = 0
        self.total_requests = 0
        self.total_failures = 0
        self.last_error: str | None = None
        self.latency_ms: int = 0

    def record_success(self, latency_ms: int):
        """Record a successful request."""
        self.is_healthy = True
        self.last_success = time.time()
        self.last_check = time.time()
        self.consecutive_failures = 0
        self.total_requests += 1
        self.latency_ms = latency_ms
        self.last_error = None

    def record_failure(self, error: str):
        """Record a failed request."""
        self.last_check = time.time()
        self.consecutive_failures += 1
        self.total_requests += 1
        self.total_failures += 1
        self.last_error = error
        # Mark unhealthy after 3 consecutive failures
        if self.consecutive_failures >= 3:
            self.is_healthy = False

    def to_dict(self) -> dict:
        """Convert to dictionary for API response."""
        return {
            "name": self.name,
            "is_healthy": self.is_healthy,
            "last_check": self.last_check,
            "last_success": self.last_success,
            "consecutive_failures": self.consecutive_failures,
            "total_requests": self.total_requests,
            "total_failures": self.total_failures,
            "last_error": self.last_error,
            "latency_ms": self.latency_ms,
            "success_rate": (
                (self.total_requests - self.total_failures) / self.total_requests * 100
                if self.total_requests > 0 else 100.0
            ),
        }


class ProviderRegistry:
    """
    Registry for managing multiple LLM providers with fallback support.

    Enables plug-and-play switching between:
    - Anthropic (Claude)
    - OpenAI (GPT-4o, o1)
    - xAI (Grok - best for X sentiment)
    - Google (Gemini)
    - Local (DeepSeek, Qwen, Llama via Ollama/LM Studio/vLLM)

    Features:
    - Automatic fallback on provider failure
    - Health monitoring for each provider
    - Hot-swapping of active provider
    """

    def __init__(self, config: Config):
        self.config = config
        self._providers: dict[str, LLMProvider] = {}
        self._connected: set[str] = set()
        self._health: dict[str, ProviderHealth] = {}
        self._active_provider: str | None = None
        self._fallback_order: list[str] = DEFAULT_FALLBACK_ORDER.copy()

    async def initialize(self) -> None:
        """Initialize all enabled providers."""
        for name, provider_config in self.config.llm_providers.items():
            if not provider_config.enabled:
                continue

            provider = self._create_provider(name, provider_config)
            if provider:
                self._providers[name] = provider
                self._health[name] = ProviderHealth(name)

        # Connect all providers
        for name, provider in self._providers.items():
            try:
                await provider.connect()
                self._connected.add(name)
                self._health[name].record_success(0)
                logger.info(f"Connected to {name} provider")
            except Exception as e:
                self._health[name].record_failure(str(e))
                logger.warning(f"Failed to connect {name}: {e}")

        # Set initial active provider based on fallback order
        for name in self._fallback_order:
            if name in self._connected and self._health[name].is_healthy:
                self._active_provider = name
                logger.info(f"Active provider set to: {name}")
                break

    async def shutdown(self) -> None:
        """Disconnect all providers."""
        for name in self._connected:
            try:
                await self._providers[name].disconnect()
            except Exception:
                pass
        self._connected.clear()

    def _create_provider(
        self,
        name: str,
        config: LLMProviderConfig,
    ) -> LLMProvider | None:
        """Create a provider instance from config."""
        provider_class = PROVIDER_CLASSES.get(name)
        if not provider_class:
            print(f"Warning: Unknown provider {name}")
            return None

        # Get API key from environment
        api_key = os.environ.get(config.api_key_env, "") if config.api_key_env else ""

        # Local models don't need API key
        if name == "local" and not api_key:
            api_key = "not-needed"

        if not api_key and name != "local":
            print(f"Warning: No API key for {name} ({config.api_key_env})")
            return None

        return provider_class(
            api_key=api_key,
            default_model=config.default_model,
            base_url=config.base_url,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )

    def get(self, name: str) -> LLMProvider | None:
        """Get a specific provider."""
        if name not in self._connected:
            return None
        return self._providers.get(name)

    def get_for_model(self, model: str) -> tuple[LLMProvider, str] | None:
        """
        Get the appropriate provider for a model.

        Returns (provider, model_name) tuple or None if not available.
        """
        # Check each provider's model list
        for name, config in self.config.llm_providers.items():
            if model in config.models and name in self._connected:
                return self._providers[name], model

        # Try to infer provider from model name
        model_lower = model.lower()
        if "claude" in model_lower:
            provider = self.get("anthropic")
            if provider:
                return provider, model
        elif "gpt" in model_lower or "o1" in model_lower:
            provider = self.get("openai")
            if provider:
                return provider, model
        elif "grok" in model_lower:
            provider = self.get("xai")
            if provider:
                return provider, model
        elif "gemini" in model_lower:
            provider = self.get("google")
            if provider:
                return provider, model
        elif any(x in model_lower for x in ["deepseek", "qwen", "llama", "mistral", "kimi"]):
            provider = self.get("local")
            if provider:
                return provider, model

        return None

    @property
    def available_providers(self) -> list[str]:
        """List of connected provider names."""
        return list(self._connected)

    @property
    def available_models(self) -> dict[str, list[str]]:
        """Map of provider -> available models."""
        result = {}
        for name in self._connected:
            config = self.config.llm_providers.get(name)
            if config:
                result[name] = config.models
        return result

    @property
    def active_provider(self) -> str | None:
        """Get currently active provider name."""
        return self._active_provider

    @property
    def fallback_order(self) -> list[str]:
        """Get current fallback order."""
        return self._fallback_order.copy()

    def set_active_provider(self, name: str) -> bool:
        """
        Set the active provider.

        Returns True if successful, False if provider not available.
        """
        if name not in self._connected:
            logger.warning(f"Cannot set active provider to {name}: not connected")
            return False
        self._active_provider = name
        logger.info(f"Active provider changed to: {name}")
        return True

    def set_fallback_order(self, order: list[str]) -> None:
        """Set custom fallback order."""
        self._fallback_order = order
        logger.info(f"Fallback order set to: {order}")

    def get_active(self) -> LLMProvider | None:
        """Get the currently active provider."""
        if self._active_provider:
            return self._providers.get(self._active_provider)
        return None

    def _get_next_healthy_provider(self, exclude: set[str] | None = None) -> str | None:
        """Get next healthy provider from fallback order."""
        exclude = exclude or set()
        for name in self._fallback_order:
            if name in exclude:
                continue
            if name in self._connected and name in self._health:
                if self._health[name].is_healthy:
                    return name
        return None

    async def complete_with_fallback(
        self,
        messages: list[Message],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        system: str | None = None,
        json_mode: bool = False,
    ) -> LLMResponse:
        """
        Complete with automatic fallback on failure.

        Tries the active provider first, then falls back through
        the fallback chain until one succeeds.
        """
        tried: set[str] = set()
        last_error: Exception | None = None

        # Start with active provider
        current = self._active_provider

        while current:
            if current in tried:
                current = self._get_next_healthy_provider(tried)
                continue

            tried.add(current)
            provider = self._providers.get(current)
            health = self._health.get(current)

            if not provider or not health:
                current = self._get_next_healthy_provider(tried)
                continue

            try:
                # Use provider's default model if none specified
                provider_config = self.config.llm_providers.get(current)
                use_model = model or (provider_config.default_model if provider_config else None)

                start_time = time.time()
                response = await provider.complete(
                    messages=messages,
                    model=use_model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    system=system,
                    json_mode=json_mode,
                )
                latency_ms = int((time.time() - start_time) * 1000)
                health.record_success(latency_ms)

                # Add metadata about which provider was used
                if response.metadata is None:
                    response.metadata = {}
                response.metadata["fallback_provider"] = current
                response.metadata["tried_providers"] = list(tried)

                return response

            except Exception as e:
                last_error = e
                health.record_failure(str(e))
                logger.warning(f"Provider {current} failed: {e}, trying fallback...")
                current = self._get_next_healthy_provider(tried)

        # All providers failed
        raise RuntimeError(
            f"All providers failed. Last error: {last_error}. "
            f"Tried: {list(tried)}"
        )

    async def check_health(self, provider_name: str | None = None) -> dict[str, ProviderHealth]:
        """
        Check health of providers.

        If provider_name is specified, only check that provider.
        Otherwise, check all connected providers.
        """
        providers_to_check = (
            [provider_name] if provider_name else list(self._connected)
        )

        for name in providers_to_check:
            if name not in self._connected:
                continue

            provider = self._providers.get(name)
            health = self._health.get(name)
            if not provider or not health:
                continue

            try:
                # Simple health check - try a minimal completion
                start_time = time.time()
                await provider.complete(
                    messages=[Message(role="user", content="Hi")],
                    max_tokens=5,
                )
                latency_ms = int((time.time() - start_time) * 1000)
                health.record_success(latency_ms)
            except Exception as e:
                health.record_failure(str(e))

        return self._health

    def get_health_status(self) -> dict[str, dict]:
        """Get health status for all providers."""
        return {name: health.to_dict() for name, health in self._health.items()}

    def get_provider_info(self) -> dict:
        """Get comprehensive provider information for API."""
        return {
            "active_provider": self._active_provider,
            "fallback_order": self._fallback_order,
            "connected_providers": list(self._connected),
            "available_models": self.available_models,
            "health": self.get_health_status(),
        }


def get_provider(
    config: Config,
    provider_name: str,
) -> LLMProvider | None:
    """
    Quick helper to get a single provider.

    For full multi-provider support, use ProviderRegistry.
    """
    provider_config = config.llm_providers.get(provider_name)
    if not provider_config or not provider_config.enabled:
        return None

    provider_class = PROVIDER_CLASSES.get(provider_name)
    if not provider_class:
        return None

    api_key = os.environ.get(provider_config.api_key_env, "")
    if not api_key and provider_name != "local":
        return None

    return provider_class(
        api_key=api_key or "not-needed",
        default_model=provider_config.default_model,
        base_url=provider_config.base_url,
        temperature=provider_config.temperature,
        max_tokens=provider_config.max_tokens,
    )
