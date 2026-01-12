"""Tests for LLM provider implementations."""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from decimal import Decimal

from src.llm.base import LLMProvider, LLMResponse, Message, Role
from src.llm.anthropic import AnthropicProvider
from src.llm.openai import OpenAIProvider
from src.llm.google import GoogleProvider
from src.llm.local import LocalProvider
from src.llm.registry import ProviderRegistry, ProviderHealth, DEFAULT_FALLBACK_ORDER
from src.core.config import Config, LLMProviderConfig


# =============================================================================
# Base Provider Tests
# =============================================================================

class TestMessage:
    """Tests for Message model."""

    def test_message_creation(self):
        msg = Message(role=Role.USER, content="Hello")
        assert msg.role == Role.USER
        assert msg.content == "Hello"

    def test_message_roles(self):
        for role in [Role.USER, Role.ASSISTANT, Role.SYSTEM]:
            msg = Message(role=role, content="test")
            assert msg.role == role


class TestLLMResponse:
    """Tests for LLMResponse model."""

    def test_response_creation(self):
        response = LLMResponse(
            content="Hello world",
            model="test-model",
            provider="test",
            tokens_input=10,
            tokens_output=5,
            latency_ms=100,
            cost_estimate=Decimal("0.001"),
            finish_reason="stop",
        )
        assert response.content == "Hello world"
        assert response.model == "test-model"
        assert response.tokens_input == 10

    def test_response_metadata(self):
        response = LLMResponse(
            content="test",
            model="test",
            provider="test",
            tokens_input=0,
            tokens_output=0,
            latency_ms=0,
            cost_estimate=Decimal("0"),
            finish_reason="stop",
            metadata={"key": "value"},
        )
        assert response.metadata["key"] == "value"


# =============================================================================
# Google Provider Tests
# =============================================================================

class TestGoogleProvider:
    """Tests for Google/Gemini provider."""

    @pytest.fixture
    def provider(self):
        return GoogleProvider(
            api_key="test-key",
            default_model="gemini-2.0-flash",
        )

    def test_provider_name(self, provider):
        assert provider.provider_name == "google"

    def test_default_model(self, provider):
        assert provider.default_model == "gemini-2.0-flash"

    @pytest.mark.asyncio
    async def test_connect(self, provider):
        with patch("google.generativeai.configure") as mock_configure:
            await provider.connect()
            mock_configure.assert_called_once_with(api_key="test-key")
            assert provider._client is not None

    @pytest.mark.asyncio
    async def test_disconnect(self, provider):
        provider._client = MagicMock()
        await provider.disconnect()
        assert provider._client is None

    @pytest.mark.asyncio
    async def test_complete_not_connected(self, provider):
        with pytest.raises(RuntimeError, match="Provider not connected"):
            await provider.complete([Message(role=Role.USER, content="test")])


# =============================================================================
# Local Provider Tests
# =============================================================================

class TestLocalProvider:
    """Tests for local model provider (Ollama/LM Studio/vLLM)."""

    @pytest.fixture
    def ollama_provider(self):
        return LocalProvider(
            base_url="http://localhost:11434/v1",
            default_model="deepseek-r1:70b",
        )

    @pytest.fixture
    def lmstudio_provider(self):
        return LocalProvider(
            base_url="http://localhost:1234/v1",
            default_model="kimi-k2",
        )

    @pytest.fixture
    def vllm_provider(self):
        return LocalProvider(
            base_url="http://localhost:8000/v1",
            default_model="llama3.3:70b",
        )

    def test_provider_name(self, ollama_provider):
        assert ollama_provider.provider_name == "local"

    def test_detect_ollama_backend(self, ollama_provider):
        backend = ollama_provider._detect_backend()
        assert backend == "ollama"

    def test_detect_lmstudio_backend(self, lmstudio_provider):
        backend = lmstudio_provider._detect_backend()
        assert backend == "lmstudio"

    def test_detect_vllm_backend(self, vllm_provider):
        backend = vllm_provider._detect_backend()
        assert backend == "vllm"

    def test_detect_custom_backend(self):
        provider = LocalProvider(base_url="http://custom-server:9999/v1")
        backend = provider._detect_backend()
        assert backend == "openai_compatible"

    @pytest.mark.asyncio
    async def test_connect_ollama(self, ollama_provider):
        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"models": [{"name": "deepseek-r1:70b"}]}

            mock_client_instance = AsyncMock()
            mock_client_instance.get = AsyncMock(return_value=mock_response)
            mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
            mock_client_instance.__aexit__ = AsyncMock()
            mock_client.return_value = mock_client_instance

            await ollama_provider.connect()
            assert ollama_provider._backend == "ollama"
            assert "deepseek-r1:70b" in ollama_provider._available_models

    def test_available_models_property(self, ollama_provider):
        ollama_provider._available_models = ["model1", "model2"]
        assert ollama_provider.available_models == ["model1", "model2"]

    def test_backend_property(self, ollama_provider):
        ollama_provider._backend = "ollama"
        assert ollama_provider.backend == "ollama"


# =============================================================================
# Provider Health Tests
# =============================================================================

class TestProviderHealth:
    """Tests for ProviderHealth tracking."""

    def test_initial_state(self):
        health = ProviderHealth("test")
        assert health.name == "test"
        assert health.is_healthy is True
        assert health.consecutive_failures == 0
        assert health.total_requests == 0

    def test_record_success(self):
        health = ProviderHealth("test")
        health.record_success(100)
        assert health.is_healthy is True
        assert health.total_requests == 1
        assert health.latency_ms == 100
        assert health.consecutive_failures == 0

    def test_record_failure(self):
        health = ProviderHealth("test")
        health.record_failure("Connection error")
        assert health.consecutive_failures == 1
        assert health.total_failures == 1
        assert health.last_error == "Connection error"
        # Still healthy after 1 failure
        assert health.is_healthy is True

    def test_unhealthy_after_three_failures(self):
        health = ProviderHealth("test")
        for _ in range(3):
            health.record_failure("Error")
        assert health.is_healthy is False
        assert health.consecutive_failures == 3

    def test_recovery_after_success(self):
        health = ProviderHealth("test")
        for _ in range(3):
            health.record_failure("Error")
        assert health.is_healthy is False

        health.record_success(50)
        assert health.is_healthy is True
        assert health.consecutive_failures == 0

    def test_to_dict(self):
        health = ProviderHealth("test")
        health.record_success(100)
        d = health.to_dict()
        assert d["name"] == "test"
        assert d["is_healthy"] is True
        assert d["latency_ms"] == 100
        assert d["success_rate"] == 100.0

    def test_success_rate_calculation(self):
        health = ProviderHealth("test")
        health.record_success(100)
        health.record_success(100)
        health.record_failure("Error")
        # 2 successes, 1 failure = 66.67% success rate
        d = health.to_dict()
        assert abs(d["success_rate"] - 66.67) < 1


# =============================================================================
# Provider Registry Tests
# =============================================================================

class TestProviderRegistry:
    """Tests for ProviderRegistry."""

    @pytest.fixture
    def config(self):
        return Config(
            llm_providers={
                "anthropic": LLMProviderConfig(
                    enabled=True,
                    models=["claude-sonnet-4-20250514"],
                    default_model="claude-sonnet-4-20250514",
                    api_key_env="ANTHROPIC_API_KEY",
                ),
                "openai": LLMProviderConfig(
                    enabled=True,
                    models=["gpt-4o"],
                    default_model="gpt-4o",
                    api_key_env="OPENAI_API_KEY",
                ),
                "google": LLMProviderConfig(
                    enabled=True,
                    models=["gemini-2.0-flash"],
                    default_model="gemini-2.0-flash",
                    api_key_env="GOOGLE_API_KEY",
                ),
                "local": LLMProviderConfig(
                    enabled=False,
                    models=["deepseek-r1:70b"],
                    default_model="deepseek-r1:70b",
                    base_url="http://localhost:11434/v1",
                ),
            }
        )

    @pytest.fixture
    def registry(self, config):
        return ProviderRegistry(config)

    def test_default_fallback_order(self):
        assert DEFAULT_FALLBACK_ORDER == ["anthropic", "openai", "google", "xai", "local"]

    def test_registry_initialization(self, registry):
        assert registry._active_provider is None
        assert len(registry._providers) == 0
        assert registry._fallback_order == DEFAULT_FALLBACK_ORDER

    def test_set_fallback_order(self, registry):
        new_order = ["local", "google", "anthropic"]
        registry.set_fallback_order(new_order)
        assert registry._fallback_order == new_order

    def test_available_providers_empty_initially(self, registry):
        assert registry.available_providers == []

    def test_set_active_provider_fails_when_not_connected(self, registry):
        result = registry.set_active_provider("anthropic")
        assert result is False

    def test_get_returns_none_when_not_connected(self, registry):
        provider = registry.get("anthropic")
        assert provider is None

    def test_get_for_model_claude(self, registry, config):
        # Need to manually set up connected state for this test
        registry._connected.add("anthropic")
        registry._providers["anthropic"] = MagicMock()
        result = registry.get_for_model("claude-sonnet-4-20250514")
        assert result is not None
        assert result[1] == "claude-sonnet-4-20250514"

    def test_get_for_model_gemini(self, registry, config):
        registry._connected.add("google")
        registry._providers["google"] = MagicMock()
        result = registry.get_for_model("gemini-2.0-flash")
        assert result is not None
        assert result[1] == "gemini-2.0-flash"

    def test_get_provider_info(self, registry):
        info = registry.get_provider_info()
        assert "active_provider" in info
        assert "fallback_order" in info
        assert "connected_providers" in info
        assert "health" in info

    @pytest.mark.asyncio
    async def test_complete_with_fallback_all_fail(self, registry):
        """Test that all providers failing raises RuntimeError."""
        # No providers connected
        with pytest.raises(RuntimeError, match="All providers failed"):
            await registry.complete_with_fallback(
                messages=[Message(role=Role.USER, content="test")]
            )


# =============================================================================
# API Endpoint Tests
# =============================================================================

class TestLLMAPIEndpoints:
    """Tests for LLM-related REST API endpoints."""

    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from src.web.app import app
        return TestClient(app)

    def test_get_providers_uninitialized(self, client):
        response = client.get("/api/llm/providers")
        assert response.status_code == 200
        data = response.json()
        assert data["registry_initialized"] is False
        assert "anthropic" in data["providers"]
        assert "google" in data["providers"]
        assert "local" in data["providers"]

    def test_get_active_provider_uninitialized(self, client):
        response = client.get("/api/llm/active")
        assert response.status_code == 200
        data = response.json()
        assert data["active_provider"] is None

    def test_get_fallback_order(self, client):
        response = client.get("/api/llm/fallback-order")
        assert response.status_code == 200
        data = response.json()
        assert "fallback_order" in data
        assert "anthropic" in data["fallback_order"]
        assert "google" in data["fallback_order"]
        assert "local" in data["fallback_order"]

    def test_set_active_provider_uninitialized(self, client):
        response = client.post(
            "/api/llm/active",
            json={"provider": "anthropic"}
        )
        assert response.status_code == 503  # Service unavailable

    def test_set_fallback_order_uninitialized(self, client):
        response = client.post(
            "/api/llm/fallback-order",
            json={"order": ["local", "anthropic"]}
        )
        assert response.status_code == 503

    def test_get_health_uninitialized(self, client):
        response = client.get("/api/llm/health")
        assert response.status_code == 200
        data = response.json()
        assert "error" in data


# =============================================================================
# Integration Tests (require actual API keys - skip in CI)
# =============================================================================

@pytest.mark.skipif(
    True,  # Set to False locally with valid API keys
    reason="Integration tests require actual API keys"
)
class TestIntegration:
    """Integration tests with actual LLM providers."""

    @pytest.mark.asyncio
    async def test_anthropic_completion(self):
        import os
        provider = AnthropicProvider(
            api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
            default_model="claude-3-5-haiku-20241022",  # Use cheaper model
        )
        await provider.connect()
        response = await provider.complete(
            messages=[Message(role=Role.USER, content="Say hello")],
            max_tokens=10,
        )
        assert len(response.content) > 0
        await provider.disconnect()

    @pytest.mark.asyncio
    async def test_google_completion(self):
        import os
        provider = GoogleProvider(
            api_key=os.environ.get("GOOGLE_API_KEY", ""),
            default_model="gemini-1.5-flash",  # Use cheaper model
        )
        await provider.connect()
        response = await provider.complete(
            messages=[Message(role=Role.USER, content="Say hello")],
            max_tokens=10,
        )
        assert len(response.content) > 0
        await provider.disconnect()
