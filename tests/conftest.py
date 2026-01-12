"""
Shared pytest fixtures and configuration.

This module provides common test fixtures used across all test files.
"""

import asyncio
import os
import sys
import tempfile
from decimal import Decimal
from pathlib import Path
from typing import AsyncGenerator, Generator

import pytest
import pytest_asyncio

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

# Set test environment
os.environ["TESTING"] = "1"
os.environ["LOG_LEVEL"] = "WARNING"


# ============================================================================
# Async Event Loop Configuration
# ============================================================================

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# ============================================================================
# Temporary Directory Fixtures
# ============================================================================

@pytest.fixture
def temp_dir() -> Generator[str, None, None]:
    """Create temporary directory for test artifacts."""
    with tempfile.TemporaryDirectory(prefix="alpha_arena_test_") as tmpdir:
        yield tmpdir


@pytest.fixture
def data_dir(temp_dir: str) -> str:
    """Create data subdirectory structure."""
    data_path = Path(temp_dir) / "data"
    (data_path / "logs").mkdir(parents=True)
    (data_path / "memory").mkdir(parents=True)
    (data_path / "backtest").mkdir(parents=True)
    return str(data_path)


# ============================================================================
# Configuration Fixtures
# ============================================================================

@pytest.fixture
def test_config(temp_dir: str) -> dict:
    """Test configuration that doesn't require real API keys."""
    return {
        "mode": "simulation",
        "log_dir": temp_dir,
        "loop_interval_seconds": 1,
        "max_iterations": 5,
        "risk": {
            "max_position_size_usdc": 100,
            "max_single_trade_usdc": 25,
            "max_open_positions": 5,
            "daily_loss_limit_usdc": 50,
            "kelly_fraction": 0.25,
        },
        "llm": {
            "provider": "mock",
            "model": "mock-model",
        },
    }


@pytest.fixture
def risk_config():
    """Risk configuration for tests."""
    from src.risk.controls import RiskConfig
    return RiskConfig(
        max_position_size_usdc=Decimal("100"),
        max_single_trade_usdc=Decimal("25"),
        max_open_positions=5,
        daily_loss_limit_usdc=Decimal("50"),
        max_orders_per_minute=10,
        kill_switch=False,
        min_confidence=Decimal("0.6"),
        min_edge=Decimal("0.05"),
    )


# ============================================================================
# Trading Fixtures
# ============================================================================

@pytest.fixture
def sample_position():
    """Sample position for tests."""
    from src.broker.base import Position
    return Position(
        market_id="test_market_001",
        token_id="test_token_001",
        outcome="YES",
        size=Decimal("50"),
        avg_entry_price=Decimal("0.55"),
        current_price=Decimal("0.60"),
        unrealized_pnl=Decimal("2.50"),
    )


@pytest.fixture
def sample_signal():
    """Sample trade signal for tests."""
    from src.broker.base import OrderSide
    from src.strategy.base import ExitPlan, SignalType, TradeSignal
    return TradeSignal(
        market_id="test_market_002",
        token_id="test_token_002",
        signal_type=SignalType.ENTER_LONG,
        side=OrderSide.BUY,
        size=Decimal("20"),
        price=Decimal("0.45"),
        confidence=Decimal("0.72"),
        expected_edge=Decimal("0.08"),
        reasoning="Test signal for unit tests",
        exit_plan=ExitPlan(
            profit_target_price=Decimal("0.60"),
            stop_loss_price=Decimal("0.35"),
            max_hold_hours=48,
        ),
    )


@pytest.fixture
def sample_market():
    """Sample market data for tests."""
    return {
        "id": "test_market_001",
        "question": "Will test condition happen?",
        "slug": "test-market",
        "outcomes": ["Yes", "No"],
        "outcomePrices": ["0.55", "0.45"],
        "volume": "50000",
        "liquidity": "10000",
        "endDate": "2025-12-31T23:59:59Z",
        "active": True,
        "closed": False,
    }


# ============================================================================
# Mock Services
# ============================================================================

class MockBroker:
    """Mock broker for testing without real API calls."""

    def __init__(self):
        self.balance = Decimal("1000")
        self.positions = []
        self.orders = []
        self.order_counter = 0

    async def get_balance(self) -> Decimal:
        return self.balance

    async def get_positions(self) -> list:
        return self.positions

    async def place_order(self, market_id: str, token_id: str, side: str,
                          size: Decimal, price: Decimal) -> dict:
        self.order_counter += 1
        order = {
            "id": f"mock_order_{self.order_counter}",
            "market_id": market_id,
            "token_id": token_id,
            "side": side,
            "size": str(size),
            "price": str(price),
            "status": "filled",
        }
        self.orders.append(order)
        return order

    async def cancel_order(self, order_id: str) -> bool:
        self.orders = [o for o in self.orders if o["id"] != order_id]
        return True


@pytest.fixture
def mock_broker() -> MockBroker:
    """Mock broker instance."""
    return MockBroker()


class MockLLMClient:
    """Mock LLM client for testing without real API calls."""

    def __init__(self, responses: list[str] | None = None):
        self.responses = responses or ["Mock LLM response"]
        self.call_count = 0
        self.last_prompt = None

    async def complete(self, prompt: str, **kwargs) -> str:
        self.last_prompt = prompt
        response = self.responses[self.call_count % len(self.responses)]
        self.call_count += 1
        return response


@pytest.fixture
def mock_llm() -> MockLLMClient:
    """Mock LLM client instance."""
    return MockLLMClient()


# ============================================================================
# Web Application Fixtures
# ============================================================================

@pytest.fixture
def app():
    """FastAPI application for testing."""
    from src.web.app import create_app
    return create_app()


@pytest_asyncio.fixture
async def async_client(app):
    """Async HTTP client for API testing."""
    from httpx import ASGITransport, AsyncClient
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test"
    ) as client:
        yield client


# ============================================================================
# Database Fixtures
# ============================================================================

@pytest_asyncio.fixture
async def metrics_logger(temp_dir: str):
    """Metrics logger with temporary storage."""
    from src.metrics.logger import MetricsLogger
    logger = MetricsLogger(
        jsonl_path=f"{temp_dir}/decisions.jsonl",
        sqlite_path=f"{temp_dir}/metrics.db",
    )
    await logger.connect()
    yield logger
    await logger.disconnect()


# ============================================================================
# Cleanup Helpers
# ============================================================================

@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset any singleton instances between tests."""
    yield
    # Add singleton cleanup here if needed


# ============================================================================
# Markers
# ============================================================================

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "api: marks tests as API tests")
    config.addinivalue_line("markers", "e2e: marks tests as end-to-end tests")
