# Alpha Arena - Testing Guide

Complete guide to the test suite and testing practices.

---

## Quick Navigation

| I want to... | Go to |
|--------------|-------|
| Run all tests | [Quick Reference](#quick-reference) |
| Run specific test suites | [Test Categories](#test-categories) |
| Understand test fixtures | [Test Fixtures](#test-fixtures) |
| Write new tests | [Writing New Tests](#writing-new-tests) |
| Mock external services | [Mocking External Services](#mocking-external-services) |
| Fix test issues | [Troubleshooting Tests](#troubleshooting-tests) |

---

## Quick Reference

```bash
# Run all tests (two ways)
./alpha test
./scripts/test.sh

# Run specific suites
./scripts/test.sh unit       # Unit tests only
./scripts/test.sh api        # API tests only
./scripts/test.sh e2e        # End-to-end tests
./scripts/test.sh fast       # Exclude slow tests
./scripts/test.sh coverage   # With coverage report
```

## Test Structure

```
tests/
├── conftest.py          # Shared fixtures and configuration
├── test_api.py          # API endpoint tests
├── test_e2e.py          # End-to-end functional tests
├── test_integration.py  # Multi-agent integration tests (NEW)
├── test_debate.py       # Debate system tests (NEW)
├── test_signals.py      # Signal aggregation tests (NEW)
├── test_llm_providers.py# LLM provider tests (NEW)
├── test_risk.py         # Risk management unit tests
├── test_pnl.py          # PnL accounting unit tests
├── test_indicators.py   # Technical indicator tests
├── test_memory.py       # Memory system tests
├── test_backtest.py     # Backtesting tests
└── test_core_types.py   # Core type tests
```

**Test count: 328+ tests**

## Test Categories

### Unit Tests

Test individual functions and classes in isolation.

```bash
./scripts/test.sh unit
```

**What's tested:**
- Risk validation logic (`test_risk.py`)
- PnL calculations (`test_pnl.py`)
- Technical indicators (`test_indicators.py`)
- Core data types (`test_core_types.py`)

**Example:**
```python
def test_position_pnl():
    """Test position-level PnL calculation."""
    position = Position(
        size=Decimal("100"),
        avg_entry_price=Decimal("0.50"),
        current_price=Decimal("0.60"),
    )
    expected_pnl = (0.60 - 0.50) * 100  # = $10
    assert position.unrealized_pnl == Decimal("10")
```

### API Tests

Test all REST API endpoints for correct behavior.

```bash
./scripts/test.sh api
```

**What's tested:**
- Health endpoints (`/api/system/health`)
- Trading control (`/api/trading/start`, `/api/trading/stop`)
- Position management (`/api/positions`)
- Market browsing (`/api/markets`)
- Configuration (`/api/config`)
- All page rendering (`/`, `/trading`, etc.)
- Error handling (404, 422, etc.)

**Example:**
```python
@pytest.mark.asyncio
async def test_health_check(async_client):
    """GET /api/system/health should return healthy status."""
    response = await async_client.get("/api/system/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
```

### Integration Tests

Test multi-component integration with mocked LLM providers.

```bash
pytest -m "integration"
```

**What's tested:**
- Full agent coordination (Research → Risk → Debate → Execution)
- Multi-agent debate with multiple personas
- Confidence calibration tracking
- Signal aggregation from multiple sources
- Event calendar impact on trading
- End-to-end trading cycle with mocked broker
- Risk rejection scenarios
- LLM failure handling
- Performance metrics tracking

**Key fixture:** `MockLLMProvider` - Returns structured JSON responses matching the real `LLMResponse` interface.

### End-to-End Tests

Test complete user workflows via API.

```bash
./scripts/test.sh e2e
```

**What's tested:**
- Trading workflow: start → check → stop
- Kill switch emergency stop
- Position management workflow
- Market browsing and filtering
- Wallet analysis workflow
- Configuration updates
- Navigation between pages
- Error recovery

**Example:**
```python
@pytest.mark.asyncio
async def test_simulation_trading_cycle(async_client):
    """Complete workflow: Check status -> Start -> Stop."""
    # Step 1: Check initial status
    response = await async_client.get("/api/trading/status")
    assert response.status_code == 200

    # Step 2: Start trading
    response = await async_client.post(
        "/api/trading/start",
        json={"mode": "simulation"}
    )

    # Step 3: Stop trading
    response = await async_client.post("/api/trading/stop")
```

## Test Fixtures

Common fixtures defined in `conftest.py`:

| Fixture | Description |
|---------|-------------|
| `temp_dir` | Temporary directory for test artifacts |
| `test_config` | Test configuration (no real API keys) |
| `risk_config` | Risk configuration for tests |
| `sample_position` | Example position data |
| `sample_signal` | Example trade signal |
| `mock_broker` | Mock broker (no real trades) |
| `mock_llm` | Mock LLM client (no real API calls) |
| `async_client` | HTTP client for API testing |
| `metrics_logger` | Logger with temporary storage |

## Running Tests

### All Tests
```bash
./scripts/test.sh
```

### Verbose Output
```bash
./scripts/test.sh -v
```

### Stop on First Failure
```bash
./scripts/test.sh -x
```

### With Coverage
```bash
./scripts/test.sh coverage

# View HTML report
open htmlcov/index.html
```

### Specific Test File
```bash
source .venv/bin/activate
pytest tests/test_risk.py -v
```

### Specific Test Function
```bash
source .venv/bin/activate
pytest tests/test_risk.py::TestRiskChecks::test_valid_signal_passes -v
```

## Test Markers

Tests can be filtered by markers:

```bash
# Exclude slow tests
pytest -m "not slow"

# Run only API tests
pytest -m "api"

# Run only E2E tests
pytest -m "e2e"

# Run only integration tests
pytest -m "integration"
```

## Writing New Tests

### Unit Test Template
```python
"""Tests for [module name]."""

import pytest
from decimal import Decimal

from src.module import MyClass


class TestMyClass:
    """Test MyClass functionality."""

    def test_basic_operation(self):
        """Should perform basic operation correctly."""
        obj = MyClass()
        result = obj.operation()
        assert result == expected_value

    @pytest.mark.asyncio
    async def test_async_operation(self):
        """Should handle async operations."""
        obj = MyClass()
        result = await obj.async_operation()
        assert result is not None
```

### API Test Template
```python
"""API tests for [endpoint group]."""

import pytest

pytestmark = pytest.mark.api


class TestMyEndpoint:
    """Test /api/my-endpoint."""

    @pytest.mark.asyncio
    async def test_get_success(self, async_client):
        """GET should return success."""
        response = await async_client.get("/api/my-endpoint")
        assert response.status_code == 200
        data = response.json()
        assert "expected_field" in data

    @pytest.mark.asyncio
    async def test_post_validation(self, async_client):
        """POST with invalid data should return 422."""
        response = await async_client.post(
            "/api/my-endpoint",
            json={"invalid": "data"}
        )
        assert response.status_code == 422
```

### E2E Test Template
```python
"""E2E tests for [workflow]."""

import pytest

pytestmark = [pytest.mark.e2e, pytest.mark.slow]


class TestMyWorkflow:
    """Test complete [workflow name] workflow."""

    @pytest.mark.asyncio
    async def test_complete_workflow(self, async_client):
        """
        Complete workflow: Step 1 -> Step 2 -> Step 3.
        """
        # Step 1
        response = await async_client.get("/api/step1")
        assert response.status_code == 200

        # Step 2
        response = await async_client.post("/api/step2", json={...})
        assert response.status_code == 200

        # Step 3
        response = await async_client.get("/api/step3")
        assert response.status_code == 200
```

## Mocking External Services

### Mock Broker
```python
def test_with_mock_broker(mock_broker):
    """Test using mock broker."""
    mock_broker.balance = Decimal("500")
    mock_broker.positions = [sample_position]

    # Your test code
    balance = await mock_broker.get_balance()
    assert balance == Decimal("500")
```

### Mock LLM
```python
def test_with_mock_llm(mock_llm):
    """Test using mock LLM."""
    mock_llm.responses = ["Buy YES at 0.55"]

    # Your test code
    response = await mock_llm.complete("Analyze market")
    assert "Buy" in response
```

## Continuous Integration

Tests run automatically on:
- Every push to main branch
- Every pull request
- Nightly scheduled runs

### Required Checks
1. All tests pass
2. Coverage above 70%
3. No security vulnerabilities

## Troubleshooting Tests

### "Module not found"
```bash
# Ensure venv is activated
source .venv/bin/activate
```

### "Async test timeout"
```python
# Increase timeout for slow tests
@pytest.mark.timeout(60)
async def test_slow_operation():
    ...
```

### "Database locked"
```bash
# Use separate temp directories for parallel tests
pytest -n auto  # Uses pytest-xdist
```

### "Fixture not found"
Ensure `conftest.py` is in the tests directory and properly imports all fixtures.
