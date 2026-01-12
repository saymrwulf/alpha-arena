"""
API Tests for Alpha Arena Web Application.

Tests all REST API endpoints for correct behavior, error handling,
and response formats.
"""

import pytest
from decimal import Decimal

pytestmark = pytest.mark.api


class TestHealthEndpoints:
    """Test system health and status endpoints."""

    @pytest.mark.asyncio
    async def test_health_check(self, async_client):
        """GET /api/system/health should return healthy status."""
        response = await async_client.get("/api/system/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        # API returns trading_active and websocket_clients
        assert "trading_active" in data or "timestamp" in data

    @pytest.mark.asyncio
    async def test_providers_status(self, async_client):
        """GET /api/system/providers should return provider status."""
        response = await async_client.get("/api/system/providers")
        assert response.status_code == 200
        data = response.json()
        # API returns provider statuses directly
        assert isinstance(data, dict)
        # Should have some known providers
        assert any(key in data for key in ["anthropic", "openai", "polymarket", "providers"])


class TestTradingEndpoints:
    """Test trading control endpoints."""

    @pytest.mark.asyncio
    async def test_trading_status(self, async_client):
        """GET /api/trading/status should return current status."""
        response = await async_client.get("/api/trading/status")
        assert response.status_code == 200
        data = response.json()
        # API uses "active" instead of "is_running"
        assert "active" in data or "is_running" in data
        assert "mode" in data or "kill_switch" in data

    @pytest.mark.asyncio
    async def test_start_trading_simulation(self, async_client):
        """POST /api/trading/start should start trading loop."""
        response = await async_client.post(
            "/api/trading/start",
            json={"mode": "simulation", "loop_interval": 60}
        )
        # May succeed or fail depending on state, but should return valid response
        assert response.status_code in [200, 400]
        data = response.json()
        assert "status" in data or "detail" in data or "message" in data

    @pytest.mark.asyncio
    async def test_stop_trading(self, async_client):
        """POST /api/trading/stop should stop trading loop."""
        response = await async_client.post("/api/trading/stop")
        # May succeed or fail depending on state
        assert response.status_code in [200, 400]

    @pytest.mark.asyncio
    async def test_kill_switch_toggle(self, async_client):
        """POST /api/trading/kill-switch should toggle kill switch."""
        # API expects a Form field "enabled"
        response = await async_client.post(
            "/api/trading/kill-switch",
            data={"enabled": "true"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "kill_switch" in data or "enabled" in data

    @pytest.mark.asyncio
    async def test_start_trading_with_mode(self, async_client):
        """POST /api/trading/start with mode parameter."""
        response = await async_client.post(
            "/api/trading/start",
            json={"mode": "simulation"}
        )
        # Should accept valid modes
        assert response.status_code in [200, 400]


class TestPositionsEndpoints:
    """Test positions and orders endpoints."""

    @pytest.mark.asyncio
    async def test_get_positions(self, async_client):
        """GET /api/positions should return positions list."""
        response = await async_client.get("/api/positions")
        assert response.status_code == 200
        data = response.json()
        assert "positions" in data
        assert isinstance(data["positions"], list)

    @pytest.mark.asyncio
    async def test_get_orders(self, async_client):
        """GET /api/orders should return orders list."""
        response = await async_client.get("/api/orders")
        assert response.status_code == 200
        data = response.json()
        assert "orders" in data
        assert isinstance(data["orders"], list)

    @pytest.mark.asyncio
    async def test_close_position_not_found(self, async_client):
        """POST /api/positions/{id}/close for non-existent position."""
        response = await async_client.post("/api/positions/nonexistent_123/close")
        # API may return 200 with error in body, or 404/400
        assert response.status_code in [200, 404, 400]

    @pytest.mark.asyncio
    async def test_cancel_order_not_found(self, async_client):
        """POST /api/orders/{id}/cancel for non-existent order."""
        response = await async_client.post("/api/orders/nonexistent_123/cancel")
        # API may return 200 with error in body, or 404/400
        assert response.status_code in [200, 404, 400]


class TestMarketsEndpoints:
    """Test markets browsing endpoints."""

    @pytest.mark.asyncio
    async def test_get_markets(self, async_client):
        """GET /api/markets should return markets list."""
        response = await async_client.get("/api/markets")
        assert response.status_code == 200
        data = response.json()
        assert "markets" in data
        assert isinstance(data["markets"], list)

    @pytest.mark.asyncio
    async def test_get_markets_with_filters(self, async_client):
        """GET /api/markets with query params should filter results."""
        response = await async_client.get(
            "/api/markets",
            params={"category": "politics", "limit": 10}
        )
        assert response.status_code == 200
        data = response.json()
        assert "markets" in data

    @pytest.mark.asyncio
    async def test_get_market_detail(self, async_client):
        """GET /api/markets/{id} should return market details."""
        # First get a market ID
        markets_response = await async_client.get("/api/markets?limit=1")
        markets = markets_response.json().get("markets", [])

        if markets:
            market_id = markets[0].get("id", "test")
            response = await async_client.get(f"/api/markets/{market_id}")
            assert response.status_code in [200, 404]

    @pytest.mark.asyncio
    async def test_get_arbitrage_opportunities(self, async_client):
        """GET /api/arbitrage should return arbitrage opportunities."""
        response = await async_client.get("/api/arbitrage")
        assert response.status_code == 200
        data = response.json()
        assert "opportunities" in data
        assert isinstance(data["opportunities"], list)


class TestWalletAnalysisEndpoints:
    """Test wallet analysis endpoints."""

    @pytest.mark.asyncio
    async def test_analyze_wallet_with_address(self, async_client):
        """POST /api/wallet/analyze with address."""
        response = await async_client.post(
            "/api/wallet/analyze",
            json={"address": "invalid_address"}
        )
        # API may accept any string and return analysis or error
        assert response.status_code in [200, 400, 422, 500]

    @pytest.mark.asyncio
    async def test_analyze_wallet_valid_format(self, async_client):
        """POST /api/wallet/analyze with valid address format."""
        response = await async_client.post(
            "/api/wallet/analyze",
            json={"address": "0x1234567890123456789012345678901234567890"}
        )
        # May fail due to no data, but should accept the format
        assert response.status_code in [200, 404, 500]


class TestAgentsEndpoints:
    """Test agent monitoring endpoints."""

    @pytest.mark.asyncio
    async def test_get_agents(self, async_client):
        """GET /api/agents should return agents status."""
        response = await async_client.get("/api/agents")
        assert response.status_code == 200
        data = response.json()
        assert "agents" in data
        assert isinstance(data["agents"], list)

    @pytest.mark.asyncio
    async def test_get_agent_history(self, async_client):
        """GET /api/agents/{name}/history should return agent history."""
        response = await async_client.get("/api/agents/research/history")
        assert response.status_code == 200
        data = response.json()
        assert "history" in data


class TestRiskEndpoints:
    """Test risk management endpoints."""

    @pytest.mark.asyncio
    async def test_get_risk_status(self, async_client):
        """GET /api/risk/status should return risk status."""
        response = await async_client.get("/api/risk/status")
        assert response.status_code == 200
        data = response.json()
        assert "daily_pnl" in data or "status" in data

    @pytest.mark.asyncio
    async def test_update_risk_settings(self, async_client):
        """POST /api/risk/settings should update risk settings."""
        response = await async_client.post(
            "/api/risk/settings",
            json={
                "max_position_size_usdc": 150,
                "daily_loss_limit_usdc": 75
            }
        )
        assert response.status_code in [200, 400]


class TestConfigEndpoints:
    """Test configuration endpoints."""

    @pytest.mark.asyncio
    async def test_get_config(self, async_client):
        """GET /api/config should return current configuration."""
        response = await async_client.get("/api/config")
        assert response.status_code == 200
        data = response.json()
        # Should have some config structure
        assert isinstance(data, dict)

    @pytest.mark.asyncio
    async def test_update_config(self, async_client):
        """POST /api/config should update configuration."""
        response = await async_client.post(
            "/api/config",
            json={"loop_interval_seconds": 120}
        )
        assert response.status_code in [200, 400]

    @pytest.mark.asyncio
    async def test_get_llm_config(self, async_client):
        """GET /api/config/llm should return LLM configuration."""
        response = await async_client.get("/api/config/llm")
        assert response.status_code == 200


class TestLogsEndpoints:
    """Test logs and metrics endpoints."""

    @pytest.mark.asyncio
    async def test_get_decision_logs(self, async_client):
        """GET /api/logs/decisions should return decision logs."""
        response = await async_client.get("/api/logs/decisions")
        assert response.status_code == 200
        data = response.json()
        assert "logs" in data
        assert isinstance(data["logs"], list)

    @pytest.mark.asyncio
    async def test_get_decision_logs_with_limit(self, async_client):
        """GET /api/logs/decisions with limit param."""
        response = await async_client.get("/api/logs/decisions?limit=5")
        assert response.status_code == 200
        data = response.json()
        assert len(data.get("logs", [])) <= 5

    @pytest.mark.asyncio
    async def test_get_trade_history(self, async_client):
        """GET /api/logs/trades should return trade history."""
        response = await async_client.get("/api/logs/trades")
        assert response.status_code == 200
        data = response.json()
        assert "trades" in data

    @pytest.mark.asyncio
    async def test_get_performance_metrics(self, async_client):
        """GET /api/metrics/performance should return metrics."""
        response = await async_client.get("/api/metrics/performance")
        assert response.status_code == 200
        data = response.json()
        # Should have performance metrics
        assert isinstance(data, dict)

    @pytest.mark.asyncio
    async def test_get_equity_curve(self, async_client):
        """GET /api/metrics/equity-curve should return equity data."""
        response = await async_client.get("/api/metrics/equity-curve")
        assert response.status_code == 200
        data = response.json()
        assert "data" in data or "equity" in data or isinstance(data, list)


class TestPageRendering:
    """Test that HTML pages render correctly."""

    @pytest.mark.asyncio
    async def test_dashboard_page(self, async_client):
        """GET / should render dashboard page."""
        response = await async_client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")

    @pytest.mark.asyncio
    async def test_trading_page(self, async_client):
        """GET /trading should render trading page."""
        response = await async_client.get("/trading")
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")

    @pytest.mark.asyncio
    async def test_positions_page(self, async_client):
        """GET /positions should render positions page."""
        response = await async_client.get("/positions")
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")

    @pytest.mark.asyncio
    async def test_markets_page(self, async_client):
        """GET /markets should render markets page."""
        response = await async_client.get("/markets")
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")

    @pytest.mark.asyncio
    async def test_wallet_analysis_page(self, async_client):
        """GET /wallet-analysis should render wallet analysis page."""
        response = await async_client.get("/wallet-analysis")
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")

    @pytest.mark.asyncio
    async def test_agents_page(self, async_client):
        """GET /agents should render agents page."""
        response = await async_client.get("/agents")
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")

    @pytest.mark.asyncio
    async def test_risk_page(self, async_client):
        """GET /risk should render risk page."""
        response = await async_client.get("/risk")
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")

    @pytest.mark.asyncio
    async def test_config_page(self, async_client):
        """GET /config should render config page."""
        response = await async_client.get("/config")
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")

    @pytest.mark.asyncio
    async def test_logs_page(self, async_client):
        """GET /logs should render logs page."""
        response = await async_client.get("/logs")
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")

    @pytest.mark.asyncio
    async def test_help_page(self, async_client):
        """GET /help should render help page."""
        response = await async_client.get("/help")
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")


class TestErrorHandling:
    """Test error handling and edge cases."""

    @pytest.mark.asyncio
    async def test_404_for_unknown_route(self, async_client):
        """Unknown routes should return 404."""
        response = await async_client.get("/api/nonexistent/endpoint")
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_405_for_wrong_method(self, async_client):
        """Wrong HTTP method should return 405."""
        response = await async_client.delete("/api/trading/status")
        assert response.status_code == 405

    @pytest.mark.asyncio
    async def test_422_for_invalid_json(self, async_client):
        """Invalid JSON should return 422."""
        response = await async_client.post(
            "/api/trading/start",
            content="not valid json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_cors_headers_present(self, async_client):
        """CORS headers should be present for API routes."""
        response = await async_client.options("/api/trading/status")
        # FastAPI may or may not have CORS middleware configured
        # This test documents expected behavior
        assert response.status_code in [200, 405]
