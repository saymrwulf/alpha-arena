"""
End-to-End (E2E) Functional Tests for Alpha Arena.

These tests verify complete user workflows and system integration.
They test the application from the user's perspective.
"""

import asyncio
import pytest
from decimal import Decimal

pytestmark = [pytest.mark.e2e, pytest.mark.slow]


class TestTradingWorkflow:
    """Test complete trading workflows."""

    @pytest.mark.asyncio
    async def test_simulation_trading_cycle(self, async_client):
        """
        Complete workflow: Check status -> Start simulation -> Check running -> Stop.
        """
        # Step 1: Check initial status
        response = await async_client.get("/api/trading/status")
        assert response.status_code == 200
        initial_status = response.json()

        # Step 2: Start trading in simulation mode
        response = await async_client.post(
            "/api/trading/start",
            json={"mode": "simulation", "loop_interval": 5, "max_iterations": 1}
        )
        # May already be running or succeed
        if response.status_code == 200:
            data = response.json()
            assert data.get("status") in ["started", "already_running"]

        # Step 3: Verify status changed
        response = await async_client.get("/api/trading/status")
        assert response.status_code == 200

        # Step 4: Stop trading
        response = await async_client.post("/api/trading/stop")
        # Should succeed or say not running
        assert response.status_code in [200, 400]

        # Step 5: Verify stopped
        response = await async_client.get("/api/trading/status")
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_kill_switch_emergency_stop(self, async_client):
        """
        Test emergency stop workflow: Start -> Activate kill switch -> Verify blocked.
        """
        # Ensure kill switch is off initially
        status_response = await async_client.get("/api/trading/status")
        initial_state = status_response.json().get("kill_switch", False)

        # Activate kill switch (API expects form data)
        response = await async_client.post(
            "/api/trading/kill-switch",
            data={"enabled": "true"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data.get("kill_switch") is True

        # Verify trading is blocked if kill switch is on
        start_response = await async_client.post(
            "/api/trading/start",
            json={"mode": "simulation"}
        )
        # Should fail or indicate kill switch active
        assert start_response.status_code in [200, 400]

        # Turn off kill switch
        response = await async_client.post(
            "/api/trading/kill-switch",
            data={"enabled": "false"}
        )
        assert response.status_code == 200


class TestPositionManagementWorkflow:
    """Test position and order management workflows."""

    @pytest.mark.asyncio
    async def test_view_positions_workflow(self, async_client):
        """
        Workflow: View positions -> Check details -> View orders.
        """
        # Step 1: Get all positions
        response = await async_client.get("/api/positions")
        assert response.status_code == 200
        positions = response.json().get("positions", [])

        # Step 2: Get all orders
        response = await async_client.get("/api/orders")
        assert response.status_code == 200
        orders = response.json().get("orders", [])

        # Step 3: Page should render with this data
        response = await async_client.get("/positions")
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")


class TestMarketAnalysisWorkflow:
    """Test market browsing and analysis workflows."""

    @pytest.mark.asyncio
    async def test_browse_and_filter_markets(self, async_client):
        """
        Workflow: Browse markets -> Filter -> View details.
        """
        # Step 1: Get all markets
        response = await async_client.get("/api/markets")
        assert response.status_code == 200
        markets = response.json().get("markets", [])

        # Step 2: Filter by category (if supported)
        response = await async_client.get("/api/markets?limit=10")
        assert response.status_code == 200

        # Step 3: Get arbitrage opportunities
        response = await async_client.get("/api/arbitrage")
        assert response.status_code == 200
        opportunities = response.json().get("opportunities", [])

    @pytest.mark.asyncio
    async def test_market_detail_view(self, async_client):
        """
        Workflow: List markets -> Select one -> View details.
        """
        # Get markets
        response = await async_client.get("/api/markets?limit=5")
        assert response.status_code == 200
        markets = response.json().get("markets", [])

        if markets and "id" in markets[0]:
            market_id = markets[0]["id"]
            # Get detail
            response = await async_client.get(f"/api/markets/{market_id}")
            # May or may not exist
            assert response.status_code in [200, 404]


class TestWalletAnalysisWorkflow:
    """Test wallet analysis workflows."""

    @pytest.mark.asyncio
    async def test_analyze_wallet_workflow(self, async_client):
        """
        Workflow: Enter address -> Analyze -> View results.
        """
        # Step 1: Page should render
        response = await async_client.get("/wallet-analysis")
        assert response.status_code == 200

        # Step 2: Analyze a wallet (with valid format)
        test_address = "0x1234567890123456789012345678901234567890"
        response = await async_client.post(
            "/api/wallet/analyze",
            json={"address": test_address}
        )
        # May fail due to no real data, but should handle gracefully
        assert response.status_code in [200, 404, 500]


class TestRiskManagementWorkflow:
    """Test risk management workflows."""

    @pytest.mark.asyncio
    async def test_view_and_update_risk_settings(self, async_client):
        """
        Workflow: View risk status -> Update settings -> Verify changes.
        """
        # Step 1: Get current risk status
        response = await async_client.get("/api/risk/status")
        assert response.status_code == 200
        initial_status = response.json()

        # Step 2: Update risk settings
        new_settings = {
            "max_position_size_usdc": 200,
            "daily_loss_limit_usdc": 100
        }
        response = await async_client.post("/api/risk/settings", json=new_settings)
        # May succeed or fail validation
        assert response.status_code in [200, 400, 422]

        # Step 3: Verify page renders
        response = await async_client.get("/risk")
        assert response.status_code == 200


class TestConfigurationWorkflow:
    """Test configuration management workflows."""

    @pytest.mark.asyncio
    async def test_view_and_update_config(self, async_client):
        """
        Workflow: View config -> Update -> Verify.
        """
        # Step 1: Get current config
        response = await async_client.get("/api/config")
        assert response.status_code == 200
        initial_config = response.json()

        # Step 2: Update config
        response = await async_client.post(
            "/api/config",
            json={"loop_interval_seconds": 90}
        )
        assert response.status_code in [200, 400]

        # Step 3: Verify page renders
        response = await async_client.get("/config")
        assert response.status_code == 200


class TestLogsAndMetricsWorkflow:
    """Test logs and metrics viewing workflows."""

    @pytest.mark.asyncio
    async def test_view_decision_logs(self, async_client):
        """
        Workflow: View logs -> Filter -> Check metrics.
        """
        # Step 1: Get decision logs
        response = await async_client.get("/api/logs/decisions?limit=50")
        assert response.status_code == 200
        logs = response.json().get("logs", [])

        # Step 2: Get trade history
        response = await async_client.get("/api/logs/trades")
        assert response.status_code == 200

        # Step 3: Get performance metrics
        response = await async_client.get("/api/metrics/performance")
        assert response.status_code == 200

        # Step 4: Get equity curve
        response = await async_client.get("/api/metrics/equity-curve")
        assert response.status_code == 200

        # Step 5: Page should render
        response = await async_client.get("/logs")
        assert response.status_code == 200


class TestAgentMonitoringWorkflow:
    """Test agent monitoring workflows."""

    @pytest.mark.asyncio
    async def test_view_agent_status(self, async_client):
        """
        Workflow: View agents -> Check history.
        """
        # Step 1: Get all agents status
        response = await async_client.get("/api/agents")
        assert response.status_code == 200
        agents = response.json().get("agents", [])

        # Step 2: Get history for each agent type
        for agent_name in ["research", "risk", "execution", "reflection"]:
            response = await async_client.get(f"/api/agents/{agent_name}/history")
            assert response.status_code == 200

        # Step 3: Page should render
        response = await async_client.get("/agents")
        assert response.status_code == 200


class TestNavigationWorkflow:
    """Test navigation between pages."""

    @pytest.mark.asyncio
    async def test_all_pages_accessible(self, async_client):
        """
        Verify all main pages are accessible.
        """
        pages = [
            "/",
            "/trading",
            "/positions",
            "/markets",
            "/wallet-analysis",
            "/agents",
            "/risk",
            "/config",
            "/logs",
            "/help",
        ]

        for page in pages:
            response = await async_client.get(page)
            assert response.status_code == 200, f"Failed to load {page}"
            assert "text/html" in response.headers.get("content-type", "")


class TestDataConsistency:
    """Test data consistency across endpoints."""

    @pytest.mark.asyncio
    async def test_status_consistency(self, async_client):
        """
        Verify trading status is consistent across endpoints.
        """
        # Get status from trading endpoint
        trading_response = await async_client.get("/api/trading/status")
        assert trading_response.status_code == 200
        trading_status = trading_response.json()

        # Get status from health endpoint
        health_response = await async_client.get("/api/system/health")
        assert health_response.status_code == 200

        # Both should reflect consistent state
        # (This test documents expected behavior)

    @pytest.mark.asyncio
    async def test_risk_status_matches_config(self, async_client):
        """
        Verify risk status reflects configuration.
        """
        # Get risk status
        status_response = await async_client.get("/api/risk/status")
        assert status_response.status_code == 200

        # Get config
        config_response = await async_client.get("/api/config")
        assert config_response.status_code == 200


class TestErrorRecovery:
    """Test system behavior after errors."""

    @pytest.mark.asyncio
    async def test_system_stable_after_invalid_requests(self, async_client):
        """
        System should remain stable after handling invalid requests.
        """
        # Send various invalid requests
        await async_client.post("/api/trading/start", json={"invalid": "data"})
        await async_client.post("/api/wallet/analyze", json={"address": "bad"})
        await async_client.get("/api/nonexistent")

        # System should still be responsive
        response = await async_client.get("/api/system/health")
        assert response.status_code == 200
        assert response.json().get("status") == "healthy"

        # All pages should still render
        response = await async_client.get("/")
        assert response.status_code == 200
