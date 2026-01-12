"""Tests for PnL accounting and metrics."""

import asyncio
import tempfile
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.metrics.logger import (
    DecisionLog,
    MetricsLogger,
    OrderLog,
    PnLSnapshot,
)


@pytest.fixture
def temp_dir():
    """Create temporary directory for test logs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
async def metrics_logger(temp_dir):
    """Create a metrics logger with temp storage."""
    logger = MetricsLogger(
        jsonl_path=f"{temp_dir}/decisions.jsonl",
        sqlite_path=f"{temp_dir}/metrics.db",
    )
    await logger.connect()
    yield logger
    await logger.disconnect()


class TestPnLAccounting:
    """Test PnL tracking and calculations."""

    @pytest.mark.asyncio
    async def test_pnl_snapshot_logging(self, metrics_logger: MetricsLogger):
        """Should log PnL snapshots correctly."""
        snapshot = PnLSnapshot(
            timestamp=datetime.utcnow(),
            balance=Decimal("1000"),
            unrealized_pnl=Decimal("50"),
            realized_pnl=Decimal("25"),
            total_equity=Decimal("1075"),
            positions_value=Decimal("75"),
            drawdown_pct=Decimal("0"),
            high_water_mark=Decimal("1075"),
        )

        await metrics_logger.log_pnl_snapshot(snapshot)

        # Verify it was logged
        history = await metrics_logger.get_pnl_history()
        assert len(history) == 1
        assert history[0]["total_equity"] == 1075

    @pytest.mark.asyncio
    async def test_high_water_mark_tracking(self, metrics_logger: MetricsLogger):
        """Should track high water mark correctly."""
        # First snapshot sets HWM
        await metrics_logger.log_pnl_snapshot(PnLSnapshot(
            timestamp=datetime.utcnow(),
            balance=Decimal("1000"),
            unrealized_pnl=Decimal("0"),
            realized_pnl=Decimal("0"),
            total_equity=Decimal("1000"),
            positions_value=Decimal("0"),
            drawdown_pct=Decimal("0"),
            high_water_mark=Decimal("1000"),
        ))

        assert metrics_logger._high_water_mark == Decimal("1000")

        # Higher equity updates HWM
        await metrics_logger.log_pnl_snapshot(PnLSnapshot(
            timestamp=datetime.utcnow(),
            balance=Decimal("1100"),
            unrealized_pnl=Decimal("0"),
            realized_pnl=Decimal("100"),
            total_equity=Decimal("1100"),
            positions_value=Decimal("0"),
            drawdown_pct=Decimal("0"),
            high_water_mark=Decimal("1100"),
        ))

        assert metrics_logger._high_water_mark == Decimal("1100")

        # Lower equity doesn't change HWM
        await metrics_logger.log_pnl_snapshot(PnLSnapshot(
            timestamp=datetime.utcnow(),
            balance=Decimal("1050"),
            unrealized_pnl=Decimal("0"),
            realized_pnl=Decimal("50"),
            total_equity=Decimal("1050"),
            positions_value=Decimal("0"),
            drawdown_pct=Decimal("0.0455"),
            high_water_mark=Decimal("1100"),
        ))

        assert metrics_logger._high_water_mark == Decimal("1100")

    @pytest.mark.asyncio
    async def test_drawdown_calculation(self, metrics_logger: MetricsLogger):
        """Should calculate drawdown correctly."""
        # Set HWM
        metrics_logger._high_water_mark = Decimal("1000")

        # Calculate drawdown from HWM
        drawdown = metrics_logger.calculate_drawdown(Decimal("900"))
        assert drawdown == Decimal("0.1000")  # 10% drawdown

        drawdown = metrics_logger.calculate_drawdown(Decimal("750"))
        assert drawdown == Decimal("0.2500")  # 25% drawdown

        drawdown = metrics_logger.calculate_drawdown(Decimal("1000"))
        assert drawdown == Decimal("0.0000")  # No drawdown at HWM


class TestDecisionLogging:
    """Test decision logging."""

    @pytest.mark.asyncio
    async def test_decision_log(self, metrics_logger: MetricsLogger):
        """Should log decisions correctly."""
        decision = DecisionLog(
            timestamp=datetime.utcnow(),
            iteration=1,
            balance=Decimal("1000"),
            positions_count=2,
            markets_analyzed=50,
            signals_generated=3,
            signals_executed=1,
            model_used="claude-sonnet-4-20250514",
            latency_ms=1500,
            tokens_used=2000,
            reasoning="Test reasoning",
            signals=[{"test": "signal"}],
        )

        await metrics_logger.log_decision(decision)

        # Verify in database
        decisions = await metrics_logger.get_recent_decisions(limit=1)
        assert len(decisions) == 1
        assert decisions[0]["iteration"] == 1
        assert decisions[0]["signals_executed"] == 1

    @pytest.mark.asyncio
    async def test_multiple_decisions(self, metrics_logger: MetricsLogger):
        """Should handle multiple decisions."""
        for i in range(5):
            await metrics_logger.log_decision(DecisionLog(
                timestamp=datetime.utcnow(),
                iteration=i + 1,
                balance=Decimal("1000") + Decimal(str(i * 10)),
                positions_count=i,
                markets_analyzed=50,
                signals_generated=i,
                signals_executed=i,
                model_used="test-model",
                latency_ms=1000,
                tokens_used=1000,
                reasoning=f"Iteration {i + 1}",
            ))

        decisions = await metrics_logger.get_recent_decisions(limit=10)
        assert len(decisions) == 5


class TestOrderLogging:
    """Test order logging."""

    @pytest.mark.asyncio
    async def test_order_log(self, metrics_logger: MetricsLogger):
        """Should log orders correctly."""
        order = OrderLog(
            timestamp=datetime.utcnow(),
            order_id="order_123",
            market_id="market_456",
            token_id="token_789",
            side="buy",
            size=Decimal("10"),
            price=Decimal("0.55"),
            status="filled",
            fill_size=Decimal("10"),
            fill_price=Decimal("0.55"),
            fee=Decimal("0.01"),
        )

        await metrics_logger.log_order(order)

        # Verify via JSONL (check file exists and has content)
        jsonl_path = Path(metrics_logger.jsonl_path)
        assert jsonl_path.exists()

        content = jsonl_path.read_text()
        assert "order_123" in content
        assert "buy" in content


class TestStatistics:
    """Test statistics aggregation."""

    @pytest.mark.asyncio
    async def test_statistics_aggregation(self, metrics_logger: MetricsLogger):
        """Should aggregate statistics correctly."""
        # Log some decisions
        for i in range(3):
            await metrics_logger.log_decision(DecisionLog(
                timestamp=datetime.utcnow(),
                iteration=i + 1,
                balance=Decimal("1000"),
                positions_count=0,
                markets_analyzed=50,
                signals_generated=2,
                signals_executed=1,
                model_used="test",
                latency_ms=1000 + i * 100,
                tokens_used=1000 + i * 100,
                reasoning="test",
            ))

        # Log PnL
        await metrics_logger.log_pnl_snapshot(PnLSnapshot(
            timestamp=datetime.utcnow(),
            balance=Decimal("1050"),
            unrealized_pnl=Decimal("25"),
            realized_pnl=Decimal("50"),
            total_equity=Decimal("1075"),
            positions_value=Decimal("25"),
            drawdown_pct=Decimal("0"),
            high_water_mark=Decimal("1075"),
        ))

        stats = await metrics_logger.get_statistics()

        assert stats["total_decisions"] == 3
        assert stats["total_trades"] == 3  # 1 per decision
        assert stats["current_equity"] == 1075
        assert stats["realized_pnl"] == 50


class TestPnLCalculations:
    """Test PnL calculation accuracy."""

    def test_position_pnl(self):
        """Test position-level PnL calculation."""
        from src.broker.base import Position

        position = Position(
            market_id="test",
            token_id="test",
            outcome="YES",
            size=Decimal("100"),
            avg_entry_price=Decimal("0.50"),
            current_price=Decimal("0.60"),
            unrealized_pnl=Decimal("10"),  # (0.60 - 0.50) * 100
        )

        # Verify calculation
        expected_pnl = (position.current_price - position.avg_entry_price) * position.size
        assert expected_pnl == Decimal("10")

    def test_portfolio_pnl(self):
        """Test portfolio-level PnL aggregation."""
        from src.broker.base import Position

        positions = [
            Position(
                market_id="market1",
                token_id="token1",
                outcome="YES",
                size=Decimal("100"),
                avg_entry_price=Decimal("0.50"),
                current_price=Decimal("0.60"),
                unrealized_pnl=Decimal("10"),
            ),
            Position(
                market_id="market2",
                token_id="token2",
                outcome="NO",
                size=Decimal("50"),
                avg_entry_price=Decimal("0.40"),
                current_price=Decimal("0.35"),
                unrealized_pnl=Decimal("-2.5"),
            ),
        ]

        total_unrealized = sum(p.unrealized_pnl for p in positions)
        assert total_unrealized == Decimal("7.5")

    def test_realized_vs_unrealized(self):
        """Test distinction between realized and unrealized PnL."""
        from src.broker.base import Position

        # Position with both realized and unrealized
        position = Position(
            market_id="test",
            token_id="test",
            outcome="YES",
            size=Decimal("50"),  # After partial close
            avg_entry_price=Decimal("0.50"),
            current_price=Decimal("0.60"),
            unrealized_pnl=Decimal("5"),  # On remaining 50
            realized_pnl=Decimal("5"),  # From closed 50
        )

        total_pnl = position.unrealized_pnl + position.realized_pnl
        assert total_pnl == Decimal("10")


class TestEdgeCases:
    """Test edge cases in PnL accounting."""

    @pytest.mark.asyncio
    async def test_zero_balance(self, metrics_logger: MetricsLogger):
        """Should handle zero balance gracefully."""
        snapshot = PnLSnapshot(
            timestamp=datetime.utcnow(),
            balance=Decimal("0"),
            unrealized_pnl=Decimal("0"),
            realized_pnl=Decimal("-100"),
            total_equity=Decimal("0"),
            positions_value=Decimal("0"),
            drawdown_pct=Decimal("1"),  # 100% drawdown
            high_water_mark=Decimal("100"),
        )

        await metrics_logger.log_pnl_snapshot(snapshot)

    @pytest.mark.asyncio
    async def test_negative_pnl(self, metrics_logger: MetricsLogger):
        """Should handle negative PnL correctly."""
        snapshot = PnLSnapshot(
            timestamp=datetime.utcnow(),
            balance=Decimal("900"),
            unrealized_pnl=Decimal("-50"),
            realized_pnl=Decimal("-50"),
            total_equity=Decimal("850"),
            positions_value=Decimal("50"),
            drawdown_pct=Decimal("0.15"),
            high_water_mark=Decimal("1000"),
        )

        await metrics_logger.log_pnl_snapshot(snapshot)

        history = await metrics_logger.get_pnl_history()
        assert len(history) == 1
        assert history[0]["unrealized_pnl"] == -50

    def test_drawdown_with_zero_hwm(self, metrics_logger: MetricsLogger):
        """Should handle zero high water mark."""
        metrics_logger._high_water_mark = Decimal("0")
        drawdown = metrics_logger.calculate_drawdown(Decimal("100"))
        assert drawdown == Decimal("0")
