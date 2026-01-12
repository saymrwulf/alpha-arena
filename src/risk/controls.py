"""Risk management and controls for trading."""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any

from ..broker.base import Order, Position
from ..strategy.base import TradeSignal


class RiskViolation(str, Enum):
    """Types of risk violations."""

    KILL_SWITCH_ACTIVE = "kill_switch_active"
    MAX_POSITION_SIZE = "max_position_size"
    MAX_SINGLE_TRADE = "max_single_trade"
    DAILY_LOSS_LIMIT = "daily_loss_limit"
    MAX_OPEN_POSITIONS = "max_open_positions"
    RATE_LIMIT = "rate_limit"
    INSUFFICIENT_BALANCE = "insufficient_balance"
    LOW_CONFIDENCE = "low_confidence"


@dataclass
class RiskCheck:
    """Result of a risk check."""

    passed: bool
    violations: list[RiskViolation] = field(default_factory=list)
    messages: list[str] = field(default_factory=list)
    adjusted_signal: TradeSignal | None = None

    def add_violation(self, violation: RiskViolation, message: str) -> None:
        """Add a risk violation."""
        self.passed = False
        self.violations.append(violation)
        self.messages.append(message)


@dataclass
class RiskConfig:
    """Risk management configuration."""

    # Position limits
    max_position_size_usdc: Decimal = Decimal("100")
    max_single_trade_usdc: Decimal = Decimal("25")
    max_open_positions: int = 5

    # Loss limits
    daily_loss_limit_usdc: Decimal = Decimal("50")

    # Rate limits
    max_orders_per_minute: int = 10

    # Kill switch
    kill_switch: bool = False

    # Confidence thresholds
    min_confidence: Decimal = Decimal("0.6")
    min_edge: Decimal = Decimal("0.05")


class RiskManager:
    """
    Manages risk controls for trading.

    Implements:
    - Max position size
    - Daily loss limit
    - Max open positions
    - Rate limits
    - Kill switch
    """

    def __init__(self, config: RiskConfig):
        self.config = config
        self._daily_pnl: Decimal = Decimal("0")
        self._daily_reset_time: datetime = datetime.utcnow().replace(
            hour=0, minute=0, second=0, microsecond=0
        ) + timedelta(days=1)
        self._order_timestamps: list[datetime] = []

    def check_signal(
        self,
        signal: TradeSignal,
        balance: Decimal,
        positions: list[Position],
    ) -> RiskCheck:
        """
        Check if a trading signal passes all risk controls.

        Returns RiskCheck with pass/fail status and any violations.
        """
        check = RiskCheck(passed=True)

        # Kill switch check
        if self.config.kill_switch:
            check.add_violation(
                RiskViolation.KILL_SWITCH_ACTIVE,
                "Kill switch is active - all trading halted",
            )
            return check

        # Skip checks for hold signals
        if signal.signal_type.value == "hold":
            return check

        # Confidence check
        if signal.confidence < self.config.min_confidence:
            check.add_violation(
                RiskViolation.LOW_CONFIDENCE,
                f"Confidence {signal.confidence} below minimum {self.config.min_confidence}",
            )

        # Edge check
        if signal.expected_edge < self.config.min_edge:
            check.add_violation(
                RiskViolation.LOW_CONFIDENCE,
                f"Expected edge {signal.expected_edge} below minimum {self.config.min_edge}",
            )

        # Balance check
        if signal.size and signal.price:
            required = signal.size * signal.price
            if required > balance:
                check.add_violation(
                    RiskViolation.INSUFFICIENT_BALANCE,
                    f"Required ${required} exceeds balance ${balance}",
                )

        # Single trade size check
        if signal.size and signal.size > self.config.max_single_trade_usdc:
            check.add_violation(
                RiskViolation.MAX_SINGLE_TRADE,
                f"Trade size ${signal.size} exceeds max ${self.config.max_single_trade_usdc}",
            )

        # Position size check (existing + new)
        existing_position = next(
            (p for p in positions if p.market_id == signal.market_id),
            None,
        )
        current_size = existing_position.size * existing_position.avg_entry_price if existing_position else Decimal("0")
        new_size = signal.size * signal.price if signal.size and signal.price else Decimal("0")

        if current_size + new_size > self.config.max_position_size_usdc:
            check.add_violation(
                RiskViolation.MAX_POSITION_SIZE,
                f"Position would be ${current_size + new_size}, max is ${self.config.max_position_size_usdc}",
            )

        # Max open positions check
        unique_markets = {p.market_id for p in positions}
        if signal.market_id not in unique_markets and len(unique_markets) >= self.config.max_open_positions:
            check.add_violation(
                RiskViolation.MAX_OPEN_POSITIONS,
                f"Already at max {self.config.max_open_positions} open positions",
            )

        # Daily loss limit check
        self._check_daily_reset()
        if self._daily_pnl <= -self.config.daily_loss_limit_usdc:
            check.add_violation(
                RiskViolation.DAILY_LOSS_LIMIT,
                f"Daily loss ${abs(self._daily_pnl)} at limit ${self.config.daily_loss_limit_usdc}",
            )

        # Rate limit check
        if not self._check_rate_limit():
            check.add_violation(
                RiskViolation.RATE_LIMIT,
                f"Rate limit: max {self.config.max_orders_per_minute} orders/minute",
            )

        return check

    def record_order(self, order: Order) -> None:
        """Record an order for rate limiting."""
        self._order_timestamps.append(datetime.utcnow())

        # Keep only last minute
        cutoff = datetime.utcnow() - timedelta(minutes=1)
        self._order_timestamps = [ts for ts in self._order_timestamps if ts > cutoff]

    def record_pnl(self, pnl: Decimal) -> None:
        """Record PnL for daily tracking."""
        self._check_daily_reset()
        self._daily_pnl += pnl

    def activate_kill_switch(self, reason: str = "") -> None:
        """Activate the kill switch."""
        self.config.kill_switch = True
        print(f"KILL SWITCH ACTIVATED: {reason}")

    def deactivate_kill_switch(self) -> None:
        """Deactivate the kill switch."""
        self.config.kill_switch = False
        print("Kill switch deactivated")

    def get_status(self) -> dict[str, Any]:
        """Get current risk status."""
        self._check_daily_reset()
        return {
            "kill_switch": self.config.kill_switch,
            "daily_pnl": str(self._daily_pnl),
            "daily_loss_limit": str(self.config.daily_loss_limit_usdc),
            "daily_limit_remaining": str(self.config.daily_loss_limit_usdc + self._daily_pnl),
            "orders_last_minute": len(self._order_timestamps),
            "rate_limit": self.config.max_orders_per_minute,
            "max_position_size": str(self.config.max_position_size_usdc),
            "max_open_positions": self.config.max_open_positions,
        }

    def _check_rate_limit(self) -> bool:
        """Check if within rate limit."""
        cutoff = datetime.utcnow() - timedelta(minutes=1)
        recent = [ts for ts in self._order_timestamps if ts > cutoff]
        return len(recent) < self.config.max_orders_per_minute

    def _check_daily_reset(self) -> None:
        """Reset daily counters if new day."""
        now = datetime.utcnow()
        if now >= self._daily_reset_time:
            self._daily_pnl = Decimal("0")
            self._daily_reset_time = now.replace(
                hour=0, minute=0, second=0, microsecond=0
            ) + timedelta(days=1)

    def adjust_signal_for_risk(
        self,
        signal: TradeSignal,
        balance: Decimal,
        positions: list[Position],
    ) -> TradeSignal | None:
        """
        Attempt to adjust a signal to pass risk controls.

        Returns adjusted signal or None if cannot be made compliant.
        """
        check = self.check_signal(signal, balance, positions)
        if check.passed:
            return signal

        # Can't adjust for these
        blocking_violations = {
            RiskViolation.KILL_SWITCH_ACTIVE,
            RiskViolation.DAILY_LOSS_LIMIT,
            RiskViolation.MAX_OPEN_POSITIONS,
            RiskViolation.RATE_LIMIT,
            RiskViolation.LOW_CONFIDENCE,
        }

        if blocking_violations & set(check.violations):
            return None

        # Try to reduce size
        new_size = signal.size
        if signal.size and signal.price:
            # Reduce for balance
            if RiskViolation.INSUFFICIENT_BALANCE in check.violations:
                max_affordable = balance / signal.price * Decimal("0.95")  # 5% buffer
                new_size = min(new_size, max_affordable)

            # Reduce for single trade limit
            if RiskViolation.MAX_SINGLE_TRADE in check.violations:
                new_size = min(new_size, self.config.max_single_trade_usdc)

            # Reduce for position size limit
            if RiskViolation.MAX_POSITION_SIZE in check.violations:
                existing = next(
                    (p for p in positions if p.market_id == signal.market_id),
                    None,
                )
                current = existing.size * existing.avg_entry_price if existing else Decimal("0")
                max_additional = (self.config.max_position_size_usdc - current) / signal.price
                new_size = min(new_size, max_additional)

        if new_size <= 0:
            return None

        # Create adjusted signal
        from dataclasses import replace
        adjusted = replace(signal, size=new_size)

        # Verify adjusted signal passes
        recheck = self.check_signal(adjusted, balance, positions)
        if recheck.passed:
            return adjusted

        return None
