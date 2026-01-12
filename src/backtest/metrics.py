"""Backtesting performance metrics."""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any
import math


def calculate_sharpe(
    returns: list[Decimal],
    risk_free_rate: Decimal = Decimal("0.05"),
    periods_per_year: int = 252,
) -> Decimal:
    """
    Calculate annualized Sharpe ratio.

    Args:
        returns: List of period returns (as decimals, e.g., 0.01 for 1%)
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods per year (252 for daily, 12 for monthly)

    Returns:
        Annualized Sharpe ratio
    """
    if not returns or len(returns) < 2:
        return Decimal("0")

    # Convert to floats for calculation
    float_returns = [float(r) for r in returns]

    # Calculate mean and std
    mean_return = sum(float_returns) / len(float_returns)
    variance = sum((r - mean_return) ** 2 for r in float_returns) / len(float_returns)
    std_dev = math.sqrt(variance) if variance > 0 else 0.0001

    # Annualize
    annual_return = mean_return * periods_per_year
    annual_std = std_dev * math.sqrt(periods_per_year)

    # Sharpe
    rf_per_period = float(risk_free_rate) / periods_per_year
    sharpe = (mean_return - rf_per_period) / std_dev if std_dev > 0 else 0

    return Decimal(str(round(sharpe * math.sqrt(periods_per_year), 4)))


def calculate_max_drawdown(equity_curve: list[Decimal]) -> tuple[Decimal, int, int]:
    """
    Calculate maximum drawdown.

    Args:
        equity_curve: List of portfolio values over time

    Returns:
        Tuple of (max_drawdown_pct, peak_index, trough_index)
    """
    if not equity_curve or len(equity_curve) < 2:
        return Decimal("0"), 0, 0

    float_curve = [float(e) for e in equity_curve]

    max_dd = 0.0
    peak_idx = 0
    trough_idx = 0
    running_peak = float_curve[0]
    running_peak_idx = 0

    for i, value in enumerate(float_curve):
        if value > running_peak:
            running_peak = value
            running_peak_idx = i

        drawdown = (running_peak - value) / running_peak if running_peak > 0 else 0

        if drawdown > max_dd:
            max_dd = drawdown
            peak_idx = running_peak_idx
            trough_idx = i

    return Decimal(str(round(max_dd * 100, 2))), peak_idx, trough_idx


def calculate_sortino(
    returns: list[Decimal],
    target_return: Decimal = Decimal("0"),
    periods_per_year: int = 252,
) -> Decimal:
    """
    Calculate annualized Sortino ratio (downside deviation only).

    Args:
        returns: List of period returns
        target_return: Minimum acceptable return
        periods_per_year: Number of periods per year

    Returns:
        Annualized Sortino ratio
    """
    if not returns or len(returns) < 2:
        return Decimal("0")

    float_returns = [float(r) for r in returns]
    target = float(target_return)

    # Calculate mean
    mean_return = sum(float_returns) / len(float_returns)

    # Calculate downside deviation
    downside_returns = [min(0, r - target) for r in float_returns]
    downside_variance = sum(r ** 2 for r in downside_returns) / len(downside_returns)
    downside_std = math.sqrt(downside_variance) if downside_variance > 0 else 0.0001

    # Annualize
    sortino = (mean_return - target) / downside_std if downside_std > 0 else 0

    return Decimal(str(round(sortino * math.sqrt(periods_per_year), 4)))


def calculate_calmar(
    total_return: Decimal,
    max_drawdown: Decimal,
    years: Decimal,
) -> Decimal:
    """
    Calculate Calmar ratio (annualized return / max drawdown).

    Args:
        total_return: Total percentage return
        max_drawdown: Maximum drawdown percentage
        years: Number of years

    Returns:
        Calmar ratio
    """
    if max_drawdown <= 0 or years <= 0:
        return Decimal("0")

    annual_return = (float(total_return) / float(years))
    calmar = annual_return / float(max_drawdown)

    return Decimal(str(round(calmar, 4)))


@dataclass
class TradeResult:
    """Result of a single trade in backtest."""
    trade_id: str
    market_id: str
    entry_time: datetime
    exit_time: datetime
    side: str  # "yes" or "no"
    entry_price: Decimal
    exit_price: Decimal
    size: Decimal
    pnl: Decimal
    pnl_pct: Decimal
    fees: Decimal
    is_winner: bool
    holding_period: timedelta


@dataclass
class BacktestMetrics:
    """Comprehensive backtest performance metrics."""
    # Basic performance
    total_return_pct: Decimal = Decimal("0")
    total_pnl: Decimal = Decimal("0")
    starting_capital: Decimal = Decimal("0")
    ending_capital: Decimal = Decimal("0")

    # Risk metrics
    sharpe_ratio: Decimal = Decimal("0")
    sortino_ratio: Decimal = Decimal("0")
    calmar_ratio: Decimal = Decimal("0")
    max_drawdown_pct: Decimal = Decimal("0")
    max_drawdown_duration: int = 0  # periods
    volatility: Decimal = Decimal("0")

    # Trade statistics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: Decimal = Decimal("0")
    avg_win: Decimal = Decimal("0")
    avg_loss: Decimal = Decimal("0")
    profit_factor: Decimal = Decimal("0")  # gross profit / gross loss
    expectancy: Decimal = Decimal("0")  # avg profit per trade

    # Position sizing
    avg_position_size: Decimal = Decimal("0")
    max_position_size: Decimal = Decimal("0")
    avg_holding_period: timedelta = field(default_factory=lambda: timedelta(0))

    # Market exposure
    time_in_market_pct: Decimal = Decimal("0")
    max_concurrent_positions: int = 0

    # By outcome
    yes_trades: int = 0
    no_trades: int = 0
    yes_win_rate: Decimal = Decimal("0")
    no_win_rate: Decimal = Decimal("0")

    # Time series
    equity_curve: list[Decimal] = field(default_factory=list)
    daily_returns: list[Decimal] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "performance": {
                "total_return_pct": float(self.total_return_pct),
                "total_pnl": float(self.total_pnl),
                "sharpe_ratio": float(self.sharpe_ratio),
                "sortino_ratio": float(self.sortino_ratio),
                "calmar_ratio": float(self.calmar_ratio),
            },
            "risk": {
                "max_drawdown_pct": float(self.max_drawdown_pct),
                "max_drawdown_duration": self.max_drawdown_duration,
                "volatility": float(self.volatility),
            },
            "trades": {
                "total": self.total_trades,
                "winners": self.winning_trades,
                "losers": self.losing_trades,
                "win_rate": float(self.win_rate),
                "profit_factor": float(self.profit_factor),
                "expectancy": float(self.expectancy),
            },
            "positions": {
                "avg_size": float(self.avg_position_size),
                "max_size": float(self.max_position_size),
                "avg_holding_period_hours": self.avg_holding_period.total_seconds() / 3600,
            },
        }

    @classmethod
    def from_trades(
        cls,
        trades: list[TradeResult],
        equity_curve: list[Decimal],
        starting_capital: Decimal,
    ) -> "BacktestMetrics":
        """Calculate metrics from trade results."""
        if not trades:
            return cls(starting_capital=starting_capital, ending_capital=starting_capital)

        # Basic stats
        winning_trades = [t for t in trades if t.is_winner]
        losing_trades = [t for t in trades if not t.is_winner]

        total_pnl = sum(t.pnl for t in trades)
        ending_capital = starting_capital + total_pnl
        total_return_pct = (total_pnl / starting_capital * 100) if starting_capital > 0 else Decimal("0")

        # Win rate
        win_rate = (Decimal(len(winning_trades)) / Decimal(len(trades)) * 100) if trades else Decimal("0")

        # Avg win/loss
        avg_win = sum(t.pnl for t in winning_trades) / len(winning_trades) if winning_trades else Decimal("0")
        avg_loss = sum(t.pnl for t in losing_trades) / len(losing_trades) if losing_trades else Decimal("0")

        # Profit factor
        gross_profit = sum(t.pnl for t in winning_trades)
        gross_loss = abs(sum(t.pnl for t in losing_trades))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else Decimal("999")

        # Expectancy
        expectancy = total_pnl / len(trades) if trades else Decimal("0")

        # Calculate daily returns from equity curve
        daily_returns = []
        for i in range(1, len(equity_curve)):
            if equity_curve[i - 1] > 0:
                ret = (equity_curve[i] - equity_curve[i - 1]) / equity_curve[i - 1]
                daily_returns.append(ret)

        # Risk metrics
        sharpe = calculate_sharpe(daily_returns)
        sortino = calculate_sortino(daily_returns)
        max_dd, _, _ = calculate_max_drawdown(equity_curve)

        # Volatility
        if daily_returns:
            float_returns = [float(r) for r in daily_returns]
            mean_ret = sum(float_returns) / len(float_returns)
            variance = sum((r - mean_ret) ** 2 for r in float_returns) / len(float_returns)
            volatility = Decimal(str(round(math.sqrt(variance) * 100, 2)))
        else:
            volatility = Decimal("0")

        # Holding period
        holding_periods = [t.holding_period for t in trades]
        avg_holding = sum(holding_periods, timedelta(0)) / len(trades) if trades else timedelta(0)

        # Position sizes
        position_sizes = [t.size for t in trades]
        avg_pos = sum(position_sizes) / len(trades) if trades else Decimal("0")
        max_pos = max(position_sizes) if trades else Decimal("0")

        # Yes/No breakdown
        yes_trades = [t for t in trades if t.side == "yes"]
        no_trades_list = [t for t in trades if t.side == "no"]

        yes_wins = len([t for t in yes_trades if t.is_winner])
        no_wins = len([t for t in no_trades_list if t.is_winner])

        yes_win_rate = (Decimal(yes_wins) / Decimal(len(yes_trades)) * 100) if yes_trades else Decimal("0")
        no_win_rate = (Decimal(no_wins) / Decimal(len(no_trades_list)) * 100) if no_trades_list else Decimal("0")

        # Days in backtest
        if trades:
            first_trade = min(t.entry_time for t in trades)
            last_trade = max(t.exit_time for t in trades)
            days = (last_trade - first_trade).days or 1
            years = Decimal(str(days / 365))
        else:
            years = Decimal("1")

        calmar = calculate_calmar(total_return_pct, max_dd, years)

        return cls(
            total_return_pct=total_return_pct.quantize(Decimal("0.01")),
            total_pnl=total_pnl.quantize(Decimal("0.01")),
            starting_capital=starting_capital,
            ending_capital=ending_capital.quantize(Decimal("0.01")),
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            max_drawdown_pct=max_dd,
            volatility=volatility,
            total_trades=len(trades),
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            win_rate=win_rate.quantize(Decimal("0.01")),
            avg_win=avg_win.quantize(Decimal("0.01")),
            avg_loss=avg_loss.quantize(Decimal("0.01")),
            profit_factor=profit_factor.quantize(Decimal("0.01")),
            expectancy=expectancy.quantize(Decimal("0.01")),
            avg_position_size=avg_pos.quantize(Decimal("0.01")),
            max_position_size=max_pos.quantize(Decimal("0.01")),
            avg_holding_period=avg_holding,
            yes_trades=len(yes_trades),
            no_trades=len(no_trades_list),
            yes_win_rate=yes_win_rate.quantize(Decimal("0.01")),
            no_win_rate=no_win_rate.quantize(Decimal("0.01")),
            equity_curve=equity_curve,
            daily_returns=daily_returns,
        )
