"""Performance analysis - calculate detailed wallet performance metrics."""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Optional
import statistics

from .wallet import WalletSummary, WalletPosition, WalletTransaction, TransactionType


@dataclass
class TimeframePnL:
    """P&L for a specific timeframe."""
    period: str  # "1d", "7d", "30d", "90d", "all"
    realized_pnl: Decimal
    unrealized_pnl: Decimal
    total_pnl: Decimal
    pnl_percent: Decimal
    trade_count: int
    win_count: int
    loss_count: int
    volume: Decimal


@dataclass
class MarketPerformance:
    """Performance in a specific market category."""
    category: str
    trade_count: int
    total_pnl: Decimal
    win_rate: Decimal
    avg_profit: Decimal
    avg_loss: Decimal
    best_trade: Decimal
    worst_trade: Decimal


@dataclass
class WalletMetrics:
    """Comprehensive wallet performance metrics."""
    # Core metrics
    total_pnl: Decimal
    total_return_pct: Decimal
    total_volume: Decimal
    total_fees: Decimal

    # Risk metrics
    sharpe_ratio: Decimal
    sortino_ratio: Decimal
    max_drawdown: Decimal
    max_drawdown_pct: Decimal
    calmar_ratio: Decimal
    var_95: Decimal  # Value at Risk 95%
    expected_shortfall: Decimal

    # Trade metrics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: Decimal
    avg_win: Decimal
    avg_loss: Decimal
    largest_win: Decimal
    largest_loss: Decimal
    profit_factor: Decimal
    expectancy: Decimal  # Expected value per trade

    # Streak analysis
    current_streak: int  # Positive = wins, negative = losses
    longest_win_streak: int
    longest_loss_streak: int

    # Timeframe breakdowns
    pnl_by_timeframe: list[TimeframePnL]

    # Market category breakdown
    performance_by_market: list[MarketPerformance]

    # Activity metrics
    avg_trades_per_day: Decimal
    avg_position_duration_hours: Decimal
    most_profitable_hour: int
    most_profitable_day: int  # 0=Monday

    # Efficiency metrics
    roi_per_trade: Decimal
    roi_per_day: Decimal
    capital_efficiency: Decimal  # PnL / max capital used


class PerformanceAnalyzer:
    """Calculate comprehensive performance metrics for a wallet."""

    def analyze(self, summary: WalletSummary) -> WalletMetrics:
        """
        Calculate all performance metrics for a wallet.

        Args:
            summary: Complete wallet summary

        Returns:
            Comprehensive wallet metrics
        """
        positions = summary.positions
        transactions = summary.transactions

        # Core calculations
        total_pnl = sum(p.total_pnl for p in positions)
        total_invested = sum(p.total_invested for p in positions if p.total_invested > 0)
        total_return_pct = (total_pnl / total_invested * 100) if total_invested > 0 else Decimal("0")

        # Trade counts
        closed_positions = [p for p in positions if p.is_closed]
        winning = [p for p in closed_positions if p.realized_pnl > 0]
        losing = [p for p in closed_positions if p.realized_pnl < 0]

        win_rate = Decimal(str(len(winning) / len(closed_positions) * 100)) if closed_positions else Decimal("0")

        # Average win/loss
        avg_win = (sum(p.realized_pnl for p in winning) / len(winning)) if winning else Decimal("0")
        avg_loss = (sum(abs(p.realized_pnl) for p in losing) / len(losing)) if losing else Decimal("0")

        # Best/worst trades
        all_pnls = [p.realized_pnl for p in closed_positions]
        largest_win = max(all_pnls) if all_pnls else Decimal("0")
        largest_loss = min(all_pnls) if all_pnls else Decimal("0")

        # Profit factor
        gross_profit = sum(p.realized_pnl for p in winning)
        gross_loss = sum(abs(p.realized_pnl) for p in losing)
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else Decimal("999")

        # Expectancy
        if closed_positions:
            expectancy = total_pnl / len(closed_positions)
        else:
            expectancy = Decimal("0")

        # Risk metrics
        returns = [float(p.pnl_percent) for p in closed_positions if p.total_invested > 0]
        sharpe = self._calculate_sharpe(returns)
        sortino = self._calculate_sortino(returns)

        # Drawdown
        max_dd, max_dd_pct = self._calculate_drawdown(closed_positions)

        # Calmar ratio
        if summary.first_transaction and summary.last_transaction:
            years = (summary.last_transaction - summary.first_transaction).days / 365.25
            annual_return = float(total_return_pct) / years if years > 0 else 0
            calmar = Decimal(str(annual_return / float(max_dd_pct))) if max_dd_pct > 0 else Decimal("0")
        else:
            calmar = Decimal("0")

        # VaR and Expected Shortfall
        var_95, es = self._calculate_var(returns)

        # Streak analysis
        current, longest_win, longest_loss = self._analyze_streaks(closed_positions)

        # Timeframe breakdowns
        pnl_by_timeframe = self._calculate_timeframe_pnl(summary)

        # Market performance
        performance_by_market = self._calculate_market_performance(positions)

        # Activity metrics
        if summary.first_transaction and summary.last_transaction:
            days_active = max((summary.last_transaction - summary.first_transaction).days, 1)
            avg_trades_per_day = Decimal(str(len(transactions) / days_active))
        else:
            avg_trades_per_day = Decimal("0")

        # Position duration
        durations = []
        for p in closed_positions:
            if p.entry_time and p.exit_time:
                hours = (p.exit_time - p.entry_time).total_seconds() / 3600
                durations.append(hours)
        avg_duration = Decimal(str(statistics.mean(durations))) if durations else Decimal("0")

        # Most profitable times
        profitable_hour, profitable_day = self._analyze_profitable_times(positions)

        # Efficiency
        roi_per_trade = total_pnl / len(closed_positions) if closed_positions else Decimal("0")
        if summary.first_transaction and summary.last_transaction:
            days = max((summary.last_transaction - summary.first_transaction).days, 1)
            roi_per_day = total_pnl / days
        else:
            roi_per_day = Decimal("0")

        max_capital = max(
            sum(p.total_invested for p in positions if p.entry_time and p.entry_time <= t.timestamp)
            for t in transactions
        ) if transactions else Decimal("1")
        capital_efficiency = (total_pnl / max_capital * 100) if max_capital > 0 else Decimal("0")

        return WalletMetrics(
            total_pnl=total_pnl,
            total_return_pct=total_return_pct,
            total_volume=summary.total_volume,
            total_fees=summary.total_fees_paid,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=max_dd,
            max_drawdown_pct=max_dd_pct,
            calmar_ratio=calmar,
            var_95=var_95,
            expected_shortfall=es,
            total_trades=len(closed_positions),
            winning_trades=len(winning),
            losing_trades=len(losing),
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            largest_win=largest_win,
            largest_loss=largest_loss,
            profit_factor=profit_factor,
            expectancy=expectancy,
            current_streak=current,
            longest_win_streak=longest_win,
            longest_loss_streak=longest_loss,
            pnl_by_timeframe=pnl_by_timeframe,
            performance_by_market=performance_by_market,
            avg_trades_per_day=avg_trades_per_day,
            avg_position_duration_hours=avg_duration,
            most_profitable_hour=profitable_hour,
            most_profitable_day=profitable_day,
            roi_per_trade=roi_per_trade,
            roi_per_day=roi_per_day,
            capital_efficiency=capital_efficiency
        )

    def _calculate_sharpe(self, returns: list[float], risk_free: float = 0.05) -> Decimal:
        """Calculate Sharpe ratio."""
        if len(returns) < 2:
            return Decimal("0")

        avg_return = statistics.mean(returns)
        std_return = statistics.stdev(returns)

        if std_return == 0:
            return Decimal("0")

        # Annualize (assume ~100 trades per year for prediction markets)
        annual_factor = (100 / len(returns)) ** 0.5
        sharpe = ((avg_return - risk_free) / std_return) * annual_factor

        return Decimal(str(round(sharpe, 2)))

    def _calculate_sortino(self, returns: list[float], target: float = 0) -> Decimal:
        """Calculate Sortino ratio (penalizes only downside volatility)."""
        if len(returns) < 2:
            return Decimal("0")

        avg_return = statistics.mean(returns)
        downside_returns = [r for r in returns if r < target]

        if not downside_returns:
            return Decimal("999")  # No downside = infinite ratio

        downside_std = statistics.stdev(downside_returns) if len(downside_returns) > 1 else 0

        if downside_std == 0:
            return Decimal("999")

        annual_factor = (100 / len(returns)) ** 0.5
        sortino = ((avg_return - target) / downside_std) * annual_factor

        return Decimal(str(round(sortino, 2)))

    def _calculate_drawdown(
        self,
        positions: list[WalletPosition]
    ) -> tuple[Decimal, Decimal]:
        """Calculate maximum drawdown in $ and %."""
        if not positions:
            return Decimal("0"), Decimal("0")

        sorted_pos = sorted(
            [p for p in positions if p.exit_time],
            key=lambda p: p.exit_time  # type: ignore
        )

        if not sorted_pos:
            return Decimal("0"), Decimal("0")

        cumulative = Decimal("0")
        peak = Decimal("0")
        max_dd = Decimal("0")
        peak_at_max = Decimal("1")

        for pos in sorted_pos:
            cumulative += pos.realized_pnl
            if cumulative > peak:
                peak = cumulative
            dd = peak - cumulative
            if dd > max_dd:
                max_dd = dd
                peak_at_max = peak

        max_dd_pct = (max_dd / peak_at_max * 100) if peak_at_max > 0 else Decimal("0")

        return max_dd, max_dd_pct

    def _calculate_var(self, returns: list[float]) -> tuple[Decimal, Decimal]:
        """Calculate Value at Risk (95%) and Expected Shortfall."""
        if len(returns) < 5:
            return Decimal("0"), Decimal("0")

        sorted_returns = sorted(returns)
        var_index = int(len(sorted_returns) * 0.05)

        var_95 = Decimal(str(abs(sorted_returns[var_index])))

        # Expected Shortfall (average of losses beyond VaR)
        tail_losses = sorted_returns[:var_index + 1]
        es = Decimal(str(abs(statistics.mean(tail_losses)))) if tail_losses else Decimal("0")

        return var_95, es

    def _analyze_streaks(
        self,
        positions: list[WalletPosition]
    ) -> tuple[int, int, int]:
        """Analyze winning and losing streaks."""
        sorted_pos = sorted(
            [p for p in positions if p.exit_time],
            key=lambda p: p.exit_time  # type: ignore
        )

        if not sorted_pos:
            return 0, 0, 0

        current_streak = 0
        longest_win = 0
        longest_loss = 0
        current_win = 0
        current_loss = 0

        for pos in sorted_pos:
            if pos.realized_pnl > 0:
                current_win += 1
                current_loss = 0
                longest_win = max(longest_win, current_win)
            elif pos.realized_pnl < 0:
                current_loss += 1
                current_win = 0
                longest_loss = max(longest_loss, current_loss)
            else:
                current_win = 0
                current_loss = 0

        # Current streak
        if current_win > 0:
            current_streak = current_win
        elif current_loss > 0:
            current_streak = -current_loss

        return current_streak, longest_win, longest_loss

    def _calculate_timeframe_pnl(self, summary: WalletSummary) -> list[TimeframePnL]:
        """Calculate P&L for different timeframes."""
        now = datetime.now()
        timeframes = [
            ("1d", timedelta(days=1)),
            ("7d", timedelta(days=7)),
            ("30d", timedelta(days=30)),
            ("90d", timedelta(days=90)),
            ("all", timedelta(days=36500)),  # ~100 years
        ]

        results = []

        for period, delta in timeframes:
            cutoff = now - delta

            # Filter positions by exit time
            period_positions = [
                p for p in summary.positions
                if p.exit_time and p.exit_time >= cutoff
            ]

            # Filter transactions
            period_txs = [
                t for t in summary.transactions
                if t.timestamp >= cutoff
            ]

            realized = sum(p.realized_pnl for p in period_positions)
            unrealized = sum(
                p.unrealized_pnl for p in summary.positions
                if not p.is_closed
            ) if period == "all" else Decimal("0")

            total = realized + unrealized

            # Volume
            volume = sum(
                t.amount for t in period_txs
                if t.tx_type in [TransactionType.BUY, TransactionType.SELL]
            )

            # Win/loss counts
            wins = sum(1 for p in period_positions if p.realized_pnl > 0)
            losses = sum(1 for p in period_positions if p.realized_pnl < 0)

            # Return %
            invested = sum(p.total_invested for p in period_positions)
            pnl_pct = (total / invested * 100) if invested > 0 else Decimal("0")

            results.append(TimeframePnL(
                period=period,
                realized_pnl=realized,
                unrealized_pnl=unrealized,
                total_pnl=total,
                pnl_percent=pnl_pct,
                trade_count=len(period_positions),
                win_count=wins,
                loss_count=losses,
                volume=volume
            ))

        return results

    def _calculate_market_performance(
        self,
        positions: list[WalletPosition]
    ) -> list[MarketPerformance]:
        """Calculate performance by market category (inferred from question)."""
        # Simple category inference from question keywords
        categories: dict[str, list[WalletPosition]] = {
            "politics": [],
            "crypto": [],
            "sports": [],
            "entertainment": [],
            "other": []
        }

        political_keywords = ["election", "president", "vote", "senate", "congress", "trump", "biden", "democrat", "republican"]
        crypto_keywords = ["bitcoin", "btc", "ethereum", "eth", "crypto", "token", "sol", "price"]
        sports_keywords = ["nfl", "nba", "mlb", "game", "super bowl", "championship", "win", "team"]
        entertainment_keywords = ["oscar", "grammy", "movie", "film", "album", "show", "netflix"]

        for pos in positions:
            question = pos.market_question.lower()

            if any(kw in question for kw in political_keywords):
                categories["politics"].append(pos)
            elif any(kw in question for kw in crypto_keywords):
                categories["crypto"].append(pos)
            elif any(kw in question for kw in sports_keywords):
                categories["sports"].append(pos)
            elif any(kw in question for kw in entertainment_keywords):
                categories["entertainment"].append(pos)
            else:
                categories["other"].append(pos)

        results = []
        for category, cat_positions in categories.items():
            if not cat_positions:
                continue

            closed = [p for p in cat_positions if p.is_closed]
            wins = [p for p in closed if p.realized_pnl > 0]
            losses = [p for p in closed if p.realized_pnl < 0]

            total_pnl = sum(p.realized_pnl for p in closed)
            win_rate = Decimal(str(len(wins) / len(closed) * 100)) if closed else Decimal("0")

            avg_profit = sum(p.realized_pnl for p in wins) / len(wins) if wins else Decimal("0")
            avg_loss = sum(abs(p.realized_pnl) for p in losses) / len(losses) if losses else Decimal("0")

            pnls = [p.realized_pnl for p in closed]
            best = max(pnls) if pnls else Decimal("0")
            worst = min(pnls) if pnls else Decimal("0")

            results.append(MarketPerformance(
                category=category,
                trade_count=len(closed),
                total_pnl=total_pnl,
                win_rate=win_rate,
                avg_profit=avg_profit,
                avg_loss=avg_loss,
                best_trade=best,
                worst_trade=worst
            ))

        return sorted(results, key=lambda x: x.total_pnl, reverse=True)

    def _analyze_profitable_times(
        self,
        positions: list[WalletPosition]
    ) -> tuple[int, int]:
        """Find most profitable trading times."""
        hour_pnl: dict[int, Decimal] = {h: Decimal("0") for h in range(24)}
        day_pnl: dict[int, Decimal] = {d: Decimal("0") for d in range(7)}

        for pos in positions:
            if pos.exit_time and pos.is_closed:
                hour = pos.exit_time.hour
                day = pos.exit_time.weekday()
                hour_pnl[hour] += pos.realized_pnl
                day_pnl[day] += pos.realized_pnl

        best_hour = max(hour_pnl, key=lambda h: hour_pnl[h])
        best_day = max(day_pnl, key=lambda d: day_pnl[d])

        return best_hour, best_day

    def generate_report(self, metrics: WalletMetrics) -> str:
        """Generate a human-readable performance report."""
        lines = [
            "=" * 60,
            "WALLET PERFORMANCE REPORT",
            "=" * 60,
            "",
            "OVERALL PERFORMANCE",
            "-" * 40,
            f"  Total P&L:          ${metrics.total_pnl:+,.2f}",
            f"  Total Return:       {metrics.total_return_pct:+.2f}%",
            f"  Total Volume:       ${metrics.total_volume:,.2f}",
            f"  Fees Paid:          ${metrics.total_fees:,.2f}",
            "",
            "RISK METRICS",
            "-" * 40,
            f"  Sharpe Ratio:       {metrics.sharpe_ratio:.2f}",
            f"  Sortino Ratio:      {metrics.sortino_ratio:.2f}",
            f"  Max Drawdown:       ${metrics.max_drawdown:,.2f} ({metrics.max_drawdown_pct:.1f}%)",
            f"  Calmar Ratio:       {metrics.calmar_ratio:.2f}",
            f"  VaR (95%):          {metrics.var_95:.2f}%",
            "",
            "TRADE STATISTICS",
            "-" * 40,
            f"  Total Trades:       {metrics.total_trades}",
            f"  Win Rate:           {metrics.win_rate:.1f}%",
            f"  Wins / Losses:      {metrics.winning_trades} / {metrics.losing_trades}",
            f"  Avg Win:            ${metrics.avg_win:,.2f}",
            f"  Avg Loss:           ${metrics.avg_loss:,.2f}",
            f"  Largest Win:        ${metrics.largest_win:,.2f}",
            f"  Largest Loss:       ${metrics.largest_loss:,.2f}",
            f"  Profit Factor:      {metrics.profit_factor:.2f}",
            f"  Expectancy:         ${metrics.expectancy:+,.2f}/trade",
            "",
            "STREAK ANALYSIS",
            "-" * 40,
            f"  Current Streak:     {metrics.current_streak:+d}",
            f"  Longest Win Streak: {metrics.longest_win_streak}",
            f"  Longest Loss Streak:{metrics.longest_loss_streak}",
            "",
            "ACTIVITY METRICS",
            "-" * 40,
            f"  Avg Trades/Day:     {metrics.avg_trades_per_day:.2f}",
            f"  Avg Hold Time:      {metrics.avg_position_duration_hours:.1f} hours",
            f"  Best Trading Hour:  {metrics.most_profitable_hour}:00",
            f"  Best Trading Day:   {['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][metrics.most_profitable_day]}",
            "",
            "EFFICIENCY",
            "-" * 40,
            f"  ROI per Trade:      ${metrics.roi_per_trade:+,.2f}",
            f"  ROI per Day:        ${metrics.roi_per_day:+,.2f}",
            f"  Capital Efficiency: {metrics.capital_efficiency:.1f}%",
            "",
        ]

        # Timeframe breakdown
        lines.extend([
            "P&L BY TIMEFRAME",
            "-" * 40,
        ])
        for tf in metrics.pnl_by_timeframe:
            lines.append(f"  {tf.period:6s}: ${tf.total_pnl:+10,.2f} ({tf.trade_count} trades, {tf.win_count}W/{tf.loss_count}L)")

        lines.append("")

        # Market performance
        if metrics.performance_by_market:
            lines.extend([
                "PERFORMANCE BY CATEGORY",
                "-" * 40,
            ])
            for mp in metrics.performance_by_market:
                lines.append(
                    f"  {mp.category:15s}: ${mp.total_pnl:+10,.2f} ({mp.win_rate:.0f}% win rate, {mp.trade_count} trades)"
                )

        lines.extend(["", "=" * 60])

        return "\n".join(lines)
