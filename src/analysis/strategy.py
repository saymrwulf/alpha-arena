"""Strategy detection - identify trading patterns and strategies from wallet history."""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Optional
import statistics

from .wallet import WalletTransaction, WalletPosition, WalletSummary, TransactionType


class StrategyType(Enum):
    """Detected strategy types."""
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    ARBITRAGE = "arbitrage"
    EVENT_DRIVEN = "event_driven"
    SCALPING = "scalping"
    SWING_TRADING = "swing_trading"
    VALUE_INVESTING = "value_investing"
    MARKET_MAKING = "market_making"
    DEGEN = "degen"  # High risk, concentrated bets
    DIVERSIFIED = "diversified"
    UNKNOWN = "unknown"


class RiskProfile(Enum):
    """Risk profile categories."""
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"
    DEGEN = "degen"


@dataclass
class StrategyPattern:
    """A detected trading pattern."""
    pattern_type: str
    confidence: Decimal
    description: str
    supporting_evidence: list[str] = field(default_factory=list)
    examples: list[str] = field(default_factory=list)


@dataclass
class TradingBehavior:
    """Analyzed trading behavior metrics."""
    # Timing patterns
    avg_hold_time_hours: Decimal
    min_hold_time_hours: Decimal
    max_hold_time_hours: Decimal
    trades_per_day: Decimal
    trades_per_week: Decimal

    # Position sizing
    avg_position_size: Decimal
    position_size_std: Decimal
    max_position_size: Decimal
    position_concentration: Decimal  # % in largest position

    # Entry/exit patterns
    avg_entry_price: Decimal
    avg_exit_price: Decimal
    prefers_yes: bool  # More YES than NO positions
    buys_dips: bool  # Tends to buy below 0.5
    sells_rallies: bool  # Tends to sell above 0.5

    # Risk metrics
    avg_profit_per_trade: Decimal
    avg_loss_per_trade: Decimal
    profit_factor: Decimal  # Gross profit / gross loss
    max_drawdown: Decimal
    sharpe_estimate: Decimal

    # Activity patterns
    most_active_hour: int
    most_active_day: int  # 0=Monday
    weekend_active: bool
    consistent_schedule: bool


@dataclass
class DetectedStrategy:
    """Complete strategy detection result."""
    primary_strategy: StrategyType
    secondary_strategies: list[StrategyType]
    confidence: Decimal
    risk_profile: RiskProfile
    behavior: TradingBehavior
    patterns: list[StrategyPattern]
    strengths: list[str]
    weaknesses: list[str]
    recommendations: list[str]
    summary: str


class StrategyDetector:
    """Detect trading strategies and patterns from wallet history."""

    def __init__(self):
        self.min_transactions_for_analysis = 10
        self.min_positions_for_analysis = 5

    def detect_strategy(self, summary: WalletSummary) -> DetectedStrategy:
        """
        Analyze wallet history to detect trading strategy.

        Args:
            summary: Complete wallet summary

        Returns:
            Detected strategy with analysis
        """
        if len(summary.transactions) < self.min_transactions_for_analysis:
            return self._insufficient_data_result(summary)

        # Analyze trading behavior
        behavior = self._analyze_behavior(summary)

        # Detect patterns
        patterns = self._detect_patterns(summary, behavior)

        # Determine strategy type
        primary, secondary, confidence = self._classify_strategy(behavior, patterns)

        # Determine risk profile
        risk_profile = self._determine_risk_profile(behavior, summary)

        # Generate insights
        strengths = self._identify_strengths(behavior, summary)
        weaknesses = self._identify_weaknesses(behavior, summary)
        recommendations = self._generate_recommendations(behavior, weaknesses)

        # Create summary
        summary_text = self._generate_summary(
            primary, risk_profile, behavior, summary
        )

        return DetectedStrategy(
            primary_strategy=primary,
            secondary_strategies=secondary,
            confidence=confidence,
            risk_profile=risk_profile,
            behavior=behavior,
            patterns=patterns,
            strengths=strengths,
            weaknesses=weaknesses,
            recommendations=recommendations,
            summary=summary_text
        )

    def _analyze_behavior(self, summary: WalletSummary) -> TradingBehavior:
        """Extract behavioral metrics from transaction history."""
        transactions = summary.transactions
        positions = summary.positions

        # Hold time analysis
        hold_times = []
        for pos in positions:
            if pos.entry_time and pos.exit_time:
                delta = pos.exit_time - pos.entry_time
                hold_times.append(delta.total_seconds() / 3600)  # Hours

        avg_hold = Decimal(str(statistics.mean(hold_times))) if hold_times else Decimal("0")
        min_hold = Decimal(str(min(hold_times))) if hold_times else Decimal("0")
        max_hold = Decimal(str(max(hold_times))) if hold_times else Decimal("0")

        # Trading frequency
        if summary.first_transaction and summary.last_transaction:
            days_active = (summary.last_transaction - summary.first_transaction).days or 1
            trades_per_day = Decimal(str(len(transactions) / days_active))
            trades_per_week = trades_per_day * 7
        else:
            trades_per_day = Decimal("0")
            trades_per_week = Decimal("0")

        # Position sizing
        position_sizes = [p.total_invested for p in positions if p.total_invested > 0]
        avg_pos_size = Decimal(str(statistics.mean([float(p) for p in position_sizes]))) if position_sizes else Decimal("0")
        pos_std = Decimal(str(statistics.stdev([float(p) for p in position_sizes]))) if len(position_sizes) > 1 else Decimal("0")
        max_pos_size = max(position_sizes) if position_sizes else Decimal("0")

        total_invested = sum(position_sizes)
        concentration = (max_pos_size / total_invested * 100) if total_invested > 0 else Decimal("0")

        # Entry/exit analysis
        buy_prices = [tx.price for tx in transactions if tx.tx_type == TransactionType.BUY and tx.price]
        sell_prices = [tx.price for tx in transactions if tx.tx_type == TransactionType.SELL and tx.price]

        avg_entry = Decimal(str(statistics.mean([float(p) for p in buy_prices]))) if buy_prices else Decimal("0.5")
        avg_exit = Decimal(str(statistics.mean([float(p) for p in sell_prices]))) if sell_prices else Decimal("0.5")

        yes_count = sum(1 for tx in transactions if tx.outcome and tx.outcome.upper() == "YES")
        no_count = sum(1 for tx in transactions if tx.outcome and tx.outcome.upper() == "NO")
        prefers_yes = yes_count > no_count

        buys_dips = avg_entry < Decimal("0.5")
        sells_rallies = avg_exit > Decimal("0.5")

        # P&L analysis
        profits = [p.realized_pnl for p in positions if p.realized_pnl > 0]
        losses = [abs(p.realized_pnl) for p in positions if p.realized_pnl < 0]

        avg_profit = Decimal(str(statistics.mean([float(p) for p in profits]))) if profits else Decimal("0")
        avg_loss = Decimal(str(statistics.mean([float(l) for l in losses]))) if losses else Decimal("0")

        gross_profit = sum(profits)
        gross_loss = sum(losses)
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else Decimal("999")

        # Drawdown (simplified)
        max_dd = self._calculate_max_drawdown(positions)

        # Sharpe estimate (simplified)
        returns = [float(p.pnl_percent) for p in positions if p.total_invested > 0]
        if len(returns) > 1:
            avg_return = statistics.mean(returns)
            std_return = statistics.stdev(returns)
            sharpe = Decimal(str(avg_return / std_return)) if std_return > 0 else Decimal("0")
        else:
            sharpe = Decimal("0")

        # Activity timing
        hours = [tx.timestamp.hour for tx in transactions]
        days = [tx.timestamp.weekday() for tx in transactions]

        most_active_hour = max(set(hours), key=hours.count) if hours else 12
        most_active_day = max(set(days), key=days.count) if days else 0

        weekend_trades = sum(1 for d in days if d >= 5)
        weekend_active = weekend_trades / len(days) > 0.2 if days else False

        hour_counts = {h: hours.count(h) for h in set(hours)}
        consistent = (max(hour_counts.values()) / len(hours) > 0.3) if hours else False

        return TradingBehavior(
            avg_hold_time_hours=avg_hold,
            min_hold_time_hours=min_hold,
            max_hold_time_hours=max_hold,
            trades_per_day=trades_per_day,
            trades_per_week=trades_per_week,
            avg_position_size=avg_pos_size,
            position_size_std=pos_std,
            max_position_size=max_pos_size,
            position_concentration=concentration,
            avg_entry_price=avg_entry,
            avg_exit_price=avg_exit,
            prefers_yes=prefers_yes,
            buys_dips=buys_dips,
            sells_rallies=sells_rallies,
            avg_profit_per_trade=avg_profit,
            avg_loss_per_trade=avg_loss,
            profit_factor=profit_factor,
            max_drawdown=max_dd,
            sharpe_estimate=sharpe,
            most_active_hour=most_active_hour,
            most_active_day=most_active_day,
            weekend_active=weekend_active,
            consistent_schedule=consistent
        )

    def _calculate_max_drawdown(self, positions: list[WalletPosition]) -> Decimal:
        """Calculate maximum drawdown from position history."""
        if not positions:
            return Decimal("0")

        # Sort by exit time
        sorted_pos = sorted(
            [p for p in positions if p.exit_time],
            key=lambda p: p.exit_time  # type: ignore
        )

        if not sorted_pos:
            return Decimal("0")

        cumulative = Decimal("0")
        peak = Decimal("0")
        max_dd = Decimal("0")

        for pos in sorted_pos:
            cumulative += pos.realized_pnl
            if cumulative > peak:
                peak = cumulative
            dd = peak - cumulative
            if dd > max_dd:
                max_dd = dd

        return max_dd

    def _detect_patterns(
        self,
        summary: WalletSummary,
        behavior: TradingBehavior
    ) -> list[StrategyPattern]:
        """Detect specific trading patterns."""
        patterns = []

        # Scalping pattern
        if behavior.avg_hold_time_hours < 24 and behavior.trades_per_day > 3:
            patterns.append(StrategyPattern(
                pattern_type="scalping",
                confidence=Decimal("0.8"),
                description="Frequent short-term trades seeking small profits",
                supporting_evidence=[
                    f"Average hold time: {behavior.avg_hold_time_hours:.1f} hours",
                    f"Trades per day: {behavior.trades_per_day:.1f}"
                ]
            ))

        # Value buying pattern
        if behavior.buys_dips and behavior.avg_entry_price < Decimal("0.4"):
            patterns.append(StrategyPattern(
                pattern_type="value_buying",
                confidence=Decimal("0.75"),
                description="Tends to buy underpriced outcomes",
                supporting_evidence=[
                    f"Average entry price: {behavior.avg_entry_price:.2f}",
                    "Prefers prices below 0.40"
                ]
            ))

        # Momentum pattern
        if behavior.sells_rallies and not behavior.buys_dips:
            patterns.append(StrategyPattern(
                pattern_type="momentum",
                confidence=Decimal("0.7"),
                description="Buys strength and sells into rallies",
                supporting_evidence=[
                    f"Average exit price: {behavior.avg_exit_price:.2f}",
                    "Sells above 0.50"
                ]
            ))

        # Concentration pattern
        if behavior.position_concentration > 50:
            patterns.append(StrategyPattern(
                pattern_type="concentrated",
                confidence=Decimal("0.85"),
                description="High conviction concentrated positions",
                supporting_evidence=[
                    f"Top position is {behavior.position_concentration:.1f}% of portfolio"
                ]
            ))

        # Diversification pattern
        if summary.unique_markets > 20 and behavior.position_concentration < 20:
            patterns.append(StrategyPattern(
                pattern_type="diversified",
                confidence=Decimal("0.8"),
                description="Spreads risk across many markets",
                supporting_evidence=[
                    f"Active in {summary.unique_markets} markets",
                    f"Max concentration: {behavior.position_concentration:.1f}%"
                ]
            ))

        # Consistent sizing pattern
        if behavior.position_size_std < behavior.avg_position_size * Decimal("0.3"):
            patterns.append(StrategyPattern(
                pattern_type="consistent_sizing",
                confidence=Decimal("0.75"),
                description="Uses consistent position sizes",
                supporting_evidence=[
                    f"Position size std: ${behavior.position_size_std:.2f}",
                    f"Average size: ${behavior.avg_position_size:.2f}"
                ]
            ))

        # Time-based patterns
        if behavior.consistent_schedule:
            patterns.append(StrategyPattern(
                pattern_type="scheduled_trading",
                confidence=Decimal("0.7"),
                description="Trades at consistent times",
                supporting_evidence=[
                    f"Most active at hour {behavior.most_active_hour}:00",
                    f"Most active on day {['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][behavior.most_active_day]}"
                ]
            ))

        # Win streak pattern
        closed = [p for p in summary.positions if p.is_closed]
        if closed:
            streak = self._find_win_streak(closed)
            if streak >= 5:
                patterns.append(StrategyPattern(
                    pattern_type="hot_streak",
                    confidence=Decimal("0.65"),
                    description=f"Had a {streak}-trade winning streak",
                    supporting_evidence=[f"Maximum consecutive wins: {streak}"]
                ))

        return patterns

    def _find_win_streak(self, positions: list[WalletPosition]) -> int:
        """Find longest winning streak."""
        sorted_pos = sorted(
            [p for p in positions if p.exit_time],
            key=lambda p: p.exit_time  # type: ignore
        )

        max_streak = 0
        current_streak = 0

        for pos in sorted_pos:
            if pos.realized_pnl > 0:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0

        return max_streak

    def _classify_strategy(
        self,
        behavior: TradingBehavior,
        patterns: list[StrategyPattern]
    ) -> tuple[StrategyType, list[StrategyType], Decimal]:
        """Classify primary and secondary strategies."""
        scores: dict[StrategyType, Decimal] = {s: Decimal("0") for s in StrategyType}

        # Score based on behavior
        if behavior.avg_hold_time_hours < 12:
            scores[StrategyType.SCALPING] += Decimal("0.4")
        elif behavior.avg_hold_time_hours < 72:
            scores[StrategyType.SWING_TRADING] += Decimal("0.3")
        else:
            scores[StrategyType.VALUE_INVESTING] += Decimal("0.3")

        if behavior.buys_dips and behavior.sells_rallies:
            scores[StrategyType.MEAN_REVERSION] += Decimal("0.4")

        if not behavior.buys_dips and behavior.sells_rallies:
            scores[StrategyType.MOMENTUM] += Decimal("0.35")

        if behavior.trades_per_day > 5:
            scores[StrategyType.SCALPING] += Decimal("0.3")
            scores[StrategyType.MARKET_MAKING] += Decimal("0.2")

        if behavior.position_concentration > 60:
            scores[StrategyType.DEGEN] += Decimal("0.3")

        if behavior.profit_factor > 2:
            scores[StrategyType.EVENT_DRIVEN] += Decimal("0.2")

        # Score based on patterns
        for pattern in patterns:
            if pattern.pattern_type == "scalping":
                scores[StrategyType.SCALPING] += pattern.confidence * Decimal("0.3")
            elif pattern.pattern_type == "value_buying":
                scores[StrategyType.VALUE_INVESTING] += pattern.confidence * Decimal("0.3")
            elif pattern.pattern_type == "momentum":
                scores[StrategyType.MOMENTUM] += pattern.confidence * Decimal("0.3")
            elif pattern.pattern_type == "concentrated":
                scores[StrategyType.DEGEN] += pattern.confidence * Decimal("0.2")
            elif pattern.pattern_type == "diversified":
                scores[StrategyType.DIVERSIFIED] += pattern.confidence * Decimal("0.3")

        # Find primary and secondary
        sorted_strategies = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        primary = sorted_strategies[0][0]
        confidence = sorted_strategies[0][1]

        # Secondary strategies (score > 0.2)
        secondary = [s for s, score in sorted_strategies[1:4] if score > Decimal("0.2")]

        # Normalize confidence to 0-1
        confidence = min(confidence, Decimal("1.0"))

        if confidence < Decimal("0.3"):
            primary = StrategyType.UNKNOWN

        return primary, secondary, confidence

    def _determine_risk_profile(
        self,
        behavior: TradingBehavior,
        summary: WalletSummary
    ) -> RiskProfile:
        """Determine risk profile from behavior."""
        risk_score = Decimal("0")

        # Position concentration
        if behavior.position_concentration > 50:
            risk_score += Decimal("2")
        elif behavior.position_concentration > 30:
            risk_score += Decimal("1")

        # Position size volatility
        if behavior.position_size_std > behavior.avg_position_size:
            risk_score += Decimal("1")

        # Win rate
        if summary.win_count + summary.loss_count > 0:
            win_rate = summary.win_count / (summary.win_count + summary.loss_count)
            if win_rate < 0.4:
                risk_score += Decimal("1")

        # Max drawdown
        if behavior.max_drawdown > behavior.avg_position_size * 3:
            risk_score += Decimal("1.5")

        # Trade frequency
        if behavior.trades_per_day > 10:
            risk_score += Decimal("0.5")

        # Classify
        if risk_score >= 4:
            return RiskProfile.DEGEN
        elif risk_score >= 2.5:
            return RiskProfile.AGGRESSIVE
        elif risk_score >= 1:
            return RiskProfile.MODERATE
        else:
            return RiskProfile.CONSERVATIVE

    def _identify_strengths(
        self,
        behavior: TradingBehavior,
        summary: WalletSummary
    ) -> list[str]:
        """Identify trading strengths."""
        strengths = []

        if summary.win_rate > 55:
            strengths.append(f"Strong win rate of {summary.win_rate:.1f}%")

        if behavior.profit_factor > 1.5:
            strengths.append(f"Excellent profit factor of {behavior.profit_factor:.2f}")

        if behavior.sharpe_estimate > 1:
            strengths.append(f"Good risk-adjusted returns (Sharpe ~{behavior.sharpe_estimate:.2f})")

        if behavior.buys_dips:
            strengths.append("Demonstrates value discipline by buying dips")

        if behavior.consistent_schedule:
            strengths.append("Maintains consistent trading schedule")

        if behavior.position_size_std < behavior.avg_position_size * Decimal("0.5"):
            strengths.append("Good position sizing discipline")

        if summary.unique_markets > 15:
            strengths.append(f"Well-diversified across {summary.unique_markets} markets")

        return strengths[:5]  # Top 5

    def _identify_weaknesses(
        self,
        behavior: TradingBehavior,
        summary: WalletSummary
    ) -> list[str]:
        """Identify areas for improvement."""
        weaknesses = []

        if summary.win_rate < 45:
            weaknesses.append(f"Low win rate of {summary.win_rate:.1f}%")

        if behavior.profit_factor < 1:
            weaknesses.append("Losing more than winning overall")

        if behavior.position_concentration > 50:
            weaknesses.append(f"High concentration risk ({behavior.position_concentration:.1f}% in top position)")

        if behavior.max_drawdown > behavior.avg_position_size * 5:
            weaknesses.append(f"Large drawdown of ${behavior.max_drawdown:.2f}")

        if behavior.avg_loss_per_trade > behavior.avg_profit_per_trade:
            weaknesses.append("Average loss exceeds average profit")

        if behavior.trades_per_day > 10:
            weaknesses.append("High trading frequency may increase costs")

        if not behavior.buys_dips and not behavior.sells_rallies:
            weaknesses.append("No clear entry/exit strategy detected")

        return weaknesses[:5]  # Top 5

    def _generate_recommendations(
        self,
        behavior: TradingBehavior,
        weaknesses: list[str]
    ) -> list[str]:
        """Generate actionable recommendations."""
        recommendations = []

        if behavior.position_concentration > 40:
            recommendations.append("Consider reducing position concentration to below 25%")

        if behavior.profit_factor < 1.5:
            recommendations.append("Focus on improving edge - aim for profit factor > 1.5")

        if behavior.avg_loss_per_trade > behavior.avg_profit_per_trade:
            recommendations.append("Implement tighter stop-losses to reduce average loss size")

        if not behavior.buys_dips:
            recommendations.append("Consider buying at lower prices for better entry points")

        if behavior.trades_per_day > 8:
            recommendations.append("Reduce trading frequency to focus on higher-quality setups")

        if behavior.max_drawdown > behavior.avg_position_size * 3:
            recommendations.append("Implement daily loss limits to control drawdowns")

        if behavior.sharpe_estimate < 0.5:
            recommendations.append("Focus on consistency - volatile returns indicate unclear edge")

        return recommendations[:5]

    def _generate_summary(
        self,
        strategy: StrategyType,
        risk_profile: RiskProfile,
        behavior: TradingBehavior,
        summary: WalletSummary
    ) -> str:
        """Generate human-readable strategy summary."""
        strategy_names = {
            StrategyType.MOMENTUM: "momentum trader",
            StrategyType.MEAN_REVERSION: "mean reversion trader",
            StrategyType.ARBITRAGE: "arbitrageur",
            StrategyType.EVENT_DRIVEN: "event-driven trader",
            StrategyType.SCALPING: "scalper",
            StrategyType.SWING_TRADING: "swing trader",
            StrategyType.VALUE_INVESTING: "value investor",
            StrategyType.MARKET_MAKING: "market maker",
            StrategyType.DEGEN: "high-risk trader",
            StrategyType.DIVERSIFIED: "diversified investor",
            StrategyType.UNKNOWN: "trader with unclear strategy",
        }

        risk_names = {
            RiskProfile.CONSERVATIVE: "conservative",
            RiskProfile.MODERATE: "moderate",
            RiskProfile.AGGRESSIVE: "aggressive",
            RiskProfile.DEGEN: "very high risk",
        }

        return (
            f"This wallet appears to be a {strategy_names[strategy]} with a "
            f"{risk_names[risk_profile]} risk profile. "
            f"Active across {summary.unique_markets} markets with a {summary.win_rate:.1f}% win rate. "
            f"Average position size is ${behavior.avg_position_size:.2f} with "
            f"trades lasting an average of {behavior.avg_hold_time_hours:.1f} hours. "
            f"Total realized P&L: ${summary.total_realized_pnl:+.2f}."
        )

    def _insufficient_data_result(self, summary: WalletSummary) -> DetectedStrategy:
        """Return result when insufficient data for analysis."""
        return DetectedStrategy(
            primary_strategy=StrategyType.UNKNOWN,
            secondary_strategies=[],
            confidence=Decimal("0"),
            risk_profile=RiskProfile.MODERATE,
            behavior=TradingBehavior(
                avg_hold_time_hours=Decimal("0"),
                min_hold_time_hours=Decimal("0"),
                max_hold_time_hours=Decimal("0"),
                trades_per_day=Decimal("0"),
                trades_per_week=Decimal("0"),
                avg_position_size=Decimal("0"),
                position_size_std=Decimal("0"),
                max_position_size=Decimal("0"),
                position_concentration=Decimal("0"),
                avg_entry_price=Decimal("0.5"),
                avg_exit_price=Decimal("0.5"),
                prefers_yes=True,
                buys_dips=False,
                sells_rallies=False,
                avg_profit_per_trade=Decimal("0"),
                avg_loss_per_trade=Decimal("0"),
                profit_factor=Decimal("0"),
                max_drawdown=Decimal("0"),
                sharpe_estimate=Decimal("0"),
                most_active_hour=12,
                most_active_day=0,
                weekend_active=False,
                consistent_schedule=False
            ),
            patterns=[],
            strengths=[],
            weaknesses=["Insufficient transaction history for analysis"],
            recommendations=["Generate more trading activity for meaningful analysis"],
            summary=f"Insufficient data: only {len(summary.transactions)} transactions found. "
                   f"Need at least {self.min_transactions_for_analysis} for analysis."
        )
