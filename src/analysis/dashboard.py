"""Dashboard - CLI and web visualization for wallet analysis."""

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Optional
import asyncio
import json

from .wallet import WalletAnalyzer, WalletSummary
from .strategy import StrategyDetector, DetectedStrategy
from .performance import PerformanceAnalyzer, WalletMetrics


@dataclass
class DashboardData:
    """Complete dashboard data for a wallet."""
    address: str
    summary: WalletSummary
    strategy: DetectedStrategy
    metrics: WalletMetrics
    generated_at: datetime


class WalletDashboard:
    """Generate wallet analysis dashboards."""

    def __init__(self):
        self.analyzer = WalletAnalyzer()
        self.strategy_detector = StrategyDetector()
        self.performance_analyzer = PerformanceAnalyzer()

    async def analyze(self, address: str) -> DashboardData:
        """
        Complete wallet analysis.

        Args:
            address: Wallet address to analyze

        Returns:
            Complete dashboard data
        """
        # Fetch wallet data
        summary = await self.analyzer.analyze_wallet(address)

        # Detect strategy
        strategy = self.strategy_detector.detect_strategy(summary)

        # Calculate metrics
        metrics = self.performance_analyzer.analyze(summary)

        return DashboardData(
            address=address,
            summary=summary,
            strategy=strategy,
            metrics=metrics,
            generated_at=datetime.now()
        )

    async def close(self):
        """Close resources."""
        await self.analyzer.close()

    def render_cli(self, data: DashboardData) -> str:
        """Render dashboard as CLI output."""
        lines = []

        # Header
        lines.extend([
            "",
            "=" * 70,
            f"  POLYMARKET WALLET ANALYSIS DASHBOARD",
            "=" * 70,
            f"  Address: {data.address}",
            f"  Generated: {data.generated_at.strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 70,
            "",
        ])

        # Quick Stats
        s = data.summary
        m = data.metrics
        lines.extend([
            "  QUICK STATS",
            "  " + "-" * 66,
            f"  | Total P&L        | Win Rate         | Sharpe Ratio     |",
            f"  | ${m.total_pnl:>14,.2f} | {m.win_rate:>14.1f}% | {m.sharpe_ratio:>16.2f} |",
            "  " + "-" * 66,
            f"  | Trades           | Volume           | Markets          |",
            f"  | {m.total_trades:>16} | ${s.total_volume:>13,.0f} | {s.unique_markets:>16} |",
            "  " + "-" * 66,
            "",
        ])

        # Strategy Profile
        st = data.strategy
        lines.extend([
            "  STRATEGY PROFILE",
            "  " + "-" * 66,
            f"  Primary Strategy: {st.primary_strategy.value.upper()}",
            f"  Risk Profile:     {st.risk_profile.value.upper()}",
            f"  Confidence:       {float(st.confidence)*100:.0f}%",
            "",
        ])

        if st.secondary_strategies:
            sec = ", ".join(s.value for s in st.secondary_strategies)
            lines.append(f"  Secondary:        {sec}")
            lines.append("")

        # Summary
        lines.extend([
            "  " + st.summary,
            "",
        ])

        # Detected Patterns
        if st.patterns:
            lines.extend([
                "  DETECTED PATTERNS",
                "  " + "-" * 66,
            ])
            for pattern in st.patterns[:5]:
                lines.append(f"  [{pattern.pattern_type.upper()}] ({float(pattern.confidence)*100:.0f}%)")
                lines.append(f"    {pattern.description}")
                for evidence in pattern.supporting_evidence[:2]:
                    lines.append(f"      - {evidence}")
            lines.append("")

        # Trading Behavior
        b = st.behavior
        lines.extend([
            "  TRADING BEHAVIOR",
            "  " + "-" * 66,
            f"  Avg Hold Time:    {float(b.avg_hold_time_hours):.1f} hours",
            f"  Trades/Day:       {float(b.trades_per_day):.2f}",
            f"  Avg Position:     ${float(b.avg_position_size):.2f}",
            f"  Concentration:    {float(b.position_concentration):.1f}% in largest",
            f"  Profit Factor:    {float(b.profit_factor):.2f}",
            "",
        ])

        # Performance Metrics
        lines.extend([
            "  RISK METRICS",
            "  " + "-" * 66,
            f"  Max Drawdown:     ${m.max_drawdown:,.2f} ({m.max_drawdown_pct:.1f}%)",
            f"  Sortino Ratio:    {m.sortino_ratio:.2f}",
            f"  VaR (95%):        {m.var_95:.2f}%",
            f"  Calmar Ratio:     {m.calmar_ratio:.2f}",
            "",
        ])

        # Trade Statistics
        lines.extend([
            "  TRADE STATISTICS",
            "  " + "-" * 66,
            f"  Wins / Losses:    {m.winning_trades} / {m.losing_trades}",
            f"  Avg Win:          ${m.avg_win:,.2f}",
            f"  Avg Loss:         ${m.avg_loss:,.2f}",
            f"  Largest Win:      ${m.largest_win:,.2f}",
            f"  Largest Loss:     ${m.largest_loss:,.2f}",
            f"  Expectancy:       ${m.expectancy:+,.2f}/trade",
            "",
        ])

        # Streak Analysis
        lines.extend([
            "  STREAK ANALYSIS",
            "  " + "-" * 66,
        ])
        if m.current_streak > 0:
            lines.append(f"  Current:          {m.current_streak} wins in a row")
        elif m.current_streak < 0:
            lines.append(f"  Current:          {abs(m.current_streak)} losses in a row")
        else:
            lines.append(f"  Current:          Neutral")
        lines.append(f"  Best Win Streak:  {m.longest_win_streak}")
        lines.append(f"  Worst Loss Streak:{m.longest_loss_streak}")
        lines.append("")

        # P&L by Timeframe
        lines.extend([
            "  P&L BY TIMEFRAME",
            "  " + "-" * 66,
        ])
        for tf in m.pnl_by_timeframe:
            wr = (tf.win_count / (tf.win_count + tf.loss_count) * 100) if (tf.win_count + tf.loss_count) > 0 else 0
            lines.append(
                f"  {tf.period:>6}: ${tf.total_pnl:>+10,.2f}  |  {tf.trade_count:>3} trades  |  {wr:.0f}% win"
            )
        lines.append("")

        # Market Performance
        if m.performance_by_market:
            lines.extend([
                "  PERFORMANCE BY CATEGORY",
                "  " + "-" * 66,
            ])
            for mp in m.performance_by_market[:5]:
                lines.append(
                    f"  {mp.category:>12}: ${mp.total_pnl:>+10,.2f}  |  {mp.trade_count:>3} trades  |  {mp.win_rate:.0f}% win"
                )
            lines.append("")

        # Strengths
        if st.strengths:
            lines.extend([
                "  STRENGTHS",
                "  " + "-" * 66,
            ])
            for strength in st.strengths:
                lines.append(f"  + {strength}")
            lines.append("")

        # Weaknesses
        if st.weaknesses:
            lines.extend([
                "  AREAS FOR IMPROVEMENT",
                "  " + "-" * 66,
            ])
            for weakness in st.weaknesses:
                lines.append(f"  - {weakness}")
            lines.append("")

        # Recommendations
        if st.recommendations:
            lines.extend([
                "  RECOMMENDATIONS",
                "  " + "-" * 66,
            ])
            for i, rec in enumerate(st.recommendations, 1):
                lines.append(f"  {i}. {rec}")
            lines.append("")

        # Recent Positions
        if data.summary.positions:
            lines.extend([
                "  RECENT POSITIONS (Last 10)",
                "  " + "-" * 66,
            ])
            recent = sorted(
                [p for p in data.summary.positions if p.exit_time],
                key=lambda p: p.exit_time,  # type: ignore
                reverse=True
            )[:10]

            for pos in recent:
                question = pos.market_question[:35] + "..." if len(pos.market_question) > 35 else pos.market_question
                status = "+" if pos.realized_pnl > 0 else "-"
                lines.append(
                    f"  {status} ${abs(pos.realized_pnl):>8,.2f}  {pos.outcome:>3}  {question}"
                )
            lines.append("")

        # Footer
        lines.extend([
            "=" * 70,
            "  Generated by Alpha Arena Wallet Analysis",
            "=" * 70,
            "",
        ])

        return "\n".join(lines)

    def render_json(self, data: DashboardData) -> str:
        """Render dashboard as JSON."""
        def decimal_to_float(obj):
            if isinstance(obj, Decimal):
                return float(obj)
            elif isinstance(obj, datetime):
                return obj.isoformat()
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

        output = {
            "address": data.address,
            "generated_at": data.generated_at.isoformat(),
            "summary": {
                "total_transactions": data.summary.total_transactions,
                "total_volume": float(data.summary.total_volume),
                "total_pnl": float(data.summary.total_realized_pnl),
                "unique_markets": data.summary.unique_markets,
                "open_positions": data.summary.open_positions,
                "closed_positions": data.summary.closed_positions,
                "win_rate": float(data.summary.win_rate),
            },
            "strategy": {
                "primary": data.strategy.primary_strategy.value,
                "secondary": [s.value for s in data.strategy.secondary_strategies],
                "confidence": float(data.strategy.confidence),
                "risk_profile": data.strategy.risk_profile.value,
                "summary": data.strategy.summary,
                "patterns": [
                    {
                        "type": p.pattern_type,
                        "confidence": float(p.confidence),
                        "description": p.description
                    }
                    for p in data.strategy.patterns
                ],
                "strengths": data.strategy.strengths,
                "weaknesses": data.strategy.weaknesses,
                "recommendations": data.strategy.recommendations,
            },
            "metrics": {
                "total_pnl": float(data.metrics.total_pnl),
                "total_return_pct": float(data.metrics.total_return_pct),
                "sharpe_ratio": float(data.metrics.sharpe_ratio),
                "sortino_ratio": float(data.metrics.sortino_ratio),
                "max_drawdown": float(data.metrics.max_drawdown),
                "max_drawdown_pct": float(data.metrics.max_drawdown_pct),
                "win_rate": float(data.metrics.win_rate),
                "profit_factor": float(data.metrics.profit_factor),
                "expectancy": float(data.metrics.expectancy),
                "total_trades": data.metrics.total_trades,
                "winning_trades": data.metrics.winning_trades,
                "losing_trades": data.metrics.losing_trades,
                "avg_win": float(data.metrics.avg_win),
                "avg_loss": float(data.metrics.avg_loss),
                "current_streak": data.metrics.current_streak,
            },
            "behavior": {
                "avg_hold_hours": float(data.strategy.behavior.avg_hold_time_hours),
                "trades_per_day": float(data.strategy.behavior.trades_per_day),
                "avg_position_size": float(data.strategy.behavior.avg_position_size),
                "position_concentration": float(data.strategy.behavior.position_concentration),
                "buys_dips": data.strategy.behavior.buys_dips,
                "sells_rallies": data.strategy.behavior.sells_rallies,
            },
            "timeframes": [
                {
                    "period": tf.period,
                    "pnl": float(tf.total_pnl),
                    "trades": tf.trade_count,
                    "win_rate": float(tf.win_count / (tf.win_count + tf.loss_count) * 100) if (tf.win_count + tf.loss_count) > 0 else 0
                }
                for tf in data.metrics.pnl_by_timeframe
            ],
            "categories": [
                {
                    "category": mp.category,
                    "pnl": float(mp.total_pnl),
                    "trades": mp.trade_count,
                    "win_rate": float(mp.win_rate)
                }
                for mp in data.metrics.performance_by_market
            ],
        }

        return json.dumps(output, indent=2)

    def render_html(self, data: DashboardData) -> str:
        """Render dashboard as HTML."""
        s = data.summary
        st = data.strategy
        m = data.metrics
        b = st.behavior

        # Color helper
        def pnl_color(val: Decimal) -> str:
            return "#22c55e" if val >= 0 else "#ef4444"

        html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Wallet Analysis - {data.address[:10]}...</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0f172a;
            color: #e2e8f0;
            line-height: 1.6;
            padding: 2rem;
        }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .header {{
            text-align: center;
            margin-bottom: 2rem;
            padding: 2rem;
            background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
            border-radius: 1rem;
            border: 1px solid #334155;
        }}
        .header h1 {{ font-size: 1.5rem; color: #f8fafc; margin-bottom: 0.5rem; }}
        .header .address {{ font-family: monospace; color: #94a3b8; font-size: 0.9rem; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 1.5rem; }}
        .card {{
            background: #1e293b;
            border-radius: 0.75rem;
            padding: 1.5rem;
            border: 1px solid #334155;
        }}
        .card h2 {{
            font-size: 0.875rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #94a3b8;
            margin-bottom: 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 1px solid #334155;
        }}
        .metric {{
            display: flex;
            justify-content: space-between;
            padding: 0.5rem 0;
            border-bottom: 1px solid #334155;
        }}
        .metric:last-child {{ border-bottom: none; }}
        .metric-label {{ color: #94a3b8; }}
        .metric-value {{ font-weight: 600; }}
        .positive {{ color: #22c55e; }}
        .negative {{ color: #ef4444; }}
        .neutral {{ color: #f59e0b; }}
        .big-number {{
            font-size: 2rem;
            font-weight: 700;
            margin-bottom: 0.25rem;
        }}
        .big-label {{ color: #94a3b8; font-size: 0.875rem; }}
        .strategy-badge {{
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
            margin-right: 0.5rem;
            margin-bottom: 0.5rem;
        }}
        .badge-primary {{ background: #3b82f6; color: white; }}
        .badge-secondary {{ background: #334155; color: #94a3b8; }}
        .badge-risk-conservative {{ background: #22c55e22; color: #22c55e; border: 1px solid #22c55e; }}
        .badge-risk-moderate {{ background: #f59e0b22; color: #f59e0b; border: 1px solid #f59e0b; }}
        .badge-risk-aggressive {{ background: #ef444422; color: #ef4444; border: 1px solid #ef4444; }}
        .badge-risk-degen {{ background: #dc262622; color: #dc2626; border: 1px solid #dc2626; }}
        .pattern {{
            background: #0f172a;
            padding: 0.75rem;
            border-radius: 0.5rem;
            margin-bottom: 0.5rem;
        }}
        .pattern-header {{ display: flex; justify-content: space-between; margin-bottom: 0.25rem; }}
        .pattern-type {{ font-weight: 600; color: #f8fafc; }}
        .pattern-confidence {{ color: #94a3b8; font-size: 0.875rem; }}
        .pattern-desc {{ color: #94a3b8; font-size: 0.875rem; }}
        .list {{ list-style: none; }}
        .list li {{ padding: 0.5rem 0; padding-left: 1.25rem; position: relative; }}
        .list li::before {{
            content: "";
            position: absolute;
            left: 0;
            top: 0.9rem;
            width: 6px;
            height: 6px;
            border-radius: 50%;
        }}
        .list.strengths li::before {{ background: #22c55e; }}
        .list.weaknesses li::before {{ background: #ef4444; }}
        .list.recommendations li::before {{ background: #3b82f6; }}
        .timeframe-row {{
            display: grid;
            grid-template-columns: 60px 1fr 80px 60px;
            padding: 0.5rem 0;
            border-bottom: 1px solid #334155;
            align-items: center;
        }}
        .timeframe-row:last-child {{ border-bottom: none; }}
        .bar-container {{ height: 8px; background: #334155; border-radius: 4px; overflow: hidden; }}
        .bar {{ height: 100%; border-radius: 4px; }}
        .summary-text {{
            color: #94a3b8;
            font-size: 0.9rem;
            line-height: 1.7;
            padding: 1rem;
            background: #0f172a;
            border-radius: 0.5rem;
        }}
        .footer {{
            text-align: center;
            margin-top: 2rem;
            padding-top: 2rem;
            border-top: 1px solid #334155;
            color: #64748b;
            font-size: 0.875rem;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Polymarket Wallet Analysis</h1>
            <div class="address">{data.address}</div>
        </div>

        <div class="grid">
            <!-- Quick Stats -->
            <div class="card">
                <h2>Performance</h2>
                <div class="big-number" style="color: {pnl_color(m.total_pnl)}">
                    ${float(m.total_pnl):+,.2f}
                </div>
                <div class="big-label">Total P&L</div>
                <div style="margin-top: 1rem;">
                    <div class="metric">
                        <span class="metric-label">Return</span>
                        <span class="metric-value" style="color: {pnl_color(m.total_return_pct)}">{float(m.total_return_pct):+.1f}%</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Win Rate</span>
                        <span class="metric-value">{float(m.win_rate):.1f}%</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Total Trades</span>
                        <span class="metric-value">{m.total_trades}</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Volume</span>
                        <span class="metric-value">${float(s.total_volume):,.0f}</span>
                    </div>
                </div>
            </div>

            <!-- Strategy -->
            <div class="card">
                <h2>Strategy Profile</h2>
                <div style="margin-bottom: 1rem;">
                    <span class="strategy-badge badge-primary">{st.primary_strategy.value}</span>
                    <span class="strategy-badge badge-risk-{st.risk_profile.value}">{st.risk_profile.value}</span>
                </div>
                {' '.join(f'<span class="strategy-badge badge-secondary">{s.value}</span>' for s in st.secondary_strategies)}
                <div class="summary-text" style="margin-top: 1rem;">
                    {st.summary}
                </div>
            </div>

            <!-- Risk Metrics -->
            <div class="card">
                <h2>Risk Metrics</h2>
                <div class="metric">
                    <span class="metric-label">Sharpe Ratio</span>
                    <span class="metric-value">{float(m.sharpe_ratio):.2f}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Sortino Ratio</span>
                    <span class="metric-value">{float(m.sortino_ratio):.2f}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Max Drawdown</span>
                    <span class="metric-value negative">${float(m.max_drawdown):,.2f}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Profit Factor</span>
                    <span class="metric-value">{float(m.profit_factor):.2f}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Expectancy</span>
                    <span class="metric-value" style="color: {pnl_color(m.expectancy)}">${float(m.expectancy):+,.2f}</span>
                </div>
            </div>

            <!-- Trading Behavior -->
            <div class="card">
                <h2>Trading Behavior</h2>
                <div class="metric">
                    <span class="metric-label">Avg Hold Time</span>
                    <span class="metric-value">{float(b.avg_hold_time_hours):.1f} hours</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Trades/Day</span>
                    <span class="metric-value">{float(b.trades_per_day):.2f}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Avg Position</span>
                    <span class="metric-value">${float(b.avg_position_size):.2f}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Concentration</span>
                    <span class="metric-value">{float(b.position_concentration):.1f}%</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Style</span>
                    <span class="metric-value">
                        {'Buys dips' if b.buys_dips else 'Buys strength'}
                    </span>
                </div>
            </div>
        </div>

        <!-- Patterns -->
        <div class="card" style="margin-top: 1.5rem;">
            <h2>Detected Patterns</h2>
            <div class="grid" style="grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));">
                {''.join(f"""
                <div class="pattern">
                    <div class="pattern-header">
                        <span class="pattern-type">{p.pattern_type.upper()}</span>
                        <span class="pattern-confidence">{float(p.confidence)*100:.0f}%</span>
                    </div>
                    <div class="pattern-desc">{p.description}</div>
                </div>
                """ for p in st.patterns[:6])}
            </div>
        </div>

        <!-- P&L by Timeframe -->
        <div class="card" style="margin-top: 1.5rem;">
            <h2>P&L by Timeframe</h2>
            {''.join(f"""
            <div class="timeframe-row">
                <span style="color: #94a3b8;">{tf.period}</span>
                <div class="bar-container">
                    <div class="bar" style="width: {min(abs(float(tf.total_pnl)) / max(abs(float(m.total_pnl)), 1) * 100, 100)}%; background: {pnl_color(tf.total_pnl)};"></div>
                </div>
                <span style="color: {pnl_color(tf.total_pnl)}; text-align: right;">${float(tf.total_pnl):+,.0f}</span>
                <span style="color: #94a3b8; text-align: right;">{tf.trade_count}t</span>
            </div>
            """ for tf in m.pnl_by_timeframe)}
        </div>

        <div class="grid" style="margin-top: 1.5rem;">
            <!-- Strengths -->
            <div class="card">
                <h2>Strengths</h2>
                <ul class="list strengths">
                    {''.join(f"<li>{s}</li>" for s in st.strengths) or "<li>Not enough data</li>"}
                </ul>
            </div>

            <!-- Weaknesses -->
            <div class="card">
                <h2>Areas for Improvement</h2>
                <ul class="list weaknesses">
                    {''.join(f"<li>{w}</li>" for w in st.weaknesses) or "<li>Looking good!</li>"}
                </ul>
            </div>

            <!-- Recommendations -->
            <div class="card">
                <h2>Recommendations</h2>
                <ul class="list recommendations">
                    {''.join(f"<li>{r}</li>" for r in st.recommendations) or "<li>Keep doing what you're doing</li>"}
                </ul>
            </div>
        </div>

        <div class="footer">
            Generated by Alpha Arena Wallet Analysis<br>
            {data.generated_at.strftime('%Y-%m-%d %H:%M:%S')}
        </div>
    </div>
</body>
</html>'''
        return html


async def run_dashboard(address: str, output_format: str = "cli") -> str:
    """
    Run wallet analysis and return formatted output.

    Args:
        address: Wallet address
        output_format: "cli", "json", or "html"

    Returns:
        Formatted output string
    """
    dashboard = WalletDashboard()

    try:
        data = await dashboard.analyze(address)

        if output_format == "json":
            return dashboard.render_json(data)
        elif output_format == "html":
            return dashboard.render_html(data)
        else:
            return dashboard.render_cli(data)
    finally:
        await dashboard.close()
