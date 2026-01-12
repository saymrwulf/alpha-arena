"""Execution Agent - Trade execution optimization."""

import json
from decimal import Decimal
from typing import Any

from ..core.types import MarketState, OrderBook
from .base import BaseAgent, AgentRole, AgentResponse


class ExecutionAgent(BaseAgent):
    """
    Execution Agent: Optimizes trade execution.

    Responsibilities:
    - Analyze order book for optimal entry/exit
    - Minimize slippage and market impact
    - Determine order type (limit vs market)
    - Split large orders if needed
    - Time entries for best execution
    """

    role = AgentRole.EXECUTION
    name = "execution"

    @property
    def system_prompt(self) -> str:
        return """You are an Execution Agent in an autonomous trading system for prediction markets.

Your role is to OPTIMIZE TRADE EXECUTION to minimize costs and slippage.

KEY RESPONSIBILITIES:
1. Analyze order book depth and liquidity
2. Determine optimal order placement strategy
3. Calculate expected slippage and market impact
4. Recommend order splitting for large trades
5. Time entries to minimize adverse selection

EXECUTION STRATEGIES:
- Passive (limit orders): Lower cost but risk of non-fill
- Aggressive (market orders): Certain fill but higher cost
- TWAP: Split orders over time for large sizes
- Iceberg: Hide true size to reduce market impact

ORDER BOOK ANALYSIS:
- Check depth at each price level
- Calculate cost to fill target size
- Identify support/resistance levels
- Detect potential spoofing or manipulation

OUTPUT FORMAT (JSON):
{
    "execution_plan": [
        {
            "market_id": "...",
            "token_id": "...",
            "side": "buy|sell",
            "total_size": XX.XX,
            "strategy": "passive|aggressive|twap|iceberg",
            "orders": [
                {
                    "type": "limit|market",
                    "price": 0.XX,
                    "size": XX.XX,
                    "urgency": "low|medium|high"
                }
            ],
            "expected_avg_price": 0.XX,
            "expected_slippage_bps": XX,
            "fill_probability": 0.XX,
            "time_to_fill_estimate": "Xm|Xh",
            "max_acceptable_price": 0.XX
        }
    ],
    "market_conditions": {
        "liquidity_assessment": "thin|moderate|deep",
        "spread_assessment": "tight|normal|wide",
        "momentum": "bullish|neutral|bearish",
        "recommended_timing": "immediate|wait_for_dip|scale_in"
    },
    "warnings": ["warning1"]
}

MINIMIZE EXECUTION COSTS - every basis point matters for edge."""

    async def analyze(
        self,
        markets: list[MarketState],
        context: str,
        other_responses: list[AgentResponse] | None = None,
    ) -> AgentResponse:
        """Plan trade execution."""
        # Get approved trades from Risk Agent
        approved_trades = []
        if other_responses:
            for resp in other_responses:
                if resp.agent_role == AgentRole.RISK:
                    for trade in resp.content.get("approved_trades", []):
                        if trade.get("approved"):
                            approved_trades.append(trade)

        if not approved_trades:
            return AgentResponse(
                agent_role=self.role,
                agent_model=self.provider.default_model,
                content={"execution_plan": [], "market_conditions": {}},
                confidence=Decimal("1.0"),
                reasoning="No approved trades to execute.",
                recommendations=["No execution required"],
                warnings=[],
            )

        # Build order book context
        orderbook_context = self._build_orderbook_context(markets, approved_trades)

        prompt = f"""Plan optimal execution for these approved trades:

APPROVED TRADES:
{json.dumps(approved_trades, indent=2, default=str)}

ORDER BOOK DATA:
{orderbook_context}

ADDITIONAL CONTEXT:
{context}

For each trade, determine:
1. Best execution strategy given order book
2. Optimal price levels for limit orders
3. Expected slippage and fill probability
4. Whether to split the order

Return your execution plan as JSON."""

        response = await self._call_llm(prompt, json_mode=True)
        plan = self._parse_plan(response.content)

        recommendations = []
        warnings = plan.get("warnings", [])

        for execution in plan.get("execution_plan", []):
            strategy = execution.get("strategy", "passive")
            slippage = execution.get("expected_slippage_bps", 0)
            recommendations.append(
                f"{strategy.upper()}: {execution.get('market_id', '?')[:16]}... "
                f"expected slippage: {slippage}bps"
            )

        return AgentResponse(
            agent_role=self.role,
            agent_model=self.provider.default_model,
            content=plan,
            confidence=Decimal("0.8"),
            reasoning=f"Execution plan for {len(approved_trades)} trades",
            recommendations=recommendations,
            warnings=warnings,
            latency_ms=response.latency_ms,
            tokens_used=response.total_tokens,
        )

    def _build_orderbook_context(
        self,
        markets: list[MarketState],
        trades: list[dict],
    ) -> str:
        """Build order book context for relevant markets."""
        lines = []
        market_ids = {t.get("market_id") for t in trades}

        for market in markets:
            if market.market_id not in market_ids:
                continue

            lines.append(f"\nMarket: {market.market_id[:20]}...")

            if market.yes_book:
                lines.append("YES Order Book:")
                lines.append(f"  Best Bid: ${market.yes_book.best_bid} | Best Ask: ${market.yes_book.best_ask}")
                lines.append(f"  Spread: {market.yes_book.spread_bps:.0f}bps")
                lines.append("  Bids: " + ", ".join(
                    f"${l.price:.3f}x{l.size:.0f}" for l in market.yes_book.bids[:5]
                ))
                lines.append("  Asks: " + ", ".join(
                    f"${l.price:.3f}x{l.size:.0f}" for l in market.yes_book.asks[:5]
                ))

            if market.no_book:
                lines.append("NO Order Book:")
                lines.append(f"  Best Bid: ${market.no_book.best_bid} | Best Ask: ${market.no_book.best_ask}")
                lines.append(f"  Spread: {market.no_book.spread_bps:.0f}bps")

        return "\n".join(lines)

    def _parse_plan(self, content: str) -> dict[str, Any]:
        """Parse execution plan response."""
        try:
            if "```" in content:
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            return json.loads(content)
        except json.JSONDecodeError:
            return {
                "execution_plan": [],
                "warnings": ["Failed to parse execution plan"],
            }

    def calculate_slippage(
        self,
        orderbook: OrderBook,
        side: str,
        size: Decimal,
    ) -> tuple[Decimal, Decimal]:
        """
        Calculate expected slippage for a given order size.

        Returns (average_price, slippage_bps).
        """
        levels = orderbook.asks if side == "buy" else orderbook.bids
        if not levels:
            return Decimal("0"), Decimal("0")

        mid = orderbook.mid_price or levels[0].price
        remaining = size
        total_cost = Decimal("0")

        for level in levels:
            fill_size = min(remaining, level.size)
            total_cost += fill_size * level.price
            remaining -= fill_size
            if remaining <= 0:
                break

        if remaining > 0:
            # Not enough liquidity
            return Decimal("0"), Decimal("10000")  # 100% slippage indicator

        filled_size = size - remaining
        avg_price = total_cost / filled_size if filled_size > 0 else Decimal("0")
        slippage_bps = ((avg_price - mid) / mid * 10000) if mid > 0 else Decimal("0")

        return avg_price, abs(slippage_bps)

    def recommend_order_type(
        self,
        orderbook: OrderBook,
        size: Decimal,
        urgency: str = "medium",
    ) -> dict[str, Any]:
        """Recommend order type based on conditions."""
        spread_bps = orderbook.spread_bps or Decimal("0")

        # Calculate slippage for market order
        _, market_slippage = self.calculate_slippage(orderbook, "buy", size)

        # Decision logic
        if urgency == "high":
            return {
                "type": "market",
                "reason": "High urgency requires immediate fill",
                "expected_slippage_bps": float(market_slippage),
            }

        if spread_bps < 50 and market_slippage < 20:
            return {
                "type": "market",
                "reason": "Tight spread and low slippage favor market order",
                "expected_slippage_bps": float(market_slippage),
            }

        if spread_bps > 200:
            return {
                "type": "limit",
                "price": float(orderbook.mid_price or 0),
                "reason": "Wide spread favors passive limit order",
                "expected_slippage_bps": 0,
            }

        return {
            "type": "limit",
            "price": float(orderbook.best_bid if orderbook.best_bid else 0),
            "reason": "Moderate conditions favor limit order at best bid",
            "expected_slippage_bps": 0,
        }
