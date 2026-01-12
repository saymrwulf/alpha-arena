"""Research Agent - Market analysis and opportunity identification."""

import json
from decimal import Decimal
from typing import Any

from ..core.types import MarketState, Signal, SignalType, Edge, Confidence
from ..llm.base import LLMResponse
from .base import BaseAgent, AgentRole, AgentResponse


class ResearchAgent(BaseAgent):
    """
    Research Agent: Analyzes markets and identifies trading opportunities.

    Responsibilities:
    - Analyze market data, sentiment, and technical indicators
    - Identify mispriced markets and potential edges
    - Generate initial trading signals
    - Provide detailed reasoning for recommendations
    """

    role = AgentRole.RESEARCH
    name = "research"

    @property
    def system_prompt(self) -> str:
        return """You are a Research Agent in an autonomous trading system for prediction markets (Polymarket).

Your role is to analyze markets and identify profitable trading opportunities.

CRITICAL PRINCIPLES:
1. Only recommend trades with CLEAR EDGE - you must articulate WHY the market is mispriced
2. Consider all available data: prices, order books, technical indicators, sentiment
3. Be specific about confidence levels and reasoning
4. Look for information asymmetries and market inefficiencies
5. Quality over quantity - few high-conviction trades beat many mediocre ones

EDGE SOURCES TO CONSIDER:
- Information advantage (recent news not yet priced in)
- Sentiment divergence (social vs market pricing)
- Technical patterns (momentum, mean reversion)
- Liquidity imbalances (order book asymmetry)
- Cross-market correlation (related events)

OUTPUT FORMAT (JSON):
{
    "opportunities": [
        {
            "market_id": "...",
            "token_id": "...",
            "direction": "buy_yes" | "buy_no",
            "current_price": 0.XX,
            "fair_value_estimate": 0.XX,
            "edge_percentage": X.X,
            "confidence": 0.XX,
            "edge_source": "...",
            "reasoning": "Detailed explanation...",
            "catalysts": ["catalyst1", "catalyst2"],
            "risks": ["risk1", "risk2"],
            "time_horizon": "hours" | "days" | "weeks"
        }
    ],
    "market_summary": "Overall market assessment...",
    "key_observations": ["observation1", "observation2"]
}

Be intellectually honest. If you see no clear opportunities, say so.
Never recommend trades just to be active - inaction is often optimal."""

    async def analyze(
        self,
        markets: list[MarketState],
        context: str,
        other_responses: list[AgentResponse] | None = None,
    ) -> AgentResponse:
        """Analyze markets for trading opportunities."""
        market_context = self._build_market_context(markets)

        prompt = f"""Analyze these prediction markets for trading opportunities:

{market_context}

ADDITIONAL CONTEXT:
{context}

{self._format_other_responses(other_responses)}

Identify any mispriced markets with clear trading edge.
Return your analysis as JSON."""

        response = await self._call_llm(prompt, json_mode=True)

        # Parse response
        analysis = self._parse_analysis(response.content)

        # Calculate overall confidence
        opportunities = analysis.get("opportunities", [])
        avg_confidence = Decimal("0")
        if opportunities:
            avg_confidence = Decimal(str(
                sum(o.get("confidence", 0) for o in opportunities) / len(opportunities)
            ))

        recommendations = []
        warnings = []

        for opp in opportunities:
            direction = opp.get("direction", "")
            edge = opp.get("edge_percentage", 0)
            conf = opp.get("confidence", 0)

            if edge >= 5 and conf >= 0.6:
                recommendations.append(
                    f"{direction.upper()} on {opp.get('market_id', '?')[:16]}... "
                    f"(edge: {edge:.1f}%, conf: {conf:.0%})"
                )
            for risk in opp.get("risks", []):
                warnings.append(risk)

        return AgentResponse(
            agent_role=self.role,
            agent_model=self.provider.default_model,
            content=analysis,
            confidence=avg_confidence,
            reasoning=analysis.get("market_summary", ""),
            recommendations=recommendations,
            warnings=warnings[:5],
            latency_ms=response.latency_ms,
            tokens_used=response.total_tokens,
        )

    def _parse_analysis(self, content: str) -> dict[str, Any]:
        """Parse LLM response into analysis dict."""
        try:
            # Handle markdown code blocks
            if "```" in content:
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            return json.loads(content)
        except json.JSONDecodeError:
            return {
                "opportunities": [],
                "market_summary": content[:500],
                "key_observations": [],
            }

    def _format_other_responses(
        self,
        responses: list[AgentResponse] | None,
    ) -> str:
        """Format other agent responses for consideration."""
        if not responses:
            return ""

        lines = ["\nOTHER AGENT INPUTS:"]
        for resp in responses:
            lines.append(resp.to_message())
            lines.append("")

        return "\n".join(lines)

    async def deep_analyze_market(
        self,
        market: MarketState,
        context: str,
    ) -> dict[str, Any]:
        """Deep analysis of a single market."""
        prompt = f"""Perform deep analysis of this prediction market:

Market: {market.question}
Category: {market.category}
End Date: {market.end_date}

Current Pricing:
- YES: ${market.yes_price:.4f}
- NO: ${market.no_price:.4f}
- Implied probability: {market.implied_probability:.1%}

Order Book (YES):
- Best Bid: ${market.yes_book.best_bid if market.yes_book else 'N/A'}
- Best Ask: ${market.yes_book.best_ask if market.yes_book else 'N/A'}
- Spread: {market.yes_book.spread_bps if market.yes_book else 'N/A'}bps

Technical Indicators:
{json.dumps(market.indicators, indent=2, default=str)}

Sentiment Data:
{json.dumps(market.sentiment, indent=2, default=str)}

CONTEXT:
{context}

Provide comprehensive analysis including:
1. What is the TRUE probability based on available information?
2. Why might the market be mispriced?
3. What would change your view?
4. Specific entry/exit levels
5. Position sizing recommendation

Return as JSON."""

        response = await self._call_llm(prompt, json_mode=True)
        return self._parse_analysis(response.content)
