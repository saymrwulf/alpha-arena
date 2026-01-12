"""Risk Agent - Risk assessment and position sizing."""

import json
from decimal import Decimal
from typing import Any

from ..core.types import MarketState, Edge, Position
from .base import BaseAgent, AgentRole, AgentResponse


class RiskAgent(BaseAgent):
    """
    Risk Agent: Assesses risk and determines optimal position sizing.

    Responsibilities:
    - Validate trading signals from Research Agent
    - Calculate Kelly-optimal position sizes
    - Monitor portfolio risk and correlation
    - Enforce risk limits and stop losses
    - Veto trades that exceed risk parameters
    """

    role = AgentRole.RISK
    name = "risk"

    def __init__(self, *args, risk_config: dict | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.risk_config = risk_config or {
            "max_position_pct": 0.10,
            "max_portfolio_risk": 0.20,
            "kelly_fraction": 0.25,
            "min_edge": 0.03,
            "max_correlation": 0.5,
        }

    @property
    def system_prompt(self) -> str:
        return f"""You are a Risk Agent in an autonomous trading system for prediction markets.

Your role is CRITICAL: Protect capital and ensure survival.

RISK PARAMETERS:
- Max position size: {self.risk_config['max_position_pct']:.0%} of portfolio
- Max portfolio risk: {self.risk_config['max_portfolio_risk']:.0%}
- Kelly fraction: {self.risk_config['kelly_fraction']} (use quarter Kelly)
- Minimum edge required: {self.risk_config['min_edge']:.1%}
- Max correlation exposure: {self.risk_config['max_correlation']}

KELLY CRITERION:
f* = (bp - q) / b
where:
  b = net odds (win_amount / loss_amount)
  p = probability of winning
  q = probability of losing (1 - p)
  f* = fraction of capital to bet

Use QUARTER Kelly (f*/4) for safety against estimation errors.

YOUR RESPONSIBILITIES:
1. Validate edge estimates from Research Agent
2. Calculate appropriate position sizes
3. Check portfolio correlation and concentration
4. Identify and flag excessive risks
5. VETO trades that violate risk parameters
6. Recommend stop-loss and take-profit levels

OUTPUT FORMAT (JSON):
{{
    "approved_trades": [
        {{
            "market_id": "...",
            "approved": true|false,
            "reason": "...",
            "kelly_fraction": 0.XX,
            "recommended_size_pct": 0.XX,
            "recommended_size_usdc": XX.XX,
            "stop_loss_price": 0.XX,
            "take_profit_price": 0.XX,
            "max_loss_usdc": XX.XX,
            "risk_reward_ratio": X.X
        }}
    ],
    "portfolio_risk_assessment": {{
        "current_exposure_pct": 0.XX,
        "projected_exposure_pct": 0.XX,
        "correlation_risk": "low|medium|high",
        "concentration_risk": "low|medium|high"
    }},
    "overall_recommendation": "proceed|reduce_size|reject",
    "warnings": ["warning1", "warning2"]
}}

BE CONSERVATIVE. Rejecting a good trade is better than accepting a bad one.
When in doubt, reduce size or reject entirely."""

    async def analyze(
        self,
        markets: list[MarketState],
        context: str,
        other_responses: list[AgentResponse] | None = None,
    ) -> AgentResponse:
        """Assess risk for proposed trades."""
        # Extract opportunities from Research Agent
        opportunities = []
        if other_responses:
            for resp in other_responses:
                if resp.agent_role == AgentRole.RESEARCH:
                    opportunities = resp.content.get("opportunities", [])

        if not opportunities:
            return AgentResponse(
                agent_role=self.role,
                agent_model=self.provider.default_model,
                content={"approved_trades": [], "overall_recommendation": "no_trades"},
                confidence=Decimal("1.0"),
                reasoning="No trading opportunities to assess.",
                recommendations=["No action required"],
                warnings=[],
            )

        prompt = f"""Assess risk for these proposed trades:

PROPOSED OPPORTUNITIES:
{json.dumps(opportunities, indent=2, default=str)}

CURRENT PORTFOLIO CONTEXT:
{context}

MARKET DATA:
{self._build_market_context(markets)}

For each opportunity:
1. Validate the edge estimate - is it realistic?
2. Calculate Kelly-optimal position size
3. Set stop-loss and take-profit levels
4. Assess correlation with existing positions
5. Determine if trade should be approved

Return your risk assessment as JSON."""

        response = await self._call_llm(prompt, json_mode=True)
        assessment = self._parse_assessment(response.content)

        # Build recommendations
        recommendations = []
        warnings = assessment.get("warnings", [])

        approved = assessment.get("approved_trades", [])
        for trade in approved:
            if trade.get("approved"):
                recommendations.append(
                    f"APPROVE {trade.get('market_id', '?')[:16]}... "
                    f"size: ${trade.get('recommended_size_usdc', 0):.2f}"
                )
            else:
                warnings.append(f"REJECT: {trade.get('reason', 'Risk too high')}")

        overall = assessment.get("overall_recommendation", "reject")
        confidence = Decimal("0.8") if overall == "proceed" else Decimal("0.5")

        return AgentResponse(
            agent_role=self.role,
            agent_model=self.provider.default_model,
            content=assessment,
            confidence=confidence,
            reasoning=f"Risk assessment: {overall}",
            recommendations=recommendations,
            warnings=warnings,
            latency_ms=response.latency_ms,
            tokens_used=response.total_tokens,
        )

    def _parse_assessment(self, content: str) -> dict[str, Any]:
        """Parse risk assessment response."""
        try:
            if "```" in content:
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            return json.loads(content)
        except json.JSONDecodeError:
            return {
                "approved_trades": [],
                "overall_recommendation": "reject",
                "warnings": ["Failed to parse risk assessment"],
            }

    def calculate_kelly(
        self,
        win_prob: Decimal,
        win_amount: Decimal,
        loss_amount: Decimal,
    ) -> Edge:
        """Calculate Kelly criterion position size."""
        return Edge.calculate(win_prob, win_amount, loss_amount, source="kelly")

    def calculate_position_size(
        self,
        edge: Edge,
        portfolio_value: Decimal,
        current_exposure: Decimal,
    ) -> Decimal:
        """
        Calculate recommended position size.

        Uses quarter Kelly with portfolio constraints.
        """
        # Base size from quarter Kelly
        kelly_size = edge.half_kelly / 2  # Quarter Kelly
        base_size = portfolio_value * kelly_size

        # Apply max position constraint
        max_position = portfolio_value * Decimal(str(self.risk_config["max_position_pct"]))
        size = min(base_size, max_position)

        # Apply portfolio risk constraint
        max_additional = portfolio_value * Decimal(str(self.risk_config["max_portfolio_risk"])) - current_exposure
        size = min(size, max(Decimal("0"), max_additional))

        return size.quantize(Decimal("0.01"))

    async def validate_trade(
        self,
        market: MarketState,
        direction: str,
        edge_estimate: Decimal,
        confidence: Decimal,
        portfolio_value: Decimal,
        current_positions: list[Position],
    ) -> dict[str, Any]:
        """Validate a single trade against risk parameters."""
        # Calculate current exposure
        current_exposure = sum(p.market_value for p in current_positions)
        exposure_pct = current_exposure / portfolio_value if portfolio_value > 0 else Decimal("0")

        # Check minimum edge
        if edge_estimate < Decimal(str(self.risk_config["min_edge"])):
            return {
                "approved": False,
                "reason": f"Edge {edge_estimate:.1%} below minimum {self.risk_config['min_edge']:.1%}",
            }

        # Check portfolio capacity
        if exposure_pct >= Decimal(str(self.risk_config["max_portfolio_risk"])):
            return {
                "approved": False,
                "reason": f"Portfolio exposure {exposure_pct:.1%} at maximum",
            }

        # Calculate position size
        entry_price = market.yes_price if "yes" in direction.lower() else market.no_price
        win_amount = Decimal("1") - entry_price
        loss_amount = entry_price

        edge = self.calculate_kelly(
            win_prob=entry_price + edge_estimate,
            win_amount=win_amount,
            loss_amount=loss_amount,
        )

        size_usdc = self.calculate_position_size(
            edge=edge,
            portfolio_value=portfolio_value,
            current_exposure=current_exposure,
        )

        # Calculate stop loss and take profit
        stop_loss = entry_price * Decimal("0.85")  # 15% stop
        take_profit = entry_price + (win_amount * Decimal("0.5"))  # 50% of max gain

        return {
            "approved": True,
            "market_id": market.market_id,
            "kelly_fraction": float(edge.kelly_fraction),
            "quarter_kelly": float(edge.half_kelly / 2),
            "recommended_size_usdc": float(size_usdc),
            "recommended_size_pct": float(size_usdc / portfolio_value) if portfolio_value > 0 else 0,
            "stop_loss_price": float(stop_loss),
            "take_profit_price": float(take_profit),
            "max_loss_usdc": float(size_usdc * entry_price),
            "risk_reward_ratio": float(win_amount / loss_amount) if loss_amount > 0 else 0,
        }
