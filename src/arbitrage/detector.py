"""Arbitrage opportunity detection."""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from difflib import SequenceMatcher

from .platforms import Platform, PlatformMarket


@dataclass
class ArbitrageOpportunity:
    """Detected arbitrage opportunity."""
    opportunity_id: str
    type: str  # "binary_complement", "cross_platform", "multi_outcome"

    # Markets involved
    market_a: PlatformMarket
    market_b: PlatformMarket | None = None

    # Trade details
    buy_platform: str = ""
    sell_platform: str = ""
    buy_side: str = ""  # "yes" or "no"
    buy_price: Decimal = Decimal("0")
    sell_price: Decimal = Decimal("0")

    # Profit calculation
    gross_profit_pct: Decimal = Decimal("0")
    net_profit_pct: Decimal = Decimal("0")  # After fees
    total_fees_pct: Decimal = Decimal("0")

    # Risk metrics
    execution_risk: str = "low"  # "low", "medium", "high"
    liquidity_risk: str = "low"
    settlement_risk: str = "low"  # Different settlement rules

    # Timing
    detected_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: datetime | None = None

    def is_profitable(self, min_profit_pct: Decimal = Decimal("0.5")) -> bool:
        """Check if opportunity meets minimum profit threshold."""
        return self.net_profit_pct >= min_profit_pct

    def to_dict(self) -> dict:
        return {
            "id": self.opportunity_id,
            "type": self.type,
            "buy_platform": self.buy_platform,
            "sell_platform": self.sell_platform,
            "buy_side": self.buy_side,
            "gross_profit_pct": float(self.gross_profit_pct),
            "net_profit_pct": float(self.net_profit_pct),
            "execution_risk": self.execution_risk,
        }


class ArbitrageDetector:
    """
    Detects arbitrage opportunities across prediction markets.

    Types of arbitrage:
    1. Binary Complement: YES + NO < $1 on same platform
    2. Cross-Platform: Same market priced differently on different platforms
    3. Multi-Outcome: Sum of all outcomes < $1
    """

    def __init__(self, platforms: list[Platform]):
        self.platforms = {p.name: p for p in platforms}
        self._market_cache: dict[str, list[PlatformMarket]] = {}

    async def refresh_markets(self) -> None:
        """Refresh market data from all platforms."""
        for name, platform in self.platforms.items():
            try:
                self._market_cache[name] = await platform.get_markets()
            except Exception as e:
                print(f"Failed to fetch {name} markets: {e}")
                self._market_cache[name] = []

    async def detect_all(self) -> list[ArbitrageOpportunity]:
        """Detect all types of arbitrage opportunities."""
        opportunities = []

        # Binary complement arbitrage (within each platform)
        for platform_name, markets in self._market_cache.items():
            opportunities.extend(
                self._detect_binary_complement(platform_name, markets)
            )

        # Cross-platform arbitrage
        if len(self._market_cache) > 1:
            opportunities.extend(
                self._detect_cross_platform()
            )

        # Sort by profit potential
        opportunities.sort(key=lambda x: x.net_profit_pct, reverse=True)

        return opportunities

    def _detect_binary_complement(
        self,
        platform_name: str,
        markets: list[PlatformMarket],
    ) -> list[ArbitrageOpportunity]:
        """Detect YES + NO < $1 opportunities."""
        opportunities = []

        for market in markets:
            # Check if buying both YES and NO costs less than $1
            total_cost = market.yes_ask + market.no_ask

            if total_cost < Decimal("1"):
                gross_profit = Decimal("1") - total_cost
                fees = market.fee_rate * 2  # Fee on both sides
                net_profit = gross_profit - fees

                if net_profit > Decimal("0"):
                    opportunities.append(ArbitrageOpportunity(
                        opportunity_id=f"bc_{market.market_id[:8]}_{datetime.utcnow().timestamp():.0f}",
                        type="binary_complement",
                        market_a=market,
                        buy_platform=platform_name,
                        sell_platform=platform_name,
                        buy_side="both",
                        buy_price=total_cost,
                        sell_price=Decimal("1"),
                        gross_profit_pct=(gross_profit * 100).quantize(Decimal("0.01")),
                        net_profit_pct=(net_profit * 100).quantize(Decimal("0.01")),
                        total_fees_pct=(fees * 100).quantize(Decimal("0.01")),
                        execution_risk="low",
                        liquidity_risk="low" if market.liquidity > 10000 else "medium",
                        settlement_risk="low",
                    ))

        return opportunities

    def _detect_cross_platform(self) -> list[ArbitrageOpportunity]:
        """Detect cross-platform arbitrage opportunities."""
        opportunities = []
        platform_names = list(self._market_cache.keys())

        if len(platform_names) < 2:
            return opportunities

        # Match markets across platforms
        for i, platform_a in enumerate(platform_names):
            for platform_b in platform_names[i + 1:]:
                matches = self._match_markets(
                    self._market_cache[platform_a],
                    self._market_cache[platform_b],
                )

                for market_a, market_b in matches:
                    opp = self._check_cross_platform_arb(market_a, market_b)
                    if opp:
                        opportunities.append(opp)

        return opportunities

    def _match_markets(
        self,
        markets_a: list[PlatformMarket],
        markets_b: list[PlatformMarket],
    ) -> list[tuple[PlatformMarket, PlatformMarket]]:
        """Match similar markets across platforms."""
        matches = []

        for ma in markets_a:
            for mb in markets_b:
                similarity = SequenceMatcher(
                    None,
                    ma.question.lower(),
                    mb.question.lower(),
                ).ratio()

                if similarity > 0.85:  # High similarity threshold
                    matches.append((ma, mb))

        return matches

    def _check_cross_platform_arb(
        self,
        market_a: PlatformMarket,
        market_b: PlatformMarket,
    ) -> ArbitrageOpportunity | None:
        """Check for arbitrage between two matched markets."""
        # Strategy 1: Buy YES on A, sell YES on B (or vice versa)
        # This requires the ability to short, which isn't always available

        # Strategy 2: Buy YES on cheaper platform, buy NO on expensive platform
        # If YES_A + NO_B < $1, there's arbitrage

        # Check YES_A + NO_B
        cost_1 = market_a.yes_ask + market_b.no_ask
        # Check YES_B + NO_A
        cost_2 = market_b.yes_ask + market_a.no_ask

        best_cost = min(cost_1, cost_2)

        if best_cost >= Decimal("1"):
            return None

        gross_profit = Decimal("1") - best_cost
        fees = market_a.fee_rate + market_b.fee_rate
        net_profit = gross_profit - fees

        if net_profit <= Decimal("0"):
            return None

        if cost_1 < cost_2:
            buy_yes_platform = market_a.platform
            buy_no_platform = market_b.platform
            yes_price = market_a.yes_ask
            no_price = market_b.no_ask
        else:
            buy_yes_platform = market_b.platform
            buy_no_platform = market_a.platform
            yes_price = market_b.yes_ask
            no_price = market_a.no_ask

        return ArbitrageOpportunity(
            opportunity_id=f"cp_{market_a.market_id[:4]}_{market_b.market_id[:4]}_{datetime.utcnow().timestamp():.0f}",
            type="cross_platform",
            market_a=market_a,
            market_b=market_b,
            buy_platform=f"{buy_yes_platform}(YES), {buy_no_platform}(NO)",
            sell_platform="settlement",
            buy_side="both",
            buy_price=best_cost,
            sell_price=Decimal("1"),
            gross_profit_pct=(gross_profit * 100).quantize(Decimal("0.01")),
            net_profit_pct=(net_profit * 100).quantize(Decimal("0.01")),
            total_fees_pct=(fees * 100).quantize(Decimal("0.01")),
            execution_risk="medium",  # Cross-platform adds complexity
            liquidity_risk="medium",
            settlement_risk="medium",  # Different platforms may settle differently
        )

    async def monitor_opportunity(
        self,
        opportunity: ArbitrageOpportunity,
    ) -> ArbitrageOpportunity | None:
        """
        Re-check if an opportunity still exists.

        Returns updated opportunity or None if it's gone.
        """
        # Refresh the specific markets
        if opportunity.type == "binary_complement":
            platform = self.platforms.get(opportunity.buy_platform)
            if not platform:
                return None

            market = await platform.get_market(opportunity.market_a.market_id)
            if not market:
                return None

            # Re-check profitability
            total_cost = market.yes_ask + market.no_ask
            if total_cost >= Decimal("1"):
                return None

            # Update opportunity
            gross_profit = Decimal("1") - total_cost
            fees = market.fee_rate * 2
            net_profit = gross_profit - fees

            return ArbitrageOpportunity(
                opportunity_id=opportunity.opportunity_id,
                type=opportunity.type,
                market_a=market,
                buy_platform=opportunity.buy_platform,
                sell_platform=opportunity.sell_platform,
                buy_side="both",
                buy_price=total_cost,
                sell_price=Decimal("1"),
                gross_profit_pct=(gross_profit * 100).quantize(Decimal("0.01")),
                net_profit_pct=(net_profit * 100).quantize(Decimal("0.01")),
                total_fees_pct=(fees * 100).quantize(Decimal("0.01")),
                execution_risk=opportunity.execution_risk,
                liquidity_risk=opportunity.liquidity_risk,
                settlement_risk=opportunity.settlement_risk,
            )

        return None
