"""LLM-based trading agent."""

import json
import time
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any

from ..broker.base import Broker, Order, OrderSide, Position
from ..data.market import MarketInfo
from ..data.sentiment import SentimentData
from ..strategy.base import ExitPlan, SignalType, TradeSignal
from .base import Agent, AgentConfig, AgentDecision, AgentObservation


class LLMProvider(str, Enum):
    """Supported LLM providers."""

    ANTHROPIC = "anthropic"
    OPENAI = "openai"


@dataclass
class LLMResponse:
    """Response from LLM call."""

    content: str
    model: str
    tokens_input: int
    tokens_output: int
    latency_ms: int


class LLMAgent(Agent):
    """Agent that uses LLM for trading decisions."""

    SYSTEM_PROMPT = """You are an autonomous trading agent for Polymarket prediction markets.
Your goal is to maximize profit (PnL) through informed trading decisions.

You will receive:
1. Current balance and positions
2. Available markets with prices
3. Sentiment data from social media
4. Recent trading activity

You must output a JSON object with your trading decisions:
{
    "reasoning": "Your analysis and thought process",
    "signals": [
        {
            "market_id": "condition_id",
            "token_id": "token_id",
            "signal_type": "enter_long|enter_short|exit|hold",
            "side": "buy|sell",
            "size": 10.0,
            "price": 0.55,
            "confidence": 0.75,
            "expected_edge": 0.08,
            "exit_plan": {
                "profit_target_price": 0.70,
                "stop_loss_price": 0.45,
                "invalidation_conditions": ["sentiment reversal", "new information"]
            }
        }
    ]
}

Guidelines:
- Only trade when you have high confidence (>0.6) and positive expected edge (>0.05)
- Consider sentiment strength and volume
- Set clear exit plans for every trade
- Prefer fewer, higher-quality trades over many low-conviction trades
- Be explicit about your reasoning
- If no good opportunities exist, return empty signals array

Output ONLY valid JSON, no additional text."""

    def __init__(self, config: AgentConfig, api_key: str):
        super().__init__(config)
        self.api_key = api_key
        self._client: Any = None

    async def connect(self) -> None:
        """Initialize LLM client."""
        if self.config.llm_provider == "anthropic":
            import anthropic
            self._client = anthropic.AsyncAnthropic(api_key=self.api_key)
        elif self.config.llm_provider == "openai":
            import openai
            self._client = openai.AsyncOpenAI(api_key=self.api_key)
        else:
            raise ValueError(f"Unknown LLM provider: {self.config.llm_provider}")

    async def disconnect(self) -> None:
        """Cleanup LLM client."""
        self._client = None

    async def observe(
        self,
        broker: Broker,
        markets: list[MarketInfo],
        sentiment: dict[str, SentimentData],
    ) -> AgentObservation:
        """Gather current market state."""
        balance = await broker.get_balance()
        positions = await broker.get_positions()
        open_orders = await broker.get_open_orders()
        recent_fills = await broker.get_fills()

        return AgentObservation(
            timestamp=datetime.utcnow(),
            balance=balance,
            positions=positions,
            markets=markets,
            sentiment=sentiment,
            open_orders=open_orders,
            recent_fills=recent_fills,
        )

    async def decide(self, observation: AgentObservation) -> AgentDecision:
        """Use LLM to make trading decisions."""
        prompt = self._build_prompt(observation)

        start_time = time.time()
        response = await self._call_llm(prompt)
        latency_ms = int((time.time() - start_time) * 1000)

        # Parse LLM response
        signals = self._parse_response(response.content)

        return AgentDecision(
            timestamp=datetime.utcnow(),
            signals=signals,
            reasoning=self._extract_reasoning(response.content),
            model_used=response.model,
            latency_ms=latency_ms,
            tokens_used=response.tokens_input + response.tokens_output,
        )

    async def execute(self, decision: AgentDecision, broker: Broker) -> list[Order]:
        """Execute the trading signals."""
        orders = []

        for signal in decision.signals:
            if signal.signal_type == SignalType.HOLD:
                continue

            if signal.side is None or signal.size is None or signal.price is None:
                continue

            try:
                order = await broker.place_order(
                    market_id=signal.market_id,
                    token_id=signal.token_id,
                    side=signal.side,
                    size=signal.size,
                    price=signal.price,
                )
                orders.append(order)
            except Exception as e:
                # Log error but continue with other orders
                print(f"Order execution failed: {e}")

        return orders

    def _build_prompt(self, obs: AgentObservation) -> str:
        """Build prompt for LLM from observation."""
        # Format positions
        positions_text = "None" if not obs.positions else "\n".join([
            f"  - {p.outcome}: size={p.size}, entry={p.avg_entry_price}, "
            f"current={p.current_price}, pnl={p.unrealized_pnl}"
            for p in obs.positions
        ])

        # Format markets (top 10 by liquidity)
        sorted_markets = sorted(obs.markets, key=lambda m: m.liquidity, reverse=True)[:10]
        markets_text = "\n".join([
            f"  - {m.question[:80]}...\n"
            f"    ID: {m.condition_id}\n"
            f"    Tokens: {', '.join(f'{t.outcome}({t.token_id[:8]}...)@{t.price}' for t in m.tokens)}\n"
            f"    Volume 24h: ${m.volume_24h}, Liquidity: ${m.liquidity}"
            for m in sorted_markets
        ])

        # Format sentiment
        sentiment_text = "None" if not obs.sentiment else "\n".join([
            f"  - {mid[:8]}...: score={s.score}, level={s.level.value}, volume={s.volume}"
            for mid, s in obs.sentiment.items()
        ])

        return f"""Current State:
- Balance: ${obs.balance} USDC
- Timestamp: {obs.timestamp.isoformat()}

Positions:
{positions_text}

Available Markets:
{markets_text}

Sentiment Data:
{sentiment_text}

Open Orders: {len(obs.open_orders)}
Recent Fills: {len(obs.recent_fills)}

Analyze the above and provide your trading decisions as JSON."""

    async def _call_llm(self, prompt: str) -> LLMResponse:
        """Call the configured LLM."""
        if self._client is None:
            raise RuntimeError("LLM client not connected")

        start_time = time.time()

        if self.config.llm_provider == "anthropic":
            response = await self._client.messages.create(
                model=self.config.llm_model,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                system=self.SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )

            return LLMResponse(
                content=response.content[0].text,
                model=response.model,
                tokens_input=response.usage.input_tokens,
                tokens_output=response.usage.output_tokens,
                latency_ms=int((time.time() - start_time) * 1000),
            )

        elif self.config.llm_provider == "openai":
            response = await self._client.chat.completions.create(
                model=self.config.llm_model,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            )

            return LLMResponse(
                content=response.choices[0].message.content or "",
                model=response.model,
                tokens_input=response.usage.prompt_tokens if response.usage else 0,
                tokens_output=response.usage.completion_tokens if response.usage else 0,
                latency_ms=int((time.time() - start_time) * 1000),
            )

        raise ValueError(f"Unknown provider: {self.config.llm_provider}")

    def _parse_response(self, content: str) -> list[TradeSignal]:
        """Parse LLM response into TradeSignals."""
        try:
            # Try to extract JSON from response
            content = content.strip()
            if content.startswith("```"):
                # Remove markdown code blocks
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]

            data = json.loads(content)
            signals = []

            for s in data.get("signals", []):
                signal_type = SignalType(s.get("signal_type", "hold"))

                exit_plan = None
                if ep := s.get("exit_plan"):
                    exit_plan = ExitPlan(
                        profit_target_price=Decimal(str(ep.get("profit_target_price", 0))) if ep.get("profit_target_price") else None,
                        stop_loss_price=Decimal(str(ep.get("stop_loss_price", 0))) if ep.get("stop_loss_price") else None,
                        invalidation_conditions=ep.get("invalidation_conditions", []),
                    )

                signals.append(TradeSignal(
                    market_id=s.get("market_id", ""),
                    token_id=s.get("token_id", ""),
                    signal_type=signal_type,
                    side=OrderSide(s["side"]) if s.get("side") else None,
                    size=Decimal(str(s["size"])) if s.get("size") else None,
                    price=Decimal(str(s["price"])) if s.get("price") else None,
                    confidence=Decimal(str(s.get("confidence", 0))),
                    expected_edge=Decimal(str(s.get("expected_edge", 0))),
                    exit_plan=exit_plan,
                    reasoning=data.get("reasoning", ""),
                ))

            return signals

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"Failed to parse LLM response: {e}")
            return []

    def _extract_reasoning(self, content: str) -> str:
        """Extract reasoning from LLM response."""
        try:
            content = content.strip()
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]

            data = json.loads(content)
            return data.get("reasoning", "")
        except (json.JSONDecodeError, KeyError):
            return content[:500]  # Return truncated raw content as fallback
