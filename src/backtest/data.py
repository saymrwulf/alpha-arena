"""Historical data loading for backtesting."""

import json
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Iterator

import httpx


@dataclass
class OHLCV:
    """OHLCV candle data."""
    timestamp: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "open": float(self.open),
            "high": float(self.high),
            "low": float(self.low),
            "close": float(self.close),
            "volume": float(self.volume),
        }


@dataclass
class MarketSnapshot:
    """Point-in-time market state for backtesting."""
    timestamp: datetime
    market_id: str
    question: str
    yes_price: Decimal
    no_price: Decimal
    yes_volume: Decimal
    no_volume: Decimal
    liquidity: Decimal
    resolved: bool = False
    outcome: str | None = None  # "yes", "no", or None


@dataclass
class HistoricalMarket:
    """Historical market with price history."""
    market_id: str
    question: str
    start_date: datetime
    end_date: datetime
    resolution_date: datetime | None = None
    outcome: str | None = None
    snapshots: list[MarketSnapshot] = field(default_factory=list)
    ohlcv: list[OHLCV] = field(default_factory=list)


class HistoricalDataLoader:
    """
    Loads historical market data for backtesting.

    Supports:
    - Local JSON/CSV files
    - Polymarket historical API
    - Synthetic data generation for testing
    """

    def __init__(self, data_dir: Path | None = None):
        self.data_dir = data_dir or Path("data/historical")
        self._cache: dict[str, HistoricalMarket] = {}

    async def load_from_polymarket(
        self,
        market_id: str,
        start_date: datetime,
        end_date: datetime,
    ) -> HistoricalMarket | None:
        """Load historical data from Polymarket API."""
        async with httpx.AsyncClient(timeout=60.0) as client:
            try:
                # Fetch market info
                market_resp = await client.get(
                    f"https://gamma-api.polymarket.com/markets/{market_id}"
                )
                if market_resp.status_code != 200:
                    return None

                market_data = market_resp.json()

                # Fetch price history
                history_resp = await client.get(
                    f"https://clob.polymarket.com/prices-history",
                    params={
                        "market": market_id,
                        "startTs": int(start_date.timestamp()),
                        "endTs": int(end_date.timestamp()),
                        "fidelity": 60,  # 1 minute
                    }
                )

                history = history_resp.json() if history_resp.status_code == 200 else {"history": []}

                # Parse into snapshots
                snapshots = []
                ohlcv = []

                for point in history.get("history", []):
                    ts = datetime.fromtimestamp(point.get("t", 0))
                    price = Decimal(str(point.get("p", 0.5)))

                    snapshots.append(MarketSnapshot(
                        timestamp=ts,
                        market_id=market_id,
                        question=market_data.get("question", ""),
                        yes_price=price,
                        no_price=Decimal("1") - price,
                        yes_volume=Decimal("0"),
                        no_volume=Decimal("0"),
                        liquidity=Decimal(str(market_data.get("liquidity", 0))),
                    ))

                    # Generate OHLCV (simplified - using same price for all)
                    ohlcv.append(OHLCV(
                        timestamp=ts,
                        open=price,
                        high=price,
                        low=price,
                        close=price,
                        volume=Decimal("0"),
                    ))

                # Determine outcome if resolved
                outcome = None
                if market_data.get("closed"):
                    outcome_prices = market_data.get("outcomePrices", [])
                    if isinstance(outcome_prices, str):
                        outcome_prices = json.loads(outcome_prices)
                    if outcome_prices and float(outcome_prices[0]) > 0.99:
                        outcome = "yes"
                    elif outcome_prices and float(outcome_prices[0]) < 0.01:
                        outcome = "no"

                return HistoricalMarket(
                    market_id=market_id,
                    question=market_data.get("question", ""),
                    start_date=start_date,
                    end_date=end_date,
                    resolution_date=datetime.fromisoformat(market_data["endDate"].replace("Z", "")) if market_data.get("endDate") else None,
                    outcome=outcome,
                    snapshots=snapshots,
                    ohlcv=ohlcv,
                )

            except Exception as e:
                print(f"Error loading historical data: {e}")
                return None

    def load_from_file(self, file_path: Path) -> HistoricalMarket | None:
        """Load historical data from local JSON file."""
        if not file_path.exists():
            return None

        with open(file_path) as f:
            data = json.load(f)

        snapshots = [
            MarketSnapshot(
                timestamp=datetime.fromisoformat(s["timestamp"]),
                market_id=data["market_id"],
                question=data["question"],
                yes_price=Decimal(str(s["yes_price"])),
                no_price=Decimal(str(s["no_price"])),
                yes_volume=Decimal(str(s.get("yes_volume", 0))),
                no_volume=Decimal(str(s.get("no_volume", 0))),
                liquidity=Decimal(str(s.get("liquidity", 0))),
            )
            for s in data.get("snapshots", [])
        ]

        ohlcv = [
            OHLCV(
                timestamp=datetime.fromisoformat(c["timestamp"]),
                open=Decimal(str(c["open"])),
                high=Decimal(str(c["high"])),
                low=Decimal(str(c["low"])),
                close=Decimal(str(c["close"])),
                volume=Decimal(str(c["volume"])),
            )
            for c in data.get("ohlcv", [])
        ]

        return HistoricalMarket(
            market_id=data["market_id"],
            question=data["question"],
            start_date=datetime.fromisoformat(data["start_date"]),
            end_date=datetime.fromisoformat(data["end_date"]),
            resolution_date=datetime.fromisoformat(data["resolution_date"]) if data.get("resolution_date") else None,
            outcome=data.get("outcome"),
            snapshots=snapshots,
            ohlcv=ohlcv,
        )

    def save_to_file(self, market: HistoricalMarket, file_path: Path) -> None:
        """Save historical data to local JSON file."""
        file_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "market_id": market.market_id,
            "question": market.question,
            "start_date": market.start_date.isoformat(),
            "end_date": market.end_date.isoformat(),
            "resolution_date": market.resolution_date.isoformat() if market.resolution_date else None,
            "outcome": market.outcome,
            "snapshots": [
                {
                    "timestamp": s.timestamp.isoformat(),
                    "yes_price": float(s.yes_price),
                    "no_price": float(s.no_price),
                    "yes_volume": float(s.yes_volume),
                    "no_volume": float(s.no_volume),
                    "liquidity": float(s.liquidity),
                }
                for s in market.snapshots
            ],
            "ohlcv": [c.to_dict() for c in market.ohlcv],
        }

        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)

    def generate_synthetic(
        self,
        market_id: str = "synthetic_001",
        question: str = "Will synthetic event happen?",
        days: int = 30,
        start_price: Decimal = Decimal("0.5"),
        volatility: Decimal = Decimal("0.02"),
        trend: Decimal = Decimal("0.001"),
        outcome: str = "yes",
    ) -> HistoricalMarket:
        """Generate synthetic market data for testing."""
        import random
        from datetime import timedelta

        random.seed(42)  # Reproducible

        start_date = datetime.utcnow() - timedelta(days=days)
        end_date = datetime.utcnow()

        snapshots = []
        ohlcv = []
        current_price = start_price

        # Generate hourly data
        current_time = start_date
        while current_time < end_date:
            # Random walk with trend
            change = Decimal(str(random.gauss(float(trend), float(volatility))))
            current_price = max(Decimal("0.01"), min(Decimal("0.99"), current_price + change))

            # Add some volume
            volume = Decimal(str(random.uniform(100, 10000)))

            snapshots.append(MarketSnapshot(
                timestamp=current_time,
                market_id=market_id,
                question=question,
                yes_price=current_price,
                no_price=Decimal("1") - current_price,
                yes_volume=volume / 2,
                no_volume=volume / 2,
                liquidity=Decimal("50000"),
            ))

            # OHLCV (simplified)
            high = current_price + Decimal(str(abs(random.gauss(0, float(volatility)))))
            low = current_price - Decimal(str(abs(random.gauss(0, float(volatility)))))

            ohlcv.append(OHLCV(
                timestamp=current_time,
                open=current_price,
                high=min(Decimal("0.99"), high),
                low=max(Decimal("0.01"), low),
                close=current_price,
                volume=volume,
            ))

            current_time += timedelta(hours=1)

        return HistoricalMarket(
            market_id=market_id,
            question=question,
            start_date=start_date,
            end_date=end_date,
            resolution_date=end_date,
            outcome=outcome,
            snapshots=snapshots,
            ohlcv=ohlcv,
        )

    def iterate_snapshots(
        self,
        market: HistoricalMarket,
    ) -> Iterator[MarketSnapshot]:
        """Iterate through market snapshots in chronological order."""
        for snapshot in sorted(market.snapshots, key=lambda x: x.timestamp):
            yield snapshot
