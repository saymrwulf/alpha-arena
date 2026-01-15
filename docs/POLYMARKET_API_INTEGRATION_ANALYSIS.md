# Polymarket API Integration Analysis

## Deep Research: Alpha Arena vs Polymarket Official Framework

**Date:** January 2025
**Author:** Alpha Arena Development Team

---

## Executive Summary

Polymarket provides a **three-tier API architecture**:

| API Layer | Purpose | Alpha Arena Integration |
|-----------|---------|------------------------|
| **Gamma API** | Market discovery, metadata, events | **Partial** - Using `/markets` only |
| **CLOB API** | Trading, orderbook, prices | **Good** - Via py-clob-client |
| **WebSocket** | Real-time orderbook, prices, order status | **Not Implemented** |
| **Data API** | Positions, activity, leaderboards | **Partial** - Custom implementation |

**Overall Integration Score: 65%**

Key gaps:
- No WebSocket real-time streaming
- Missing Data API integration
- Incomplete Gamma API coverage
- No GraphQL subgraph access

---

## 1. Polymarket API Architecture

### 1.1 Official API Structure

```
┌─────────────────────────────────────────────────────────────────┐
│                    POLYMARKET API ECOSYSTEM                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐   │
│  │  GAMMA API   │  │  CLOB API    │  │     DATA API         │   │
│  │  (Discovery) │  │  (Trading)   │  │  (Portfolio/History) │   │
│  └──────┬───────┘  └──────┬───────┘  └──────────┬───────────┘   │
│         │                 │                      │               │
│  gamma-api.       clob.polymarket.    data-api.polymarket.      │
│  polymarket.com        .com                  .com                │
│         │                 │                      │               │
│         │                 │                      │               │
│         ▼                 ▼                      ▼               │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    WEBSOCKET LAYER                        │   │
│  │         wss://ws-subscriptions-clob.polymarket.com        │   │
│  │                                                           │   │
│  │  ┌─────────────┐              ┌─────────────────────┐    │   │
│  │  │   MARKET    │              │        USER         │    │   │
│  │  │  Channel    │              │      Channel        │    │   │
│  │  │ (public)    │              │   (authenticated)   │    │   │
│  │  │             │              │                     │    │   │
│  │  │ - Orderbook │              │ - Order updates     │    │   │
│  │  │ - Prices    │              │ - Trade fills       │    │   │
│  │  │ - Trades    │              │ - Position changes  │    │   │
│  │  └─────────────┘              └─────────────────────┘    │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                   GRAPHQL SUBGRAPHS                       │   │
│  │                   (Goldsky-hosted)                        │   │
│  │                                                           │   │
│  │  Activity | Orderbook | Positions | PnL | Sports Oracle  │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 API Base URLs

| API | Base URL | Purpose |
|-----|----------|---------|
| Gamma REST | `https://gamma-api.polymarket.com` | Market discovery & metadata |
| CLOB REST | `https://clob.polymarket.com` | Trading operations |
| Data REST | `https://data-api.polymarket.com` | Portfolio & analytics |
| CLOB WebSocket | `wss://ws-subscriptions-clob.polymarket.com/ws/` | Real-time updates |
| Live Data WebSocket | `wss://ws-live-data.polymarket.com` | Crypto prices, comments |

---

## 2. Current Alpha Arena Integration

### 2.1 What's Implemented

#### Gamma API (Market Discovery)

| Endpoint | Implemented | Used In |
|----------|-------------|---------|
| `GET /markets` | Yes | `MarketData`, `PolymarketBroker` |
| `GET /markets/{id}` | Yes | `MarketData` |
| `GET /events` | No | - |
| `GET /events/{id}` | No | - |
| `GET /tags` | No | - |
| `GET /sports` | No | - |
| `GET /markets/slug/{slug}` | No | - |
| `GET /search` | No | - |

**Coverage: ~30%**

#### CLOB API (Trading)

| Endpoint | Implemented | Used In |
|----------|-------------|---------|
| `GET /price` | Indirect (via py-clob-client) | - |
| `GET /book` | Yes | `MarketData`, `PolymarketBroker` |
| `GET /midpoint` | Yes | `MarketData`, `PolymarketBroker` |
| `POST /order` | Yes (via py-clob-client) | `PolymarketBroker` |
| `DELETE /order` | Yes (via py-clob-client) | `PolymarketBroker` |
| `GET /prices-history` | Yes | `HistoricalDataLoader` |
| `GET /tick-sizes` | No | - |
| `GET /neg-risk` | No | - |

**Coverage: ~70%**

#### Data API (Portfolio)

| Endpoint | Implemented | Used In |
|----------|-------------|---------|
| `GET /positions` | Partial (via CLOB) | `WalletAnalyzer` |
| `GET /activity` | Yes | `WalletAnalyzer` |
| `GET /trades` | Partial (via Gamma) | `WalletAnalyzer` |
| `GET /holders` | No | - |
| `GET /leaderboard` | No | - |
| `GET /pnl-timeseries` | No | - |

**Coverage: ~40%**

#### WebSocket (Real-time)

| Channel | Implemented | Used In |
|---------|-------------|---------|
| Market Channel (orderbook) | No | - |
| Market Channel (prices) | No | - |
| User Channel (orders) | No | - |
| User Channel (trades) | No | - |

**Coverage: 0%**

### 2.2 Integration Method

```python
# Current Alpha Arena approach:

# 1. py-clob-client (v0.34.4) for CLOB operations
from py_clob_client.client import ClobClient
client = ClobClient(
    host="https://clob.polymarket.com",
    chain_id=137,  # Polygon
    key=private_key,
)

# 2. Direct httpx for Gamma API
async with httpx.AsyncClient() as client:
    response = await client.get(
        "https://gamma-api.polymarket.com/markets",
        params={"active": "true"}
    )

# 3. Polling for updates (no WebSocket)
while trading:
    markets = await fetch_markets()  # HTTP poll
    await asyncio.sleep(30)  # Wait interval
```

---

## 3. Gap Analysis

### 3.1 Critical Gaps

#### Gap 1: No WebSocket Integration

**Impact: HIGH**

Current approach uses polling every 30 seconds. WebSocket would provide:
- Sub-second orderbook updates
- Instant order fill notifications
- Reduced API calls (better rate limit management)
- Lower latency for execution

**Polymarket WebSocket Capabilities:**

```
Market Channel (Public):
├── price_change - Real-time price updates
├── book_update - Orderbook changes (incremental)
└── trade - Recent trades

User Channel (Authenticated):
├── order_update - Order status changes
├── trade - User's trade fills
└── position_change - Position updates
```

#### Gap 2: Missing Data API

**Impact: MEDIUM**

The Data API (`data-api.polymarket.com`) provides:
- Better position tracking than CLOB
- P&L timeseries data
- Leaderboard rankings
- Top holders analysis

Currently, Alpha Arena reconstructs this from CLOB/Gamma, which is less efficient.

#### Gap 3: Incomplete Gamma API Usage

**Impact: MEDIUM**

Missing endpoints:
- `/events` - Event-based organization (markets grouped by event)
- `/tags` - Category/tag system for filtering
- `/search` - Cross-entity search
- `/sports` - Sports-specific metadata

### 3.2 Moderate Gaps

| Gap | Current Approach | Better Approach |
|-----|-----------------|-----------------|
| Rate limiting | No tracking | Parse `X-RateLimit-*` headers |
| Error typing | Generic exceptions | Custom Polymarket exceptions |
| Batch requests | Sequential | Parallel with semaphores |
| Response validation | Manual parsing | Pydantic models |

### 3.3 Minor Gaps

| Gap | Impact | Priority |
|-----|--------|----------|
| No GraphQL support | Low | P3 |
| No sports metadata | Low | P3 |
| No comment streaming | Very Low | P4 |

---

## 4. Polymarket Official Agents Framework

### 4.1 Architecture Comparison

**Polymarket/agents (Official):**

```
polymarket/agents/
├── gamma.py          # GammaMarketClient - market discovery
├── polymarket.py     # Order building, signing, execution
├── chroma.py         # Vector DB for RAG
├── objects.py        # Pydantic data models
├── connectors/
│   ├── news.py       # News API integration
│   └── search.py     # Web search
└── cli.py            # User interface
```

**Alpha Arena:**

```
alpha-arena/src/
├── data/market.py           # Market discovery (similar to gamma.py)
├── broker/polymarket.py     # Trading (similar to polymarket.py)
├── signals/news.py          # News integration
├── signals/aggregator.py    # Signal combination
├── agents/                  # Multi-agent system (UNIQUE)
│   ├── research.py
│   ├── risk.py
│   ├── execution.py
│   └── reflection.py
├── backtest/                # Backtesting (UNIQUE)
└── core/resilience.py       # Circuit breakers (UNIQUE)
```

### 4.2 Key Differences

| Feature | Polymarket/agents | Alpha Arena |
|---------|------------------|-------------|
| Multi-agent debate | No | Yes |
| Risk management | Basic | Advanced (Kelly, limits) |
| Backtesting | No | Comprehensive |
| WebSocket | Not documented | Not implemented |
| Vector DB (RAG) | Chroma | SQLite memory |
| Signal aggregation | Simple | Multi-source weighted |
| Circuit breakers | No | Yes |
| Technical indicators | No | EMA, RSI, MACD, ATR |

**Alpha Arena Advantages:**
- More sophisticated risk management
- Multi-agent coordination with debate
- Comprehensive backtesting suite
- Production resilience patterns
- Technical indicator integration

**Polymarket/agents Advantages:**
- Official Polymarket support
- Potentially better API alignment
- Vector DB for RAG
- Simpler architecture

---

## 5. Recommended Integration Improvements

### 5.1 Priority 1: WebSocket Integration

**New file: `src/data/websocket.py`**

```python
class PolymarketWebSocket:
    """Real-time Polymarket data via WebSocket."""

    WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/"

    async def connect(self):
        """Establish WebSocket connection."""

    async def subscribe_market(self, token_ids: list[str]):
        """Subscribe to market channel for orderbook/price updates."""

    async def subscribe_user(self, api_key: str, markets: list[str]):
        """Subscribe to user channel for order/trade updates."""

    async def handle_message(self, message: dict):
        """Process incoming WebSocket messages."""

    async def maintain_heartbeat(self):
        """Send periodic heartbeats to keep connection alive."""
```

**Benefits:**
- Real-time orderbook updates (sub-second)
- Instant order fill notifications
- Reduced API polling (better rate limits)
- Lower execution latency

### 5.2 Priority 2: Data API Integration

**Enhancement to `src/analysis/wallet.py`**

```python
class PolymarketDataClient:
    """Client for Polymarket Data API."""

    BASE_URL = "https://data-api.polymarket.com"

    async def get_positions(self, address: str) -> list[Position]:
        """Get user positions with enriched data."""

    async def get_pnl_timeseries(self, address: str) -> list[PnLPoint]:
        """Get historical P&L curve."""

    async def get_leaderboard(self, period: str = "daily") -> list[LeaderEntry]:
        """Get trader leaderboard."""

    async def get_top_holders(self, token_id: str) -> list[Holder]:
        """Get top position holders for a market."""
```

### 5.3 Priority 3: Complete Gamma API

**Enhancement to `src/data/market.py`**

```python
# Add missing Gamma endpoints:

async def get_events(self, active_only: bool = True) -> list[Event]:
    """Get events (grouped markets)."""

async def get_event(self, event_id: str) -> Event:
    """Get single event with all markets."""

async def get_tags(self) -> list[Tag]:
    """Get all market tags/categories."""

async def search(self, query: str) -> SearchResults:
    """Cross-entity search."""
```

### 5.4 Priority 4: Use polymarket-apis Package

Consider migrating to the unified `polymarket-apis` package:

```python
# Instead of multiple custom clients:
from polymarket_apis import (
    PolymarketClobClient,
    PolymarketGammaClient,
    PolymarketDataClient,
    PolymarketWebsocketsClient,
)

# Unified, validated, maintained by community
```

**Pros:**
- Pydantic validation
- All APIs in one package
- WebSocket client included
- GraphQL support

**Cons:**
- Requires Python 3.12+
- External dependency
- Less control over implementation

---

## 6. Implementation Roadmap

### Phase 1: WebSocket (Week 1-2)

1. Implement `PolymarketWebSocket` class
2. Add market channel subscription for orderbook
3. Add user channel subscription for order updates
4. Integrate with `EnhancedTradingRunner`
5. Add reconnection logic with exponential backoff

### Phase 2: Data API (Week 3)

1. Create `PolymarketDataClient`
2. Implement position tracking via Data API
3. Add P&L timeseries fetching
4. Integrate with wallet analysis

### Phase 3: Complete Gamma (Week 4)

1. Add events endpoint support
2. Add tags/categories support
3. Add search functionality
4. Improve market filtering

### Phase 4: Cleanup & Optimization (Week 5)

1. Add rate limit tracking
2. Create Pydantic response models
3. Add comprehensive error handling
4. Consider polymarket-apis migration

---

## 7. Current Code References

### 7.1 Broker Implementation

**File:** `src/broker/polymarket.py`

```python
# Line 1-50: Class definition and constants
CLOB_HOST = "https://clob.polymarket.com"
GAMMA_HOST = "https://gamma-api.polymarket.com"

# Line 100-150: Order placement via py-clob-client
async def place_order(self, ...):
    order_args = OrderArgs(...)
    signed_order = self._client.create_order(order_args)
    response = self._client.post_order(signed_order, OrderType.GTC)

# Line 200-250: Market data via httpx
async def get_markets(self, ...):
    response = await self._http.get(f"{self.GAMMA_HOST}/markets", ...)
```

### 7.2 Market Data Implementation

**File:** `src/data/market.py`

```python
# Line 50-100: Circuit breaker protection
self._gamma_breaker = CircuitBreaker("gamma-api", ...)

# Line 150-200: Gamma API calls
async def fetch_markets(self, ...):
    async with self._gamma_breaker:
        response = await self._http.get(f"{self.GAMMA_HOST}/markets", ...)

# Line 250-300: CLOB API calls
async def get_orderbook(self, token_id: str):
    response = await self._http.get(f"{self.CLOB_HOST}/book", ...)
```

---

## 8. Conclusion

Alpha Arena has **solid foundational integration** with Polymarket, particularly for trading operations via py-clob-client. However, significant improvements are possible:

| Area | Current State | Target State |
|------|--------------|--------------|
| Gamma API | 30% coverage | 80% coverage |
| CLOB API | 70% coverage | 90% coverage |
| Data API | 40% coverage | 80% coverage |
| WebSocket | 0% coverage | 100% coverage |

**The biggest opportunity is WebSocket integration**, which would transform Alpha Arena from a polling-based system to a real-time trading platform with sub-second market awareness.

The multi-agent architecture, risk management, and backtesting capabilities are **unique strengths** that differentiate Alpha Arena from Polymarket's official agents framework. These should be preserved and enhanced alongside improved API integration.

---

## Sources

- [Polymarket Documentation](https://docs.polymarket.com/)
- [Gamma API Overview](https://docs.polymarket.com/developers/gamma-markets-api/overview)
- [CLOB WebSocket Documentation](https://docs.polymarket.com/developers/CLOB/websocket/wss-overview)
- [API Endpoints Reference](https://docs.polymarket.com/quickstart/reference/endpoints)
- [Polymarket/agents GitHub](https://github.com/Polymarket/agents)
- [polymarket-apis PyPI](https://pypi.org/project/polymarket-apis/)
