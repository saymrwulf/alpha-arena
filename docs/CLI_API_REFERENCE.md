# Alpha Arena CLI & API Reference

**Complete reference for command-line interface and REST API**

---

## Part 1: Command Line Interface (CLI)

### Quick Reference

```bash
# Server Management
./alpha start              # Start server
./alpha stop               # Stop server
./alpha restart            # Restart server
./alpha status             # Check status
./alpha logs               # View live logs
./alpha open               # Open web UI

# Diagnostics
./alpha test               # Run tests
./alpha check              # System diagnostics
./alpha help               # Show help
```

---

## 1.1 Server Commands

### `./alpha start`

Start the Alpha Arena server in the background.

```bash
./alpha start
```

**What it does:**
1. Checks if virtual environment exists (creates if needed)
2. Starts the FastAPI server on port 8000
3. Waits for server to be healthy (up to 30 seconds)
4. Shows macOS notification when ready

**Options via environment variables:**

| Variable | Default | Description |
|----------|---------|-------------|
| `ALPHA_HOST` | `127.0.0.1` | Host to bind to |
| `ALPHA_PORT` | `8000` | Port to bind to |

**Examples:**

```bash
# Start on default port (8000)
./alpha start

# Start on custom port
ALPHA_PORT=9000 ./alpha start

# Start accessible from LAN
ALPHA_HOST=0.0.0.0 ./alpha start

# Combine options
ALPHA_HOST=0.0.0.0 ALPHA_PORT=9000 ./alpha start
```

**Output:**

```
Starting Alpha Arena server...
Server starting on http://127.0.0.1:8000
Waiting for server to be ready...
✓ Server is running!

Open in browser: http://127.0.0.1:8000
Or run: ./alpha open
```

---

### `./alpha stop`

Stop the running server gracefully.

```bash
./alpha stop
```

**What it does:**
1. Reads PID from `.alpha.pid`
2. Sends SIGTERM to process
3. Waits for clean shutdown
4. Shows notification when stopped

**Output:**

```
Stopping Alpha Arena server...
✓ Server stopped
```

---

### `./alpha restart`

Stop and start the server.

```bash
./alpha restart
```

**Equivalent to:**
```bash
./alpha stop && ./alpha start
```

---

### `./alpha status`

Check if the server is running and healthy.

```bash
./alpha status
```

**Output when running:**

```
Alpha Arena Status
==================
Status:  RUNNING
URL:     http://127.0.0.1:8000
PID:     12345
Health:  OK
Uptime:  2h 34m
```

**Output when stopped:**

```
Alpha Arena Status
==================
Status:  STOPPED
```

---

### `./alpha logs`

Watch live server logs.

```bash
./alpha logs
```

**What it does:**
- Tails `data/logs/server.log`
- Shows real-time log output
- Press `Ctrl+C` to exit

**Example output:**

```
2024-01-15 14:32:15 INFO  Trading cycle started
2024-01-15 14:32:16 INFO  Analyzing 5 markets
2024-01-15 14:32:18 INFO  Signal generated: BUY on market_123 (conf: 0.82)
2024-01-15 14:32:19 INFO  Trade executed: 50 USDC
```

---

### `./alpha open`

Open the web interface in your default browser.

```bash
./alpha open
```

**What it does:**
1. Checks if server is running
2. Starts server if not running
3. Opens browser to http://localhost:8000

---

## 1.2 Diagnostic Commands

### `./alpha test`

Run the test suite.

```bash
./alpha test
```

**What it does:**
- Activates virtual environment
- Runs pytest with all tests
- Shows pass/fail summary

**Options:**

```bash
# Run all tests
./alpha test

# Run specific test file
./scripts/test.sh tests/test_api.py

# Run with coverage
./scripts/test.sh coverage

# Run only unit tests
./scripts/test.sh unit
```

---

### `./alpha check`

Run system diagnostics.

```bash
./alpha check
```

**What it checks:**
- Python version
- Virtual environment
- Dependencies installed
- API keys configured
- Network connectivity
- Disk space

**Output:**

```
Alpha Arena System Check
========================
✓ Python 3.11.5
✓ Virtual environment exists
✓ Dependencies installed
✓ Anthropic API key configured
✗ OpenAI API key not set
✓ Network connectivity OK
✓ Disk space OK (45GB free)

Status: 6/7 checks passed
```

---

### `./alpha help`

Show help and usage information.

```bash
./alpha help
```

---

## 1.3 Python CLI (`cli.py`)

For advanced usage, there's also a Python CLI with more commands.

### Trading Commands

```bash
# Start live trading
python cli.py run

# Start enhanced multi-agent trading
python cli.py run_enhanced

# Show current status
python cli.py status

# Run backtest
python cli.py backtest

# Scan for arbitrage
python cli.py arbitrage
```

### Analysis Commands

```bash
# List markets
python cli.py markets

# Show technical indicators
python cli.py indicators

# Analyze a wallet
python cli.py analyze_wallet 0x1234...

# Compare wallets
python cli.py compare_wallets 0x1234... 0x5678...

# Generate wallet report
python cli.py wallet_report 0x1234...

# Show leaderboard
python cli.py leaderboard
```

### Configuration Commands

```bash
# List providers
python cli.py providers

# Compare LLM models
python cli.py compare_models

# Show statistics
python cli.py stats
```

---

## Part 2: REST API Reference

### Base URL

```
http://localhost:8000/api
```

### Authentication

Currently no authentication required for local access. For production, configure API keys in settings.

### Response Format

All responses are JSON:

```json
{
  "status": "success",
  "data": { ... },
  "message": "Optional message"
}
```

Error responses:

```json
{
  "status": "error",
  "error": "Error description",
  "code": 400
}
```

---

## 2.1 System Endpoints

### GET `/api/system/health`

Check server health.

**Request:**
```bash
curl http://localhost:8000/api/system/health
```

**Response:**
```json
{
  "status": "ok",
  "timestamp": "2024-01-15T14:32:15Z",
  "version": "1.0.0",
  "uptime": 9245.5
}
```

---

### GET `/api/system/providers`

Get configured provider status.

**Request:**
```bash
curl http://localhost:8000/api/system/providers
```

**Response:**
```json
{
  "anthropic": {
    "configured": true,
    "model": "claude-3-5-sonnet-20241022"
  },
  "openai": {
    "configured": false
  },
  "polymarket": {
    "configured": true,
    "address": "0x1234..."
  }
}
```

---

## 2.2 Trading Endpoints

### POST `/api/trading/start`

Start the trading loop.

**Request:**
```bash
curl -X POST http://localhost:8000/api/trading/start \
  -H "Content-Type: application/json" \
  -d '{
    "mode": "simulation",
    "interval": 60,
    "max_iterations": null,
    "features": {
      "multi_agent": true,
      "technical_indicators": true
    }
  }'
```

**Parameters:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `mode` | string | `"simulation"` | `"simulation"` or `"live"` |
| `interval` | integer | `60` | Seconds between cycles |
| `max_iterations` | integer | `null` | Max cycles (null = unlimited) |
| `features.multi_agent` | boolean | `true` | Enable multi-agent system |
| `features.technical_indicators` | boolean | `true` | Use technical analysis |
| `features.arbitrage` | boolean | `false` | Enable arbitrage detection |

**Response:**
```json
{
  "status": "success",
  "message": "Trading started",
  "mode": "simulation"
}
```

---

### POST `/api/trading/stop`

Stop the trading loop.

**Request:**
```bash
curl -X POST http://localhost:8000/api/trading/stop
```

**Response:**
```json
{
  "status": "success",
  "message": "Trading stopped"
}
```

---

### GET `/api/trading/status`

Get current trading status.

**Request:**
```bash
curl http://localhost:8000/api/trading/status
```

**Response:**
```json
{
  "is_active": true,
  "mode": "simulation",
  "total_pnl": 127.45,
  "daily_pnl": 23.67,
  "open_positions": 3,
  "trades_today": 12,
  "last_cycle_time": "2024-01-15T14:32:15Z",
  "kill_switch": false
}
```

---

### POST `/api/trading/kill-switch`

Toggle the emergency kill switch.

**Request:**
```bash
curl -X POST http://localhost:8000/api/trading/kill-switch \
  -H "Content-Type: application/json" \
  -d '{"activate": true, "reason": "Manual trigger"}'
```

**Parameters:**

| Field | Type | Description |
|-------|------|-------------|
| `activate` | boolean | `true` to activate, `false` to deactivate |
| `reason` | string | Reason for activation (optional) |

**Response:**
```json
{
  "status": "success",
  "kill_switch": true,
  "reason": "Manual trigger"
}
```

---

## 2.3 Position Endpoints

### GET `/api/positions`

Get all open positions.

**Request:**
```bash
curl http://localhost:8000/api/positions
```

**Response:**
```json
{
  "positions": [
    {
      "id": "pos_123",
      "market_id": "market_456",
      "market_title": "Will Trump win 2024?",
      "side": "YES",
      "size": 50.0,
      "entry_price": 0.45,
      "current_price": 0.52,
      "pnl": 7.78,
      "pnl_percent": 15.5,
      "opened_at": "2024-01-15T10:00:00Z"
    }
  ],
  "total_pnl": 7.78,
  "count": 1
}
```

---

### POST `/api/positions/{position_id}/close`

Close a specific position.

**Request:**
```bash
curl -X POST http://localhost:8000/api/positions/pos_123/close
```

**Response:**
```json
{
  "status": "success",
  "position_id": "pos_123",
  "close_price": 0.52,
  "realized_pnl": 7.78
}
```

---

### GET `/api/orders`

Get all open orders.

**Request:**
```bash
curl http://localhost:8000/api/orders
```

**Response:**
```json
{
  "orders": [
    {
      "id": "order_789",
      "market_id": "market_456",
      "side": "BUY",
      "outcome": "YES",
      "size": 25.0,
      "price": 0.48,
      "status": "pending",
      "created_at": "2024-01-15T14:00:00Z"
    }
  ],
  "count": 1
}
```

---

### POST `/api/orders/{order_id}/cancel`

Cancel a pending order.

**Request:**
```bash
curl -X POST http://localhost:8000/api/orders/order_789/cancel
```

**Response:**
```json
{
  "status": "success",
  "order_id": "order_789"
}
```

---

## 2.4 Market Endpoints

### GET `/api/markets`

Get active markets with optional filters.

**Request:**
```bash
curl "http://localhost:8000/api/markets?category=politics&min_liquidity=10000"
```

**Query Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `category` | string | Filter by category |
| `min_liquidity` | number | Minimum liquidity in USDC |
| `min_volume` | number | Minimum 24h volume |
| `status` | string | `open`, `closing_soon`, `resolved` |
| `limit` | integer | Max results (default: 50) |
| `offset` | integer | Pagination offset |

**Response:**
```json
{
  "markets": [
    {
      "id": "market_456",
      "title": "Will Trump win 2024 election?",
      "category": "politics",
      "end_date": "2024-11-05T00:00:00Z",
      "outcomes": {
        "yes": {"price": 0.52, "change_24h": 0.021},
        "no": {"price": 0.48, "change_24h": -0.021}
      },
      "liquidity": 450000,
      "volume_24h": 125000
    }
  ],
  "total": 1,
  "limit": 50,
  "offset": 0
}
```

---

### GET `/api/markets/{market_id}`

Get detailed market information.

**Request:**
```bash
curl http://localhost:8000/api/markets/market_456
```

**Response:**
```json
{
  "id": "market_456",
  "title": "Will Trump win 2024 election?",
  "description": "This market resolves YES if...",
  "category": "politics",
  "created_at": "2024-01-01T00:00:00Z",
  "end_date": "2024-11-05T00:00:00Z",
  "resolution_source": "AP News",
  "outcomes": {
    "yes": {
      "price": 0.52,
      "change_24h": 0.021,
      "volume_24h": 75000
    },
    "no": {
      "price": 0.48,
      "change_24h": -0.021,
      "volume_24h": 50000
    }
  },
  "liquidity": 450000,
  "total_volume": 12500000,
  "order_book": {
    "bids": [...],
    "asks": [...]
  }
}
```

---

### GET `/api/markets/{market_id}/indicators`

Get technical indicators for a market.

**Request:**
```bash
curl http://localhost:8000/api/markets/market_456/indicators
```

**Response:**
```json
{
  "market_id": "market_456",
  "timestamp": "2024-01-15T14:32:15Z",
  "indicators": {
    "ema_12": 0.51,
    "ema_26": 0.49,
    "rsi_14": 65.4,
    "macd": 0.02,
    "macd_signal": 0.015,
    "bollinger_upper": 0.58,
    "bollinger_lower": 0.42
  },
  "signals": {
    "trend": "bullish",
    "momentum": "strong",
    "overbought": false
  }
}
```

---

### GET `/api/arbitrage`

Find arbitrage opportunities.

**Request:**
```bash
curl http://localhost:8000/api/arbitrage
```

**Response:**
```json
{
  "opportunities": [
    {
      "market_id": "market_789",
      "title": "BTC > 100k by Dec?",
      "yes_price": 0.35,
      "no_price": 0.62,
      "spread": 0.03,
      "potential_profit": 3.0,
      "confidence": "medium"
    }
  ],
  "count": 1
}
```

---

## 2.5 Wallet Analysis Endpoints

### POST `/api/wallet/analyze`

Analyze a wallet address.

**Request:**
```bash
curl -X POST http://localhost:8000/api/wallet/analyze \
  -H "Content-Type: application/json" \
  -d '{"address": "0x1234567890abcdef1234567890abcdef12345678"}'
```

**Response:**
```json
{
  "address": "0x1234...5678",
  "analysis": {
    "total_pnl": 12456.78,
    "win_rate": 0.67,
    "total_trades": 234,
    "avg_position_size": 150.0,
    "avg_holding_time_hours": 100.8,
    "max_drawdown": -1200.0,
    "sharpe_ratio": 1.45
  },
  "strategy": {
    "style": "contrarian",
    "preferred_categories": ["politics", "crypto"],
    "avg_confidence": 0.72,
    "risk_level": "moderate"
  },
  "recent_trades": [...]
}
```

---

### GET `/api/wallet/{address}/html`

Get wallet analysis as HTML report.

**Request:**
```bash
curl http://localhost:8000/api/wallet/0x1234.../html
```

**Response:** HTML page with formatted analysis report.

---

## 2.6 Agent Endpoints

### GET `/api/agents`

Get all agent statuses.

**Request:**
```bash
curl http://localhost:8000/api/agents
```

**Response:**
```json
{
  "agents": [
    {
      "name": "fundamentals",
      "status": "active",
      "last_decision": "2024-01-15T14:30:00Z",
      "decisions_today": 45,
      "accuracy": 0.72
    },
    {
      "name": "sentiment",
      "status": "active",
      "last_decision": "2024-01-15T14:28:00Z",
      "decisions_today": 38,
      "accuracy": 0.68
    },
    {
      "name": "technical",
      "status": "active",
      "last_decision": "2024-01-15T14:31:00Z",
      "decisions_today": 52,
      "accuracy": 0.65
    }
  ]
}
```

---

### GET `/api/agents/{agent_name}/history`

Get agent decision history.

**Request:**
```bash
curl "http://localhost:8000/api/agents/fundamentals/history?limit=10"
```

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `limit` | integer | 50 | Max results |
| `offset` | integer | 0 | Pagination offset |

**Response:**
```json
{
  "agent": "fundamentals",
  "decisions": [
    {
      "timestamp": "2024-01-15T14:30:00Z",
      "market_id": "market_456",
      "decision": "BUY",
      "confidence": 0.82,
      "reasoning": "Strong fundamental indicators...",
      "outcome": "win",
      "pnl": 12.34
    }
  ],
  "total": 1
}
```

---

## 2.7 Risk Endpoints

### GET `/api/risk/status`

Get current risk status.

**Request:**
```bash
curl http://localhost:8000/api/risk/status
```

**Response:**
```json
{
  "daily_loss": 45.67,
  "daily_loss_limit": 100.0,
  "daily_loss_percent": 45.67,
  "drawdown": 8.5,
  "max_drawdown": 15.0,
  "drawdown_percent": 56.67,
  "open_positions": 3,
  "max_positions": 5,
  "kill_switch_active": false,
  "warnings": []
}
```

---

### POST `/api/risk/settings`

Update risk settings.

**Request:**
```bash
curl -X POST http://localhost:8000/api/risk/settings \
  -H "Content-Type: application/json" \
  -d '{
    "max_position_size": 500,
    "max_positions": 5,
    "daily_loss_limit": 100,
    "max_drawdown_percent": 15,
    "kelly_fraction": 0.25,
    "auto_kill_switch": true
  }'
```

**Response:**
```json
{
  "status": "success",
  "message": "Risk settings updated"
}
```

---

## 2.8 Configuration Endpoints

### GET `/api/config`

Get current configuration.

**Request:**
```bash
curl http://localhost:8000/api/config
```

**Response:**
```json
{
  "trading": {
    "default_mode": "simulation",
    "loop_interval": 60,
    "min_confidence": 0.6,
    "min_edge": 0.05
  },
  "risk": {
    "max_position_size": 500,
    "max_positions": 5,
    "daily_loss_limit": 100
  },
  "features": {
    "multi_agent": true,
    "technical_indicators": true,
    "arbitrage": false
  }
}
```

---

### POST `/api/config`

Update configuration.

**Request:**
```bash
curl -X POST http://localhost:8000/api/config \
  -H "Content-Type: application/json" \
  -d '{
    "trading": {
      "loop_interval": 120,
      "min_confidence": 0.7
    }
  }'
```

**Response:**
```json
{
  "status": "success",
  "message": "Configuration updated"
}
```

---

### GET `/api/config/llm`

Get LLM provider configuration.

**Request:**
```bash
curl http://localhost:8000/api/config/llm
```

**Response:**
```json
{
  "primary_provider": "anthropic",
  "model": "claude-3-5-sonnet-20241022",
  "temperature": 0.7,
  "max_tokens": 4096,
  "fallback_provider": "openai"
}
```

---

### POST `/api/config/llm`

Update LLM provider configuration.

**Request:**
```bash
curl -X POST http://localhost:8000/api/config/llm \
  -H "Content-Type: application/json" \
  -d '{
    "primary_provider": "anthropic",
    "model": "claude-3-5-sonnet-20241022",
    "temperature": 0.5
  }'
```

---

## 2.9 Logs & Metrics Endpoints

### GET `/api/logs/decisions`

Get decision logs.

**Request:**
```bash
curl "http://localhost:8000/api/logs/decisions?limit=100"
```

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `limit` | integer | 100 | Max results |
| `offset` | integer | 0 | Pagination offset |
| `start_date` | string | - | Filter from date |
| `end_date` | string | - | Filter to date |

**Response:**
```json
{
  "logs": [
    {
      "timestamp": "2024-01-15T14:32:15Z",
      "market_id": "market_456",
      "decision": "BUY",
      "side": "YES",
      "confidence": 0.82,
      "size": 50.0,
      "reasoning": "Strong fundamentals...",
      "agents": ["fundamentals", "sentiment"],
      "outcome": "pending"
    }
  ],
  "total": 1
}
```

---

### GET `/api/logs/trades`

Get trade history.

**Request:**
```bash
curl "http://localhost:8000/api/logs/trades?limit=50"
```

**Response:**
```json
{
  "trades": [
    {
      "id": "trade_123",
      "timestamp": "2024-01-15T14:32:16Z",
      "market_id": "market_456",
      "side": "BUY",
      "outcome": "YES",
      "size": 50.0,
      "price": 0.45,
      "fees": 0.25,
      "status": "filled"
    }
  ],
  "total": 1
}
```

---

### GET `/api/metrics/performance`

Get performance metrics.

**Request:**
```bash
curl http://localhost:8000/api/metrics/performance
```

**Response:**
```json
{
  "total_pnl": 1234.56,
  "daily_pnl": 45.67,
  "weekly_pnl": 234.56,
  "monthly_pnl": 789.01,
  "win_rate": 0.65,
  "profit_factor": 1.8,
  "sharpe_ratio": 1.45,
  "max_drawdown": -234.56,
  "total_trades": 156,
  "winning_trades": 101,
  "losing_trades": 55
}
```

---

### GET `/api/metrics/equity-curve`

Get equity curve data.

**Request:**
```bash
curl "http://localhost:8000/api/metrics/equity-curve?period=7d"
```

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `period` | string | `7d` | `1d`, `7d`, `30d`, `all` |

**Response:**
```json
{
  "period": "7d",
  "data": [
    {"timestamp": "2024-01-09T00:00:00Z", "equity": 10000.0},
    {"timestamp": "2024-01-10T00:00:00Z", "equity": 10125.5},
    {"timestamp": "2024-01-11T00:00:00Z", "equity": 10089.2}
  ]
}
```

---

## 2.10 HTMX Partial Endpoints

These endpoints return HTML fragments for dynamic updates.

### GET `/partials/positions-table`

Returns HTML table of positions.

### GET `/partials/orders-table`

Returns HTML table of orders.

### GET `/partials/alerts`

Returns HTML list of alerts.

### GET `/partials/metrics-cards`

Returns HTML metric cards.

---

## Part 3: WebSocket API

### Connection

```javascript
const ws = new WebSocket('ws://localhost:8000/ws');

ws.onopen = () => {
  console.log('Connected');
};

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Received:', data);
};
```

### Message Types

#### `init`

Sent when connection established.

```json
{
  "type": "init",
  "data": {
    "status": "connected",
    "server_time": "2024-01-15T14:32:15Z"
  }
}
```

#### `trading_started`

Sent when trading starts.

```json
{
  "type": "trading_started",
  "data": {
    "mode": "simulation",
    "timestamp": "2024-01-15T14:32:15Z"
  }
}
```

#### `trading_stopped`

Sent when trading stops.

```json
{
  "type": "trading_stopped",
  "data": {
    "reason": "manual",
    "timestamp": "2024-01-15T14:32:15Z"
  }
}
```

#### `cycle_complete`

Sent after each trading cycle.

```json
{
  "type": "cycle_complete",
  "data": {
    "cycle_number": 47,
    "pnl": 127.45,
    "positions": 3,
    "trades_this_cycle": 1,
    "timestamp": "2024-01-15T14:32:15Z"
  }
}
```

#### `kill_switch`

Sent when kill switch activated.

```json
{
  "type": "kill_switch",
  "data": {
    "activated": true,
    "reason": "Daily loss limit exceeded",
    "timestamp": "2024-01-15T14:32:15Z"
  }
}
```

#### `position_update`

Sent when position changes.

```json
{
  "type": "position_update",
  "data": {
    "action": "opened",
    "position_id": "pos_123",
    "market_id": "market_456",
    "size": 50.0,
    "price": 0.45
  }
}
```

#### `alert`

Sent for important notifications.

```json
{
  "type": "alert",
  "data": {
    "level": "warning",
    "message": "Approaching daily loss limit",
    "timestamp": "2024-01-15T14:32:15Z"
  }
}
```

---

## Quick Reference Tables

### HTTP Status Codes

| Code | Meaning |
|------|---------|
| 200 | Success |
| 201 | Created |
| 400 | Bad Request |
| 401 | Unauthorized |
| 404 | Not Found |
| 500 | Server Error |

### Common Query Parameters

| Parameter | Description |
|-----------|-------------|
| `limit` | Max results per page |
| `offset` | Skip N results |
| `start_date` | Filter from date |
| `end_date` | Filter to date |
| `sort` | Sort field |
| `order` | `asc` or `desc` |

### CLI Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | General error |
| 2 | Invalid arguments |
| 3 | Server not running |
| 4 | Connection failed |

---

*Last updated: January 2025*
