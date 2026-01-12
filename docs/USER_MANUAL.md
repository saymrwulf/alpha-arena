# Alpha Arena User Manual

**Polymarket Multi-Agent Trading Harness** | Complete CLI & API Reference

---

## Quick Navigation

| I want to... | Go to |
|--------------|-------|
| Get started fast | [Quick Start](#2-quick-start) |
| Run CLI commands | [CLI Commands](#5-cli-commands) |
| Understand the agents | [Multi-Agent System](#6-multi-agent-system) |
| Configure LLM providers | [LLM Providers](#7-llm-providers) |
| Set up risk controls | [Risk Management](#10-risk-management) |
| Use technical indicators | [Technical Indicators](#11-technical-indicators) |
| Run backtests | [Backtesting](#13-backtesting) |
| Fix problems | [Troubleshooting](#16-troubleshooting) |

---

## Table of Contents

| Section | Topics |
|---------|--------|
| [1. Introduction](#1-introduction) | What is Alpha Arena, key features, architecture |
| [2. Quick Start](#2-quick-start) | Five-minute setup, first trade |
| [3. Installation & Setup](#3-installation--setup) | Requirements, environment, wallet, API keys |
| [4. Configuration](#4-configuration) | config.yaml, environment variables |
| [5. CLI Commands](#5-cli-commands) | run, markets, positions, backtest, arbitrage |
| [6. Multi-Agent System](#6-multi-agent-system) | Research, Risk, Execution, Reflection agents |
| [7. LLM Providers](#7-llm-providers) | Anthropic, OpenAI, xAI, local models |
| [8. Market Analysis](#8-market-analysis) | Data structure, fetching, filtering |
| [9. Trading Strategies](#9-trading-strategies) | Built-in strategies, edge calculation, entries/exits |
| [10. Risk Management](#10-risk-management) | Position limits, Kelly sizing, kill switch |
| [11. Technical Indicators](#11-technical-indicators) | EMA, RSI, MACD, ATR, volume |
| [12. Arbitrage Detection](#12-arbitrage-detection) | Binary complement, cross-platform |
| [13. Backtesting](#13-backtesting) | Synthetic data, strategies, metrics |
| [14. Memory System](#14-memory-system) | Short-term, long-term, episodic memory |
| [15. Logging & Monitoring](#15-logging--monitoring) | Decision logs, metrics, dashboards |
| [16. Troubleshooting](#16-troubleshooting) | Common issues, diagnostics, recovery |
| [17. API Reference](#17-api-reference) | Core types, broker, agents, memory |

---

## 1. Introduction

### 1.1 What is Alpha Arena?

Alpha Arena is a world-class autonomous trading harness for Polymarket prediction markets. It employs a sophisticated multi-agent architecture where specialized AI agents collaborate to:

- **Research** market opportunities and gather intelligence
- **Assess risk** using Kelly Criterion and technical analysis
- **Execute trades** with optimal timing and position sizing
- **Learn** from outcomes to continuously improve

### 1.2 Key Features

| Feature | Description |
|---------|-------------|
| **Multi-Agent Architecture** | Research, Risk, Execution, and Reflection agents working in concert |
| **Multi-LLM Support** | Anthropic Claude, OpenAI GPT-4o/o1, xAI Grok, and local models |
| **Kelly Criterion Sizing** | Mathematically optimal position sizing based on edge |
| **Technical Analysis** | EMA, RSI, MACD, ATR, volume analysis, support/resistance |
| **Cross-Platform Arbitrage** | Detect and exploit price discrepancies |
| **Memory & Learning** | Short-term, long-term, and episodic memory for continuous improvement |
| **Comprehensive Backtesting** | Test strategies on historical and synthetic data |
| **Real-Time Execution** | Live trading on Polymarket with risk controls |

### 1.3 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        CLI INTERFACE                             │
│    run | run-enhanced | backtest | arbitrage | indicators       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    AGENT COORDINATOR                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐    │
│  │ Research │  │   Risk   │  │Execution │  │  Reflection  │    │
│  │  Agent   │  │  Agent   │  │  Agent   │  │    Agent     │    │
│  └──────────┘  └──────────┘  └──────────┘  └──────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                              │
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│  LLM Providers  │  │   Indicators    │  │ Memory System   │
│ Claude/GPT/Grok │  │ EMA/RSI/MACD    │  │ Short/Long/Epi  │
└─────────────────┘  └─────────────────┘  └─────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      BROKER INTERFACE                            │
│            Polymarket CLOB API | Order Execution                │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Quick Start

### 2.1 Five-Minute Setup

```bash
# 1. Clone and enter directory
cd alpha-arena

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure credentials
cp .env.example .env
# Edit .env with your API keys

# 5. Verify setup
python cli.py providers

# 6. Start simulation
python cli.py run --simulation
```

### 2.2 First Live Trade

```bash
# Ensure .env has WALLET_PRIVATE_KEY and Polymarket credentials

# Check market status
python cli.py markets

# Start with enhanced multi-agent mode (recommended)
python cli.py run-enhanced --dry-run  # Preview only

# When ready for live trading
python cli.py run-enhanced
```

---

## 3. Installation & Setup

### 3.1 System Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| Python | 3.11+ | 3.12 |
| RAM | 4GB | 8GB+ |
| Storage | 1GB | 10GB |
| Network | Stable broadband | Low-latency connection |
| OS | macOS/Linux | Ubuntu 22.04 LTS |

### 3.2 Python Environment

```bash
# Create isolated environment
python -m venv venv

# Activate (macOS/Linux)
source venv/bin/activate

# Activate (Windows)
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "from src.core import Edge, Confidence; print('Core OK')"
python -c "from src.agents import AgentCoordinator; print('Agents OK')"
python -c "from src.broker import PolymarketBroker; print('Broker OK')"
```

### 3.3 Wallet Setup

1. **Create Polygon Wallet**: Use MetaMask or any Polygon-compatible wallet
2. **Fund with USDC**: Transfer USDC to your Polygon address
3. **Export Private Key**: Settings → Security → Export Private Key
4. **Configure in .env**:
   ```
   WALLET_PRIVATE_KEY=your_private_key_here
   WALLET_ADDRESS=0x_your_address_here
   ```

### 3.4 Polymarket API Credentials

1. Visit [Polymarket](https://polymarket.com)
2. Connect your wallet
3. Navigate to Account → API
4. Generate API credentials
5. Add to `.env`:
   ```
   POLYMARKET_API_KEY=your_key
   POLYMARKET_API_SECRET=your_secret
   POLYMARKET_API_PASSPHRASE=your_passphrase
   ```

### 3.5 LLM Provider Setup

At least one LLM provider is required:

**Anthropic (Recommended)**
```
ANTHROPIC_API_KEY=sk-ant-api...
```

**OpenAI**
```
OPENAI_API_KEY=sk-...
```

**xAI (Grok)**
```
XAI_API_KEY=xai-...
```

**Local Models (Ollama)**
```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull models
ollama pull deepseek-v3
ollama pull qwen2.5

# No API key needed - runs locally
```

---

## 4. Configuration

### 4.1 Configuration Files

| File | Purpose |
|------|---------|
| `.env` | Sensitive credentials (API keys, private keys) |
| `config.yaml` | System configuration (strategies, risk limits) |

### 4.2 config.yaml Reference

```yaml
# Operating mode
mode: live  # "live" or "simulation"

# Agent settings
agent:
  loop_interval_seconds: 60  # Time between analysis cycles
  max_iterations: null       # null = infinite, or set limit

# LLM configuration
llm:
  default_provider: anthropic
  default_model: claude-sonnet-4-20250514

  providers:
    anthropic:
      models:
        - claude-sonnet-4-20250514
        - claude-haiku-3-5-20241022
        - claude-opus-4-20250514
      temperature: 0.3
      max_tokens: 4096

    openai:
      models:
        - gpt-4o
        - gpt-4o-mini
        - o1-preview
        - o1-mini
      temperature: 0.3
      max_tokens: 4096

    xai:
      models:
        - grok-2-latest
        - grok-3-latest
      temperature: 0.3
      max_tokens: 4096

    local:
      backend: ollama
      base_url: http://localhost:11434
      models:
        - deepseek-v3
        - qwen2.5
        - llama3.3

# Multi-agent configuration
agents:
  research_model: claude-sonnet-4-20250514
  risk_model: claude-sonnet-4-20250514
  execution_model: claude-haiku-3-5-20241022
  reflection_model: claude-sonnet-4-20250514
  enable_debate: true     # Agents debate before decisions
  debate_rounds: 2        # Number of debate rounds
  enable_reflection: true # Learn from outcomes

# Risk controls
risk:
  max_position_size_usdc: 100    # Max per position
  daily_loss_limit_usdc: 50      # Daily loss stop
  max_open_positions: 5          # Position count limit
  max_single_trade_usdc: 25      # Per-trade maximum
  rate_limit_orders_per_minute: 10
  kill_switch: false             # Emergency stop all trading
  max_kelly_fraction: 0.25       # Quarter-Kelly sizing
  min_edge_for_trade: 0.05       # 5% edge required

# Strategy settings
strategy:
  default: multi_agent
  min_confidence: 0.6    # Minimum confidence to trade
  min_edge: 0.05         # Minimum expected edge

# Exit planning
exit:
  profit_target_pct: 0.15  # Take profit at 15%
  stop_loss_pct: 0.10      # Stop loss at 10%
  max_hold_hours: 72       # Maximum hold time

# Market filters
markets:
  categories: []  # Empty = all, or ["politics", "crypto", "sports"]
  min_liquidity_usdc: 5000
  min_volume_24h_usdc: 1000
  max_markets_per_cycle: 20
```

### 4.3 Environment Variables

```bash
# Required
WALLET_PRIVATE_KEY=      # Polygon wallet private key
WALLET_ADDRESS=          # Polygon wallet address
POLYMARKET_API_KEY=      # Polymarket API key
POLYMARKET_API_SECRET=   # Polymarket API secret
POLYMARKET_API_PASSPHRASE= # Polymarket passphrase

# LLM Providers (at least one)
ANTHROPIC_API_KEY=       # Claude
OPENAI_API_KEY=          # GPT-4o
XAI_API_KEY=             # Grok

# Optional - Risk Overrides
MAX_POSITION_SIZE_USDC=100
DAILY_LOSS_LIMIT_USDC=50
MAX_OPEN_POSITIONS=5
KILL_SWITCH=false

# Optional - Agent Overrides
DEFAULT_LLM_PROVIDER=anthropic
DEFAULT_LLM_MODEL=claude-sonnet-4-20250514
AGENT_LOOP_INTERVAL_SECONDS=60
```

---

## 5. CLI Commands

### 5.1 Command Overview

```bash
python cli.py --help
```

| Command | Description |
|---------|-------------|
| `run` | Start basic trading loop |
| `run-enhanced` | Start multi-agent trading loop |
| `markets` | List active markets |
| `positions` | Show current positions |
| `history` | View trade history |
| `balance` | Check wallet balance |
| `backtest` | Run strategy backtesting |
| `arbitrage` | Scan for arbitrage opportunities |
| `indicators` | View technical indicators |
| `providers` | Check LLM provider status |
| `config` | Display current configuration |

### 5.2 run - Basic Trading Loop

```bash
# Start in simulation mode
python cli.py run --simulation

# Start live trading
python cli.py run

# Limit iterations
python cli.py run --max-iterations 10

# Custom interval
python cli.py run --interval 30  # 30 seconds between cycles

# Specify config file
python cli.py run --config custom-config.yaml
```

### 5.3 run-enhanced - Multi-Agent Mode

The enhanced runner activates the full multi-agent architecture with debate, reflection, and comprehensive analysis.

```bash
# Dry run (no actual trades)
python cli.py run-enhanced --dry-run

# Live trading with multi-agent
python cli.py run-enhanced

# Custom settings
python cli.py run-enhanced \
  --config config.yaml \
  --max-iterations 100 \
  --interval 60

# Options:
#   --config, -c        Configuration file path
#   --dry-run          Preview mode, no actual execution
#   --max-iterations   Maximum cycles (default: unlimited)
#   --interval, -i     Seconds between cycles (default: 60)
```

### 5.4 markets - List Markets

```bash
# Show all markets
python cli.py markets

# Filter by category
python cli.py markets --category politics

# Show detailed view
python cli.py markets --detailed

# Limit results
python cli.py markets --limit 10

# Output format
python cli.py markets --format json > markets.json
```

### 5.5 positions - Current Positions

```bash
# Show all positions
python cli.py positions

# Detailed P&L breakdown
python cli.py positions --detailed

# Filter by status
python cli.py positions --status open

# JSON export
python cli.py positions --format json
```

### 5.6 history - Trade History

```bash
# Recent trades
python cli.py history

# Last N trades
python cli.py history --limit 50

# Date range
python cli.py history --start 2024-01-01 --end 2024-01-31

# Filter by market
python cli.py history --market "Presidential Election"

# Export to CSV
python cli.py history --format csv > trades.csv
```

### 5.7 balance - Wallet Balance

```bash
# Show balance
python cli.py balance

# Include pending orders
python cli.py balance --include-pending
```

### 5.8 backtest - Strategy Backtesting

```bash
# Basic backtest with synthetic data
python cli.py backtest

# Custom parameters
python cli.py backtest \
  --days 90 \
  --capital 10000 \
  --buy-threshold 0.35 \
  --sell-threshold 0.65

# Options:
#   --days           Days of data to backtest (default: 30)
#   --capital        Starting capital in USDC (default: 10000)
#   --buy-threshold  Buy when price below this (default: 0.40)
#   --sell-threshold Sell when price above this (default: 0.60)
```

### 5.9 arbitrage - Opportunity Scanner

```bash
# Scan for opportunities
python cli.py arbitrage

# Set minimum profit threshold
python cli.py arbitrage --min-profit 0.5  # 0.5%

# Continuous monitoring
python cli.py arbitrage --watch

# Filter by type
python cli.py arbitrage --type binary_complement
python cli.py arbitrage --type cross_platform
```

### 5.10 indicators - Technical Analysis

```bash
# View indicators for a market
python cli.py indicators --market "market_id_here"

# All indicators
python cli.py indicators --all

# Specific indicators
python cli.py indicators --indicator rsi --indicator macd
```

### 5.11 providers - LLM Status

```bash
# Check all providers
python cli.py providers

# Test specific provider
python cli.py providers --test anthropic

# Output:
# ┌─────────────┬──────────┬─────────────────────────┐
# │ Provider    │ Status   │ Models                  │
# ├─────────────┼──────────┼─────────────────────────┤
# │ anthropic   │ ✓ Ready  │ claude-sonnet-4, ...    │
# │ openai      │ ✓ Ready  │ gpt-4o, o1-preview      │
# │ xai         │ ✓ Ready  │ grok-2-latest           │
# │ local       │ ✓ Ready  │ deepseek-v3, qwen2.5    │
# └─────────────┴──────────┴─────────────────────────┘
```

### 5.12 config - View Configuration

```bash
# Display current config
python cli.py config

# Show specific section
python cli.py config --section risk
python cli.py config --section agents
python cli.py config --section llm
```

---

## 6. Multi-Agent System

### 6.1 Agent Architecture

Alpha Arena employs four specialized agents that collaborate through a coordinator:

```
                    ┌─────────────────────┐
                    │  Agent Coordinator  │
                    │                     │
                    │  - Orchestration    │
                    │  - Debate Protocol  │
                    │  - Consensus        │
                    └─────────┬───────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│   Research    │    │     Risk      │    │   Execution   │
│    Agent      │    │    Agent      │    │    Agent      │
├───────────────┤    ├───────────────┤    ├───────────────┤
│ - Market data │    │ - Kelly sizing│    │ - Order entry │
│ - News/events │    │ - Exposure    │    │ - Timing      │
│ - Sentiment   │    │ - Stop-loss   │    │ - Slippage    │
│ - Probability │    │ - Correlation │    │ - Monitoring  │
└───────────────┘    └───────────────┘    └───────────────┘
                              │
                              ▼
                    ┌───────────────┐
                    │  Reflection   │
                    │    Agent      │
                    ├───────────────┤
                    │ - Outcome     │
                    │   analysis    │
                    │ - Learning    │
                    │ - Memory      │
                    │   storage     │
                    └───────────────┘
```

### 6.2 Research Agent

**Purpose**: Gather intelligence and form probability estimates

**Responsibilities**:
- Fetch and analyze market data
- Parse news and events relevant to markets
- Assess sentiment (social media, news)
- Generate probability estimates
- Identify market catalysts

**Configuration**:
```yaml
agents:
  research_model: claude-sonnet-4-20250514
```

**Example Analysis Output**:
```json
{
  "market_id": "0x123abc",
  "analysis": {
    "fair_probability": 0.65,
    "confidence": 0.78,
    "reasoning": "Recent polling data shows...",
    "key_factors": [
      "Polling trend +3% in last week",
      "Major endorsement received",
      "Historical correlation with similar events"
    ],
    "catalysts": [
      {"event": "Debate on Jan 15", "impact": "high"},
      {"event": "Jobs report Jan 10", "impact": "medium"}
    ]
  }
}
```

### 6.3 Risk Agent

**Purpose**: Ensure optimal position sizing and risk management

**Responsibilities**:
- Calculate Kelly Criterion sizing
- Monitor portfolio exposure
- Set stop-loss and take-profit levels
- Assess correlation between positions
- Enforce risk limits

**Kelly Criterion Formula**:
```
f* = (b × p - q) / b

Where:
  f* = Optimal fraction of bankroll
  b  = Odds received (e.g., 2:1 = 2)
  p  = Probability of winning
  q  = Probability of losing (1 - p)
```

**Example**:
```python
# Market price: 0.40 (implies 40% probability)
# Your estimate: 55% probability
# Edge: 15%

# Kelly calculation:
# b = (1/0.40) - 1 = 1.5  (potential profit ratio)
# p = 0.55, q = 0.45

# f* = (1.5 × 0.55 - 0.45) / 1.5
# f* = (0.825 - 0.45) / 1.5
# f* = 0.25 or 25% of bankroll

# With quarter-Kelly (max_kelly_fraction: 0.25):
# Position = 0.25 × 0.25 = 6.25% of bankroll
```

**Configuration**:
```yaml
agents:
  risk_model: claude-sonnet-4-20250514

risk:
  max_kelly_fraction: 0.25    # Use quarter-Kelly
  min_edge_for_trade: 0.05    # Require 5% edge
  max_position_size_usdc: 100
  daily_loss_limit_usdc: 50
```

### 6.4 Execution Agent

**Purpose**: Optimal trade execution and order management

**Responsibilities**:
- Determine optimal entry timing
- Manage order placement
- Monitor fills and slippage
- Handle partial fills
- Execute exit strategies

**Configuration**:
```yaml
agents:
  execution_model: claude-haiku-3-5-20241022  # Fast model for execution

exit:
  profit_target_pct: 0.15  # Take profit at 15%
  stop_loss_pct: 0.10      # Stop loss at 10%
  max_hold_hours: 72       # Max position duration
```

**Execution Modes**:

| Mode | Description | Use Case |
|------|-------------|----------|
| `market` | Immediate execution | Urgent entries/exits |
| `limit` | Price-specific | Normal trading |
| `twap` | Time-weighted average | Large positions |
| `iceberg` | Hidden size | Reduce market impact |

### 6.5 Reflection Agent

**Purpose**: Learn from outcomes and improve future decisions

**Responsibilities**:
- Analyze completed trades
- Identify patterns in successes/failures
- Update memory with learnings
- Suggest strategy adjustments
- Generate performance reports

**Configuration**:
```yaml
agents:
  reflection_model: claude-sonnet-4-20250514
  enable_reflection: true
```

**Learning Categories**:
- **Market Analysis**: Was probability estimate accurate?
- **Timing**: Did we enter/exit at good times?
- **Sizing**: Was position size appropriate?
- **Risk**: Did we manage downside effectively?

### 6.6 Agent Debate Protocol

When enabled, agents debate before making decisions:

```yaml
agents:
  enable_debate: true
  debate_rounds: 2
```

**Debate Flow**:

```
Round 1:
┌──────────────┐     ┌──────────────┐
│   Research   │ ──► │     Risk     │
│   "Buy at    │     │   "Sizing    │
│    0.40"     │     │    looks     │
│              │     │    high"     │
└──────────────┘     └──────────────┘
        │                   │
        └───────┬───────────┘
                ▼
Round 2:
┌──────────────┐     ┌──────────────┐
│   Research   │ ◄── │     Risk     │
│  "Confirmed  │     │  "Adjusted   │
│   with new   │     │   to 0.15    │
│   catalyst"  │     │   Kelly"     │
└──────────────┘     └──────────────┘
                │
                ▼
         ┌──────────────┐
         │  Consensus   │
         │   Decision   │
         └──────────────┘
```

### 6.7 Consensus Mechanisms

**Voting**: Each agent votes on proposed actions
```python
# Simple majority
votes = {
    "research": "buy",
    "risk": "buy",
    "execution": "hold"  # Concerned about liquidity
}
# Result: Buy (2-1)
```

**Weighted Consensus**: Different weights per agent type
```python
weights = {
    "research": 0.4,
    "risk": 0.35,
    "execution": 0.25
}
```

**Veto Power**: Risk agent can veto any trade that violates limits

---

## 7. LLM Providers

### 7.1 Provider Overview

| Provider | Models | Best For | Cost |
|----------|--------|----------|------|
| Anthropic | Claude Opus 4, Sonnet 4, Haiku 3.5 | Complex reasoning, research | $$$ |
| OpenAI | GPT-4o, o1-preview, o1-mini | General analysis | $$$ |
| xAI | Grok 2, Grok 3 | Real-time X/Twitter sentiment | $$ |
| Local | DeepSeek-v3, Qwen 2.5, Llama 3.3 | Cost-free, privacy | Free |

### 7.2 Anthropic (Claude)

**Recommended for**: Research and risk analysis

**Models**:
| Model | Capabilities | Use Case |
|-------|-------------|----------|
| claude-opus-4-20250514 | Most capable | Complex market analysis |
| claude-sonnet-4-20250514 | Balanced | Default for all agents |
| claude-haiku-3-5-20241022 | Fast, efficient | Execution, quick decisions |

**Setup**:
```bash
# Get API key from console.anthropic.com
export ANTHROPIC_API_KEY=sk-ant-api...
```

**Configuration**:
```yaml
llm:
  default_provider: anthropic
  default_model: claude-sonnet-4-20250514

  providers:
    anthropic:
      models:
        - claude-opus-4-20250514
        - claude-sonnet-4-20250514
        - claude-haiku-3-5-20241022
      temperature: 0.3
      max_tokens: 4096
```

### 7.3 OpenAI (GPT-4o, o1)

**Recommended for**: General analysis and reasoning

**Models**:
| Model | Capabilities | Use Case |
|-------|-------------|----------|
| gpt-4o | Multimodal, fast | General analysis |
| gpt-4o-mini | Efficient | Quick tasks |
| o1-preview | Advanced reasoning | Complex probability |
| o1-mini | Fast reasoning | Quick reasoning tasks |

**Setup**:
```bash
export OPENAI_API_KEY=sk-...
```

**Configuration**:
```yaml
llm:
  providers:
    openai:
      models:
        - gpt-4o
        - gpt-4o-mini
        - o1-preview
        - o1-mini
      temperature: 0.3
      max_tokens: 4096
```

### 7.4 xAI (Grok)

**Recommended for**: Real-time social sentiment analysis

**Unique Capability**: Direct access to X/Twitter data for sentiment analysis

**Models**:
| Model | Capabilities |
|-------|-------------|
| grok-2-latest | Real-time X sentiment |
| grok-3-latest | Enhanced reasoning + sentiment |

**Setup**:
```bash
export XAI_API_KEY=xai-...
```

**Configuration**:
```yaml
llm:
  providers:
    xai:
      models:
        - grok-2-latest
        - grok-3-latest
      temperature: 0.3
      max_tokens: 4096
```

**Use Case Example**:
```python
# Grok excels at real-time sentiment:
# "What is the current Twitter sentiment around [candidate]?"
# "Are there trending topics affecting [market]?"
```

### 7.5 Local Models (Ollama)

**Recommended for**: Cost-sensitive operations, privacy, offline usage

**Setup**:
```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Start Ollama service
ollama serve

# Pull recommended models
ollama pull deepseek-v3       # Best for coding/analysis
ollama pull qwen2.5           # Strong general model
ollama pull llama3.3          # Meta's latest
```

**Configuration**:
```yaml
llm:
  providers:
    local:
      backend: ollama
      base_url: http://localhost:11434
      models:
        - deepseek-v3
        - qwen2.5
        - llama3.3
```

**Comparison**:
| Model | Parameters | VRAM Required | Best For |
|-------|------------|---------------|----------|
| deepseek-v3 | 70B (MoE) | 32GB+ | Analysis |
| qwen2.5 | 72B | 40GB+ | General |
| llama3.3 | 70B | 40GB+ | Reasoning |

### 7.6 Provider Selection Strategy

**Recommended Configuration**:
```yaml
agents:
  # Use Claude for deep analysis
  research_model: claude-sonnet-4-20250514
  risk_model: claude-sonnet-4-20250514

  # Use Haiku for fast execution decisions
  execution_model: claude-haiku-3-5-20241022

  # Use Sonnet for reflection/learning
  reflection_model: claude-sonnet-4-20250514
```

**Cost Optimization**:
```yaml
# For cost-sensitive operation, use local models:
agents:
  research_model: deepseek-v3    # Local
  risk_model: qwen2.5            # Local
  execution_model: llama3.3      # Local
  reflection_model: deepseek-v3  # Local
```

**Hybrid Approach**:
```yaml
# Mix cloud + local for balance:
agents:
  research_model: claude-sonnet-4-20250514  # Cloud for accuracy
  risk_model: deepseek-v3                   # Local for cost
  execution_model: claude-haiku-3-5-20241022 # Cloud for speed
  reflection_model: deepseek-v3             # Local for cost
```

### 7.7 Fallback Chain

The system automatically falls back if a provider fails:

```python
fallback_chain = [
    "anthropic",  # Try first
    "openai",     # If Anthropic fails
    "xai",        # If OpenAI fails
    "local"       # Final fallback
]
```

---

## 8. Market Analysis

### 8.1 Market Data Structure

```python
@dataclass
class Market:
    condition_id: str          # Unique market identifier
    question: str              # Market question
    outcomes: list[str]        # Possible outcomes
    end_date: datetime         # Resolution date
    volume_24h: Decimal        # 24-hour volume
    liquidity: Decimal         # Available liquidity

    # Order book
    yes_bid: Decimal          # Best YES bid
    yes_ask: Decimal          # Best YES ask
    no_bid: Decimal           # Best NO bid
    no_ask: Decimal           # Best NO ask
```

### 8.2 Fetching Markets

```bash
# CLI
python cli.py markets --detailed

# Programmatic
from src.data import PolymarketDataFetcher

async def get_markets():
    fetcher = PolymarketDataFetcher()
    markets = await fetcher.fetch_active_markets()

    for market in markets:
        print(f"{market.question}")
        print(f"  YES: {market.yes_ask:.2f} / NO: {market.no_ask:.2f}")
        print(f"  Volume: ${market.volume_24h:,.2f}")
```

### 8.3 Market Filtering

```yaml
# config.yaml
markets:
  categories: ["politics", "crypto"]  # Filter by category
  min_liquidity_usdc: 5000            # Minimum liquidity
  min_volume_24h_usdc: 1000           # Minimum 24h volume
  max_markets_per_cycle: 20           # Limit per analysis cycle
```

### 8.4 Price History

```python
from src.data import PolymarketDataFetcher

async def analyze_history():
    fetcher = PolymarketDataFetcher()

    # Get price history
    history = await fetcher.fetch_price_history(
        market_id="0x123abc",
        interval="1h",  # 1m, 5m, 15m, 1h, 4h, 1d
        limit=168       # Last 7 days hourly
    )

    for point in history:
        print(f"{point.timestamp}: {point.price:.4f}")
```

### 8.5 Order Book Analysis

```python
from src.broker import PolymarketBroker

async def analyze_orderbook():
    broker = PolymarketBroker()

    orderbook = await broker.get_orderbook("0x123abc")

    # Best prices
    best_bid = orderbook.bids[0] if orderbook.bids else None
    best_ask = orderbook.asks[0] if orderbook.asks else None

    # Spread
    spread = best_ask.price - best_bid.price if best_bid and best_ask else None

    # Depth
    bid_depth = sum(order.size for order in orderbook.bids[:10])
    ask_depth = sum(order.size for order in orderbook.asks[:10])
```

### 8.6 Market Categories

| Category | Description | Example Markets |
|----------|-------------|-----------------|
| politics | Elections, policy | "Will X win election?" |
| crypto | Cryptocurrency | "Will BTC exceed $100k?" |
| sports | Sports outcomes | "Will team X win?" |
| entertainment | Pop culture | "Will movie X win Oscar?" |
| business | Corporate events | "Will merger complete?" |
| science | Scientific events | "Will discovery happen?" |

---

## 9. Trading Strategies

### 9.1 Built-in Strategies

| Strategy | Description | Risk Level |
|----------|-------------|------------|
| `multi_agent` | Full agent collaboration | Medium |
| `momentum` | Follow price trends | Medium-High |
| `mean_reversion` | Bet on price normalization | Medium |
| `arbitrage` | Exploit price differences | Low |
| `event_driven` | Trade around catalysts | High |

### 9.2 Multi-Agent Strategy (Default)

The full multi-agent pipeline:

```
1. RESEARCH PHASE
   └── Gather market data, news, sentiment
   └── Generate probability estimates
   └── Identify opportunities with edge

2. RISK PHASE
   └── Calculate Kelly sizing
   └── Check position limits
   └── Assess portfolio correlation

3. DEBATE PHASE (if enabled)
   └── Agents present arguments
   └── Multiple rounds of refinement
   └── Reach consensus

4. EXECUTION PHASE
   └── Determine optimal entry
   └── Place orders
   └── Monitor fills

5. REFLECTION PHASE (ongoing)
   └── Track outcome
   └── Learn from result
   └── Update memory
```

### 9.3 Edge Calculation

```python
# Fair value from research agent
fair_value = 0.65  # 65% probability

# Market price
market_price = 0.55  # Trading at 55%

# Edge calculation
if fair_value > market_price:
    # Buy YES
    edge = fair_value - market_price  # 10% edge
    direction = "BUY_YES"
else:
    # Buy NO (or sell YES)
    edge = market_price - fair_value
    direction = "BUY_NO"

# Only trade if edge exceeds minimum
min_edge = 0.05  # 5%
if edge >= min_edge:
    # Proceed with trade
    pass
```

### 9.4 Position Sizing with Kelly

```python
def calculate_position_size(
    edge: float,
    market_price: float,
    bankroll: float,
    max_kelly: float = 0.25
) -> float:
    """Calculate position size using Kelly Criterion."""

    # Convert to odds
    if market_price < 0.5:
        # Buying YES
        p = market_price + edge  # Our probability
        b = (1 / market_price) - 1  # Payoff ratio
    else:
        # Buying NO
        p = (1 - market_price) + edge
        b = (1 / (1 - market_price)) - 1

    q = 1 - p  # Probability of loss

    # Kelly formula
    kelly = (b * p - q) / b

    # Apply fraction (quarter-Kelly recommended)
    kelly = kelly * max_kelly

    # Cap at maximum
    kelly = min(kelly, 0.10)  # Never more than 10%

    return bankroll * kelly
```

### 9.5 Entry Strategies

**Immediate Entry**:
```python
# For high-conviction opportunities
order = await broker.place_order(
    market_id=market.condition_id,
    side="BUY",
    outcome="YES",
    amount=position_size,
    order_type="MARKET"
)
```

**Limit Entry**:
```python
# For price-sensitive entries
target_price = market.yes_bid + Decimal("0.01")  # 1 cent above bid

order = await broker.place_order(
    market_id=market.condition_id,
    side="BUY",
    outcome="YES",
    amount=position_size,
    price=target_price,
    order_type="LIMIT"
)
```

**Scaled Entry**:
```python
# Split into multiple orders
total_size = position_size
num_orders = 3
prices = [
    market.yes_bid,
    market.yes_bid + Decimal("0.01"),
    market.yes_bid + Decimal("0.02")
]

for price in prices:
    await broker.place_order(
        market_id=market.condition_id,
        side="BUY",
        outcome="YES",
        amount=total_size / num_orders,
        price=price,
        order_type="LIMIT"
    )
```

### 9.6 Exit Strategies

**Take Profit**:
```yaml
exit:
  profit_target_pct: 0.15  # Exit at 15% profit
```

**Stop Loss**:
```yaml
exit:
  stop_loss_pct: 0.10  # Exit at 10% loss
```

**Time-Based**:
```yaml
exit:
  max_hold_hours: 72  # Exit after 72 hours regardless
```

**Event-Driven**:
- Exit before major catalyst if uncertainty too high
- Exit if thesis invalidated by new information
- Exit if better opportunity identified

---

## 10. Risk Management

### 10.1 Risk Controls Overview

```yaml
risk:
  # Position limits
  max_position_size_usdc: 100    # Per position
  max_open_positions: 5          # Total positions
  max_single_trade_usdc: 25      # Per trade

  # Loss limits
  daily_loss_limit_usdc: 50      # Daily stop

  # Sizing
  max_kelly_fraction: 0.25       # Quarter-Kelly
  min_edge_for_trade: 0.05       # Require 5% edge

  # Rate limiting
  rate_limit_orders_per_minute: 10

  # Emergency
  kill_switch: false
```

### 10.2 Position Limits

| Control | Purpose | Default |
|---------|---------|---------|
| `max_position_size_usdc` | Max capital per position | 100 |
| `max_open_positions` | Max concurrent positions | 5 |
| `max_single_trade_usdc` | Max per individual trade | 25 |

### 10.3 Daily Loss Limit

The system tracks daily P&L and stops trading when limit is hit:

```python
# Automatic enforcement
daily_loss = sum(closed_pnl for trade in today_trades)

if daily_loss <= -daily_loss_limit:
    # Trading halted for the day
    log.warning(f"Daily loss limit hit: ${daily_loss}")
    return TradingHalted(reason="daily_loss_limit")
```

### 10.4 Kelly Criterion Sizing

**Full Kelly** (aggressive): f* = (bp - q) / b
**Half Kelly** (moderate): f* × 0.5
**Quarter Kelly** (conservative): f* × 0.25

```yaml
# Recommended: Quarter Kelly
risk:
  max_kelly_fraction: 0.25
```

**Why Quarter Kelly?**
- Full Kelly assumes perfect probability estimates
- Reduces volatility significantly
- Still captures most of the growth

### 10.5 Minimum Edge Requirement

```yaml
risk:
  min_edge_for_trade: 0.05  # 5% edge
```

Only trade when: `|fair_value - market_price| >= min_edge`

### 10.6 Kill Switch

Emergency stop all trading:

```bash
# Via environment
export KILL_SWITCH=true

# Via config
risk:
  kill_switch: true

# Via CLI (if implemented)
python cli.py kill-switch --enable
```

### 10.7 Risk Monitoring Dashboard

```
┌─────────────────────────────────────────────────────────────┐
│                    RISK DASHBOARD                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Daily P&L:     -$23.45  ████████░░░░░░░ (47% of limit)    │
│  Open Exposure: $287.50  █████████████░░ (57% of limit)    │
│  Positions:     3/5      ██████░░░░░░░░░                   │
│                                                             │
│  Position Breakdown:                                        │
│  ┌──────────────────┬────────┬────────┬─────────┐          │
│  │ Market           │ Size   │ P&L    │ Risk    │          │
│  ├──────────────────┼────────┼────────┼─────────┤          │
│  │ Presidential     │ $95.00 │ +$8.50 │ LOW     │          │
│  │ BTC > 100k       │ $87.50 │ -$12.3 │ MEDIUM  │          │
│  │ Fed Rate Cut     │ $105.0 │ -$19.6 │ HIGH    │          │
│  └──────────────────┴────────┴────────┴─────────┘          │
│                                                             │
│  Correlation Matrix:                                        │
│  Presidential  ─┬─ BTC: 0.12 (low)                         │
│                └─ Fed: 0.45 (moderate)                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 11. Technical Indicators

### 11.1 Available Indicators

| Indicator | Full Name | Purpose |
|-----------|-----------|---------|
| EMA | Exponential Moving Average | Trend direction |
| RSI | Relative Strength Index | Overbought/oversold |
| MACD | Moving Average Convergence Divergence | Momentum |
| ATR | Average True Range | Volatility |
| Volume | Volume Analysis | Confirmation |
| S/R | Support/Resistance | Key levels |

### 11.2 EMA (Exponential Moving Average)

```yaml
indicators:
  ema_periods: [9, 21, 50]  # Short, medium, long
```

**Interpretation**:
- Price > EMA: Bullish
- Price < EMA: Bearish
- EMA9 > EMA21 > EMA50: Strong uptrend
- EMA9 < EMA21 < EMA50: Strong downtrend

### 11.3 RSI (Relative Strength Index)

```yaml
indicators:
  rsi_period: 14
  rsi_oversold: 30
  rsi_overbought: 70
```

**Interpretation**:
- RSI > 70: Overbought (potential sell)
- RSI < 30: Oversold (potential buy)
- RSI 30-70: Neutral zone

### 11.4 MACD

```yaml
indicators:
  macd_fast: 12
  macd_slow: 26
  macd_signal: 9
```

**Components**:
- MACD Line: EMA12 - EMA26
- Signal Line: EMA9 of MACD Line
- Histogram: MACD Line - Signal Line

**Signals**:
- MACD crosses above Signal: Bullish
- MACD crosses below Signal: Bearish
- Histogram increasing: Momentum strengthening

### 11.5 ATR (Average True Range)

```yaml
indicators:
  atr_period: 14
```

**Use Cases**:
- Position sizing (smaller in volatile markets)
- Stop-loss placement (1-2 × ATR)
- Volatility filtering

### 11.6 Volume Analysis

```yaml
indicators:
  volume_ma_period: 20
```

**Signals**:
- High volume + price move: Confirmed trend
- Low volume + price move: Weak/suspicious move
- Volume spike: Potential reversal or breakout

### 11.7 Using Indicators Programmatically

```python
from src.indicators import IndicatorCalculator, MarketOHLCV

calculator = IndicatorCalculator()

# Prepare OHLCV data
ohlcv_data = [
    MarketOHLCV(
        timestamp=datetime.now() - timedelta(hours=i),
        open=Decimal("0.45"),
        high=Decimal("0.47"),
        low=Decimal("0.44"),
        close=Decimal("0.46"),
        volume=Decimal("10000")
    )
    for i in range(100)
]

# Calculate all indicators
result = calculator.calculate_all(ohlcv_data)

# Access individual indicators
print(f"RSI: {result.rsi.value}")
print(f"MACD: {result.macd.macd_line}")
print(f"Signal Strength: {result.signal_strength}")
print(f"Trend: {result.trend}")  # BULLISH, BEARISH, NEUTRAL
```

### 11.8 CLI Indicator View

```bash
python cli.py indicators --market "0x123abc"

# Output:
# ┌─────────────────────────────────────────────────────────┐
# │ Technical Analysis: Will X happen by Y?                 │
# ├─────────────────────────────────────────────────────────┤
# │ Price: 0.4500                                           │
# │                                                         │
# │ EMAs:                                                   │
# │   EMA9:  0.4520 (price below - bearish)                │
# │   EMA21: 0.4480 (price above - bullish)                │
# │   EMA50: 0.4400 (price above - bullish)                │
# │                                                         │
# │ RSI(14): 42.5 (neutral)                                │
# │                                                         │
# │ MACD:                                                   │
# │   MACD Line: 0.0012                                    │
# │   Signal:    0.0008                                    │
# │   Histogram: 0.0004 (bullish)                          │
# │                                                         │
# │ ATR(14): 0.0234                                        │
# │                                                         │
# │ Overall Signal: SLIGHTLY BULLISH (strength: 0.35)      │
# └─────────────────────────────────────────────────────────┘
```

---

## 12. Arbitrage Detection

### 12.1 Arbitrage Types

| Type | Description | Example |
|------|-------------|---------|
| Binary Complement | YES + NO should = 1 | YES=0.45, NO=0.50 → 5% arb |
| Cross-Platform | Same market, different prices | PM: 0.40, Kalshi: 0.45 |
| Correlated Markets | Related markets mispriced | Similar events, different odds |

### 12.2 Binary Complement Arbitrage

In prediction markets: P(YES) + P(NO) = 1

If YES = 0.45 and NO = 0.50:
- Total = 0.95 (should be 1.00)
- Buy both: Guaranteed 0.05 profit per share

```python
from src.arbitrage import ArbitrageDetector

detector = ArbitrageDetector()

# Find opportunities
opportunities = await detector.find_binary_complement_arbs(
    markets=markets,
    min_profit_pct=0.5  # 0.5% minimum
)

for opp in opportunities:
    print(f"Market: {opp.market_id}")
    print(f"YES: {opp.yes_price}, NO: {opp.no_price}")
    print(f"Profit: {opp.profit_pct}%")
```

### 12.3 Cross-Platform Arbitrage

```python
# Requires multiple platform integrations
opportunities = await detector.find_cross_platform_arbs(
    platforms=["polymarket", "kalshi"],
    min_profit_pct=0.5
)

for opp in opportunities:
    print(f"Market: {opp.question}")
    print(f"Polymarket: {opp.pm_price}")
    print(f"Kalshi: {opp.kalshi_price}")
    print(f"Action: Buy on {opp.buy_platform}, Sell on {opp.sell_platform}")
    print(f"Profit: {opp.profit_pct}%")
```

### 12.4 CLI Arbitrage Scanner

```bash
# Scan for opportunities
python cli.py arbitrage

# Set minimum profit
python cli.py arbitrage --min-profit 1.0

# Continuous watch mode
python cli.py arbitrage --watch

# Output:
# ┌─────────────────────────────────────────────────────────┐
# │                 ARBITRAGE OPPORTUNITIES                 │
# ├─────────────────────────────────────────────────────────┤
# │                                                         │
# │ [BINARY COMPLEMENT]                                     │
# │ Market: Will BTC exceed $100k?                          │
# │ YES: 0.4500 | NO: 0.5200 | Sum: 0.9700                 │
# │ Profit: 3.00% (risk-free)                              │
# │ Liquidity: $5,230 available                            │
# │                                                         │
# │ [CROSS-PLATFORM] (if Kalshi enabled)                   │
# │ Market: Presidential Election                           │
# │ Polymarket: 0.5200 | Kalshi: 0.5450                    │
# │ Action: Buy PM, Sell Kalshi                            │
# │ Profit: 2.50%                                          │
# │                                                         │
# └─────────────────────────────────────────────────────────┘
```

### 12.5 Configuration

```yaml
arbitrage:
  enabled: true
  min_profit_pct: 0.5  # Minimum 0.5% profit
  platforms:
    - polymarket
    # - kalshi  # Uncomment if you have Kalshi API
```

---

## 13. Backtesting

### 13.1 Overview

Test strategies on historical or synthetic data before live trading.

### 13.2 Running Backtests

```bash
# Basic backtest
python cli.py backtest

# Custom parameters
python cli.py backtest \
  --days 90 \
  --capital 10000 \
  --buy-threshold 0.35 \
  --sell-threshold 0.65
```

### 13.3 Synthetic Data Generation

The backtester can generate realistic market data:

```python
from src.backtest import generate_synthetic_market

# Generate synthetic market
market = generate_synthetic_market(
    market_id="synthetic_1",
    question="Test Market",
    duration_days=30,
    initial_price=Decimal("0.50"),
    volatility=Decimal("0.02"),
    trend=Decimal("0.001"),  # Slight upward drift
    resolution_price=Decimal("1.0")  # Resolves YES
)
```

### 13.4 Built-in Strategies

**Mean Reversion Strategy**:
```python
from src.backtest import SimpleStrategy

strategy = SimpleStrategy(
    buy_threshold=Decimal("0.35"),   # Buy below 0.35
    sell_threshold=Decimal("0.65"),  # Sell above 0.65
    position_size=Decimal("100")     # $100 per trade
)
```

### 13.5 Custom Strategies

```python
from src.backtest import BacktestStrategy, BacktestOrder

class MyStrategy(BacktestStrategy):
    async def on_data(
        self,
        timestamp: datetime,
        market_data: dict[str, MarketSnapshot],
        portfolio: BacktestPortfolio
    ) -> list[BacktestOrder]:
        orders = []

        for market_id, snapshot in market_data.items():
            # Your logic here
            if snapshot.price < Decimal("0.30"):
                orders.append(BacktestOrder(
                    market_id=market_id,
                    side="BUY",
                    size=Decimal("50"),
                    price=snapshot.price
                ))

        return orders
```

### 13.6 Performance Metrics

| Metric | Description | Good Value |
|--------|-------------|------------|
| Total Return | Overall profit/loss | > 0% |
| Sharpe Ratio | Risk-adjusted return | > 1.0 |
| Sortino Ratio | Downside risk-adjusted | > 1.5 |
| Max Drawdown | Largest peak-to-trough | < 20% |
| Win Rate | % profitable trades | > 50% |
| Profit Factor | Gross profit / loss | > 1.5 |

### 13.7 Backtest Report

```
┌─────────────────────────────────────────────────────────────┐
│                    BACKTEST RESULTS                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Period: 2024-01-01 to 2024-03-31 (90 days)                │
│  Starting Capital: $10,000.00                               │
│  Ending Capital:   $11,234.56                               │
│                                                             │
│  PERFORMANCE METRICS                                        │
│  ─────────────────                                          │
│  Total Return:     12.35%                                   │
│  Sharpe Ratio:     1.45                                     │
│  Sortino Ratio:    1.89                                     │
│  Calmar Ratio:     2.12                                     │
│  Max Drawdown:     5.83%                                    │
│                                                             │
│  TRADE STATISTICS                                           │
│  ────────────────                                           │
│  Total Trades:     47                                       │
│  Win Rate:         63.8%                                    │
│  Avg Win:          $89.23                                   │
│  Avg Loss:         $45.67                                   │
│  Profit Factor:    2.45                                     │
│                                                             │
│  EQUITY CURVE                                               │
│  ────────────                                               │
│  $11.5k ┤                                        ╭──        │
│  $11.0k ┤                              ╭────────╯           │
│  $10.5k ┤              ╭───────────────╯                    │
│  $10.0k ┼──────────────╯                                    │
│         └────────────────────────────────────────           │
│         Jan        Feb        Mar        Apr                │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 14. Memory System

### 14.1 Memory Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      MEMORY SYSTEM                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────┐                                        │
│  │  SHORT-TERM     │  Recent events (last hour)            │
│  │  Memory         │  - Current positions                   │
│  │  (In-memory)    │  - Recent price moves                  │
│  │                 │  - Active orders                       │
│  └────────┬────────┘                                        │
│           │ (promote important items)                       │
│           ▼                                                 │
│  ┌─────────────────┐                                        │
│  │  LONG-TERM      │  Persistent knowledge                  │
│  │  Memory         │  - Historical trades                   │
│  │  (SQLite)       │  - Market patterns                     │
│  │                 │  - Learned strategies                  │
│  └────────┬────────┘                                        │
│           │ (query for context)                             │
│           ▼                                                 │
│  ┌─────────────────┐                                        │
│  │  EPISODIC       │  Trade lifecycle memories              │
│  │  Memory         │  - Full trade context                  │
│  │  (SQLite)       │  - Decision reasoning                  │
│  │                 │  - Outcome analysis                    │
│  └─────────────────┘                                        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 14.2 Configuration

```yaml
memory:
  db_path: data/memory.db
  short_term_capacity: 100      # Items in short-term
  short_term_ttl_minutes: 60    # TTL for short-term
  long_term_capacity: 10000     # Items in long-term
  enable_semantic_search: true  # Similarity search
```

### 14.3 Short-Term Memory

Stores recent, high-frequency information:

```python
from src.memory import ShortTermMemory, MemoryEntry

stm = ShortTermMemory(capacity=100, ttl_minutes=60)

# Store recent event
await stm.add(MemoryEntry(
    memory_type="price_update",
    content="BTC market moved from 0.45 to 0.48",
    importance=0.6,
    metadata={"market_id": "0x123", "old_price": 0.45, "new_price": 0.48}
))

# Recall recent memories
recent = await stm.recall(memory_type="price_update", limit=10)
```

### 14.4 Long-Term Memory

Persistent storage for important learnings:

```python
from src.memory import LongTermMemory

ltm = LongTermMemory.connect("data/memory.db")

# Store learning
await ltm.store(MemoryEntry(
    memory_type="strategy_insight",
    content="RSI below 25 on political markets often precedes reversal",
    importance=0.9,
    metadata={"category": "political", "indicator": "RSI"}
))

# Query similar memories
similar = await ltm.query(
    "RSI signals in political markets",
    limit=5
)
```

### 14.5 Episodic Memory

Complete trade lifecycle records:

```python
from src.memory import EpisodicMemory

em = EpisodicMemory.connect("data/memory.db")

# Record episode start
episode_id = await em.start_episode(
    episode_type="trade",
    context={
        "market_id": "0x123",
        "thesis": "Event will resolve YES due to polling",
        "entry_price": 0.45
    }
)

# Add events during trade
await em.add_event(episode_id, "order_filled", {"price": 0.45, "size": 100})
await em.add_event(episode_id, "price_update", {"price": 0.52})

# Complete episode
await em.complete_episode(episode_id, {
    "exit_price": 0.58,
    "profit": 28.89,
    "outcome": "success",
    "learnings": ["Polling data was predictive"]
})
```

### 14.6 Memory-Informed Decisions

Agents query memory before making decisions:

```python
# Research agent queries for similar markets
similar_trades = await ltm.query(
    f"trades on {market.category} markets with similar volume",
    limit=10
)

# Extract insights
win_rate = sum(1 for t in similar_trades if t.metadata["profit"] > 0) / len(similar_trades)
avg_edge = mean(t.metadata["edge"] for t in similar_trades)

# Inform decision
if win_rate < 0.5:
    confidence *= 0.8  # Reduce confidence based on history
```

---

## 15. Logging & Monitoring

### 15.1 Logging Configuration

```yaml
logging:
  level: INFO          # DEBUG, INFO, WARNING, ERROR
  jsonl_file: logs/decisions.jsonl
  sqlite_file: logs/metrics.db
```

### 15.2 Log Levels

| Level | Use |
|-------|-----|
| DEBUG | Detailed diagnostic info |
| INFO | Normal operation events |
| WARNING | Potential issues |
| ERROR | Failures requiring attention |

### 15.3 Decision Log (JSONL)

Every trading decision is logged:

```json
{
  "timestamp": "2024-01-15T10:30:00Z",
  "cycle_id": "abc123",
  "market_id": "0x123abc",
  "action": "BUY_YES",
  "research": {
    "fair_value": 0.65,
    "confidence": 0.78,
    "reasoning": "Polling data supports outcome"
  },
  "risk": {
    "edge": 0.10,
    "kelly_size": 0.0625,
    "position_usdc": 62.50
  },
  "execution": {
    "order_type": "LIMIT",
    "price": 0.55,
    "status": "FILLED"
  }
}
```

### 15.4 Metrics Database

```sql
-- Performance metrics table
CREATE TABLE metrics (
    timestamp TEXT,
    portfolio_value REAL,
    daily_pnl REAL,
    open_positions INTEGER,
    sharpe_30d REAL,
    win_rate_30d REAL
);

-- Trade log table
CREATE TABLE trades (
    id TEXT PRIMARY KEY,
    market_id TEXT,
    side TEXT,
    entry_price REAL,
    exit_price REAL,
    size REAL,
    pnl REAL,
    entry_time TEXT,
    exit_time TEXT
);
```

### 15.5 Real-Time Monitoring

```bash
# Watch decision log
tail -f logs/decisions.jsonl | jq .

# Monitor specific market
tail -f logs/decisions.jsonl | jq 'select(.market_id == "0x123")'

# Watch errors only
tail -f logs/app.log | grep ERROR
```

### 15.6 Performance Dashboard

```
┌─────────────────────────────────────────────────────────────┐
│                  PERFORMANCE DASHBOARD                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  PORTFOLIO                                                  │
│  ─────────                                                  │
│  Balance: $10,542.30                                        │
│  Open P&L: +$127.45                                         │
│  Daily P&L: +$89.20                                         │
│                                                             │
│  30-DAY METRICS                                             │
│  ─────────────                                              │
│  Return: +8.45%                                             │
│  Sharpe: 1.67                                               │
│  Win Rate: 58.3%                                            │
│  Trades: 24                                                 │
│                                                             │
│  RECENT ACTIVITY                                            │
│  ───────────────                                            │
│  10:30 BUY  Presidential YES @ 0.55  ✓ Filled              │
│  10:28 SELL BTC > 100k NO @ 0.48     ✓ Filled              │
│  10:15 BUY  Fed Rate Cut YES @ 0.32  ⏳ Pending             │
│                                                             │
│  ALERTS                                                     │
│  ──────                                                     │
│  ⚠ Daily P&L approaching 80% of limit                      │
│  ℹ High volume detected on Presidential market              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 16. Troubleshooting

### 16.1 Common Issues

#### API Connection Failed
```
Error: Failed to connect to Polymarket API
```
**Solutions**:
- Check internet connection
- Verify API credentials in `.env`
- Ensure credentials haven't expired
- Check Polymarket status page

#### Insufficient Funds
```
Error: Insufficient USDC balance
```
**Solutions**:
- Check wallet balance: `python cli.py balance`
- Transfer more USDC to wallet
- Reduce position size in config

#### LLM Provider Error
```
Error: Anthropic API rate limited
```
**Solutions**:
- Wait and retry (automatic)
- Switch to backup provider
- Check API key validity
- Upgrade API tier if needed

#### Order Rejected
```
Error: Order rejected - price outside bounds
```
**Solutions**:
- Market price moved; use fresh prices
- Widen limit order spread
- Use market orders for urgent fills

### 16.2 Diagnostic Commands

```bash
# Check configuration
python cli.py config

# Test LLM providers
python cli.py providers --test

# Check connectivity
python cli.py balance

# View recent logs
tail -100 logs/app.log

# Check memory database
sqlite3 data/memory.db "SELECT COUNT(*) FROM long_term_memory"
```

### 16.3 Recovery Procedures

**After Crash**:
```bash
# 1. Check positions
python cli.py positions

# 2. Review open orders
python cli.py orders

# 3. Cancel stale orders if needed
# (manual in Polymarket UI for safety)

# 4. Restart with caution
python cli.py run-enhanced --dry-run
```

**After Loss Limit Hit**:
```bash
# 1. Review what happened
python cli.py history --today

# 2. Analyze decisions
cat logs/decisions.jsonl | jq 'select(.timestamp > "2024-01-15")'

# 3. Reset daily counter (next day automatic)

# 4. Adjust strategy if needed
```

### 16.4 Debug Mode

```bash
# Enable verbose logging
export LOG_LEVEL=DEBUG
python cli.py run-enhanced

# Or in config
logging:
  level: DEBUG
```

### 16.5 Getting Help

1. Check this manual first
2. Review `docs/OPERATIONAL_MANUAL.md` for ops issues
3. Check logs: `logs/app.log`, `logs/decisions.jsonl`
4. File issue: https://github.com/anthropics/claude-code/issues

---

## 17. API Reference

### 17.1 Core Types

```python
from src.core import Edge, Confidence, Signal, Position, Trade

# Edge: Expected advantage
edge = Edge(
    value=Decimal("0.10"),      # 10% edge
    confidence=Decimal("0.75"),  # 75% confident in estimate
    source="research_agent"
)

# Confidence: Multi-factor confidence
confidence = Confidence(
    base=Decimal("0.70"),
    data_quality=Decimal("0.80"),
    model_agreement=Decimal("0.85"),
    market_efficiency=Decimal("0.60")
)

# Signal: Trading signal
signal = Signal(
    direction="BUY",
    strength=Decimal("0.65"),
    confidence=confidence,
    timestamp=datetime.now()
)

# Position
position = Position(
    market_id="0x123",
    outcome="YES",
    size=Decimal("100"),
    entry_price=Decimal("0.55"),
    current_price=Decimal("0.58"),
    unrealized_pnl=Decimal("5.45")
)
```

### 17.2 Broker Interface

```python
from src.broker import PolymarketBroker

broker = PolymarketBroker()

# Place order
order = await broker.place_order(
    market_id="0x123abc",
    side="BUY",
    outcome="YES",
    amount=Decimal("50"),
    price=Decimal("0.55"),
    order_type="LIMIT"
)

# Cancel order
await broker.cancel_order(order.id)

# Get positions
positions = await broker.get_positions()

# Get order book
orderbook = await broker.get_orderbook("0x123abc")
```

### 17.3 Data Fetcher

```python
from src.data import PolymarketDataFetcher

fetcher = PolymarketDataFetcher()

# Fetch markets
markets = await fetcher.fetch_active_markets()

# Fetch specific market
market = await fetcher.fetch_market("0x123abc")

# Price history
history = await fetcher.fetch_price_history(
    market_id="0x123abc",
    interval="1h",
    limit=168
)
```

### 17.4 Agent Coordinator

```python
from src.agents import AgentCoordinator

coordinator = AgentCoordinator(config)

# Full analysis
result = await coordinator.analyze_market(market)

# Access individual analyses
print(result.research.probability_estimate)
print(result.risk.kelly_fraction)
print(result.consensus.should_trade)
```

### 17.5 Indicator Calculator

```python
from src.indicators import IndicatorCalculator

calculator = IndicatorCalculator()

# Calculate all
result = calculator.calculate_all(ohlcv_data)

# Individual indicators
ema = calculator.calculate_ema(prices, period=21)
rsi = calculator.calculate_rsi(prices, period=14)
macd = calculator.calculate_macd(prices)
```

### 17.6 Arbitrage Detector

```python
from src.arbitrage import ArbitrageDetector

detector = ArbitrageDetector()

# Binary complement arbitrage
binary_arbs = await detector.find_binary_complement_arbs(markets)

# Cross-platform (if enabled)
cross_arbs = await detector.find_cross_platform_arbs(
    platforms=["polymarket", "kalshi"]
)
```

### 17.7 Memory Manager

```python
from src.memory import MemoryManager

memory = MemoryManager(config)

# Store
await memory.store(entry)

# Recall short-term
recent = await memory.recall_recent(limit=10)

# Query long-term
relevant = await memory.query("similar market patterns", limit=5)

# Record episode
await memory.record_trade_episode(trade_context)
```

---

## Appendix A: Glossary

| Term | Definition |
|------|------------|
| **Edge** | Expected advantage over market price |
| **Kelly Criterion** | Formula for optimal bet sizing |
| **CLOB** | Central Limit Order Book |
| **Arbitrage** | Risk-free profit from price discrepancies |
| **Sharpe Ratio** | Risk-adjusted return metric |
| **Drawdown** | Peak-to-trough decline |
| **Slippage** | Difference between expected and actual price |
| **USDC** | USD Coin stablecoin on Polygon |

## Appendix B: Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `Ctrl+C` | Graceful shutdown |
| `Ctrl+Z` | Suspend (use `fg` to resume) |

## Appendix C: File Structure

```
alpha-arena/
├── cli.py                 # Main CLI entry point
├── config.yaml            # Configuration
├── .env                   # Credentials (gitignored)
├── src/
│   ├── core/              # Core types and config
│   ├── agents/            # Multi-agent system
│   ├── broker/            # Exchange interface
│   ├── data/              # Data fetching
│   ├── indicators/        # Technical analysis
│   ├── arbitrage/         # Arb detection
│   ├── memory/            # Memory system
│   ├── backtest/          # Backtesting
│   └── runner/            # Trading loops
├── tests/                 # Test suite
├── logs/                  # Runtime logs
├── data/                  # Databases
└── docs/                  # Documentation
```

---

*Alpha Arena User Manual v1.0*
*For support, file issues at the project repository.*
