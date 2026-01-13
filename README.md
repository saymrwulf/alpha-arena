# Alpha Arena

**Autonomous trading system for Polymarket prediction markets**

Single objective: **maximize PnL**

---

## Start in 10 Seconds

```bash
./alpha start
```

Then open **http://localhost:8000**

That's it. Everything runs in the background with macOS notifications.

---

## Daily Commands

| Command | What it does |
|---------|--------------|
| `./alpha start` | Start server (background, notifies when ready) |
| `./alpha stop` | Stop server (graceful shutdown) |
| `./alpha status` | Check if running |
| `./alpha logs` | Watch live logs |
| `./alpha open` | Open web UI in browser |

See **[QUICKSTART.md](QUICKSTART.md)** for the complete command reference.

---

## What You Get

### Native macOS Menu Bar App
Control Alpha Arena from your menu bar:
- One-click server start/stop
- Real-time P&L updates via WebSocket
- Native notifications for trades and alerts
- Quick access to dashboard and logs
- Network mode toggle (localhost/LAN)

**Requires:** macOS 13.0+, Xcode 15.0+ (from App Store)

```bash
# Build and install
cd macos-app && ./build.sh install
```

See **[docs/MACOS_APP_MANUAL.md](docs/MACOS_APP_MANUAL.md)** for full documentation.

### Web Control Center
Real-time dashboard at localhost:8000 with:
- Live P&L and positions
- Start/stop trading controls
- Market browser and analysis
- Risk management settings
- Decision logs and metrics

### Multi-Agent Trading
Four specialized AI agents collaborating:
- **Research** - Market analysis, probability estimation
- **Risk** - Position sizing with Kelly Criterion
- **Execution** - Optimal order timing
- **Reflection** - Learning from outcomes

### Safety First
- Kill switch for emergency stop
- Daily loss limits
- Position size caps
- Simulation mode for testing

---

## Requirements

- Python 3.11+ (tested with 3.14)
- macOS, Linux, or WSL
- At least one LLM API key (Anthropic, OpenAI, or xAI)

---

## First-Time Setup

The `./alpha start` command handles setup automatically. For manual control:

```bash
# 1. Run setup
./scripts/setup.sh

# 2. Add your API keys
nano .env

# 3. Verify everything works
./scripts/check.sh

# 4. Start
./alpha start
```

### Required API Keys

Add to `.env`:

```bash
# At least one LLM provider
ANTHROPIC_API_KEY=sk-ant-...
# or
OPENAI_API_KEY=sk-...

# For live trading (optional)
POLYMARKET_API_KEY=...
POLYMARKET_API_SECRET=...
POLYMARKET_PASSPHRASE=...
```

---

## Architecture

```
┌──────────────────┐     ┌────────────────────────────────────────┐
│  macOS Menu Bar  │     │       Web Application (FastAPI)        │
│   (SwiftUI App)  │     │     Dashboard | Trading | Markets      │
└────────┬─────────┘     └───────────────────┬────────────────────┘
         │                                   │
         └──────────────┬────────────────────┘
                        │ REST API + WebSocket
                        ▼
          ┌─────────────┼─────────────┐
          ▼             ▼             ▼
    ┌──────────┐  ┌──────────┐  ┌──────────┐
    │ Trading  │  │  Agent   │  │   Risk   │
    │   Loop   │  │ System   │  │ Manager  │
    └──────────┘  └──────────┘  └──────────┘
          │             │
          ▼             ▼
    ┌──────────┐  ┌──────────┐
    │ Broker   │  │   LLM    │
    │Polymarket│  │ Providers│
    └──────────┘  └──────────┘
```

---

## Project Structure

```
alpha-arena/
├── alpha                 # Master control script
├── QUICKSTART.md         # Daily command reference
├── scripts/
│   ├── setup.sh          # First-time setup
│   ├── check.sh          # System diagnostics
│   └── test.sh           # Run tests
├── src/
│   ├── web/              # Web application
│   ├── broker/           # Market execution
│   ├── agents/           # AI agents + debate system
│   ├── signals/          # News, events, sentiment
│   ├── risk/             # Risk management
│   └── llm/              # LLM provider abstraction
├── macos-app/            # Native SwiftUI menu bar app
│   ├── AlphaArena/       # Swift source files
│   └── build.sh          # Build script
├── tests/                # Test suite (200+ tests)
├── docs/                 # Detailed documentation
├── .venv/                # Isolated Python (auto-created)
└── data/                 # Runtime data (auto-created)
```

---

## Documentation

| Document | Purpose |
|----------|---------|
| **[QUICKSTART.md](QUICKSTART.md)** | Daily commands, quick reference |
| **[docs/MACOS_APP_MANUAL.md](docs/MACOS_APP_MANUAL.md)** | Native macOS menu bar app |
| **[docs/WEB_APP_MANUAL.md](docs/WEB_APP_MANUAL.md)** | Web interface guide |
| **[docs/USER_MANUAL.md](docs/USER_MANUAL.md)** | Full CLI and API reference |
| **[docs/TESTING.md](docs/TESTING.md)** | Testing guide |

---

## Troubleshooting

### Server won't start
```bash
./scripts/check.sh --fix
```

### Missing dependencies
```bash
./scripts/setup.sh
```

### API key issues
Check `.env` has valid keys, then restart.

### Something else broke
```bash
./alpha logs    # See what's happening
./alpha status  # Check process status
```

---

## Isolation

Everything runs in complete isolation:

- **`.venv/`** - Python packages (nothing global)
- **`data/`** - All runtime data
- **`.env`** - Your configuration

Uninstall completely by deleting the folder.

---

## License

MIT

---

## Disclaimer

This software is for educational purposes. Trading prediction markets involves financial risk. Past performance does not guarantee future results. Use at your own risk.
