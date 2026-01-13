# Alpha Arena Documentation

Welcome to Alpha Arena. This guide helps you find the right documentation for your needs.

---

## Quick Navigation

### Just want to run it?
**[../QUICKSTART.md](../QUICKSTART.md)** - Daily commands, start/stop, basic troubleshooting

### Using the web interface?
**[WEB_APP_MANUAL.md](WEB_APP_MANUAL.md)** - Complete web UI guide with screenshots and workflows

### Using the macOS menu bar app?
**[MACOS_APP_MANUAL.md](MACOS_APP_MANUAL.md)** - Native SwiftUI app for menu bar control

### Need the full reference?
**[USER_MANUAL.md](USER_MANUAL.md)** - CLI commands, multi-agent system, strategies, API reference

### Running in production?
**[OPERATIONAL_MANUAL.md](OPERATIONAL_MANUAL.md)** - Deployment, monitoring, backup, security

### Writing or running tests?
**[TESTING.md](TESTING.md)** - Test structure, fixtures, writing tests

---

## Documentation Overview

| Document | Audience | Content |
|----------|----------|---------|
| [QUICKSTART.md](../QUICKSTART.md) | Everyone | 5-minute guide, daily commands |
| [WEB_APP_MANUAL.md](WEB_APP_MANUAL.md) | Operators | Web interface, dashboards, controls |
| [MACOS_APP_MANUAL.md](MACOS_APP_MANUAL.md) | macOS users | Native menu bar app |
| [USER_MANUAL.md](USER_MANUAL.md) | Developers | Full CLI, agents, strategies, API |
| [OPERATIONAL_MANUAL.md](OPERATIONAL_MANUAL.md) | DevOps | Deployment, monitoring, security |
| [TESTING.md](TESTING.md) | Developers | Test suite, fixtures, patterns |

---

## Learning Path

### Beginner
1. Read [../QUICKSTART.md](../QUICKSTART.md) - Get running
2. Explore [WEB_APP_MANUAL.md](WEB_APP_MANUAL.md) - Use the web UI
3. Or try [MACOS_APP_MANUAL.md](MACOS_APP_MANUAL.md) - Native menu bar app
4. Try simulation mode before live trading

### Intermediate
1. Study [USER_MANUAL.md](USER_MANUAL.md) Sections 6-10 - Agents, strategies, risk
2. Learn technical indicators (Section 11)
3. Understand memory system (Section 14)

### Advanced
1. Review [OPERATIONAL_MANUAL.md](OPERATIONAL_MANUAL.md) - Production deployment
2. Customize strategies via config
3. Integrate with other systems via API

---

## Quick Reference

### Start the System
```bash
./alpha start        # Start server
./alpha open         # Open web UI
```

### Stop the System
```bash
./alpha stop         # Graceful shutdown
```

### Check Status
```bash
./alpha status       # Is it running?
./alpha logs         # What's happening?
```

### Run Tests
```bash
./alpha test         # All tests
./scripts/test.sh    # More options
```

---

## Key Concepts

| Concept | What it means |
|---------|---------------|
| **Multi-Agent** | Four AI agents (Research, Risk, Execution, Reflection) collaborate |
| **Kelly Criterion** | Mathematical formula for optimal position sizing |
| **Kill Switch** | Emergency stop all trading |
| **Simulation Mode** | Paper trading, no real money |
| **Edge** | Your expected advantage over market price |

---

## Getting Help

1. Check the troubleshooting section in [../QUICKSTART.md](../QUICKSTART.md)
2. Review logs: `./alpha logs`
3. Run diagnostics: `./scripts/check.sh`
4. Search the relevant manual for your issue
