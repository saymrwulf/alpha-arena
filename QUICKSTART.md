# Quick Start Guide

Your daily reference for Alpha Arena commands.

---

## Essential Commands

### Start
```bash
./alpha start
```
Starts the server in the background. You'll get a macOS notification when it's ready.

### Stop
```bash
./alpha stop
```
Graceful shutdown with notification.

### Status
```bash
./alpha status
```
Shows if running, the URL, and health status.

### Logs
```bash
./alpha logs
```
Watch live server logs. Press `Ctrl+C` to exit.

### Open Web UI
```bash
./alpha open
```
Opens http://localhost:8000 in your browser (starts server if needed).

---

## All Commands

| Command | Description |
|---------|-------------|
| `./alpha start` | Start in background, notify when ready |
| `./alpha stop` | Stop gracefully, notify when stopped |
| `./alpha restart` | Stop then start |
| `./alpha status` | Show running status and URL |
| `./alpha logs` | Watch live logs (`Ctrl+C` to exit) |
| `./alpha open` | Open web UI in browser |
| `./alpha test` | Run all tests |
| `./alpha check` | Run system diagnostics |
| `./alpha help` | Show all commands |

---

## Web Interface

Once started, open: **http://localhost:8000**

| Page | What it does |
|------|--------------|
| Dashboard | Live P&L, positions, system status |
| Trading | Start/stop trading, simulation/live modes |
| Positions | View and manage open positions |
| Markets | Browse Polymarket markets |
| Wallet Analysis | Analyze any wallet's strategy |
| Agents | Monitor AI agent status |
| Risk | Configure limits and kill switch |
| Config | All system settings |
| Logs | Decision logs and metrics |

---

## Notifications

You'll get macOS notifications when:
- Server starts successfully
- Server stops
- Errors occur

---

## Environment Variables

Override defaults without editing config:

```bash
# Change port
ALPHA_PORT=9000 ./alpha start

# Change host (for network access)
ALPHA_HOST=0.0.0.0 ./alpha start
```

---

## Menu Bar App (macOS)

For a native macOS experience, use the menu bar app instead of terminal commands.

### Requirements
- macOS 13.0 or later
- Xcode 15.0+ (for building from source)

### Build & Install

```bash
# Build the app
cd macos-app
./build.sh

# Or build and install to /Applications
./build.sh install
```

### Open in Xcode (for development)
```bash
open macos-app/AlphaArena.xcodeproj
```

### Features
- Native SwiftUI menu bar app
- Status icon in menu bar (green=running, gray=stopped)
- Start/Stop server with one click
- Real-time P&L and position updates via WebSocket
- Network mode toggle (localhost vs LAN-accessible)
- Rich notifications with action buttons
- Open Dashboard, View Logs, Copy URL shortcuts
- Preferences window for configuration

### Menu Bar Controls
```
┌────────────────────────────────────────┐
│ ● Server Running                       │
│   http://127.0.0.1:8000                │
├────────────────────────────────────────┤
│ P&L: +$127.45 | 5 positions            │
│ Trading: Active (simulation)           │
├────────────────────────────────────────┤
│ Stop Server                   ⌘⇧S      │
│ Open Dashboard                ⌘D       │
│ Copy Server URL               ⌘C       │
│ View Logs                     ⌘L       │
├────────────────────────────────────────┤
│ Network Mode                        ▶  │
│   ✓ Localhost Only (127.0.0.1)        │
│     LAN Access (0.0.0.0)              │
├────────────────────────────────────────┤
│ Preferences...                ⌘,       │
│ Quit Alpha Arena              ⌘Q       │
└────────────────────────────────────────┘
```

### Keyboard Shortcuts
| Shortcut | Action |
|----------|--------|
| ⌘S | Start Server |
| ⌘⇧S | Stop Server |
| ⌘D | Open Dashboard |
| ⌘C | Copy Server URL |
| ⌘L | View Logs |
| ⌘, | Preferences |
| ⌘Q | Quit |

---

## Auto-Start at Login

To have Alpha Arena start automatically when you log in:

```bash
./scripts/install-autostart.sh
```

To remove auto-start:

```bash
./scripts/install-autostart.sh remove
```

---

## Testing

```bash
# Run all tests
./alpha test

# Or use the test script directly
./scripts/test.sh          # All tests
./scripts/test.sh unit     # Unit tests only
./scripts/test.sh api      # API tests only
./scripts/test.sh e2e      # End-to-end tests
./scripts/test.sh coverage # With coverage report
```

---

## First-Time Setup

If `./alpha start` says setup is needed:

```bash
./scripts/setup.sh
```

Then add your API keys to `.env`:

```bash
ANTHROPIC_API_KEY=sk-ant-...
```

---

## Troubleshooting

### Server won't start
```bash
./scripts/check.sh --fix
```

### Check what's running
```bash
./alpha status
```

### See error details
```bash
./alpha logs
```

### Reset everything
```bash
./alpha stop
rm -rf .venv
./scripts/setup.sh
./alpha start
```

---

## Next Steps

- **[docs/WEB_APP_MANUAL.md](docs/WEB_APP_MANUAL.md)** - Full web interface guide
- **[docs/USER_MANUAL.md](docs/USER_MANUAL.md)** - Complete CLI and API reference
- **[docs/TESTING.md](docs/TESTING.md)** - Testing guide
