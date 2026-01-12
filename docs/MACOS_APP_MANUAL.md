# Alpha Arena macOS Menu Bar App

A native SwiftUI menu bar application for controlling Alpha Arena from your Mac's menu bar.

---

## Overview

The Alpha Arena menu bar app provides a convenient way to:

- Start/stop the trading server with one click
- Monitor real-time P&L and position updates
- Receive native macOS notifications for important events
- Quick access to the web dashboard
- Configure LLM providers and network settings

```
┌─────────────────────────────────────────────────────────────────┐
│                     Swift Menu Bar App                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │   SwiftUI    │  │   AppKit     │  │  UserNotifications   │  │
│  │   MenuBar    │  │  StatusItem  │  │      Framework       │  │
│  └──────┬───────┘  └──────┬───────┘  └──────────┬───────────┘  │
│         │                 │                      │              │
│  ┌──────┴─────────────────┴──────────────────────┴───────────┐  │
│  │                    AlphaArenaService                       │  │
│  │  - Server process control (via ./alpha script)            │  │
│  │  - REST API client (health, trading status)               │  │
│  │  - WebSocket client (real-time P&L updates)               │  │
│  └──────────────────────────┬────────────────────────────────┘  │
└─────────────────────────────┼───────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              │                               │
              ▼                               ▼
    ┌─────────────────┐            ┌─────────────────┐
    │  ./alpha script │            │  FastAPI Server │
    │  (Process Ctrl) │            │  (REST + WS)    │
    └─────────────────┘            └─────────────────┘
```

---

## Requirements

| Requirement | Version |
|-------------|---------|
| macOS | 13.0 (Ventura) or later |
| Xcode | 15.0+ (for building from source) |
| Alpha Arena | Project must be installed |

---

## Installation

### Option 1: Build from Source (Recommended)

```bash
# Navigate to the macOS app directory
cd macos-app

# Build the app
./build.sh

# Build and install to /Applications
./build.sh install
```

### Option 2: Build with Xcode

```bash
# Open project in Xcode
open macos-app/AlphaArena.xcodeproj

# In Xcode:
# 1. Select "AlphaArena" scheme
# 2. Product > Build (⌘B)
# 3. Product > Archive for distribution
```

### First Launch

1. **Locate the app**: After building, find `AlphaArena.app` in:
   - `macos-app/build/Release/AlphaArena.app` (after `./build.sh`)
   - `/Applications/AlphaArena.app` (after `./build.sh install`)

2. **First launch security**: On first launch, macOS may show a security warning:
   - Go to **System Settings > Privacy & Security**
   - Click **"Open Anyway"** next to the Alpha Arena warning
   - Or right-click the app and select **Open**

3. **Grant permissions**: The app will request:
   - **Notifications**: For trade alerts and server status
   - **Network access**: For communicating with the trading server

4. **Set project path**: If the app can't find Alpha Arena automatically:
   - Click the menu bar icon
   - Select **Preferences** (⌘,)
   - Browse to select your Alpha Arena project folder

---

## Menu Bar Interface

### Status Icon

The menu bar icon indicates server status:

| Icon | Status |
|------|--------|
| ● (Green) | Server running, trading active |
| ◐ (Half green) | Server running, trading stopped |
| ○ (Gray) | Server stopped |
| ◉ (Red) | Error state |

### Menu Structure

```
┌────────────────────────────────────────┐
│ Alpha Arena                            │
├────────────────────────────────────────┤
│ ● Server Running                       │
│   http://127.0.0.1:8000                │
├────────────────────────────────────────┤
│ 📊 Dashboard                           │
│   P&L: +$127.45  ▲ 3.2%               │
│   Positions: 5 open                    │
│   Trading: Active                      │
├────────────────────────────────────────┤
│ ▶ Start Server           ⌘S           │
│ ◼ Stop Server            ⌘⇧S          │
│ ─────────────────────────             │
│ 🌐 Open Dashboard        ⌘D           │
│ 📋 Copy Server URL       ⌘C           │
│ 📜 View Logs             ⌘L           │
├────────────────────────────────────────┤
│ Network Mode                        ▶  │
│   ├─ ✓ Localhost Only                 │
│   └─   LAN Access                     │
├────────────────────────────────────────┤
│ ⚙ Preferences...        ⌘,           │
│ ─────────────────────────             │
│ Quit Alpha Arena         ⌘Q           │
└────────────────────────────────────────┘
```

### Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| ⌘S | Start server |
| ⌘⇧S | Stop server |
| ⌘D | Open web dashboard |
| ⌘C | Copy server URL |
| ⌘L | View logs in Terminal |
| ⌘, | Open Preferences |
| ⌘Q | Quit app |

---

## Preferences

Access via menu bar icon > **Preferences** (⌘,)

### General Tab

| Setting | Description | Default |
|---------|-------------|---------|
| **Project Path** | Location of Alpha Arena installation | Auto-detected |
| **Default Port** | Server port (8000, 8080, or 3000) | 8000 |
| **Auto-connect WebSocket** | Connect for real-time updates automatically | On |

### LLM Tab

Manage LLM provider configuration:

- View connected providers and their health status
- Set active provider for trading
- Configure fallback order
- Refresh provider connections

### Network Tab

| Setting | Description |
|---------|-------------|
| **Localhost Only** | Server only accessible from this Mac (127.0.0.1) |
| **LAN Access** | Server accessible from other devices on network (0.0.0.0) |

### Notifications Tab

| Setting | Description | Default |
|---------|-------------|---------|
| **Show Notifications** | Enable/disable all notifications | On |
| **Server Events** | Notify on start/stop | On |
| **Trade Events** | Notify on trades executed | On |
| **P&L Alerts** | Notify on significant P&L changes | On |

---

## Notifications

The app sends native macOS notifications for:

### Server Events
- Server started successfully
- Server stopped
- Server error/crash

### Trading Events
- Trading started/stopped
- Trade executed
- Kill switch activated

### Alerts
- Daily loss limit approached
- Position limit reached
- Connection lost

### Notification Actions

Notifications include action buttons:

| Button | Action |
|--------|--------|
| **Open Dashboard** | Opens web UI in browser |
| **View Logs** | Opens Terminal with live logs |
| **Stop Server** | Stops the trading server |

---

## Real-Time Updates

The app connects via WebSocket for live updates:

- **P&L**: Updates every trade cycle
- **Positions**: Current open position count
- **Trading Status**: Active/stopped state
- **Connection**: Auto-reconnects on disconnect

The dashboard section in the menu shows:
```
P&L: +$127.45  ▲ 3.2%
Positions: 5 open
Trading: Active (simulation)
```

---

## Network Modes

### Localhost Only (Default)

- Server binds to `127.0.0.1`
- Only accessible from this Mac
- Most secure option
- Use for personal trading

### LAN Access

- Server binds to `0.0.0.0`
- Accessible from other devices on your network
- URL shows your local IP (e.g., `http://192.168.1.100:8000`)
- Useful for monitoring from phone/tablet
- **Security note**: Only use on trusted networks

---

## Troubleshooting

### App Won't Launch

**"App is damaged and can't be opened"**
```bash
# Remove quarantine attribute
xattr -cr /Applications/AlphaArena.app
```

**"App from unidentified developer"**
1. System Settings > Privacy & Security
2. Click "Open Anyway"

### Server Won't Start

**"Project path not found"**
1. Open Preferences (⌘,)
2. Click Browse and select the alpha-arena folder
3. Ensure the folder contains the `alpha` script

**"Port already in use"**
```bash
# Find what's using the port
lsof -i :8000

# Kill the process
kill -9 <PID>
```

**"Python not found"**
- Ensure Python 3.11+ is installed
- Run `./scripts/setup.sh` in the project directory

### No Real-Time Updates

**WebSocket not connecting**
1. Check server is running (status icon should be green)
2. Open Preferences > General
3. Enable "Auto-connect WebSocket"
4. Restart the app

**Updates delayed**
- WebSocket may reconnect after network changes
- Wait 5-10 seconds for reconnection

### Notifications Not Working

1. System Settings > Notifications > Alpha Arena
2. Ensure "Allow Notifications" is enabled
3. Check notification style is "Alerts" or "Banners"
4. In app Preferences > Notifications, ensure enabled

### High CPU/Memory Usage

- Check for runaway server process: `./alpha status`
- Restart the server: Stop then Start
- Check logs for errors: `./alpha logs`

---

## Building for Development

### Prerequisites

```bash
# Install Xcode from App Store
# Or install Command Line Tools
xcode-select --install
```

### Build Debug Version

```bash
cd macos-app
xcodebuild -scheme AlphaArena -configuration Debug build
```

### Run Tests

```bash
cd macos-app
xcodebuild -scheme AlphaArena test
```

### Project Structure

```
macos-app/
├── AlphaArena/
│   ├── AlphaArenaApp.swift      # App entry point
│   ├── AppDelegate.swift        # App lifecycle
│   ├── Views/
│   │   ├── MenuBarView.swift    # Main menu content
│   │   ├── StatusItemView.swift # Menu bar icon
│   │   ├── DashboardView.swift  # P&L display
│   │   └── PreferencesView.swift# Settings window
│   ├── Services/
│   │   ├── AlphaArenaService.swift  # Main coordinator
│   │   ├── ProcessController.swift  # ./alpha execution
│   │   ├── APIClient.swift          # REST API
│   │   └── WebSocketClient.swift    # Real-time updates
│   ├── Models/
│   │   ├── ServerState.swift    # Server status
│   │   ├── TradingStatus.swift  # Trading state
│   │   ├── LLMProviderState.swift # LLM config
│   │   └── AppSettings.swift    # User preferences
│   └── Notifications/
│       └── NotificationManager.swift
├── AlphaArenaTests/             # Unit tests
├── AlphaArena.xcodeproj         # Xcode project
├── build.sh                     # Build script
└── Package.swift                # Swift package manifest
```

---

## Uninstalling

### Remove App

```bash
# If installed to Applications
rm -rf /Applications/AlphaArena.app

# Remove preferences
rm ~/Library/Preferences/com.alpha-arena.controller.plist
rm ~/.alpha-arena-path
```

### Keep Server Running

The server runs independently. Uninstalling the menu bar app does not affect:
- The Python trading server
- Your configuration in `.env`
- Trading history and logs

---

## FAQ

**Q: Does the app need to be running for trading to work?**

No. The trading server runs independently. The app is just a convenient control interface. You can also use terminal commands (`./alpha start/stop`).

**Q: Can I run multiple instances?**

No. The app is a singleton. If already running, clicking the app again will focus the existing instance.

**Q: Does the app auto-start with macOS?**

Not by default. To enable:
1. System Settings > General > Login Items
2. Click + and add Alpha Arena

**Q: How do I update the app?**

Pull the latest code and rebuild:
```bash
git pull
cd macos-app
./build.sh install
```

---

## Support

- **Issues**: https://github.com/saymrwulf/alpha-arena/issues
- **Logs**: `./alpha logs` or View Logs from menu
- **Diagnostics**: `./scripts/check.sh`
