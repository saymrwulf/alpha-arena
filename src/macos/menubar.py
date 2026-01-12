#!/usr/bin/env python3
"""
Alpha Arena Menu Bar Application.

A native macOS menu bar app for controlling the Alpha Arena trading server.

Features:
- Status icon (green=running, gray=stopped, red=error)
- Start/Stop server controls
- Network mode toggle (localhost vs LAN)
- Real-time P&L display
- Rich notifications

Usage:
    python -m src.macos.menubar
    # or
    python src/macos/menubar.py
"""

import os
import sys
import time
import threading
import webbrowser
import subprocess
from pathlib import Path
from typing import Optional

import rumps

from .server_bridge import ServerBridge, ServerStatus, WebSocketClient
from .notifications import NotificationManager


# Icon paths (relative to this file)
ICONS_DIR = Path(__file__).parent / "icons"

# Icon filenames
ICON_RUNNING = "icon_running.png"
ICON_STOPPED = "icon_stopped.png"
ICON_ERROR = "icon_error.png"

# Unicode symbols for menu (fallback if icons not found)
SYMBOL_RUNNING = "●"  # Green dot
SYMBOL_STOPPED = "○"  # Empty circle
SYMBOL_ERROR = "◉"    # Red dot

# Polling interval in seconds
POLL_INTERVAL = 4


class AlphaArenaMenuBar(rumps.App):
    """
    Alpha Arena Menu Bar Application.

    Provides native macOS menu bar integration for server control.
    """

    def __init__(self):
        # Initialize with title (shown if no icon)
        super().__init__(
            name="Alpha Arena",
            title=SYMBOL_STOPPED,
            quit_button=None  # We'll add custom quit
        )

        # Find project directory
        self.project_dir = self._find_project_dir()

        # Initialize components
        self.bridge = ServerBridge(project_dir=self.project_dir)
        self.notifications = NotificationManager(
            dashboard_url=self.bridge.base_url,
            log_path=self.bridge.get_log_path()
        )

        # WebSocket client for real-time updates
        self.ws_client: Optional[WebSocketClient] = None

        # State
        self.network_mode = "localhost"
        self._last_pnl = 0.0

        # Build menu
        self._build_menu()

        # Try to set icon
        self._update_icon()

        # Start polling
        self._start_polling()

    def _find_project_dir(self) -> Path:
        """Find the Alpha Arena project directory."""
        # Try relative to this file (src/macos/menubar.py -> project root)
        current = Path(__file__).resolve()
        candidate = current.parent.parent.parent
        if (candidate / "alpha").exists():
            return candidate

        # Try environment variable
        if "ALPHA_ARENA_DIR" in os.environ:
            path = Path(os.environ["ALPHA_ARENA_DIR"])
            if (path / "alpha").exists():
                return path

        # Try common locations
        home = Path.home()
        for path in [
            home / "GitClone/ClaudeCodeProjects/alpha-arena",
            home / "Projects/alpha-arena",
            home / "alpha-arena",
        ]:
            if (path / "alpha").exists():
                return path

        # Fall back to current directory
        return Path.cwd()

    def _build_menu(self):
        """Build the menu structure."""
        # Status header (non-clickable)
        self.status_item = rumps.MenuItem("Status: Checking...", callback=None)

        # URL display
        self.url_item = rumps.MenuItem("", callback=None)

        # P&L display
        self.pnl_item = rumps.MenuItem("", callback=None)

        # Start/Stop button
        self.toggle_item = rumps.MenuItem("Start Server", callback=self.toggle_server)

        # Open Dashboard
        self.dashboard_item = rumps.MenuItem("Open Dashboard", callback=self.open_dashboard)

        # Network mode submenu
        self.localhost_item = rumps.MenuItem(
            "Localhost Only (127.0.0.1)",
            callback=lambda _: self.set_network_mode("localhost")
        )
        self.lan_item = rumps.MenuItem(
            "LAN Accessible (0.0.0.0)",
            callback=lambda _: self.set_network_mode("lan")
        )
        self.localhost_item.state = 1  # Checked by default

        network_menu = rumps.MenuItem("Network Mode")
        network_menu.add(self.localhost_item)
        network_menu.add(self.lan_item)

        # Utilities
        self.copy_url_item = rumps.MenuItem("Copy URL", callback=self.copy_url)
        self.view_logs_item = rumps.MenuItem("View Logs", callback=self.view_logs)

        # Quit
        self.quit_item = rumps.MenuItem("Quit Alpha Arena", callback=self.quit_app)

        # Build menu
        self.menu = [
            self.status_item,
            self.url_item,
            self.pnl_item,
            None,  # Separator
            self.toggle_item,
            self.dashboard_item,
            None,  # Separator
            network_menu,
            None,  # Separator
            self.copy_url_item,
            self.view_logs_item,
            None,  # Separator
            self.quit_item,
        ]

    def _get_icon_path(self, icon_name: str) -> Optional[str]:
        """Get full path to icon if it exists."""
        path = ICONS_DIR / icon_name
        if path.exists():
            return str(path)
        return None

    def _update_icon(self):
        """Update menu bar icon based on status."""
        status = self.bridge.state.status

        if status == ServerStatus.RUNNING:
            icon = self._get_icon_path(ICON_RUNNING)
            symbol = SYMBOL_RUNNING
        elif status == ServerStatus.ERROR:
            icon = self._get_icon_path(ICON_ERROR)
            symbol = SYMBOL_ERROR
        else:
            icon = self._get_icon_path(ICON_STOPPED)
            symbol = SYMBOL_STOPPED

        if icon:
            self.icon = icon
            self.title = None
        else:
            self.icon = None
            self.title = f"{symbol} Alpha"

    def _update_menu(self):
        """Update menu items based on current state."""
        state = self.bridge.state

        # Status text
        if state.status == ServerStatus.RUNNING:
            self.status_item.title = "Server Running"
            self.toggle_item.title = "Stop Server"
            self.url_item.title = f"  {state.url}"
            self.dashboard_item.set_callback(self.open_dashboard)
        elif state.status == ServerStatus.STARTING:
            self.status_item.title = "Starting..."
            self.toggle_item.title = "Starting..."
            self.url_item.title = ""
        elif state.status == ServerStatus.STOPPING:
            self.status_item.title = "Stopping..."
            self.toggle_item.title = "Stopping..."
        elif state.status == ServerStatus.ERROR:
            self.status_item.title = f"Error: {state.error_message[:30]}"
            self.toggle_item.title = "Start Server"
            self.url_item.title = ""
        else:
            self.status_item.title = "Server Stopped"
            self.toggle_item.title = "Start Server"
            self.url_item.title = ""

        # P&L display
        if state.status == ServerStatus.RUNNING and state.trading_active:
            pnl = state.daily_pnl
            pnl_str = f"+${pnl:.2f}" if pnl >= 0 else f"-${abs(pnl):.2f}"
            self.pnl_item.title = f"  P&L: {pnl_str} | {state.positions_count} pos"
        elif state.status == ServerStatus.RUNNING:
            self.pnl_item.title = f"  Mode: {state.trading_mode}"
        else:
            self.pnl_item.title = ""

        # Update icon
        self._update_icon()

    def _start_polling(self):
        """Start background health polling."""
        def poll_loop():
            while True:
                try:
                    self.bridge.refresh_state()
                    # Update UI on main thread
                    rumps.Timer(0, lambda _: self._update_menu()).start()
                except Exception as e:
                    print(f"Poll error: {e}")
                time.sleep(POLL_INTERVAL)

        thread = threading.Thread(target=poll_loop, daemon=True)
        thread.start()

    def _start_websocket(self):
        """Start WebSocket connection for real-time updates."""
        if self.ws_client:
            self.ws_client.stop()

        self.ws_client = WebSocketClient(
            bridge=self.bridge,
            on_message=self._handle_ws_message
        )
        self.ws_client.start()

    def _stop_websocket(self):
        """Stop WebSocket connection."""
        if self.ws_client:
            self.ws_client.stop()
            self.ws_client = None

    def _handle_ws_message(self, data: dict):
        """Handle incoming WebSocket message."""
        msg_type = data.get("type")

        if msg_type == "cycle_complete":
            cycle_data = data.get("data", {})
            pnl = cycle_data.get("daily_pnl", 0)

            # Update state
            self.bridge.state.daily_pnl = float(pnl)
            self.bridge.state.cycle_count = cycle_data.get("cycle_count", 0)

            # Notify on significant P&L change
            if abs(pnl - self._last_pnl) >= 5.0:
                self.notifications.notify_summary(
                    pnl=pnl,
                    positions=self.bridge.state.positions_count
                )
                self._last_pnl = pnl

        elif msg_type == "trading_started":
            mode = data.get("data", {}).get("mode", "simulation")
            self.notifications.notify_trading_started(mode)

        elif msg_type == "trading_stopped":
            self.notifications.notify_trading_stopped()

        elif msg_type == "kill_switch":
            activated = data.get("data", {}).get("enabled", False)
            self.notifications.notify_kill_switch(activated)

        # Trigger menu update
        rumps.Timer(0, lambda _: self._update_menu()).start()

    # Menu callbacks

    def toggle_server(self, sender):
        """Start or stop the server."""
        if self.bridge.state.status == ServerStatus.RUNNING:
            self._stop_server()
        else:
            self._start_server()

    def _start_server(self):
        """Start the server."""
        def start_thread():
            lan_mode = self.network_mode == "lan"
            success, message = self.bridge.start_server(lan_mode=lan_mode)

            if success:
                self.notifications.notify_server_started(self.bridge.base_url)
                self._start_websocket()
            else:
                self.notifications.notify_server_error(message)

            # Update UI
            rumps.Timer(0, lambda _: self._update_menu()).start()

        threading.Thread(target=start_thread, daemon=True).start()
        self._update_menu()

    def _stop_server(self):
        """Stop the server."""
        def stop_thread():
            self._stop_websocket()
            success, message = self.bridge.stop_server()

            if success:
                self.notifications.notify_server_stopped()
            else:
                self.notifications.notify_server_error(message)

            # Update UI
            rumps.Timer(0, lambda _: self._update_menu()).start()

        threading.Thread(target=stop_thread, daemon=True).start()
        self._update_menu()

    def open_dashboard(self, sender):
        """Open the web dashboard in browser."""
        if self.bridge.state.status == ServerStatus.RUNNING:
            webbrowser.open(self.bridge.base_url)
        else:
            rumps.notification(
                "Alpha Arena",
                "Server not running",
                "Start the server first to open the dashboard."
            )

    def set_network_mode(self, mode: str):
        """Set network mode (localhost or lan)."""
        self.network_mode = mode

        # Update checkmarks
        self.localhost_item.state = 1 if mode == "localhost" else 0
        self.lan_item.state = 1 if mode == "lan" else 0

        # Update bridge
        if mode == "lan":
            self.bridge.host = "0.0.0.0"
        else:
            self.bridge.host = "127.0.0.1"

        # Notify if server is running (will need restart)
        if self.bridge.state.status == ServerStatus.RUNNING:
            rumps.notification(
                "Alpha Arena",
                "Network Mode Changed",
                "Restart the server for this change to take effect."
            )

    def copy_url(self, sender):
        """Copy server URL to clipboard."""
        url = self.bridge.base_url
        subprocess.run(
            ["pbcopy"],
            input=url.encode(),
            check=True
        )
        rumps.notification(
            "Alpha Arena",
            "URL Copied",
            f"Copied {url} to clipboard"
        )

    def view_logs(self, sender):
        """Open log file in Console.app or Terminal."""
        log_path = self.bridge.get_log_path()

        if log_path.exists():
            # Try Console.app first
            result = subprocess.run(
                ["open", "-a", "Console", str(log_path)],
                capture_output=True
            )
            if result.returncode != 0:
                # Fall back to Terminal
                subprocess.run(["open", "-a", "Terminal", str(log_path)])
        else:
            rumps.notification(
                "Alpha Arena",
                "No Logs",
                "No log file found. Start the server first."
            )

    def quit_app(self, sender):
        """Quit the menu bar app."""
        # Stop WebSocket
        self._stop_websocket()

        # Ask about server if running
        if self.bridge.state.status == ServerStatus.RUNNING:
            response = rumps.alert(
                title="Quit Alpha Arena",
                message="The server is still running. Do you want to stop it?",
                ok="Stop & Quit",
                cancel="Quit (Keep Running)"
            )
            if response == 1:  # OK clicked
                self.bridge.stop_server()

        rumps.quit_application()


def main():
    """Main entry point."""
    app = AlphaArenaMenuBar()
    app.run()


if __name__ == "__main__":
    # Add parent directory to path for imports
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    main()
