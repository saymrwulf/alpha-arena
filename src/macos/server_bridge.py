"""
Server Bridge - Communication layer between menu bar app and Alpha Arena server.

Handles:
- Process control via ./alpha script
- HTTP API communication
- WebSocket connection for real-time updates
"""

import os
import subprocess
import threading
import json
from pathlib import Path
from typing import Optional, Callable
from dataclasses import dataclass, field
from enum import Enum

import httpx


class ServerStatus(Enum):
    """Server status states."""
    RUNNING = "running"
    STOPPED = "stopped"
    ERROR = "error"
    STARTING = "starting"
    STOPPING = "stopping"


@dataclass
class ServerState:
    """Current server state."""
    status: ServerStatus = ServerStatus.STOPPED
    trading_active: bool = False
    trading_mode: str = "simulation"
    balance: float = 0.0
    daily_pnl: float = 0.0
    total_pnl: float = 0.0
    positions_count: int = 0
    cycle_count: int = 0
    websocket_clients: int = 0
    uptime: str = ""
    url: str = ""
    network_mode: str = "localhost"
    error_message: str = ""


class ServerBridge:
    """
    Bridge between menu bar app and Alpha Arena server.

    Manages process control and API communication.
    """

    def __init__(
        self,
        project_dir: Optional[Path] = None,
        host: str = "127.0.0.1",
        port: int = 8000
    ):
        self.project_dir = project_dir or self._find_project_dir()
        self.host = host
        self.port = port
        self.alpha_script = self.project_dir / "alpha"
        self._state = ServerState()
        self._callbacks: list[Callable[[ServerState], None]] = []

    def _find_project_dir(self) -> Path:
        """Find the Alpha Arena project directory."""
        # Try relative to this file
        current = Path(__file__).resolve()

        # Go up from src/macos/server_bridge.py to project root
        candidate = current.parent.parent.parent
        if (candidate / "alpha").exists():
            return candidate

        # Try common locations
        home = Path.home()
        common_paths = [
            home / "GitClone/ClaudeCodeProjects/alpha-arena",
            home / "Projects/alpha-arena",
            home / "alpha-arena",
            Path("/opt/alpha-arena"),
        ]

        for path in common_paths:
            if (path / "alpha").exists():
                return path

        # Fall back to current directory
        return Path.cwd()

    @property
    def base_url(self) -> str:
        """Get the base URL for API calls."""
        return f"http://{self.host}:{self.port}"

    @property
    def state(self) -> ServerState:
        """Get current server state."""
        return self._state

    def add_state_callback(self, callback: Callable[[ServerState], None]):
        """Add callback to be notified of state changes."""
        self._callbacks.append(callback)

    def _notify_callbacks(self):
        """Notify all callbacks of state change."""
        for callback in self._callbacks:
            try:
                callback(self._state)
            except Exception as e:
                print(f"Callback error: {e}")

    def start_server(self, lan_mode: bool = False) -> tuple[bool, str]:
        """
        Start the server using ./alpha start.

        Args:
            lan_mode: If True, bind to 0.0.0.0 for LAN access

        Returns:
            Tuple of (success, message)
        """
        self._state.status = ServerStatus.STARTING
        self._state.network_mode = "lan" if lan_mode else "localhost"
        self._notify_callbacks()

        env = os.environ.copy()
        env["ALPHA_HOST"] = "0.0.0.0" if lan_mode else "127.0.0.1"
        env["ALPHA_PORT"] = str(self.port)

        try:
            result = subprocess.run(
                [str(self.alpha_script), "start"],
                env=env,
                capture_output=True,
                text=True,
                cwd=str(self.project_dir),
                timeout=60
            )

            if result.returncode == 0:
                self._state.status = ServerStatus.RUNNING
                self._state.url = f"http://{env['ALPHA_HOST']}:{self.port}"
                self._state.error_message = ""
                self._notify_callbacks()
                return True, result.stdout
            else:
                self._state.status = ServerStatus.ERROR
                self._state.error_message = result.stderr or result.stdout
                self._notify_callbacks()
                return False, result.stderr or result.stdout

        except subprocess.TimeoutExpired:
            self._state.status = ServerStatus.ERROR
            self._state.error_message = "Start timeout"
            self._notify_callbacks()
            return False, "Server start timed out"
        except Exception as e:
            self._state.status = ServerStatus.ERROR
            self._state.error_message = str(e)
            self._notify_callbacks()
            return False, str(e)

    def stop_server(self) -> tuple[bool, str]:
        """
        Stop the server using ./alpha stop.

        Returns:
            Tuple of (success, message)
        """
        self._state.status = ServerStatus.STOPPING
        self._notify_callbacks()

        try:
            result = subprocess.run(
                [str(self.alpha_script), "stop"],
                capture_output=True,
                text=True,
                cwd=str(self.project_dir),
                timeout=30
            )

            self._state.status = ServerStatus.STOPPED
            self._state.trading_active = False
            self._state.error_message = ""
            self._notify_callbacks()
            return result.returncode == 0, result.stdout

        except subprocess.TimeoutExpired:
            self._state.status = ServerStatus.ERROR
            self._state.error_message = "Stop timeout"
            self._notify_callbacks()
            return False, "Server stop timed out"
        except Exception as e:
            self._state.status = ServerStatus.ERROR
            self._state.error_message = str(e)
            self._notify_callbacks()
            return False, str(e)

    def check_health(self) -> bool:
        """
        Check server health via API.

        Returns:
            True if server is healthy
        """
        try:
            with httpx.Client(timeout=2.0) as client:
                response = client.get(f"{self.base_url}/api/system/health")

                if response.status_code == 200:
                    data = response.json()
                    self._state.status = ServerStatus.RUNNING
                    self._state.websocket_clients = data.get("websocket_clients", 0)
                    self._state.uptime = data.get("uptime", "")
                    self._state.trading_active = data.get("trading_active", False)
                    self._state.error_message = ""
                    return True
                else:
                    self._state.status = ServerStatus.ERROR
                    self._state.error_message = f"Health check returned {response.status_code}"
                    return False

        except httpx.ConnectError:
            self._state.status = ServerStatus.STOPPED
            self._state.error_message = ""
            return False
        except Exception as e:
            self._state.status = ServerStatus.ERROR
            self._state.error_message = str(e)
            return False

    def get_trading_status(self) -> Optional[dict]:
        """
        Get detailed trading status from API.

        Returns:
            Trading status dict or None if unavailable
        """
        try:
            with httpx.Client(timeout=2.0) as client:
                response = client.get(f"{self.base_url}/api/trading/status")

                if response.status_code == 200:
                    data = response.json()
                    self._state.trading_active = data.get("active", False)
                    self._state.trading_mode = data.get("mode", "simulation")
                    self._state.balance = float(data.get("balance", 0))
                    self._state.daily_pnl = float(data.get("daily_pnl", 0))
                    self._state.total_pnl = float(data.get("total_pnl", 0))
                    self._state.positions_count = len(data.get("positions", []))
                    self._state.cycle_count = data.get("cycle_count", 0)
                    return data

        except Exception:
            pass
        return None

    def refresh_state(self) -> ServerState:
        """
        Refresh full server state.

        Returns:
            Updated ServerState
        """
        if self.check_health():
            self.get_trading_status()
        self._notify_callbacks()
        return self._state

    def get_log_path(self) -> Path:
        """Get path to server log file."""
        return self.project_dir / "data" / "logs" / "server.log"

    def get_pid_file(self) -> Path:
        """Get path to PID file."""
        return self.project_dir / ".alpha.pid"

    def is_process_running(self) -> bool:
        """Check if server process is running via PID file."""
        pid_file = self.get_pid_file()
        if not pid_file.exists():
            return False

        try:
            pid = int(pid_file.read_text().strip())
            # Check if process exists
            os.kill(pid, 0)
            return True
        except (ValueError, OSError, ProcessLookupError):
            return False


class WebSocketClient:
    """
    WebSocket client for real-time server updates.
    """

    def __init__(
        self,
        bridge: ServerBridge,
        on_message: Optional[Callable[[dict], None]] = None
    ):
        self.bridge = bridge
        self.on_message = on_message
        self._ws = None
        self._thread: Optional[threading.Thread] = None
        self._running = False

    def start(self):
        """Start WebSocket connection in background thread."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop WebSocket connection."""
        self._running = False
        if self._ws:
            try:
                self._ws.close()
            except Exception:
                pass

    def _run_loop(self):
        """WebSocket connection loop with reconnection."""
        import websocket

        ws_url = self.bridge.base_url.replace("http", "ws") + "/ws"

        while self._running:
            try:
                self._ws = websocket.WebSocketApp(
                    ws_url,
                    on_message=self._on_message,
                    on_error=self._on_error,
                    on_close=self._on_close,
                    on_open=self._on_open
                )
                self._ws.run_forever(ping_interval=30, ping_timeout=10)
            except Exception as e:
                print(f"WebSocket error: {e}")

            # Wait before reconnecting
            if self._running:
                import time
                time.sleep(3)

    def _on_message(self, ws, message: str):
        """Handle incoming WebSocket message."""
        try:
            data = json.loads(message)
            if self.on_message:
                self.on_message(data)
        except json.JSONDecodeError:
            pass

    def _on_error(self, ws, error):
        """Handle WebSocket error."""
        print(f"WebSocket error: {error}")

    def _on_close(self, ws, close_status_code, close_msg):
        """Handle WebSocket close."""
        print(f"WebSocket closed: {close_status_code} {close_msg}")

    def _on_open(self, ws):
        """Handle WebSocket open."""
        print("WebSocket connected")
