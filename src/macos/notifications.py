"""
Rich macOS Notifications using UserNotifications framework.

Provides:
- Notifications with action buttons (Open Dashboard, View Logs)
- Notification categories (server, trading, summary)
- Sound options
- Fallback to osascript for simpler notifications
"""

import subprocess
import webbrowser
from typing import Optional, Callable
from pathlib import Path


# Try to import pyobjc for rich notifications
try:
    import objc
    from Foundation import NSObject
    from UserNotifications import (
        UNUserNotificationCenter,
        UNMutableNotificationContent,
        UNNotificationRequest,
        UNNotificationAction,
        UNNotificationCategory,
        UNNotificationCategoryOptionNone,
        UNNotificationActionOptionForeground,
        UNNotificationSound,
        UNAuthorizationOptionAlert,
        UNAuthorizationOptionSound,
    )
    PYOBJC_AVAILABLE = True
except ImportError:
    PYOBJC_AVAILABLE = False


# Action identifiers
ACTION_OPEN_DASHBOARD = "OPEN_DASHBOARD"
ACTION_VIEW_LOGS = "VIEW_LOGS"
ACTION_DISMISS = "DISMISS"

# Category identifiers
CATEGORY_SERVER = "SERVER_STATUS"
CATEGORY_TRADING = "TRADING_ALERT"
CATEGORY_SUMMARY = "PERIODIC_SUMMARY"


class NotificationDelegate(NSObject if PYOBJC_AVAILABLE else object):
    """
    Handles notification action responses.
    """

    def init(self):
        if not PYOBJC_AVAILABLE:
            return self
        self = objc.super(NotificationDelegate, self).init()
        if self is None:
            return None
        self.dashboard_url = "http://127.0.0.1:8000"
        self.log_path = None
        return self

    if PYOBJC_AVAILABLE:
        def userNotificationCenter_didReceiveNotificationResponse_withCompletionHandler_(
            self, center, response, handler
        ):
            """Handle notification action response."""
            action_id = response.actionIdentifier()

            if action_id == ACTION_OPEN_DASHBOARD:
                webbrowser.open(self.dashboard_url)
            elif action_id == ACTION_VIEW_LOGS:
                if self.log_path and Path(self.log_path).exists():
                    subprocess.run(["open", "-a", "Console", str(self.log_path)])
                else:
                    # Open Terminal with tail
                    subprocess.run(["open", "-a", "Terminal"])

            handler()

        def userNotificationCenter_willPresentNotification_withCompletionHandler_(
            self, center, notification, handler
        ):
            """Allow notifications to show even when app is in foreground."""
            from UserNotifications import (
                UNNotificationPresentationOptionBanner,
                UNNotificationPresentationOptionSound,
            )
            handler(UNNotificationPresentationOptionBanner | UNNotificationPresentationOptionSound)


class NotificationManager:
    """
    Manages macOS notifications with rich actions.

    Falls back to simple osascript notifications if pyobjc is unavailable.
    """

    def __init__(self, dashboard_url: str = "http://127.0.0.1:8000", log_path: Optional[Path] = None):
        self.dashboard_url = dashboard_url
        self.log_path = log_path
        self._delegate = None
        self._center = None
        self._authorized = False

        if PYOBJC_AVAILABLE:
            self._setup_rich_notifications()

    def _setup_rich_notifications(self):
        """Set up UserNotifications framework."""
        try:
            self._center = UNUserNotificationCenter.currentNotificationCenter()
            self._delegate = NotificationDelegate.alloc().init()
            self._delegate.dashboard_url = self.dashboard_url
            self._delegate.log_path = str(self.log_path) if self.log_path else None
            self._center.setDelegate_(self._delegate)
            self._request_authorization()
            self._register_categories()
        except Exception as e:
            print(f"Failed to set up rich notifications: {e}")
            self._center = None

    def _request_authorization(self):
        """Request notification permissions."""
        def completion(granted, error):
            self._authorized = granted
            if error:
                print(f"Notification auth error: {error}")

        self._center.requestAuthorizationWithOptions_completionHandler_(
            UNAuthorizationOptionAlert | UNAuthorizationOptionSound,
            completion
        )

    def _register_categories(self):
        """Register notification categories with actions."""
        # Server status actions
        open_dashboard = UNNotificationAction.actionWithIdentifier_title_options_(
            ACTION_OPEN_DASHBOARD,
            "Open Dashboard",
            UNNotificationActionOptionForeground
        )
        view_logs = UNNotificationAction.actionWithIdentifier_title_options_(
            ACTION_VIEW_LOGS,
            "View Logs",
            UNNotificationActionOptionForeground
        )

        # Server category
        server_category = UNNotificationCategory.categoryWithIdentifier_actions_intentIdentifiers_options_(
            CATEGORY_SERVER,
            [open_dashboard, view_logs],
            [],
            UNNotificationCategoryOptionNone
        )

        # Trading category
        trading_category = UNNotificationCategory.categoryWithIdentifier_actions_intentIdentifiers_options_(
            CATEGORY_TRADING,
            [open_dashboard],
            [],
            UNNotificationCategoryOptionNone
        )

        # Summary category
        summary_category = UNNotificationCategory.categoryWithIdentifier_actions_intentIdentifiers_options_(
            CATEGORY_SUMMARY,
            [open_dashboard],
            [],
            UNNotificationCategoryOptionNone
        )

        self._center.setNotificationCategories_({
            server_category,
            trading_category,
            summary_category
        })

    def _send_rich_notification(
        self,
        identifier: str,
        title: str,
        body: str,
        category: str = CATEGORY_SERVER,
        sound: str = "default"
    ):
        """Send a rich notification using UserNotifications."""
        content = UNMutableNotificationContent.alloc().init()
        content.setTitle_(title)
        content.setBody_(body)
        content.setCategoryIdentifier_(category)

        # Set sound
        if sound == "default":
            content.setSound_(UNNotificationSound.defaultSound())
        elif sound:
            content.setSound_(UNNotificationSound.soundNamed_(sound))

        request = UNNotificationRequest.requestWithIdentifier_content_trigger_(
            identifier,
            content,
            None  # Immediate delivery
        )

        def completion(error):
            if error:
                print(f"Notification error: {error}")

        self._center.addNotificationRequest_withCompletionHandler_(request, completion)

    def _send_simple_notification(self, title: str, message: str, sound: str = "default"):
        """Send a simple notification using osascript (fallback)."""
        script = f'display notification "{message}" with title "{title}"'
        if sound:
            script += f' sound name "{sound}"'

        try:
            subprocess.run(
                ["osascript", "-e", script],
                capture_output=True,
                timeout=5
            )
        except Exception as e:
            print(f"osascript notification failed: {e}")

    def notify(
        self,
        title: str,
        message: str,
        category: str = CATEGORY_SERVER,
        sound: str = "default",
        identifier: Optional[str] = None
    ):
        """
        Send a notification.

        Uses rich notifications if available, otherwise falls back to osascript.
        """
        if self._center and PYOBJC_AVAILABLE:
            self._send_rich_notification(
                identifier or f"notification_{id(message)}",
                title,
                message,
                category,
                sound
            )
        else:
            self._send_simple_notification(title, message, sound)

    # Convenience methods

    def notify_server_started(self, url: str):
        """Notify that server has started."""
        self.notify(
            "Alpha Arena Started",
            f"Server running at {url}",
            category=CATEGORY_SERVER,
            sound="Glass",
            identifier="server_started"
        )

    def notify_server_stopped(self):
        """Notify that server has stopped."""
        self.notify(
            "Alpha Arena Stopped",
            "Trading server has been stopped",
            category=CATEGORY_SERVER,
            sound="Pop",
            identifier="server_stopped"
        )

    def notify_server_error(self, error: str):
        """Notify of server error."""
        self.notify(
            "Alpha Arena Error",
            error[:100],  # Truncate long errors
            category=CATEGORY_SERVER,
            sound="Basso",
            identifier="server_error"
        )

    def notify_trading_started(self, mode: str):
        """Notify that trading has started."""
        self.notify(
            "Trading Started",
            f"Running in {mode} mode",
            category=CATEGORY_TRADING,
            sound="Pop",
            identifier="trading_started"
        )

    def notify_trading_stopped(self):
        """Notify that trading has stopped."""
        self.notify(
            "Trading Stopped",
            "Trading loop has been stopped",
            category=CATEGORY_TRADING,
            sound="Pop",
            identifier="trading_stopped"
        )

    def notify_summary(self, pnl: float, positions: int, trades: int = 0):
        """Send periodic trading summary."""
        pnl_str = f"+${pnl:.2f}" if pnl >= 0 else f"-${abs(pnl):.2f}"
        self.notify(
            "Trading Summary",
            f"P&L: {pnl_str} | {positions} positions | {trades} trades",
            category=CATEGORY_SUMMARY,
            sound="",  # No sound for summaries
            identifier="summary"
        )

    def notify_alert(self, title: str, message: str, critical: bool = False):
        """Send trading alert."""
        self.notify(
            title,
            message,
            category=CATEGORY_TRADING,
            sound="Basso" if critical else "Pop",
            identifier=f"alert_{id(message)}"
        )

    def notify_kill_switch(self, activated: bool):
        """Notify kill switch state change."""
        if activated:
            self.notify(
                "Kill Switch ACTIVATED",
                "All trading has been halted",
                category=CATEGORY_TRADING,
                sound="Basso",
                identifier="kill_switch"
            )
        else:
            self.notify(
                "Kill Switch Deactivated",
                "Trading can resume",
                category=CATEGORY_TRADING,
                sound="Pop",
                identifier="kill_switch"
            )

    def notify_loss_limit(self, current_loss: float, limit: float):
        """Notify that daily loss limit was hit."""
        self.notify(
            "Daily Loss Limit Hit",
            f"Loss: ${abs(current_loss):.2f} / Limit: ${limit:.2f}",
            category=CATEGORY_TRADING,
            sound="Basso",
            identifier="loss_limit"
        )
