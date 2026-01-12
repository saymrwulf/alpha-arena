import Foundation
import UserNotifications
import AppKit

/// Notification categories
enum NotificationCategory: String {
    case serverStatus = "SERVER_STATUS"
    case tradingAlert = "TRADING_ALERT"
    case criticalAlert = "CRITICAL_ALERT"
}

/// Notification actions
enum NotificationAction: String {
    case openDashboard = "OPEN_DASHBOARD"
    case viewLogs = "VIEW_LOGS"
    case dismiss = "DISMISS"
}

/// Manages macOS notifications for Alpha Arena
class NotificationManager: NSObject, UNUserNotificationCenterDelegate {
    private var isAuthorized = false
    private let center = UNUserNotificationCenter.current()

    override init() {
        super.init()
        center.delegate = self
        setupCategories()
        requestPermission()
    }

    // MARK: - Permission

    /// Request notification permission
    func requestPermission() {
        center.requestAuthorization(options: [.alert, .sound, .badge]) { [weak self] granted, error in
            DispatchQueue.main.async {
                self?.isAuthorized = granted
                if let error = error {
                    print("Notification permission error: \(error)")
                }
            }
        }
    }

    // MARK: - Categories Setup

    private func setupCategories() {
        // Server status actions
        let openDashboard = UNNotificationAction(
            identifier: NotificationAction.openDashboard.rawValue,
            title: "Open Dashboard",
            options: .foreground
        )

        let viewLogs = UNNotificationAction(
            identifier: NotificationAction.viewLogs.rawValue,
            title: "View Logs",
            options: .foreground
        )

        let dismiss = UNNotificationAction(
            identifier: NotificationAction.dismiss.rawValue,
            title: "Dismiss",
            options: .destructive
        )

        // Server status category
        let serverCategory = UNNotificationCategory(
            identifier: NotificationCategory.serverStatus.rawValue,
            actions: [openDashboard, viewLogs, dismiss],
            intentIdentifiers: [],
            options: .customDismissAction
        )

        // Trading alert category
        let tradingCategory = UNNotificationCategory(
            identifier: NotificationCategory.tradingAlert.rawValue,
            actions: [openDashboard, dismiss],
            intentIdentifiers: [],
            options: .customDismissAction
        )

        // Critical alert category
        let criticalCategory = UNNotificationCategory(
            identifier: NotificationCategory.criticalAlert.rawValue,
            actions: [openDashboard, viewLogs],
            intentIdentifiers: [],
            options: [.customDismissAction]
        )

        center.setNotificationCategories([serverCategory, tradingCategory, criticalCategory])
    }

    // MARK: - Send Notifications

    /// Send server started notification
    func sendServerStartedNotification(url: String) {
        guard AppSettings.shared.showNotifications else { return }

        send(
            title: "Alpha Arena Started",
            body: "Server is running at \(url)",
            category: .serverStatus,
            userInfo: ["url": url]
        )
    }

    /// Send server stopped notification
    func sendServerStoppedNotification() {
        guard AppSettings.shared.showNotifications else { return }

        send(
            title: "Alpha Arena Stopped",
            body: "Server has been stopped",
            category: .serverStatus
        )
    }

    /// Send error notification
    func sendErrorNotification(message: String) {
        guard AppSettings.shared.showNotifications else { return }

        send(
            title: "Alpha Arena Error",
            body: message,
            category: .criticalAlert,
            sound: .defaultCritical
        )
    }

    /// Send trading started notification
    func sendTradingStartedNotification(mode: TradingMode) {
        guard AppSettings.shared.showNotifications else { return }

        send(
            title: "Trading Started",
            body: "Trading is now active in \(mode.rawValue) mode",
            category: .tradingAlert
        )
    }

    /// Send trading stopped notification
    func sendTradingStoppedNotification() {
        guard AppSettings.shared.showNotifications else { return }

        send(
            title: "Trading Stopped",
            body: "Trading has been stopped",
            category: .tradingAlert
        )
    }

    /// Send kill switch notification
    func sendKillSwitchNotification(reason: String) {
        send(
            title: "Kill Switch Activated",
            body: reason,
            category: .criticalAlert,
            sound: .defaultCritical
        )
    }

    /// Send P&L update notification
    func sendPnLNotification(pnl: Double, positions: Int) {
        guard AppSettings.shared.showNotifications else { return }

        let sign = pnl >= 0 ? "+" : ""
        let body = "P&L: \(sign)$\(String(format: "%.2f", pnl)) | \(positions) positions"

        send(
            title: "Alpha Arena Update",
            body: body,
            category: .tradingAlert
        )
    }

    // MARK: - Private Methods

    private func send(
        title: String,
        body: String,
        category: NotificationCategory,
        sound: UNNotificationSound = .default,
        userInfo: [String: Any] = [:]
    ) {
        guard isAuthorized else {
            // Fall back to simple alert if not authorized
            showFallbackAlert(title: title, body: body)
            return
        }

        let content = UNMutableNotificationContent()
        content.title = title
        content.body = body
        content.sound = sound
        content.categoryIdentifier = category.rawValue
        content.userInfo = userInfo

        let request = UNNotificationRequest(
            identifier: UUID().uuidString,
            content: content,
            trigger: nil // Deliver immediately
        )

        center.add(request) { error in
            if let error = error {
                print("Failed to send notification: \(error)")
            }
        }
    }

    private func showFallbackAlert(title: String, body: String) {
        DispatchQueue.main.async {
            let alert = NSAlert()
            alert.messageText = title
            alert.informativeText = body
            alert.alertStyle = .informational
            alert.addButton(withTitle: "OK")
            alert.runModal()
        }
    }

    // MARK: - UNUserNotificationCenterDelegate

    func userNotificationCenter(
        _ center: UNUserNotificationCenter,
        willPresent notification: UNNotification,
        withCompletionHandler completionHandler: @escaping (UNNotificationPresentationOptions) -> Void
    ) {
        // Show notification even when app is in foreground
        completionHandler([.banner, .sound])
    }

    func userNotificationCenter(
        _ center: UNUserNotificationCenter,
        didReceive response: UNNotificationResponse,
        withCompletionHandler completionHandler: @escaping () -> Void
    ) {
        let actionIdentifier = response.actionIdentifier

        switch actionIdentifier {
        case NotificationAction.openDashboard.rawValue, UNNotificationDefaultActionIdentifier:
            openDashboard()

        case NotificationAction.viewLogs.rawValue:
            openLogs()

        default:
            break
        }

        completionHandler()
    }

    // MARK: - Actions

    private func openDashboard() {
        let settings = AppSettings.shared
        let host = settings.networkMode == .lan ? (getLocalIPAddress() ?? "127.0.0.1") : "127.0.0.1"
        let url = URL(string: "http://\(host):\(settings.serverPort)")!
        NSWorkspace.shared.open(url)
    }

    private func openLogs() {
        let settings = AppSettings.shared
        let projectPath = settings.projectPath

        guard !projectPath.isEmpty else { return }

        let script = """
        tell application "Terminal"
            do script "cd '\(projectPath)' && ./alpha logs"
            activate
        end tell
        """

        var error: NSDictionary?
        if let scriptObject = NSAppleScript(source: script) {
            scriptObject.executeAndReturnError(&error)
        }
    }
}
