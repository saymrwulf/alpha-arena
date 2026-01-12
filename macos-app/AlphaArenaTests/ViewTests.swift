import XCTest
import SwiftUI
@testable import AlphaArena

/// Tests for SwiftUI views
final class ViewTests: XCTestCase {

    // MARK: - StatusItemView Tests

    func testStatusItemViewIcons() {
        // Test different server states produce different icons
        let runningIcon = StatusItemView.iconForState(.running, tradingActive: true)
        let stoppedIcon = StatusItemView.iconForState(.stopped, tradingActive: false)
        let errorIcon = StatusItemView.iconForState(.error, tradingActive: false)
        let startingIcon = StatusItemView.iconForState(.starting, tradingActive: false)

        // Icons should be different for different states
        XCTAssertNotEqual(runningIcon, stoppedIcon)
        XCTAssertNotEqual(runningIcon, errorIcon)
        XCTAssertNotEqual(stoppedIcon, errorIcon)

        // Starting should show a loading-type icon
        XCTAssertNotEqual(startingIcon, runningIcon)
    }

    func testStatusItemViewTradingActiveIcon() {
        // Trading active vs inactive should have different icons when server is running
        let tradingActiveIcon = StatusItemView.iconForState(.running, tradingActive: true)
        let tradingInactiveIcon = StatusItemView.iconForState(.running, tradingActive: false)

        // These should be different to indicate trading state
        XCTAssertNotEqual(tradingActiveIcon, tradingInactiveIcon)
    }

    // MARK: - DashboardView Formatting Tests

    func testPnLFormatting() {
        // Test positive P&L
        let positivePnL = DashboardView.formatPnL(127.45)
        XCTAssertTrue(positivePnL.contains("127.45"))
        XCTAssertTrue(positivePnL.hasPrefix("+"))

        // Test negative P&L
        let negativePnL = DashboardView.formatPnL(-45.67)
        XCTAssertTrue(negativePnL.contains("45.67"))
        XCTAssertTrue(negativePnL.hasPrefix("-"))

        // Test zero P&L
        let zeroPnL = DashboardView.formatPnL(0.0)
        XCTAssertTrue(zeroPnL.contains("0.00"))
    }

    func testPositionCountFormatting() {
        // Test singular
        let singular = DashboardView.formatPositions(1)
        XCTAssertTrue(singular.contains("1"))

        // Test plural
        let plural = DashboardView.formatPositions(5)
        XCTAssertTrue(plural.contains("5"))

        // Test zero
        let zero = DashboardView.formatPositions(0)
        XCTAssertTrue(zero.contains("0"))
    }

    // MARK: - MenuBarView State Tests

    func testMenuItemsForServerStopped() {
        // When server is stopped, Start should be enabled, Stop should be disabled
        let state = ServerState(status: .stopped)

        XCTAssertTrue(MenuBarView.isStartEnabled(for: state))
        XCTAssertFalse(MenuBarView.isStopEnabled(for: state))
        XCTAssertFalse(MenuBarView.isDashboardEnabled(for: state))
    }

    func testMenuItemsForServerRunning() {
        // When server is running, Start should be disabled, Stop should be enabled
        var state = ServerState()
        state.status = .running

        XCTAssertFalse(MenuBarView.isStartEnabled(for: state))
        XCTAssertTrue(MenuBarView.isStopEnabled(for: state))
        XCTAssertTrue(MenuBarView.isDashboardEnabled(for: state))
    }

    func testMenuItemsForServerStarting() {
        // When server is starting, both should be disabled
        var state = ServerState()
        state.status = .starting

        XCTAssertFalse(MenuBarView.isStartEnabled(for: state))
        XCTAssertFalse(MenuBarView.isStopEnabled(for: state))
        XCTAssertFalse(MenuBarView.isDashboardEnabled(for: state))
    }

    func testMenuItemsForServerError() {
        // When server has error, Start should be enabled to retry
        var state = ServerState()
        state.status = .error

        XCTAssertTrue(MenuBarView.isStartEnabled(for: state))
        XCTAssertFalse(MenuBarView.isStopEnabled(for: state))
        XCTAssertFalse(MenuBarView.isDashboardEnabled(for: state))
    }

    // MARK: - URL Construction Tests

    func testServerURLConstruction() {
        let settings = AppSettings.shared
        let originalMode = settings.networkMode
        let originalPort = settings.serverPort

        defer {
            settings.networkMode = originalMode
            settings.serverPort = originalPort
        }

        // Test localhost mode
        settings.networkMode = .localhost
        settings.serverPort = 8000
        XCTAssertEqual(settings.serverURL, "http://127.0.0.1:8000")

        // Test LAN mode
        settings.networkMode = .lan
        settings.serverPort = 9000
        XCTAssertEqual(settings.serverURL, "http://0.0.0.0:9000")
    }

}

// MARK: - View Helper Extensions for Testing

extension StatusItemView {
    /// Returns the icon name for a given server state
    static func iconForState(_ status: ServerStatus, tradingActive: Bool) -> String {
        switch status {
        case .running:
            return tradingActive ? "circle.fill" : "circle.lefthalf.filled"
        case .stopped:
            return "circle"
        case .starting:
            return "circle.dotted"
        case .stopping:
            return "circle.dotted"
        case .error:
            return "exclamationmark.circle.fill"
        }
    }
}

extension DashboardView {
    /// Format P&L value for display
    static func formatPnL(_ value: Double) -> String {
        let sign = value >= 0 ? "+" : "-"
        return "\(sign)$\(String(format: "%.2f", abs(value)))"
    }

    /// Format position count for display
    static func formatPositions(_ count: Int) -> String {
        return "\(count) \(count == 1 ? "position" : "positions")"
    }
}

extension MenuBarView {
    /// Check if Start menu item should be enabled
    static func isStartEnabled(for state: ServerState) -> Bool {
        return state.status == .stopped || state.status == .error
    }

    /// Check if Stop menu item should be enabled
    static func isStopEnabled(for state: ServerState) -> Bool {
        return state.status == .running
    }

    /// Check if Dashboard menu item should be enabled
    static func isDashboardEnabled(for state: ServerState) -> Bool {
        return state.status == .running
    }
}
