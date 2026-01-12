import XCTest
@testable import AlphaArena

/// Tests for data models
final class ModelTests: XCTestCase {

    // MARK: - ServerState Tests

    func testServerStateDefaults() {
        let state = ServerState()

        XCTAssertEqual(state.status, .stopped)
        XCTAssertEqual(state.host, "127.0.0.1")
        XCTAssertEqual(state.port, 8000)
        XCTAssertNil(state.errorMessage)
        XCTAssertFalse(state.isRunning)
    }

    func testServerStateIsRunning() {
        var state = ServerState()

        state.status = .running
        XCTAssertTrue(state.isRunning)

        state.status = .stopped
        XCTAssertFalse(state.isRunning)

        state.status = .starting
        XCTAssertFalse(state.isRunning)

        state.status = .error
        XCTAssertFalse(state.isRunning)
    }

    func testServerStateDisplayURL() {
        var state = ServerState()
        state.host = "127.0.0.1"
        state.port = 8000

        XCTAssertEqual(state.displayURL, "http://127.0.0.1:8000")

        state.host = "0.0.0.0"
        state.port = 9000
        XCTAssertEqual(state.displayURL, "http://0.0.0.0:9000")
    }

    func testServerStateEquality() {
        var state1 = ServerState()
        var state2 = ServerState()

        XCTAssertEqual(state1, state2)

        state1.status = .running
        XCTAssertNotEqual(state1, state2)

        state2.status = .running
        XCTAssertEqual(state1, state2)
    }

    // MARK: - HealthResponse Tests

    func testHealthResponseDecoding() throws {
        let json = """
        {
            "status": "ok",
            "timestamp": "2024-01-15T14:32:15Z",
            "version": "1.0.0",
            "uptime": 9245.5
        }
        """

        let data = json.data(using: .utf8)!
        let response = try JSONDecoder().decode(HealthResponse.self, from: data)

        XCTAssertEqual(response.status, "ok")
        XCTAssertEqual(response.version, "1.0.0")
        XCTAssertEqual(response.uptime, 9245.5)
        XCTAssertTrue(response.isHealthy)
    }

    func testHealthResponseIsHealthy() throws {
        // Test "ok" status
        let okJson = """
        {"status": "ok"}
        """
        let okResponse = try JSONDecoder().decode(HealthResponse.self, from: okJson.data(using: .utf8)!)
        XCTAssertTrue(okResponse.isHealthy)

        // Test "healthy" status
        let healthyJson = """
        {"status": "healthy"}
        """
        let healthyResponse = try JSONDecoder().decode(HealthResponse.self, from: healthyJson.data(using: .utf8)!)
        XCTAssertTrue(healthyResponse.isHealthy)

        // Test unhealthy status
        let errorJson = """
        {"status": "error"}
        """
        let errorResponse = try JSONDecoder().decode(HealthResponse.self, from: errorJson.data(using: .utf8)!)
        XCTAssertFalse(errorResponse.isHealthy)
    }

    // MARK: - TradingStatus Tests

    func testTradingStatusDefaults() {
        let status = TradingStatus()

        XCTAssertFalse(status.isActive)
        XCTAssertEqual(status.mode, .simulation)
        XCTAssertEqual(status.totalPnL, 0.0)
        XCTAssertEqual(status.dailyPnL, 0.0)
        XCTAssertEqual(status.openPositions, 0)
    }

    func testTradingStatusPnLFormatted() {
        var status = TradingStatus()

        status.totalPnL = 127.45
        XCTAssertEqual(status.pnlFormatted, "+$127.45")

        status.totalPnL = -45.67
        XCTAssertEqual(status.pnlFormatted, "$-45.67")

        status.totalPnL = 0.0
        XCTAssertEqual(status.pnlFormatted, "+$0.00")
    }

    func testTradingStatusDailyPnLFormatted() {
        var status = TradingStatus()

        status.dailyPnL = 23.67
        XCTAssertEqual(status.dailyPnLFormatted, "+$23.67")

        status.dailyPnL = -12.34
        XCTAssertEqual(status.dailyPnLFormatted, "$-12.34")
    }

    func testTradingStatusText() {
        var status = TradingStatus()

        status.isActive = true
        XCTAssertEqual(status.statusText, "Trading Active")

        status.isActive = false
        XCTAssertEqual(status.statusText, "Trading Stopped")
    }

    // MARK: - TradingStatusResponse Tests

    func testTradingStatusResponseDecoding() throws {
        let json = """
        {
            "is_active": true,
            "mode": "simulation",
            "total_pnl": 127.45,
            "daily_pnl": 23.67,
            "open_positions": 3,
            "last_cycle_time": "2024-01-15T14:32:15Z"
        }
        """

        let data = json.data(using: .utf8)!
        let response = try JSONDecoder().decode(TradingStatusResponse.self, from: data)

        XCTAssertTrue(response.isActive)
        XCTAssertEqual(response.mode, "simulation")
        XCTAssertEqual(response.totalPnL, 127.45)
        XCTAssertEqual(response.dailyPnL, 23.67)
        XCTAssertEqual(response.openPositions, 3)
    }

    func testTradingStatusResponsePartialDecoding() throws {
        // Test with minimal fields
        let json = """
        {
            "is_active": false
        }
        """

        let data = json.data(using: .utf8)!
        let response = try JSONDecoder().decode(TradingStatusResponse.self, from: data)

        XCTAssertFalse(response.isActive)
        XCTAssertNil(response.mode)
        XCTAssertNil(response.totalPnL)
    }

    // MARK: - WebSocketMessage Tests

    func testWebSocketMessageDecoding() throws {
        let json = """
        {
            "type": "cycle_complete",
            "data": {
                "pnl": 127.45,
                "positions": 3
            }
        }
        """

        let data = json.data(using: .utf8)!
        let message = try JSONDecoder().decode(WebSocketMessage.self, from: data)

        XCTAssertEqual(message.type, "cycle_complete")
        XCTAssertEqual(message.data?.pnl, 127.45)
        XCTAssertEqual(message.data?.positions, 3)
    }

    func testWebSocketMessageTypes() {
        XCTAssertEqual(WebSocketMessageType.`init`.rawValue, "init")
        XCTAssertEqual(WebSocketMessageType.tradingStarted.rawValue, "trading_started")
        XCTAssertEqual(WebSocketMessageType.tradingStopped.rawValue, "trading_stopped")
        XCTAssertEqual(WebSocketMessageType.killSwitch.rawValue, "kill_switch")
        XCTAssertEqual(WebSocketMessageType.cycleComplete.rawValue, "cycle_complete")
    }

    // MARK: - TradingMode Tests

    func testTradingModeRawValues() {
        XCTAssertEqual(TradingMode.simulation.rawValue, "simulation")
        XCTAssertEqual(TradingMode.live.rawValue, "live")
    }

    // MARK: - NetworkMode Tests

    func testNetworkModeRawValues() {
        XCTAssertEqual(NetworkMode.localhost.rawValue, "127.0.0.1")
        XCTAssertEqual(NetworkMode.lan.rawValue, "0.0.0.0")
    }

    func testNetworkModeDisplayNames() {
        XCTAssertEqual(NetworkMode.localhost.displayName, "Localhost Only")
        XCTAssertEqual(NetworkMode.lan.displayName, "LAN Access")
    }

    func testNetworkModeDescriptions() {
        XCTAssertFalse(NetworkMode.localhost.description.isEmpty)
        XCTAssertFalse(NetworkMode.lan.description.isEmpty)
    }

    func testNetworkModeAllCases() {
        XCTAssertEqual(NetworkMode.allCases.count, 2)
        XCTAssertTrue(NetworkMode.allCases.contains(.localhost))
        XCTAssertTrue(NetworkMode.allCases.contains(.lan))
    }
}
