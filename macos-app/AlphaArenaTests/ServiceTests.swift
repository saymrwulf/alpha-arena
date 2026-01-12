import XCTest
@testable import AlphaArena

/// Tests for service classes
final class ServiceTests: XCTestCase {

    // MARK: - AppSettings Tests

    func testAppSettingsSharedInstance() {
        let settings1 = AppSettings.shared
        let settings2 = AppSettings.shared

        XCTAssertTrue(settings1 === settings2, "Shared instance should be singleton")
    }

    func testAppSettingsDefaultValues() {
        let settings = AppSettings.shared

        // These should have reasonable defaults
        XCTAssertEqual(settings.serverPort, 8000)
        XCTAssertEqual(settings.networkMode, .localhost)
        XCTAssertTrue(settings.autoConnectWebSocket)
        XCTAssertTrue(settings.showNotifications)
    }

    func testAppSettingsServerURL() {
        let settings = AppSettings.shared
        let originalMode = settings.networkMode
        let originalPort = settings.serverPort

        defer {
            settings.networkMode = originalMode
            settings.serverPort = originalPort
        }

        settings.networkMode = .localhost
        settings.serverPort = 8000
        XCTAssertEqual(settings.serverURL, "http://127.0.0.1:8000")

        settings.networkMode = .lan
        settings.serverPort = 9000
        XCTAssertEqual(settings.serverURL, "http://0.0.0.0:9000")
    }

    func testAppSettingsNetworkModePersistence() {
        let settings = AppSettings.shared
        let originalMode = settings.networkMode

        defer {
            settings.networkMode = originalMode
        }

        settings.networkMode = .lan
        XCTAssertEqual(settings.networkMode, .lan)

        settings.networkMode = .localhost
        XCTAssertEqual(settings.networkMode, .localhost)
    }

    func testAppSettingsPortPersistence() {
        let settings = AppSettings.shared
        let originalPort = settings.serverPort

        defer {
            settings.serverPort = originalPort
        }

        settings.serverPort = 9000
        XCTAssertEqual(settings.serverPort, 9000)

        settings.serverPort = 8080
        XCTAssertEqual(settings.serverPort, 8080)
    }

    // MARK: - ProcessError Tests

    func testProcessErrorDescriptions() {
        XCTAssertNotNil(ProcessError.projectNotFound.errorDescription)
        XCTAssertNotNil(ProcessError.scriptNotFound.errorDescription)
        XCTAssertNotNil(ProcessError.timeout.errorDescription)

        let executionError = ProcessError.executionFailed("Test error")
        XCTAssertTrue(executionError.errorDescription?.contains("Test error") ?? false)
    }

    // MARK: - APIError Tests

    func testAPIErrorDescriptions() {
        XCTAssertNotNil(APIError.invalidURL.errorDescription)
        XCTAssertNotNil(APIError.serverNotRunning.errorDescription)

        let requestError = APIError.requestFailed(404)
        XCTAssertTrue(requestError.errorDescription?.contains("404") ?? false)

        let networkError = APIError.networkError(NSError(domain: "test", code: -1))
        XCTAssertNotNil(networkError.errorDescription)
    }

    // MARK: - WebSocketClient Tests

    func testWebSocketClientInitialization() {
        let client = WebSocketClient()

        XCTAssertFalse(client.isConnected)
        XCTAssertEqual(client.latestPnL, 0)
        XCTAssertEqual(client.openPositions, 0)
        XCTAssertFalse(client.isTradingActive)
    }

    func testWebSocketClientDisconnect() {
        let client = WebSocketClient()

        // Connect then disconnect
        client.connect()
        client.disconnect()

        // Should be disconnected
        XCTAssertFalse(client.isConnected)
    }

    func testWebSocketClientUpdateURL() {
        let settings = AppSettings.shared
        let originalMode = settings.networkMode
        let originalPort = settings.serverPort

        defer {
            settings.networkMode = originalMode
            settings.serverPort = originalPort
        }

        let client = WebSocketClient(settings: settings)

        settings.networkMode = .lan
        settings.serverPort = 9000
        client.updateURL()

        // URL should be updated (we can't easily verify the private URL)
        // but at least ensure no crash
        XCTAssertFalse(client.isConnected)
    }

    // MARK: - ProcessController Tests

    func testProcessControllerInitialization() async {
        let controller = ProcessController()

        // Just verify initialization doesn't crash
        XCTAssertNotNil(controller)
    }

    // MARK: - APIClient Tests

    func testAPIClientInitialization() async {
        let client = APIClient()

        // Verify initialization
        XCTAssertNotNil(client)

        // Test that server not running returns false
        let isRunning = await client.isServerRunning()
        // Note: This will likely be false unless server is actually running
        XCTAssertFalse(isRunning) // Expected when server not running
    }

    func testAPIClientUpdateBaseURL() async {
        let settings = AppSettings.shared
        let originalMode = settings.networkMode

        defer {
            settings.networkMode = originalMode
        }

        let client = APIClient(settings: settings)

        settings.networkMode = .lan
        await client.updateBaseURL()

        // Verify no crash - can't easily test private baseURL
        XCTAssertNotNil(client)
    }
}

// MARK: - Mock Tests

/// Tests using mock data
final class MockServiceTests: XCTestCase {

    // MARK: - JSON Parsing Tests

    func testParseHealthResponse() throws {
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

        XCTAssertTrue(response.isHealthy)
        XCTAssertEqual(response.status, "ok")
    }

    func testParseTradingStatusResponse() throws {
        let json = """
        {
            "is_active": true,
            "mode": "simulation",
            "total_pnl": 127.45,
            "daily_pnl": 23.67,
            "open_positions": 3
        }
        """

        let data = json.data(using: .utf8)!
        let response = try JSONDecoder().decode(TradingStatusResponse.self, from: data)

        XCTAssertTrue(response.isActive)
        XCTAssertEqual(response.mode, "simulation")
        XCTAssertEqual(response.totalPnL, 127.45)
    }

    func testParseWebSocketInitMessage() throws {
        let json = """
        {
            "type": "init",
            "data": {
                "pnl": 0,
                "positions": 0
            }
        }
        """

        let data = json.data(using: .utf8)!
        let message = try JSONDecoder().decode(WebSocketMessage.self, from: data)

        XCTAssertEqual(message.type, "init")
    }

    func testParseWebSocketCycleCompleteMessage() throws {
        let json = """
        {
            "type": "cycle_complete",
            "data": {
                "pnl": 127.45,
                "positions": 3,
                "mode": "simulation"
            }
        }
        """

        let data = json.data(using: .utf8)!
        let message = try JSONDecoder().decode(WebSocketMessage.self, from: data)

        XCTAssertEqual(message.type, "cycle_complete")
        XCTAssertEqual(message.data?.pnl, 127.45)
        XCTAssertEqual(message.data?.positions, 3)
    }

    func testParseWebSocketTradingStartedMessage() throws {
        let json = """
        {
            "type": "trading_started",
            "data": {
                "mode": "simulation"
            }
        }
        """

        let data = json.data(using: .utf8)!
        let message = try JSONDecoder().decode(WebSocketMessage.self, from: data)

        XCTAssertEqual(message.type, "trading_started")
        XCTAssertEqual(message.data?.mode, "simulation")
    }

    func testParseWebSocketKillSwitchMessage() throws {
        let json = """
        {
            "type": "kill_switch",
            "data": {
                "reason": "Daily loss limit exceeded"
            }
        }
        """

        let data = json.data(using: .utf8)!
        let message = try JSONDecoder().decode(WebSocketMessage.self, from: data)

        XCTAssertEqual(message.type, "kill_switch")
        XCTAssertEqual(message.data?.reason, "Daily loss limit exceeded")
    }

    // MARK: - Edge Case Tests

    func testParseEmptyWebSocketData() throws {
        let json = """
        {
            "type": "trading_stopped",
            "data": null
        }
        """

        let data = json.data(using: .utf8)!
        let message = try JSONDecoder().decode(WebSocketMessage.self, from: data)

        XCTAssertEqual(message.type, "trading_stopped")
        XCTAssertNil(message.data)
    }

    func testParseMinimalHealthResponse() throws {
        let json = """
        {"status": "ok"}
        """

        let data = json.data(using: .utf8)!
        let response = try JSONDecoder().decode(HealthResponse.self, from: data)

        XCTAssertEqual(response.status, "ok")
        XCTAssertNil(response.timestamp)
        XCTAssertNil(response.version)
        XCTAssertNil(response.uptime)
    }

    func testParseMinimalTradingStatus() throws {
        let json = """
        {"is_active": false}
        """

        let data = json.data(using: .utf8)!
        let response = try JSONDecoder().decode(TradingStatusResponse.self, from: data)

        XCTAssertFalse(response.isActive)
        XCTAssertNil(response.mode)
        XCTAssertNil(response.totalPnL)
        XCTAssertNil(response.openPositions)
    }
}
