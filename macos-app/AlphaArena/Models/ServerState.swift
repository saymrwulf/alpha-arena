import Foundation

/// Represents the current state of the Alpha Arena server
enum ServerStatus: String, Codable {
    case running
    case stopped
    case starting
    case stopping
    case error
}

/// Server state with connection details
struct ServerState: Equatable {
    var status: ServerStatus = .stopped
    var url: String = "http://127.0.0.1:8000"
    var host: String = "127.0.0.1"
    var port: Int = 8000
    var errorMessage: String?
    var lastChecked: Date?

    var isRunning: Bool {
        status == .running
    }

    var displayURL: String {
        "http://\(host):\(port)"
    }
}

/// Health check response from the API
struct HealthResponse: Codable {
    let status: String
    let timestamp: String?
    let version: String?
    let uptime: Double?

    var isHealthy: Bool {
        status == "ok" || status == "healthy"
    }
}
