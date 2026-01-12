import Foundation

/// Trading mode
enum TradingMode: String, Codable {
    case simulation
    case live
}

/// Trading state
enum TradingState: String, Codable {
    case active
    case stopped
    case paused
}

/// Response from /api/trading/status
struct TradingStatusResponse: Codable {
    let isActive: Bool
    let mode: String?
    let totalPnL: Double?
    let dailyPnL: Double?
    let openPositions: Int?
    let lastCycleTime: String?

    enum CodingKeys: String, CodingKey {
        case isActive = "is_active"
        case mode
        case totalPnL = "total_pnl"
        case dailyPnL = "daily_pnl"
        case openPositions = "open_positions"
        case lastCycleTime = "last_cycle_time"
    }
}

/// Trading status for display
struct TradingStatus: Equatable {
    var isActive: Bool = false
    var mode: TradingMode = .simulation
    var totalPnL: Double = 0.0
    var dailyPnL: Double = 0.0
    var openPositions: Int = 0
    var lastUpdate: Date?

    var pnlFormatted: String {
        let sign = totalPnL >= 0 ? "+" : ""
        return String(format: "%@$%.2f", sign, totalPnL)
    }

    var dailyPnLFormatted: String {
        let sign = dailyPnL >= 0 ? "+" : ""
        return String(format: "%@$%.2f", sign, dailyPnL)
    }

    var statusText: String {
        isActive ? "Trading Active" : "Trading Stopped"
    }
}

/// WebSocket message types
enum WebSocketMessageType: String, Codable {
    case `init`
    case tradingStarted = "trading_started"
    case tradingStopped = "trading_stopped"
    case killSwitch = "kill_switch"
    case cycleComplete = "cycle_complete"
    case positionUpdate = "position_update"
    case error
}

/// WebSocket message structure
struct WebSocketMessage: Codable {
    let type: String
    let data: WebSocketData?
}

struct WebSocketData: Codable {
    let pnl: Double?
    let positions: Int?
    let mode: String?
    let reason: String?
    let message: String?

    enum CodingKeys: String, CodingKey {
        case pnl
        case positions
        case mode
        case reason
        case message
    }
}
