import Foundation
import Combine

/// WebSocket client for real-time updates from Alpha Arena
class WebSocketClient: NSObject, ObservableObject {
    @Published var isConnected = false
    @Published var latestPnL: Double = 0
    @Published var dailyPnL: Double = 0
    @Published var openPositions: Int = 0
    @Published var isTradingActive: Bool = false
    @Published var lastMessage: String = ""

    private var webSocketTask: URLSessionWebSocketTask?
    private var session: URLSession!
    private var url: URL
    private var reconnectTimer: Timer?
    private var pingTimer: Timer?
    private let settings: AppSettings

    private var shouldReconnect = true
    private var reconnectAttempts = 0
    private let maxReconnectAttempts = 5
    private let reconnectDelay: TimeInterval = 3.0

    init(settings: AppSettings = .shared) {
        self.settings = settings
        self.url = URL(string: "ws://127.0.0.1:8000/ws")!
        super.init()
        self.session = URLSession(configuration: .default, delegate: self, delegateQueue: .main)
    }

    /// Update WebSocket URL when settings change
    func updateURL() {
        let host = settings.networkMode.rawValue
        let port = settings.serverPort
        self.url = URL(string: "ws://\(host):\(port)/ws")!
    }

    /// Connect to WebSocket
    func connect() {
        guard !isConnected else { return }

        shouldReconnect = true
        reconnectAttempts = 0
        establishConnection()
    }

    /// Disconnect from WebSocket
    func disconnect() {
        shouldReconnect = false
        reconnectTimer?.invalidate()
        reconnectTimer = nil
        pingTimer?.invalidate()
        pingTimer = nil

        webSocketTask?.cancel(with: .goingAway, reason: nil)
        webSocketTask = nil

        DispatchQueue.main.async {
            self.isConnected = false
        }
    }

    // MARK: - Private Methods

    private func establishConnection() {
        webSocketTask?.cancel()
        webSocketTask = session.webSocketTask(with: url)
        webSocketTask?.resume()

        receiveMessage()
        startPingTimer()
    }

    private func receiveMessage() {
        webSocketTask?.receive { [weak self] result in
            guard let self = self else { return }

            switch result {
            case .success(let message):
                self.handleMessage(message)
                self.receiveMessage() // Continue listening
            case .failure(let error):
                print("WebSocket receive error: \(error)")
                self.handleDisconnect()
            }
        }
    }

    private func handleMessage(_ message: URLSessionWebSocketTask.Message) {
        switch message {
        case .string(let text):
            parseMessage(text)
        case .data(let data):
            if let text = String(data: data, encoding: .utf8) {
                parseMessage(text)
            }
        @unknown default:
            break
        }
    }

    private func parseMessage(_ text: String) {
        DispatchQueue.main.async {
            self.lastMessage = text
        }

        guard let data = text.data(using: .utf8) else { return }

        do {
            let decoder = JSONDecoder()
            let message = try decoder.decode(WebSocketMessage.self, from: data)

            DispatchQueue.main.async {
                self.handleParsedMessage(message)
            }
        } catch {
            print("Failed to parse WebSocket message: \(error)")
        }
    }

    private func handleParsedMessage(_ message: WebSocketMessage) {
        guard let type = WebSocketMessageType(rawValue: message.type) else {
            print("Unknown message type: \(message.type)")
            return
        }

        switch type {
        case .`init`:
            // Initial connection state
            if let data = message.data {
                if let pnl = data.pnl {
                    latestPnL = pnl
                }
                if let positions = data.positions {
                    openPositions = positions
                }
            }
            isConnected = true
            reconnectAttempts = 0

        case .tradingStarted:
            isTradingActive = true
            if let mode = message.data?.mode {
                print("Trading started in \(mode) mode")
            }

        case .tradingStopped:
            isTradingActive = false

        case .killSwitch:
            isTradingActive = false
            if let reason = message.data?.reason {
                print("Kill switch triggered: \(reason)")
            }

        case .cycleComplete:
            if let data = message.data {
                if let pnl = data.pnl {
                    latestPnL = pnl
                }
                if let positions = data.positions {
                    openPositions = positions
                }
            }

        case .positionUpdate:
            if let positions = message.data?.positions {
                openPositions = positions
            }

        case .error:
            if let errorMessage = message.data?.message {
                print("WebSocket error: \(errorMessage)")
            }
        }
    }

    private func handleDisconnect() {
        DispatchQueue.main.async {
            self.isConnected = false
        }

        pingTimer?.invalidate()
        pingTimer = nil

        guard shouldReconnect else { return }
        guard reconnectAttempts < maxReconnectAttempts else {
            print("Max reconnect attempts reached")
            return
        }

        reconnectAttempts += 1
        let delay = reconnectDelay * Double(reconnectAttempts)

        print("Attempting to reconnect in \(delay) seconds (attempt \(reconnectAttempts)/\(maxReconnectAttempts))")

        DispatchQueue.main.asyncAfter(deadline: .now() + delay) { [weak self] in
            self?.establishConnection()
        }
    }

    private func startPingTimer() {
        pingTimer?.invalidate()
        pingTimer = Timer.scheduledTimer(withTimeInterval: 30.0, repeats: true) { [weak self] _ in
            self?.sendPing()
        }
    }

    private func sendPing() {
        webSocketTask?.sendPing { [weak self] error in
            if let error = error {
                print("Ping failed: \(error)")
                self?.handleDisconnect()
            }
        }
    }
}

// MARK: - URLSessionWebSocketDelegate

extension WebSocketClient: URLSessionWebSocketDelegate {
    func urlSession(_ session: URLSession, webSocketTask: URLSessionWebSocketTask, didOpenWithProtocol protocol: String?) {
        DispatchQueue.main.async {
            self.isConnected = true
            self.reconnectAttempts = 0
        }
        print("WebSocket connected")
    }

    func urlSession(_ session: URLSession, webSocketTask: URLSessionWebSocketTask, didCloseWith closeCode: URLSessionWebSocketTask.CloseCode, reason: Data?) {
        DispatchQueue.main.async {
            self.isConnected = false
        }

        if let reason = reason, let reasonString = String(data: reason, encoding: .utf8) {
            print("WebSocket closed: \(reasonString)")
        }

        if shouldReconnect {
            handleDisconnect()
        }
    }
}
