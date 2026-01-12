import Foundation
import Combine
import AppKit

/// Main service coordinator for Alpha Arena
/// Manages server state, API communication, and WebSocket connection
@MainActor
class AlphaArenaService: ObservableObject {
    // MARK: - Published State

    @Published var serverState = ServerState()
    @Published var tradingStatus = TradingStatus()
    @Published var isLoading = false
    @Published var errorMessage: String?

    // MARK: - Dependencies

    let settings: AppSettings
    let processController: ProcessController
    let apiClient: APIClient
    let webSocketClient: WebSocketClient
    let notificationManager: NotificationManager

    // MARK: - Private Properties

    private var healthCheckTimer: Timer?
    private var cancellables = Set<AnyCancellable>()
    private var isStoppingServer = false

    // MARK: - Initialization

    init(
        settings: AppSettings = .shared,
        processController: ProcessController? = nil,
        apiClient: APIClient? = nil,
        webSocketClient: WebSocketClient? = nil,
        notificationManager: NotificationManager? = nil
    ) {
        self.settings = settings
        self.processController = processController ?? ProcessController(settings: settings)
        self.apiClient = apiClient ?? APIClient(settings: settings)
        self.webSocketClient = webSocketClient ?? WebSocketClient(settings: settings)
        self.notificationManager = notificationManager ?? NotificationManager()

        setupBindings()
        startHealthCheck()
    }

    deinit {
        healthCheckTimer?.invalidate()
    }

    // MARK: - Server Control

    /// Start the Alpha Arena server
    func startServer() async {
        guard !isLoading else { return }

        isLoading = true
        errorMessage = nil
        serverState.status = .starting

        do {
            let lanAccess = settings.networkMode == .lan
            try await processController.startServer(lanAccess: lanAccess)

            // Wait a moment for server to start
            try await Task.sleep(nanoseconds: 2_000_000_000) // 2 seconds

            // Check if actually running
            await checkServerHealth()

            if serverState.status == .running {
                notificationManager.sendServerStartedNotification(url: serverState.displayURL)

                // Connect WebSocket if enabled
                if settings.autoConnectWebSocket {
                    webSocketClient.connect()
                }
            }
        } catch {
            serverState.status = .error
            serverState.errorMessage = error.localizedDescription
            errorMessage = error.localizedDescription
            notificationManager.sendErrorNotification(message: error.localizedDescription)
        }

        isLoading = false
    }

    /// Stop the Alpha Arena server
    func stopServer() async {
        guard !isLoading else { return }

        isLoading = true
        isStoppingServer = true
        errorMessage = nil
        serverState.status = .stopping

        // Pause health check timer during stop
        healthCheckTimer?.invalidate()
        healthCheckTimer = nil

        // Disconnect WebSocket first
        webSocketClient.disconnect()

        do {
            try await processController.stopServer()

            // Wait for server to fully stop
            try await Task.sleep(nanoseconds: 2_000_000_000) // 2 seconds

            // Verify it's actually stopped
            let stillRunning = await apiClient.isServerRunning()
            if stillRunning {
                // Try stopping again with force
                try? await processController.stopServer()
                try await Task.sleep(nanoseconds: 1_000_000_000)
            }

            serverState.status = .stopped
            tradingStatus = TradingStatus()
            notificationManager.sendServerStoppedNotification()
        } catch {
            serverState.status = .error
            serverState.errorMessage = error.localizedDescription
            errorMessage = error.localizedDescription
        }

        isLoading = false
        isStoppingServer = false

        // Restart health check timer after a delay
        DispatchQueue.main.asyncAfter(deadline: .now() + 3.0) { [weak self] in
            self?.startHealthCheck()
        }
    }

    /// Restart the server
    func restartServer() async {
        await stopServer()
        try? await Task.sleep(nanoseconds: 500_000_000) // 0.5 second
        await startServer()
    }

    // MARK: - Trading Control

    /// Start trading
    func startTrading(mode: TradingMode = .simulation) async {
        guard serverState.isRunning else {
            errorMessage = "Server must be running to start trading"
            return
        }

        do {
            try await apiClient.startTrading(mode: mode)
            tradingStatus.isActive = true
            tradingStatus.mode = mode
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    /// Stop trading
    func stopTrading() async {
        do {
            try await apiClient.stopTrading()
            tradingStatus.isActive = false
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    /// Trigger kill switch
    func triggerKillSwitch(reason: String = "Manual trigger") async {
        do {
            try await apiClient.triggerKillSwitch(reason: reason)
            tradingStatus.isActive = false
            notificationManager.sendKillSwitchNotification(reason: reason)
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    // MARK: - Actions

    /// Open dashboard in browser
    func openDashboard() {
        Task {
            await processController.openDashboard()
        }
    }

    /// Open logs in Terminal
    func openLogs() {
        Task {
            try? await processController.openLogs()
        }
    }

    /// Copy server URL to clipboard
    func copyURL() {
        Task {
            await processController.copyURL()
        }
    }

    // MARK: - Health Check

    /// Check server health
    func checkServerHealth() async {
        // Skip health check if we're in the process of stopping
        guard !isStoppingServer else { return }

        let isRunning = await apiClient.isServerRunning()

        if isRunning {
            serverState.status = .running
            serverState.lastChecked = Date()

            // Fetch trading status
            await fetchTradingStatus()
        } else {
            // Only update status if we're not actively stopping
            guard !isStoppingServer else { return }

            // Check via process controller
            do {
                let status = try await processController.getStatus()
                // Don't override stopped status unless truly running
                if status == .running {
                    serverState.status = status
                } else if serverState.status != .stopping {
                    serverState.status = status
                }
            } catch {
                if serverState.status != .stopping {
                    serverState.status = .stopped
                }
            }
        }

        serverState.host = settings.networkMode.rawValue
        serverState.port = settings.serverPort
    }

    /// Fetch trading status from API
    func fetchTradingStatus() async {
        guard serverState.isRunning else { return }

        do {
            let response = try await apiClient.getTradingStatus()
            tradingStatus.isActive = response.isActive
            tradingStatus.totalPnL = response.totalPnL ?? 0
            tradingStatus.dailyPnL = response.dailyPnL ?? 0
            tradingStatus.openPositions = response.openPositions ?? 0
            tradingStatus.lastUpdate = Date()

            if let modeStr = response.mode, let mode = TradingMode(rawValue: modeStr) {
                tradingStatus.mode = mode
            }
        } catch {
            // Silently fail - server might not have trading status endpoint
        }
    }

    // MARK: - Private Methods

    private func setupBindings() {
        // Bind WebSocket updates to trading status
        webSocketClient.$latestPnL
            .receive(on: DispatchQueue.main)
            .sink { [weak self] pnl in
                self?.tradingStatus.totalPnL = pnl
            }
            .store(in: &cancellables)

        webSocketClient.$openPositions
            .receive(on: DispatchQueue.main)
            .sink { [weak self] positions in
                self?.tradingStatus.openPositions = positions
            }
            .store(in: &cancellables)

        webSocketClient.$isTradingActive
            .receive(on: DispatchQueue.main)
            .sink { [weak self] isActive in
                self?.tradingStatus.isActive = isActive
            }
            .store(in: &cancellables)

        webSocketClient.$isConnected
            .receive(on: DispatchQueue.main)
            .sink { [weak self] isConnected in
                if isConnected {
                    self?.serverState.status = .running
                }
            }
            .store(in: &cancellables)

        // Watch for settings changes
        settings.$networkMode
            .receive(on: DispatchQueue.main)
            .sink { [weak self] mode in
                self?.serverState.host = mode.rawValue
                self?.webSocketClient.updateURL()
                Task {
                    await self?.apiClient.updateBaseURL()
                }
            }
            .store(in: &cancellables)

        settings.$serverPort
            .receive(on: DispatchQueue.main)
            .sink { [weak self] port in
                self?.serverState.port = port
                self?.webSocketClient.updateURL()
                Task {
                    await self?.apiClient.updateBaseURL()
                }
            }
            .store(in: &cancellables)
    }

    private func startHealthCheck() {
        // Check immediately
        Task {
            await checkServerHealth()
        }

        // Then check every 4 seconds
        healthCheckTimer = Timer.scheduledTimer(withTimeInterval: 4.0, repeats: true) { [weak self] _ in
            Task { @MainActor in
                await self?.checkServerHealth()
            }
        }
    }
}
