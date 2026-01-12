import Foundation

/// Errors that can occur during API calls
enum APIError: Error, LocalizedError {
    case invalidURL
    case requestFailed(Int)
    case decodingFailed(Error)
    case networkError(Error)
    case serverNotRunning

    var errorDescription: String? {
        switch self {
        case .invalidURL:
            return "Invalid server URL"
        case .requestFailed(let statusCode):
            return "Request failed with status code \(statusCode)"
        case .decodingFailed(let error):
            return "Failed to decode response: \(error.localizedDescription)"
        case .networkError(let error):
            return "Network error: \(error.localizedDescription)"
        case .serverNotRunning:
            return "Server is not running"
        }
    }
}

/// HTTP client for Alpha Arena REST API
actor APIClient {
    private var baseURL: URL
    private let session: URLSession
    private let settings: AppSettings

    init(settings: AppSettings = .shared) {
        self.settings = settings
        self.baseURL = URL(string: "http://127.0.0.1:8000")!

        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest = 5
        config.timeoutIntervalForResource = 10
        self.session = URLSession(configuration: config)
    }

    /// Update base URL when settings change
    func updateBaseURL() async {
        let host = await MainActor.run { settings.networkMode.rawValue }
        let port = await MainActor.run { settings.serverPort }
        self.baseURL = URL(string: "http://\(host):\(port)")!
    }

    // MARK: - Health Check

    /// Check if the server is healthy
    func getHealth() async throws -> HealthResponse {
        let url = baseURL.appendingPathComponent("/api/system/health")
        return try await get(url: url)
    }

    /// Quick check if server is reachable
    func isServerRunning() async -> Bool {
        do {
            let health = try await getHealth()
            return health.isHealthy
        } catch {
            return false
        }
    }

    // MARK: - Trading Status

    /// Get current trading status
    func getTradingStatus() async throws -> TradingStatusResponse {
        let url = baseURL.appendingPathComponent("/api/trading/status")
        return try await get(url: url)
    }

    // MARK: - Trading Control

    /// Start trading
    func startTrading(mode: TradingMode = .simulation) async throws {
        let url = baseURL.appendingPathComponent("/api/trading/start")
        let body = ["mode": mode.rawValue]
        try await post(url: url, body: body)
    }

    /// Stop trading
    func stopTrading() async throws {
        let url = baseURL.appendingPathComponent("/api/trading/stop")
        try await post(url: url, body: [:] as [String: String])
    }

    /// Trigger kill switch
    func triggerKillSwitch(reason: String = "Manual trigger from menu bar app") async throws {
        let url = baseURL.appendingPathComponent("/api/trading/kill-switch")
        let body = ["reason": reason]
        try await post(url: url, body: body)
    }

    // MARK: - LLM Provider Management

    /// Get LLM providers status
    func getLLMProviders() async throws -> LLMProvidersResponse {
        let url = baseURL.appendingPathComponent("/api/llm/providers")
        return try await get(url: url)
    }

    /// Initialize LLM providers
    func initializeLLMProviders() async throws -> LLMInitializeResponse {
        let url = baseURL.appendingPathComponent("/api/llm/initialize")
        return try await postWithResponse(url: url, body: [:] as [String: String])
    }

    /// Set active provider
    func setActiveProvider(_ provider: String) async throws {
        let url = baseURL.appendingPathComponent("/api/llm/active")
        let body = SetActiveProviderRequest(provider: provider)
        try await post(url: url, body: body)
    }

    /// Set fallback order
    func setFallbackOrder(_ order: [String]) async throws {
        let url = baseURL.appendingPathComponent("/api/llm/fallback-order")
        let body = SetFallbackOrderRequest(order: order)
        try await post(url: url, body: body)
    }

    /// Check health of providers
    func checkLLMHealth() async throws -> [String: ProviderHealth] {
        let url = baseURL.appendingPathComponent("/api/llm/health/check")
        let response: [String: [String: ProviderHealth]] = try await postWithResponse(url: url, body: [:] as [String: String])
        return response["health"] ?? [:]
    }

    // MARK: - Private Methods

    private func get<T: Decodable>(url: URL) async throws -> T {
        var request = URLRequest(url: url)
        request.httpMethod = "GET"
        request.setValue("application/json", forHTTPHeaderField: "Accept")

        do {
            let (data, response) = try await session.data(for: request)

            guard let httpResponse = response as? HTTPURLResponse else {
                throw APIError.serverNotRunning
            }

            guard (200...299).contains(httpResponse.statusCode) else {
                throw APIError.requestFailed(httpResponse.statusCode)
            }

            do {
                let decoder = JSONDecoder()
                return try decoder.decode(T.self, from: data)
            } catch {
                throw APIError.decodingFailed(error)
            }
        } catch let error as APIError {
            throw error
        } catch {
            throw APIError.networkError(error)
        }
    }

    private func post<T: Encodable>(url: URL, body: T) async throws {
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("application/json", forHTTPHeaderField: "Accept")

        let encoder = JSONEncoder()
        request.httpBody = try encoder.encode(body)

        do {
            let (_, response) = try await session.data(for: request)

            guard let httpResponse = response as? HTTPURLResponse else {
                throw APIError.serverNotRunning
            }

            guard (200...299).contains(httpResponse.statusCode) else {
                throw APIError.requestFailed(httpResponse.statusCode)
            }
        } catch let error as APIError {
            throw error
        } catch {
            throw APIError.networkError(error)
        }
    }

    private func postWithResponse<T: Encodable, R: Decodable>(url: URL, body: T) async throws -> R {
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("application/json", forHTTPHeaderField: "Accept")

        let encoder = JSONEncoder()
        request.httpBody = try encoder.encode(body)

        do {
            let (data, response) = try await session.data(for: request)

            guard let httpResponse = response as? HTTPURLResponse else {
                throw APIError.serverNotRunning
            }

            guard (200...299).contains(httpResponse.statusCode) else {
                throw APIError.requestFailed(httpResponse.statusCode)
            }

            do {
                let decoder = JSONDecoder()
                return try decoder.decode(R.self, from: data)
            } catch {
                throw APIError.decodingFailed(error)
            }
        } catch let error as APIError {
            throw error
        } catch {
            throw APIError.networkError(error)
        }
    }
}
