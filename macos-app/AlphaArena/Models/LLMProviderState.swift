import Foundation

/// Represents an LLM provider
enum LLMProviderType: String, CaseIterable, Codable, Identifiable {
    case anthropic
    case openai
    case google
    case xai
    case local

    var id: String { rawValue }

    var displayName: String {
        switch self {
        case .anthropic: return "Claude"
        case .openai: return "OpenAI"
        case .google: return "Gemini"
        case .xai: return "Grok"
        case .local: return "Local"
        }
    }

    var icon: String {
        switch self {
        case .anthropic: return "brain.head.profile"
        case .openai: return "sparkles"
        case .google: return "diamond"
        case .xai: return "bolt.fill"
        case .local: return "house.fill"
        }
    }

    var description: String {
        switch self {
        case .anthropic: return "Anthropic Claude models"
        case .openai: return "GPT-4o, o1 models"
        case .google: return "Gemini models"
        case .xai: return "Grok models (best for X/Twitter)"
        case .local: return "Local models via Ollama/LM Studio"
        }
    }
}

/// Health status for a provider
struct ProviderHealth: Codable {
    let name: String
    let isHealthy: Bool
    let lastCheck: Double?
    let lastSuccess: Double?
    let consecutiveFailures: Int
    let totalRequests: Int
    let totalFailures: Int
    let lastError: String?
    let latencyMs: Int?
    let successRate: Double?

    enum CodingKeys: String, CodingKey {
        case name
        case isHealthy = "is_healthy"
        case lastCheck = "last_check"
        case lastSuccess = "last_success"
        case consecutiveFailures = "consecutive_failures"
        case totalRequests = "total_requests"
        case totalFailures = "total_failures"
        case lastError = "last_error"
        case latencyMs = "latency_ms"
        case successRate = "success_rate"
    }

    var status: HealthStatus {
        if isHealthy { return .healthy }
        if consecutiveFailures > 0 && consecutiveFailures < 3 { return .degraded }
        return .unhealthy
    }
}

enum HealthStatus: String {
    case healthy
    case degraded
    case unhealthy
    case unknown

    var color: String {
        switch self {
        case .healthy: return "green"
        case .degraded: return "yellow"
        case .unhealthy: return "red"
        case .unknown: return "gray"
        }
    }
}

/// Response from /api/llm/providers
struct LLMProvidersResponse: Codable {
    let activeProvider: String?
    let fallbackOrder: [String]?
    let connectedProviders: [String]?
    let availableModels: [String: [String]]?
    let health: [String: ProviderHealth]?
    let registryInitialized: Bool?

    enum CodingKeys: String, CodingKey {
        case activeProvider = "active_provider"
        case fallbackOrder = "fallback_order"
        case connectedProviders = "connected_providers"
        case availableModels = "available_models"
        case health
        case registryInitialized = "registry_initialized"
    }
}

/// Response from /api/llm/initialize
struct LLMInitializeResponse: Codable {
    let status: String
    let activeProvider: String?
    let connectedProviders: [String]?
    let health: [String: ProviderHealth]?

    enum CodingKeys: String, CodingKey {
        case status
        case activeProvider = "active_provider"
        case connectedProviders = "connected_providers"
        case health
    }
}

/// Request body for setting active provider
struct SetActiveProviderRequest: Encodable {
    let provider: String
}

/// Request body for setting fallback order
struct SetFallbackOrderRequest: Encodable {
    let order: [String]
}

/// Observable state for LLM providers
class LLMProviderState: ObservableObject {
    @Published var isInitialized = false
    @Published var isInitializing = false
    @Published var activeProvider: LLMProviderType?
    @Published var connectedProviders: Set<LLMProviderType> = []
    @Published var fallbackOrder: [LLMProviderType] = [.anthropic, .openai, .google, .xai, .local]
    @Published var health: [LLMProviderType: ProviderHealth] = [:]
    @Published var errorMessage: String?

    func update(from response: LLMProvidersResponse) {
        isInitialized = response.registryInitialized ?? false

        if let active = response.activeProvider {
            activeProvider = LLMProviderType(rawValue: active)
        }

        if let connected = response.connectedProviders {
            connectedProviders = Set(connected.compactMap { LLMProviderType(rawValue: $0) })
        }

        if let order = response.fallbackOrder {
            fallbackOrder = order.compactMap { LLMProviderType(rawValue: $0) }
        }

        if let healthData = response.health {
            for (key, value) in healthData {
                if let provider = LLMProviderType(rawValue: key) {
                    health[provider] = value
                }
            }
        }
    }

    func getHealthStatus(for provider: LLMProviderType) -> HealthStatus {
        guard let h = health[provider] else { return .unknown }
        return h.status
    }
}
