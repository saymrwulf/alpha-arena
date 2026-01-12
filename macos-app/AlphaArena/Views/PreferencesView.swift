import SwiftUI
import UniformTypeIdentifiers

/// Preferences window view
struct PreferencesView: View {
    @ObservedObject var settings = AppSettings.shared
    @StateObject private var llmState = LLMProviderState()
    @State private var showingPathPicker = false

    private let apiClient = APIClient()

    var body: some View {
        TabView {
            generalTab
                .tabItem {
                    Label("General", systemImage: "gear")
                }

            llmProvidersTab
                .tabItem {
                    Label("LLM", systemImage: "brain")
                }

            networkTab
                .tabItem {
                    Label("Network", systemImage: "network")
                }

            notificationsTab
                .tabItem {
                    Label("Notifications", systemImage: "bell")
                }

            aboutTab
                .tabItem {
                    Label("About", systemImage: "info.circle")
                }
        }
        .frame(width: 500, height: 380)
        .task {
            await fetchLLMState()
        }
    }

    // MARK: - General Tab

    private var generalTab: some View {
        Form {
            Section {
                HStack {
                    TextField("Project Path", text: $settings.projectPath)
                        .textFieldStyle(.roundedBorder)

                    Button("Browse...") {
                        showingPathPicker = true
                    }
                }
                .fileImporter(
                    isPresented: $showingPathPicker,
                    allowedContentTypes: [.folder]
                ) { result in
                    if case .success(let url) = result {
                        settings.projectPath = url.path
                        settings.saveProjectPath()
                    }
                }

                Text("Path to the Alpha Arena project directory containing the 'alpha' script.")
                    .font(.caption)
                    .foregroundColor(.secondary)
            } header: {
                Text("Project Location")
            }

            Section {
                Picker("Default Port", selection: $settings.serverPort) {
                    Text("8000").tag(8000)
                    Text("8080").tag(8080)
                    Text("3000").tag(3000)
                }
                .pickerStyle(.menu)
            } header: {
                Text("Server")
            }

            Section {
                Toggle("Auto-connect WebSocket", isOn: $settings.autoConnectWebSocket)
                Text("Automatically connect to the WebSocket for real-time updates when the server starts.")
                    .font(.caption)
                    .foregroundColor(.secondary)
            } header: {
                Text("Connection")
            }
        }
        .padding()
    }

    // MARK: - LLM Providers Tab

    private var llmProvidersTab: some View {
        Form {
            Section {
                if !llmState.isInitialized {
                    VStack(spacing: 12) {
                        Image(systemName: "exclamationmark.triangle")
                            .font(.title)
                            .foregroundColor(.orange)
                        Text("Provider registry not initialized")
                            .font(.headline)
                        Text("Start the server and click Initialize to connect to LLM providers.")
                            .font(.caption)
                            .foregroundColor(.secondary)
                            .multilineTextAlignment(.center)
                        Button("Initialize Providers") {
                            Task { await initializeProviders() }
                        }
                        .buttonStyle(.borderedProminent)
                        .disabled(llmState.isInitializing)
                    }
                    .frame(maxWidth: .infinity)
                    .padding()
                } else {
                    HStack {
                        Text("Active Provider:")
                            .foregroundColor(.secondary)
                        Text(llmState.activeProvider?.displayName ?? "None")
                            .fontWeight(.semibold)
                        Spacer()
                        Button("Refresh") {
                            Task { await fetchLLMState() }
                        }
                    }
                }
            } header: {
                Text("Status")
            }

            if llmState.isInitialized {
                Section {
                    ForEach(LLMProviderType.allCases) { provider in
                        ProviderRow(
                            provider: provider,
                            isActive: llmState.activeProvider == provider,
                            isConnected: llmState.connectedProviders.contains(provider),
                            healthStatus: llmState.getHealthStatus(for: provider)
                        ) {
                            Task { await setActiveProvider(provider) }
                        }
                    }
                } header: {
                    Text("Providers")
                }

                Section {
                    VStack(alignment: .leading, spacing: 8) {
                        Text("If the active provider fails, the system will try providers in this order:")
                            .font(.caption)
                            .foregroundColor(.secondary)

                        ForEach(Array(llmState.fallbackOrder.enumerated()), id: \.element) { index, provider in
                            HStack {
                                Text("\(index + 1).")
                                    .foregroundColor(.secondary)
                                    .frame(width: 20)
                                Text(provider.displayName)
                                Spacer()
                                if index > 0 {
                                    Button("↑") {
                                        moveFallback(index: index, direction: -1)
                                    }
                                    .buttonStyle(.plain)
                                }
                                if index < llmState.fallbackOrder.count - 1 {
                                    Button("↓") {
                                        moveFallback(index: index, direction: 1)
                                    }
                                    .buttonStyle(.plain)
                                }
                            }
                            .padding(.vertical, 2)
                        }

                        Button("Save Fallback Order") {
                            Task { await saveFallbackOrder() }
                        }
                        .buttonStyle(.bordered)
                        .padding(.top, 8)
                    }
                } header: {
                    Text("Fallback Order")
                }
            }
        }
        .padding()
    }

    // MARK: - LLM Helper Methods

    private func fetchLLMState() async {
        do {
            let response = try await apiClient.getLLMProviders()
            await MainActor.run {
                llmState.update(from: response)
            }
        } catch {
            // Server might not be running
        }
    }

    private func initializeProviders() async {
        await MainActor.run {
            llmState.isInitializing = true
        }
        do {
            let response = try await apiClient.initializeLLMProviders()
            await MainActor.run {
                llmState.isInitialized = true
                if let active = response.activeProvider {
                    llmState.activeProvider = LLMProviderType(rawValue: active)
                }
                if let connected = response.connectedProviders {
                    llmState.connectedProviders = Set(connected.compactMap { LLMProviderType(rawValue: $0) })
                }
            }
        } catch {
            await MainActor.run {
                llmState.errorMessage = error.localizedDescription
            }
        }
        await MainActor.run {
            llmState.isInitializing = false
        }
    }

    private func setActiveProvider(_ provider: LLMProviderType) async {
        guard llmState.connectedProviders.contains(provider) else { return }
        do {
            try await apiClient.setActiveProvider(provider.rawValue)
            await MainActor.run {
                llmState.activeProvider = provider
            }
        } catch {
            await MainActor.run {
                llmState.errorMessage = error.localizedDescription
            }
        }
    }

    private func moveFallback(index: Int, direction: Int) {
        let newIndex = index + direction
        guard newIndex >= 0 && newIndex < llmState.fallbackOrder.count else { return }
        llmState.fallbackOrder.swapAt(index, newIndex)
    }

    private func saveFallbackOrder() async {
        do {
            let order = llmState.fallbackOrder.map { $0.rawValue }
            try await apiClient.setFallbackOrder(order)
        } catch {
            await MainActor.run {
                llmState.errorMessage = error.localizedDescription
            }
        }
    }

    // MARK: - Network Tab

    private var networkTab: some View {
        Form {
            Section {
                Picker("Default Network Mode", selection: $settings.networkMode) {
                    ForEach(NetworkMode.allCases, id: \.rawValue) { mode in
                        VStack(alignment: .leading) {
                            Text(mode.displayName)
                            Text(mode.description)
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                        .tag(mode)
                    }
                }
                .pickerStyle(.radioGroup)
            } header: {
                Text("Network Mode")
            }

            Section {
                VStack(alignment: .leading, spacing: 8) {
                    Label {
                        VStack(alignment: .leading) {
                            Text("Localhost Only (127.0.0.1)")
                                .font(.headline)
                            Text("Server is only accessible from this machine. More secure.")
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                    } icon: {
                        Image(systemName: "lock.shield")
                            .foregroundColor(.green)
                    }

                    Label {
                        VStack(alignment: .leading) {
                            Text("LAN Access (0.0.0.0)")
                                .font(.headline)
                            Text("Server is accessible from other devices on your local network.")
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                    } icon: {
                        Image(systemName: "network")
                            .foregroundColor(.blue)
                    }
                }
            } header: {
                Text("About Network Modes")
            }
        }
        .padding()
    }

    // MARK: - Notifications Tab

    private var notificationsTab: some View {
        Form {
            Section {
                Toggle("Show Notifications", isOn: $settings.showNotifications)
                Text("Show macOS notifications for server status changes and trading events.")
                    .font(.caption)
                    .foregroundColor(.secondary)
            } header: {
                Text("Notifications")
            }

            Section {
                VStack(alignment: .leading, spacing: 8) {
                    NotificationExample(
                        title: "Alpha Arena Started",
                        message: "Server is running at http://127.0.0.1:8000"
                    )

                    NotificationExample(
                        title: "Kill Switch Activated",
                        message: "Trading stopped due to loss limit"
                    )
                }
            } header: {
                Text("Example Notifications")
            }
        }
        .padding()
    }

    // MARK: - About Tab

    private var aboutTab: some View {
        VStack(spacing: 16) {
            Image(systemName: "chart.line.uptrend.xyaxis.circle.fill")
                .font(.system(size: 64))
                .foregroundStyle(.green, .green.opacity(0.3))

            Text("Alpha Arena")
                .font(.title)
                .fontWeight(.bold)

            Text("Version 1.0.0")
                .foregroundColor(.secondary)

            Text("Menu bar controller for Alpha Arena trading system")
                .font(.caption)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)

            Spacer()

            HStack(spacing: 20) {
                Button("View Logs") {
                    openLogs()
                }

                Button("Open Dashboard") {
                    openDashboard()
                }
                .buttonStyle(.borderedProminent)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .padding()
    }

    // MARK: - Helper Methods

    private func openLogs() {
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

    private func openDashboard() {
        let host = settings.networkMode == .lan ? (getLocalIPAddress() ?? "127.0.0.1") : "127.0.0.1"
        if let url = URL(string: "http://\(host):\(settings.serverPort)") {
            NSWorkspace.shared.open(url)
        }
    }
}

// MARK: - Notification Example View

struct NotificationExample: View {
    let title: String
    let message: String

    var body: some View {
        HStack(alignment: .top, spacing: 12) {
            Image(systemName: "app.badge")
                .font(.title2)
                .foregroundColor(.accentColor)

            VStack(alignment: .leading, spacing: 2) {
                Text(title)
                    .font(.caption)
                    .fontWeight(.semibold)
                Text(message)
                    .font(.caption2)
                    .foregroundColor(.secondary)
            }
        }
        .padding(8)
        .background(Color(NSColor.controlBackgroundColor))
        .cornerRadius(8)
    }
}

// MARK: - Provider Row

struct ProviderRow: View {
    let provider: LLMProviderType
    let isActive: Bool
    let isConnected: Bool
    let healthStatus: HealthStatus
    let onSelect: () -> Void

    var body: some View {
        HStack {
            Image(systemName: provider.icon)
                .foregroundColor(isConnected ? .accentColor : .secondary)
                .frame(width: 20)

            VStack(alignment: .leading, spacing: 2) {
                Text(provider.displayName)
                    .fontWeight(isActive ? .semibold : .regular)
                Text(provider.description)
                    .font(.caption2)
                    .foregroundColor(.secondary)
            }

            Spacer()

            // Health indicator
            Circle()
                .fill(healthColor)
                .frame(width: 8, height: 8)

            // Status badge
            if isActive {
                Text("Active")
                    .font(.caption2)
                    .padding(.horizontal, 6)
                    .padding(.vertical, 2)
                    .background(Color.accentColor.opacity(0.2))
                    .foregroundColor(.accentColor)
                    .cornerRadius(4)
            } else if isConnected {
                Button("Select") {
                    onSelect()
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
            } else {
                Text("Not Available")
                    .font(.caption2)
                    .foregroundColor(.secondary)
            }
        }
        .padding(.vertical, 4)
    }

    private var healthColor: Color {
        switch healthStatus {
        case .healthy: return .green
        case .degraded: return .yellow
        case .unhealthy: return .red
        case .unknown: return .gray
        }
    }
}

#Preview {
    PreferencesView()
}
