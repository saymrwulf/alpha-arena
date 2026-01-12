import SwiftUI
import AppKit

/// Main menu bar content view
struct MenuBarView: View {
    @EnvironmentObject var service: AlphaArenaService

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            // Header
            headerSection

            Divider()
                .padding(.vertical, 4)

            // Dashboard summary
            if service.serverState.isRunning {
                dashboardSection

                Divider()
                    .padding(.vertical, 4)
            }

            // Server controls
            serverControlsSection

            Divider()
                .padding(.vertical, 4)

            // Quick actions
            quickActionsSection

            Divider()
                .padding(.vertical, 4)

            // Network mode
            networkModeSection

            Divider()
                .padding(.vertical, 4)

            // Footer
            footerSection
        }
        .padding(.vertical, 8)
        .frame(width: 280)
    }

    // MARK: - Header Section

    private var headerSection: some View {
        HStack {
            Circle()
                .fill(statusColor)
                .frame(width: 10, height: 10)

            Text(statusText)
                .font(.headline)

            Spacer()

            if service.isLoading {
                ProgressView()
                    .scaleEffect(0.6)
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 4)
    }

    private var statusColor: Color {
        switch service.serverState.status {
        case .running:
            return .green
        case .stopped:
            return .gray
        case .starting, .stopping:
            return .orange
        case .error:
            return .red
        }
    }

    private var statusText: String {
        switch service.serverState.status {
        case .running:
            return "Server Running"
        case .stopped:
            return "Server Stopped"
        case .starting:
            return "Starting..."
        case .stopping:
            return "Stopping..."
        case .error:
            return "Error"
        }
    }

    // MARK: - Dashboard Section

    private var dashboardSection: some View {
        VStack(alignment: .leading, spacing: 6) {
            // URL
            HStack {
                Image(systemName: "link")
                    .foregroundColor(.secondary)
                    .frame(width: 16)
                Text(service.serverState.displayURL)
                    .font(.caption)
                    .foregroundColor(.secondary)
            }

            // P&L
            HStack {
                Image(systemName: "chart.line.uptrend.xyaxis")
                    .foregroundColor(.secondary)
                    .frame(width: 16)
                Text("P&L: \(service.tradingStatus.pnlFormatted)")
                    .font(.system(.caption, design: .monospaced))
                    .foregroundColor(service.tradingStatus.totalPnL >= 0 ? .green : .red)

                Spacer()

                Text("\(service.tradingStatus.openPositions) positions")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }

            // Trading status
            HStack {
                Image(systemName: service.tradingStatus.isActive ? "play.fill" : "stop.fill")
                    .foregroundColor(service.tradingStatus.isActive ? .green : .secondary)
                    .frame(width: 16)
                Text(service.tradingStatus.statusText)
                    .font(.caption)
                    .foregroundColor(service.tradingStatus.isActive ? .primary : .secondary)

                if service.tradingStatus.isActive {
                    Text("(\(service.tradingStatus.mode.rawValue))")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
            }
        }
        .padding(.horizontal, 12)
    }

    // MARK: - Server Controls Section

    private var serverControlsSection: some View {
        VStack(spacing: 2) {
            if service.serverState.isRunning {
                MenuButton(
                    title: "Stop Server",
                    icon: "stop.fill",
                    shortcut: "S",
                    modifiers: [.command, .shift]
                ) {
                    Task {
                        await service.stopServer()
                    }
                }
                .disabled(service.isLoading)
            } else {
                MenuButton(
                    title: "Start Server",
                    icon: "play.fill",
                    shortcut: "S",
                    modifiers: [.command]
                ) {
                    Task {
                        await service.startServer()
                    }
                }
                .disabled(service.isLoading)
            }
        }
    }

    // MARK: - Quick Actions Section

    private var quickActionsSection: some View {
        VStack(spacing: 2) {
            MenuButton(
                title: "Open Dashboard",
                icon: "globe",
                shortcut: "D",
                modifiers: [.command]
            ) {
                service.openDashboard()
            }
            .disabled(!service.serverState.isRunning)

            MenuButton(
                title: "Copy Server URL",
                icon: "doc.on.doc",
                shortcut: "C",
                modifiers: [.command]
            ) {
                service.copyURL()
            }

            MenuButton(
                title: "View Logs",
                icon: "terminal",
                shortcut: "L",
                modifiers: [.command]
            ) {
                service.openLogs()
            }
        }
    }

    // MARK: - Network Mode Section

    private var networkModeSection: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text("Network Mode")
                .font(.caption)
                .foregroundColor(.secondary)
                .padding(.horizontal, 12)
                .padding(.bottom, 4)

            ForEach(NetworkMode.allCases, id: \.rawValue) { mode in
                NetworkModeButton(
                    mode: mode,
                    isSelected: service.settings.networkMode == mode
                ) {
                    service.settings.networkMode = mode
                }
            }
        }
    }

    // MARK: - Footer Section

    private var footerSection: some View {
        VStack(spacing: 2) {
            MenuButton(
                title: "Preferences...",
                icon: "gear",
                shortcut: ",",
                modifiers: [.command]
            ) {
                openPreferences()
            }

            Divider()
                .padding(.vertical, 4)

            MenuButton(
                title: "Quit Alpha Arena",
                icon: "power",
                shortcut: "Q",
                modifiers: [.command]
            ) {
                NSApplication.shared.terminate(nil)
            }
        }
    }
}

// MARK: - Menu Button Component

struct MenuButton: View {
    let title: String
    let icon: String
    var shortcut: String? = nil
    var modifiers: EventModifiers = []
    let action: () -> Void

    @State private var isHovering = false

    var body: some View {
        Button(action: action) {
            HStack {
                Image(systemName: icon)
                    .frame(width: 20)
                Text(title)
                Spacer()
                if let shortcut = shortcut {
                    Text(shortcutText)
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 6)
            .background(isHovering ? Color.accentColor.opacity(0.1) : Color.clear)
            .cornerRadius(4)
        }
        .buttonStyle(.plain)
        .onHover { hovering in
            isHovering = hovering
        }
    }

    private var shortcutText: String {
        var parts: [String] = []
        if modifiers.contains(.command) { parts.append("⌘") }
        if modifiers.contains(.shift) { parts.append("⇧") }
        if modifiers.contains(.option) { parts.append("⌥") }
        if modifiers.contains(.control) { parts.append("⌃") }
        if let shortcut = shortcut { parts.append(shortcut) }
        return parts.joined()
    }
}

// MARK: - Network Mode Button

struct NetworkModeButton: View {
    let mode: NetworkMode
    let isSelected: Bool
    let action: () -> Void

    @State private var isHovering = false

    var body: some View {
        Button(action: action) {
            HStack {
                Image(systemName: isSelected ? "checkmark.circle.fill" : "circle")
                    .foregroundColor(isSelected ? .accentColor : .secondary)
                    .frame(width: 20)

                VStack(alignment: .leading, spacing: 1) {
                    Text(mode.displayName)
                        .font(.caption)
                    Text(mode.rawValue)
                        .font(.caption2)
                        .foregroundColor(.secondary)
                }

                Spacer()
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 4)
            .background(isHovering ? Color.accentColor.opacity(0.1) : Color.clear)
            .cornerRadius(4)
        }
        .buttonStyle(.plain)
        .onHover { hovering in
            isHovering = hovering
        }
    }
}

// MARK: - Preferences Helper

private func openPreferences() {
    // Close the menu bar popover first
    NSApp.keyWindow?.close()

    // Small delay to allow menu to close, then open preferences window
    DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) {
        PreferencesWindowController.shared.showPreferences()
    }
}

#Preview {
    MenuBarView()
        .environmentObject(AlphaArenaService())
}
