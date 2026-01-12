import SwiftUI

/// Menu bar status icon view
struct StatusItemView: View {
    @EnvironmentObject var service: AlphaArenaService

    var body: some View {
        Image(systemName: iconName)
            .symbolRenderingMode(.palette)
            .foregroundStyle(iconColor, .clear)
    }

    private var iconName: String {
        switch service.serverState.status {
        case .running:
            return service.tradingStatus.isActive ? "circle.fill" : "circle.lefthalf.filled"
        case .stopped:
            return "circle"
        case .starting, .stopping:
            return "circle.dotted"
        case .error:
            return "exclamationmark.circle.fill"
        }
    }

    private var iconColor: Color {
        switch service.serverState.status {
        case .running:
            return service.tradingStatus.isActive ? .green : .green.opacity(0.7)
        case .stopped:
            return .gray
        case .starting, .stopping:
            return .orange
        case .error:
            return .red
        }
    }
}

#Preview {
    StatusItemView()
        .environmentObject(AlphaArenaService())
}
