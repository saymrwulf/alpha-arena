import SwiftUI

/// Dashboard summary view for the menu bar
struct DashboardView: View {
    @EnvironmentObject var service: AlphaArenaService

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            // P&L Card
            pnlCard

            // Positions Card
            positionsCard

            // Trading Status Card
            tradingCard
        }
        .padding(12)
        .background(Color(NSColor.controlBackgroundColor))
        .cornerRadius(8)
    }

    private var pnlCard: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Image(systemName: "chart.line.uptrend.xyaxis")
                    .foregroundColor(.secondary)
                Text("P&L")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }

            HStack(alignment: .firstTextBaseline) {
                Text(service.tradingStatus.pnlFormatted)
                    .font(.system(.title2, design: .rounded, weight: .semibold))
                    .foregroundColor(service.tradingStatus.totalPnL >= 0 ? .green : .red)

                Spacer()

                if service.tradingStatus.dailyPnL != 0 {
                    VStack(alignment: .trailing) {
                        Text("Today")
                            .font(.caption2)
                            .foregroundColor(.secondary)
                        Text(service.tradingStatus.dailyPnLFormatted)
                            .font(.caption)
                            .foregroundColor(service.tradingStatus.dailyPnL >= 0 ? .green : .red)
                    }
                }
            }
        }
    }

    private var positionsCard: some View {
        HStack {
            Image(systemName: "square.stack.3d.up")
                .foregroundColor(.secondary)
            Text("Open Positions")
                .font(.caption)
                .foregroundColor(.secondary)

            Spacer()

            Text("\(service.tradingStatus.openPositions)")
                .font(.system(.body, design: .rounded, weight: .medium))
        }
    }

    private var tradingCard: some View {
        HStack {
            Image(systemName: service.tradingStatus.isActive ? "play.circle.fill" : "stop.circle")
                .foregroundColor(service.tradingStatus.isActive ? .green : .secondary)
            Text("Trading")
                .font(.caption)
                .foregroundColor(.secondary)

            Spacer()

            HStack(spacing: 4) {
                Circle()
                    .fill(service.tradingStatus.isActive ? Color.green : Color.gray)
                    .frame(width: 8, height: 8)
                Text(service.tradingStatus.isActive ? "Active" : "Stopped")
                    .font(.caption)
                    .foregroundColor(service.tradingStatus.isActive ? .primary : .secondary)
            }

            if service.tradingStatus.isActive {
                Text("(\(service.tradingStatus.mode.rawValue))")
                    .font(.caption2)
                    .foregroundColor(.secondary)
            }
        }
    }
}

#Preview {
    DashboardView()
        .environmentObject(AlphaArenaService())
        .frame(width: 260)
}
