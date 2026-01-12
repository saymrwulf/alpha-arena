import SwiftUI

/// Alpha Arena Menu Bar Application
@main
struct AlphaArenaApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) var appDelegate
    @StateObject private var service = AlphaArenaService()

    var body: some Scene {
        // Menu bar extra - the main interface
        MenuBarExtra {
            MenuBarView()
                .environmentObject(service)
        } label: {
            StatusItemView()
                .environmentObject(service)
        }
        .menuBarExtraStyle(.window)

        // Settings window
        Settings {
            PreferencesView()
        }
    }
}
