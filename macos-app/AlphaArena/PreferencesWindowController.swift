import SwiftUI
import AppKit

/// Controller for the preferences window
class PreferencesWindowController {
    static let shared = PreferencesWindowController()

    private var window: NSWindow?

    private init() {}

    func showPreferences() {
        // If window exists, just bring it to front
        if let existingWindow = window, existingWindow.isVisible {
            existingWindow.makeKeyAndOrderFront(nil)
            NSApp.activate(ignoringOtherApps: true)
            return
        }

        // Create the SwiftUI view
        let preferencesView = PreferencesView()

        // Create hosting controller
        let hostingController = NSHostingController(rootView: preferencesView)

        // Create window
        let newWindow = NSWindow(contentViewController: hostingController)
        newWindow.title = "Alpha Arena Preferences"
        newWindow.styleMask = [.titled, .closable]
        newWindow.setContentSize(NSSize(width: 450, height: 320))
        newWindow.center()
        newWindow.isReleasedWhenClosed = false

        // Set minimum size
        newWindow.minSize = NSSize(width: 400, height: 280)

        // Handle window close to restore accessory policy
        NotificationCenter.default.addObserver(
            forName: NSWindow.willCloseNotification,
            object: newWindow,
            queue: .main
        ) { [weak self] _ in
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) {
                // Only hide dock if no other windows are visible
                let visibleWindows = NSApp.windows.filter {
                    $0.isVisible && $0 !== self?.window && !$0.title.isEmpty
                }
                if visibleWindows.isEmpty {
                    NSApp.setActivationPolicy(.accessory)
                }
            }
        }

        self.window = newWindow

        // Become regular app so window shows in dock/cmd-tab
        NSApp.setActivationPolicy(.regular)

        // Show window
        newWindow.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
    }

    func closePreferences() {
        window?.close()
    }
}
