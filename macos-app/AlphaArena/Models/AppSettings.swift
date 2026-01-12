import Foundation

/// Network mode for server binding
enum NetworkMode: String, CaseIterable {
    case localhost = "127.0.0.1"
    case lan = "0.0.0.0"

    var displayName: String {
        switch self {
        case .localhost: return "Localhost Only"
        case .lan: return "LAN Access"
        }
    }

    var description: String {
        switch self {
        case .localhost: return "Only accessible from this machine"
        case .lan: return "Accessible from other devices on network"
        }
    }
}

/// Persistent app settings
class AppSettings: ObservableObject {
    static let shared = AppSettings()

    private let defaults = UserDefaults.standard

    private enum Keys {
        static let projectPath = "projectPath"
        static let networkMode = "networkMode"
        static let serverPort = "serverPort"
        static let autoConnectWebSocket = "autoConnectWebSocket"
        static let showNotifications = "showNotifications"
    }

    @Published var projectPath: String {
        didSet {
            defaults.set(projectPath, forKey: Keys.projectPath)
        }
    }

    @Published var networkMode: NetworkMode {
        didSet {
            defaults.set(networkMode.rawValue, forKey: Keys.networkMode)
        }
    }

    @Published var serverPort: Int {
        didSet {
            defaults.set(serverPort, forKey: Keys.serverPort)
        }
    }

    @Published var autoConnectWebSocket: Bool {
        didSet {
            defaults.set(autoConnectWebSocket, forKey: Keys.autoConnectWebSocket)
        }
    }

    @Published var showNotifications: Bool {
        didSet {
            defaults.set(showNotifications, forKey: Keys.showNotifications)
        }
    }

    var serverURL: String {
        "http://\(networkMode.rawValue):\(serverPort)"
    }

    var displayURL: String {
        let host = networkMode == .lan ? getLocalIPAddress() ?? "0.0.0.0" : "127.0.0.1"
        return "http://\(host):\(serverPort)"
    }

    private init() {
        // Load saved values or use defaults
        self.projectPath = defaults.string(forKey: Keys.projectPath) ?? Self.findDefaultProjectPath()
        self.networkMode = NetworkMode(rawValue: defaults.string(forKey: Keys.networkMode) ?? "") ?? .localhost
        self.serverPort = defaults.integer(forKey: Keys.serverPort) != 0 ? defaults.integer(forKey: Keys.serverPort) : 8000
        self.autoConnectWebSocket = defaults.object(forKey: Keys.autoConnectWebSocket) as? Bool ?? true
        self.showNotifications = defaults.object(forKey: Keys.showNotifications) as? Bool ?? true
    }

    /// Find the default project path by checking common locations
    private static func findDefaultProjectPath() -> String {
        // Check config file first
        let configPath = NSHomeDirectory() + "/.alpha-arena-path"
        if let savedPath = try? String(contentsOfFile: configPath, encoding: .utf8).trimmingCharacters(in: .whitespacesAndNewlines),
           FileManager.default.fileExists(atPath: savedPath + "/alpha") {
            return savedPath
        }

        // Search common locations
        let searchPaths = [
            NSHomeDirectory() + "/GitClone/ClaudeCodeProjects/alpha-arena",
            NSHomeDirectory() + "/Projects/alpha-arena",
            NSHomeDirectory() + "/alpha-arena",
            "/opt/alpha-arena"
        ]

        for path in searchPaths {
            if FileManager.default.fileExists(atPath: path + "/alpha") {
                // Save for future use
                try? path.write(toFile: configPath, atomically: true, encoding: .utf8)
                return path
            }
        }

        return ""
    }

    /// Save project path to config file for other tools
    func saveProjectPath() {
        let configPath = NSHomeDirectory() + "/.alpha-arena-path"
        try? projectPath.write(toFile: configPath, atomically: true, encoding: .utf8)
    }
}

/// Get local IP address for LAN mode display
func getLocalIPAddress() -> String? {
    var address: String?
    var ifaddr: UnsafeMutablePointer<ifaddrs>?

    if getifaddrs(&ifaddr) == 0 {
        var ptr = ifaddr
        while ptr != nil {
            defer { ptr = ptr?.pointee.ifa_next }

            guard let interface = ptr?.pointee else { continue }
            let addrFamily = interface.ifa_addr.pointee.sa_family

            if addrFamily == UInt8(AF_INET) {
                let name = String(cString: interface.ifa_name)
                if name == "en0" || name == "en1" {
                    var hostname = [CChar](repeating: 0, count: Int(NI_MAXHOST))
                    getnameinfo(interface.ifa_addr, socklen_t(interface.ifa_addr.pointee.sa_len),
                               &hostname, socklen_t(hostname.count),
                               nil, socklen_t(0), NI_NUMERICHOST)
                    address = String(cString: hostname)
                }
            }
        }
        freeifaddrs(ifaddr)
    }

    return address
}
