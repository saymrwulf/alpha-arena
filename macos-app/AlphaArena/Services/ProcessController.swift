import Foundation
import AppKit

/// Errors that can occur during process control
enum ProcessError: Error, LocalizedError {
    case projectNotFound
    case scriptNotFound
    case executionFailed(String)
    case timeout

    var errorDescription: String? {
        switch self {
        case .projectNotFound:
            return "Alpha Arena project directory not found"
        case .scriptNotFound:
            return "The 'alpha' script was not found in the project directory"
        case .executionFailed(let message):
            return "Command failed: \(message)"
        case .timeout:
            return "Command timed out"
        }
    }
}

/// Controls the Alpha Arena server process via the ./alpha script
actor ProcessController {
    private let settings: AppSettings

    init(settings: AppSettings = .shared) {
        self.settings = settings
    }

    /// Start the server
    /// - Parameter lanAccess: If true, bind to 0.0.0.0 for LAN access
    func startServer(lanAccess: Bool = false) async throws {
        var environment = ProcessInfo.processInfo.environment
        if lanAccess {
            environment["ALPHA_HOST"] = "0.0.0.0"
        }
        _ = try await runAlphaScript("start", environment: environment)
    }

    /// Stop the server
    func stopServer() async throws {
        _ = try await runAlphaScript("stop")
    }

    /// Get server status
    func getStatus() async throws -> ServerStatus {
        let output = try await runAlphaScript("status")
        return parseStatus(output)
    }

    /// Open logs in Terminal
    func openLogs() async throws {
        let projectPath = await MainActor.run { settings.projectPath }

        let script = """
        tell application "Terminal"
            do script "cd '\(projectPath)' && ./alpha logs"
            activate
        end tell
        """

        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/osascript")
        process.arguments = ["-e", script]

        try process.run()
        process.waitUntilExit()
    }

    /// Open dashboard in default browser
    func openDashboard() async {
        let port = await MainActor.run { settings.serverPort }
        let host = await MainActor.run { settings.networkMode == .lan ? (getLocalIPAddress() ?? "127.0.0.1") : "127.0.0.1" }

        if let url = URL(string: "http://\(host):\(port)") {
            await MainActor.run {
                NSWorkspace.shared.open(url)
            }
        }
    }

    /// Copy server URL to clipboard
    func copyURL() async {
        let port = await MainActor.run { settings.serverPort }
        let host = await MainActor.run { settings.networkMode == .lan ? (getLocalIPAddress() ?? "127.0.0.1") : "127.0.0.1" }
        let url = "http://\(host):\(port)"

        await MainActor.run {
            NSPasteboard.general.clearContents()
            NSPasteboard.general.setString(url, forType: .string)
        }
    }

    // MARK: - Private Methods

    private func runAlphaScript(_ command: String, environment: [String: String]? = nil) async throws -> String {
        let projectPath = await MainActor.run { settings.projectPath }

        guard !projectPath.isEmpty else {
            throw ProcessError.projectNotFound
        }

        let scriptPath = projectPath + "/alpha"
        guard FileManager.default.fileExists(atPath: scriptPath) else {
            throw ProcessError.scriptNotFound
        }

        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/bin/bash")
        process.arguments = ["-c", "cd '\(projectPath)' && ./alpha \(command)"]
        process.currentDirectoryURL = URL(fileURLWithPath: projectPath)

        if let env = environment {
            var currentEnv = ProcessInfo.processInfo.environment
            for (key, value) in env {
                currentEnv[key] = value
            }
            process.environment = currentEnv
        }

        let outputPipe = Pipe()
        let errorPipe = Pipe()
        process.standardOutput = outputPipe
        process.standardError = errorPipe

        try process.run()

        // Wait with timeout
        let timeout = DispatchTime.now() + .seconds(30)
        let semaphore = DispatchSemaphore(value: 0)

        DispatchQueue.global().async {
            process.waitUntilExit()
            semaphore.signal()
        }

        if semaphore.wait(timeout: timeout) == .timedOut {
            process.terminate()
            throw ProcessError.timeout
        }

        let outputData = outputPipe.fileHandleForReading.readDataToEndOfFile()
        let errorData = errorPipe.fileHandleForReading.readDataToEndOfFile()

        let output = String(data: outputData, encoding: .utf8) ?? ""
        let errorOutput = String(data: errorData, encoding: .utf8) ?? ""

        if process.terminationStatus != 0 && !errorOutput.isEmpty {
            throw ProcessError.executionFailed(errorOutput)
        }

        return output
    }

    private func parseStatus(_ output: String) -> ServerStatus {
        let lowercased = output.lowercased()

        // Check "not running" BEFORE "running" since "not running" contains "running"
        if lowercased.contains("not running") || lowercased.contains("stopped") {
            return .stopped
        } else if lowercased.contains("running") || lowercased.contains("server is running") {
            return .running
        } else if lowercased.contains("starting") {
            return .starting
        } else if lowercased.contains("error") {
            return .error
        }

        return .stopped
    }
}
