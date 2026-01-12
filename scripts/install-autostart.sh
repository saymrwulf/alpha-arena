#!/bin/bash
#
# Install Alpha Arena as a macOS Launch Agent (auto-start at login)
#
# Usage:
#   ./scripts/install-autostart.sh          Install auto-start
#   ./scripts/install-autostart.sh remove   Remove auto-start
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

PLIST_NAME="com.alpha-arena.server"
PLIST_PATH="$HOME/Library/LaunchAgents/${PLIST_NAME}.plist"

install_launchd() {
    echo "Installing Alpha Arena auto-start..."

    # Create LaunchAgents directory if needed
    mkdir -p "$HOME/Library/LaunchAgents"

    # Create plist
    cat > "$PLIST_PATH" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>${PLIST_NAME}</string>
    <key>ProgramArguments</key>
    <array>
        <string>${PROJECT_DIR}/alpha</string>
        <string>start</string>
    </array>
    <key>WorkingDirectory</key>
    <string>${PROJECT_DIR}</string>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <false/>
    <key>StandardOutPath</key>
    <string>${PROJECT_DIR}/data/logs/launchd.log</string>
    <key>StandardErrorPath</key>
    <string>${PROJECT_DIR}/data/logs/launchd.log</string>
</dict>
</plist>
EOF

    # Load the agent
    launchctl unload "$PLIST_PATH" 2>/dev/null || true
    launchctl load "$PLIST_PATH"

    echo "✓ Installed: $PLIST_PATH"
    echo ""
    echo "Alpha Arena will now start automatically when you log in."
    echo ""
    echo "To remove auto-start:"
    echo "  ./scripts/install-autostart.sh remove"
}

remove_launchd() {
    if [ -f "$PLIST_PATH" ]; then
        echo "Removing Alpha Arena auto-start..."
        launchctl unload "$PLIST_PATH" 2>/dev/null || true
        rm -f "$PLIST_PATH"
        echo "✓ Removed"
    else
        echo "Auto-start not installed"
    fi
}

case "${1:-}" in
    remove|uninstall)
        remove_launchd
        ;;
    *)
        install_launchd
        ;;
esac
