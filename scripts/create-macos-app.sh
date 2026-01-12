#!/bin/bash
#
# Create Alpha Arena.app - Simple wrapper approach
#
# This creates a minimal .app bundle that launches the Python menu bar app.
# Works without py2app by using a shell script as the executable.
#
# Usage:
#   ./scripts/create-macos-app.sh          Create in dist/
#   ./scripts/create-macos-app.sh install  Create and install to /Applications
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

APP_NAME="Alpha Arena"
BUNDLE_ID="com.alpha-arena.controller"
VERSION="1.0.0"

# Output locations
DIST_DIR="$PROJECT_DIR/dist"
APP_PATH="$DIST_DIR/$APP_NAME.app"

echo "========================================"
echo "  Creating $APP_NAME.app"
echo "========================================"
echo ""

# Clean previous builds
rm -rf "$APP_PATH"
mkdir -p "$DIST_DIR"

# Create app bundle structure
mkdir -p "$APP_PATH/Contents/MacOS"
mkdir -p "$APP_PATH/Contents/Resources"

# Create Info.plist
cat > "$APP_PATH/Contents/Info.plist" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key>
    <string>$APP_NAME</string>
    <key>CFBundleDisplayName</key>
    <string>$APP_NAME</string>
    <key>CFBundleIdentifier</key>
    <string>$BUNDLE_ID</string>
    <key>CFBundleVersion</key>
    <string>$VERSION</string>
    <key>CFBundleShortVersionString</key>
    <string>$VERSION</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleExecutable</key>
    <string>launcher</string>
    <key>LSMinimumSystemVersion</key>
    <string>12.0</string>
    <key>LSUIElement</key>
    <true/>
    <key>NSHighResolutionCapable</key>
    <true/>
    <key>NSUserNotificationAlertStyle</key>
    <string>alert</string>
</dict>
</plist>
EOF

# Create launcher script
cat > "$APP_PATH/Contents/MacOS/launcher" << 'LAUNCHER'
#!/bin/bash
#
# Alpha Arena Menu Bar Launcher
#

# Find the project directory
# The app stores the project path in a config file
CONFIG_FILE="$HOME/.alpha-arena-path"

if [ -f "$CONFIG_FILE" ]; then
    PROJECT_DIR=$(cat "$CONFIG_FILE")
else
    # Default locations to search
    SEARCH_PATHS=(
        "$HOME/GitClone/ClaudeCodeProjects/alpha-arena"
        "$HOME/Projects/alpha-arena"
        "$HOME/alpha-arena"
        "/opt/alpha-arena"
    )

    for path in "${SEARCH_PATHS[@]}"; do
        if [ -f "$path/alpha" ]; then
            PROJECT_DIR="$path"
            echo "$PROJECT_DIR" > "$CONFIG_FILE"
            break
        fi
    done
fi

if [ -z "$PROJECT_DIR" ] || [ ! -f "$PROJECT_DIR/alpha" ]; then
    osascript -e 'display alert "Alpha Arena" message "Could not find Alpha Arena project directory. Please run the app from the project folder first."'
    exit 1
fi

cd "$PROJECT_DIR"

# Ensure virtual environment exists
if [ ! -d ".venv" ]; then
    osascript -e 'display alert "Alpha Arena" message "Virtual environment not found. Please run ./scripts/setup.sh first."'
    exit 1
fi

# Activate and run
source .venv/bin/activate
exec python -m src.macos.menubar
LAUNCHER

chmod +x "$APP_PATH/Contents/MacOS/launcher"

# Copy icons if they exist
ICONS_DIR="$PROJECT_DIR/src/macos/icons"
if [ -d "$ICONS_DIR" ]; then
    cp -r "$ICONS_DIR"/* "$APP_PATH/Contents/Resources/" 2>/dev/null || true
fi

# Create a simple app icon (colored circle) if iconutil is available
# For now, we'll skip the .icns creation as it requires more setup

echo "Created: $APP_PATH"
echo ""

# Store the project path for the launcher
echo "$PROJECT_DIR" > "$HOME/.alpha-arena-path"
echo "Saved project path to ~/.alpha-arena-path"
echo ""

# Install if requested
if [ "$1" == "install" ]; then
    echo "Installing to /Applications..."
    rm -rf "/Applications/$APP_NAME.app"
    cp -r "$APP_PATH" /Applications/
    echo ""
    echo "========================================"
    echo "  Installation Complete!"
    echo "========================================"
    echo ""
    echo "You can now launch Alpha Arena from:"
    echo "  - Spotlight (Cmd+Space, type 'Alpha Arena')"
    echo "  - Applications folder"
    echo "  - Launchpad"
    echo ""
else
    echo "To install to /Applications, run:"
    echo "  cp -r '$APP_PATH' /Applications/"
    echo ""
    echo "Or run this script with 'install':"
    echo "  ./scripts/create-macos-app.sh install"
fi

echo "Done!"
