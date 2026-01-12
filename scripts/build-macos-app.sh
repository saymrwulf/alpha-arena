#!/bin/bash
#
# Build Alpha Arena.app for macOS
#
# This script:
# 1. Generates icon assets (if Pillow is available)
# 2. Builds the .app bundle using py2app
# 3. Optionally installs to /Applications
#
# Usage:
#   ./scripts/build-macos-app.sh          Build only
#   ./scripts/build-macos-app.sh install  Build and install to /Applications
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo "========================================"
echo "  Alpha Arena macOS App Builder"
echo "========================================"
echo ""

# Check for virtual environment
if [ ! -d ".venv" ]; then
    echo "Error: Virtual environment not found."
    echo "Run ./scripts/setup.sh first."
    exit 1
fi

# Activate venv
source .venv/bin/activate

# Check for py2app
if ! python -c "import py2app" 2>/dev/null; then
    echo "Installing py2app..."
    pip install py2app
fi

# Check for rumps
if ! python -c "import rumps" 2>/dev/null; then
    echo "Installing rumps and other macOS dependencies..."
    pip install rumps pyobjc-framework-Cocoa pyobjc-framework-UserNotifications websocket-client
fi

# Generate icons
echo ""
echo "Step 1: Generating icons..."
if python -c "from PIL import Image" 2>/dev/null; then
    python scripts/generate-icons.py
else
    echo "Pillow not installed. Installing..."
    pip install Pillow
    python scripts/generate-icons.py
fi

# Clean previous builds
echo ""
echo "Step 2: Cleaning previous builds..."
rm -rf build dist

# Build the app
echo ""
echo "Step 3: Building Alpha Arena.app..."
python scripts/macos-app-setup.py py2app

# Check if build succeeded
APP_PATH="dist/Alpha Arena.app"
if [ -d "$APP_PATH" ]; then
    echo ""
    echo "========================================"
    echo "  Build Successful!"
    echo "========================================"
    echo ""
    echo "App location: $PROJECT_DIR/$APP_PATH"
    echo ""

    # Install if requested
    if [ "$1" == "install" ]; then
        echo "Installing to /Applications..."
        rm -rf "/Applications/Alpha Arena.app"
        cp -r "$APP_PATH" /Applications/
        echo "Installed to /Applications/Alpha Arena.app"
        echo ""
        echo "You can now launch Alpha Arena from:"
        echo "  - Spotlight (Cmd+Space, type 'Alpha Arena')"
        echo "  - Applications folder"
        echo "  - Launchpad"
    else
        echo "To install, run:"
        echo "  cp -r '$APP_PATH' /Applications/"
        echo ""
        echo "Or run this script with 'install' argument:"
        echo "  ./scripts/build-macos-app.sh install"
    fi
else
    echo ""
    echo "Build failed! Check the output above for errors."
    exit 1
fi

echo ""
echo "Done!"
