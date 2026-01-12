#!/bin/bash
#
# Build Alpha Arena macOS Menu Bar App
#
# Requirements:
#   - Xcode 15.0+ installed (not just Command Line Tools)
#   - macOS 13.0+
#
# Usage:
#   ./build.sh          Build the app
#   ./build.sh install  Build and install to /Applications
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

APP_NAME="AlphaArena"
BUILD_DIR="$SCRIPT_DIR/build"
APP_PATH="$BUILD_DIR/Release/$APP_NAME.app"

echo "========================================"
echo "  Alpha Arena macOS App Builder"
echo "========================================"
echo ""

# Check for Xcode
if ! command -v xcodebuild &> /dev/null; then
    echo "Error: xcodebuild not found."
    echo ""
    echo "Please install Xcode from the App Store:"
    echo "  https://apps.apple.com/app/xcode/id497799835"
    echo ""
    echo "After installing, run:"
    echo "  sudo xcode-select -s /Applications/Xcode.app/Contents/Developer"
    exit 1
fi

# Check Xcode is properly selected
XCODE_PATH=$(xcode-select -p 2>/dev/null)
if [[ "$XCODE_PATH" == *"CommandLineTools"* ]]; then
    echo "Error: Xcode Command Line Tools selected instead of Xcode."
    echo ""
    echo "Please run:"
    echo "  sudo xcode-select -s /Applications/Xcode.app/Contents/Developer"
    exit 1
fi

echo "Using Xcode at: $XCODE_PATH"
echo ""

# Clean previous builds
echo "Step 1: Cleaning previous builds..."
rm -rf "$BUILD_DIR"

# Build the app
echo ""
echo "Step 2: Building $APP_NAME.app..."
cd "$SCRIPT_DIR"

xcodebuild \
    -project AlphaArena.xcodeproj \
    -scheme AlphaArena \
    -configuration Release \
    -derivedDataPath "$BUILD_DIR" \
    CONFIGURATION_BUILD_DIR="$BUILD_DIR/Release" \
    build

# Check if build succeeded
if [ -d "$APP_PATH" ]; then
    echo ""
    echo "========================================"
    echo "  Build Successful!"
    echo "========================================"
    echo ""
    echo "App location: $APP_PATH"
    echo ""

    # Store project path for the app
    echo "$PROJECT_DIR" > "$HOME/.alpha-arena-path"
    echo "Saved project path to ~/.alpha-arena-path"
    echo ""

    # Install if requested
    if [ "$1" == "install" ]; then
        echo "Installing to /Applications..."
        rm -rf "/Applications/$APP_NAME.app"
        cp -r "$APP_PATH" /Applications/
        echo ""
        echo "Installed to /Applications/$APP_NAME.app"
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
        echo "  ./build.sh install"
    fi
else
    echo ""
    echo "Build failed! Check the output above for errors."
    exit 1
fi

echo ""
echo "Done!"
