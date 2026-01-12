"""
py2app setup configuration for Alpha Arena Menu Bar App.

This creates a standalone .app bundle that can be placed in /Applications.

Usage:
    python scripts/macos-app-setup.py py2app
"""

from setuptools import setup
from pathlib import Path

# Project paths
PROJECT_DIR = Path(__file__).parent.parent
SRC_DIR = PROJECT_DIR / "src"
MACOS_DIR = SRC_DIR / "macos"

# Main entry point
APP = [str(MACOS_DIR / "menubar.py")]

# Data files to include
DATA_FILES = []

# Check for icon files
icons_dir = MACOS_DIR / "icons"
if icons_dir.exists():
    icon_files = list(icons_dir.glob("*.png"))
    if icon_files:
        DATA_FILES.append(("icons", [str(f) for f in icon_files]))

# Check for .icns app icon
icns_file = MACOS_DIR / "resources" / "Alpha Arena.icns"
if not icns_file.exists():
    icns_file = None

# py2app options
OPTIONS = {
    'argv_emulation': False,
    'plist': {
        'CFBundleName': 'Alpha Arena',
        'CFBundleDisplayName': 'Alpha Arena',
        'CFBundleIdentifier': 'com.alpha-arena.controller',
        'CFBundleVersion': '1.0.0',
        'CFBundleShortVersionString': '1.0.0',
        'LSMinimumSystemVersion': '12.0',
        'LSUIElement': True,  # Menu bar app - no dock icon
        'LSBackgroundOnly': False,
        'NSHighResolutionCapable': True,
        'NSUserNotificationAlertStyle': 'alert',
        'NSAppleEventsUsageDescription': 'Alpha Arena needs to send notifications.',
    },
    'packages': [
        'rumps',
        'httpx',
        'websocket',
        'certifi',
    ],
    'includes': [
        'src.macos',
        'src.macos.server_bridge',
        'src.macos.notifications',
        'src.macos.menubar',
    ],
    'excludes': [
        'tkinter',
        'matplotlib',
        'numpy',
        'pandas',
        'scipy',
    ],
    'resources': [],
    'frameworks': [],
}

# Add icon if exists
if icns_file:
    OPTIONS['iconfile'] = str(icns_file)

setup(
    name='Alpha Arena',
    version='1.0.0',
    description='Alpha Arena Menu Bar Controller',
    author='Alpha Arena',
    app=APP,
    data_files=DATA_FILES,
    options={'py2app': OPTIONS},
    setup_requires=['py2app'],
)
