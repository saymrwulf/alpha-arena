# Alpha Arena Master Control Web Application
"""
Comprehensive web-based control center for the Polymarket Trading Harness.
Provides operator and supervisor interfaces for complete system management.
"""

from .app import create_app, app

__all__ = ["create_app", "app"]
