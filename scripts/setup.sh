#!/bin/bash
# Alpha Arena - Setup Script
# This script sets up the isolated Python environment and installs all dependencies

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo "=============================================="
echo "  Alpha Arena - Environment Setup"
echo "=============================================="
echo ""

# Check Python version
PYTHON_CMD=""
for cmd in python3.14 python3.13 python3.12 python3.11 python3; do
    if command -v "$cmd" &> /dev/null; then
        version=$("$cmd" --version 2>&1 | grep -oE '[0-9]+\.[0-9]+' | head -1)
        major=$(echo "$version" | cut -d. -f1)
        minor=$(echo "$version" | cut -d. -f2)
        if [ "$major" -ge 3 ] && [ "$minor" -ge 11 ]; then
            PYTHON_CMD="$cmd"
            break
        fi
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    echo "ERROR: Python 3.11+ is required but not found"
    exit 1
fi

echo "Using Python: $PYTHON_CMD ($($PYTHON_CMD --version))"
echo ""

# Create or verify virtual environment
if [ -d ".venv" ]; then
    echo "Virtual environment exists at .venv"
    # Verify it's valid
    if [ ! -f ".venv/bin/python" ]; then
        echo "Virtual environment is corrupted, recreating..."
        rm -rf .venv
        "$PYTHON_CMD" -m venv .venv
    fi
else
    echo "Creating virtual environment..."
    "$PYTHON_CMD" -m venv .venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt

# Create .env if it doesn't exist
if [ ! -f ".env" ]; then
    echo ""
    echo "Creating .env file from template..."
    if [ -f ".env.example" ]; then
        cp .env.example .env
        echo "Created .env from .env.example"
    else
        cat > .env << 'EOF'
# Alpha Arena Configuration
# Copy this file to .env and fill in your values

# Polymarket Credentials (required for live trading)
POLYMARKET_API_KEY=
POLYMARKET_API_SECRET=
POLYMARKET_PASSPHRASE=
POLYGON_PRIVATE_KEY=

# LLM Provider API Keys (at least one required)
ANTHROPIC_API_KEY=
OPENAI_API_KEY=
XAI_API_KEY=

# Optional: Local LLM endpoint
LOCAL_LLM_URL=http://localhost:11434

# Web App Settings
WEB_HOST=127.0.0.1
WEB_PORT=8000
EOF
        echo "Created default .env file"
    fi
    echo ""
    echo "IMPORTANT: Edit .env with your API keys before running!"
fi

# Create data directories
echo ""
echo "Creating data directories..."
mkdir -p data/logs
mkdir -p data/memory
mkdir -p data/backtest

echo ""
echo "=============================================="
echo "  Setup Complete!"
echo "=============================================="
echo ""
echo "To activate the environment:"
echo "  source .venv/bin/activate"
echo ""
echo "To run the web application:"
echo "  ./scripts/run_web.sh"
echo ""
echo "Or manually:"
echo "  source .venv/bin/activate"
echo "  python -m uvicorn src.web.app:app --reload --port 8000"
echo ""
