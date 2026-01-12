#!/bin/bash
# Alpha Arena - Run Web Application
# This script activates the virtual environment and starts the web server

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# Check if .venv exists
if [ ! -d ".venv" ]; then
    echo "ERROR: Virtual environment not found."
    echo "Please run ./scripts/setup.sh first"
    exit 1
fi

# Activate virtual environment
source .venv/bin/activate

# Check if dependencies are installed
if ! python -c "import fastapi" 2>/dev/null; then
    echo "ERROR: Dependencies not installed."
    echo "Please run ./scripts/setup.sh first"
    exit 1
fi

# Parse arguments
HOST="${WEB_HOST:-127.0.0.1}"
PORT="${WEB_PORT:-8000}"
RELOAD=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --host)
            HOST="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --reload)
            RELOAD="--reload"
            shift
            ;;
        --production)
            RELOAD=""
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --host HOST      Bind to host (default: 127.0.0.1)"
            echo "  --port PORT      Bind to port (default: 8000)"
            echo "  --reload         Enable auto-reload (development)"
            echo "  --production     Disable auto-reload"
            echo "  -h, --help       Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=============================================="
echo "  Alpha Arena - Master Control Web App"
echo "=============================================="
echo ""
echo "Starting server at http://${HOST}:${PORT}"
echo ""
echo "Press Ctrl+C to stop"
echo ""

# Run the web server
python -m uvicorn src.web.app:app --host "$HOST" --port "$PORT" $RELOAD
