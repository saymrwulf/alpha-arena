#!/bin/bash
#
# ============================================================================
#   ALPHA ARENA - TEST RUNNER
# ============================================================================
#
# Run all tests or specific test suites.
#
# USAGE:
#   ./scripts/test.sh              Run all tests
#   ./scripts/test.sh unit         Run unit tests only
#   ./scripts/test.sh api          Run API tests only
#   ./scripts/test.sh e2e          Run end-to-end tests only
#   ./scripts/test.sh fast         Run fast tests (exclude slow)
#   ./scripts/test.sh coverage     Run with coverage report
#   ./scripts/test.sh -v           Run with verbose output
#   ./scripts/test.sh -h           Show this help
#
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_header() {
    echo ""
    echo -e "${BLUE}============================================${NC}"
    echo -e "${BLUE}  ALPHA ARENA - TEST RUNNER${NC}"
    echo -e "${BLUE}============================================${NC}"
    echo ""
}

print_help() {
    echo "Usage: $0 [command] [options]"
    echo ""
    echo "Commands:"
    echo "  (none)      Run all tests"
    echo "  unit        Run unit tests only (test_*.py excluding api/e2e)"
    echo "  api         Run API tests only (test_api.py)"
    echo "  e2e         Run end-to-end tests only (test_e2e.py)"
    echo "  fast        Run fast tests (exclude @slow marker)"
    echo "  coverage    Run with coverage report"
    echo ""
    echo "Options:"
    echo "  -v, --verbose    Verbose output"
    echo "  -x, --exitfirst  Exit on first failure"
    echo "  -h, --help       Show this help"
    echo ""
    echo "Examples:"
    echo "  $0                   # Run all tests"
    echo "  $0 unit -v           # Run unit tests verbosely"
    echo "  $0 api               # Run API tests"
    echo "  $0 coverage          # Run with coverage"
    echo ""
}

check_venv() {
    if [ ! -d ".venv" ]; then
        echo -e "${RED}ERROR: Virtual environment not found.${NC}"
        echo ""
        echo "Please run setup first:"
        echo "  ./scripts/setup.sh"
        echo ""
        exit 1
    fi
}

check_dependencies() {
    if ! .venv/bin/python -c "import pytest" 2>/dev/null; then
        echo -e "${RED}ERROR: pytest not installed.${NC}"
        echo ""
        echo "Please run setup first:"
        echo "  ./scripts/setup.sh"
        echo ""
        exit 1
    fi
}

run_tests() {
    local test_args="$1"
    local extra_args="${@:2}"

    echo -e "${GREEN}Running tests...${NC}"
    echo ""

    .venv/bin/python -m pytest $test_args $extra_args

    local exit_code=$?

    echo ""
    if [ $exit_code -eq 0 ]; then
        echo -e "${GREEN}All tests passed!${NC}"
    else
        echo -e "${RED}Some tests failed.${NC}"
    fi

    return $exit_code
}

# Parse arguments
COMMAND=""
EXTRA_ARGS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            print_help
            exit 0
            ;;
        -v|--verbose)
            EXTRA_ARGS="$EXTRA_ARGS -v"
            shift
            ;;
        -x|--exitfirst)
            EXTRA_ARGS="$EXTRA_ARGS -x"
            shift
            ;;
        unit|api|e2e|fast|coverage)
            COMMAND="$1"
            shift
            ;;
        *)
            EXTRA_ARGS="$EXTRA_ARGS $1"
            shift
            ;;
    esac
done

print_header
check_venv
check_dependencies

# Activate venv
source .venv/bin/activate

case $COMMAND in
    unit)
        echo -e "${YELLOW}Running UNIT tests...${NC}"
        run_tests "tests/ --ignore=tests/test_api.py --ignore=tests/test_e2e.py" $EXTRA_ARGS
        ;;
    api)
        echo -e "${YELLOW}Running API tests...${NC}"
        run_tests "tests/test_api.py" $EXTRA_ARGS
        ;;
    e2e)
        echo -e "${YELLOW}Running END-TO-END tests...${NC}"
        run_tests "tests/test_e2e.py" $EXTRA_ARGS
        ;;
    fast)
        echo -e "${YELLOW}Running FAST tests (excluding slow)...${NC}"
        run_tests "tests/ -m 'not slow'" $EXTRA_ARGS
        ;;
    coverage)
        echo -e "${YELLOW}Running tests with COVERAGE...${NC}"
        run_tests "tests/ --cov=src --cov-report=html --cov-report=term-missing" $EXTRA_ARGS
        echo ""
        echo -e "${GREEN}Coverage report generated: htmlcov/index.html${NC}"
        ;;
    *)
        echo -e "${YELLOW}Running ALL tests...${NC}"
        run_tests "tests/" $EXTRA_ARGS
        ;;
esac
