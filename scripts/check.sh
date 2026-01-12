#!/bin/bash
#
# ============================================================================
#   ALPHA ARENA - SYSTEM CHECK
# ============================================================================
#
# Comprehensive system check to verify everything is set up correctly.
# Run this whenever something seems wrong.
#
# USAGE:
#   ./scripts/check.sh           Run all checks
#   ./scripts/check.sh --fix     Attempt to fix issues automatically
#   ./scripts/check.sh -h        Show help
#
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Counters
PASS=0
FAIL=0
WARN=0

FIX_MODE=false

print_header() {
    echo ""
    echo -e "${BLUE}============================================${NC}"
    echo -e "${BLUE}  ALPHA ARENA - SYSTEM CHECK${NC}"
    echo -e "${BLUE}============================================${NC}"
    echo ""
}

print_help() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --fix       Attempt to fix issues automatically"
    echo "  -h, --help  Show this help"
    echo ""
    echo "This script checks:"
    echo "  - Python version (3.11+ required)"
    echo "  - Virtual environment setup"
    echo "  - Dependencies installed"
    echo "  - Configuration files present"
    echo "  - Directory structure"
    echo "  - Import test for main modules"
    echo ""
}

check_pass() {
    echo -e "  ${GREEN}[PASS]${NC} $1"
    ((PASS++))
}

check_fail() {
    echo -e "  ${RED}[FAIL]${NC} $1"
    ((FAIL++))
}

check_warn() {
    echo -e "  ${YELLOW}[WARN]${NC} $1"
    ((WARN++))
}

check_info() {
    echo -e "  ${BLUE}[INFO]${NC} $1"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --fix)
            FIX_MODE=true
            shift
            ;;
        -h|--help)
            print_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            print_help
            exit 1
            ;;
    esac
done

print_header

echo "Checking system requirements..."
echo ""

# ============================================================================
# Check Python
# ============================================================================
echo -e "${YELLOW}1. Python Environment${NC}"

# Find Python
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

if [ -n "$PYTHON_CMD" ]; then
    check_pass "Python found: $($PYTHON_CMD --version)"
else
    check_fail "Python 3.11+ not found"
    echo "         Please install Python 3.11 or higher"
fi

# ============================================================================
# Check Virtual Environment
# ============================================================================
echo ""
echo -e "${YELLOW}2. Virtual Environment${NC}"

if [ -d ".venv" ]; then
    check_pass "Virtual environment exists at .venv/"

    if [ -f ".venv/bin/python" ]; then
        VENV_PYTHON=$(.venv/bin/python --version 2>&1)
        check_pass "Python in venv: $VENV_PYTHON"
    else
        check_fail "Python not found in .venv/bin/"
        if [ "$FIX_MODE" = true ]; then
            echo "         Attempting to recreate venv..."
            rm -rf .venv
            $PYTHON_CMD -m venv .venv
        fi
    fi

    # Check isolation
    if grep -q "include-system-site-packages = false" .venv/pyvenv.cfg 2>/dev/null; then
        check_pass "Virtual environment is isolated"
    else
        check_warn "Virtual environment may include system packages"
    fi
else
    check_fail "Virtual environment not found"
    if [ "$FIX_MODE" = true ] && [ -n "$PYTHON_CMD" ]; then
        echo "         Creating virtual environment..."
        $PYTHON_CMD -m venv .venv
        check_info "Virtual environment created"
    else
        echo "         Run: ./scripts/setup.sh"
    fi
fi

# ============================================================================
# Check Dependencies
# ============================================================================
echo ""
echo -e "${YELLOW}3. Dependencies${NC}"

if [ -f ".venv/bin/python" ]; then
    # Check core dependencies
    DEPS=("fastapi" "uvicorn" "httpx" "pydantic" "jinja2" "pytest")

    for dep in "${DEPS[@]}"; do
        if .venv/bin/python -c "import $dep" 2>/dev/null; then
            check_pass "$dep installed"
        else
            check_fail "$dep NOT installed"
            if [ "$FIX_MODE" = true ]; then
                echo "         Installing $dep..."
                .venv/bin/pip install $dep
            fi
        fi
    done
else
    check_warn "Cannot check dependencies (venv not ready)"
fi

# ============================================================================
# Check Configuration Files
# ============================================================================
echo ""
echo -e "${YELLOW}4. Configuration Files${NC}"

if [ -f ".env" ]; then
    check_pass ".env file exists"

    # Check for required variables (without exposing values)
    ENV_VARS=("ANTHROPIC_API_KEY" "OPENAI_API_KEY")
    CONFIGURED=0
    for var in "${ENV_VARS[@]}"; do
        if grep -q "^${var}=." .env 2>/dev/null; then
            ((CONFIGURED++))
        fi
    done

    if [ $CONFIGURED -gt 0 ]; then
        check_pass "At least one API key configured"
    else
        check_warn "No API keys configured in .env"
        echo "         Edit .env and add your API keys"
    fi
else
    check_fail ".env file missing"
    if [ "$FIX_MODE" = true ]; then
        if [ -f ".env.example" ]; then
            cp .env.example .env
            check_info "Created .env from .env.example"
        fi
    else
        echo "         Run: ./scripts/setup.sh"
    fi
fi

if [ -f "requirements.txt" ]; then
    check_pass "requirements.txt exists"
else
    check_fail "requirements.txt missing"
fi

if [ -f "pyproject.toml" ]; then
    check_pass "pyproject.toml exists"
else
    check_warn "pyproject.toml missing"
fi

# ============================================================================
# Check Directory Structure
# ============================================================================
echo ""
echo -e "${YELLOW}5. Directory Structure${NC}"

REQUIRED_DIRS=("src" "src/web" "src/web/templates" "tests" "scripts" "docs")

for dir in "${REQUIRED_DIRS[@]}"; do
    if [ -d "$dir" ]; then
        check_pass "$dir/ exists"
    else
        check_fail "$dir/ missing"
        if [ "$FIX_MODE" = true ]; then
            mkdir -p "$dir"
            check_info "Created $dir/"
        fi
    fi
done

# Check data directories
DATA_DIRS=("data/logs" "data/memory" "data/backtest")
for dir in "${DATA_DIRS[@]}"; do
    if [ -d "$dir" ]; then
        check_pass "$dir/ exists"
    else
        check_warn "$dir/ missing (will be created on first run)"
        if [ "$FIX_MODE" = true ]; then
            mkdir -p "$dir"
            check_info "Created $dir/"
        fi
    fi
done

# ============================================================================
# Check Module Imports
# ============================================================================
echo ""
echo -e "${YELLOW}6. Module Import Test${NC}"

if [ -f ".venv/bin/python" ]; then
    MODULES=("src.web.app" "src.risk.controls" "src.broker.base" "src.strategy.base")

    for module in "${MODULES[@]}"; do
        if .venv/bin/python -c "import $module" 2>/dev/null; then
            check_pass "Can import $module"
        else
            check_fail "Cannot import $module"
        fi
    done
else
    check_warn "Cannot test imports (venv not ready)"
fi

# ============================================================================
# Check Scripts
# ============================================================================
echo ""
echo -e "${YELLOW}7. Scripts${NC}"

SCRIPTS=("scripts/setup.sh" "scripts/run_web.sh" "scripts/test.sh" "scripts/check.sh")

for script in "${SCRIPTS[@]}"; do
    if [ -f "$script" ]; then
        if [ -x "$script" ]; then
            check_pass "$script is executable"
        else
            check_warn "$script exists but not executable"
            if [ "$FIX_MODE" = true ]; then
                chmod +x "$script"
                check_info "Made $script executable"
            fi
        fi
    else
        check_fail "$script missing"
    fi
done

# ============================================================================
# Summary
# ============================================================================
echo ""
echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}  SUMMARY${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""
echo -e "  ${GREEN}Passed:${NC}   $PASS"
echo -e "  ${RED}Failed:${NC}   $FAIL"
echo -e "  ${YELLOW}Warnings:${NC} $WARN"
echo ""

if [ $FAIL -eq 0 ]; then
    echo -e "${GREEN}System check passed!${NC}"
    echo ""
    echo "You can now run:"
    echo "  ./scripts/run_web.sh     # Start web application"
    echo "  ./scripts/test.sh        # Run tests"
    echo ""
    exit 0
else
    echo -e "${RED}System check failed.${NC}"
    echo ""
    if [ "$FIX_MODE" = false ]; then
        echo "Try running with --fix to auto-fix issues:"
        echo "  ./scripts/check.sh --fix"
        echo ""
        echo "Or run setup:"
        echo "  ./scripts/setup.sh"
    fi
    echo ""
    exit 1
fi
