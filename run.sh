#!/usr/bin/env bash
set -e

# Detect Python command
if command -v python3 &>/dev/null; then
    PYTHON_CMD="python3"
elif command -v python &>/dev/null; then
    PYTHON_CMD="python"
else
    echo "Error: Python 3 is required but not installed." >&2
    exit 1
fi

# Set up virtual environment if not present
if [ ! -d ".venv" ]; then
    echo "==> Creating virtual environment (.venv)..."
    $PYTHON_CMD -m venv .venv
    source .venv/bin/activate
    echo "==> Installing dependencies..."
    pip install --upgrade pip --quiet
    pip install -r requirements.txt --quiet
    touch .venv/.installed
fi

# Activate virtual environment
source .venv/bin/activate

# Optional explicit dependency check with --install flag
if [ "$1" == "--install" ]; then
    echo "==> Verifying dependencies..."
    pip install -r requirements.txt --quiet
    touch .venv/.installed
    shift
fi

# Launch CSRT Tracker with Pan-Tilt Servoing directly
exec python main.py "$@"
