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

# Select requirements file
REQ_FILE="requirements.txt"
if [ "$1" == "--pi" ] || [ "$1" == "--csrt-only" ]; then
    REQ_FILE="requirements-pi.txt"
    echo "==> Using lightweight Raspberry Pi profile ($REQ_FILE)..."
    shift
fi

# Set up virtual environment if not present
if [ ! -d ".venv" ]; then
    echo "==> Creating virtual environment (.venv)..."
    $PYTHON_CMD -m venv .venv
    source .venv/bin/activate
    echo "==> Installing dependencies..."
    pip install --upgrade pip
    pip install -r "$REQ_FILE"
    touch .venv/.installed
fi

# Activate virtual environment
source .venv/bin/activate

# Auto-install missing dependencies if environment is incomplete
if [ "$REQ_FILE" == "requirements.txt" ]; then
    if ! python -c "import ultralytics" &>/dev/null; then
        echo "==> Ultralytics not detected in virtual environment. Installing dependencies from $REQ_FILE..."
        pip install -r "$REQ_FILE"
        touch .venv/.installed
    fi
elif [ "$REQ_FILE" == "requirements-pi.txt" ]; then
    if ! python -c "import cv2" &>/dev/null; then
        echo "==> Missing dependencies detected. Installing from $REQ_FILE..."
        pip install -r "$REQ_FILE"
        touch .venv/.installed
    fi
fi

# Optional explicit dependency check with --install flag
if [ "$1" == "--install" ]; then
    echo "==> Verifying dependencies..."
    pip install -r "$REQ_FILE"
    touch .venv/.installed
    shift
fi

# Launch AI Vision Tracker with Pan-Tilt Servoing directly
exec python main.py "$@"
