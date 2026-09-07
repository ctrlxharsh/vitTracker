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
fi

# Activate virtual environment
source .venv/bin/activate

# Install requirements
echo "==> Verifying dependencies..."
pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet

# Launch tracker
if [ "$1" == "--vittrack" ] || [ "$1" == "-v" ]; then
    echo "==> Launching OpenCV VitTrack (Lightweight/Edge mode)..."
    python track_vittrack.py
else
    echo "==> Launching EdgeTAM Tracker (Auto-GPU accelerated)..."
    python track.py
fi
