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
    pip install --upgrade pip --quiet

    # If on Linux aarch64 (Raspberry Pi) and installing full requirements, use CPU PyTorch to avoid NVIDIA CUDA bloat
    if [ "$REQ_FILE" == "requirements.txt" ] && [ "$(uname -s)" = "Linux" ] && [ "$(uname -m)" = "aarch64" ] && ! command -v nvidia-smi &>/dev/null; then
        echo "==> Raspberry Pi (aarch64) detected: installing CPU-only PyTorch (skipping NVIDIA CUDA)..."
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu --quiet || true
    fi

    pip install -r "$REQ_FILE" --quiet
    touch .venv/.installed
fi

# Activate virtual environment
source .venv/bin/activate

# Optional explicit dependency check with --install flag
if [ "$1" == "--install" ]; then
    echo "==> Verifying dependencies..."
    if [ "$REQ_FILE" == "requirements.txt" ] && [ "$(uname -s)" = "Linux" ] && [ "$(uname -m)" = "aarch64" ] && ! command -v nvidia-smi &>/dev/null; then
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu --quiet || true
    fi
    pip install -r "$REQ_FILE" --quiet
    touch .venv/.installed
    shift
fi

# Launch CSRT Tracker with Pan-Tilt Servoing directly
exec python main.py "$@"
