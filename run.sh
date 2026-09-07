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
    $PYTHON_CMD -m venv --system-site-packages .venv
    source .venv/bin/activate
    echo "==> Installing dependencies..."
    pip install --upgrade pip
    if [ "$REQ_FILE" == "requirements.txt" ] && [ "$(uname -m)" == "aarch64" ]; then
        echo "==> Pre-installing PyTorch CPU wheel for ARM64..."
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    fi
    pip install -r "$REQ_FILE"
    pip uninstall -y opencv-python 2>/dev/null || true
    touch .venv/.installed
fi

# Ensure system site packages are accessible for picamera2 / libcamera
if [ -f ".venv/pyvenv.cfg" ]; then
    sed -i 's/include-system-site-packages = false/include-system-site-packages = true/' .venv/pyvenv.cfg 2>/dev/null || true
fi

# Activate virtual environment
source .venv/bin/activate

# Auto-install missing dependencies if environment is incomplete
if [ "$REQ_FILE" == "requirements.txt" ]; then
    if ! python -c "import ultralytics" &>/dev/null; then
        echo "==> Ultralytics not detected in virtual environment. Installing dependencies from $REQ_FILE..."
        if [ "$(uname -m)" == "aarch64" ] && ! python -c "import torch" &>/dev/null; then
            pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
        fi
        pip install -r "$REQ_FILE"
        pip uninstall -y opencv-python 2>/dev/null || true
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
    if [ "$REQ_FILE" == "requirements.txt" ] && [ "$(uname -m)" == "aarch64" ]; then
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    fi
    pip install -r "$REQ_FILE"
    pip uninstall -y opencv-python 2>/dev/null || true
    touch .venv/.installed
    shift
fi

# Ensure opencv-contrib-python is active for CSRT tracker support
if ! python -c "import cv2; assert hasattr(cv2, 'TrackerCSRT_create')" &>/dev/null; then
    pip uninstall -y opencv-python 2>/dev/null || true
    pip install --force-reinstall --no-deps "opencv-contrib-python>=4.10.0" 2>/dev/null || true
fi


# Launch AI Vision Tracker with Pan-Tilt Servoing directly
exec python main.py "$@"
