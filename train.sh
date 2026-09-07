#!/usr/bin/env bash
# One-Click YOLOv11n Bounding Box Training Runner
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

if [ -f "./.venv/bin/python" ]; then
    PYTHON_CMD="./.venv/bin/python"
else
    PYTHON_CMD="python3"
fi

echo "Starting YOLOv11n Bounding Box Detector Training..."
$PYTHON_CMD train_yolo.py "$@"
