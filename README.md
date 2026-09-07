# vitTracker

Real-time vision tracking from live camera feeds with automatic GPU acceleration.

Supports:
- **EdgeTAM** ([`yonigozlan/EdgeTAM-hf`](https://huggingface.co/yonigozlan/EdgeTAM-hf)): Track Anything Model on edge devices. Segment & track objects with a single click or bounding box.
- **VitTrack**: Ultra-fast, lightweight Vision Transformer tracker running via OpenCV DNN (ideal for Raspberry Pi and low-spec CPUs).

---

## Hardware Acceleration

The tracker automatically detects the best available hardware engine:
- **NVIDIA GPU**: CUDA acceleration with `bfloat16` / `float16`.
- **Apple Silicon (Mac M1/M2/M3/M4)**: Metal Performance Shaders (`MPS`) with `bfloat16`.
- **CPU Fallback**: Standard `float32` execution on any x86 or ARM CPU.

---

## Quick Start (Any Device)

### Option 1: One-Click Bash Script (Linux & macOS)

Clone and run:
```bash
git clone https://github.com/ctrlxharsh/vitTracker.git
cd vitTracker
chmod +x run.sh
./run.sh
```
`run.sh` will automatically create a virtual environment (`.venv`), install dependencies, and launch the tracker.

For the lightweight CPU/Raspberry Pi mode:
```bash
./run.sh --vittrack
```

---

### Option 2: Manual Setup

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run EdgeTAM Tracker (Default)**:
   ```bash
   python track.py
   ```

3. **Run OpenCV VitTrack (Lightweight / Raspberry Pi)**:
   ```bash
   python track_vittrack.py
   ```

---

## Controls

| Action | Control |
|---|---|
| **Point-to-Track** | Left Click directly on any object |
| **Box-to-Track** | Press `SPACE` to freeze frame, drag crosshairs, and press `ENTER`/`SPACE` |
| **Clear Tracker** | Press `c` |
| **Quit** | Press `q` or `ESC` |

---

## License

Apache 2.0 / MIT
