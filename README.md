# AI Vision Tracker & Pan-Tilt Visual Servoing

Real-time vision tracking and autonomous target acquisition for drones and ground mounts.
Features dual tracking engines with closed-loop 2-DoF Pan-Tilt visual servoing.

---

## Dual Tracking Engines

1. **YOLO Autonomous Detection & Tracking**:
   - Uses YOLO model (`harsh-awasthi/bluemockdrone`, weights `best.pt`).
   - Automatically detects and tracks the target blue mock drone in the video feed without any manual intervention.
   - Persistent tracking across frames using ByteTrack / BoT-SORT ID persistence.
   - Automatically drives pan-tilt servos toward the detected object.

2. **OpenCV CSRT Manual Selection**:
   - Discriminative correlation filter with Bayes color likelihood and Epanechnikov 2D spatial reliability maps.
   - Sudden-jerk re-acquisition via projected velocity momentum and template matching.
   - Click-and-drag ROI selection for arbitrary objects.

---

## Controls & Mode Switching

- **In the GUI**:
  - Use the **`TRACKING ENGINE`** segmented button in the sidebar to toggle between **`YOLO Auto`** and **`CSRT Manual`**.
  - Or press **`t`** on the keyboard at any time to switch instantly between engines.
- **In CLI Mode (`--cv-only`)**:
  - Press **`t`** to toggle between YOLO and CSRT.
  - Press **`SPACE`** to pause (YOLO) or select bounding box (CSRT).
  - Press **`c`** to reset tracker and recenter servos.
  - Press **`s`** to toggle servo output on/off.
  - Press **`q`** or **`ESC`** to quit.

---

## Quick Start

### 1. Launch with GUI (YOLO mode default)
```bash
./run.sh
```
*or directly with python:*
```bash
source .venv/bin/activate
python main.py
```

### 2. Launch directly in CSRT mode
```bash
python main.py --mode csrt
```

### 3. Launch Standalone OpenCV Window (Lightweight / No Tkinter)
```bash
python main.py --cv-only
```

### 4. Custom Weights / Source
```bash
python main.py --weights best.pt --source path/to/video.mp4
```

---

## Hardware Servoing (Raspberry Pi)
- **Pan (Azimuth)**: GPIO 12 (PWM0)
- **Tilt (Elevation)**: GPIO 13 (PWM1)
- Automatic mock/simulation fallback on macOS/Windows/Linux without pigpiod.
