#!/usr/bin/env python3
"""
test_system.py
==============
Automated verification tests for:
- AxisController & Servo Mock
- PanTiltServoing Feedback Loop
- CSRT Tracker Engine on Synthetic Frames
- YOLO Autonomous Object Detection/Tracking (harsh-awasthi/bluemockdrone)
"""

import os
import sys
import time
import numpy as np
import cv2

from tracker import CSRTTrackerEngine, compute_spatial_reliability_map
from servo_controller import PanTiltServoing, AxisController, Servo
from yolo_tracker import YOLOTrackerEngine


def test_axis_controller():
    print("Testing AxisController...")
    ctl = AxisController(kp=0.45, kd=0.06, deadband_px=8.0, max_step_deg=2.0)
    
    # Test deadband
    delta = ctl.step(err_px=4.0, deg_per_px=0.1, dt=0.02)
    assert delta == 0.0, f"Expected 0.0 inside deadband, got {delta}"

    # Test proportional step
    delta = ctl.step(err_px=50.0, deg_per_px=0.1, dt=0.02)
    assert delta > 0.0, f"Expected positive step for positive error, got {delta}"
    assert delta <= 2.0, f"Expected clamped step <= 2.0, got {delta}"
    print("  AxisController tests passed!")


def test_servo_mock():
    print("Testing Servo mock mode...")
    servo = Servo(pi=None, gpio=12, lo_deg=-80.0, hi_deg=80.0, start_deg=0.0)
    assert servo.angle == 0.0
    
    ang = servo.write(45.0)
    assert ang == 45.0
    assert servo.angle == 45.0

    # Test clamping
    ang_clamped = servo.write(100.0)
    assert ang_clamped == 80.0
    assert servo.angle == 80.0
    print("  Servo mock tests passed!")


def test_pantilt_servoing():
    print("Testing PanTiltServoing...")
    pt = PanTiltServoing(force_mock=True)
    assert pt.is_mock is True
    assert pt.pan_angle == 0.0
    assert pt.tilt_angle == 0.0

    frame_w, frame_h = 640, 480
    # Center is at (320, 240)
    # Target at (400, 300) -> right and below center
    # PAN_SIGN is +1, so err_x = +80 px -> pan should move in positive direction (toward target)
    # TILT_SIGN is -1, so err_y = +60 px -> tilt should move in negative direction (down toward target)
    for _ in range(5):
        pan, tilt = pt.update(center=(400.0, 300.0), frame_w=frame_w, frame_h=frame_h, dt=0.02)

    assert pan > 0.0, f"Expected pan > 0 for target to the right, got {pan}"
    assert tilt < 0.0, f"Expected tilt < 0 for target below center, got {tilt}"

    # Recenter test
    pt.center_servos()
    assert pt.pan_angle == 0.0
    assert pt.tilt_angle == 0.0
    pt.release()
    print("  PanTiltServoing tests passed!")


def test_smoothness_and_no_jerk():
    print("Testing AxisController smoothness and derivative kick suppression...")
    ctl = AxisController(kp=1.8, kd=0.04, deadband_px=10.0, max_step_deg=0.8, max_speed_deg_s=25.0)

    # Sudden large offset (target appears 200px off-center)
    dt = 0.033  # ~30 FPS
    step1 = ctl.step(err_px=200.0, deg_per_px=0.1, dt=dt)

    # Step 1 must be strictly bounded by max velocity * dt (no derivative kick)
    assert step1 <= 25.0 * dt + 1e-5, f"Step 1 exceeded max velocity: {step1}"
    assert step1 <= 0.8, f"Step 1 exceeded max_step_deg: {step1}"
    assert step1 > 0.0, f"Step 1 should be positive: {step1}"

    # Verify smooth evolution over 20 steps
    steps = [step1]
    for _ in range(19):
        s = ctl.step(err_px=200.0, deg_per_px=0.1, dt=dt)
        steps.append(s)
        assert s <= 0.8 and s <= 25.0 * dt + 1e-5

    # No sudden spikes or discontinuities
    for i in range(1, len(steps)):
        diff = abs(steps[i] - steps[i - 1])
        assert diff < 0.2, f"Step acceleration discontinuity at frame {i}: {diff}"
    print("  Smoothness and jerk suppression tests passed!")


def test_tracker_synthetic():
    print("Testing CSRTTrackerEngine on synthetic frames...")
    engine = CSRTTrackerEngine()

    frame_h, frame_w = 480, 640
    box_x, box_y, box_w, box_h = 200, 150, 50, 50

    frame1 = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)
    cv2.rectangle(frame1, (box_x, box_y), (box_x + box_w, box_y + box_h), (200, 50, 50), -1)

    heatmap, mask = compute_spatial_reliability_map(frame1, (box_x, box_y, box_w, box_h))
    assert heatmap is not None and mask is not None, "Spatial map computation failed"

    success = engine.init(frame1, (box_x, box_y, box_w, box_h))
    assert success, "Tracker initialization failed"
    assert engine.is_tracking is True

    frame2 = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)
    box_x2 = box_x + 5
    box_y2 = box_y + 5
    cv2.rectangle(frame2, (box_x2, box_y2), (box_x2 + box_w, box_y2 + box_h), (200, 50, 50), -1)

    result = engine.update(frame2)
    assert result.success is True, "Tracker update failed on moved object"
    assert result.center is not None
    engine.reset()
    assert engine.is_tracking is False
    print("  CSRTTrackerEngine tests passed!")


def test_yolo_tracker():
    print("Testing YOLOTrackerEngine (Drone Detector)...")
    weights = "best.pt" if os.path.exists("best.pt") else None
    engine = YOLOTrackerEngine(weights_path=weights)

    # 1. Blank frame test (no false positive)
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    res_blank = engine.update(blank)
    assert res_blank.success is False

    # 2. Real preview test if available
    preview_path = "dataset/previews/preview_00.jpg"
    if os.path.exists(preview_path):
        drone_frame = cv2.imread(preview_path)
        res = engine.update(drone_frame)
        assert res.success is True, "YOLO should detect drone in preview_00.jpg"
        assert res.confidence > 0.5, f"Expected confidence > 0.5, got {res.confidence}"
        print(f"  Detected drone at {res.center} with confidence {res.confidence:.1%}")

    engine.reset()
    print("  YOLOTrackerEngine tests passed!")


def main():
    test_axis_controller()
    test_servo_mock()
    test_pantilt_servoing()
    test_smoothness_and_no_jerk()
    test_tracker_synthetic()
    test_yolo_tracker()
    print("\nALL SYSTEM TESTS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
