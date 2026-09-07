#!/usr/bin/env python3
"""
main.py
=======
AI Vision Tracker with Pan-Tilt Visual Servoing.

Supports dual tracking engines:
1. YOLO Auto-Detection & Tracking (harsh-awasthi/bluemockdrone best.pt)
2. OpenCV CSRT Manual Selection with Spatial Reliability Maps

Launchable as modern CustomTkinter GUI or standalone OpenCV CLI window.
"""

import argparse
import sys
import threading
import time
import cv2

from camera import open_video_capture
from servo_controller import PanTiltServoing
from tracker import CSRTTrackerEngine, TrackingResult, compute_spatial_reliability_map
from yolo_tracker import YOLOTrackerEngine, get_yolo_weights


def run_opencv_cli(
    source=0,
    servo_controller: PanTiltServoing = None,
    initial_mode: str = "yolo",
    weights_path: str = None,
    conf: float = 0.80,
):
    """Standalone OpenCV window runner supporting live switching between YOLO and CSRT."""
    print(f"==> Launching AI Vision Tracker in Standalone CLI Mode on source {source}...")
    cap = open_video_capture(source)

    if not cap.isOpened():
        print(f"Error: Cannot open video source {source}.")
        return

    win_name = "AI Vision Tracker - Visual Servoing"
    cv2.namedWindow(win_name)

    mode = initial_mode.lower()
    csrt_tracker = CSRTTrackerEngine()
    yolo_tracker = None

    try:
        yolo_tracker = YOLOTrackerEngine(weights_path=weights_path, conf=conf)
    except Exception as e:
        print(f"Notice: YOLO initialization skipped ({e}). Operating in CSRT mode.")
        mode = "csrt"

    servo = servo_controller or PanTiltServoing()
    show_spatial_map = False

    print("\nInitial Mode:", f"[{mode.upper()}]")
    print("Servo Mode:", "HARDWARE (pigpio)" if servo.is_hardware else "MOCK (Simulation)")
    print("\nControls:")
    print("  [t]     Toggle between YOLO Auto-Detect and CSRT Manual")
    print("  [SPACE] Select target bounding box (CSRT mode) / Pause (YOLO mode)")
    print("  [m]     Toggle Spatial Reliability Map heatmap (CSRT)")
    print("  [c]     Clear / Reset tracker & center servos")
    print("  [s]     Toggle servo tracking on/off")
    print("  [q/ESC] Quit\n")

    prev_t = time.monotonic()

    worker_frame = None
    worker_result = TrackingResult(success=False, bbox=None, confidence=0.0, center=None)
    worker_busy = False
    track_lock = threading.Lock()
    cli_running = True

    def tracker_worker():
        nonlocal worker_frame, worker_result, worker_busy
        while cli_running:
            f_proc = None
            with track_lock:
                if worker_frame is not None:
                    f_proc = worker_frame
                    worker_frame = None
                    worker_busy = True
            if f_proc is not None:
                try:
                    if mode == "yolo" and yolo_tracker is not None:
                        res = yolo_tracker.update(f_proc)
                    else:
                        res = csrt_tracker.update(f_proc)
                except Exception:
                    res = TrackingResult(success=False, bbox=None, confidence=0.0, center=None)
                with track_lock:
                    worker_result = res
                    worker_busy = False
            else:
                time.sleep(0.005)

    th = threading.Thread(target=tracker_worker, daemon=True)
    th.start()

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            if isinstance(source, int) and not getattr(cap, "is_picamera", False):
                frame = cv2.flip(frame, 1)

            fh, fw = frame.shape[:2]
            t_now = time.monotonic()
            dt = max(1e-3, t_now - prev_t)
            prev_t = t_now

            # Dispatch frame to worker & get latest tracking result
            with track_lock:
                if not worker_busy:
                    worker_frame = frame.copy()
                result = worker_result

            # Update servos at full frame rate
            pan_ang, tilt_ang = servo.update(
                center=result.center if result.success else None,
                frame_w=fw,
                frame_h=fh,
                dt=dt,
            )

            # Draw optical boresight crosshair
            cx, cy = fw // 2, fh // 2
            cv2.line(frame, (cx - 10, cy), (cx + 10, cy), (100, 100, 100), 1)
            cv2.line(frame, (cx, cy - 10), (cx, cy + 10), (100, 100, 100), 1)

            if result.success:
                x, y, w, h = result.bbox
                if mode == "csrt":
                    if show_spatial_map:
                        heatmap, _ = compute_spatial_reliability_map(frame, (x, y, w, h))
                        if heatmap is not None:
                            x1, y1 = max(0, x), max(0, y)
                            x2, y2 = min(fw, x + w), min(fh, y + h)
                            rh_crop = cv2.resize(heatmap, (x2 - x1, y2 - y1))
                            frame[y1:y2, x1:x2] = cv2.addWeighted(rh_crop, 0.45, frame[y1:y2, x1:x2], 0.55, 0)
                    color = (0, 240, 255) if result.is_recovered else (0, 230, 118)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(
                        frame,
                        f"CSRT [{int(result.confidence * 100)}%]",
                        (x, max(20, y - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        1,
                    )
                else:
                    # YOLO Drone overlay
                    color = (255, 180, 0)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(
                        frame,
                        f"BLUE DRONE [{int(result.confidence * 100)}%]",
                        (x, max(20, y - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        1,
                    )
            elif result.is_recovering and result.bbox is not None:
                bx, by, bw, bh = result.bbox
                cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (0, 165, 255), 1)
                cv2.putText(frame, "RECOVERING MOTION...", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
            else:
                if mode == "yolo":
                    cv2.putText(frame, "YOLO: Scanning for Blue Drone...", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 180, 0), 2)
                elif csrt_tracker.is_tracking:
                    cv2.putText(frame, "TARGET LOST - Press SPACE to re-select", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                else:
                    cv2.putText(frame, "CSRT: Press SPACE to select ROI", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2)

            # Telemetry HUD
            servo_status = "ENABLED" if servo.enabled else "DISABLED"
            mode_desc = f"ESP32:{servo.port_name}" if servo.is_hardware else "ESP32:MOCK"
            hud_text = f"[{mode.upper()}] [{mode_desc}|{servo_status}] Pan: {pan_ang:+5.1f}d ({servo.pan_us}us) | Tilt: {tilt_ang:+5.1f}d ({servo.tilt_us}us) | [P] PanInv [I] TiltInv"
            cv2.putText(frame, hud_text, (16, fh - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (240, 240, 240), 1)

            cv2.imshow(win_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            elif key == ord("t"):
                with track_lock:
                    mode = "csrt" if mode == "yolo" else "yolo"
                    worker_frame = None
                    worker_result = TrackingResult(success=False, bbox=None, confidence=0.0, center=None)
                print(f"Switched mode to: [{mode.upper()}]")
            elif key == ord("c"):
                with track_lock:
                    worker_frame = None
                    worker_result = TrackingResult(success=False, bbox=None, confidence=0.0, center=None)
                    if mode == "yolo" and yolo_tracker:
                        yolo_tracker.reset()
                    else:
                        csrt_tracker.reset()
                servo.center_servos()
            elif key == ord("s"):
                servo.enabled = not servo.enabled
            elif key == ord("p"):
                servo.pan_sign = -servo.pan_sign
                print(f"Pan sign toggled to: {servo.pan_sign:+d}")
            elif key == ord("i"):
                servo.tilt_sign = -servo.tilt_sign
                print(f"Tilt sign toggled to: {servo.tilt_sign:+d}")
            elif key == ord("m"):
                show_spatial_map = not show_spatial_map
            elif key == ord(" "):
                if mode == "csrt":
                    roi = cv2.selectROI(win_name, frame, fromCenter=False, showCrosshair=True)
                    if roi[2] > 10 and roi[3] > 10:
                        with track_lock:
                            csrt_tracker.init(frame, roi)
                            worker_result = TrackingResult(
                                success=True,
                                bbox=roi,
                                confidence=1.0,
                                center=(roi[0] + roi[2] / 2.0, roi[1] + roi[3] / 2.0),
                            )

    finally:
        cli_running = False
        cap.release()
        servo.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="AI Vision Tracker & Visual Servoing (YOLO Drone & CSRT)")
    parser.add_argument("--source", type=str, default="auto", help="Video source (camera index e.g. 0, 'auto' for best camera, or video file path)")
    parser.add_argument("--mode", type=str, choices=["yolo", "csrt"], default="yolo", help="Initial tracking mode (yolo or csrt)")
    parser.add_argument("--weights", type=str, default=None, help="Path to custom YOLO weights (.pt)")
    parser.add_argument("--conf", type=float, default=0.80, help="Confidence threshold for YOLO (default: 0.80)")
    parser.add_argument("--cv-only", action="store_true", help="Launch in standalone OpenCV window without CustomTkinter GUI")
    parser.add_argument("--port", type=str, default="auto", help="Serial port for NodeMCU ESP32 (e.g. /dev/cu.usbserial-0001 or 'auto')")
    parser.add_argument("--baud", type=int, default=115200, help="Serial baud rate for ESP32 (default: 115200)")
    parser.add_argument("--invert-pan", action="store_true", help="Invert pan servo direction (default: pan_sign=+1)")
    parser.add_argument("--invert-tilt", action="store_true", help="Invert tilt servo direction (default: tilt_sign=-1)")
    parser.add_argument("--no-servo", action="store_true", help="Disable servo controller")
    parser.add_argument("--mock-servo", action="store_true", help="Force mock servo mode (skip ESP32 serial connection)")
    args = parser.parse_args()

    if args.source.lower() in ("auto", "none"):
        from camera import find_best_camera_source
        source = find_best_camera_source()
        print(f"Auto-detected primary camera source: Index {source}")
    elif args.source.isdigit():
        source = int(args.source)
    else:
        source = args.source

    pan_sign = -1 if args.invert_pan else 1
    tilt_sign = 1 if args.invert_tilt else -1

    servo = PanTiltServoing(
        serial_port=args.port,
        baud=args.baud,
        pan_sign=pan_sign,
        tilt_sign=tilt_sign,
        force_mock=args.mock_servo,
    )
    if args.no_servo:
        servo.enabled = False

    default_gui_mode = "YOLO Auto" if args.mode == "yolo" else "CSRT Manual"

    if args.cv_only:
        run_opencv_cli(
            source=source,
            servo_controller=servo,
            initial_mode=args.mode,
            weights_path=args.weights,
            conf=args.conf,
        )
    else:
        try:
            import customtkinter as ctk
            from gui.app import CSRTTrackerApp
        except ImportError as e:
            print(f"Warning: GUI libraries failed to import ({e}). Falling back to standalone OpenCV CLI mode...")
            run_opencv_cli(
                source=source,
                servo_controller=servo,
                initial_mode=args.mode,
                weights_path=args.weights,
                conf=args.conf,
            )
            return

        root = ctk.CTk()
        CSRTTrackerApp(
            root,
            video_source=source,
            servo_controller=servo,
            default_mode=default_gui_mode,
            yolo_weights=args.weights,
            yolo_conf=args.conf,
        )
        root.mainloop()


if __name__ == "__main__":
    main()
