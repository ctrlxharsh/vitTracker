"""
camera.py
=========
Cross-platform video capture abstraction for AI Vision Tracker.

Supports:
1. Raspberry Pi CSI camera modules (OV5647, IMX219, IMX477, IMX708) via libcamera / picamera2.
2. Standard USB webcams and V4L2 devices via OpenCV VideoCapture.
3. Video file sources.
"""

import json
import os
import platform
import subprocess
import threading
import time
import cv2

# Deprioritize buggy MSMF backend on Windows and silence noisy C++ backend warnings
if platform.system() == "Windows":
    os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")


def _create_cv_capture(source, is_camera: bool = False):
    """Creates a cv2.VideoCapture instance preferring DirectShow on Windows."""
    if is_camera and platform.system() == "Windows":
        # DirectShow resolves MSMF async ReadSample/grabFrame error -1072873821 on Windows
        return cv2.VideoCapture(source, cv2.CAP_DSHOW)
    return cv2.VideoCapture(source)


class PiCameraCapture:
    """Wrapper for Raspberry Pi libcamera via picamera2 matching cv2.VideoCapture API."""

    def __init__(self, width: int = 640, height: int = 480):
        from picamera2 import Picamera2

        self.width = width
        self.height = height
        self.picam2 = Picamera2()
        # In Picamera2/libcamera, 'RGB888' format outputs a [B, G, R] NumPy array,
        # perfectly matching OpenCV's native BGR representation.
        config = self.picam2.create_video_configuration(
            main={"size": (width, height), "format": "RGB888"}
        )
        self.picam2.configure(config)
        self.picam2.start()
        self._opened = True
        self.is_picamera = True

    def isOpened(self) -> bool:
        return self._opened

    def read(self):
        if not self._opened:
            return False, None
        try:
            frame = self.picam2.capture_array()
            if frame is None:
                return False, None
            if frame.ndim == 3 and frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            return True, frame
        except Exception:
            return False, None

    def release(self):
        if self._opened:
            self._opened = False
            try:
                self.picam2.stop()
                self.picam2.close()
            except Exception:
                pass

    def get(self, prop: int) -> float:
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self.width)
        elif prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self.height)
        elif prop == cv2.CAP_PROP_FPS:
            return 30.0
        return 0.0

    def set(self, prop: int, val: float) -> bool:
        return True


class OpenCVCapture:
    """
    Threaded wrapper around cv2.VideoCapture providing non-blocking read().
    Prevents webcam driver exposure/I/O latency from blocking the Tkinter GUI thread.
    """

    def __init__(self, cap, source_desc: str = "Webcam", is_live: bool = True, cam_index: int = 0):
        self.cap = cap
        self.is_picamera = False
        self.source_desc = source_desc
        self.is_live = is_live
        self._cam_index = cam_index
        self._lock = threading.Lock()
        self._running = True
        self._latest_frame = None
        self._latest_ret = False

        # Pre-warm with initial frames (USB webcams often deliver a black frame on first read)
        if self.cap.isOpened():
            for _ in range(5):
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    self._latest_ret = True
                    self._latest_frame = frame
                    if frame.mean() > 10:
                        break
                time.sleep(0.03)

        if self.is_live:
            self._thread = threading.Thread(target=self._reader_loop, daemon=True, name="CameraReader")
            self._thread.start()
        else:
            self._thread = None

    def _reader_loop(self):
        while self._running:
            if not self.cap.isOpened():
                time.sleep(0.05)
                continue
            ret, frame = self.cap.read()
            if ret and frame is not None:
                with self._lock:
                    self._latest_ret = True
                    self._latest_frame = frame
            else:
                time.sleep(0.005)

    def isOpened(self) -> bool:
        return self.cap.isOpened()

    def read(self):
        if self.is_live:
            with self._lock:
                if self._latest_frame is not None:
                    return self._latest_ret, self._latest_frame.copy()
                return False, None
        return self.cap.read()

    def release(self):
        self._running = False
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=0.1)
        self._thread = None
        old_cap = self.cap
        # Release in background thread to avoid blocking GUI thread during AVFoundation session teardown
        threading.Thread(target=old_cap.release, daemon=True).start()

    def get(self, prop: int) -> float:
        return self.cap.get(prop)

    def set(self, prop: int, val: float) -> bool:
        return self.cap.set(prop, val)


_CACHED_CAMERAS = None


def get_available_cameras(refresh: bool = False):
    """Returns list of (index, display_label) for active cameras."""
    global _CACHED_CAMERAS
    if _CACHED_CAMERAS is not None and not refresh:
        return list(_CACHED_CAMERAS)

    cameras = []
    # Test indices 0 through 4 (handles laptops with built-in + external USB cameras)
    for idx in range(5):
        cap = _create_cv_capture(idx, is_camera=True)
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            if ret and frame is not None:
                h, w = frame.shape[:2]
                if platform.system() == "Darwin":
                    label = f"USB Camera (Index {idx} - {w}x{h})" if idx == 0 else f"Camera (Index {idx} - {w}x{h})"
                elif platform.system() == "Windows":
                    label = f"Camera {idx} (USB/Webcam - {w}x{h})"
                else:
                    label = f"Camera {idx} ({w}x{h})"
                cameras.append((idx, label))
        elif platform.system() == "Windows" and idx >= 2:
            break

    if not cameras:
        cameras.append((0, "Camera 0 [Default]"))
    _CACHED_CAMERAS = cameras
    return list(cameras)


def find_best_camera_source() -> int:
    """Returns the first detected active camera index."""
    cams = get_available_cameras()
    if cams:
        return cams[0][0]
    return 0


def open_video_capture(source="auto", width: int = 1280, height: int = 720):
    """
    Opens video stream from Raspberry Pi CSI camera, USB webcam, or video file.
    Returns capture object with cv2.VideoCapture compatible API and .is_picamera attribute.
    """
    is_cam_index = False
    cam_index = 0

    if source is None or str(source).lower() in ("auto", "none", ""):
        cam_index = find_best_camera_source()
        is_cam_index = True
    elif isinstance(source, int):
        is_cam_index = True
        cam_index = source
    elif isinstance(source, str):
        if source.isdigit():
            is_cam_index = True
            cam_index = int(source)
        elif source.lower() in ("picam", "rpicam", "csi", "/dev/video0"):
            is_cam_index = True
            cam_index = 0

    if is_cam_index and cam_index == 0 and platform.system() == "Linux":
        # First attempt Raspberry Pi Camera via picamera2 on Linux
        try:
            picam = PiCameraCapture(width=width, height=height)
            ret, test_frame = picam.read()
            if ret and test_frame is not None:
                picam.source_desc = "Raspberry Pi Camera (CSI)"
                return picam
            picam.release()
        except Exception:
            pass

    # Fall back to standard cv2.VideoCapture with OS-optimized backend
    actual_source = cam_index if is_cam_index else source
    cap = _create_cv_capture(actual_source, is_camera=is_cam_index)

    if is_cam_index:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # If OpenCV opened /dev/video0 on Linux but cannot read frames (typical with Pi Unicam),
        # try picamera2 as fallback
        if platform.system() == "Linux":
            has_frame = False
            if cap.isOpened():
                try:
                    ret, _ = cap.read()
                    has_frame = ret
                except Exception:
                    has_frame = False

            if not has_frame:
                cap.release()
                try:
                    picam = PiCameraCapture(width=width, height=height)
                    ret, test_frame = picam.read()
                    if ret and test_frame is not None:
                        picam.source_desc = "Raspberry Pi Camera (CSI)"
                        return picam
                    picam.release()
                except Exception:
                    pass
                # Reopen standard capture if picamera2 also failed
                cap = _create_cv_capture(actual_source, is_camera=True)

        return OpenCVCapture(cap, source_desc=f"Webcam ({cam_index})", is_live=True, cam_index=cam_index)
    else:
        return OpenCVCapture(cap, source_desc=os.path.basename(str(source)), is_live=False, cam_index=-1)

