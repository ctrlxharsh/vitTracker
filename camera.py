"""
camera.py
=========
Cross-platform video capture abstraction for AI Vision Tracker.

Supports:
1. Raspberry Pi CSI camera modules (OV5647, IMX219, IMX477, IMX708) via libcamera / picamera2.
2. Standard USB webcams and V4L2 devices via OpenCV VideoCapture.
3. Video file sources.
"""

import os
import cv2


class PiCameraCapture:
    """Wrapper for Raspberry Pi libcamera via picamera2 matching cv2.VideoCapture API."""

    def __init__(self, width: int = 640, height: int = 480):
        from picamera2 import Picamera2

        self.width = width
        self.height = height
        self.picam2 = Picamera2()
        config = self.picam2.create_video_configuration(
            main={"size": (width, height), "format": "BGR888"}
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


def open_video_capture(source=0, width: int = 640, height: int = 480):
    """
    Opens video stream from Raspberry Pi CSI camera, USB webcam, or video file.
    Returns capture object with cv2.VideoCapture compatible API and .is_picamera attribute.
    """
    is_cam_index = False
    cam_index = 0

    if isinstance(source, int):
        is_cam_index = True
        cam_index = source
    elif isinstance(source, str):
        if source.isdigit():
            is_cam_index = True
            cam_index = int(source)
        elif source.lower() in ("picam", "rpicam", "csi", "/dev/video0"):
            is_cam_index = True
            cam_index = 0

    if is_cam_index and cam_index == 0:
        # First attempt Raspberry Pi Camera via picamera2
        try:
            picam = PiCameraCapture(width=width, height=height)
            ret, test_frame = picam.read()
            if ret and test_frame is not None:
                picam.source_desc = "Raspberry Pi Camera (CSI)"
                return picam
            picam.release()
        except Exception:
            pass

    # Fall back to standard cv2.VideoCapture
    actual_source = cam_index if is_cam_index else source
    cap = cv2.VideoCapture(actual_source)
    cap.is_picamera = False

    if is_cam_index:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # If OpenCV opened /dev/video0 but cannot read frames (typical with Pi Unicam),
        # try picamera2 as fallback
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
            cap = cv2.VideoCapture(actual_source)
            cap.is_picamera = False

        cap.source_desc = f"Webcam ({cam_index})"
    else:
        cap.source_desc = os.path.basename(str(source))

    return cap
