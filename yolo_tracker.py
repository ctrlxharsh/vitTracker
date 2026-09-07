"""
yolo_tracker.py
===============
YOLO-based autonomous object detector and tracker for the blue mock drone.

Uses trained weights from Hugging Face ("harsh-awasthi/bluemockdrone") or local "best.pt".
Features:
- Automatic weights discovery (local cache -> workspace root -> Hugging Face Hub download).
- Persistent multi-frame object tracking (ByteTrack / BoT-SORT via model.track).
- Primary target selection with trajectory history and confidence telemetry.
- Seamless interface compatibility with TrackingResult for pan-tilt visual servoing.
"""

from collections import deque
import os
from typing import Optional, Tuple
import numpy as np

from tracker import TrackingResult


def get_yolo_weights(repo_id: str = "harsh-awasthi/bluemockdrone", filename: str = "best.pt") -> str:
    """
    Finds or downloads the YOLO weights.
    Prioritizes local file, then runs/ directory, then downloads from Hugging Face Hub.
    """
    # 1. Check current directory
    if os.path.exists(filename):
        return os.path.abspath(filename)

    # 2. Check runs training directory
    local_candidates = [
        os.path.join("runs", "detect", "runs", "detect", "train_yolov11n", "weights", "best.pt"),
        os.path.join("runs", "detect", "train_yolov11n", "weights", "best.pt"),
    ]
    for cand in local_candidates:
        if os.path.exists(cand):
            return os.path.abspath(cand)

    # 3. Download from Hugging Face Hub
    print(f"Downloading {filename} from Hugging Face ({repo_id})...")
    try:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(repo_id=repo_id, filename=filename)
        print(f"Weights downloaded to: {path}")
        return path
    except Exception as e:
        print(f"Failed to download from Hugging Face: {e}")
        # Fallback to local default if present
        if os.path.exists("yolo11n.pt"):
            print("Falling back to local 'yolo11n.pt'...")
            return os.path.abspath("yolo11n.pt")
        raise RuntimeError(f"Could not obtain YOLO weights: {e}")


class YOLOTrackerEngine:
    """
    Autonomous YOLO Detection and Tracking engine.
    Detects blue mock drone targets and assigns persistent track IDs.
    """

    def __init__(self, weights_path: Optional[str] = None, conf: float = 0.80, max_trajectory: int = 40):
        self.conf = conf
        self.weights_path = weights_path or get_yolo_weights()
        self.trajectory = deque(maxlen=max_trajectory)
        self.active_track_id: Optional[int] = None
        self.last_bbox: Optional[Tuple[int, int, int, int]] = None

        print(f"Loading YOLO model from: {self.weights_path}...")
        from ultralytics import YOLO

        self.model = YOLO(self.weights_path)
        self.class_names = self.model.names
        print(f"YOLO model ready. Detected classes: {self.class_names}")

    def reset(self):
        """Reset active track identity and trajectory."""
        self.trajectory.clear()
        self.active_track_id = None
        self.last_bbox = None

    def update(self, frame: np.ndarray) -> TrackingResult:
        """
        Runs YOLO tracking/detection on the incoming video frame.
        Returns a standardized TrackingResult containing the primary target centroid.
        """
        fh, fw = frame.shape[:2]

        try:
            # Use model.track with persistence for continuous ID assignment
            results = self.model.track(frame, persist=True, conf=self.conf, verbose=False)
        except Exception:
            # Fallback to predict if tracker algorithm is initializing
            results = self.model.predict(frame, conf=self.conf, verbose=False)

        if not results or len(results) == 0:
            return TrackingResult(
                success=False,
                bbox=None,
                center=None,
                confidence=0.0,
                status="YOLO SEARCHING...",
            )

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return TrackingResult(
                success=False,
                bbox=None,
                center=None,
                confidence=0.0,
                status="YOLO SEARCHING...",
            )

        # Parse detected boxes: xyxy, conf, id
        xyxy_arr = boxes.xyxy.cpu().numpy()
        conf_arr = boxes.conf.cpu().numpy()
        id_arr = boxes.id.cpu().numpy() if boxes.id is not None else None

        best_idx = 0
        # If we already have an active track ID, prefer following that same target
        if id_arr is not None and self.active_track_id is not None:
            matches = np.where(id_arr == self.active_track_id)[0]
            if len(matches) > 0:
                best_idx = matches[0]
            else:
                best_idx = int(np.argmax(conf_arr))
        else:
            best_idx = int(np.argmax(conf_arr))

        if id_arr is not None and len(id_arr) > best_idx:
            self.active_track_id = int(id_arr[best_idx])

        # Primary target coordinates
        bx1, by1, bx2, by2 = xyxy_arr[best_idx]
        x1 = max(0, min(fw - 1, int(bx1)))
        y1 = max(0, min(fh - 1, int(by1)))
        x2 = max(0, min(fw, int(bx2)))
        y2 = max(0, min(fh, int(by2)))
        w = max(1, x2 - x1)
        h = max(1, y2 - y1)

        cx = x1 + w / 2.0
        cy = y1 + h / 2.0
        conf_val = float(conf_arr[best_idx])

        self.last_bbox = (x1, y1, w, h)
        self.trajectory.append((int(cx), int(cy)))

        class_id = int(boxes.cls[best_idx].item()) if boxes.cls is not None else 0
        label = self.class_names.get(class_id, "target")
        track_str = f"#{self.active_track_id}" if self.active_track_id is not None else ""

        return TrackingResult(
            success=True,
            bbox=(x1, y1, w, h),
            center=(cx, cy),
            confidence=conf_val,
            status=f"YOLO TRACKING {label.upper()} {track_str}".strip(),
        )
