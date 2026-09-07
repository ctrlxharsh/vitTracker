"""
tracker.py
==========
OpenCV CSRT (Channel and Spatial Reliability Tracker) Engine.

Decoupled core tracking logic with:
- Robust CSRT tracker factory with expanded search padding for high-speed tracking.
- Bayes spatial reliability map calculation with Epanechnikov spatial prior.
- Instantaneous velocity momentum estimation (EMA smoothed).
- Sudden jerk re-acquisition via projected template matching.
- Appearance verification via color histogram Bhattacharyya distance to suppress distractors.
"""

from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple
import cv2
import numpy as np


def create_csrt_tracker(padding: float = 4.5, admm_iter: int = 10):
    """
    Create and configure an OpenCV CSRT tracker instance.
    padding=4.5 expands search radius to reliably capture sudden jerks/rapid motion.
    """
    params = None
    if hasattr(cv2, "TrackerCSRT_Params"):
        params = cv2.TrackerCSRT_Params()
        params.use_segmentation = True       # Enable spatial reliability map calculation
        params.use_channel_weights = True   # Enable channel reliability weighting
        params.use_color_names = True       # Color name features for illumination invariance
        params.use_hog = True               # HOG features for edge/structure
        params.padding = float(padding)     # Wider search window for sudden jerks (default 3.0)
        params.admm_iterations = int(admm_iter)
        params.histogram_bins = 16
        params.filter_lr = 0.025            # Rapid adaptation to pose changes
        params.psr_threshold = 0.035

    if hasattr(cv2, "TrackerCSRT_create"):
        return cv2.TrackerCSRT_create(params) if params else cv2.TrackerCSRT_create()
    elif hasattr(cv2, "TrackerCSRT") and hasattr(cv2.TrackerCSRT, "create"):
        return cv2.TrackerCSRT.create(params) if params else cv2.TrackerCSRT.create()
    elif hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerCSRT_create"):
        return cv2.legacy.TrackerCSRT_create()
    else:
        raise RuntimeError(
            "OpenCV CSRT tracker not found. Ensure 'opencv-contrib-python' is installed: "
            "pip install opencv-contrib-python"
        )


def compute_spatial_reliability_map(
    frame: np.ndarray, bbox: Tuple[int, int, int, int], bg_factor: float = 2.0, num_bins: int = 16
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Computes the spatial reliability map m(x) for the target ROI.

    Mathematics of CSRT Spatial Reliability:
      1. Computes Hue-Saturation 2D color histograms for target foreground (B)
         and surrounding background context (W \\ B).
      2. Computes Bayes posterior probability:
            P(fg | c) = P(c | fg) / (P(c | fg) + P(c | bg) + eps)
      3. Applies an Epanechnikov 2D spatial prior centered on target.
      4. Suppresses stationary background pixels while assigning high weights to foreground pixels.

    Returns:
      (heatmap_bgr, reliability_mask_gray) or (None, None) if ROI is invalid.
    """
    x, y, w, h = [int(v) for v in bbox]
    img_h, img_w = frame.shape[:2]

    # Target ROI bounds
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(img_w, x + w), min(img_h, y + h)

    roi_w, roi_h = x2 - x1, y2 - y1
    if roi_w < 6 or roi_h < 6:
        return None, None

    # Context window surrounding target
    pad_w = int(w * (bg_factor - 1.0) / 2.0)
    pad_h = int(h * (bg_factor - 1.0) / 2.0)
    bx1, by1 = max(0, x - pad_w), max(0, y - pad_h)
    bx2, by2 = min(img_w, x + w + pad_w), min(img_h, y + h + pad_h)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    fg_crop = hsv[y1:y2, x1:x2]
    bg_crop = hsv[by1:by2, bx1:bx2]

    # 2D Color Histograms (Hue: 0-180, Saturation: 0-256)
    hist_fg = cv2.calcHist([fg_crop], [0, 1], None, [num_bins, num_bins], [0, 180, 0, 256])
    hist_bg = cv2.calcHist([bg_crop], [0, 1], None, [num_bins, num_bins], [0, 180, 0, 256])

    cv2.normalize(hist_fg, hist_fg, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(hist_bg, hist_bg, 0, 1, cv2.NORM_MINMAX)

    # Pixel-wise Bayes foreground likelihood
    h_channel = fg_crop[:, :, 0]
    s_channel = fg_crop[:, :, 1]
    h_idx = np.clip((h_channel.astype(np.float32) / 180.0 * (num_bins - 1)).astype(np.int32), 0, num_bins - 1)
    s_idx = np.clip((s_channel.astype(np.float32) / 256.0 * (num_bins - 1)).astype(np.int32), 0, num_bins - 1)

    p_fg = hist_fg[h_idx, s_idx]
    p_bg = hist_bg[h_idx, s_idx]
    bayes_map = p_fg / (p_fg + p_bg + 1e-5)

    # Epanechnikov 2D spatial center kernel
    cy, cx = roi_h / 2.0, roi_w / 2.0
    yy, xx = np.ogrid[:roi_h, :roi_w]
    dist_sq = ((xx - cx) / (cx + 1e-5)) ** 2 + ((yy - cy) / (cy + 1e-5)) ** 2
    spatial_prior = np.maximum(0.0, 1.0 - dist_sq)

    # Combined spatial reliability map
    reliability = bayes_map * spatial_prior
    max_val = reliability.max()
    if max_val > 0:
        reliability = (reliability / max_val * 255.0).astype(np.uint8)
    else:
        reliability = np.zeros((roi_h, roi_w), dtype=np.uint8)

    heatmap = cv2.applyColorMap(reliability, cv2.COLORMAP_TURBO)
    return heatmap, reliability


@dataclass
class TrackingResult:
    success: bool
    bbox: Optional[Tuple[int, int, int, int]]
    center: Optional[Tuple[float, float]]
    confidence: float
    status: str = "IDLE"
    is_recovered: bool = False
    is_recovering: bool = False


class CSRTTrackerEngine:
    """
    Stateful CSRT Tracker managing:
    - Target initialization with color fingerprint
    - Frame-to-frame tracking
    - Distractor rejection via Bhattacharyya histogram comparison
    - Sudden-jerk re-acquisition using velocity projection & template matching
    - Trajectory tracking
    """

    def __init__(self, max_recovery_frames: int = 15, max_trajectory: int = 40):
        self.max_recovery_frames = max_recovery_frames
        self.tracker = None
        self.is_tracking = False
        self.current_bbox: Optional[Tuple[int, int, int, int]] = None
        self.target_template: Optional[np.ndarray] = None
        self.target_hist: Optional[np.ndarray] = None
        self.velocity = (0.0, 0.0)
        self.recovery_frames = 0
        self.trajectory = deque(maxlen=max_trajectory)
        self.tracking_score = 0.0

    def init(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> bool:
        """Initializes the tracker on a given bounding box (x, y, w, h)."""
        x, y, w, h = [int(v) for v in bbox]
        if w < 6 or h < 6:
            return False

        try:
            tracker = create_csrt_tracker(padding=4.5, admm_iter=10)
            tracker.init(frame, (x, y, w, h))
            self.tracker = tracker
            self.current_bbox = (x, y, w, h)
            self.is_tracking = True
            self.trajectory.clear()
            self.velocity = (0.0, 0.0)
            self.recovery_frames = 0

            # Store high-res reference template and color fingerprint
            self.target_template = frame[y : y + h, x : x + w].copy()
            init_hsv = cv2.cvtColor(self.target_template, cv2.COLOR_BGR2HSV)
            self.target_hist = cv2.calcHist([init_hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
            cv2.normalize(self.target_hist, self.target_hist, 0, 1, cv2.NORM_MINMAX)

            cx = x + w / 2.0
            cy = y + h / 2.0
            self.trajectory.append((int(cx), int(cy)))
            self.tracking_score = 1.0
            return True
        except Exception as e:
            print(f"Error initializing CSRT tracker: {e}")
            self.reset()
            return False

    def reset(self):
        """Resets tracker state."""
        self.tracker = None
        self.is_tracking = False
        self.current_bbox = None
        self.target_template = None
        self.target_hist = None
        self.velocity = (0.0, 0.0)
        self.recovery_frames = 0
        self.trajectory.clear()
        self.tracking_score = 0.0

    def update(self, frame: np.ndarray) -> TrackingResult:
        """Updates tracking state for the new video frame."""
        if not self.is_tracking or self.tracker is None:
            return TrackingResult(
                success=False,
                bbox=None,
                center=None,
                confidence=0.0,
                status="IDLE READY",
            )

        success, bbox = self.tracker.update(frame)

        if success:
            x, y, w, h = [int(v) for v in bbox]

            # Distractor verification against color fingerprint
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(frame.shape[1], x + w), min(frame.shape[0], y + h)
            crop = frame[y1:y2, x1:x2]

            is_distractor = False
            if crop.size > 0 and self.target_hist is not None:
                cand_hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
                cand_hist = cv2.calcHist([cand_hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
                cv2.normalize(cand_hist, cand_hist, 0, 1, cv2.NORM_MINMAX)
                bhatt_dist = cv2.compareHist(self.target_hist, cand_hist, cv2.HISTCMP_BHATTACHARYYA)
                if bhatt_dist > 0.62:
                    is_distractor = True

            if is_distractor:
                success = False

        if success:
            x, y, w, h = [int(v) for v in bbox]

            # Instantaneous velocity momentum (EMA filtered)
            if self.current_bbox is not None:
                prev_cx = self.current_bbox[0] + self.current_bbox[2] / 2.0
                prev_cy = self.current_bbox[1] + self.current_bbox[3] / 2.0
                new_cx = x + w / 2.0
                new_cy = y + h / 2.0
                inst_vx = new_cx - prev_cx
                inst_vy = new_cy - prev_cy
                self.velocity = (
                    0.6 * self.velocity[0] + 0.4 * inst_vx,
                    0.6 * self.velocity[1] + 0.4 * inst_vy,
                )

            self.current_bbox = (x, y, w, h)
            self.recovery_frames = 0
            cx = x + w / 2.0
            cy = y + h / 2.0
            self.trajectory.append((int(cx), int(cy)))

            # Tracking confidence score
            score = -1.0
            if hasattr(self.tracker, "getTrackingScore"):
                try:
                    score = float(self.tracker.getTrackingScore())
                except Exception:
                    score = -1.0

            if score >= 0.0:
                self.tracking_score = min(1.0, max(0.0, score))
            else:
                self.tracking_score = 0.95

            # Adaptively update reference template when tracking is solid
            if self.tracking_score > 0.6 and (
                x >= 0 and y >= 0 and x + w <= frame.shape[1] and y + h <= frame.shape[0]
            ):
                crop = frame[y : y + h, x : x + w]
                if crop.size > 0 and self.target_template is not None and crop.shape == self.target_template.shape:
                    self.target_template = cv2.addWeighted(crop, 0.1, self.target_template, 0.9, 0)

            return TrackingResult(
                success=True,
                bbox=(x, y, w, h),
                center=(cx, cy),
                confidence=self.tracking_score,
                status="TRACKING ACTIVE",
            )

        # ---------------------------------------------------------------------
        # Sudden Jerk Recovery & Re-acquisition Engine
        # ---------------------------------------------------------------------
        if (
            self.current_bbox is not None
            and self.target_template is not None
            and self.recovery_frames < self.max_recovery_frames
        ):
            self.recovery_frames += 1
            curr_x, curr_y, curr_w, curr_h = self.current_bbox
            vx, vy = self.velocity

            # 1. Project expected location along velocity momentum vector
            proj_x = int(curr_x + vx * 1.5)
            proj_y = int(curr_y + vy * 1.5)

            # 2. Search expanded neighborhood around projected position
            search_pad_x = int(curr_w * 2.0)
            search_pad_y = int(curr_h * 2.0)
            sx1 = max(0, proj_x - search_pad_x)
            sy1 = max(0, proj_y - search_pad_y)
            sx2 = min(frame.shape[1], proj_x + curr_w + search_pad_x)
            sy2 = min(frame.shape[0], proj_y + curr_h + search_pad_y)

            search_roi = frame[sy1:sy2, sx1:sx2]
            tw, th = self.target_template.shape[1], self.target_template.shape[0]

            if search_roi.shape[0] >= th and search_roi.shape[1] >= tw:
                res = cv2.matchTemplate(search_roi, self.target_template, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(res)

                if max_val >= 0.52:
                    rec_x = sx1 + max_loc[0]
                    rec_y = sy1 + max_loc[1]
                    rec_crop = frame[rec_y : rec_y + curr_h, rec_x : rec_x + curr_w]

                    b_ok = True
                    if rec_crop.size > 0 and self.target_hist is not None:
                        c_hsv = cv2.cvtColor(rec_crop, cv2.COLOR_BGR2HSV)
                        c_hist = cv2.calcHist([c_hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
                        cv2.normalize(c_hist, c_hist, 0, 1, cv2.NORM_MINMAX)
                        if cv2.compareHist(self.target_hist, c_hist, cv2.HISTCMP_BHATTACHARYYA) > 0.60:
                            b_ok = False

                    if b_ok:
                        rec_bbox = (rec_x, rec_y, curr_w, curr_h)
                        new_tracker = create_csrt_tracker(padding=4.5, admm_iter=10)
                        new_tracker.init(frame, rec_bbox)
                        self.tracker = new_tracker
                        self.current_bbox = rec_bbox
                        self.recovery_frames = 0
                        self.tracking_score = float(max_val)
                        cx = rec_x + curr_w / 2.0
                        cy = rec_y + curr_h / 2.0
                        self.trajectory.append((int(cx), int(cy)))

                        return TrackingResult(
                            success=True,
                            bbox=rec_bbox,
                            center=(cx, cy),
                            confidence=self.tracking_score,
                            status="RECOVERED TARGET",
                            is_recovered=True,
                        )

        # Coasting or completely lost
        if self.recovery_frames < self.max_recovery_frames and self.current_bbox is not None:
            vx, vy = self.velocity
            cx = self.current_bbox[0] + vx + self.current_bbox[2] / 2.0
            cy = self.current_bbox[1] + vy + self.current_bbox[3] / 2.0
            return TrackingResult(
                success=False,
                bbox=self.current_bbox,
                center=(cx, cy),
                confidence=0.2,
                status="RECOVERING FROM MOTION...",
                is_recovering=True,
            )
        else:
            self.tracking_score = 0.0
            return TrackingResult(
                success=False,
                bbox=None,
                center=None,
                confidence=0.0,
                status="TARGET LOST - RE-SELECT",
            )
