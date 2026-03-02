"""
Pitch Keypoint Detection Service

Detects 32 standard pitch keypoints (line intersections, circle points,
penalty spots, etc.) using a trained keypoint detection model.

Used to compute a robust homography for pixel-to-pitch coordinate mapping,
superior to the Hough-line based approach.

Requires a trained pitch_keypoints.pt model - until available, the existing
Hough-line calibration in PitchMapper works as fallback.
"""
import numpy as np
import cv2
from typing import List, Optional, Tuple, Dict
from pathlib import Path

from config import settings
from models.schemas import Position, PixelPosition


# 32 standard pitch keypoints (from Roboflow sports pitch config)
# These are the intersections and notable points on a standard football pitch.
# Coordinates in meters from bottom-left corner (0,0).
PITCH_KEYPOINTS = {
    # Corners
    0: ("bottom_left_corner", 0.0, 0.0),
    1: ("bottom_right_corner", 105.0, 0.0),
    2: ("top_right_corner", 105.0, 68.0),
    3: ("top_left_corner", 0.0, 68.0),

    # Halfway line
    4: ("halfway_bottom", 52.5, 0.0),
    5: ("halfway_top", 52.5, 68.0),
    6: ("center_spot", 52.5, 34.0),
    7: ("center_circle_top", 52.5, 43.15),
    8: ("center_circle_bottom", 52.5, 24.85),
    9: ("center_circle_left", 43.35, 34.0),
    10: ("center_circle_right", 61.65, 34.0),

    # Left penalty area (16.5m from goal line, 40.32m wide)
    11: ("left_penalty_top_left", 0.0, 54.16),
    12: ("left_penalty_top_right", 16.5, 54.16),
    13: ("left_penalty_bottom_right", 16.5, 13.84),
    14: ("left_penalty_bottom_left", 0.0, 13.84),
    15: ("left_penalty_spot", 11.0, 34.0),

    # Left goal area (5.5m from goal line, 18.32m wide)
    16: ("left_goal_area_top_left", 0.0, 43.16),
    17: ("left_goal_area_top_right", 5.5, 43.16),
    18: ("left_goal_area_bottom_right", 5.5, 24.84),
    19: ("left_goal_area_bottom_left", 0.0, 24.84),

    # Right penalty area
    20: ("right_penalty_top_right", 105.0, 54.16),
    21: ("right_penalty_top_left", 88.5, 54.16),
    22: ("right_penalty_bottom_left", 88.5, 13.84),
    23: ("right_penalty_bottom_right", 105.0, 13.84),
    24: ("right_penalty_spot", 94.0, 34.0),

    # Right goal area
    25: ("right_goal_area_top_right", 105.0, 43.16),
    26: ("right_goal_area_top_left", 99.5, 43.16),
    27: ("right_goal_area_bottom_left", 99.5, 24.84),
    28: ("right_goal_area_bottom_right", 105.0, 24.84),

    # Goal posts (on the goal line)
    29: ("left_goal_top", 0.0, 37.66),
    30: ("left_goal_bottom", 0.0, 30.34),
    31: ("right_goal_top", 105.0, 37.66),
}
# Note: index 31 brings us to 32 keypoints (0-31)


class PitchKeypointDetector:
    """
    Detects pitch keypoints from a video frame using a trained model.

    Uses the keypoints to compute a homography transformation from
    pixel coordinates to pitch coordinates (meters).
    """

    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.model_path = model_path or settings.PITCH_KEYPOINT_MODEL_PATH
        self._loaded = False

    def _load_model(self) -> bool:
        """Lazy-load the keypoint detection model."""
        if self._loaded:
            return self.model is not None

        model_file = Path(self.model_path)
        if not model_file.exists():
            print(f"[PITCH_KP] Model not found at {self.model_path}, using fallback calibration")
            self._loaded = True
            return False

        try:
            from ultralytics import YOLO
            self.model = YOLO(str(model_file))
            if not settings.USE_GPU:
                self.model.to('cpu')
            print(f"[PITCH_KP] Keypoint model loaded: {self.model_path}")
            self._loaded = True
            return True
        except Exception as e:
            print(f"[PITCH_KP] Failed to load keypoint model: {e}")
            self._loaded = True
            return False

    def detect_keypoints(
        self, frame: np.ndarray, confidence_threshold: float = 0.5
    ) -> Dict[int, Tuple[float, float, float]]:
        """
        Detect pitch keypoints in a frame.

        Args:
            frame: BGR image
            confidence_threshold: Minimum confidence for a keypoint

        Returns:
            Dict mapping keypoint index -> (pixel_x, pixel_y, confidence)
        """
        if not self._load_model():
            return {}

        results = self.model(frame, verbose=False)

        detected = {}
        for result in results:
            if result.keypoints is None:
                continue

            keypoints = result.keypoints
            if keypoints.xy is None:
                continue

            # keypoints.xy shape: (N, num_keypoints, 2)
            # keypoints.conf shape: (N, num_keypoints)
            for person_idx in range(keypoints.xy.shape[0]):
                kps = keypoints.xy[person_idx]  # (num_keypoints, 2)
                confs = keypoints.conf[person_idx] if keypoints.conf is not None else None

                for kp_idx in range(kps.shape[0]):
                    px, py = float(kps[kp_idx, 0]), float(kps[kp_idx, 1])
                    conf = float(confs[kp_idx]) if confs is not None else 1.0

                    if conf >= confidence_threshold and px > 0 and py > 0:
                        if kp_idx in PITCH_KEYPOINTS:
                            detected[kp_idx] = (px, py, conf)

        return detected

    def compute_homography(
        self, detected_keypoints: Dict[int, Tuple[float, float, float]]
    ) -> Optional[np.ndarray]:
        """
        Compute homography matrix from detected keypoints.

        Args:
            detected_keypoints: Dict from detect_keypoints()

        Returns:
            3x3 homography matrix or None if insufficient points
        """
        if len(detected_keypoints) < 4:
            print(f"[PITCH_KP] Only {len(detected_keypoints)} keypoints detected, need 4+")
            return None

        src_points = []  # pixel coordinates
        dst_points = []  # pitch coordinates (meters)

        for kp_idx, (px, py, conf) in detected_keypoints.items():
            if kp_idx in PITCH_KEYPOINTS:
                name, pitch_x, pitch_y = PITCH_KEYPOINTS[kp_idx]
                src_points.append([px, py])
                dst_points.append([pitch_x, pitch_y])

        src = np.array(src_points, dtype=np.float32)
        dst = np.array(dst_points, dtype=np.float32)

        if len(src) == 4:
            H = cv2.getPerspectiveTransform(src, dst)
        else:
            H, status = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
            if status is not None:
                inliers = status.sum()
                print(f"[PITCH_KP] Homography: {inliers}/{len(src)} inliers")

        return H

    def get_pitch_coordinates(self, kp_idx: int) -> Optional[Tuple[float, float]]:
        """Get the real-world pitch coordinates for a keypoint index."""
        if kp_idx in PITCH_KEYPOINTS:
            _, x, y = PITCH_KEYPOINTS[kp_idx]
            return (x, y)
        return None


def get_all_keypoint_positions() -> Dict[str, Position]:
    """Get all 32 keypoint positions as Position objects."""
    positions = {}
    for idx, (name, x, y) in PITCH_KEYPOINTS.items():
        positions[name] = Position(x=x, y=y)
    return positions
