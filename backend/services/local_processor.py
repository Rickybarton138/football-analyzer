"""
Local CPU Video Processing Service

Processes video locally using CPU for player detection and tracking.
Slower than GPU but works without cloud setup.

Detection pipeline targets:
  - 20 outfield players (10 per team)
  - 2 goalkeepers
  - 1 referee
  - Filters out: linesmen, coaching staff, substitutes, ball boys
"""
import cv2
import numpy as np
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from pathlib import Path
from collections import deque
from sklearn.cluster import KMeans
import json
import time
from datetime import datetime

# Import for player highlights tracking and jersey detection
from models.schemas import DetectedPlayer, BoundingBox, PixelPosition, TeamSide
from services.player_highlights import player_highlights_service
from services.jersey_ocr import jersey_ocr_service
from services.ai_jersey_detection import ai_jersey_detection_service
from services.tracking import TrackingService
from config import settings

# Import professional analytics services (VEO-style)
from services.match_statistics import match_statistics_service, EventType as StatsEventType
from services.pitch_visualization import pitch_visualization_service
from services.event_detector import event_detector, DetectedEventType
from services.coach_assist import coach_assist_service

# Phase 1: Wire existing analytics services
from services.pass_detector import pass_detector
from services.xg_model import xg_model, Shot
from services.formation_detector import formation_detector
from services.tactical_events import tactical_detector

# Phase 2: AI Coaching Intelligence
from services.tactical_intelligence import tactical_intelligence
from services.ai_coach import ai_coach


@dataclass
class Detection:
    """Single detection in a frame."""
    bbox: List[float]  # [x1, y1, x2, y2]
    confidence: float
    class_id: int
    class_name: str
    track_id: Optional[int] = None
    team: Optional[str] = None  # 'home', 'away', 'referee'


class BallTracker:
    """Enhanced ball tracking using multiple detection methods."""

    def __init__(self):
        self.prev_frame: Optional[np.ndarray] = None
        self.prev_positions: deque = deque(maxlen=10)
        self.kalman = self._init_kalman()

    def _init_kalman(self):
        """Initialize Kalman filter for ball trajectory prediction."""
        kalman = cv2.KalmanFilter(4, 2)
        kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        kalman.transitionMatrix = np.array([
            [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]
        ], np.float32)
        kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5
        return kalman

    def detect_ball(self, frame: np.ndarray, yolo_ball_pos: Optional[List[float]] = None,
                    player_boxes: List[List[float]] = None) -> Optional[List[float]]:
        """
        Detect ball using multiple methods combined.

        Returns: [center_x, center_y] or None
        """
        candidates = []

        # Method 1: YOLO detection (if available)
        if yolo_ball_pos:
            candidates.append(('yolo', yolo_ball_pos, 0.9))

        # Method 2: Motion-based detection
        motion_pos = self._detect_motion(frame)
        if motion_pos:
            candidates.append(('motion', motion_pos, 0.7))

        # Method 3: Color-based detection (white ball)
        color_pos = self._detect_color(frame, player_boxes or [])
        if color_pos:
            candidates.append(('color', color_pos, 0.6))

        # Method 4: Kalman prediction (if we have history)
        if len(self.prev_positions) >= 3:
            pred_pos = self._predict_position()
            if pred_pos:
                candidates.append(('kalman', pred_pos, 0.4))

        # Store frame for next iteration
        self.prev_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame.copy()

        if not candidates:
            return None

        # Select best candidate
        candidates.sort(key=lambda x: x[2], reverse=True)
        best_method, best_pos, conf = candidates[0]

        # Validate position is reasonable
        if self._validate_position(best_pos, frame.shape):
            # Update tracking
            self.prev_positions.append(best_pos)
            self._update_kalman(best_pos)
            return best_pos

        return None

    def _detect_motion(self, frame: np.ndarray) -> Optional[List[float]]:
        """Detect ball using frame differencing."""
        if self.prev_frame is None:
            return None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame

        # Ensure frames have same size (handle resolution changes)
        if gray.shape != self.prev_frame.shape:
            self.prev_frame = cv2.resize(self.prev_frame, (gray.shape[1], gray.shape[0]))

        # Frame difference
        diff = cv2.absdiff(gray, self.prev_frame)
        _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

        # Clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_candidate = None
        best_score = 0

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 30 or area > 3000:  # Ball-sized objects
                continue

            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue

            circularity = 4 * np.pi * area / (perimeter ** 2)
            if circularity < 0.4:  # Must be somewhat circular
                continue

            (x, y), radius = cv2.minEnclosingCircle(contour)
            if radius < 3 or radius > 40:
                continue

            # Score based on circularity and size
            score = circularity * min(1.0, area / 500)
            if score > best_score:
                best_score = score
                best_candidate = [float(x), float(y)]

        return best_candidate

    def _detect_color(self, frame: np.ndarray, player_boxes: List[List[float]]) -> Optional[List[float]]:
        """Detect white/light ball using color filtering."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Create mask for player areas to exclude
        player_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        for box in player_boxes:
            x1, y1, x2, y2 = map(int, box[:4])
            player_mask[max(0,y1):min(frame.shape[0],y2), max(0,x1):min(frame.shape[1],x2)] = 255

        # White ball detection (standard football)
        lower_white = np.array([0, 0, 180])
        upper_white = np.array([180, 50, 255])
        white_mask = cv2.inRange(hsv, lower_white, upper_white)

        # Also try yellow/orange (some balls)
        lower_yellow = np.array([15, 80, 150])
        upper_yellow = np.array([35, 255, 255])
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

        # Combine masks
        ball_mask = cv2.bitwise_or(white_mask, yellow_mask)

        # Remove player areas
        ball_mask = cv2.bitwise_and(ball_mask, cv2.bitwise_not(player_mask))

        # Clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        ball_mask = cv2.morphologyEx(ball_mask, cv2.MORPH_OPEN, kernel)
        ball_mask = cv2.morphologyEx(ball_mask, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(ball_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_candidate = None
        best_score = 0

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 20 or area > 2000:
                continue

            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue

            circularity = 4 * np.pi * area / (perimeter ** 2)
            if circularity < 0.5:
                continue

            (x, y), radius = cv2.minEnclosingCircle(contour)
            if radius < 3 or radius > 35:
                continue

            score = circularity
            if score > best_score:
                best_score = score
                best_candidate = [float(x), float(y)]

        return best_candidate

    def _predict_position(self) -> Optional[List[float]]:
        """Predict ball position using Kalman filter."""
        try:
            prediction = self.kalman.predict()
            x, y = float(prediction[0][0]), float(prediction[1][0])
            return [x, y]
        except:
            return None

    def _update_kalman(self, pos: List[float]):
        """Update Kalman filter with measured position."""
        measurement = np.array([[pos[0]], [pos[1]]], dtype=np.float32)
        self.kalman.correct(measurement)

    def _validate_position(self, pos: List[float], frame_shape: Tuple) -> bool:
        """Validate ball position is within frame bounds."""
        h, w = frame_shape[:2]
        x, y = pos
        return 0 <= x < w and 0 <= y < h

    def reset(self):
        """Reset tracker state."""
        self.prev_frame = None
        self.prev_positions.clear()
        self.kalman = self._init_kalman()


@dataclass
class FrameAnalysis:
    """Analysis result for a single frame."""
    frame_number: int
    timestamp: float
    detections: List[Detection] = field(default_factory=list)
    ball_position: Optional[List[float]] = None
    player_count: int = 0
    home_players: int = 0
    away_players: int = 0


@dataclass
class MatchAnalysis:
    """Full match analysis result."""
    video_path: str
    duration_seconds: float
    total_frames: int
    analyzed_frames: int
    fps_analyzed: float
    start_time: str
    end_time: Optional[str] = None
    frame_analyses: List[FrameAnalysis] = field(default_factory=list)

    # Aggregated stats
    avg_home_players: float = 0
    avg_away_players: float = 0
    ball_possession_estimate: Dict = field(default_factory=dict)
    heatmap_data: Optional[List] = None


class LocalVideoProcessor:
    """Process video locally using CPU."""

    # Detection thresholds - optimized for full 22-player coverage
    CONF_THRESHOLD_PRIMARY = 0.3    # Primary detection pass
    CONF_THRESHOLD_RETRY = 0.2      # Retry when too few players detected
    CONF_THRESHOLD_AGGRESSIVE = 0.12  # Aggressive retry for distant players

    # Expected player counts for validation
    EXPECTED_ON_PITCH = 23         # 20 outfield + 2 GK + 1 referee
    EXPECTED_PLAYERS_MIN = 18      # Minimum reasonable (some occlusion)
    EXPECTED_PLAYERS_MAX = 26      # Maximum (22 + refs + some overlap)

    # Multi-scale detection settings
    DETECTION_SCALES = [1.0, 1.5, 2.0]  # Upscale for distant player detection
    NMS_IOU_THRESHOLD = 0.5        # For merging multi-scale detections

    # Pitch boundary filtering (VEO camera perspective)
    # Bottom ~8% of frame is typically advertising boards / sideline area
    # Top ~3% can be crowd/sky
    PITCH_Y_MIN_RATIO = 0.03      # Top boundary (above = crowd/sky)
    PITCH_Y_MAX_RATIO = 0.92      # Bottom boundary (below = sideline/boards)
    # Sideline exclusion: detections in bottom strip with very small bbox = sideline people
    SIDELINE_Y_THRESHOLD = 0.85   # Bottom 15% needs extra validation
    SIDELINE_MIN_HEIGHT_RATIO = 0.04  # Min player height relative to frame (below = too small/far)

    def __init__(self):
        self.is_processing = False
        self.current_video = None
        self.progress = 0
        self.status = "idle"
        self.current_analysis: Optional[MatchAnalysis] = None
        self.model = None
        self.model_name: str = "none"
        self.is_football_model: bool = False
        self.ball_tracker = BallTracker()
        self.tracking_service = TrackingService()  # Real multi-object tracker
        self.use_multi_scale = False  # Disable for faster processing (set True for better detection)
        self.track_counter = 0  # Legacy fallback counter

        # Team color classification (auto-detected)
        self._home_color: Optional[np.ndarray] = None  # BGR
        self._away_color: Optional[np.ndarray] = None  # BGR
        self._team_colors_detected: bool = False
        self._color_samples: List[np.ndarray] = []  # Collect during warmup

        # Possession inertia state (prevents flickering)
        self._possession_track_id: Optional[int] = None
        self._possession_team: Optional[str] = None
        self._possession_inertia: int = 0

        # Phase 2: Cached coaching intelligence results
        self._coaching_analysis: Optional[Dict] = None
        self._tactical_alerts_summary: Optional[Dict] = None
        self._training_focus: Optional[Dict] = None

    def _determine_ball_possession(
        self,
        ball_pos: Optional[List[float]],
        player_detections: List[Dict]
    ) -> Tuple[Optional[int], Optional[str]]:
        """
        Determine which player has possession of the ball.
        Uses 5-frame inertia to prevent flickering between players.

        Returns (track_id, team_str) of the possessing player, or (None, None).
        """
        POSSESSION_DISTANCE = 80  # pixels (~2m real-world)
        INERTIA_FRAMES = 5  # Must see different player for N frames before switching

        if ball_pos is None or not player_detections:
            return (self._possession_track_id, self._possession_team)

        closest_player = None
        closest_team = None
        min_distance = float('inf')

        for det in player_detections:
            bbox = det['bbox']
            player_x = (bbox[0] + bbox[2]) / 2
            player_y = bbox[3]  # Bottom of bbox (feet)

            dist = np.sqrt((player_x - ball_pos[0])**2 + (player_y - ball_pos[1])**2)

            if dist < min_distance and dist < POSSESSION_DISTANCE:
                min_distance = dist
                closest_player = det.get('track_id')
                closest_team = det.get('team')

        if closest_player is None:
            return (self._possession_track_id, self._possession_team)

        # Apply inertia: only switch after INERTIA_FRAMES consecutive frames
        if closest_player != self._possession_track_id:
            self._possession_inertia += 1
            if self._possession_inertia >= INERTIA_FRAMES:
                self._possession_track_id = closest_player
                self._possession_team = closest_team
                self._possession_inertia = 0
        else:
            self._possession_inertia = 0

        return (self._possession_track_id, self._possession_team)

    def _load_model(self):
        """Load YOLO model using proper fallback chain from config."""
        if self.model is None:
            try:
                from ultralytics import YOLO

                # Use the proper model fallback chain (football_best.pt → yolo11m → yolov8m)
                model_chain = settings.YOLO_MODEL_CHAIN if settings.USE_GPU else settings.YOLO_MODEL_CHAIN_CPU

                for model_path in model_chain:
                    try:
                        print(f"[PROCESSOR] Trying model: {model_path}")
                        self.model = YOLO(model_path)
                        self.model_name = model_path

                        if not settings.USE_GPU:
                            self.model.to('cpu')
                            print(f"[PROCESSOR] Model loaded: {model_path} (CPU)")
                        else:
                            print(f"[PROCESSOR] Model loaded: {model_path} (GPU)")

                        self.is_football_model = "football" in model_path
                        if self.is_football_model:
                            print("[PROCESSOR] Using fine-tuned football model (classes: ball=0, goalkeeper=1, player=2, referee=3)")
                        break
                    except Exception as e:
                        print(f"[PROCESSOR] Failed to load {model_path}: {e}")
                        continue

                if self.model is None:
                    print("[PROCESSOR] All models in fallback chain failed. Using mock detection.")

            except ImportError:
                print("ultralytics not installed, using mock detection")
                self.model = None
        return self.model

    def _extract_jersey_color(self, frame: np.ndarray, bbox: List[float]) -> Optional[np.ndarray]:
        """Extract dominant jersey color from player bounding box (torso region)."""
        x1, y1, x2, y2 = map(int, bbox)

        h_frame, w_frame = frame.shape[:2]
        x1, x2 = max(0, x1), min(w_frame, x2)
        y1, y2 = max(0, y1), min(h_frame, y2)

        h = y2 - y1
        w = x2 - x1
        if h < 10 or w < 5:
            return None

        # Focus on torso (upper-middle of bbox)
        if h < 30 or w < 15:
            torso_y1 = max(0, int(h * 0.25))
            torso_y2 = min(h, int(h * 0.75))
            torso_x1 = max(0, int(w * 0.1))
            torso_x2 = min(w, int(w * 0.9))
        else:
            torso_y1 = int(h * 0.15)
            torso_y2 = int(h * 0.5)
            torso_x1 = int(w * 0.2)
            torso_x2 = int(w * 0.8)

        roi = frame[y1:y2, x1:x2]
        torso = roi[torso_y1:torso_y2, torso_x1:torso_x2]
        if torso.size == 0:
            torso = roi

        pixels = torso.reshape(-1, 3)

        # Filter out grass colors, dark pixels, and overly bright pixels
        pixels_bgr = pixels.reshape(-1, 1, 3).astype(np.uint8)
        pixels_hsv = cv2.cvtColor(pixels_bgr, cv2.COLOR_BGR2HSV).reshape(-1, 3)

        filtered = []
        for i, pixel in enumerate(pixels):
            h_val, s, v = pixels_hsv[i]
            b, g, r = pixel
            is_grass = (35 <= h_val <= 85 and s > 30 and v > 30)
            is_dark = (v < 30)
            is_bright = (r > 220 and g > 220 and b > 220)
            if not is_grass and not is_dark and not is_bright:
                filtered.append(pixel)

        if len(filtered) < 10:
            filtered = pixels.tolist()

        pixels = np.array(filtered)

        try:
            n_clusters = min(3, len(pixels))
            if n_clusters < 1:
                return None
            kmeans = KMeans(n_clusters=n_clusters, n_init=5, max_iter=100, random_state=42)
            kmeans.fit(pixels)

            labels, counts = np.unique(kmeans.labels_, return_counts=True)
            best_score = -1
            best_color = None

            for idx in range(n_clusters):
                center = kmeans.cluster_centers_[idx]
                cluster_size = counts[labels == idx][0] if idx in labels else 0
                center_bgr = center.astype(np.uint8).reshape(1, 1, 3)
                center_hsv = cv2.cvtColor(center_bgr, cv2.COLOR_BGR2HSV).reshape(3)
                h_val, s, v = center_hsv

                score = float(cluster_size)
                # Penalize red/orange (could be red card, advertising)
                if h_val <= 25 and 30 < s < 170 and v > 60:
                    score *= 0.3
                # Penalize grass colors
                if 35 <= h_val <= 85 and s > 30 and v > 30:
                    score *= 0.2

                if score > best_score:
                    best_score = score
                    best_color = center

            return best_color.astype(np.uint8) if best_color is not None else None
        except Exception:
            return np.mean(pixels, axis=0).astype(np.uint8) if len(pixels) > 0 else None

    def _bgr_to_lab(self, bgr_color: np.ndarray) -> np.ndarray:
        """Convert a single BGR color to CIELAB."""
        bgr_pixel = bgr_color.astype(np.uint8).reshape(1, 1, 3)
        lab_pixel = cv2.cvtColor(bgr_pixel, cv2.COLOR_BGR2LAB)
        return lab_pixel.reshape(3).astype(float)

    def _auto_detect_team_colors(self, colors: List[np.ndarray]):
        """Auto-detect home/away team colors using KMeans on collected jersey colors."""
        if len(colors) < 6:
            return

        color_array = np.array(colors)
        try:
            kmeans = KMeans(n_clusters=2, n_init=10, random_state=42)
            kmeans.fit(color_array)
            self._home_color = kmeans.cluster_centers_[0].astype(np.uint8)
            self._away_color = kmeans.cluster_centers_[1].astype(np.uint8)
            self._team_colors_detected = True

            home_lab = self._bgr_to_lab(self._home_color)
            away_lab = self._bgr_to_lab(self._away_color)
            separation = np.linalg.norm(home_lab - away_lab)
            print(f"[PROCESSOR] Auto-detected team colors: "
                  f"Home BGR={self._home_color.tolist()}, Away BGR={self._away_color.tolist()}, "
                  f"LAB separation={separation:.1f}")
        except Exception as e:
            print(f"[PROCESSOR] Team color auto-detection failed: {e}")

    def _classify_team(self, jersey_color: Optional[np.ndarray]) -> str:
        """Classify team using CIELAB color distance (not hardcoded red!)."""
        if jersey_color is None or not self._team_colors_detected:
            return 'unknown'

        color_lab = self._bgr_to_lab(jersey_color)
        home_lab = self._bgr_to_lab(self._home_color)
        away_lab = self._bgr_to_lab(self._away_color)

        home_dist = np.linalg.norm(color_lab - home_lab)
        away_dist = np.linalg.norm(color_lab - away_lab)

        min_dist = min(home_dist, away_dist)
        max_dist = max(home_dist, away_dist)

        # Too far from both teams = referee or other non-player
        if min_dist > settings.TEAM_COLOR_DISTANCE_THRESHOLD:
            return 'referee'

        # Too ambiguous (colors similar distance from both teams)
        if max_dist > 0 and min_dist / max_dist > settings.TEAM_AMBIGUITY_RATIO:
            return 'unknown'

        return 'home' if home_dist < away_dist else 'away'

    def _is_referee_by_color(self, jersey_color: Optional[np.ndarray]) -> bool:
        """Detect referee by jersey color characteristics (black/yellow/fluoro)."""
        if jersey_color is None:
            return False

        bgr_pixel = jersey_color.astype(np.uint8).reshape(1, 1, 3)
        hsv_pixel = cv2.cvtColor(bgr_pixel, cv2.COLOR_BGR2HSV).reshape(3)
        h, s, v = hsv_pixel

        is_black_kit = (v < 60 and s < 80)
        is_yellow_kit = (25 <= h <= 35 and s > 100 and v > 100)
        is_fluoro_green = (35 <= h <= 75 and s > 150 and v > 150)

        if not (is_black_kit or is_yellow_kit or is_fluoro_green):
            return False

        # Confirm by checking distance from both team colors
        if self._team_colors_detected:
            color_lab = self._bgr_to_lab(jersey_color)
            home_lab = self._bgr_to_lab(self._home_color)
            away_lab = self._bgr_to_lab(self._away_color)
            home_dist = np.linalg.norm(color_lab - home_lab)
            away_dist = np.linalg.norm(color_lab - away_lab)
            if home_dist > 50 and away_dist > 50:
                return True

        return False

    def _is_on_pitch(self, x1: float, y1: float, x2: float, y2: float,
                     frame_w: int, frame_h: int) -> bool:
        """
        Filter out detections that are clearly off the pitch.
        Removes: linesmen at touchlines, coaching staff, substitutes, ball boys.

        VEO camera perspective:
          - Bottom of frame = near touchline (camera side, sideline people here)
          - Top of frame = far touchline
          - Very bottom strip = advertising boards, coaching area
        """
        center_y = (y1 + y2) / 2
        center_x = (x1 + x2) / 2
        bbox_h = y2 - y1
        bbox_w = x2 - x1
        y_ratio = center_y / frame_h
        h_ratio = bbox_h / frame_h

        # Above top boundary (crowd/sky)
        if y_ratio < self.PITCH_Y_MIN_RATIO:
            return False

        # Below bottom boundary (advertising boards, coaching zone)
        if y_ratio > self.PITCH_Y_MAX_RATIO:
            return False

        # In the sideline zone (bottom portion): need extra validation
        # Sideline people tend to be stationary, partially cut off, or very small
        if y_ratio > self.SIDELINE_Y_THRESHOLD:
            # Very small detections in sideline zone are likely coaches/subs
            if h_ratio < self.SIDELINE_MIN_HEIGHT_RATIO:
                return False
            # Detections at extreme left/right edges in bottom zone = linesmen
            x_ratio = center_x / frame_w
            if x_ratio < 0.05 or x_ratio > 0.95:
                return False

        # Aspect ratio check (humans are taller than wide)
        if bbox_w > 0 and bbox_h > 0:
            aspect_ratio = bbox_h / bbox_w
            if aspect_ratio < settings.DETECTION_ASPECT_RATIO_MIN or aspect_ratio > settings.DETECTION_ASPECT_RATIO_MAX:
                return False

        # Minimum size check
        if bbox_w < 5 or bbox_h < 10:
            return False

        # Maximum size check (too large = false detection)
        if bbox_w > frame_w * 0.3 or bbox_h > frame_h * 0.5:
            return False

        return True

    def _detect_players_multiscale(
        self,
        frame: np.ndarray,
        model,
        conf_override: Optional[float] = None
    ) -> Tuple[List[Dict], Optional[List[float]]]:
        """
        Detect players with support for football model classes and pitch boundary filtering.

        Handles both:
          - Football model: ball=0, goalkeeper=1, player=2, referee=3
          - COCO model: person=0, sports_ball=32

        Returns:
            Tuple of (player_detections, ball_position)
        """
        height, width = frame.shape[:2]
        all_detections = []
        ball_position = None

        conf = conf_override or self.CONF_THRESHOLD_PRIMARY
        scales_to_use = self.DETECTION_SCALES if self.use_multi_scale else [1.0]

        for scale in scales_to_use:
            if scale == 1.0:
                scaled_frame = frame
                scale_conf = conf
            else:
                new_w = int(width * scale)
                new_h = int(height * scale)
                scaled_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                scale_conf = max(0.10, conf - 0.05)  # Lower conf for upscaled

            # Determine classes to detect
            if self.is_football_model:
                classes = None  # All football model classes
            else:
                classes = [settings.PLAYER_CLASS_ID]  # COCO: person only

            results = model(
                scaled_frame,
                conf=scale_conf,
                classes=classes,
                verbose=False,
                imgsz=settings.DETECTION_IMGSZ
            )

            for r in results:
                boxes = r.boxes
                if boxes is None:
                    continue

                for box in boxes:
                    cls = int(box.cls[0])
                    box_conf = float(box.conf[0])

                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    if scale != 1.0:
                        x1 /= scale
                        y1 /= scale
                        x2 /= scale
                        y2 /= scale

                    # Clamp to frame bounds
                    x1 = max(0, min(x1, width - 1))
                    y1 = max(0, min(y1, height - 1))
                    x2 = max(0, min(x2, width))
                    y2 = max(0, min(y2, height))

                    if self.is_football_model:
                        # Football model classes: ball=0, goalkeeper=1, player=2, referee=3
                        if cls == 0:  # Ball
                            if ball_position is None and box_conf > 0.15:
                                ball_position = [(x1 + x2) / 2, (y1 + y2) / 2]
                            continue

                        if cls in (1, 2, 3):  # goalkeeper, player, referee
                            # Apply pitch boundary filter
                            if not self._is_on_pitch(x1, y1, x2, y2, width, height):
                                continue
                            role = 'goalkeeper' if cls == 1 else ('referee' if cls == 3 else 'player')
                            all_detections.append({
                                'bbox': [x1, y1, x2, y2],
                                'confidence': box_conf,
                                'scale': scale,
                                'role': role,
                            })
                    else:
                        # COCO model: person=0, sports_ball=32
                        if cls == settings.PLAYER_CLASS_ID:
                            if not self._is_on_pitch(x1, y1, x2, y2, width, height):
                                continue
                            all_detections.append({
                                'bbox': [x1, y1, x2, y2],
                                'confidence': box_conf,
                                'scale': scale,
                                'role': 'player',
                            })
                        elif cls == settings.SPORTS_BALL_CLASS_ID and box_conf > 0.12 and ball_position is None:
                            ball_position = [(x1 + x2) / 2, (y1 + y2) / 2]

        merged = self._apply_nms(all_detections)

        # Adaptive retry: if too few detections, retry with lower confidence
        if len(merged) < self.EXPECTED_PLAYERS_MIN and conf > self.CONF_THRESHOLD_RETRY:
            print(f"[PROCESSOR] Only {len(merged)} detections at conf={conf:.2f}, retrying at {self.CONF_THRESHOLD_RETRY:.2f}")
            retry_dets, retry_ball = self._detect_players_multiscale(
                frame, model, conf_override=self.CONF_THRESHOLD_RETRY
            )
            if len(retry_dets) > len(merged):
                merged = retry_dets
                if retry_ball and not ball_position:
                    ball_position = retry_ball

        # Second retry with aggressive threshold
        if len(merged) < self.EXPECTED_PLAYERS_MIN and conf > self.CONF_THRESHOLD_AGGRESSIVE:
            if conf_override != self.CONF_THRESHOLD_RETRY:  # Avoid double-retry
                pass  # Already retried above
            else:
                print(f"[PROCESSOR] Still only {len(merged)} detections, aggressive retry at {self.CONF_THRESHOLD_AGGRESSIVE:.2f}")
                retry_dets2, retry_ball2 = self._detect_players_multiscale(
                    frame, model, conf_override=self.CONF_THRESHOLD_AGGRESSIVE
                )
                if len(retry_dets2) > len(merged):
                    merged = retry_dets2
                    if retry_ball2 and not ball_position:
                        ball_position = retry_ball2

        return merged, ball_position

    def _apply_nms(self, detections: List[Dict]) -> List[Dict]:
        """
        Apply Non-Maximum Suppression to merge overlapping detections.
        """
        if not detections:
            return []

        # Sort by confidence (highest first)
        detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)

        kept = []
        suppressed = set()

        for i, det_i in enumerate(detections):
            if i in suppressed:
                continue

            kept.append(det_i)

            for j, det_j in enumerate(detections[i + 1:], start=i + 1):
                if j in suppressed:
                    continue

                iou = self._calculate_iou(det_i['bbox'], det_j['bbox'])
                if iou > self.NMS_IOU_THRESHOLD:
                    suppressed.add(j)

        return kept

    def _calculate_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Calculate Intersection over Union between two bounding boxes."""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)

        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

        union = area1 + area2 - intersection

        return intersection / max(union, 1e-6)

    async def process_video(
        self,
        video_path: str,
        fps: int = 5,  # Lower FPS for CPU
        output_path: Optional[str] = None,
        progress_callback: Optional[callable] = None
    ) -> MatchAnalysis:
        """
        Process a video file for player detection and tracking.

        Args:
            video_path: Path to video file
            fps: Frames per second to analyze (lower = faster)
            output_path: Optional path to save analysis JSON
            progress_callback: Optional callback for progress updates
        """
        self.is_processing = True
        self.current_video = video_path
        self.status = "loading"

        # Reset ball tracker for new video
        self.ball_tracker.reset()

        # Load model
        model = self._load_model()

        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self.status = "error"
            self.is_processing = False
            raise ValueError(f"Could not open video: {video_path}")

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / video_fps
        frame_skip = max(1, int(video_fps / fps))

        self.current_analysis = MatchAnalysis(
            video_path=video_path,
            duration_seconds=duration,
            total_frames=total_frames,
            analyzed_frames=0,
            fps_analyzed=fps,
            start_time=datetime.now().isoformat()
        )

        # Initialize player highlights service for clip extraction
        player_highlights_service.reset()
        player_highlights_service.set_video_info(
            video_path=video_path,
            fps=video_fps,
            width=frame_width,
            height=frame_height,
            total_frames=total_frames
        )
        self.track_counter = 0  # Reset track counter for new video
        self.tracking_service.reset()  # Reset tracker for new video

        # Reset team color auto-detection for new video
        self._home_color = None
        self._away_color = None
        self._team_colors_detected = False
        self._color_samples = []

        # Reset possession inertia
        self._possession_track_id = None
        self._possession_team = None
        self._possession_inertia = 0

        # Reset Phase 1 analytics services
        pass_detector.reset()
        xg_model.reset()
        formation_detector.reset()
        tactical_detector.reset()

        # Initialize professional analytics services (VEO-style)
        match_statistics_service.reset()
        match_statistics_service.set_video_info(fps=video_fps, total_frames=total_frames)

        pitch_visualization_service.reset()
        pitch_visualization_service.set_video_info(fps=video_fps, total_frames=total_frames)

        event_detector.reset()
        event_detector.set_video_info(fps=video_fps, total_frames=total_frames, frame_width=frame_width, frame_height=frame_height)

        coach_assist_service.reset()

        # Reset Phase 2 coaching intelligence services
        tactical_intelligence.reset()
        ai_coach.reset()
        self._coaching_analysis = None
        self._tactical_alerts_summary = None
        self._training_focus = None

        # Initialize jersey detection services
        # AI jersey detection (GPT-4V/Claude Vision) is preferred over traditional OCR
        jersey_ocr_service.reset()
        ai_jersey_detection_service.reset()

        ai_jersey_enabled = False
        ocr_enabled = False

        # Try AI jersey detection first (more accurate)
        if settings.AI_JERSEY_DETECTION_ENABLED and settings.OPENAI_API_KEY:
            try:
                ai_jersey_enabled = await ai_jersey_detection_service.initialize(
                    openai_api_key=settings.OPENAI_API_KEY,
                    anthropic_api_key=getattr(settings, 'ANTHROPIC_API_KEY', None),
                    provider=settings.AI_JERSEY_PROVIDER
                )
                if ai_jersey_enabled:
                    print("AI Jersey Detection: Initialized successfully")
            except Exception as e:
                print(f"AI Jersey Detection initialization failed: {e}")
                ai_jersey_enabled = False

        # Fall back to traditional OCR if AI not available
        if not ai_jersey_enabled:
            try:
                await jersey_ocr_service.initialize()
                ocr_enabled = jersey_ocr_service.ocr_engine is not None
                if ocr_enabled:
                    print("Traditional OCR: Initialized as fallback")
            except Exception as e:
                print(f"OCR initialization failed: {e}. Continuing without jersey detection.")
                ocr_enabled = False

        self.status = "processing"
        frame_count = 0
        analyzed_count = 0

        home_player_counts = []
        away_player_counts = []

        start_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_skip == 0:
                timestamp = frame_count / video_fps

                frame_analysis = FrameAnalysis(
                    frame_number=frame_count,
                    timestamp=timestamp
                )

                # Run multi-scale detection for better coverage of all 22 players
                player_boxes = []
                yolo_ball_pos = None
                detected_players_for_highlights = []  # For player clip tracking

                if model is not None:
                    # Use enhanced detection with football model support + pitch filtering
                    player_detections, yolo_ball_pos = self._detect_players_multiscale(frame, model)

                    # Convert raw detections to DetectedPlayer objects for the tracker
                    raw_detected_players = []
                    for det in player_detections:
                        bbox = det['bbox']
                        conf = det['confidence']
                        role = det.get('role', 'player')
                        player_boxes.append(bbox)

                        # Extract jersey color for team classification
                        jersey_color = self._extract_jersey_color(frame, bbox)

                        # Collect color samples for auto-detection (first 30 frames)
                        if not self._team_colors_detected and jersey_color is not None:
                            self._color_samples.append(jersey_color)
                            # Auto-detect team colors once we have enough samples
                            if len(self._color_samples) >= 30:
                                self._auto_detect_team_colors(self._color_samples)

                        # Determine team and role
                        is_gk = (role == 'goalkeeper')
                        is_ref = (role == 'referee')

                        if is_ref:
                            team = 'referee'
                            team_side = TeamSide.UNKNOWN
                        elif self._team_colors_detected:
                            # Use CIELAB distance classification
                            team = self._classify_team(jersey_color)
                            # Also check referee by color for COCO model
                            if team == 'referee' or (not self.is_football_model and self._is_referee_by_color(jersey_color)):
                                team = 'referee'
                                team_side = TeamSide.UNKNOWN
                                is_ref = True
                            else:
                                team_side = TeamSide.HOME if team == 'home' else (
                                    TeamSide.AWAY if team == 'away' else TeamSide.UNKNOWN
                                )
                        else:
                            # Team colors not yet detected — mark unknown for now
                            team = 'unknown'
                            team_side = TeamSide.UNKNOWN

                        # Goalkeeper detection for COCO model (position-based)
                        if not self.is_football_model and not is_gk and not is_ref:
                            center_x = (bbox[0] + bbox[2]) / 2
                            if center_x < frame_width * 0.12 or center_x > frame_width * 0.88:
                                # Near goal line + distinct color = likely goalkeeper
                                if jersey_color is not None and self._team_colors_detected:
                                    color_lab = self._bgr_to_lab(jersey_color)
                                    home_lab = self._bgr_to_lab(self._home_color)
                                    away_lab = self._bgr_to_lab(self._away_color)
                                    home_dist = np.linalg.norm(color_lab - home_lab)
                                    away_dist = np.linalg.norm(color_lab - away_lab)
                                    if home_dist > 50 or away_dist > 50:
                                        is_gk = True

                        raw_player = DetectedPlayer(
                            track_id=-1,
                            bbox=BoundingBox(
                                x1=int(bbox[0]), y1=int(bbox[1]),
                                x2=int(bbox[2]), y2=int(bbox[3]),
                                confidence=conf
                            ),
                            pixel_position=PixelPosition(
                                x=int((bbox[0] + bbox[2]) / 2),
                                y=int((bbox[1] + bbox[3]) / 2)
                            ),
                            team=team_side,
                            jersey_color=jersey_color.tolist() if jersey_color is not None else None,
                            is_goalkeeper=is_gk,
                        )
                        raw_detected_players.append(raw_player)

                    # Log detection quality for first few frames
                    if analyzed_count < 5:
                        n_home = sum(1 for p in raw_detected_players if p.team == TeamSide.HOME)
                        n_away = sum(1 for p in raw_detected_players if p.team == TeamSide.AWAY)
                        n_unknown = sum(1 for p in raw_detected_players if p.team == TeamSide.UNKNOWN)
                        n_gk = sum(1 for p in raw_detected_players if p.is_goalkeeper)
                        print(f"[PROCESSOR] Frame {frame_count}: {len(raw_detected_players)} detections "
                              f"(home={n_home}, away={n_away}, unknown/ref={n_unknown}, gk={n_gk})")

                    # Run real tracking to get consistent IDs across frames
                    tracked_players = await self.tracking_service.update(
                        raw_detected_players, frame_count, frame=frame
                    )

                    # Process tracked players with persistent IDs
                    for tracked_player in tracked_players:
                        track_id = tracked_player.track_id
                        team_side = tracked_player.team
                        team = 'home' if team_side == TeamSide.HOME else (
                            'away' if team_side == TeamSide.AWAY else 'unknown'
                        )

                        detection = Detection(
                            bbox=[tracked_player.bbox.x1, tracked_player.bbox.y1,
                                  tracked_player.bbox.x2, tracked_player.bbox.y2],
                            confidence=tracked_player.bbox.confidence,
                            class_id=0,
                            class_name='person',
                            team=team,
                            track_id=track_id
                        )
                        frame_analysis.detections.append(detection)

                        if team == 'home':
                            frame_analysis.home_players += 1
                        elif team == 'away':
                            frame_analysis.away_players += 1

                        # Check if we already know this player's jersey number from OCR
                        known_jersey = jersey_ocr_service.get_jersey_number(track_id)

                        detected_player = DetectedPlayer(
                            track_id=track_id,
                            bbox=tracked_player.bbox,
                            pixel_position=tracked_player.pixel_position,
                            team=team_side,
                            jersey_color=tracked_player.jersey_color,
                            is_goalkeeper=tracked_player.is_goalkeeper,
                            jersey_number=known_jersey
                        )
                        detected_players_for_highlights.append(detected_player)

                    # Update player_detections for downstream ball possession code
                    player_detections = [
                        {
                            'bbox': [p.bbox.x1, p.bbox.y1, p.bbox.x2, p.bbox.y2],
                            'confidence': p.bbox.confidence,
                            'track_id': p.track_id,
                            'team': 'home' if p.team == TeamSide.HOME else (
                                'away' if p.team == TeamSide.AWAY else 'unknown'
                            )
                        }
                        for p in tracked_players
                    ]

                # Run jersey number detection
                # AI jersey detection is preferred (more accurate), falls back to OCR
                if detected_players_for_highlights:
                    if ai_jersey_enabled:
                        # AI detection handles its own frame interval internally
                        detected_players_for_highlights = await ai_jersey_detection_service.process_frame(
                            frame=frame,
                            players=detected_players_for_highlights,
                            frame_number=frame_count
                        )
                    elif ocr_enabled and analyzed_count % 30 == 0:
                        # Traditional OCR fallback (every 30 frames to save CPU)
                        detected_players_for_highlights = await jersey_ocr_service.process_players(
                            frame=frame,
                            players=detected_players_for_highlights,
                            frame_number=frame_count
                        )

                # Use enhanced ball tracker (combines YOLO, motion, and color detection)
                try:
                    ball_pos = self.ball_tracker.detect_ball(frame, yolo_ball_pos, player_boxes)
                    if ball_pos:
                        frame_analysis.ball_position = ball_pos
                except Exception as e:
                    # Ball tracker can fail on some frames - don't crash the whole pipeline
                    if analyzed_count <= 3:
                        print(f"Ball tracker error (will continue): {e}")
                    ball_pos = yolo_ball_pos  # Fall back to YOLO ball position

                # Determine ball possession and track player moments
                ball_possessed_by, possession_team = self._determine_ball_possession(ball_pos, player_detections if model else [])

                # Feed frame data to player highlights service for clip extraction
                if detected_players_for_highlights:
                    player_highlights_service.process_frame(
                        frame_number=frame_count,
                        players=detected_players_for_highlights,
                        ball_possessed_by=ball_possessed_by,
                        timestamp_ms=int(timestamp * 1000)
                    )

                # Feed data to professional analytics services
                timestamp_ms = int(timestamp * 1000)

                # Convert pixel positions to normalized pitch coordinates (0-100)
                # Assuming camera covers full pitch width
                for det_player in detected_players_for_highlights:
                    # Normalize: x across pitch length, y across pitch width
                    pitch_x = (det_player.pixel_position.x / frame_width) * 100
                    pitch_y = (det_player.pixel_position.y / frame_height) * 100

                    # Record for heatmaps and 2D radar
                    team_str = "home" if det_player.team == TeamSide.HOME else (
                        "away" if det_player.team == TeamSide.AWAY else "unknown"
                    )
                    if team_str in ["home", "away"]:
                        # Get best jersey number (confirmed, pending, or detected)
                        # This allows visualization even during early processing
                        jersey = det_player.jersey_number
                        if jersey is None:
                            # Try AI detection service - use best guess (confirmed or pending)
                            jersey = ai_jersey_detection_service.get_best_jersey_number(det_player.track_id)

                        # Only record if we have a valid jersey number (1-99)
                        # This prevents random track_ids from polluting the visualization
                        if jersey is not None and 1 <= jersey <= 99:
                            # Record player position for pitch visualization
                            pitch_visualization_service.record_player_position(
                                team=team_str,
                                jersey_number=jersey,
                                x=pitch_x,
                                y=pitch_y,
                                frame_number=frame_count,
                                timestamp_ms=timestamp_ms,
                                has_ball=(det_player.track_id == ball_possessed_by)
                            )

                        # Record possession data for match statistics
                        if det_player.track_id == ball_possessed_by:
                            match_statistics_service.record_possession(frame_count, team_str, timestamp_ms)

                # Record ball position for event detection and visualization
                if ball_pos:
                    ball_pitch_x = (ball_pos[0] / frame_width) * 100
                    ball_pitch_y = (ball_pos[1] / frame_height) * 100

                    pitch_visualization_service.record_ball_position(
                        x=ball_pitch_x,
                        y=ball_pitch_y,
                        frame_number=frame_count,
                        timestamp_ms=timestamp_ms
                    )

                    # Feed ball position to event detector for auto-detection of goals, corners, etc.
                    possessing_team = possession_team

                    detected_events = event_detector.process_frame(
                        frame_number=frame_count,
                        timestamp_ms=timestamp_ms,
                        ball_x=ball_pitch_x,
                        ball_y=ball_pitch_y,
                        possessing_team=possessing_team
                    )

                    # Wire xG model: calculate xG for any shot events
                    if detected_events:
                        for evt in detected_events:
                            if evt.event_type in (
                                DetectedEventType.SHOT,
                                DetectedEventType.SHOT_ON_TARGET,
                                DetectedEventType.SHOT_OFF_TARGET,
                                DetectedEventType.SHOT_BLOCKED,
                            ):
                                try:
                                    shot = Shot(
                                        x=evt.position_x or ball_pitch_x,
                                        y=evt.position_y or ball_pitch_y,
                                        frame_number=frame_count,
                                        timestamp_ms=timestamp_ms,
                                        team=evt.team or "unknown",
                                        player_jersey=evt.player_jersey,
                                        is_goal=(evt.event_type == DetectedEventType.GOAL),
                                        on_target=(evt.event_type == DetectedEventType.SHOT_ON_TARGET),
                                        blocked=(evt.event_type == DetectedEventType.SHOT_BLOCKED),
                                    )
                                    xg_result = xg_model.add_shot(shot)
                                except Exception as e:
                                    print(f"[xG] Error adding shot: {e}")

                # Build frame_data dict for Phase 1 analytics services
                frame_data = {
                    'frame_number': frame_count,
                    'timestamp': timestamp,
                    'ball_position': ball_pos,
                    'detections': player_detections if model else [],
                }

                # Wire pass detector (every frame — lightweight)
                try:
                    pass_detector.process_frame(frame_data)
                except Exception as e:
                    if analyzed_count <= 3:
                        print(f"[PASS] Error: {e}")

                # Wire formation detector (every 5th analyzed frame)
                if analyzed_count % 5 == 0:
                    try:
                        formation_detector.process_frame(frame_data)
                    except Exception as e:
                        if analyzed_count <= 5:
                            print(f"[FORMATION] Error: {e}")

                # Wire tactical event detector (every 2nd analyzed frame)
                if analyzed_count % 2 == 0:
                    try:
                        tactical_detector.process_frame(frame_data)
                    except Exception as e:
                        if analyzed_count <= 3:
                            print(f"[TACTICAL] Error: {e}")

                # Phase 2: Wire tactical intelligence (every 5th frame)
                if analyzed_count % 5 == 0:
                    try:
                        # Transform player_detections into tactical_intelligence format
                        # Expects separate home/away lists with {x, y, jersey_number, has_ball}
                        ti_home = []
                        ti_away = []
                        for det in (player_detections if model else []):
                            bbox = det.get('bbox', [0, 0, 0, 0])
                            cx = (bbox[0] + bbox[2]) / 2
                            cy = (bbox[1] + bbox[3]) / 2
                            # Normalize to 0-100 pitch coordinates
                            px = (cx / frame_width) * 100 if frame_width > 0 else 0
                            py = (cy / frame_height) * 100 if frame_height > 0 else 0
                            tid = det.get('track_id')
                            has_ball = (ball_possessed_by is not None and tid == ball_possessed_by)
                            jersey = ai_jersey_detection_service.get_best_jersey_number(tid) if tid else None
                            player_entry = {
                                'x': px, 'y': py,
                                'jersey_number': jersey or 0,
                                'has_ball': has_ball,
                            }
                            if det.get('team') == 'home':
                                ti_home.append(player_entry)
                            elif det.get('team') == 'away':
                                ti_away.append(player_entry)

                        bp = None
                        if ball_pos:
                            bp = ((ball_pos[0] / frame_width) * 100, (ball_pos[1] / frame_height) * 100)

                        tactical_intelligence.process_frame(
                            frame_number=frame_count,
                            timestamp_ms=int(timestamp * 1000),
                            home_players=ti_home,
                            away_players=ti_away,
                            ball_position=bp,
                            possession_team=possession_team,
                        )
                    except Exception as e:
                        if analyzed_count <= 5:
                            print(f"[TACTICAL_INTEL] Error: {e}")

                frame_analysis.player_count = len(frame_analysis.detections)
                home_player_counts.append(frame_analysis.home_players)
                away_player_counts.append(frame_analysis.away_players)

                self.current_analysis.frame_analyses.append(frame_analysis)
                analyzed_count += 1

                # Update progress
                self.progress = (frame_count / total_frames) * 100
                self.current_analysis.analyzed_frames = analyzed_count

                # Yield to event loop periodically to allow status checks
                if analyzed_count % 5 == 0:
                    await asyncio.sleep(0)

                if progress_callback and analyzed_count % 10 == 0:
                    elapsed = time.time() - start_time
                    fps_actual = analyzed_count / elapsed if elapsed > 0 else 0
                    await progress_callback({
                        'progress': self.progress,
                        'analyzed': analyzed_count,
                        'total': total_frames // frame_skip,
                        'fps': fps_actual,
                        'elapsed': elapsed
                    })

            frame_count += 1

        cap.release()

        # Calculate aggregated stats
        if home_player_counts:
            self.current_analysis.avg_home_players = np.mean(home_player_counts)
        if away_player_counts:
            self.current_analysis.avg_away_players = np.mean(away_player_counts)

        # Detection quality summary
        total_counts = [h + a for h, a in zip(home_player_counts, away_player_counts)]
        if total_counts:
            avg_total = np.mean(total_counts)
            good_coverage = sum(1 for t in total_counts if t >= self.EXPECTED_PLAYERS_MIN)
            coverage_pct = 100 * good_coverage / len(total_counts)
            print(f"\n=== Detection Summary ===")
            print(f"Average players detected: {avg_total:.1f} (home: {self.current_analysis.avg_home_players:.1f}, away: {self.current_analysis.avg_away_players:.1f})")
            print(f"Frames with good coverage (>={self.EXPECTED_PLAYERS_MIN}): {good_coverage}/{len(total_counts)} ({coverage_pct:.1f}%)")
            print(f"Min/Max detections: {min(total_counts)}/{max(total_counts)}")

        # Jersey detection summary
        if ai_jersey_enabled:
            jersey_stats = ai_jersey_detection_service.get_stats()
            print(f"\n=== AI Jersey Detection Summary ===")
            print(f"Provider: {jersey_stats['provider']}")
            print(f"API calls: {jersey_stats['api_calls']}")
            print(f"Players processed: {jersey_stats['total_players_processed']}")
            print(f"Successful detections: {jersey_stats['successful_detections']}")
            print(f"Confirmed players: {jersey_stats['confirmed_players']}")
        elif ocr_enabled:
            ocr_stats = jersey_ocr_service.get_stats()
            print(f"\n=== OCR Jersey Detection Summary ===")
            print(f"OCR attempts: {ocr_stats['total_ocr_attempts']}")
            print(f"Successful reads: {ocr_stats['successful_reads']}")
            print(f"Players identified: {ocr_stats['players_identified']}")

        # Finalize professional analytics services
        print(f"\n=== Professional Analytics Summary ===")

        # Get match statistics summary
        match_summary = match_statistics_service.get_match_summary()
        print(f"Possession: Home {match_summary.get('possession', {}).get('home', 0):.1f}% - Away {match_summary.get('possession', {}).get('away', 0):.1f}%")

        # Get detected events
        events = event_detector.get_events()
        print(f"Auto-detected events: {len(events)}")
        for event in events[:5]:  # Show first 5
            print(f"  - {event.event_type.value} at {event.timestamp_ms/1000:.1f}s")

        # Phase 1: Pass detection summary
        try:
            pass_stats = pass_detector.get_pass_stats()
            home_passes = pass_stats.get('home', {})
            away_passes = pass_stats.get('away', {})
            print(f"\n=== Pass Detection Summary ===")
            print(f"Home: {home_passes.get('total', 0)} passes ({home_passes.get('successful', 0)} successful, {home_passes.get('accuracy', 0):.0f}%)")
            print(f"Away: {away_passes.get('total', 0)} passes ({away_passes.get('successful', 0)} successful, {away_passes.get('accuracy', 0):.0f}%)")
        except Exception as e:
            pass_stats = {}
            print(f"[PASS] Stats error: {e}")

        # Phase 1: Formation detection summary
        try:
            formation_stats = formation_detector.calculate_stats()
            print(f"\n=== Formation Detection Summary ===")
            for team, stats in formation_stats.items():
                print(f"{team.capitalize()}: {stats.primary_formation} ({stats.formation_changes} changes)")
        except Exception as e:
            formation_stats = {}
            print(f"[FORMATION] Stats error: {e}")

        # Phase 1: xG summary
        try:
            xg_data = xg_model.get_shot_map_data()
            xg_shots = xg_data.get('shots', [])
            print(f"\n=== xG Summary ===")
            print(f"Total shots: {len(xg_shots)}")
            print(f"Home xG: {xg_data.get('total_xg', {}).get('home', 0):.2f}")
            print(f"Away xG: {xg_data.get('total_xg', {}).get('away', 0):.2f}")
        except Exception as e:
            xg_data = {}
            print(f"[xG] Stats error: {e}")

        # Phase 1: Tactical events summary
        try:
            tactical_summary = tactical_detector.get_events_summary()
            tactical_total = tactical_summary.get('total_events', 0)
            event_counts = tactical_summary.get('event_counts', {})
            print(f"\n=== Tactical Events Summary ===")
            print(f"Total tactical events: {tactical_total}")
            for event_type, count in sorted(event_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"  - {event_type}: {count}")
        except Exception as e:
            tactical_summary = {}
            print(f"[TACTICAL] Stats error: {e}")

        # Convert formation_stats dataclasses to dicts for downstream consumers
        formation_stats_dicts = {}
        try:
            for team, stats in formation_stats.items():
                formation_stats_dicts[team] = {
                    'primary_formation': stats.primary_formation,
                    'formation_counts': stats.formation_counts,
                    'avg_defensive_line': stats.avg_defensive_line_height,
                    'avg_compactness': stats.avg_compactness,
                    'formation_changes': stats.formation_changes,
                }
        except Exception:
            formation_stats_dicts = {}

        # Initialize coach assist with collected data + Phase 1 analytics
        coach_assist_service.load_match_data(
            match_stats=match_statistics_service,
            pitch_viz=pitch_visualization_service,
            event_detector=event_detector,
            player_highlights=player_highlights_service,
            pass_stats=pass_stats,
            formation_stats=formation_stats_dicts,
            xg_data=xg_data,
            tactical_summary=tactical_summary,
        )
        print(f"\nCoach Assist AI initialized with match data + Phase 1 analytics")

        # Phase 2: Run AI Coach analysis
        try:
            self._coaching_analysis = ai_coach.analyze_match(
                pass_stats=pass_stats,
                formation_stats=formation_stats_dicts,
                tactical_events=tactical_summary,
                frame_analyses=None,
            )
            summary = ai_coach.match_summary
            if summary:
                print(f"\n=== AI Coach Analysis ===")
                print(f"Overall rating: {summary.overall_rating}/10")
                print(f"Strengths: {', '.join(summary.key_strengths[:3])}")
                print(f"Insights generated: {len(ai_coach.insights)}")
        except Exception as e:
            self._coaching_analysis = {}
            print(f"[AI_COACH] Analysis error: {e}")

        # Phase 2: Collect tactical intelligence summary
        try:
            self._tactical_alerts_summary = tactical_intelligence.get_full_analysis()
            alert_count = len(self._tactical_alerts_summary.get('all_alerts', []))
            print(f"\n=== Tactical Intelligence ===")
            print(f"Total tactical alerts: {alert_count}")
        except Exception as e:
            self._tactical_alerts_summary = {}
            print(f"[TACTICAL_INTEL] Summary error: {e}")

        # Phase 2: Generate training focus — THE DIFFERENTIATOR
        try:
            self._training_focus = coach_assist_service.generate_training_focus()
            priorities = self._training_focus.get('priority_areas', [])
            print(f"\n=== Training Focus ===")
            for p in priorities[:3]:
                print(f"  [{p.get('severity', '').upper()}] {p.get('area')} ({p.get('team')}) — {p.get('drill')}")
        except Exception as e:
            self._training_focus = {}
            print(f"[TRAINING] Focus error: {e}")

        self.current_analysis.end_time = datetime.now().isoformat()
        self.status = "complete"
        self.is_processing = False
        self.progress = 100

        # Save to file if output path provided
        if output_path:
            self._save_analysis(output_path)

        return self.current_analysis

    def _save_analysis(self, output_path: str):
        """Save analysis to JSON file with full analytics data."""
        if self.current_analysis:
            data = {
                'video_path': self.current_analysis.video_path,
                'duration_seconds': self.current_analysis.duration_seconds,
                'total_frames': self.current_analysis.total_frames,
                'analyzed_frames': self.current_analysis.analyzed_frames,
                'fps_analyzed': self.current_analysis.fps_analyzed,
                'start_time': self.current_analysis.start_time,
                'end_time': self.current_analysis.end_time,
                'avg_home_players': self.current_analysis.avg_home_players,
                'avg_away_players': self.current_analysis.avg_away_players,
                'frame_count': len(self.current_analysis.frame_analyses),
                'frames': [
                    {
                        'frame_number': f.frame_number,
                        'timestamp': f.timestamp,
                        'player_count': f.player_count,
                        'home_players': f.home_players,
                        'away_players': f.away_players,
                        'ball_position': f.ball_position,
                        'detections': [asdict(d) for d in f.detections]
                    }
                    for f in self.current_analysis.frame_analyses
                ]
            }

            # Enrich with Phase 1 analytics
            try:
                data['pass_stats'] = pass_detector.get_pass_stats()
            except Exception:
                data['pass_stats'] = {}

            try:
                formation_stats = formation_detector.calculate_stats()
                data['formations'] = {
                    team: {
                        'primary_formation': s.primary_formation,
                        'formation_counts': s.formation_counts,
                        'avg_defensive_line_height': s.avg_defensive_line_height,
                        'avg_compactness': s.avg_compactness,
                        'formation_changes': s.formation_changes,
                    }
                    for team, s in formation_stats.items()
                }
            except Exception:
                data['formations'] = {}

            try:
                data['xg'] = xg_model.get_shot_map_data()
            except Exception:
                data['xg'] = {}

            try:
                data['tactical_events'] = tactical_detector.get_events_summary()
            except Exception:
                data['tactical_events'] = {}

            # Enrich with Phase 2 coaching intelligence
            data['coaching_analysis'] = self._coaching_analysis or {}
            data['tactical_alerts'] = self._tactical_alerts_summary or {}
            data['training_focus'] = self._training_focus or {}

            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)

            print(f"Analysis saved to: {output_path}")

    def get_status(self) -> Dict:
        """Get current processing status."""
        return {
            'is_processing': self.is_processing,
            'video_path': self.current_video,
            'progress': self.progress,
            'status': self.status,
            'analyzed_frames': self.current_analysis.analyzed_frames if self.current_analysis else 0
        }

    def get_coaching_analysis(self) -> Dict:
        """Get cached AI Coach analysis from pipeline run."""
        return self._coaching_analysis or {}

    def get_tactical_alerts(self) -> Dict:
        """Get cached tactical intelligence alerts from pipeline run."""
        return self._tactical_alerts_summary or {}

    def get_training_focus(self) -> Dict:
        """Get cached training focus / session plan from pipeline run."""
        return self._training_focus or {}


# Global instance
local_processor = LocalVideoProcessor()
