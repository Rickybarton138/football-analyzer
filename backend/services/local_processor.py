"""
Local CPU Video Processing Service

Processes video locally using CPU for player detection and tracking.
Slower than GPU but works without cloud setup.
"""
import cv2
import numpy as np
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from pathlib import Path
from collections import deque
import json
import time
from datetime import datetime

# Import for player highlights tracking and jersey detection
from models.schemas import DetectedPlayer, BoundingBox, PixelPosition, TeamSide
from services.player_highlights import player_highlights_service
from services.jersey_ocr import jersey_ocr_service
from services.ai_jersey_detection import ai_jersey_detection_service
from config import settings

# Import professional analytics services (VEO-style)
from services.match_statistics import match_statistics_service, EventType as StatsEventType
from services.pitch_visualization import pitch_visualization_service
from services.event_detector import event_detector
from services.coach_assist import coach_assist_service


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

    # Detection thresholds - optimized for football pitch coverage
    CONF_THRESHOLD_HIGH = 0.4      # Near camera players (large bboxes)
    CONF_THRESHOLD_MEDIUM = 0.25   # Mid-distance players
    CONF_THRESHOLD_LOW = 0.15      # Distant players (small bboxes)

    # Expected player counts for validation
    EXPECTED_PLAYERS_MIN = 18      # Minimum reasonable (some occlusion)
    EXPECTED_PLAYERS_MAX = 26      # Maximum (22 + refs + subs visible)

    # Multi-scale detection settings
    DETECTION_SCALES = [1.0, 1.5, 2.0]  # Upscale for distant player detection
    NMS_IOU_THRESHOLD = 0.5        # For merging multi-scale detections

    def __init__(self):
        self.is_processing = False
        self.current_video = None
        self.progress = 0
        self.status = "idle"
        self.current_analysis: Optional[MatchAnalysis] = None
        self.model = None
        self.ball_tracker = BallTracker()
        self.use_multi_scale = False  # Disable for faster processing (set True for better detection)
        self.track_counter = 0  # For assigning track IDs

    def _determine_ball_possession(
        self,
        ball_pos: Optional[List[float]],
        player_detections: List[Dict]
    ) -> Optional[int]:
        """
        Determine which player has possession of the ball.

        Returns the track_id of the player closest to the ball, if within
        possession distance threshold.
        """
        if ball_pos is None or not player_detections:
            return None

        POSSESSION_DISTANCE = 60  # pixels - player must be within this distance

        closest_player = None
        min_distance = float('inf')

        for det in player_detections:
            bbox = det['bbox']
            # Calculate player foot position (bottom center of bbox)
            player_x = (bbox[0] + bbox[2]) / 2
            player_y = bbox[3]  # Bottom of bbox (feet)

            # Distance to ball
            dist = np.sqrt((player_x - ball_pos[0])**2 + (player_y - ball_pos[1])**2)

            if dist < min_distance and dist < POSSESSION_DISTANCE:
                min_distance = dist
                closest_player = det.get('track_id')

        return closest_player

    def _load_model(self):
        """Load YOLOv8 model for CPU inference."""
        if self.model is None:
            try:
                from ultralytics import YOLO
                # Use nano model for faster CPU inference
                self.model = YOLO('yolov8n.pt')
                self.model.to('cpu')
                print("YOLOv8 nano model loaded for CPU inference")
            except ImportError:
                print("ultralytics not installed, using mock detection")
                self.model = None
        return self.model

    def _detect_team_color(self, frame: np.ndarray, bbox: List[float]) -> str:
        """Detect team based on jersey color. Home team = RED jerseys."""
        x1, y1, x2, y2 = map(int, bbox)

        # Clamp coordinates to frame bounds
        h_frame, w_frame = frame.shape[:2]
        x1, x2 = max(0, x1), min(w_frame, x2)
        y1, y2 = max(0, y1), min(h_frame, y2)

        # Get upper body region (jersey area)
        h = y2 - y1
        jersey_y1 = y1 + int(h * 0.15)
        jersey_y2 = y1 + int(h * 0.45)

        if jersey_y2 > jersey_y1 and x2 > x1:
            jersey = frame[jersey_y1:jersey_y2, x1:x2]
            if jersey.size > 0:
                # Convert to HSV for better color detection
                hsv = cv2.cvtColor(jersey, cv2.COLOR_BGR2HSV)

                # Count red pixels (red wraps around in HSV: 0-15 and 165-180)
                # Widened range and lowered saturation/value thresholds for varying lighting
                lower_red1 = np.array([0, 50, 40])
                upper_red1 = np.array([15, 255, 255])
                lower_red2 = np.array([165, 50, 40])
                upper_red2 = np.array([180, 255, 255])

                mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
                mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
                red_mask = mask_red1 | mask_red2

                red_ratio = np.sum(red_mask > 0) / red_mask.size

                # Check for black/dark (referee)
                avg_val = np.mean(hsv[:, :, 2])
                avg_sat = np.mean(hsv[:, :, 1])

                # Referee detection: dark jerseys (black)
                if avg_val < 50 and avg_sat < 60:
                    return 'referee'
                # Home team: RED jerseys (at least 8% red pixels - lowered threshold)
                elif red_ratio > 0.08:
                    return 'home'
                else:
                    return 'away'

        return 'unknown'

    def _get_adaptive_confidence(self, bbox_height: float, frame_height: float) -> float:
        """
        Get adaptive confidence threshold based on bounding box size.

        Smaller bboxes (distant players) get lower thresholds since they
        naturally have lower detection confidence.
        """
        # Calculate relative size (0-1 where 1 = full frame height)
        relative_size = bbox_height / frame_height

        if relative_size > 0.15:  # Large player (close to camera)
            return self.CONF_THRESHOLD_HIGH
        elif relative_size > 0.08:  # Medium distance
            return self.CONF_THRESHOLD_MEDIUM
        else:  # Distant player
            return self.CONF_THRESHOLD_LOW

    def _detect_players_multiscale(
        self,
        frame: np.ndarray,
        model
    ) -> Tuple[List[Dict], Optional[List[float]]]:
        """
        Run multi-scale detection to catch players at all distances.

        Returns:
            Tuple of (player_detections, ball_position)
        """
        height, width = frame.shape[:2]
        all_detections = []
        ball_position = None

        scales_to_use = self.DETECTION_SCALES if self.use_multi_scale else [1.0]

        for scale in scales_to_use:
            if scale == 1.0:
                scaled_frame = frame
                # Use lower conf for base scale to catch more
                base_conf = self.CONF_THRESHOLD_LOW
            else:
                # Upscale frame to make distant players larger
                new_w = int(width * scale)
                new_h = int(height * scale)
                scaled_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                # Even lower confidence for upscaled (these are distant players)
                base_conf = 0.12

            # Run YOLO
            results = model(scaled_frame, conf=base_conf, verbose=False)

            for r in results:
                boxes = r.boxes
                if boxes is None:
                    continue

                for box in boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])

                    # Scale coordinates back to original frame
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    if scale != 1.0:
                        x1 = x1 / scale
                        y1 = y1 / scale
                        x2 = x2 / scale
                        y2 = y2 / scale

                    # Clamp to frame bounds
                    x1 = max(0, min(x1, width - 1))
                    y1 = max(0, min(y1, height - 1))
                    x2 = max(0, min(x2, width))
                    y2 = max(0, min(y2, height))

                    bbox_height = y2 - y1

                    # Class 0 = person
                    if cls == 0:
                        # Apply adaptive confidence based on player size
                        required_conf = self._get_adaptive_confidence(bbox_height, height)
                        if conf >= required_conf:
                            all_detections.append({
                                'bbox': [x1, y1, x2, y2],
                                'confidence': conf,
                                'scale': scale
                            })

                    # Class 32 = sports ball
                    elif cls == 32 and conf > 0.12 and ball_position is None:
                        center_x = (x1 + x2) / 2
                        center_y = (y1 + y2) / 2
                        ball_position = [center_x, center_y]

        # Apply NMS to merge overlapping detections from different scales
        merged_detections = self._apply_nms(all_detections)

        return merged_detections, ball_position

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

        # Initialize professional analytics services (VEO-style)
        match_statistics_service.reset()
        match_statistics_service.set_video_info(fps=video_fps, total_frames=total_frames)

        pitch_visualization_service.reset()
        pitch_visualization_service.set_video_info(fps=video_fps, total_frames=total_frames)

        event_detector.reset()
        event_detector.set_video_info(fps=video_fps, total_frames=total_frames, frame_width=frame_width, frame_height=frame_height)

        coach_assist_service.reset()

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
                    # Use enhanced multi-scale detection
                    player_detections, yolo_ball_pos = self._detect_players_multiscale(frame, model)

                    # Process each detected player
                    for det in player_detections:
                        bbox = det['bbox']
                        conf = det['confidence']
                        player_boxes.append(bbox)

                        # Assign track ID (simple incrementing for now)
                        track_id = self.track_counter
                        self.track_counter += 1
                        det['track_id'] = track_id

                        # Classify team by jersey color
                        team = self._detect_team_color(frame, bbox)
                        team_side = TeamSide.HOME if team == 'home' else (
                            TeamSide.AWAY if team == 'away' else TeamSide.UNKNOWN
                        )

                        detection = Detection(
                            bbox=bbox,
                            confidence=conf,
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

                        # Create DetectedPlayer for highlights tracking
                        # Check if we already know this player's jersey number from OCR
                        known_jersey = jersey_ocr_service.get_jersey_number(track_id)

                        detected_player = DetectedPlayer(
                            track_id=track_id,
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
                            jersey_number=known_jersey  # From OCR if identified
                        )
                        detected_players_for_highlights.append(detected_player)

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
                ball_possessed_by = self._determine_ball_possession(ball_pos, player_detections if model else [])

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
                    possessing_team = None
                    if ball_possessed_by is not None:
                        for det in player_detections if model else []:
                            if det.get('track_id') == ball_possessed_by:
                                possessing_team = det.get('team', None)
                                if possessing_team:
                                    possessing_team = "home" if possessing_team == "home" else "away"
                                break

                    event_detector.process_frame(
                        frame_number=frame_count,
                        timestamp_ms=timestamp_ms,
                        ball_x=ball_pitch_x,
                        ball_y=ball_pitch_y,
                        possessing_team=possessing_team
                    )

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

        # Initialize coach assist with collected data
        coach_assist_service.load_match_data(
            match_stats=match_statistics_service,
            pitch_viz=pitch_visualization_service,
            event_detector=event_detector,
            player_highlights=player_highlights_service
        )
        print(f"Coach Assist AI initialized with match data")

        self.current_analysis.end_time = datetime.now().isoformat()
        self.status = "complete"
        self.is_processing = False
        self.progress = 100

        # Save to file if output path provided
        if output_path:
            self._save_analysis(output_path)

        return self.current_analysis

    def _save_analysis(self, output_path: str):
        """Save analysis to JSON file."""
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


# Global instance
local_processor = LocalVideoProcessor()
