"""
Ball Detection Service

Specialized detection for the football using SAHI sliced inference
as primary method, with fallback to standard YOLO, motion-based,
and color-based detection.

Includes parabolic interpolation for aerial balls and adaptive
Kalman filtering.
"""
import numpy as np
import cv2
from typing import Optional, List, Tuple
from collections import deque

from config import settings
from models.schemas import DetectedBall, BoundingBox, PixelPosition, Position, Velocity


class BallDetectionService:
    """
    Service for detecting and tracking the football.

    Detection pipeline (in priority order):
    1. SAHI sliced inference (primary - catches small/distant balls)
    2. Standard YOLO detection (fast fallback)
    3. Motion-based detection
    4. Color-based detection
    5. Kalman prediction + parabolic interpolation
    """

    def __init__(self):
        self.model = None
        self.prev_frame: Optional[np.ndarray] = None
        self.prev_positions: deque = deque(maxlen=30)  # Increased for interpolation
        self.kalman = self._init_kalman_filter()
        self._sahi_available = False
        self._frames_since_detection = 0
        self._is_aerial = False  # Track if ball is in aerial trajectory

        # Ball appearance parameters
        self.ball_min_radius = 5
        self.ball_max_radius = 30
        self.ball_color_lower = np.array([0, 0, 200])
        self.ball_color_upper = np.array([180, 30, 255])

    def _init_kalman_filter(self):
        """Initialize adaptive Kalman filter for ball tracking."""
        kalman = cv2.KalmanFilter(4, 2)

        kalman.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], np.float32)

        kalman.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], np.float32)

        # Default noise (adjusted adaptively for aerial vs ground)
        kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5

        return kalman

    def _adapt_kalman_noise(self):
        """Adjust Kalman noise based on whether ball is aerial or on ground."""
        if self._is_aerial:
            # Aerial: higher process noise (gravity affects trajectory)
            self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.1
            self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1.0
        else:
            # Ground: lower process noise (more predictable)
            self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
            self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5

    async def initialize(self):
        """Initialize ball detection model."""
        try:
            from ultralytics import YOLO

            model_name = "yolov8s.pt" if settings.USE_GPU else "yolov8n.pt"
            self.model = YOLO(model_name)

            if not settings.USE_GPU:
                self.model.to('cpu')

            print("[BALL] YOLO ball detection initialized")
        except ImportError:
            print("Warning: ultralytics not installed. Using fallback ball detection.")
            self.model = None

        # Check SAHI availability
        try:
            from sahi import AutoDetectionModel
            self._sahi_available = True
            print("[BALL] SAHI sliced inference available")
        except ImportError:
            self._sahi_available = False
            print("[BALL] SAHI not available, using standard detection")

    async def detect(
        self,
        frame: np.ndarray,
        player_boxes: Optional[List[BoundingBox]] = None
    ) -> Optional[DetectedBall]:
        """
        Detect the ball in a frame.

        Args:
            frame: BGR image as numpy array
            player_boxes: Optional player bounding boxes to exclude

        Returns:
            DetectedBall or None if not found
        """
        candidates = []

        # Method 1: SAHI sliced inference (primary for small balls)
        if self._sahi_available and self.model is not None:
            sahi_ball = await self._detect_sahi(frame)
            if sahi_ball:
                candidates.append(("sahi", sahi_ball, 0.95))

        # Method 2: Standard YOLO detection
        yolo_ball = await self._detect_yolo(frame)
        if yolo_ball:
            candidates.append(("yolo", yolo_ball, 0.9))

        # Method 3: Motion-based detection
        motion_ball = await self._detect_motion(frame)
        if motion_ball:
            candidates.append(("motion", motion_ball, 0.7))

        # Method 4: Color-based detection
        color_ball = await self._detect_color(frame, player_boxes)
        if color_ball:
            candidates.append(("color", color_ball, 0.5))

        # Method 5: Parabolic interpolation for aerial balls
        if not candidates and self._frames_since_detection <= settings.BALL_MAX_INTERPOLATION_GAP:
            interp_ball = self._parabolic_interpolation()
            if interp_ball:
                candidates.append(("interpolated", interp_ball, 0.2))

        # Method 6: Kalman prediction
        if not candidates:
            predicted = self._predict_position()
            if predicted:
                candidates.append(("predicted", predicted, 0.1))

        if not candidates:
            self._frames_since_detection += 1
            self.prev_frame = frame.copy()
            return None

        # Sort by confidence and select best
        candidates.sort(key=lambda x: x[2], reverse=True)
        best_method, best_ball, _ = candidates[0]

        # Update state
        self._update_kalman(best_ball.pixel_position)
        self._detect_aerial_state(best_ball.pixel_position)

        velocity = self._calculate_velocity(best_ball.pixel_position)
        best_ball.velocity = velocity

        self.prev_frame = frame.copy()
        self.prev_positions.append((
            best_ball.pixel_position.x,
            best_ball.pixel_position.y,
            self._frames_since_detection == 0  # True if real detection
        ))
        self._frames_since_detection = 0 if best_method != "interpolated" else self._frames_since_detection + 1

        return best_ball

    async def _detect_sahi(self, frame: np.ndarray) -> Optional[DetectedBall]:
        """Detect ball using SAHI sliced inference for small object detection."""
        try:
            from sahi import AutoDetectionModel
            from sahi.predict import get_sliced_prediction

            # Create SAHI detection model wrapper
            detection_model = AutoDetectionModel.from_pretrained(
                model_type="ultralytics",
                model=self.model,
                confidence_threshold=0.2,
                device="cuda:0" if settings.USE_GPU else "cpu",
            )

            # Run sliced prediction
            result = get_sliced_prediction(
                image=frame,
                detection_model=detection_model,
                slice_height=settings.SAHI_SLICE_SIZE,
                slice_width=settings.SAHI_SLICE_SIZE,
                overlap_height_ratio=settings.SAHI_OVERLAP_RATIO,
                overlap_width_ratio=settings.SAHI_OVERLAP_RATIO,
                verbose=0,
            )

            # Find ball detections (sports ball class = 32 in COCO)
            best_ball = None
            best_conf = 0

            for pred in result.object_prediction_list:
                if pred.category.id == settings.SPORTS_BALL_CLASS_ID:
                    conf = pred.score.value
                    if conf > best_conf:
                        best_conf = conf
                        bbox = pred.bbox
                        x1, y1 = int(bbox.minx), int(bbox.miny)
                        x2, y2 = int(bbox.maxx), int(bbox.maxy)
                        cx = (x1 + x2) // 2
                        cy = (y1 + y2) // 2

                        best_ball = DetectedBall(
                            bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, confidence=conf),
                            pixel_position=PixelPosition(x=cx, y=cy)
                        )

            return best_ball

        except Exception as e:
            # SAHI failed, fall through to standard YOLO
            return None

    async def _detect_yolo(self, frame: np.ndarray) -> Optional[DetectedBall]:
        """Detect ball using standard YOLO."""
        if self.model is None:
            return None

        results = self.model(
            frame,
            conf=0.3,
            classes=[settings.SPORTS_BALL_CLASS_ID],
            verbose=False
        )

        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue

            best_idx = boxes.conf.argmax()
            box = boxes[best_idx]

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            confidence = float(box.conf[0])

            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            return DetectedBall(
                bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, confidence=confidence),
                pixel_position=PixelPosition(x=center_x, y=center_y)
            )

        return None

    async def _detect_motion(self, frame: np.ndarray) -> Optional[DetectedBall]:
        """Detect ball using frame differencing and motion analysis."""
        if self.prev_frame is None:
            return None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        prev_gray = cv2.cvtColor(self.prev_frame, cv2.COLOR_BGR2GRAY)

        if gray.shape != prev_gray.shape:
            prev_gray = cv2.resize(prev_gray, (gray.shape[1], gray.shape[0]))

        diff = cv2.absdiff(gray, prev_gray)
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        ball_candidates = []

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 50 or area > 2000:
                continue

            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter ** 2)

            if circularity < 0.5:
                continue

            (x, y), radius = cv2.minEnclosingCircle(contour)

            if radius < self.ball_min_radius or radius > self.ball_max_radius:
                continue

            ball_candidates.append({
                "center": (int(x), int(y)),
                "radius": int(radius),
                "circularity": circularity,
                "area": area
            })

        if not ball_candidates:
            return None

        ball_candidates.sort(key=lambda b: b["circularity"], reverse=True)
        best = ball_candidates[0]

        x, y = best["center"]
        r = best["radius"]

        return DetectedBall(
            bbox=BoundingBox(
                x1=x-r, y1=y-r, x2=x+r, y2=y+r,
                confidence=best["circularity"]
            ),
            pixel_position=PixelPosition(x=x, y=y)
        )

    async def _detect_color(
        self,
        frame: np.ndarray,
        player_boxes: Optional[List[BoundingBox]] = None
    ) -> Optional[DetectedBall]:
        """Detect ball using color filtering."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 30, 255])
        mask = cv2.inRange(hsv, lower_white, upper_white)

        if player_boxes:
            for box in player_boxes:
                mask[box.y1:box.y2, box.x1:box.x2] = 0

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 30 or area > 1500:
                continue

            (x, y), radius = cv2.minEnclosingCircle(contour)

            if radius < self.ball_min_radius or radius > self.ball_max_radius:
                continue

            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter ** 2)

            if circularity > 0.6:
                return DetectedBall(
                    bbox=BoundingBox(
                        x1=int(x-radius), y1=int(y-radius),
                        x2=int(x+radius), y2=int(y+radius),
                        confidence=circularity * 0.5
                    ),
                    pixel_position=PixelPosition(x=int(x), y=int(y))
                )

        return None

    def _detect_aerial_state(self, position: PixelPosition):
        """Detect if ball is in an aerial trajectory based on vertical movement."""
        if len(self.prev_positions) < 3:
            self._is_aerial = False
            return

        # Check vertical acceleration (gravity indicator)
        positions = list(self.prev_positions)
        recent = positions[-3:]

        # Calculate vertical differences
        dy1 = recent[1][1] - recent[0][1]
        dy2 = recent[2][1] - recent[1][1]

        # If vertical acceleration is consistent (ball going up then down), it's aerial
        # Gravity causes increasing downward velocity
        vertical_accel = dy2 - dy1
        self._is_aerial = abs(vertical_accel) > 2  # Pixel threshold for acceleration

        self._adapt_kalman_noise()

    def _parabolic_interpolation(self) -> Optional[DetectedBall]:
        """
        Interpolate ball position using parabolic (gravity-aware) trajectory.

        Fits a parabola to the last known positions to predict where the
        ball should be during aerial phases.
        """
        # Need at least 3 real detections for parabolic fit
        real_positions = [(x, y) for x, y, is_real in self.prev_positions if is_real]

        if len(real_positions) < 3:
            return self._predict_position()

        # Use last 5-10 real positions
        recent = real_positions[-10:]
        n = len(recent)

        # Fit parabola: y = at² + bt + c (t = frame index)
        t = np.arange(n, dtype=np.float64)
        xs = np.array([p[0] for p in recent], dtype=np.float64)
        ys = np.array([p[1] for p in recent], dtype=np.float64)

        # Linear fit for x (horizontal motion is approximately linear)
        if n >= 2:
            x_coeffs = np.polyfit(t, xs, 1)
        else:
            return None

        # Quadratic fit for y (vertical motion follows parabola under gravity)
        if n >= 3:
            y_coeffs = np.polyfit(t, ys, 2)
        else:
            y_coeffs = np.polyfit(t, ys, 1)

        # Predict next position
        t_pred = float(n + self._frames_since_detection)
        pred_x = int(np.polyval(x_coeffs, t_pred))
        pred_y = int(np.polyval(y_coeffs, t_pred))

        # Validate prediction is reasonable
        if pred_x < 0 or pred_x > 1920 or pred_y < 0 or pred_y > 1080:
            return None

        # Confidence decreases with interpolation gap
        conf = max(0.1, 0.5 - self._frames_since_detection * 0.02)

        return DetectedBall(
            bbox=BoundingBox(
                x1=pred_x - 10, y1=pred_y - 10,
                x2=pred_x + 10, y2=pred_y + 10,
                confidence=conf
            ),
            pixel_position=PixelPosition(x=pred_x, y=pred_y)
        )

    def _predict_position(self) -> Optional[DetectedBall]:
        """Predict ball position using Kalman filter."""
        if len(self.prev_positions) < 3:
            return None

        prediction = self.kalman.predict()
        pred_flat = prediction.flatten()
        x, y = int(pred_flat[0]), int(pred_flat[1])

        if x < 0 or x > 1920 or y < 0 or y > 1080:
            return None

        return DetectedBall(
            bbox=BoundingBox(
                x1=x-10, y1=y-10, x2=x+10, y2=y+10,
                confidence=0.3
            ),
            pixel_position=PixelPosition(x=x, y=y)
        )

    def _update_kalman(self, position: PixelPosition):
        """Update Kalman filter with measured position."""
        measurement = np.array([[position.x], [position.y]], dtype=np.float32)
        self.kalman.correct(measurement)

    def _calculate_velocity(self, current_pos: PixelPosition) -> Optional[Velocity]:
        """Calculate ball velocity from recent positions."""
        if len(self.prev_positions) < 2:
            return None

        prev_x, prev_y = self.prev_positions[-1][0], self.prev_positions[-1][1]
        dx = current_pos.x - prev_x
        dy = current_pos.y - prev_y

        fps = settings.LIVE_FPS
        vx = dx * fps
        vy = dy * fps

        pixel_to_meter = 0.05
        vx_ms = vx * pixel_to_meter
        vy_ms = vy * pixel_to_meter

        speed_ms = float(np.sqrt(vx_ms**2 + vy_ms**2))
        speed_kmh = speed_ms * 3.6

        return Velocity(vx=float(vx_ms), vy=float(vy_ms), speed_kmh=float(speed_kmh))

    def get_ball_trajectory(self, num_points: int = 5) -> List[Tuple[int, int]]:
        """Get recent ball trajectory points."""
        return [(x, y) for x, y, _ in list(self.prev_positions)[-num_points:]]

    def is_ball_in_motion(self, threshold_kmh: float = 5.0) -> bool:
        """Check if ball is moving above threshold speed."""
        if len(self.prev_positions) < 2:
            return False

        p1 = self.prev_positions[-2]
        p2 = self.prev_positions[-1]

        dist_pixels = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
        dist_meters = dist_pixels * 0.05

        speed_ms = dist_meters * settings.LIVE_FPS
        speed_kmh = speed_ms * 3.6

        return speed_kmh > threshold_kmh

    def draw_ball(
        self,
        frame: np.ndarray,
        ball: DetectedBall,
        draw_trajectory: bool = True
    ) -> np.ndarray:
        """Draw ball detection and trajectory on frame."""
        annotated = frame.copy()

        cv2.circle(
            annotated,
            (ball.pixel_position.x, ball.pixel_position.y),
            max(10, (ball.bbox.x2 - ball.bbox.x1) // 2),
            (0, 255, 255), 2
        )

        if draw_trajectory and len(self.prev_positions) > 1:
            points = [(x, y) for x, y, _ in self.prev_positions]
            for i in range(1, len(points)):
                alpha = i / len(points)
                color = (0, int(255 * alpha), int(255 * alpha))
                cv2.line(annotated, points[i-1], points[i], color, 2)

        if ball.velocity and ball.velocity.speed_kmh > 5:
            scale = 2
            end_x = int(ball.pixel_position.x + ball.velocity.vx * scale)
            end_y = int(ball.pixel_position.y + ball.velocity.vy * scale)
            cv2.arrowedLine(
                annotated,
                (ball.pixel_position.x, ball.pixel_position.y),
                (end_x, end_y),
                (255, 0, 255), 2
            )

        return annotated

    def reset(self):
        """Reset ball tracking state."""
        self.prev_frame = None
        self.prev_positions.clear()
        self.kalman = self._init_kalman_filter()
        self._frames_since_detection = 0
        self._is_aerial = False
