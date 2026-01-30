"""
Ball Detection Service

Specialized detection for the football using motion analysis
and deep learning inference.
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

    Combines multiple approaches:
    1. YOLO detection for sports ball class
    2. Motion-based detection for fast-moving ball
    3. Color-based detection as fallback
    4. Kalman filtering for trajectory prediction
    """

    def __init__(self):
        self.model = None
        self.prev_frame: Optional[np.ndarray] = None
        self.prev_positions: deque = deque(maxlen=10)  # Last 10 ball positions
        self.kalman = self._init_kalman_filter()

        # Ball appearance parameters
        self.ball_min_radius = 5  # pixels
        self.ball_max_radius = 30  # pixels
        self.ball_color_lower = np.array([0, 0, 200])  # White ball lower bound (BGR)
        self.ball_color_upper = np.array([180, 30, 255])  # White ball upper bound

    def _init_kalman_filter(self):
        """Initialize Kalman filter for ball tracking."""
        kalman = cv2.KalmanFilter(4, 2)  # 4 state vars (x, y, vx, vy), 2 measurements (x, y)

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

        kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5

        return kalman

    async def initialize(self):
        """Initialize ball detection model."""
        try:
            from ultralytics import YOLO

            # Use same model as player detection
            model_name = "yolov8s.pt" if settings.USE_GPU else "yolov8n.pt"
            self.model = YOLO(model_name)

            if not settings.USE_GPU:
                self.model.to('cpu')

        except ImportError:
            print("Warning: ultralytics not installed. Using fallback ball detection.")
            self.model = None

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

        # Method 1: YOLO detection
        yolo_ball = await self._detect_yolo(frame)
        if yolo_ball:
            candidates.append(("yolo", yolo_ball, 0.9))

        # Method 2: Motion-based detection
        motion_ball = await self._detect_motion(frame)
        if motion_ball:
            candidates.append(("motion", motion_ball, 0.7))

        # Method 3: Color-based detection
        color_ball = await self._detect_color(frame, player_boxes)
        if color_ball:
            candidates.append(("color", color_ball, 0.5))

        # Method 4: Kalman prediction (if we have history)
        predicted = self._predict_position()
        if predicted:
            candidates.append(("predicted", predicted, 0.3))

        # Select best candidate
        if not candidates:
            self.prev_frame = frame.copy()
            return None

        # Sort by confidence and select best
        candidates.sort(key=lambda x: x[2], reverse=True)
        best_method, best_ball, _ = candidates[0]

        # Update Kalman filter
        self._update_kalman(best_ball.pixel_position)

        # Calculate velocity
        velocity = self._calculate_velocity(best_ball.pixel_position)
        best_ball.velocity = velocity

        # Store for next frame
        self.prev_frame = frame.copy()
        self.prev_positions.append((best_ball.pixel_position.x, best_ball.pixel_position.y))

        return best_ball

    async def _detect_yolo(self, frame: np.ndarray) -> Optional[DetectedBall]:
        """Detect ball using YOLO."""
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

            # Get highest confidence ball detection
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

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        prev_gray = cv2.cvtColor(self.prev_frame, cv2.COLOR_BGR2GRAY)

        # Ensure frames have same size (handle resolution changes)
        if gray.shape != prev_gray.shape:
            prev_gray = cv2.resize(prev_gray, (gray.shape[1], gray.shape[0]))

        # Frame difference
        diff = cv2.absdiff(gray, prev_gray)
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        ball_candidates = []

        for contour in contours:
            area = cv2.contourArea(contour)

            # Filter by size (ball should be small)
            if area < 50 or area > 2000:
                continue

            # Check circularity
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter ** 2)

            if circularity < 0.5:  # Not circular enough
                continue

            # Get bounding circle
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

        # Select best candidate (most circular, appropriate size)
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
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Create mask for white/bright objects
        # Standard football is white
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 30, 255])
        mask = cv2.inRange(hsv, lower_white, upper_white)

        # Mask out player regions
        if player_boxes:
            for box in player_boxes:
                mask[box.y1:box.y2, box.x1:box.x2] = 0

        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 30 or area > 1500:
                continue

            (x, y), radius = cv2.minEnclosingCircle(contour)

            if radius < self.ball_min_radius or radius > self.ball_max_radius:
                continue

            # Check circularity
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

    def _predict_position(self) -> Optional[DetectedBall]:
        """Predict ball position using Kalman filter."""
        if len(self.prev_positions) < 3:
            return None

        prediction = self.kalman.predict()
        # Kalman predict returns 2D array (4x1), flatten and get x,y
        pred_flat = prediction.flatten()
        x, y = int(pred_flat[0]), int(pred_flat[1])

        # Validate prediction is within frame bounds (assume 1920x1080)
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

        prev_x, prev_y = self.prev_positions[-1]
        dx = current_pos.x - prev_x
        dy = current_pos.y - prev_y

        # Assume ~10 FPS for velocity calculation
        fps = settings.LIVE_FPS
        vx = dx * fps  # pixels per second
        vy = dy * fps

        # Convert to approximate m/s (rough estimate: 1 pixel ≈ 0.05m at typical camera view)
        pixel_to_meter = 0.05
        vx_ms = vx * pixel_to_meter
        vy_ms = vy * pixel_to_meter

        speed_ms = float(np.sqrt(vx_ms**2 + vy_ms**2))
        speed_kmh = speed_ms * 3.6

        return Velocity(vx=float(vx_ms), vy=float(vy_ms), speed_kmh=float(speed_kmh))

    def get_ball_trajectory(self, num_points: int = 5) -> List[Tuple[int, int]]:
        """Get recent ball trajectory points."""
        return list(self.prev_positions)[-num_points:]

    def is_ball_in_motion(self, threshold_kmh: float = 5.0) -> bool:
        """Check if ball is moving above threshold speed."""
        if len(self.prev_positions) < 2:
            return False

        # Simple check using last two positions
        p1 = self.prev_positions[-2]
        p2 = self.prev_positions[-1]

        dist_pixels = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
        dist_meters = dist_pixels * 0.05  # Rough conversion

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

        # Draw ball circle
        cv2.circle(
            annotated,
            (ball.pixel_position.x, ball.pixel_position.y),
            max(10, (ball.bbox.x2 - ball.bbox.x1) // 2),
            (0, 255, 255),  # Yellow
            2
        )

        # Draw trajectory
        if draw_trajectory and len(self.prev_positions) > 1:
            points = list(self.prev_positions)
            for i in range(1, len(points)):
                alpha = i / len(points)
                color = (0, int(255 * alpha), int(255 * alpha))
                cv2.line(annotated, points[i-1], points[i], color, 2)

        # Draw velocity vector
        if ball.velocity and ball.velocity.speed_kmh > 5:
            scale = 2  # Scale factor for visualization
            end_x = int(ball.pixel_position.x + ball.velocity.vx * scale)
            end_y = int(ball.pixel_position.y + ball.velocity.vy * scale)
            cv2.arrowedLine(
                annotated,
                (ball.pixel_position.x, ball.pixel_position.y),
                (end_x, end_y),
                (255, 0, 255),  # Magenta
                2
            )

        return annotated

    def reset(self):
        """Reset ball tracking state."""
        self.prev_frame = None
        self.prev_positions.clear()
        self.kalman = self._init_kalman_filter()
