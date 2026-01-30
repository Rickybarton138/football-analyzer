"""
Pitch Mapping Service

Handles homography transformation to convert pixel coordinates to
real-world pitch coordinates in meters.

VEO SIDELINE CAMERA PERSPECTIVE:
- Camera is positioned at the SIDE of the pitch near the halfway line
- Video X-axis (0-1920) = pitch LENGTH (goal to goal)
- Video Y-axis (0-1080) = pitch WIDTH (far touchline at top, near touchline at bottom)
- Perspective distortion: players far from camera appear smaller/closer together
- Homography transformation corrects for this distortion
"""
import numpy as np
import cv2
from typing import List, Optional, Tuple
from dataclasses import dataclass

from config import settings
from models.schemas import Position, PixelPosition


# Video dimensions
VIDEO_WIDTH = 1920
VIDEO_HEIGHT = 1080


@dataclass
class PitchKeypoint:
    """A known point on the pitch for calibration."""
    name: str
    pixel: PixelPosition
    pitch: Position  # Real-world position in meters


class PitchMapper:
    """
    Service for mapping pixel coordinates to pitch coordinates.

    Uses homography transformation based on known pitch landmarks
    (corners, penalty spots, center circle, etc.).

    For VEO sideline camera:
    - Accounts for perspective distortion (far players appear smaller)
    - Maps video coordinates to true bird's eye view pitch coordinates
    """

    def __init__(self):
        self.homography_matrix: Optional[np.ndarray] = None
        self.inverse_homography: Optional[np.ndarray] = None
        self.calibration_points: List[PitchKeypoint] = []

        # Standard pitch dimensions in meters
        self.pitch_length = settings.PITCH_LENGTH  # 105m
        self.pitch_width = settings.PITCH_WIDTH  # 68m

        # Pre-defined pitch landmarks (in meters from bottom-left corner)
        self.pitch_landmarks = self._get_standard_landmarks()

        # Initialize with default VEO sideline camera calibration
        self._setup_default_veo_calibration()

    def _get_standard_landmarks(self) -> dict:
        """Get standard pitch landmark positions in meters."""
        L = self.pitch_length  # 105
        W = self.pitch_width  # 68

        return {
            # Corners
            "bottom_left": Position(x=0, y=0),
            "bottom_right": Position(x=L, y=0),
            "top_left": Position(x=0, y=W),
            "top_right": Position(x=L, y=W),

            # Goal line centers
            "left_goal_center": Position(x=0, y=W/2),
            "right_goal_center": Position(x=L, y=W/2),

            # Center circle
            "center_spot": Position(x=L/2, y=W/2),
            "center_top": Position(x=L/2, y=W/2 + 9.15),
            "center_bottom": Position(x=L/2, y=W/2 - 9.15),

            # Penalty areas (16.5m from goal line, 40.32m wide)
            "left_penalty_top": Position(x=16.5, y=W/2 + 20.16),
            "left_penalty_bottom": Position(x=16.5, y=W/2 - 20.16),
            "right_penalty_top": Position(x=L-16.5, y=W/2 + 20.16),
            "right_penalty_bottom": Position(x=L-16.5, y=W/2 - 20.16),

            # Penalty spots (11m from goal line)
            "left_penalty_spot": Position(x=11, y=W/2),
            "right_penalty_spot": Position(x=L-11, y=W/2),

            # Goal area (5.5m from goal line, 18.32m wide)
            "left_goal_area_top": Position(x=5.5, y=W/2 + 9.16),
            "left_goal_area_bottom": Position(x=5.5, y=W/2 - 9.16),
            "right_goal_area_top": Position(x=L-5.5, y=W/2 + 9.16),
            "right_goal_area_bottom": Position(x=L-5.5, y=W/2 - 9.16),

            # Halfway line
            "halfway_top": Position(x=L/2, y=W),
            "halfway_bottom": Position(x=L/2, y=0),
        }

    def _setup_default_veo_calibration(self):
        """
        Set up default homography for VEO sideline camera using proper
        perspective projection geometry.

        VEO Camera Position (typical setup):
        - Positioned at sideline near halfway line
        - Elevated approximately 6-8 meters (on a platform/stand)
        - Distance from touchline: approximately 5-10 meters
        - Wide-angle lens capturing full pitch length

        The key insight from professional systems (HUDL, Roboflow sports):
        - Use at least 4 corresponding points for basic homography
        - More points improve accuracy
        - Perspective causes far touchline to appear narrower
        - The video Y-axis maps to pitch width (across the pitch)
        - The video X-axis maps to pitch length (goal to goal)

        Since we don't have ML-based line detection, we use a geometric
        model based on typical VEO camera placement and field of view.
        """
        L = self.pitch_length  # 105m
        W = self.pitch_width   # 68m

        # VEO Cam 3 typical field of view and placement:
        # - Camera at halfway line, ~8m high, ~5m from near touchline
        # - Wide angle captures goal line to goal line
        # - Video frame: 1920x1080 pixels
        #
        # For a sideline camera looking ACROSS the pitch:
        # - Video LEFT edge (x=0) = left goal line
        # - Video RIGHT edge (x=1920) = right goal line
        # - Video TOP (y=0) = far touchline (appears smaller due to distance)
        # - Video BOTTOM (y=1080) = near touchline (camera side, appears larger)
        #
        # The perspective transformation follows projective geometry:
        # - Objects further away appear smaller
        # - Parallel lines (touchlines) converge toward vanishing point
        # - The far touchline spans fewer pixels than the near touchline

        # Estimate video coordinates based on VEO typical framing
        # These values approximate where pitch landmarks appear in a typical VEO shot
        # that captures the full pitch with some margin

        # Horizon and vanishing point estimation
        # For an 8m high camera, ~5m from touchline, looking across 68m pitch
        # The far touchline is roughly 73m away, near is ~5m away
        # This creates significant perspective compression (ratio ~14:1)

        # Practical video coordinates (based on typical VEO framing):
        # The pitch typically fills 80-90% of the frame with some grass/sky margin

        # Near touchline (bottom of pitch in video, y ~ 900-1000)
        near_y = 950  # pixels from top
        # Far touchline (top of pitch in video, y ~ 100-200)
        far_y = 150   # pixels from top

        # Goal lines at near touchline (spread wide because close to camera)
        # Typically pitch fills ~90% of horizontal frame
        near_left_x = 80      # Left goal line at near touchline
        near_right_x = 1840   # Right goal line at near touchline

        # Goal lines at far touchline (compressed due to perspective)
        # The compression ratio depends on camera position
        # For typical VEO setup, far touchline appears about 40-60% as wide
        compression = 0.50  # far touchline width / near touchline width
        near_width = near_right_x - near_left_x  # ~1760 pixels
        far_width = near_width * compression  # ~880 pixels
        center_x = VIDEO_WIDTH / 2  # 960

        far_left_x = center_x - far_width / 2   # ~520
        far_right_x = center_x + far_width / 2  # ~1400

        # Halfway line (center of pitch length, vertical line in video)
        # At near touchline, halfway line is at video center
        halfway_near_x = center_x  # 960
        halfway_far_x = center_x   # 960 (vertical line stays at center)

        # Mid-pitch Y position (halfway across the width)
        mid_y = (near_y + far_y) / 2  # ~550

        # Additional calibration points for better accuracy
        # Quarter lines (25% and 75% along pitch length)
        quarter_near_left = near_left_x + (halfway_near_x - near_left_x) / 2
        quarter_near_right = halfway_near_x + (near_right_x - halfway_near_x) / 2
        quarter_far_left = far_left_x + (halfway_far_x - far_left_x) / 2
        quarter_far_right = halfway_far_x + (far_right_x - halfway_far_x) / 2

        # Source points (video pixel coordinates)
        # Using 8 points distributed across the pitch for better homography
        src_points = np.array([
            # Four corners (most important for homography)
            [far_left_x, far_y],        # Far touchline, left goal
            [far_right_x, far_y],       # Far touchline, right goal
            [near_right_x, near_y],     # Near touchline, right goal
            [near_left_x, near_y],      # Near touchline, left goal
            # Halfway line points (help with center accuracy)
            [halfway_far_x, far_y],     # Far touchline, halfway
            [halfway_near_x, near_y],   # Near touchline, halfway
            # Mid-pitch points (improve interior accuracy)
            [quarter_far_left, far_y],  # Far touchline, left quarter
            [quarter_near_right, near_y], # Near touchline, right quarter
        ], dtype=np.float32)

        # Destination points (real pitch coordinates in meters)
        # Standard pitch coordinate system:
        # - X: 0 to 105m (pitch length, left goal to right goal)
        # - Y: 0 to 68m (pitch width, near touchline to far touchline)
        dst_points = np.array([
            # Four corners
            [0, W],              # Far touchline (y=68), left goal (x=0)
            [L, W],              # Far touchline (y=68), right goal (x=105)
            [L, 0],              # Near touchline (y=0), right goal (x=105)
            [0, 0],              # Near touchline (y=0), left goal (x=0)
            # Halfway line points
            [L/2, W],            # Far touchline, halfway (x=52.5)
            [L/2, 0],            # Near touchline, halfway (x=52.5)
            # Quarter points
            [L/4, W],            # Far touchline, left quarter (x=26.25)
            [3*L/4, 0],          # Near touchline, right quarter (x=78.75)
        ], dtype=np.float32)

        # Compute homography matrix
        self.homography_matrix, status = cv2.findHomography(
            src_points, dst_points, cv2.RANSAC, 5.0
        )

        if self.homography_matrix is not None:
            try:
                self.inverse_homography = np.linalg.inv(self.homography_matrix)
            except np.linalg.LinAlgError:
                self.inverse_homography = None

    async def calibrate(
        self,
        pitch_corners: List[PixelPosition],
        additional_points: Optional[List[Tuple[PixelPosition, str]]] = None
    ):
        """
        Calibrate pitch mapping using corner points.

        Args:
            pitch_corners: Four corners in order: [bottom_left, bottom_right, top_right, top_left]
            additional_points: Optional list of (pixel_position, landmark_name) tuples
        """
        if len(pitch_corners) != 4:
            raise ValueError("Exactly 4 corner points required")

        # Source points (pixels)
        src_points = np.array([
            [pitch_corners[0].x, pitch_corners[0].y],  # bottom_left
            [pitch_corners[1].x, pitch_corners[1].y],  # bottom_right
            [pitch_corners[2].x, pitch_corners[2].y],  # top_right
            [pitch_corners[3].x, pitch_corners[3].y],  # top_left
        ], dtype=np.float32)

        # Destination points (pitch coordinates in meters)
        landmarks = self.pitch_landmarks
        dst_points = np.array([
            [landmarks["bottom_left"].x, landmarks["bottom_left"].y],
            [landmarks["bottom_right"].x, landmarks["bottom_right"].y],
            [landmarks["top_right"].x, landmarks["top_right"].y],
            [landmarks["top_left"].x, landmarks["top_left"].y],
        ], dtype=np.float32)

        # Add additional calibration points if provided
        if additional_points:
            for pixel_pos, landmark_name in additional_points:
                if landmark_name in landmarks:
                    src_points = np.vstack([
                        src_points,
                        [[pixel_pos.x, pixel_pos.y]]
                    ])
                    dst_points = np.vstack([
                        dst_points,
                        [[landmarks[landmark_name].x, landmarks[landmark_name].y]]
                    ])

        # Calculate homography matrix
        if len(src_points) == 4:
            self.homography_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        else:
            self.homography_matrix, _ = cv2.findHomography(src_points, dst_points, cv2.RANSAC)

        # Calculate inverse for mapping pitch -> pixels
        self.inverse_homography = np.linalg.inv(self.homography_matrix)

        # Store calibration points
        self.calibration_points = []
        corner_names = ["bottom_left", "bottom_right", "top_right", "top_left"]
        for i, corner in enumerate(pitch_corners):
            self.calibration_points.append(PitchKeypoint(
                name=corner_names[i],
                pixel=corner,
                pitch=landmarks[corner_names[i]]
            ))

    async def auto_calibrate(self, frame: np.ndarray) -> bool:
        """
        Automatically detect pitch lines and calibrate.

        Uses line detection to find pitch boundaries.

        Args:
            frame: Video frame as numpy array

        Returns:
            True if calibration successful
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect edges
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)

        # Detect lines using Hough transform
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=100,
            minLineLength=100,
            maxLineGap=10
        )

        if lines is None:
            return False

        # Find dominant horizontal and vertical lines
        horizontal_lines = []
        vertical_lines = []

        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi

            if abs(angle) < 20 or abs(angle) > 160:
                horizontal_lines.append(line[0])
            elif 70 < abs(angle) < 110:
                vertical_lines.append(line[0])

        if len(horizontal_lines) < 2 or len(vertical_lines) < 2:
            return False

        # Find pitch boundary lines (outermost)
        # Sort horizontal by y-coordinate
        horizontal_lines.sort(key=lambda l: (l[1] + l[3]) / 2)
        top_line = horizontal_lines[-1]
        bottom_line = horizontal_lines[0]

        # Sort vertical by x-coordinate
        vertical_lines.sort(key=lambda l: (l[0] + l[2]) / 2)
        left_line = vertical_lines[0]
        right_line = vertical_lines[-1]

        # Find intersections (corners)
        corners = [
            self._line_intersection(bottom_line, left_line),
            self._line_intersection(bottom_line, right_line),
            self._line_intersection(top_line, right_line),
            self._line_intersection(top_line, left_line),
        ]

        if any(c is None for c in corners):
            return False

        # Calibrate with detected corners
        corner_pixels = [PixelPosition(x=int(c[0]), y=int(c[1])) for c in corners]
        await self.calibrate(corner_pixels)

        return True

    def _line_intersection(
        self,
        line1: np.ndarray,
        line2: np.ndarray
    ) -> Optional[Tuple[float, float]]:
        """Find intersection point of two lines."""
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2

        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-6:
            return None

        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom

        px = x1 + t * (x2 - x1)
        py = y1 + t * (y2 - y1)

        return (px, py)

    def pixel_to_pitch(self, pixel: PixelPosition) -> Optional[Position]:
        """
        Convert pixel coordinates to pitch coordinates.

        Args:
            pixel: Position in pixel coordinates

        Returns:
            Position in pitch coordinates (meters) or None if not calibrated
        """
        if self.homography_matrix is None:
            return None

        # Apply homography transformation
        point = np.array([[[pixel.x, pixel.y]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(point, self.homography_matrix)

        x, y = transformed[0][0]

        # Clamp to pitch bounds
        x = max(0, min(x, self.pitch_length))
        y = max(0, min(y, self.pitch_width))

        return Position(x=float(x), y=float(y))

    def video_to_pitch_normalized(self, video_x: float, video_y: float) -> Tuple[float, float]:
        """
        Convert video pixel coordinates to normalized pitch coordinates (0-100).

        This properly accounts for perspective distortion from the sideline camera.

        Args:
            video_x: X position in video (0-1920)
            video_y: Y position in video (0-1080)

        Returns:
            (pitch_x_pct, pitch_y_pct) where:
            - pitch_x_pct: 0-100 along pitch length (0=left goal, 100=right goal)
            - pitch_y_pct: 0-100 along pitch width (0=near touchline, 100=far touchline)
        """
        if self.homography_matrix is None:
            # Fallback to simple linear mapping if no calibration
            return (video_x / VIDEO_WIDTH * 100, (1 - video_y / VIDEO_HEIGHT) * 100)

        # Apply homography transformation
        point = np.array([[[video_x, video_y]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(point, self.homography_matrix)

        pitch_x_m, pitch_y_m = transformed[0][0]

        # Convert meters to percentage (0-100)
        pitch_x_pct = (pitch_x_m / self.pitch_length) * 100
        pitch_y_pct = (pitch_y_m / self.pitch_width) * 100

        # Clamp to valid range and convert to Python float (not numpy)
        pitch_x_pct = float(max(0, min(100, pitch_x_pct)))
        pitch_y_pct = float(max(0, min(100, pitch_y_pct)))

        return (pitch_x_pct, pitch_y_pct)

    def transform_detections_to_pitch(self, detections: List[dict]) -> List[dict]:
        """
        Transform a list of player detections to pitch coordinates.

        Args:
            detections: List of detection dicts with 'bbox' field

        Returns:
            List of detections with added 'pitch_x' and 'pitch_y' fields (0-100 normalized)
        """
        transformed = []
        for det in detections:
            bbox = det.get('bbox', [0, 0, 0, 0])
            # Use center-bottom of bbox (player's feet position)
            video_x = (bbox[0] + bbox[2]) / 2
            video_y = bbox[3]  # Bottom of bounding box

            pitch_x, pitch_y = self.video_to_pitch_normalized(video_x, video_y)

            transformed_det = det.copy()
            transformed_det['pitch_x'] = pitch_x
            transformed_det['pitch_y'] = pitch_y
            transformed.append(transformed_det)

        return transformed

    def pitch_to_pixel(self, pitch: Position) -> Optional[PixelPosition]:
        """
        Convert pitch coordinates to pixel coordinates.

        Args:
            pitch: Position in pitch coordinates (meters)

        Returns:
            Position in pixel coordinates or None if not calibrated
        """
        if self.inverse_homography is None:
            return None

        point = np.array([[[pitch.x, pitch.y]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(point, self.inverse_homography)

        x, y = transformed[0][0]
        return PixelPosition(x=int(x), y=int(y))

    def calculate_distance(self, pos1: Position, pos2: Position) -> float:
        """
        Calculate real-world distance between two pitch positions.

        Args:
            pos1: First position
            pos2: Second position

        Returns:
            Distance in meters
        """
        return np.sqrt((pos2.x - pos1.x)**2 + (pos2.y - pos1.y)**2)

    def calculate_speed(
        self,
        pos1: Position,
        pos2: Position,
        time_delta_ms: int
    ) -> float:
        """
        Calculate speed between two positions.

        Args:
            pos1: Starting position
            pos2: Ending position
            time_delta_ms: Time difference in milliseconds

        Returns:
            Speed in km/h
        """
        distance_m = self.calculate_distance(pos1, pos2)
        time_s = time_delta_ms / 1000
        speed_ms = distance_m / max(time_s, 0.001)
        return speed_ms * 3.6  # Convert to km/h

    def draw_pitch_overlay(
        self,
        frame: np.ndarray,
        draw_lines: bool = True,
        draw_zones: bool = False
    ) -> np.ndarray:
        """
        Draw pitch lines overlay on frame.

        Args:
            frame: Original frame
            draw_lines: Draw pitch boundary lines
            draw_zones: Draw zone divisions

        Returns:
            Frame with overlay
        """
        if self.inverse_homography is None:
            return frame

        overlay = frame.copy()

        # Draw pitch boundary
        if draw_lines:
            corners = ["bottom_left", "bottom_right", "top_right", "top_left"]
            corner_pixels = []

            for corner_name in corners:
                pitch_pos = self.pitch_landmarks[corner_name]
                pixel = self.pitch_to_pixel(pitch_pos)
                if pixel:
                    corner_pixels.append((pixel.x, pixel.y))

            if len(corner_pixels) == 4:
                pts = np.array(corner_pixels, dtype=np.int32)
                cv2.polylines(overlay, [pts], True, (255, 255, 255), 2)

            # Draw center line
            center_top = self.pitch_to_pixel(self.pitch_landmarks["halfway_top"])
            center_bottom = self.pitch_to_pixel(self.pitch_landmarks["halfway_bottom"])
            if center_top and center_bottom:
                cv2.line(overlay, (center_top.x, center_top.y),
                        (center_bottom.x, center_bottom.y), (255, 255, 255), 2)

            # Draw center circle
            center = self.pitch_to_pixel(self.pitch_landmarks["center_spot"])
            if center:
                cv2.circle(overlay, (center.x, center.y), 5, (255, 255, 255), -1)

        return cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

    def get_zone(self, position: Position) -> str:
        """
        Get tactical zone name for a position.

        Divides pitch into 18 zones (3x6 grid).

        Args:
            position: Pitch position

        Returns:
            Zone name (e.g., "defensive_left", "midfield_center", "attacking_right")
        """
        L = self.pitch_length
        W = self.pitch_width

        # Horizontal zones: defensive (0-35m), midfield (35-70m), attacking (70-105m)
        if position.x < L / 3:
            h_zone = "defensive"
        elif position.x < 2 * L / 3:
            h_zone = "midfield"
        else:
            h_zone = "attacking"

        # Vertical zones: left, center, right
        if position.y < W / 3:
            v_zone = "left"
        elif position.y < 2 * W / 3:
            v_zone = "center"
        else:
            v_zone = "right"

        return f"{h_zone}_{v_zone}"

    def is_in_penalty_area(self, position: Position, side: str = "right") -> bool:
        """Check if position is in penalty area."""
        L = self.pitch_length
        W = self.pitch_width

        if side == "left":
            return position.x < 16.5 and (W/2 - 20.16) < position.y < (W/2 + 20.16)
        else:
            return position.x > (L - 16.5) and (W/2 - 20.16) < position.y < (W/2 + 20.16)

    def is_in_goal_area(self, position: Position, side: str = "right") -> bool:
        """Check if position is in goal area (6-yard box)."""
        L = self.pitch_length
        W = self.pitch_width

        if side == "left":
            return position.x < 5.5 and (W/2 - 9.16) < position.y < (W/2 + 9.16)
        else:
            return position.x > (L - 5.5) and (W/2 - 9.16) < position.y < (W/2 + 9.16)
