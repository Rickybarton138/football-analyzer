"""
Player Detection Service

Uses YOLOv8 for detecting players and classifying teams by jersey color.
Supports both local CPU inference and cloud GPU inference.
Enhanced with multi-scale detection and player count validation for
consistent tracking of all 22 players on the pitch.
"""
import asyncio
from typing import List, Optional, Tuple, Dict
import numpy as np
import cv2
from sklearn.cluster import KMeans

from config import settings
from models.schemas import DetectedPlayer, BoundingBox, PixelPosition, TeamSide


class DetectionService:
    """Service for player detection using YOLOv8."""

    # Expected player counts
    EXPECTED_FIELD_PLAYERS = 22  # 11 per team
    MIN_EXPECTED_PLAYERS = 18  # Allow for some occlusion

    def __init__(self):
        self.model = None
        self.home_color: Optional[np.ndarray] = None
        self.away_color: Optional[np.ndarray] = None
        self.goalkeeper_colors: List[np.ndarray] = []
        self._cloud_client = None

        # Multi-scale detection settings for better coverage
        # More scales = better detection of distant/small players
        self.detection_scales = [1.0, 1.5, 0.75, 0.5]  # Process at multiple scales including upscale
        self.use_multi_scale = True  # Enable for better distant player detection

        # Detection confidence tiers - balanced for accuracy
        self.high_confidence_thresh = 0.5
        self.medium_confidence_thresh = 0.35  # Higher to reduce false positives
        self.low_confidence_thresh = 0.25     # Minimum threshold for distant players

        # NMS settings for merging multi-scale detections
        self.nms_iou_threshold = 0.5  # Standard threshold for NMS

        # Adaptive detection settings
        self.target_player_count = 22
        self.min_acceptable_players = 14  # Can be lower for cameras not showing full pitch
        self.max_detection_retries = 3

        # Maximum players expected per team (to filter obvious outliers)
        self.max_players_per_team = 14  # 11 + subs/staff visible

    async def initialize(self):
        """Initialize the detection model."""
        if settings.CLOUD_INFERENCE_ENABLED:
            await self._init_cloud_client()
        else:
            await self._init_local_model()

    async def _init_local_model(self):
        """Initialize local YOLO model."""
        try:
            from ultralytics import YOLO

            # Use YOLOv8m (medium) for better accuracy, or YOLOv8n (nano) if memory constrained
            # Model performance ranking: yolov8x > yolov8l > yolov8m > yolov8s > yolov8n
            if settings.USE_GPU:
                model_name = "yolov8m.pt"  # Medium model for GPU - good balance
            else:
                model_name = "yolov8s.pt"  # Small model for CPU - better than nano for sports

            print(f"[DETECTION] Loading YOLO model: {model_name}")
            self.model = YOLO(model_name)
            print(f"[DETECTION] Model loaded successfully")

            # Configure for CPU/GPU
            if not settings.USE_GPU:
                self.model.to('cpu')
                print("[DETECTION] Running on CPU")
            else:
                print("[DETECTION] Running on GPU")

        except ImportError as e:
            print(f"Warning: ultralytics not installed ({e}). Using mock detection.")
            self.model = None
        except Exception as e:
            print(f"Warning: Failed to load YOLO model ({e}). Using mock detection.")
            self.model = None

    async def _init_cloud_client(self):
        """Initialize cloud inference client."""
        import httpx
        self._cloud_client = httpx.AsyncClient(
            base_url=settings.CLOUD_INFERENCE_URL,
            headers={"Authorization": f"Bearer {settings.CLOUD_API_KEY}"},
            timeout=settings.CLOUD_TIMEOUT_MS / 1000
        )

    async def detect(self, frame: np.ndarray) -> List[DetectedPlayer]:
        """
        Detect players in a frame.

        Args:
            frame: BGR image as numpy array

        Returns:
            List of detected players
        """
        if settings.CLOUD_INFERENCE_ENABLED and self._cloud_client:
            return await self._detect_cloud(frame)
        else:
            return await self._detect_local(frame)

    async def _detect_local(self, frame: np.ndarray) -> List[DetectedPlayer]:
        """
        Run detection locally using YOLO with multi-scale processing.

        Uses adaptive confidence thresholds and multi-scale detection to
        ensure all 22 players are detected, even when distant or partially occluded.
        """
        if self.model is None:
            # Return mock detection for testing
            return self._mock_detection(frame)

        height, width = frame.shape[:2]

        # Adaptive detection: start with standard settings, lower thresholds if needed
        best_detections = []

        for retry in range(self.max_detection_retries):
            all_detections = []

            # Adjust confidence based on retry (lower each time)
            retry_factor = 1.0 - (retry * 0.3)  # 1.0, 0.7, 0.4

            # Multi-scale detection for better coverage of distant players
            scales_to_use = self.detection_scales if self.use_multi_scale else [1.0]

            for scale in scales_to_use:
                if scale >= 1.0:
                    # Original or upscaled - use medium confidence
                    conf_thresh = self.medium_confidence_thresh * retry_factor
                elif scale >= 0.75:
                    # Slightly downscaled - use low confidence
                    conf_thresh = self.low_confidence_thresh * retry_factor
                else:
                    # Heavily downscaled - use very low confidence
                    conf_thresh = max(0.05, self.low_confidence_thresh * 0.5 * retry_factor)

                # Resize frame
                if scale != 1.0:
                    new_w = int(width * scale)
                    new_h = int(height * scale)
                    scaled_frame = cv2.resize(frame, (new_w, new_h))
                else:
                    scaled_frame = frame

                # Run YOLO inference
                results = self.model(
                    scaled_frame,
                    conf=conf_thresh,
                    classes=[settings.PLAYER_CLASS_ID],  # Only detect persons
                    verbose=False,
                    imgsz=max(320, min(1280, scaled_frame.shape[1]))  # Adaptive image size
                )

                for result in results:
                    boxes = result.boxes
                    if boxes is None:
                        continue

                    for box in boxes:
                        # Scale coordinates back to original frame size
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        if scale != 1.0:
                            x1 = int(x1 / scale)
                            y1 = int(y1 / scale)
                            x2 = int(x2 / scale)
                            y2 = int(y2 / scale)
                        else:
                            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                        # Clamp to frame bounds
                        x1 = max(0, min(x1, width - 1))
                        y1 = max(0, min(y1, height - 1))
                        x2 = max(0, min(x2, width))
                        y2 = max(0, min(y2, height))

                        # Filter out unreasonably sized detections
                        box_width = x2 - x1
                        box_height = y2 - y1

                        # Players should have reasonable aspect ratio and size
                        if box_width < 5 or box_height < 10:
                            continue  # Too small
                        if box_width > width * 0.3 or box_height > height * 0.5:
                            continue  # Too large (probably not a player)

                        aspect_ratio = box_height / max(box_width, 1)
                        if aspect_ratio < 0.5 or aspect_ratio > 5.0:
                            continue  # Unreasonable aspect ratio

                        confidence = float(box.conf[0])

                        all_detections.append({
                            'bbox': [x1, y1, x2, y2],
                            'confidence': confidence,
                            'scale': scale
                        })

            # Apply NMS to merge overlapping detections from different scales
            merged_detections = self._apply_nms(all_detections)

            # Keep the best result so far
            if len(merged_detections) > len(best_detections):
                best_detections = merged_detections

            # If we found enough players, stop retrying
            if len(merged_detections) >= self.min_acceptable_players:
                print(f"[DETECTION] Found {len(merged_detections)} players on retry {retry}")
                break

        # Use the best detection result
        merged_detections = best_detections
        print(f"[DETECTION] Final count: {len(merged_detections)} players detected")

        # Convert to DetectedPlayer objects
        players = []
        for det in merged_detections:
            x1, y1, x2, y2 = det['bbox']
            confidence = det['confidence']

            # Extract jersey color for team classification
            player_roi = frame[y1:y2, x1:x2]
            jersey_color = self._extract_jersey_color(player_roi)

            # Classify team
            team = self._classify_team(jersey_color)

            # Check if goalkeeper
            is_gk = self._is_goalkeeper(jersey_color, x1, x2, frame.shape[1])

            # Calculate center position
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            player = DetectedPlayer(
                track_id=-1,  # Will be assigned by tracker
                bbox=BoundingBox(
                    x1=x1, y1=y1, x2=x2, y2=y2,
                    confidence=confidence
                ),
                pixel_position=PixelPosition(x=center_x, y=center_y),
                team=team,
                jersey_color=jersey_color.tolist() if jersey_color is not None else None,
                is_goalkeeper=is_gk
            )
            players.append(player)

        # Log if we're detecting fewer players than expected
        if len(players) < self.MIN_EXPECTED_PLAYERS:
            print(f"[DETECTION] Warning: Only detected {len(players)} players (expected {self.MIN_EXPECTED_PLAYERS}+)")

        return players

    def _apply_nms(self, detections: List[Dict]) -> List[Dict]:
        """
        Apply Non-Maximum Suppression to merge overlapping detections.

        Keeps the detection with highest confidence when boxes overlap.
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
                if iou > self.nms_iou_threshold:
                    suppressed.add(j)

        return kept

    def _calculate_iou(self, bbox1: List[int], bbox2: List[int]) -> float:
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

    async def _detect_cloud(self, frame: np.ndarray) -> List[DetectedPlayer]:
        """Run detection via cloud GPU service."""
        import base64

        # Encode frame
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        frame_b64 = base64.b64encode(buffer).decode('utf-8')

        # Send to cloud
        response = await self._cloud_client.post(
            "/detect",
            json={"image": frame_b64}
        )

        if response.status_code != 200:
            return []

        # Parse response
        data = response.json()
        players = []

        for det in data.get("detections", []):
            player = DetectedPlayer(
                track_id=det.get("track_id", -1),
                bbox=BoundingBox(**det["bbox"]),
                pixel_position=PixelPosition(**det["position"]),
                team=TeamSide(det.get("team", "unknown")),
                jersey_color=det.get("jersey_color"),
                is_goalkeeper=det.get("is_goalkeeper", False)
            )
            players.append(player)

        return players

    def _mock_detection(self, frame: np.ndarray) -> List[DetectedPlayer]:
        """Generate mock detections for testing without YOLO."""
        height, width = frame.shape[:2]

        # Generate some mock player positions
        mock_players = []
        positions = [
            (0.2, 0.5), (0.3, 0.3), (0.3, 0.7), (0.4, 0.5),
            (0.5, 0.4), (0.5, 0.6), (0.6, 0.3), (0.6, 0.7),
            (0.7, 0.5), (0.8, 0.4), (0.8, 0.6),
            (0.15, 0.5),  # Goalkeeper
        ]

        for i, (px, py) in enumerate(positions):
            x = int(px * width)
            y = int(py * height)

            # Mock bounding box (player ~50px wide, ~100px tall)
            x1 = x - 25
            y1 = y - 50
            x2 = x + 25
            y2 = y + 50

            # Alternate teams
            team = TeamSide.HOME if i < 6 else TeamSide.AWAY
            is_gk = i == 11

            player = DetectedPlayer(
                track_id=-1,
                bbox=BoundingBox(
                    x1=max(0, x1),
                    y1=max(0, y1),
                    x2=min(width, x2),
                    y2=min(height, y2),
                    confidence=0.85
                ),
                pixel_position=PixelPosition(x=x, y=y),
                team=team,
                jersey_color=[255, 0, 0] if team == TeamSide.HOME else [0, 0, 255],
                is_goalkeeper=is_gk
            )
            mock_players.append(player)

        return mock_players

    def _extract_jersey_color(self, player_roi: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract dominant jersey color from player ROI.

        Uses the upper-middle portion of the bounding box (torso area).
        For small bounding boxes (wide-angle footage), uses entire ROI with grass filtering.
        """
        if player_roi.size == 0:
            return None

        h, w = player_roi.shape[:2]

        # For very small bounding boxes (common in wide-angle footage),
        # use the center of the ROI instead of trying to find torso
        if h < 30 or w < 15:
            # Use center region for small detections
            torso_y1 = max(0, int(h * 0.25))
            torso_y2 = min(h, int(h * 0.75))
            torso_x1 = max(0, int(w * 0.1))
            torso_x2 = min(w, int(w * 0.9))
        else:
            # Focus on torso area (upper middle of bounding box) for larger detections
            torso_y1 = int(h * 0.15)
            torso_y2 = int(h * 0.5)
            torso_x1 = int(w * 0.2)
            torso_x2 = int(w * 0.8)

        torso = player_roi[torso_y1:torso_y2, torso_x1:torso_x2]

        if torso.size == 0:
            # Fallback to entire ROI
            torso = player_roi

        # Reshape pixels
        pixels = torso.reshape(-1, 3)

        # Filter out grass-like colors (green dominant pixels)
        # Grass typically has G > R and G > B with moderate saturation
        filtered_pixels = []
        for pixel in pixels:
            b, g, r = pixel  # OpenCV uses BGR
            # Skip grass-like pixels: green is dominant and significantly higher than others
            is_grass = (g > r * 1.2 and g > b * 1.1 and g > 40 and g < 180)
            # Also skip very dark pixels (shadows) and very bright pixels (sky/lines)
            is_dark = (r < 30 and g < 30 and b < 30)
            is_bright = (r > 220 and g > 220 and b > 220)
            if not is_grass and not is_dark and not is_bright:
                filtered_pixels.append(pixel)

        # If we filtered out too much, use original pixels
        if len(filtered_pixels) < 10:
            filtered_pixels = pixels.tolist()

        pixels = np.array(filtered_pixels)

        # Use K-means to find dominant color
        try:
            # Use 2 clusters to separate jersey from remaining noise
            n_clusters = min(2, len(pixels))
            if n_clusters < 1:
                return None

            kmeans = KMeans(n_clusters=n_clusters, n_init=5, max_iter=100, random_state=42)
            kmeans.fit(pixels)

            # Get the cluster with more pixels (usually jersey, not skin)
            labels, counts = np.unique(kmeans.labels_, return_counts=True)
            dominant_idx = labels[np.argmax(counts)]
            dominant_color = kmeans.cluster_centers_[dominant_idx]

            return dominant_color.astype(np.uint8)
        except Exception:
            # Fallback to mean color
            return np.mean(pixels, axis=0).astype(np.uint8)

    def _classify_team(self, jersey_color: Optional[np.ndarray]) -> TeamSide:
        """Classify team based on jersey color."""
        if jersey_color is None:
            return TeamSide.UNKNOWN

        if self.home_color is None or self.away_color is None:
            return TeamSide.UNKNOWN

        # Calculate color distance to each team
        home_dist = np.linalg.norm(jersey_color.astype(float) - self.home_color.astype(float))
        away_dist = np.linalg.norm(jersey_color.astype(float) - self.away_color.astype(float))

        # Use threshold to avoid misclassification
        min_dist = min(home_dist, away_dist)
        if min_dist > 100:  # Too different from both, might be referee
            return TeamSide.UNKNOWN

        return TeamSide.HOME if home_dist < away_dist else TeamSide.AWAY

    def _is_goalkeeper(
        self,
        jersey_color: Optional[np.ndarray],
        x1: int,
        x2: int,
        frame_width: int
    ) -> bool:
        """
        Determine if player is a goalkeeper.

        Based on:
        1. Position near goal (edge of frame)
        2. Distinct jersey color from field players
        """
        if jersey_color is None:
            return False

        # Check position (near edges)
        center_x = (x1 + x2) / 2
        near_left = center_x < frame_width * 0.15
        near_right = center_x > frame_width * 0.85

        if not (near_left or near_right):
            return False

        # Check if color is distinct from team colors
        if self.goalkeeper_colors:
            for gk_color in self.goalkeeper_colors:
                dist = np.linalg.norm(jersey_color.astype(float) - gk_color.astype(float))
                if dist < 50:
                    return True

        return near_left or near_right

    async def set_team_colors(
        self,
        home_color: List[int],
        away_color: List[int],
        home_gk_color: Optional[List[int]] = None,
        away_gk_color: Optional[List[int]] = None
    ):
        """
        Set team jersey colors for classification.

        Args:
            home_color: RGB color of home team
            away_color: RGB color of away team
            home_gk_color: RGB color of home goalkeeper (optional)
            away_gk_color: RGB color of away goalkeeper (optional)
        """
        self.home_color = np.array(home_color, dtype=np.uint8)
        self.away_color = np.array(away_color, dtype=np.uint8)

        self.goalkeeper_colors = []
        if home_gk_color:
            self.goalkeeper_colors.append(np.array(home_gk_color, dtype=np.uint8))
        if away_gk_color:
            self.goalkeeper_colors.append(np.array(away_gk_color, dtype=np.uint8))

    async def auto_detect_team_colors(
        self,
        detections: List[DetectedPlayer]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Automatically detect team colors from player detections.

        Uses K-means clustering on jersey colors to identify two teams.

        Args:
            detections: List of detected players

        Returns:
            Tuple of (home_color, away_color) as numpy arrays
        """
        colors = []
        for det in detections:
            if det.jersey_color:
                colors.append(det.jersey_color)

        if len(colors) < 4:
            raise ValueError("Not enough players detected for team color analysis")

        colors = np.array(colors)

        # Cluster into 2-3 groups (2 teams + possibly referee)
        kmeans = KMeans(n_clusters=2, n_init=10, random_state=42)
        kmeans.fit(colors)

        team_colors = kmeans.cluster_centers_.astype(np.uint8)

        # Assign based on which side of pitch players are on
        # (home team typically starts on left)
        self.home_color = team_colors[0]
        self.away_color = team_colors[1]

        return self.home_color, self.away_color

    def draw_detections(
        self,
        frame: np.ndarray,
        detections: List[DetectedPlayer],
        show_ids: bool = True,
        show_team: bool = True
    ) -> np.ndarray:
        """
        Draw detection boxes and labels on frame.

        Args:
            frame: Original frame
            detections: List of detected players
            show_ids: Show track IDs
            show_team: Show team labels

        Returns:
            Frame with annotations
        """
        annotated = frame.copy()

        for det in detections:
            # Color based on team
            if det.team == TeamSide.HOME:
                color = (0, 255, 0)  # Green
            elif det.team == TeamSide.AWAY:
                color = (0, 0, 255)  # Red
            else:
                color = (128, 128, 128)  # Gray

            # Draw box
            cv2.rectangle(
                annotated,
                (det.bbox.x1, det.bbox.y1),
                (det.bbox.x2, det.bbox.y2),
                color,
                2
            )

            # Draw label
            label_parts = []
            if show_ids and det.track_id >= 0:
                label_parts.append(f"#{det.track_id}")
            if show_team:
                label_parts.append(det.team.value[:1].upper())
            if det.is_goalkeeper:
                label_parts.append("GK")

            if label_parts:
                label = " ".join(label_parts)
                cv2.putText(
                    annotated,
                    label,
                    (det.bbox.x1, det.bbox.y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2
                )

        return annotated
