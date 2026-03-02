"""
Player Detection Service

Uses YOLO with model fallback chain for detecting players and classifying
teams by jersey color. Integrates supervision.Detections as the internal
detection format for seamless bridge to tracking.

Supports: fine-tuned football model > YOLO11 > YOLOv8 fallback chain.
"""
import asyncio
from typing import List, Optional, Tuple, Dict
import numpy as np
import cv2
from sklearn.cluster import KMeans

from config import settings
from models.schemas import DetectedPlayer, BoundingBox, PixelPosition, TeamSide


class DetectionService:
    """Service for player detection using YOLO with model fallback chain."""

    EXPECTED_FIELD_PLAYERS = 22
    MIN_EXPECTED_PLAYERS = 18

    def __init__(self):
        self.model = None
        self.model_name: str = "none"
        self.home_color: Optional[np.ndarray] = None
        self.away_color: Optional[np.ndarray] = None
        self.goalkeeper_colors: List[np.ndarray] = []
        self._cloud_client = None
        self._sv_available = False
        self._team_classifier = None  # SigLIP classifier (lazy init in Step 5)

        # Detection settings
        self.target_player_count = 22
        self.min_acceptable_players = 14

    async def initialize(self):
        """Initialize the detection model."""
        if settings.CLOUD_INFERENCE_ENABLED:
            await self._init_cloud_client()
        else:
            await self._init_local_model()

        # Check if supervision is available
        try:
            import supervision as sv
            self._sv_available = True
            print("[DETECTION] supervision library available")
        except ImportError:
            self._sv_available = False

        # Try to initialize SigLIP team classifier
        try:
            from services.team_classifier import TeamClassifier
            self._team_classifier = TeamClassifier()
            print("[DETECTION] SigLIP team classifier available (lazy init)")
        except ImportError:
            print("[DETECTION] SigLIP team classifier not available, using LAB color fallback")

    async def _init_local_model(self):
        """Initialize local YOLO model with fallback chain."""
        try:
            from ultralytics import YOLO

            # Select model chain based on GPU availability
            model_chain = settings.YOLO_MODEL_CHAIN if settings.USE_GPU else settings.YOLO_MODEL_CHAIN_CPU

            for model_path in model_chain:
                try:
                    print(f"[DETECTION] Trying model: {model_path}")
                    self.model = YOLO(model_path)
                    self.model_name = model_path

                    if not settings.USE_GPU:
                        self.model.to('cpu')
                        print(f"[DETECTION] Model loaded: {model_path} (CPU)")
                    else:
                        print(f"[DETECTION] Model loaded: {model_path} (GPU)")
                    break
                except Exception as e:
                    print(f"[DETECTION] Failed to load {model_path}: {e}")
                    continue

            if self.model is None:
                print("[DETECTION] All models in fallback chain failed. Using mock detection.")

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
        Run detection locally using YOLO at native 1280 resolution.

        Simplified from multi-scale approach - modern YOLO handles scale
        natively at imgsz=1280 which is sufficient for full-pitch footage.
        """
        if self.model is None:
            return self._mock_detection(frame)

        height, width = frame.shape[:2]
        is_football_model = "football" in self.model_name

        # Determine classes to detect
        if is_football_model:
            # Fine-tuned model has football-specific classes: ball, goalkeeper, player, referee
            classes = None  # Use all classes from the football model
        else:
            # COCO model: only detect persons
            classes = [settings.PLAYER_CLASS_ID]

        # Single-pass detection at native resolution
        results = self.model(
            frame,
            conf=settings.DETECTION_CONFIDENCE,
            classes=classes,
            verbose=False,
            imgsz=settings.DETECTION_IMGSZ
        )

        # Convert to supervision Detections if available, else manual parse
        if self._sv_available:
            detections = self._parse_with_supervision(results, frame, is_football_model)
        else:
            detections = self._parse_manual(results, frame, width, height, is_football_model)

        if len(detections) < self.MIN_EXPECTED_PLAYERS:
            print(f"[DETECTION] Warning: Only detected {len(detections)} players (expected {self.MIN_EXPECTED_PLAYERS}+)")

        print(f"[DETECTION] {len(detections)} players detected (model: {self.model_name})")
        return detections

    def _parse_with_supervision(
        self,
        results,
        frame: np.ndarray,
        is_football_model: bool
    ) -> List[DetectedPlayer]:
        """Parse YOLO results via supervision.Detections for consistent format."""
        import supervision as sv

        players = []
        height, width = frame.shape[:2]

        for result in results:
            sv_detections = sv.Detections.from_ultralytics(result)

            if len(sv_detections) == 0:
                continue

            for i in range(len(sv_detections)):
                x1, y1, x2, y2 = sv_detections.xyxy[i].astype(int)
                confidence = float(sv_detections.confidence[i])
                class_id = int(sv_detections.class_id[i])

                # For COCO model, only keep persons
                if not is_football_model and class_id != settings.PLAYER_CLASS_ID:
                    continue

                # For football model, skip ball class (handled by ball_detection)
                if is_football_model and class_id == 0:  # ball class
                    continue

                # Clamp to frame bounds
                x1 = max(0, min(x1, width - 1))
                y1 = max(0, min(y1, height - 1))
                x2 = max(0, min(x2, width))
                y2 = max(0, min(y2, height))

                # Filter unreasonable detections
                if not self._is_valid_player_bbox(x1, y1, x2, y2, width, height):
                    continue

                # Determine if goalkeeper/referee from football model classes
                is_gk = False
                is_referee = False
                if is_football_model:
                    is_gk = (class_id == 1)  # goalkeeper class
                    is_referee = (class_id == 3)  # referee class

                # Extract jersey color for team classification
                box_area = (x2 - x1) * (y2 - y1)
                if box_area < settings.DETECTION_MIN_BOX_AREA:
                    jersey_color = None
                else:
                    player_roi = frame[y1:y2, x1:x2]
                    jersey_color = self._extract_jersey_color(player_roi)

                # Team classification
                if is_referee:
                    team = TeamSide.UNKNOWN
                elif not is_football_model:
                    # COCO model: check referee by color
                    if self._is_referee(jersey_color):
                        team = TeamSide.UNKNOWN
                        is_referee = True
                    else:
                        team = self._classify_team(jersey_color)
                else:
                    team = self._classify_team(jersey_color)

                # Goalkeeper detection for COCO model
                if not is_football_model and not is_gk:
                    is_gk = self._is_goalkeeper(jersey_color, x1, x2, width)

                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                player = DetectedPlayer(
                    track_id=-1,
                    bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, confidence=confidence),
                    pixel_position=PixelPosition(x=center_x, y=center_y),
                    team=team,
                    jersey_color=jersey_color.tolist() if jersey_color is not None else None,
                    is_goalkeeper=is_gk
                )
                players.append(player)

        return players

    def _parse_manual(
        self,
        results,
        frame: np.ndarray,
        width: int,
        height: int,
        is_football_model: bool
    ) -> List[DetectedPlayer]:
        """Parse YOLO results manually (fallback when supervision not available)."""
        players = []

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])

                if not is_football_model and class_id != settings.PLAYER_CLASS_ID:
                    continue

                if is_football_model and class_id == 0:
                    continue

                x1 = max(0, min(x1, width - 1))
                y1 = max(0, min(y1, height - 1))
                x2 = max(0, min(x2, width))
                y2 = max(0, min(y2, height))

                if not self._is_valid_player_bbox(x1, y1, x2, y2, width, height):
                    continue

                is_gk = is_football_model and (class_id == 1)
                is_referee = is_football_model and (class_id == 3)

                box_area = (x2 - x1) * (y2 - y1)
                if box_area < settings.DETECTION_MIN_BOX_AREA:
                    jersey_color = None
                else:
                    player_roi = frame[y1:y2, x1:x2]
                    jersey_color = self._extract_jersey_color(player_roi)

                if is_referee:
                    team = TeamSide.UNKNOWN
                elif not is_football_model and self._is_referee(jersey_color):
                    team = TeamSide.UNKNOWN
                else:
                    team = self._classify_team(jersey_color)

                if not is_football_model and not is_gk:
                    is_gk = self._is_goalkeeper(jersey_color, x1, x2, width)

                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                player = DetectedPlayer(
                    track_id=-1,
                    bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, confidence=confidence),
                    pixel_position=PixelPosition(x=center_x, y=center_y),
                    team=team,
                    jersey_color=jersey_color.tolist() if jersey_color is not None else None,
                    is_goalkeeper=is_gk
                )
                players.append(player)

        return players

    def _is_valid_player_bbox(
        self, x1: int, y1: int, x2: int, y2: int, frame_w: int, frame_h: int
    ) -> bool:
        """Validate that a bounding box is reasonable for a player."""
        box_width = x2 - x1
        box_height = y2 - y1

        if box_width < 5 or box_height < 10:
            return False
        if box_width > frame_w * 0.3 or box_height > frame_h * 0.5:
            return False

        aspect_ratio = box_height / max(box_width, 1)
        if aspect_ratio < settings.DETECTION_ASPECT_RATIO_MIN or aspect_ratio > settings.DETECTION_ASPECT_RATIO_MAX:
            return False

        return True

    async def classify_teams_siglip(
        self, frame: np.ndarray, detections: List[DetectedPlayer]
    ) -> List[DetectedPlayer]:
        """
        Classify teams using SigLIP embeddings if available.
        Called from the processing pipeline after detection.
        Falls back to existing LAB color method silently.
        """
        if self._team_classifier is None:
            return detections

        try:
            return self._team_classifier.classify(frame, detections)
        except Exception as e:
            print(f"[DETECTION] SigLIP classification failed, using color fallback: {e}")
            return detections

    async def _detect_cloud(self, frame: np.ndarray) -> List[DetectedPlayer]:
        """Run detection via cloud GPU service."""
        import base64

        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        frame_b64 = base64.b64encode(buffer).decode('utf-8')

        response = await self._cloud_client.post(
            "/detect",
            json={"image": frame_b64}
        )

        if response.status_code != 200:
            return []

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

        mock_players = []
        positions = [
            (0.2, 0.5), (0.3, 0.3), (0.3, 0.7), (0.4, 0.5),
            (0.5, 0.4), (0.5, 0.6), (0.6, 0.3), (0.6, 0.7),
            (0.7, 0.5), (0.8, 0.4), (0.8, 0.6),
            (0.15, 0.5),
        ]

        for i, (px, py) in enumerate(positions):
            x = int(px * width)
            y = int(py * height)

            x1 = x - 25
            y1 = y - 50
            x2 = x + 25
            y2 = y + 50

            team = TeamSide.HOME if i < 6 else TeamSide.AWAY
            is_gk = i == 11

            player = DetectedPlayer(
                track_id=-1,
                bbox=BoundingBox(
                    x1=max(0, x1), y1=max(0, y1),
                    x2=min(width, x2), y2=min(height, y2),
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
        """
        if player_roi.size == 0:
            return None

        h, w = player_roi.shape[:2]

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

        torso = player_roi[torso_y1:torso_y2, torso_x1:torso_x2]

        if torso.size == 0:
            torso = player_roi

        pixels = torso.reshape(-1, 3)

        # Filter out grass-like colors using HSV
        pixels_bgr = pixels.reshape(-1, 1, 3).astype(np.uint8)
        pixels_hsv = cv2.cvtColor(pixels_bgr, cv2.COLOR_BGR2HSV).reshape(-1, 3)

        filtered_pixels = []
        for i, pixel in enumerate(pixels):
            h_val, s, v = pixels_hsv[i]
            b, g, r = pixel

            is_grass = (35 <= h_val <= 85 and s > 30 and v > 30)
            is_dark = (v < 30)
            is_bright = (r > 220 and g > 220 and b > 220)
            if not is_grass and not is_dark and not is_bright:
                filtered_pixels.append(pixel)

        if len(filtered_pixels) < 10:
            filtered_pixels = pixels.tolist()

        pixels = np.array(filtered_pixels)

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

                if h_val <= 25 and 30 < s < 170 and v > 60:
                    score *= 0.3
                if 35 <= h_val <= 85 and s > 30 and v > 30:
                    score *= 0.2

                if score > best_score:
                    best_score = score
                    best_color = center

            if best_color is not None:
                return best_color.astype(np.uint8)
            return None
        except Exception:
            return np.mean(pixels, axis=0).astype(np.uint8)

    def _bgr_to_lab(self, bgr_color: np.ndarray) -> np.ndarray:
        """Convert a single BGR color to CIELAB."""
        bgr_pixel = bgr_color.astype(np.uint8).reshape(1, 1, 3)
        lab_pixel = cv2.cvtColor(bgr_pixel, cv2.COLOR_BGR2LAB)
        return lab_pixel.reshape(3).astype(float)

    def _classify_team(self, jersey_color: Optional[np.ndarray]) -> TeamSide:
        """Classify team based on jersey color using CIELAB distance."""
        if jersey_color is None:
            return TeamSide.UNKNOWN

        if self.home_color is None or self.away_color is None:
            return TeamSide.UNKNOWN

        color_lab = self._bgr_to_lab(jersey_color)
        home_lab = self._bgr_to_lab(self.home_color)
        away_lab = self._bgr_to_lab(self.away_color)

        home_dist = np.linalg.norm(color_lab - home_lab)
        away_dist = np.linalg.norm(color_lab - away_lab)

        min_dist = min(home_dist, away_dist)
        max_dist = max(home_dist, away_dist)

        if min_dist > settings.TEAM_COLOR_DISTANCE_THRESHOLD:
            return TeamSide.UNKNOWN

        if max_dist > 0 and min_dist / max_dist > settings.TEAM_AMBIGUITY_RATIO:
            return TeamSide.UNKNOWN

        return TeamSide.HOME if home_dist < away_dist else TeamSide.AWAY

    def _is_referee(self, jersey_color: Optional[np.ndarray]) -> bool:
        """Detect referee/linesman by jersey color characteristics."""
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

        if self.home_color is not None and self.away_color is not None:
            color_lab = self._bgr_to_lab(jersey_color)
            home_lab = self._bgr_to_lab(self.home_color)
            away_lab = self._bgr_to_lab(self.away_color)

            home_dist = np.linalg.norm(color_lab - home_lab)
            away_dist = np.linalg.norm(color_lab - away_lab)

            if home_dist > 60 and away_dist > 60:
                return True

        return False

    def _is_goalkeeper(
        self, jersey_color: Optional[np.ndarray], x1: int, x2: int, frame_width: int
    ) -> bool:
        """Determine if player is a goalkeeper based on position + color distinctness."""
        if jersey_color is None:
            return False

        center_x = (x1 + x2) / 2
        near_left = center_x < frame_width * 0.15
        near_right = center_x > frame_width * 0.85

        if not (near_left or near_right):
            return False

        if self.goalkeeper_colors:
            for gk_color in self.goalkeeper_colors:
                gk_lab = self._bgr_to_lab(gk_color)
                color_lab = self._bgr_to_lab(jersey_color)
                dist = np.linalg.norm(color_lab - gk_lab)
                if dist < 50:
                    return True

        color_distinct = False
        if self.home_color is not None and self.away_color is not None:
            color_lab = self._bgr_to_lab(jersey_color)
            home_lab = self._bgr_to_lab(self.home_color)
            away_lab = self._bgr_to_lab(self.away_color)

            home_dist = np.linalg.norm(color_lab - home_lab)
            away_dist = np.linalg.norm(color_lab - away_lab)

            color_distinct = home_dist > 60 or away_dist > 60

        return color_distinct

    async def set_team_colors(
        self,
        home_color: List[int],
        away_color: List[int],
        home_gk_color: Optional[List[int]] = None,
        away_gk_color: Optional[List[int]] = None
    ):
        """Set team jersey colors for classification."""
        self.home_color = np.array(home_color, dtype=np.uint8)
        self.away_color = np.array(away_color, dtype=np.uint8)

        self.goalkeeper_colors = []
        if home_gk_color:
            self.goalkeeper_colors.append(np.array(home_gk_color, dtype=np.uint8))
        if away_gk_color:
            self.goalkeeper_colors.append(np.array(away_gk_color, dtype=np.uint8))

    async def auto_detect_team_colors(
        self, detections: List[DetectedPlayer]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Automatically detect team colors from player detections using K-means."""
        colors = []
        for det in detections:
            if det.jersey_color:
                colors.append(det.jersey_color)

        if len(colors) < 4:
            raise ValueError("Not enough players detected for team color analysis")

        colors = np.array(colors)

        kmeans = KMeans(n_clusters=2, n_init=10, random_state=42)
        kmeans.fit(colors)

        team_colors = kmeans.cluster_centers_.astype(np.uint8)

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
        """Draw detection boxes and labels on frame."""
        annotated = frame.copy()

        for det in detections:
            if det.team == TeamSide.HOME:
                color = (0, 255, 0)
            elif det.team == TeamSide.AWAY:
                color = (0, 0, 255)
            else:
                color = (128, 128, 128)

            cv2.rectangle(
                annotated,
                (det.bbox.x1, det.bbox.y1),
                (det.bbox.x2, det.bbox.y2),
                color, 2
            )

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
                    annotated, label,
                    (det.bbox.x1, det.bbox.y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
                )

        return annotated
