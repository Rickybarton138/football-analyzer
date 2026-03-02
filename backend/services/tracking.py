"""
Player Tracking Service

Uses supervision ByteTrack for multi-object tracking with consistent player IDs.
Replaces the custom ByteTrack implementation with the battle-tested supervision
library while keeping the exact same public API.

Falls back to a minimal IoU tracker if supervision is not installed.
"""
import numpy as np
import cv2
from typing import List, Dict, Optional, Tuple
from collections import defaultdict, deque, Counter

from config import settings
from models.schemas import DetectedPlayer, BoundingBox, PixelPosition, TeamSide


class TrackingService:
    """
    Multi-object tracking service using supervision ByteTrack.

    Maintains consistent player IDs across frames and handles:
    - Track initialization from detections
    - ByteTrack matching (high + low confidence two-pass)
    - Team temporal smoothing (30-frame majority vote)
    - Track history for analytics
    """

    EXPECTED_PLAYERS = 22
    MIN_EXPECTED_PLAYERS = 18

    def __init__(
        self,
        track_activation_threshold: float = None,
        lost_track_buffer: int = None,
        minimum_matching_threshold: float = None,
    ):
        self.track_activation_threshold = (
            track_activation_threshold or settings.TRACK_ACTIVATION_THRESHOLD
        )
        self.lost_track_buffer = lost_track_buffer or settings.TRACK_LOST_BUFFER
        self.minimum_matching_threshold = (
            minimum_matching_threshold or settings.TRACK_MINIMUM_MATCHING_THRESHOLD
        )

        self.frame_count = 0

        # Track metadata (team, color, gk flag) keyed by track_id
        self._track_meta: Dict[int, dict] = {}

        # Team temporal smoothing: track_id -> deque of recent TeamSide assignments
        self._team_history: Dict[int, deque] = defaultdict(
            lambda: deque(maxlen=settings.TEAM_TEMPORAL_SMOOTHING_WINDOW)
        )

        # Track history for analytics: track_id -> [(frame_number, bbox_array)]
        self.track_history: Dict[int, List[Tuple[int, np.ndarray]]] = defaultdict(list)

        # Player count statistics for monitoring
        self.player_count_history: List[int] = []

        # Initialize tracker
        self._sv_tracker = None
        self._sv_available = False
        self._init_tracker()

    def _init_tracker(self):
        """Initialize supervision ByteTrack or fallback."""
        try:
            import supervision as sv
            self._sv_tracker = sv.ByteTrack(
                track_activation_threshold=self.track_activation_threshold,
                lost_track_buffer=self.lost_track_buffer,
                minimum_matching_threshold=self.minimum_matching_threshold,
                frame_rate=30,
            )
            self._sv_available = True
            print(f"[TRACKING] supervision ByteTrack initialized "
                  f"(buffer={self.lost_track_buffer}, thresh={self.minimum_matching_threshold})")
        except ImportError:
            print("[TRACKING] supervision not available, using minimal IoU fallback tracker")
            self._sv_available = False
            self._fallback_next_id = 1
            self._fallback_tracks: Dict[int, dict] = {}

    async def update(
        self,
        detections: List[DetectedPlayer],
        frame_number: int,
        frame: np.ndarray = None
    ) -> List[DetectedPlayer]:
        """
        Update tracks with new detections.

        Args:
            detections: List of detected players (without track IDs)
            frame_number: Current frame number
            frame: Optional BGR frame (kept for API compat, not used by ByteTrack)

        Returns:
            List of detected players with assigned track IDs
        """
        self.frame_count = frame_number

        if not detections:
            self.player_count_history.append(0)
            return []

        if self._sv_available:
            tracked = self._update_supervision(detections)
        else:
            tracked = self._update_fallback(detections)

        # Apply team temporal smoothing to all tracked players
        for player in tracked:
            if player.track_id >= 0:
                player.team = self._smooth_team(player.track_id, player.team)

                # Update track metadata
                self._track_meta[player.track_id] = {
                    "team": player.team,
                    "jersey_color": player.jersey_color,
                    "is_goalkeeper": player.is_goalkeeper,
                }

                # Store in history
                bbox_arr = np.array([
                    player.bbox.x1, player.bbox.y1,
                    player.bbox.x2, player.bbox.y2
                ])
                self.track_history[player.track_id].append(
                    (frame_number, bbox_arr)
                )

        self.player_count_history.append(len(tracked))
        return tracked

    def _update_supervision(self, detections: List[DetectedPlayer]) -> List[DetectedPlayer]:
        """Update tracking using supervision ByteTrack."""
        import supervision as sv

        # Build numpy arrays for supervision Detections
        n = len(detections)
        xyxy = np.zeros((n, 4), dtype=np.float32)
        confidence = np.zeros(n, dtype=np.float32)
        class_id = np.zeros(n, dtype=int)

        for i, det in enumerate(detections):
            xyxy[i] = [det.bbox.x1, det.bbox.y1, det.bbox.x2, det.bbox.y2]
            confidence[i] = det.bbox.confidence
            class_id[i] = 0  # All players are class 0 for tracking

        sv_dets = sv.Detections(
            xyxy=xyxy,
            confidence=confidence,
            class_id=class_id,
        )

        # Run ByteTrack
        tracked_dets = self._sv_tracker.update_with_detections(sv_dets)

        # Map tracked results back to DetectedPlayer objects
        tracked_players = []
        for i in range(len(tracked_dets)):
            x1, y1, x2, y2 = tracked_dets.xyxy[i].astype(int)
            conf = float(tracked_dets.confidence[i])
            track_id = int(tracked_dets.tracker_id[i])

            # Find the best matching original detection for metadata
            best_det = self._find_closest_detection(
                x1, y1, x2, y2, detections
            )

            team = best_det.team if best_det else TeamSide.UNKNOWN
            jersey_color = best_det.jersey_color if best_det else None
            is_gk = best_det.is_goalkeeper if best_det else False

            # Inherit metadata from track history if detection is ambiguous
            if team == TeamSide.UNKNOWN and track_id in self._track_meta:
                meta = self._track_meta[track_id]
                team = meta.get("team", TeamSide.UNKNOWN)
                if jersey_color is None:
                    jersey_color = meta.get("jersey_color")
                if not is_gk:
                    is_gk = meta.get("is_goalkeeper", False)

            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            player = DetectedPlayer(
                track_id=track_id,
                bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, confidence=conf),
                pixel_position=PixelPosition(x=center_x, y=center_y),
                team=team,
                jersey_color=jersey_color,
                is_goalkeeper=is_gk,
            )
            tracked_players.append(player)

        return tracked_players

    def _update_fallback(self, detections: List[DetectedPlayer]) -> List[DetectedPlayer]:
        """Minimal IoU-based tracking fallback when supervision is not available."""
        tracked_players = []

        # Simple greedy IoU matching
        det_bboxes = []
        for det in detections:
            det_bboxes.append(np.array([det.bbox.x1, det.bbox.y1, det.bbox.x2, det.bbox.y2]))

        matched_tracks = set()
        matched_dets = set()
        matches = []

        # Match existing tracks to detections by IoU
        for tid, track in self._fallback_tracks.items():
            best_iou = 0
            best_idx = -1
            for j, det_bbox in enumerate(det_bboxes):
                if j in matched_dets:
                    continue
                iou = self._calculate_iou(track["bbox"], det_bbox)
                if iou > best_iou and iou > 0.3:
                    best_iou = iou
                    best_idx = j
            if best_idx >= 0:
                matches.append((tid, best_idx))
                matched_tracks.add(tid)
                matched_dets.add(best_idx)

        # Update matched tracks
        for tid, det_idx in matches:
            det = detections[det_idx]
            self._fallback_tracks[tid]["bbox"] = det_bboxes[det_idx]
            self._fallback_tracks[tid]["age"] = 0

            player = DetectedPlayer(
                track_id=tid,
                bbox=det.bbox,
                pixel_position=det.pixel_position,
                team=det.team,
                jersey_color=det.jersey_color,
                is_goalkeeper=det.is_goalkeeper,
            )
            tracked_players.append(player)

        # Create new tracks for unmatched detections
        for j, det in enumerate(detections):
            if j in matched_dets:
                continue
            tid = self._fallback_next_id
            self._fallback_next_id += 1
            self._fallback_tracks[tid] = {
                "bbox": det_bboxes[j],
                "age": 0,
            }

            player = DetectedPlayer(
                track_id=tid,
                bbox=det.bbox,
                pixel_position=det.pixel_position,
                team=det.team,
                jersey_color=det.jersey_color,
                is_goalkeeper=det.is_goalkeeper,
            )
            tracked_players.append(player)

        # Age and remove stale tracks
        for tid in list(self._fallback_tracks.keys()):
            if tid not in matched_tracks:
                self._fallback_tracks[tid]["age"] += 1
                if self._fallback_tracks[tid]["age"] > self.lost_track_buffer:
                    del self._fallback_tracks[tid]

        return tracked_players

    def _find_closest_detection(
        self,
        x1: int, y1: int, x2: int, y2: int,
        detections: List[DetectedPlayer]
    ) -> Optional[DetectedPlayer]:
        """Find the detection with highest IoU to a tracked box."""
        best_iou = 0
        best_det = None
        tracked_bbox = np.array([x1, y1, x2, y2], dtype=float)

        for det in detections:
            det_bbox = np.array([det.bbox.x1, det.bbox.y1, det.bbox.x2, det.bbox.y2], dtype=float)
            iou = self._calculate_iou(tracked_bbox, det_bbox)
            if iou > best_iou:
                best_iou = iou
                best_det = det

        return best_det

    def _smooth_team(self, track_id: int, current_team: TeamSide) -> TeamSide:
        """Apply temporal smoothing to team assignment using majority vote."""
        if current_team != TeamSide.UNKNOWN:
            self._team_history[track_id].append(current_team)

        history = self._team_history[track_id]
        if len(history) < 2:
            return current_team

        counts = Counter(history)
        most_common_team, most_common_count = counts.most_common(1)[0]

        if most_common_count / len(history) >= settings.TEAM_CONSENSUS_THRESHOLD:
            return most_common_team

        return current_team

    def _calculate_iou(self, bbox1: np.ndarray, bbox2: np.ndarray) -> float:
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

    def get_player_count_stats(self) -> Dict:
        """Get statistics about player detection counts."""
        if not self.player_count_history:
            return {"avg": 0, "min": 0, "max": 0, "current": 0}

        return {
            "avg": sum(self.player_count_history) / len(self.player_count_history),
            "min": min(self.player_count_history),
            "max": max(self.player_count_history),
            "current": self.player_count_history[-1] if self.player_count_history else 0,
            "frames_below_22": sum(1 for c in self.player_count_history if c < 22),
            "total_frames": len(self.player_count_history),
        }

    def get_track_history(self, track_id: int) -> List[Tuple[int, np.ndarray]]:
        """Get position history for a track."""
        return self.track_history.get(track_id, [])

    def get_active_track_ids(self) -> List[int]:
        """Get IDs of currently active tracks."""
        if self._sv_available:
            # Return track IDs seen in recent frames
            recent = set()
            for tid, history in self.track_history.items():
                if history and self.frame_count - history[-1][0] < 3:
                    recent.add(tid)
            return list(recent)
        else:
            return list(self._fallback_tracks.keys())

    def reset(self):
        """Reset all tracking state."""
        self.track_history.clear()
        self.player_count_history.clear()
        self._track_meta.clear()
        self._team_history.clear()
        self.frame_count = 0

        if self._sv_available:
            self._init_tracker()
        else:
            self._fallback_next_id = 1
            self._fallback_tracks.clear()
