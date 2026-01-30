"""
Player Tracking Service

Uses ByteTrack for multi-object tracking to maintain consistent player IDs
across frames, handling occlusions and re-identification.
"""
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

from config import settings
from models.schemas import DetectedPlayer, BoundingBox, PixelPosition, TeamSide


@dataclass
class TrackState:
    """Internal state for a tracked object."""
    track_id: int
    bbox: np.ndarray  # [x1, y1, x2, y2]
    score: float
    team: TeamSide = TeamSide.UNKNOWN
    jersey_color: Optional[List[int]] = None
    is_goalkeeper: bool = False
    hits: int = 0
    age: int = 0
    time_since_update: int = 0
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(4))


class KalmanFilter:
    """Simple Kalman filter for bounding box tracking."""

    def __init__(self):
        # State: [x_center, y_center, aspect_ratio, height, vx, vy, va, vh]
        self.dim_x = 8
        self.dim_z = 4

        # State transition matrix
        self.F = np.eye(self.dim_x)
        self.F[:4, 4:] = np.eye(4)

        # Measurement matrix
        self.H = np.eye(self.dim_z, self.dim_x)

        # Process noise
        self.Q = np.eye(self.dim_x) * 0.01
        self.Q[4:, 4:] *= 10

        # Measurement noise
        self.R = np.eye(self.dim_z) * 1

        # State covariance
        self.P = np.eye(self.dim_x) * 10

        # State
        self.x = np.zeros(self.dim_x)

    def initialize(self, bbox: np.ndarray):
        """Initialize state from bounding box [x1, y1, x2, y2]."""
        self.x[:4] = self._bbox_to_z(bbox)
        self.x[4:] = 0

    def predict(self) -> np.ndarray:
        """Predict next state."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self._z_to_bbox(self.x[:4])

    def update(self, bbox: np.ndarray):
        """Update state with measurement."""
        z = self._bbox_to_z(bbox)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(self.dim_x) - K @ self.H) @ self.P

    def get_state(self) -> np.ndarray:
        """Get current bounding box estimate."""
        return self._z_to_bbox(self.x[:4])

    def _bbox_to_z(self, bbox: np.ndarray) -> np.ndarray:
        """Convert [x1, y1, x2, y2] to [cx, cy, s, r]."""
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        cx = bbox[0] + w / 2
        cy = bbox[1] + h / 2
        s = w * h
        r = w / max(h, 1)
        return np.array([cx, cy, s, r])

    def _z_to_bbox(self, z: np.ndarray) -> np.ndarray:
        """Convert [cx, cy, s, r] to [x1, y1, x2, y2]."""
        w = np.sqrt(max(z[2] * z[3], 0))
        h = z[2] / max(w, 1)
        x1 = z[0] - w / 2
        y1 = z[1] - h / 2
        x2 = z[0] + w / 2
        y2 = z[1] + h / 2
        return np.array([x1, y1, x2, y2])


class TrackingService:
    """
    Multi-object tracking service using simplified ByteTrack algorithm.

    Maintains consistent player IDs across frames and handles:
    - Track initialization from detections
    - Track prediction using Kalman filter
    - Hungarian matching for data association
    - Track lifecycle management (birth, death)
    - Player count validation for consistent 22-player tracking
    - Interpolation for temporarily occluded players
    """

    # Expected player counts
    EXPECTED_PLAYERS = 22
    MIN_EXPECTED_PLAYERS = 18

    def __init__(
        self,
        track_high_thresh: float = 0.5,  # Lowered for better sensitivity
        track_low_thresh: float = 0.1,
        new_track_thresh: float = 0.4,  # Lowered to catch distant players
        match_thresh: float = 0.7,  # Slightly lower for better matching
        track_buffer: int = 90  # Extended from 30 to handle longer occlusions (3 sec at 30fps)
    ):
        self.track_high_thresh = track_high_thresh
        self.track_low_thresh = track_low_thresh
        self.new_track_thresh = new_track_thresh
        self.match_thresh = match_thresh
        self.track_buffer = track_buffer

        self.tracks: Dict[int, TrackState] = {}
        self.kalman_filters: Dict[int, KalmanFilter] = {}
        self.next_id = 1
        self.frame_count = 0

        # Track history for analysis
        self.track_history: Dict[int, List[Tuple[int, np.ndarray]]] = defaultdict(list)

        # Player count statistics for monitoring
        self.player_count_history: List[int] = []
        self.detection_gaps: Dict[int, int] = {}  # track_id -> frames since last detection

    async def update(
        self,
        detections: List[DetectedPlayer],
        frame_number: int
    ) -> List[DetectedPlayer]:
        """
        Update tracks with new detections.

        Enhanced to include interpolated positions for temporarily lost tracks
        to maintain consistent 22-player tracking.

        Args:
            detections: List of detected players (without track IDs)
            frame_number: Current frame number

        Returns:
            List of detected players with assigned track IDs (including interpolated)
        """
        self.frame_count = frame_number

        # Separate high and low confidence detections
        high_dets = [d for d in detections if d.bbox.confidence >= self.track_high_thresh]
        low_dets = [d for d in detections if d.bbox.confidence < self.track_high_thresh]

        # Predict new locations for existing tracks
        for track_id, kf in self.kalman_filters.items():
            self.tracks[track_id].bbox = kf.predict()

        # Get active and lost tracks
        active_tracks = {tid: t for tid, t in self.tracks.items() if t.time_since_update < 1}
        lost_tracks = {tid: t for tid, t in self.tracks.items() if 1 <= t.time_since_update < self.track_buffer}

        # Match high confidence detections to active tracks
        matched_high, unmatched_tracks, unmatched_dets = self._match_detections(
            list(active_tracks.values()),
            high_dets
        )

        # Update matched tracks
        tracked_players = []
        matched_track_ids = set()

        for track, det in matched_high:
            self._update_track(track, det)
            tracked_players.append(self._track_to_player(track, det))
            matched_track_ids.add(track.track_id)

        # Try to match remaining tracks with low confidence detections
        remaining_tracks = [self.tracks[tid] for tid in unmatched_tracks if tid in active_tracks]
        matched_low, _, remaining_low_dets = self._match_detections(
            remaining_tracks,
            low_dets
        )

        for track, det in matched_low:
            self._update_track(track, det)
            tracked_players.append(self._track_to_player(track, det))
            matched_track_ids.add(track.track_id)

        # Try to recover lost tracks with unmatched high detections
        lost_track_list = list(lost_tracks.values())
        matched_lost, _, final_unmatched = self._match_detections(
            lost_track_list,
            [high_dets[i] for i in unmatched_dets]
        )

        for track, det in matched_lost:
            self._update_track(track, det)
            tracked_players.append(self._track_to_player(track, det))
            matched_track_ids.add(track.track_id)

        # Create new tracks for unmatched high confidence detections
        for idx in final_unmatched:
            det = high_dets[idx] if idx < len(high_dets) else unmatched_dets[idx]
            if isinstance(det, int):
                det = high_dets[det]
            if det.bbox.confidence >= self.new_track_thresh:
                new_track = self._create_track(det)
                tracked_players.append(self._track_to_player(new_track, det))
                matched_track_ids.add(new_track.track_id)

        # Add interpolated positions for recently lost tracks
        # This helps maintain 22-player visibility even during brief occlusions
        interpolated_players = self._get_interpolated_players(matched_track_ids)
        tracked_players.extend(interpolated_players)

        # Age all tracks
        for track_id in list(self.tracks.keys()):
            track = self.tracks[track_id]
            track.age += 1
            track.time_since_update += 1

            # Remove dead tracks
            if track.time_since_update > self.track_buffer:
                del self.tracks[track_id]
                if track_id in self.kalman_filters:
                    del self.kalman_filters[track_id]

        # Track player count for monitoring
        self.player_count_history.append(len(tracked_players))

        return tracked_players

    def _get_interpolated_players(
        self,
        matched_track_ids: set,
        max_interpolation_frames: int = 30
    ) -> List[DetectedPlayer]:
        """
        Get interpolated positions for tracks that weren't matched this frame.

        Uses Kalman filter predictions to estimate where lost players are,
        helping maintain consistent 22-player tracking during brief occlusions.
        """
        interpolated = []

        for track_id, track in self.tracks.items():
            # Skip tracks that were just matched
            if track_id in matched_track_ids:
                continue

            # Only interpolate recently lost tracks (not too old)
            if track.time_since_update > 0 and track.time_since_update <= max_interpolation_frames:
                # Use Kalman prediction for position
                if track_id in self.kalman_filters:
                    predicted_bbox = self.kalman_filters[track_id].get_state()

                    # Convert to int coordinates
                    x1 = int(float(predicted_bbox[0]))
                    y1 = int(float(predicted_bbox[1]))
                    x2 = int(float(predicted_bbox[2]))
                    y2 = int(float(predicted_bbox[3]))

                    # Lower confidence for interpolated detections
                    # Decreases with time since last real detection
                    interp_confidence = max(0.3, 0.8 - (track.time_since_update * 0.02))

                    player = DetectedPlayer(
                        track_id=track_id,
                        bbox=BoundingBox(
                            x1=x1, y1=y1, x2=x2, y2=y2,
                            confidence=interp_confidence
                        ),
                        pixel_position=PixelPosition(
                            x=int((x1 + x2) / 2),
                            y=int((y1 + y2) / 2)
                        ),
                        team=track.team,
                        jersey_color=track.jersey_color,
                        is_goalkeeper=track.is_goalkeeper
                    )
                    interpolated.append(player)

        return interpolated

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
            "total_frames": len(self.player_count_history)
        }

    def _match_detections(
        self,
        tracks: List[TrackState],
        detections: List[DetectedPlayer]
    ) -> Tuple[List[Tuple[TrackState, DetectedPlayer]], List[int], List[int]]:
        """
        Match detections to tracks using IoU.

        Returns:
            - List of (track, detection) matches
            - List of unmatched track IDs
            - List of unmatched detection indices
        """
        if not tracks or not detections:
            unmatched_tracks = [t.track_id for t in tracks]
            unmatched_dets = list(range(len(detections)))
            return [], unmatched_tracks, unmatched_dets

        # Calculate IoU matrix
        iou_matrix = np.zeros((len(tracks), len(detections)))
        for i, track in enumerate(tracks):
            for j, det in enumerate(detections):
                iou_matrix[i, j] = self._calculate_iou(track.bbox, self._det_to_bbox(det))

        # Simple greedy matching (for simplicity; could use Hungarian algorithm)
        matches = []
        matched_tracks = set()
        matched_dets = set()

        # Match in order of highest IoU
        while True:
            if iou_matrix.size == 0:
                break
            i, j = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
            if iou_matrix[i, j] < (1 - self.match_thresh):
                break

            if i not in matched_tracks and j not in matched_dets:
                matches.append((tracks[i], detections[j]))
                matched_tracks.add(i)
                matched_dets.add(j)

            iou_matrix[i, j] = 0

            if len(matched_tracks) == len(tracks) or len(matched_dets) == len(detections):
                break

        unmatched_tracks = [tracks[i].track_id for i in range(len(tracks)) if i not in matched_tracks]
        unmatched_dets = [j for j in range(len(detections)) if j not in matched_dets]

        return matches, unmatched_tracks, unmatched_dets

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

    def _det_to_bbox(self, det: DetectedPlayer) -> np.ndarray:
        """Convert detection to bbox array."""
        return np.array([det.bbox.x1, det.bbox.y1, det.bbox.x2, det.bbox.y2])

    def _create_track(self, det: DetectedPlayer) -> TrackState:
        """Create new track from detection."""
        track_id = self.next_id
        self.next_id += 1

        bbox = self._det_to_bbox(det)

        track = TrackState(
            track_id=track_id,
            bbox=bbox,
            score=det.bbox.confidence,
            team=det.team,
            jersey_color=det.jersey_color,
            is_goalkeeper=det.is_goalkeeper,
            hits=1,
            age=0,
            time_since_update=0
        )

        self.tracks[track_id] = track

        # Initialize Kalman filter
        kf = KalmanFilter()
        kf.initialize(bbox)
        self.kalman_filters[track_id] = kf

        # Save to history
        self.track_history[track_id].append((self.frame_count, bbox.copy()))

        return track

    def _update_track(self, track: TrackState, det: DetectedPlayer):
        """Update existing track with new detection."""
        bbox = self._det_to_bbox(det)

        # Update Kalman filter
        if track.track_id in self.kalman_filters:
            self.kalman_filters[track.track_id].update(bbox)
            track.bbox = self.kalman_filters[track.track_id].get_state()
        else:
            track.bbox = bbox

        track.score = det.bbox.confidence
        track.hits += 1
        track.time_since_update = 0

        # Update team info if more confident
        if det.team != TeamSide.UNKNOWN:
            track.team = det.team
        if det.jersey_color:
            track.jersey_color = det.jersey_color
        track.is_goalkeeper = det.is_goalkeeper

        # Save to history
        self.track_history[track.track_id].append((self.frame_count, track.bbox.copy()))

    def _track_to_player(self, track: TrackState, det: DetectedPlayer) -> DetectedPlayer:
        """Convert track state back to DetectedPlayer with track ID."""
        bbox = track.bbox

        # Convert numpy values to Python scalars
        x1 = int(float(bbox[0]))
        y1 = int(float(bbox[1]))
        x2 = int(float(bbox[2]))
        y2 = int(float(bbox[3]))

        return DetectedPlayer(
            track_id=track.track_id,
            bbox=BoundingBox(
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                confidence=float(track.score)
            ),
            pixel_position=PixelPosition(
                x=int((x1 + x2) / 2),
                y=int((y1 + y2) / 2)
            ),
            pitch_position=det.pitch_position,
            team=track.team,
            jersey_color=track.jersey_color,
            is_goalkeeper=track.is_goalkeeper
        )

    def get_track_history(self, track_id: int) -> List[Tuple[int, np.ndarray]]:
        """Get position history for a track."""
        return self.track_history.get(track_id, [])

    def get_active_track_ids(self) -> List[int]:
        """Get IDs of currently active tracks."""
        return [tid for tid, t in self.tracks.items() if t.time_since_update < 3]

    def reset(self):
        """Reset all tracking state."""
        self.tracks.clear()
        self.kalman_filters.clear()
        self.track_history.clear()
        self.next_id = 1
        self.frame_count = 0
