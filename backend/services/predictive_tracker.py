"""
Predictive Player Tracking System

Uses Kalman filtering and motion models to:
1. Track players across frames with consistent IDs
2. Predict positions when players leave the frame
3. Re-identify players when they return to view
4. Estimate player trajectories and likely destinations
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
from enum import Enum
import math
from scipy.optimize import linear_sum_assignment


class PlayerState(Enum):
    """Current visibility state of a tracked player."""
    VISIBLE = "visible"           # Currently detected in frame
    PREDICTED = "predicted"       # Out of frame, position predicted
    OCCLUDED = "occluded"        # Temporarily hidden by another player
    LOST = "lost"                # Lost track, no longer predicting


@dataclass
class KalmanFilter2D:
    """
    2D Kalman filter for position and velocity estimation.

    State vector: [x, y, vx, vy]
    Measurement: [x, y]
    """
    # State vector [x, y, vx, vy]
    x: np.ndarray = field(default_factory=lambda: np.zeros(4))

    # State covariance matrix
    P: np.ndarray = field(default_factory=lambda: np.eye(4) * 100)

    # Process noise covariance
    Q: np.ndarray = field(default_factory=lambda: np.diag([1, 1, 0.5, 0.5]))

    # Measurement noise covariance
    R: np.ndarray = field(default_factory=lambda: np.diag([5, 5]))

    # State transition matrix (will be updated with dt)
    F: np.ndarray = field(default_factory=lambda: np.eye(4))

    # Measurement matrix
    H: np.ndarray = field(default_factory=lambda: np.array([[1, 0, 0, 0], [0, 1, 0, 0]]))

    def __post_init__(self):
        """Initialize matrices properly."""
        self.x = np.zeros(4)
        self.P = np.eye(4) * 100
        self.Q = np.diag([1, 1, 0.5, 0.5])
        self.R = np.diag([5, 5])
        self.F = np.eye(4)
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)

    def initialize(self, x: float, y: float, vx: float = 0, vy: float = 0):
        """Initialize filter with position and optional velocity."""
        self.x = np.array([x, y, vx, vy])
        self.P = np.eye(4) * 100

    def predict(self, dt: float = 1/30) -> np.ndarray:
        """
        Predict next state.

        Args:
            dt: Time step in seconds (default 1/30 for 30fps)

        Returns:
            Predicted state [x, y, vx, vy]
        """
        # Update state transition matrix with time step
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        # Predict state
        self.x = self.F @ self.x

        # Predict covariance
        self.P = self.F @ self.P @ self.F.T + self.Q

        return self.x.copy()

    def update(self, z: np.ndarray) -> np.ndarray:
        """
        Update state with measurement.

        Args:
            z: Measurement [x, y]

        Returns:
            Updated state [x, y, vx, vy]
        """
        # Innovation (measurement residual)
        y = z - self.H @ self.x

        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R

        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # Update state
        self.x = self.x + K @ y

        # Update covariance
        I = np.eye(4)
        self.P = (I - K @ self.H) @ self.P

        return self.x.copy()

    @property
    def position(self) -> Tuple[float, float]:
        """Get current position estimate."""
        return (self.x[0], self.x[1])

    @property
    def velocity(self) -> Tuple[float, float]:
        """Get current velocity estimate."""
        return (self.x[2], self.x[3])

    @property
    def speed(self) -> float:
        """Get current speed (magnitude of velocity)."""
        return np.sqrt(self.x[2]**2 + self.x[3]**2)


@dataclass
class TrackedPlayer:
    """A player being tracked across frames."""
    track_id: int
    team: str  # 'home' or 'away'
    kalman: KalmanFilter2D = field(default_factory=KalmanFilter2D)
    state: PlayerState = PlayerState.VISIBLE

    # Tracking history
    position_history: List[Tuple[float, float]] = field(default_factory=list)
    velocity_history: List[Tuple[float, float]] = field(default_factory=list)

    # Timing
    last_seen_frame: int = 0
    frames_since_seen: int = 0
    total_frames_tracked: int = 0

    # Prediction info
    predicted_positions: List[Tuple[float, float]] = field(default_factory=list)
    exit_direction: Optional[str] = None  # 'left', 'right', 'top', 'bottom'
    exit_position: Optional[Tuple[float, float]] = None

    # Re-identification features
    avg_bbox_size: Tuple[float, float] = (50, 100)  # width, height
    color_histogram: Optional[np.ndarray] = None

    # Movement patterns
    avg_speed: float = 0
    max_speed: float = 0
    direction_tendency: float = 0  # Radians, 0 = moving right

    def __post_init__(self):
        """Initialize Kalman filter."""
        self.kalman = KalmanFilter2D()
        self.position_history = []
        self.velocity_history = []
        self.predicted_positions = []

    def update_with_detection(self, x: float, y: float, frame_num: int,
                              bbox: Optional[Tuple[float, float, float, float]] = None):
        """Update tracker with a new detection."""
        if self.total_frames_tracked == 0:
            # First detection - initialize Kalman filter
            self.kalman.initialize(x, y)
        else:
            # Predict then update
            self.kalman.predict()
            self.kalman.update(np.array([x, y]))

        # Update state
        self.state = PlayerState.VISIBLE
        self.last_seen_frame = frame_num
        self.frames_since_seen = 0
        self.total_frames_tracked += 1

        # Store history
        self.position_history.append((x, y))
        if len(self.position_history) > 100:
            self.position_history.pop(0)

        vel = self.kalman.velocity
        self.velocity_history.append(vel)
        if len(self.velocity_history) > 100:
            self.velocity_history.pop(0)

        # Update movement stats
        speed = self.kalman.speed
        self.avg_speed = (self.avg_speed * (self.total_frames_tracked - 1) + speed) / self.total_frames_tracked
        self.max_speed = max(self.max_speed, speed)

        if speed > 1:
            self.direction_tendency = math.atan2(vel[1], vel[0])

        # Update bbox size estimate
        if bbox:
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            alpha = 0.1  # Smoothing factor
            self.avg_bbox_size = (
                self.avg_bbox_size[0] * (1 - alpha) + w * alpha,
                self.avg_bbox_size[1] * (1 - alpha) + h * alpha
            )

        # Clear predictions since we have a real detection
        self.predicted_positions = []
        self.exit_direction = None
        self.exit_position = None

    def predict_position(self, frame_num: int, frame_width: int = 1920,
                        frame_height: int = 1080) -> Tuple[float, float]:
        """
        Predict position for a frame where player is not detected.

        Returns:
            Predicted (x, y) position
        """
        self.frames_since_seen = frame_num - self.last_seen_frame

        # Predict using Kalman filter
        state = self.kalman.predict()
        pred_x, pred_y = state[0], state[1]

        # Check if player has left the frame
        margin = 50
        if pred_x < -margin:
            self.exit_direction = 'left'
            self.state = PlayerState.PREDICTED
        elif pred_x > frame_width + margin:
            self.exit_direction = 'right'
            self.state = PlayerState.PREDICTED
        elif pred_y < -margin:
            self.exit_direction = 'top'
            self.state = PlayerState.PREDICTED
        elif pred_y > frame_height + margin:
            self.exit_direction = 'bottom'
            self.state = PlayerState.PREDICTED
        else:
            # Still in frame but not detected - likely occluded
            self.state = PlayerState.OCCLUDED

        # Store exit position
        if self.exit_direction and not self.exit_position:
            self.exit_position = (pred_x, pred_y)

        # Store prediction
        self.predicted_positions.append((pred_x, pred_y))

        # Mark as lost if not seen for too long
        if self.frames_since_seen > 90:  # 3 seconds at 30fps
            self.state = PlayerState.LOST

        return (pred_x, pred_y)

    def predict_reentry(self, frames_ahead: int = 30,
                       frame_width: int = 1920,
                       frame_height: int = 1080) -> Optional[Dict]:
        """
        Predict when and where the player might re-enter the frame.

        Args:
            frames_ahead: How many frames to look ahead

        Returns:
            Dictionary with reentry prediction or None
        """
        if self.state != PlayerState.PREDICTED:
            return None

        vx, vy = self.kalman.velocity

        # Check if velocity suggests returning
        if self.exit_direction == 'left' and vx <= 0:
            return None  # Moving further away
        if self.exit_direction == 'right' and vx >= 0:
            return None
        if self.exit_direction == 'top' and vy <= 0:
            return None
        if self.exit_direction == 'bottom' and vy >= 0:
            return None

        # Simulate forward
        x, y = self.kalman.position
        for i in range(1, frames_ahead + 1):
            x += vx / 30  # Assuming 30fps
            y += vy / 30

            # Check if back in frame
            if 0 <= x <= frame_width and 0 <= y <= frame_height:
                return {
                    'frames_until_reentry': i,
                    'reentry_position': (x, y),
                    'reentry_side': self._opposite_direction(self.exit_direction),
                    'confidence': max(0.3, 1.0 - i / frames_ahead)
                }

        return None

    def _opposite_direction(self, direction: str) -> str:
        """Get opposite direction."""
        opposites = {
            'left': 'right',
            'right': 'left',
            'top': 'bottom',
            'bottom': 'top'
        }
        return opposites.get(direction, direction)

    def get_trajectory_prediction(self, frames_ahead: int = 15) -> List[Tuple[float, float]]:
        """
        Get predicted trajectory for the next N frames.

        Args:
            frames_ahead: Number of frames to predict

        Returns:
            List of (x, y) predicted positions
        """
        trajectory = []

        # Save current state
        orig_x = self.kalman.x.copy()
        orig_P = self.kalman.P.copy()

        # Predict forward
        for _ in range(frames_ahead):
            state = self.kalman.predict()
            trajectory.append((state[0], state[1]))

        # Restore state
        self.kalman.x = orig_x
        self.kalman.P = orig_P

        return trajectory

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'track_id': self.track_id,
            'team': self.team,
            'state': self.state.value,
            'position': self.kalman.position,
            'velocity': self.kalman.velocity,
            'speed': self.kalman.speed,
            'last_seen_frame': self.last_seen_frame,
            'frames_since_seen': self.frames_since_seen,
            'exit_direction': self.exit_direction,
            'exit_position': self.exit_position,
            'avg_speed': self.avg_speed,
            'max_speed': self.max_speed,
            'total_frames_tracked': self.total_frames_tracked
        }


class PredictiveTracker:
    """
    Main predictive tracking system for football players.

    Features:
    - Kalman filter-based tracking with velocity estimation
    - Position prediction when players leave frame
    - Re-identification when players return
    - Team-aware tracking
    - Movement pattern analysis
    """

    def __init__(self, frame_width: int = 1920, frame_height: int = 1080):
        self.frame_width = frame_width
        self.frame_height = frame_height

        # Active trackers
        self.trackers: Dict[int, TrackedPlayer] = {}
        self.next_track_id = 1

        # Tracking parameters
        self.max_distance_for_match = 100  # pixels
        self.max_frames_to_predict = 90    # 3 seconds at 30fps
        self.iou_threshold = 0.3

        # Statistics
        self.total_detections = 0
        self.total_predictions = 0
        self.successful_reids = 0

    def reset(self):
        """Reset tracker state."""
        self.trackers = {}
        self.next_track_id = 1
        self.total_detections = 0
        self.total_predictions = 0
        self.successful_reids = 0

    def process_frame(self, frame_num: int, detections: List[Dict],
                      timestamp: float = 0) -> Dict:
        """
        Process a frame and update all trackers.

        Args:
            frame_num: Current frame number
            detections: List of detections with bbox, team, etc.
            timestamp: Frame timestamp

        Returns:
            Dictionary with tracked and predicted players
        """
        # Extract player detections (exclude ball, referee, etc.)
        player_detections = [
            d for d in detections
            if d.get('team') in ['home', 'away']
        ]

        self.total_detections += len(player_detections)

        # Match detections to existing trackers
        matched, unmatched_dets, unmatched_tracks = self._match_detections(
            player_detections, frame_num
        )

        # Update matched trackers
        for track_id, det in matched:
            bbox = det.get('bbox', [0, 0, 0, 0])
            x = (bbox[0] + bbox[2]) / 2  # Center x
            y = bbox[3]  # Bottom y (feet position)

            self.trackers[track_id].update_with_detection(
                x, y, frame_num, tuple(bbox)
            )

        # Create new trackers for unmatched detections
        for det in unmatched_dets:
            bbox = det.get('bbox', [0, 0, 0, 0])
            x = (bbox[0] + bbox[2]) / 2
            y = bbox[3]
            team = det.get('team', 'unknown')

            tracker = TrackedPlayer(
                track_id=self.next_track_id,
                team=team
            )
            tracker.update_with_detection(x, y, frame_num, tuple(bbox))

            self.trackers[self.next_track_id] = tracker
            self.next_track_id += 1

        # Predict positions for unmatched trackers
        for track_id in unmatched_tracks:
            tracker = self.trackers[track_id]
            if tracker.state != PlayerState.LOST:
                tracker.predict_position(
                    frame_num, self.frame_width, self.frame_height
                )
                self.total_predictions += 1

        # Clean up lost trackers
        lost_ids = [
            tid for tid, t in self.trackers.items()
            if t.state == PlayerState.LOST
        ]
        for tid in lost_ids:
            del self.trackers[tid]

        # Build result
        return self._build_frame_result(frame_num, timestamp)

    def _match_detections(self, detections: List[Dict],
                          frame_num: int) -> Tuple[List, List, List]:
        """
        Match detections to existing trackers using Hungarian algorithm.

        Returns:
            (matched_pairs, unmatched_detections, unmatched_tracker_ids)
        """
        if not detections or not self.trackers:
            return [], detections, list(self.trackers.keys())

        # Build cost matrix
        track_ids = list(self.trackers.keys())
        costs = np.zeros((len(detections), len(track_ids)))

        for i, det in enumerate(detections):
            bbox = det.get('bbox', [0, 0, 0, 0])
            det_x = (bbox[0] + bbox[2]) / 2
            det_y = bbox[3]
            det_team = det.get('team', 'unknown')

            for j, track_id in enumerate(track_ids):
                tracker = self.trackers[track_id]

                # Predict where tracker should be
                pred_x, pred_y = tracker.kalman.position

                # Distance cost
                dist = np.sqrt((det_x - pred_x)**2 + (det_y - pred_y)**2)

                # Team mismatch penalty
                if tracker.team != det_team:
                    dist += 500  # Large penalty for team mismatch

                costs[i, j] = dist

        # Hungarian algorithm for optimal assignment
        row_indices, col_indices = linear_sum_assignment(costs)

        matched = []
        used_dets = set()
        used_tracks = set()

        for i, j in zip(row_indices, col_indices):
            if costs[i, j] < self.max_distance_for_match:
                matched.append((track_ids[j], detections[i]))
                used_dets.add(i)
                used_tracks.add(j)

        unmatched_dets = [d for i, d in enumerate(detections) if i not in used_dets]
        unmatched_tracks = [t for j, t in enumerate(track_ids) if j not in used_tracks]

        return matched, unmatched_dets, unmatched_tracks

    def _build_frame_result(self, frame_num: int, timestamp: float) -> Dict:
        """Build result dictionary for the frame."""
        visible_players = []
        predicted_players = []
        occluded_players = []

        for tracker in self.trackers.values():
            player_data = {
                'track_id': tracker.track_id,
                'team': tracker.team,
                'position': tracker.kalman.position,
                'velocity': tracker.kalman.velocity,
                'speed_pixels_per_sec': tracker.kalman.speed * 30,  # Convert to per second
                'speed_meters_per_sec': (tracker.kalman.speed * 30) * (105 / 1920),  # Rough conversion
                'state': tracker.state.value,
                'frames_tracked': tracker.total_frames_tracked
            }

            if tracker.state == PlayerState.VISIBLE:
                visible_players.append(player_data)
            elif tracker.state == PlayerState.PREDICTED:
                player_data['exit_direction'] = tracker.exit_direction
                player_data['exit_position'] = tracker.exit_position
                player_data['frames_since_seen'] = tracker.frames_since_seen

                # Add reentry prediction
                reentry = tracker.predict_reentry()
                if reentry:
                    player_data['reentry_prediction'] = reentry

                predicted_players.append(player_data)
            elif tracker.state == PlayerState.OCCLUDED:
                player_data['frames_since_seen'] = tracker.frames_since_seen
                occluded_players.append(player_data)

        return {
            'frame_number': frame_num,
            'timestamp': timestamp,
            'visible_players': visible_players,
            'predicted_players': predicted_players,
            'occluded_players': occluded_players,
            'total_tracked': len(self.trackers),
            'home_count': sum(1 for t in self.trackers.values() if t.team == 'home'),
            'away_count': sum(1 for t in self.trackers.values() if t.team == 'away')
        }

    def get_player_trajectory(self, track_id: int,
                              frames_ahead: int = 15) -> Optional[Dict]:
        """
        Get trajectory prediction for a specific player.

        Args:
            track_id: The player's track ID
            frames_ahead: Number of frames to predict

        Returns:
            Dictionary with trajectory info or None
        """
        if track_id not in self.trackers:
            return None

        tracker = self.trackers[track_id]
        trajectory = tracker.get_trajectory_prediction(frames_ahead)

        return {
            'track_id': track_id,
            'team': tracker.team,
            'current_position': tracker.kalman.position,
            'current_velocity': tracker.kalman.velocity,
            'predicted_trajectory': trajectory,
            'avg_speed': tracker.avg_speed,
            'direction_tendency': tracker.direction_tendency
        }

    def get_all_trajectories(self, frames_ahead: int = 15) -> Dict:
        """Get trajectory predictions for all tracked players."""
        return {
            'home': [
                self.get_player_trajectory(tid, frames_ahead)
                for tid, t in self.trackers.items()
                if t.team == 'home' and t.state != PlayerState.LOST
            ],
            'away': [
                self.get_player_trajectory(tid, frames_ahead)
                for tid, t in self.trackers.items()
                if t.team == 'away' and t.state != PlayerState.LOST
            ]
        }

    def get_out_of_frame_players(self) -> List[Dict]:
        """Get list of players currently predicted to be out of frame."""
        return [
            {
                'track_id': t.track_id,
                'team': t.team,
                'exit_direction': t.exit_direction,
                'exit_position': t.exit_position,
                'predicted_position': t.kalman.position,
                'frames_since_seen': t.frames_since_seen,
                'reentry_prediction': t.predict_reentry()
            }
            for t in self.trackers.values()
            if t.state == PlayerState.PREDICTED
        ]

    def analyze_from_frames(self, frame_analyses: List[Dict]) -> Dict:
        """
        Analyze a sequence of frames with predictive tracking.

        Args:
            frame_analyses: List of frame data with detections

        Returns:
            Complete tracking analysis
        """
        self.reset()

        tracked_frames = []
        out_of_frame_events = []

        for frame_data in frame_analyses:
            frame_num = frame_data.get('frame_number', 0)
            timestamp = frame_data.get('timestamp', 0)
            detections = frame_data.get('detections', [])

            # Process frame
            result = self.process_frame(frame_num, detections, timestamp)
            tracked_frames.append(result)

            # Record out-of-frame events
            for pred in result.get('predicted_players', []):
                if pred.get('frames_since_seen') == 1:  # Just left frame
                    out_of_frame_events.append({
                        'frame': frame_num,
                        'timestamp': timestamp,
                        'track_id': pred['track_id'],
                        'team': pred['team'],
                        'exit_direction': pred.get('exit_direction'),
                        'exit_position': pred.get('exit_position')
                    })

        # Calculate summary statistics
        total_visible = sum(len(f['visible_players']) for f in tracked_frames)
        total_predicted = sum(len(f['predicted_players']) for f in tracked_frames)
        total_occluded = sum(len(f['occluded_players']) for f in tracked_frames)

        return {
            'summary': {
                'total_frames_analyzed': len(frame_analyses),
                'unique_players_tracked': self.next_track_id - 1,
                'total_visible_detections': total_visible,
                'total_predicted_positions': total_predicted,
                'total_occluded_frames': total_occluded,
                'prediction_ratio': total_predicted / max(1, total_visible + total_predicted)
            },
            'out_of_frame_events': out_of_frame_events,
            'final_trajectories': self.get_all_trajectories(),
            'current_out_of_frame': self.get_out_of_frame_players()
        }


# Singleton instance
predictive_tracker = PredictiveTracker()
