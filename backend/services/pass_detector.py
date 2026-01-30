"""
Pass Detection Service

Detects passes by tracking ball movement and possession changes.
Determines pass success/failure based on whether the receiving team matches the passing team.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
from enum import Enum


class PassResult(Enum):
    SUCCESSFUL = "successful"
    INTERCEPTED = "intercepted"
    OUT_OF_PLAY = "out_of_play"
    INCOMPLETE = "incomplete"


@dataclass
class Pass:
    """Represents a detected pass."""
    frame_start: int
    frame_end: int
    timestamp_start: float
    timestamp_end: float
    start_position: Tuple[float, float]
    end_position: Tuple[float, float]
    team: str  # 'home' or 'away'
    result: PassResult
    distance_pixels: float
    duration_seconds: float
    speed_pixels_per_sec: float

    # Optional: player IDs if tracking is available
    passer_id: Optional[int] = None
    receiver_id: Optional[int] = None

    @property
    def is_successful(self) -> bool:
        return self.result == PassResult.SUCCESSFUL

    @property
    def is_forward(self) -> bool:
        """Check if pass was played forward (towards opponent goal)."""
        # Assuming home attacks right (positive x direction)
        if self.team == 'home':
            return self.end_position[0] > self.start_position[0]
        else:
            return self.end_position[0] < self.start_position[0]

    @property
    def distance_meters_estimate(self) -> float:
        """Rough estimate assuming 1920px = 105m pitch length."""
        return self.distance_pixels * (105 / 1920)


@dataclass
class PossessionPeriod:
    """A period of sustained possession by one team."""
    team: str
    frame_start: int
    frame_end: int
    timestamp_start: float
    timestamp_end: float
    touches: int = 0
    passes_attempted: int = 0
    passes_completed: int = 0

    @property
    def duration_seconds(self) -> float:
        return self.timestamp_end - self.timestamp_start


@dataclass
class PassStats:
    """Aggregated pass statistics for a team."""
    team: str
    total_passes: int = 0
    successful_passes: int = 0
    intercepted_passes: int = 0
    incomplete_passes: int = 0
    forward_passes: int = 0
    backward_passes: int = 0
    total_distance_pixels: float = 0
    avg_pass_speed: float = 0

    @property
    def pass_accuracy(self) -> float:
        if self.total_passes == 0:
            return 0.0
        return (self.successful_passes / self.total_passes) * 100

    @property
    def forward_pass_ratio(self) -> float:
        if self.total_passes == 0:
            return 0.0
        return (self.forward_passes / self.total_passes) * 100


class PassDetector:
    """
    Detects passes from ball tracking and player positions.

    Pass detection algorithm:
    1. Track ball position across frames
    2. Detect when ball is near a player (possession)
    3. When ball moves away from one player toward another = pass attempt
    4. If receiving player is same team = successful pass
    5. If receiving player is other team = intercepted
    6. If ball goes out of bounds or no reception = incomplete
    """

    def __init__(self):
        self.passes: List[Pass] = []
        self.possession_periods: List[PossessionPeriod] = []
        self.home_stats = PassStats(team='home')
        self.away_stats = PassStats(team='away')

        # Detection parameters
        self.possession_radius = 50  # pixels - ball considered "possessed" if within this distance
        self.min_pass_distance = 30  # pixels - minimum distance to count as pass (not dribble)
        self.max_pass_frames = 90    # max frames for a pass (3 seconds at 30fps)
        self.min_ball_speed = 5      # pixels/frame - minimum speed to detect pass start

        # State tracking
        self.current_possession_team: Optional[str] = None
        self.current_possession_start: Optional[int] = None
        self.ball_history: List[Tuple[int, float, float]] = []  # (frame, x, y)
        self.possession_player_pos: Optional[Tuple[float, float]] = None

    def reset(self):
        """Reset detector state for new analysis."""
        self.passes = []
        self.possession_periods = []
        self.home_stats = PassStats(team='home')
        self.away_stats = PassStats(team='away')
        self.current_possession_team = None
        self.current_possession_start = None
        self.ball_history = []
        self.possession_player_pos = None

    def process_frame(self, frame_data: dict) -> Optional[Pass]:
        """
        Process a single frame and detect passes.

        Args:
            frame_data: Dictionary containing:
                - frame_number: int
                - timestamp: float
                - ball_position: [x, y] or None
                - detections: list of player detections with bbox, team

        Returns:
            Pass object if a pass was completed this frame, None otherwise
        """
        frame_num = frame_data.get('frame_number', 0)
        timestamp = frame_data.get('timestamp', 0.0)
        ball_pos = frame_data.get('ball_position')
        detections = frame_data.get('detections', [])

        if ball_pos is None:
            return None

        ball_x, ball_y = ball_pos[0], ball_pos[1]

        # Add to ball history
        self.ball_history.append((frame_num, ball_x, ball_y))
        if len(self.ball_history) > 150:  # Keep 5 seconds of history
            self.ball_history.pop(0)

        # Find nearest player to ball
        nearest_player = None
        nearest_distance = float('inf')
        nearest_team = None

        for det in detections:
            if det.get('team') not in ['home', 'away']:
                continue

            bbox = det.get('bbox', [0, 0, 0, 0])
            # Player foot position (bottom center of bbox)
            player_x = (bbox[0] + bbox[2]) / 2
            player_y = bbox[3]  # Bottom of bbox

            dist = np.sqrt((ball_x - player_x)**2 + (ball_y - player_y)**2)

            if dist < nearest_distance:
                nearest_distance = dist
                nearest_player = (player_x, player_y)
                nearest_team = det.get('team')

        # Check for possession change
        detected_pass = None

        if nearest_distance < self.possession_radius and nearest_team:
            # Ball is possessed by a player
            if self.current_possession_team != nearest_team:
                # Possession changed
                if self.current_possession_team is not None and self.possession_player_pos is not None:
                    # A pass or turnover occurred
                    detected_pass = self._create_pass(
                        frame_num, timestamp, nearest_player, nearest_team
                    )

                # Start new possession period
                self._end_possession_period(frame_num, timestamp)
                self.current_possession_team = nearest_team
                self.current_possession_start = frame_num
                self.possession_player_pos = nearest_player
                self._start_possession_period(nearest_team, frame_num, timestamp)
            else:
                # Same team still has possession - check if it's a pass within team
                if self.possession_player_pos:
                    dist_from_last = np.sqrt(
                        (nearest_player[0] - self.possession_player_pos[0])**2 +
                        (nearest_player[1] - self.possession_player_pos[1])**2
                    )
                    if dist_from_last > self.min_pass_distance:
                        # Ball moved significantly - likely a pass
                        detected_pass = self._create_pass(
                            frame_num, timestamp, nearest_player, nearest_team
                        )
                        self.possession_player_pos = nearest_player

        return detected_pass

    def _create_pass(self, frame_end: int, timestamp_end: float,
                     end_pos: Tuple[float, float], receiving_team: str) -> Pass:
        """Create a Pass object from current state."""

        # Find pass start from ball history
        frame_start = self.current_possession_start or frame_end - 30
        timestamp_start = timestamp_end - ((frame_end - frame_start) / 30)  # Assume 30fps
        start_pos = self.possession_player_pos or end_pos

        # Calculate pass metrics
        distance = np.sqrt(
            (end_pos[0] - start_pos[0])**2 +
            (end_pos[1] - start_pos[1])**2
        )
        duration = max(0.1, timestamp_end - timestamp_start)
        speed = distance / duration

        # Determine pass result
        if receiving_team == self.current_possession_team:
            result = PassResult.SUCCESSFUL
        else:
            result = PassResult.INTERCEPTED

        passing_team = self.current_possession_team or 'unknown'

        pass_obj = Pass(
            frame_start=frame_start,
            frame_end=frame_end,
            timestamp_start=timestamp_start,
            timestamp_end=timestamp_end,
            start_position=start_pos,
            end_position=end_pos,
            team=passing_team,
            result=result,
            distance_pixels=distance,
            duration_seconds=duration,
            speed_pixels_per_sec=speed
        )

        self.passes.append(pass_obj)
        self._update_stats(pass_obj)

        # Update possession period
        if self.possession_periods:
            period = self.possession_periods[-1]
            period.passes_attempted += 1
            if pass_obj.is_successful:
                period.passes_completed += 1

        return pass_obj

    def _start_possession_period(self, team: str, frame: int, timestamp: float):
        """Start tracking a new possession period."""
        period = PossessionPeriod(
            team=team,
            frame_start=frame,
            frame_end=frame,
            timestamp_start=timestamp,
            timestamp_end=timestamp
        )
        self.possession_periods.append(period)

    def _end_possession_period(self, frame: int, timestamp: float):
        """End the current possession period."""
        if self.possession_periods:
            period = self.possession_periods[-1]
            period.frame_end = frame
            period.timestamp_end = timestamp

    def _update_stats(self, pass_obj: Pass):
        """Update team statistics with new pass."""
        stats = self.home_stats if pass_obj.team == 'home' else self.away_stats

        stats.total_passes += 1
        stats.total_distance_pixels += pass_obj.distance_pixels

        if pass_obj.is_successful:
            stats.successful_passes += 1
        elif pass_obj.result == PassResult.INTERCEPTED:
            stats.intercepted_passes += 1
        else:
            stats.incomplete_passes += 1

        if pass_obj.is_forward:
            stats.forward_passes += 1
        else:
            stats.backward_passes += 1

        # Update average speed
        if stats.total_passes > 0:
            stats.avg_pass_speed = (
                (stats.avg_pass_speed * (stats.total_passes - 1) + pass_obj.speed_pixels_per_sec)
                / stats.total_passes
            )

    def get_possession_percentages(self) -> Dict[str, float]:
        """Calculate possession percentage for each team."""
        home_time = sum(
            p.duration_seconds for p in self.possession_periods
            if p.team == 'home'
        )
        away_time = sum(
            p.duration_seconds for p in self.possession_periods
            if p.team == 'away'
        )

        total = home_time + away_time
        if total == 0:
            return {'home': 50.0, 'away': 50.0}

        return {
            'home': round((home_time / total) * 100, 1),
            'away': round((away_time / total) * 100, 1)
        }

    def get_pass_stats(self) -> Dict:
        """Get comprehensive pass statistics."""
        possession = self.get_possession_percentages()

        return {
            'home': {
                'total_passes': self.home_stats.total_passes,
                'successful_passes': self.home_stats.successful_passes,
                'pass_accuracy': round(self.home_stats.pass_accuracy, 1),
                'forward_passes': self.home_stats.forward_passes,
                'forward_pass_ratio': round(self.home_stats.forward_pass_ratio, 1),
                'avg_pass_distance_meters': round(
                    (self.home_stats.total_distance_pixels / max(1, self.home_stats.total_passes)) * (105/1920), 1
                ),
                'possession_percent': possession['home']
            },
            'away': {
                'total_passes': self.away_stats.total_passes,
                'successful_passes': self.away_stats.successful_passes,
                'pass_accuracy': round(self.away_stats.pass_accuracy, 1),
                'forward_passes': self.away_stats.forward_passes,
                'forward_pass_ratio': round(self.away_stats.forward_pass_ratio, 1),
                'avg_pass_distance_meters': round(
                    (self.away_stats.total_distance_pixels / max(1, self.away_stats.total_passes)) * (105/1920), 1
                ),
                'possession_percent': possession['away']
            },
            'total_passes': len(self.passes),
            'possession_changes': len(self.possession_periods)
        }

    def get_passes_in_period(self, start_time: float, end_time: float) -> List[Pass]:
        """Get all passes within a time period."""
        return [
            p for p in self.passes
            if start_time <= p.timestamp_start <= end_time
        ]

    def analyze_from_frames(self, frame_analyses: List[dict]) -> Dict:
        """
        Analyze passes from a list of frame analysis data.

        Args:
            frame_analyses: List of frame data dictionaries

        Returns:
            Dictionary with pass statistics
        """
        self.reset()

        for frame_data in frame_analyses:
            self.process_frame(frame_data)

        # End final possession period
        if frame_analyses:
            last_frame = frame_analyses[-1]
            self._end_possession_period(
                last_frame.get('frame_number', 0),
                last_frame.get('timestamp', 0)
            )

        return self.get_pass_stats()


# Singleton instance
pass_detector = PassDetector()
