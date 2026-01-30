"""
Event Detection Service

Detects match events like passes, shots, tackles, and possession changes
from tracking data and ball position.

Enhanced with:
- Pass direction tracking (forward/sideways/backward)
- Pitch thirds classification
- Team statistics aggregation
- Set piece detection (corners, free kicks, goal kicks, throw-ins)
- Header detection
- Match period tracking (first/second half)
"""
import uuid
from typing import List, Optional, Dict, Tuple
from collections import deque, defaultdict
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

from config import settings
from models.schemas import (
    MatchEvent, EventType, DetectedPlayer, DetectedBall,
    Position, TeamSide
)


class PassDirection(str, Enum):
    """Pass direction relative to attacking goal."""
    FORWARD = "forward"
    SIDEWAYS = "sideways"
    BACKWARD = "backward"


class PitchThird(str, Enum):
    """Horizontal pitch thirds."""
    DEFENSIVE = "defensive"
    MIDDLE = "middle"
    ATTACKING = "attacking"


@dataclass
class PossessionState:
    """Current possession state."""
    team: TeamSide
    player_id: int
    start_time_ms: int
    position: Position
    pitch_third: PitchThird = PitchThird.MIDDLE


@dataclass
class EventCandidate:
    """Potential event being evaluated."""
    event_type: EventType
    start_frame: int
    start_time_ms: int
    start_position: Position
    player_id: int
    team: TeamSide
    confidence: float = 0.0
    metadata: Dict = None


@dataclass
class TeamStatistics:
    """Team statistics for a match period."""
    team: TeamSide
    period: int = 0  # 0 = full match, 1 = first half, 2 = second half

    # Possession
    possession_frames: int = 0
    total_frames: int = 0
    possession_defensive_frames: int = 0
    possession_middle_frames: int = 0
    possession_attacking_frames: int = 0

    # Passing
    passes_total: int = 0
    passes_successful: int = 0
    passes_failed: int = 0
    passes_forward: int = 0
    passes_sideways: int = 0
    passes_backward: int = 0
    passes_defensive_third: int = 0
    passes_middle_third: int = 0
    passes_attacking_third: int = 0
    current_pass_sequence: int = 0
    longest_pass_sequence: int = 0

    # Shooting
    shots_total: int = 0
    shots_on_target: int = 0
    shots_off_target: int = 0
    goals: int = 0

    # Other
    tackles: int = 0
    interceptions: int = 0
    headers: int = 0
    corners: int = 0
    free_kicks: int = 0
    throw_ins: int = 0
    goal_kicks: int = 0

    @property
    def possession_pct(self) -> float:
        if self.total_frames == 0:
            return 0.0
        return (self.possession_frames / self.total_frames) * 100

    @property
    def pass_accuracy_pct(self) -> float:
        if self.passes_total == 0:
            return 0.0
        return (self.passes_successful / self.passes_total) * 100

    @property
    def shot_accuracy_pct(self) -> float:
        if self.shots_total == 0:
            return 0.0
        return (self.shots_on_target / self.shots_total) * 100


@dataclass
class BallState:
    """Current ball state for set piece detection."""
    position: Optional[Position] = None
    pixel_position: Tuple[int, int] = (0, 0)
    is_stationary: bool = False
    stationary_frames: int = 0
    last_move_frame: int = 0


class EventDetectionService:
    """
    Service for detecting match events from tracking data.

    Events detected:
    - Passes (completed and incomplete, with direction)
    - Shots (on target, off target, blocked)
    - Tackles and interceptions
    - Possession changes
    - Headers
    - Set pieces (corners, free kicks, goal kicks, throw-ins)
    - Match periods (first/second half)
    """

    # Thresholds from EVENT_DETECTION_SPEC
    HEADER_ZONE_RATIO = 0.30  # Top 30% of bbox = header zone
    STATIONARY_THRESHOLD_M = 0.5  # Ball movement threshold in meters
    STATIONARY_DURATION_FRAMES = 60  # ~2 seconds at 30fps for set piece
    TACKLE_PROXIMITY_M = 2.0  # ~60 pixels at typical resolution

    # Pitch zone boundaries (normalized 0-100)
    DEFENSIVE_THIRD_END = 33.3
    MIDDLE_THIRD_END = 66.6

    def __init__(self):
        # State tracking
        self.possession: Optional[PossessionState] = None
        self.prev_possession: Optional[PossessionState] = None
        self.events: List[MatchEvent] = []
        self.event_candidates: List[EventCandidate] = []

        # History buffers
        self.ball_history: deque = deque(maxlen=30)  # ~3 seconds at 10 FPS
        self.player_history: Dict[int, deque] = {}

        # Ball state for set piece detection
        self.ball_state = BallState()

        # Detection thresholds
        self.possession_distance = 3.0  # meters - distance to consider ball possessed
        self.pass_min_distance = 5.0  # meters - minimum pass distance
        self.shot_zone_x = 35.0  # meters from goal line to consider shot zone
        self.shot_speed_threshold = 40.0  # km/h minimum for shot
        self.tackle_distance = 2.0  # meters - distance for tackle detection

        # Pitch dimensions
        self.pitch_length = settings.PITCH_LENGTH
        self.pitch_width = settings.PITCH_WIDTH

        # Match period tracking
        self.current_period = 1  # 1 = first half, 2 = second half
        self.frames_since_play = 0
        self.home_attacking_right = True  # Teams swap at half

        # Team statistics
        self.home_stats = TeamStatistics(team=TeamSide.HOME, period=0)
        self.away_stats = TeamStatistics(team=TeamSide.AWAY, period=0)
        self.home_stats_h1 = TeamStatistics(team=TeamSide.HOME, period=1)
        self.away_stats_h1 = TeamStatistics(team=TeamSide.AWAY, period=1)
        self.home_stats_h2 = TeamStatistics(team=TeamSide.HOME, period=2)
        self.away_stats_h2 = TeamStatistics(team=TeamSide.AWAY, period=2)

        # Team color classification
        self.home_team_color: Optional[np.ndarray] = None
        self.away_team_color: Optional[np.ndarray] = None
        self.team_colors_initialized = False

    def set_team_colors(self, home_color: List[int], away_color: List[int]):
        """Set team colors for classification."""
        self.home_team_color = np.array(home_color, dtype=np.uint8)
        self.away_team_color = np.array(away_color, dtype=np.uint8)
        self.team_colors_initialized = True

    def auto_detect_team_colors(self, players: List[DetectedPlayer]) -> bool:
        """
        Auto-detect team colors from player detections using clustering.
        Excludes grass colors (greens) and clusters remaining colors.
        """
        from sklearn.cluster import KMeans

        colors = []
        for player in players:
            if player.jersey_color:
                color = np.array(player.jersey_color)
                # Note: jersey_color from OpenCV is BGR order
                b, g, r = color[0], color[1], color[2]
                # Exclude grass-like colors (high green, low other channels)
                if g > r * 1.2 and g > b * 1.1 and g > 40:
                    continue  # Skip grass-colored detections
                # Also skip very dark pixels (shadows) and very bright pixels
                if (r < 30 and g < 30 and b < 30) or (r > 220 and g > 220 and b > 220):
                    continue
                colors.append(color)

        if len(colors) < 6:
            return False

        colors = np.array(colors)

        try:
            # Cluster into 2 teams
            kmeans = KMeans(n_clusters=2, n_init=10, random_state=42)
            kmeans.fit(colors)

            self.home_team_color = kmeans.cluster_centers_[0].astype(np.uint8)
            self.away_team_color = kmeans.cluster_centers_[1].astype(np.uint8)
            self.team_colors_initialized = True

            print(f"[EVENT_DETECTION] Auto-detected team colors: Home={self.home_team_color.tolist()}, Away={self.away_team_color.tolist()}")
            return True
        except Exception as e:
            print(f"[EVENT_DETECTION] Failed to auto-detect team colors: {e}")
            return False

    def classify_player_team(self, jersey_color: Optional[List[int]]) -> TeamSide:
        """Classify a player's team based on jersey color."""
        if not self.team_colors_initialized or jersey_color is None:
            return TeamSide.UNKNOWN

        color = np.array(jersey_color, dtype=np.uint8)

        home_dist = np.linalg.norm(color.astype(float) - self.home_team_color.astype(float))
        away_dist = np.linalg.norm(color.astype(float) - self.away_team_color.astype(float))

        # Threshold to avoid misclassifying referees etc.
        min_dist = min(home_dist, away_dist)
        if min_dist > 100:
            return TeamSide.UNKNOWN

        return TeamSide.HOME if home_dist < away_dist else TeamSide.AWAY

    def get_pitch_third(self, x: float, team: TeamSide) -> PitchThird:
        """
        Get the pitch third for a position.
        Takes into account which direction the team is attacking.
        """
        # Normalize x to 0-100 scale
        norm_x = (x / self.pitch_length) * 100 if self.pitch_length > 0 else 50

        attacking_right = (team == TeamSide.HOME) == self.home_attacking_right

        if attacking_right:
            # Attacking toward x=100 (high x values)
            if norm_x < self.DEFENSIVE_THIRD_END:
                return PitchThird.DEFENSIVE
            elif norm_x < self.MIDDLE_THIRD_END:
                return PitchThird.MIDDLE
            else:
                return PitchThird.ATTACKING
        else:
            # Attacking toward x=0 (low x values)
            if norm_x > self.MIDDLE_THIRD_END:
                return PitchThird.DEFENSIVE
            elif norm_x > self.DEFENSIVE_THIRD_END:
                return PitchThird.MIDDLE
            else:
                return PitchThird.ATTACKING

    def get_pass_direction(self, start_x: float, end_x: float, team: TeamSide) -> PassDirection:
        """Determine pass direction relative to attacking goal."""
        attacking_right = (team == TeamSide.HOME) == self.home_attacking_right

        dx = end_x - start_x

        if attacking_right:
            if dx > 2:  # Forward toward high x
                return PassDirection.FORWARD
            elif dx < -2:
                return PassDirection.BACKWARD
            else:
                return PassDirection.SIDEWAYS
        else:
            if dx < -2:  # Forward toward low x
                return PassDirection.FORWARD
            elif dx > 2:
                return PassDirection.BACKWARD
            else:
                return PassDirection.SIDEWAYS

    def detect_header(
        self,
        player: DetectedPlayer,
        ball_pixel_y: int
    ) -> bool:
        """Check if ball contact is in the header zone (top 30% of bbox)."""
        # Header zone is top 30% of bounding box
        header_y_threshold = player.bbox.y1 + (player.bbox.y2 - player.bbox.y1) * self.HEADER_ZONE_RATIO

        # Check if ball is in header zone
        if player.bbox.y1 <= ball_pixel_y <= header_y_threshold:
            return True

        return False

    async def process_frame(
        self,
        players: List[DetectedPlayer],
        ball: Optional[DetectedBall],
        frame_number: int,
        timestamp_ms: int
    ) -> List[MatchEvent]:
        """
        Process a frame and detect events.

        Args:
            players: Detected players with pitch positions
            ball: Detected ball with pitch position
            frame_number: Current frame number
            timestamp_ms: Current timestamp in milliseconds

        Returns:
            List of newly detected events
        """
        new_events = []

        # Auto-detect team colors if not initialized
        if not self.team_colors_initialized and len(players) >= 10:
            self.auto_detect_team_colors(players)

        # Classify player teams if colors are set
        if self.team_colors_initialized:
            for player in players:
                if player.team == TeamSide.UNKNOWN:
                    player.team = self.classify_player_team(player.jersey_color)

        # Update histories
        self._update_histories(players, ball, frame_number, timestamp_ms)

        # Update total frames for stats
        self._update_total_frames()

        if ball is None or ball.pitch_position is None:
            self.frames_since_play += 1
            return new_events

        # Check for set piece (stationary ball detection)
        set_piece_event = await self._detect_set_piece(ball, players, frame_number, timestamp_ms)
        if set_piece_event:
            new_events.append(set_piece_event)
            self.events.append(set_piece_event)
            self._update_set_piece_stats(set_piece_event)

        # Determine possession
        possession_change = await self._update_possession(
            players, ball, frame_number, timestamp_ms
        )

        # Check for events
        if possession_change:
            event = await self._classify_possession_change(
                players, ball, frame_number, timestamp_ms
            )
            if event:
                new_events.append(event)
                self.events.append(event)
                self._update_event_stats(event)

        # Track possession time
        if self.possession:
            self._update_possession_stats(self.possession.team, ball.pitch_position.x)

        # Check for shot
        shot_event = await self._detect_shot(ball, frame_number, timestamp_ms)
        if shot_event:
            new_events.append(shot_event)
            self.events.append(shot_event)
            self._update_shot_stats(shot_event)

        # Check for pressing opportunity
        pressing_event = await self._detect_pressing_opportunity(
            players, ball, frame_number, timestamp_ms
        )
        if pressing_event:
            new_events.append(pressing_event)

        # Detect half-time (extended stoppage > 3 minutes)
        if self.frames_since_play > 5400 and self.current_period == 1:
            self.current_period = 2
            self.home_attacking_right = not self.home_attacking_right
            print(f"[EVENT_DETECTION] Second half detected at frame {frame_number}")

        self.frames_since_play = 0  # Reset since we processed ball

        return new_events

    def _update_total_frames(self):
        """Update total frames counter for possession calculation."""
        self.home_stats.total_frames += 1
        self.away_stats.total_frames += 1
        if self.current_period == 1:
            self.home_stats_h1.total_frames += 1
            self.away_stats_h1.total_frames += 1
        else:
            self.home_stats_h2.total_frames += 1
            self.away_stats_h2.total_frames += 1

    def _update_possession_stats(self, team: TeamSide, ball_x: float):
        """Update possession statistics."""
        if team == TeamSide.UNKNOWN:
            return

        stats = self.home_stats if team == TeamSide.HOME else self.away_stats
        period_stats = (self.home_stats_h1 if self.current_period == 1 else self.home_stats_h2) \
            if team == TeamSide.HOME else \
            (self.away_stats_h1 if self.current_period == 1 else self.away_stats_h2)

        stats.possession_frames += 1
        period_stats.possession_frames += 1

        third = self.get_pitch_third(ball_x, team)
        if third == PitchThird.DEFENSIVE:
            stats.possession_defensive_frames += 1
            period_stats.possession_defensive_frames += 1
        elif third == PitchThird.MIDDLE:
            stats.possession_middle_frames += 1
            period_stats.possession_middle_frames += 1
        else:
            stats.possession_attacking_frames += 1
            period_stats.possession_attacking_frames += 1

    def _update_event_stats(self, event: MatchEvent):
        """Update statistics based on an event."""
        if event.team == TeamSide.UNKNOWN:
            return

        stats = self.home_stats if event.team == TeamSide.HOME else self.away_stats
        period_stats = (self.home_stats_h1 if self.current_period == 1 else self.home_stats_h2) \
            if event.team == TeamSide.HOME else \
            (self.away_stats_h1 if self.current_period == 1 else self.away_stats_h2)

        if event.event_type == EventType.PASS:
            stats.passes_total += 1
            period_stats.passes_total += 1

            if event.success:
                stats.passes_successful += 1
                period_stats.passes_successful += 1
                stats.current_pass_sequence += 1
                period_stats.current_pass_sequence += 1
                if stats.current_pass_sequence > stats.longest_pass_sequence:
                    stats.longest_pass_sequence = stats.current_pass_sequence
                if period_stats.current_pass_sequence > period_stats.longest_pass_sequence:
                    period_stats.longest_pass_sequence = period_stats.current_pass_sequence
            else:
                stats.passes_failed += 1
                period_stats.passes_failed += 1
                stats.current_pass_sequence = 0
                period_stats.current_pass_sequence = 0

            # Track pass direction from metadata
            direction = event.metadata.get("direction") if event.metadata else None
            if direction == "forward":
                stats.passes_forward += 1
                period_stats.passes_forward += 1
            elif direction == "sideways":
                stats.passes_sideways += 1
                period_stats.passes_sideways += 1
            elif direction == "backward":
                stats.passes_backward += 1
                period_stats.passes_backward += 1

            # Track pass by third
            third = event.metadata.get("pitch_third") if event.metadata else None
            if third == "defensive":
                stats.passes_defensive_third += 1
                period_stats.passes_defensive_third += 1
            elif third == "middle":
                stats.passes_middle_third += 1
                period_stats.passes_middle_third += 1
            elif third == "attacking":
                stats.passes_attacking_third += 1
                period_stats.passes_attacking_third += 1

            # Check for header
            if event.metadata and event.metadata.get("is_header"):
                stats.headers += 1
                period_stats.headers += 1

        elif event.event_type == EventType.TACKLE:
            stats.tackles += 1
            period_stats.tackles += 1
            # Reset opponent's pass sequence
            opp_stats = self.away_stats if event.team == TeamSide.HOME else self.home_stats
            opp_stats.current_pass_sequence = 0

        elif event.event_type == EventType.INTERCEPTION:
            stats.interceptions += 1
            period_stats.interceptions += 1
            # Reset opponent's pass sequence
            opp_stats = self.away_stats if event.team == TeamSide.HOME else self.home_stats
            opp_stats.current_pass_sequence = 0

    def _update_shot_stats(self, event: MatchEvent):
        """Update shot statistics."""
        if event.team == TeamSide.UNKNOWN:
            return

        stats = self.home_stats if event.team == TeamSide.HOME else self.away_stats
        period_stats = (self.home_stats_h1 if self.current_period == 1 else self.home_stats_h2) \
            if event.team == TeamSide.HOME else \
            (self.away_stats_h1 if self.current_period == 1 else self.away_stats_h2)

        stats.shots_total += 1
        period_stats.shots_total += 1

        on_target = event.metadata.get("on_target", False) if event.metadata else False
        if on_target:
            stats.shots_on_target += 1
            period_stats.shots_on_target += 1
        else:
            stats.shots_off_target += 1
            period_stats.shots_off_target += 1

        if event.metadata and event.metadata.get("is_goal"):
            stats.goals += 1
            period_stats.goals += 1

    def _update_set_piece_stats(self, event: MatchEvent):
        """Update set piece statistics."""
        if event.team == TeamSide.UNKNOWN:
            return

        stats = self.home_stats if event.team == TeamSide.HOME else self.away_stats
        period_stats = (self.home_stats_h1 if self.current_period == 1 else self.home_stats_h2) \
            if event.team == TeamSide.HOME else \
            (self.away_stats_h1 if self.current_period == 1 else self.away_stats_h2)

        if event.event_type == EventType.CORNER:
            stats.corners += 1
            period_stats.corners += 1
        elif event.event_type == EventType.FREE_KICK:
            stats.free_kicks += 1
            period_stats.free_kicks += 1
        elif event.event_type == EventType.THROW_IN:
            stats.throw_ins += 1
            period_stats.throw_ins += 1
        # Goal kicks handled via metadata

    async def _detect_set_piece(
        self,
        ball: DetectedBall,
        players: List[DetectedPlayer],
        frame_number: int,
        timestamp_ms: int
    ) -> Optional[MatchEvent]:
        """
        Detect set piece events based on ball position and stationary duration.
        """
        if ball.pitch_position is None:
            return None

        # Check if ball is stationary
        if self.ball_state.position:
            dist = self._calculate_distance(self.ball_state.position, ball.pitch_position)
            if dist < self.STATIONARY_THRESHOLD_M:
                self.ball_state.stationary_frames += 1
            else:
                self.ball_state.stationary_frames = 0
                self.ball_state.last_move_frame = frame_number
        else:
            self.ball_state.stationary_frames = 0

        self.ball_state.position = ball.pitch_position

        # Need stationary for set piece detection
        if self.ball_state.stationary_frames < self.STATIONARY_DURATION_FRAMES:
            return None

        # Normalize position to 0-100
        norm_x = (ball.pitch_position.x / self.pitch_length) * 100 if self.pitch_length > 0 else 50
        norm_y = (ball.pitch_position.y / self.pitch_width) * 100 if self.pitch_width > 0 else 50

        # Find nearest player to attribute event
        nearest = self._find_nearest_player(players, ball)
        player_id = nearest.track_id if nearest else None
        team = nearest.team if nearest else TeamSide.UNKNOWN

        event_type = None

        # Goal kick: near goal line (x < 5 or x > 95), central area
        if (norm_x < 5 or norm_x > 95) and 20 < norm_y < 80:
            event_type = EventType.CORNER  # Using CORNER for goal kick area

        # Kickoff: center of pitch
        elif 45 <= norm_x <= 55 and 45 <= norm_y <= 55:
            # This is a kickoff/restart
            pass  # Don't record as event, just reset tracking

        # Corner: in corner arcs
        elif (norm_x < 5 or norm_x > 95) and (norm_y < 10 or norm_y > 90):
            event_type = EventType.CORNER

        # Free kick: stationary outside both penalty areas
        elif 18 < norm_x < 82:
            event_type = EventType.FREE_KICK

        if event_type:
            # Reset stationary counter
            self.ball_state.stationary_frames = 0

            return MatchEvent(
                event_id=str(uuid.uuid4()),
                event_type=event_type,
                timestamp_ms=timestamp_ms,
                frame_number=frame_number,
                position=ball.pitch_position,
                player_id=player_id,
                team=team,
                success=True,
                metadata={"set_piece": True}
            )

        return None

    def _find_nearest_player(
        self,
        players: List[DetectedPlayer],
        ball: DetectedBall
    ) -> Optional[DetectedPlayer]:
        """Find nearest player to ball."""
        if not players or ball.pitch_position is None:
            return None

        min_dist = float('inf')
        nearest = None

        for player in players:
            if player.pitch_position is None:
                continue
            dist = self._calculate_distance(player.pitch_position, ball.pitch_position)
            if dist < min_dist:
                min_dist = dist
                nearest = player

        return nearest

    def _update_histories(
        self,
        players: List[DetectedPlayer],
        ball: Optional[DetectedBall],
        frame_number: int,
        timestamp_ms: int
    ):
        """Update position histories."""
        # Ball history
        if ball and ball.pitch_position:
            self.ball_history.append({
                "frame": frame_number,
                "time": timestamp_ms,
                "position": ball.pitch_position,
                "velocity": ball.velocity
            })

        # Player histories
        for player in players:
            if player.pitch_position is None:
                continue

            if player.track_id not in self.player_history:
                self.player_history[player.track_id] = deque(maxlen=30)

            self.player_history[player.track_id].append({
                "frame": frame_number,
                "time": timestamp_ms,
                "position": player.pitch_position,
                "team": player.team
            })

    async def _update_possession(
        self,
        players: List[DetectedPlayer],
        ball: DetectedBall,
        frame_number: int,
        timestamp_ms: int
    ) -> bool:
        """
        Update ball possession state.

        Returns True if possession changed.
        """
        # Find nearest player to ball
        min_distance = float('inf')
        nearest_player = None

        for player in players:
            if player.pitch_position is None:
                continue

            distance = self._calculate_distance(
                player.pitch_position,
                ball.pitch_position
            )

            if distance < min_distance:
                min_distance = distance
                nearest_player = player

        # Check if ball is possessed
        if min_distance > self.possession_distance:
            # Ball is free (no possession)
            return False

        if nearest_player is None:
            return False

        # Check for possession change
        if self.possession is None:
            # First possession
            self.possession = PossessionState(
                team=nearest_player.team,
                player_id=nearest_player.track_id,
                start_time_ms=timestamp_ms,
                position=ball.pitch_position
            )
            return True

        # Check if possession changed
        if (self.possession.player_id != nearest_player.track_id and
            self.possession.team != nearest_player.team):
            # Possession changed teams
            self.prev_possession = self.possession
            self.possession = PossessionState(
                team=nearest_player.team,
                player_id=nearest_player.track_id,
                start_time_ms=timestamp_ms,
                position=ball.pitch_position
            )
            return True

        elif self.possession.player_id != nearest_player.track_id:
            # Same team, different player (potential pass)
            self.prev_possession = self.possession
            self.possession = PossessionState(
                team=nearest_player.team,
                player_id=nearest_player.track_id,
                start_time_ms=timestamp_ms,
                position=ball.pitch_position
            )
            return True

        return False

    async def _classify_possession_change(
        self,
        players: List[DetectedPlayer],
        ball: DetectedBall,
        frame_number: int,
        timestamp_ms: int
    ) -> Optional[MatchEvent]:
        """
        Classify the type of possession change event.

        Returns appropriate event (pass, interception, tackle, etc.)
        Enhanced with pass direction and pitch third tracking.
        """
        if self.prev_possession is None:
            return None

        # Calculate ball travel distance
        ball_distance = self._calculate_distance(
            self.prev_possession.position,
            ball.pitch_position
        )

        # Get pitch third and pass direction
        pitch_third = self.get_pitch_third(
            self.prev_possession.position.x,
            self.prev_possession.team
        )
        pass_direction = self.get_pass_direction(
            self.prev_possession.position.x,
            ball.pitch_position.x,
            self.prev_possession.team
        )

        # Check for header (if ball contact in upper portion of player bbox)
        is_header = False
        if ball.pixel_position:
            current_player = next(
                (p for p in players if p.track_id == self.possession.player_id),
                None
            )
            if current_player:
                is_header = self.detect_header(current_player, ball.pixel_position.y)

        # Same team - likely a pass
        if self.possession.team == self.prev_possession.team:
            if ball_distance >= self.pass_min_distance:
                return MatchEvent(
                    event_id=str(uuid.uuid4()),
                    event_type=EventType.PASS,
                    timestamp_ms=timestamp_ms,
                    frame_number=frame_number,
                    position=self.prev_possession.position,
                    player_id=self.prev_possession.player_id,
                    team=self.prev_possession.team,
                    recipient_id=self.possession.player_id,
                    success=True,
                    metadata={
                        "distance": ball_distance,
                        "direction": pass_direction.value,
                        "pitch_third": pitch_third.value,
                        "is_header": is_header,
                        "start_position": {
                            "x": self.prev_possession.position.x,
                            "y": self.prev_possession.position.y
                        },
                        "end_position": {
                            "x": ball.pitch_position.x,
                            "y": ball.pitch_position.y
                        }
                    }
                )

        # Different team - interception or tackle
        else:
            # Check if there was a duel (players close together)
            prev_player_pos = self._get_player_position(
                self.prev_possession.player_id
            )
            curr_player_pos = self._get_player_position(
                self.possession.player_id
            )

            if prev_player_pos and curr_player_pos:
                player_distance = self._calculate_distance(
                    prev_player_pos, curr_player_pos
                )

                if player_distance <= self.tackle_distance:
                    # Tackle (close contact)
                    return MatchEvent(
                        event_id=str(uuid.uuid4()),
                        event_type=EventType.TACKLE,
                        timestamp_ms=timestamp_ms,
                        frame_number=frame_number,
                        position=ball.pitch_position,
                        player_id=self.possession.player_id,
                        team=self.possession.team,
                        success=True,
                        metadata={
                            "tackled_player": self.prev_possession.player_id,
                            "pitch_third": pitch_third.value
                        }
                    )

            # Interception (no close contact) = failed pass
            return MatchEvent(
                event_id=str(uuid.uuid4()),
                event_type=EventType.PASS,  # Record as failed pass
                timestamp_ms=timestamp_ms,
                frame_number=frame_number,
                position=self.prev_possession.position,
                player_id=self.prev_possession.player_id,
                team=self.prev_possession.team,
                recipient_id=self.possession.player_id,
                success=False,  # Failed pass
                metadata={
                    "distance": ball_distance,
                    "direction": pass_direction.value,
                    "pitch_third": pitch_third.value,
                    "intercepted_by": self.possession.player_id,
                    "intercepted_by_team": self.possession.team.value if self.possession.team else None
                }
            )

        return None

    async def _detect_shot(
        self,
        ball: DetectedBall,
        frame_number: int,
        timestamp_ms: int
    ) -> Optional[MatchEvent]:
        """Detect shot on goal."""
        if ball.velocity is None or ball.pitch_position is None:
            return None

        # Check if in shot zone (near goal)
        in_shot_zone = (
            ball.pitch_position.x > (self.pitch_length - self.shot_zone_x) or
            ball.pitch_position.x < self.shot_zone_x
        )

        if not in_shot_zone:
            return None

        # Check ball speed
        if ball.velocity.speed_kmh < self.shot_speed_threshold:
            return None

        # Check ball direction (moving toward goal)
        if ball.pitch_position.x > self.pitch_length / 2:
            # Attacking right goal
            moving_toward_goal = ball.velocity.vx > 0
        else:
            # Attacking left goal
            moving_toward_goal = ball.velocity.vx < 0

        if not moving_toward_goal:
            return None

        # Determine shooting player
        shooting_player = None
        if self.possession:
            shooting_player = self.possession.player_id

        return MatchEvent(
            event_id=str(uuid.uuid4()),
            event_type=EventType.SHOT,
            timestamp_ms=timestamp_ms,
            frame_number=frame_number,
            position=ball.pitch_position,
            player_id=shooting_player,
            team=self.possession.team if self.possession else None,
            metadata={
                "speed_kmh": ball.velocity.speed_kmh,
                "from_distance": abs(
                    self.pitch_length - ball.pitch_position.x
                    if ball.pitch_position.x > self.pitch_length / 2
                    else ball.pitch_position.x
                )
            }
        )

    async def _detect_pressing_opportunity(
        self,
        players: List[DetectedPlayer],
        ball: DetectedBall,
        frame_number: int,
        timestamp_ms: int
    ) -> Optional[MatchEvent]:
        """
        Detect when pressing opportunity exists.

        Criteria:
        - Opposition has ball
        - Multiple teammates nearby
        - Opposition player has limited passing options
        """
        if self.possession is None:
            return None

        # Get defending team
        defending_team = (
            TeamSide.AWAY if self.possession.team == TeamSide.HOME
            else TeamSide.HOME
        )

        # Count defenders near ball
        defenders_near = 0
        pressing_distance = settings.PRESSING_DISTANCE_M

        for player in players:
            if player.team != defending_team or player.pitch_position is None:
                continue

            distance = self._calculate_distance(
                player.pitch_position,
                ball.pitch_position
            )

            if distance <= pressing_distance:
                defenders_near += 1

        # Pressing opportunity if 2+ defenders close
        if defenders_near >= 2:
            return MatchEvent(
                event_id=str(uuid.uuid4()),
                event_type=EventType.TACKLE,  # Use as pressing trigger
                timestamp_ms=timestamp_ms,
                frame_number=frame_number,
                position=ball.pitch_position,
                team=defending_team,
                metadata={
                    "event_subtype": "pressing_opportunity",
                    "defenders_nearby": defenders_near,
                    "target_player": self.possession.player_id
                }
            )

        return None

    def _calculate_distance(self, pos1: Position, pos2: Position) -> float:
        """Calculate Euclidean distance between two positions."""
        return np.sqrt(
            (pos2.x - pos1.x) ** 2 +
            (pos2.y - pos1.y) ** 2
        )

    def _get_player_position(self, player_id: int) -> Optional[Position]:
        """Get latest position for a player."""
        if player_id not in self.player_history:
            return None

        history = self.player_history[player_id]
        if not history:
            return None

        return history[-1]["position"]

    def get_possession_stats(self) -> Dict[str, float]:
        """Calculate possession statistics."""
        home_time = 0
        away_time = 0

        # Analyze event history for possession
        prev_time = 0
        prev_team = None

        for event in self.events:
            if event.team:
                if prev_team:
                    duration = event.timestamp_ms - prev_time
                    if prev_team == TeamSide.HOME:
                        home_time += duration
                    else:
                        away_time += duration

                prev_time = event.timestamp_ms
                prev_team = event.team

        total = home_time + away_time
        if total == 0:
            return {"home": 50.0, "away": 50.0}

        return {
            "home": (home_time / total) * 100,
            "away": (away_time / total) * 100
        }

    def get_pass_stats(self, team: TeamSide) -> Dict[str, int]:
        """Get passing statistics for a team."""
        passes = [e for e in self.events
                  if e.event_type == EventType.PASS and e.team == team]

        completed = len([p for p in passes if p.success])
        total = len(passes)

        return {
            "total": total,
            "completed": completed,
            "accuracy": (completed / total * 100) if total > 0 else 0
        }

    def get_events_in_range(
        self,
        start_ms: int,
        end_ms: int,
        event_type: Optional[EventType] = None
    ) -> List[MatchEvent]:
        """Get events within a time range."""
        events = [
            e for e in self.events
            if start_ms <= e.timestamp_ms <= end_ms
        ]

        if event_type:
            events = [e for e in events if e.event_type == event_type]

        return events

    def reset(self):
        """Reset all state."""
        self.possession = None
        self.prev_possession = None
        self.events.clear()
        self.event_candidates.clear()
        self.ball_history.clear()
        self.player_history.clear()

        # Reset match period
        self.current_period = 1
        self.frames_since_play = 0
        self.home_attacking_right = True

        # Reset ball state
        self.ball_state = BallState()

        # Reset statistics
        self.home_stats = TeamStatistics(team=TeamSide.HOME, period=0)
        self.away_stats = TeamStatistics(team=TeamSide.AWAY, period=0)
        self.home_stats_h1 = TeamStatistics(team=TeamSide.HOME, period=1)
        self.away_stats_h1 = TeamStatistics(team=TeamSide.AWAY, period=1)
        self.home_stats_h2 = TeamStatistics(team=TeamSide.HOME, period=2)
        self.away_stats_h2 = TeamStatistics(team=TeamSide.AWAY, period=2)

    def get_team_stats(self) -> Dict:
        """Get comprehensive team statistics."""
        def stats_to_dict(stats: TeamStatistics) -> Dict:
            return {
                "possession_pct": round(stats.possession_pct, 1),
                "possession_defensive_pct": round(
                    (stats.possession_defensive_frames / max(1, stats.possession_frames)) * 100, 1
                ) if stats.possession_frames > 0 else 0,
                "possession_middle_pct": round(
                    (stats.possession_middle_frames / max(1, stats.possession_frames)) * 100, 1
                ) if stats.possession_frames > 0 else 0,
                "possession_attacking_pct": round(
                    (stats.possession_attacking_frames / max(1, stats.possession_frames)) * 100, 1
                ) if stats.possession_frames > 0 else 0,
                "passes_total": stats.passes_total,
                "passes_successful": stats.passes_successful,
                "passes_failed": stats.passes_failed,
                "pass_accuracy_pct": round(stats.pass_accuracy_pct, 1),
                "passes_forward": stats.passes_forward,
                "passes_sideways": stats.passes_sideways,
                "passes_backward": stats.passes_backward,
                "passes_defensive_third": stats.passes_defensive_third,
                "passes_middle_third": stats.passes_middle_third,
                "passes_attacking_third": stats.passes_attacking_third,
                "longest_pass_sequence": stats.longest_pass_sequence,
                "shots_total": stats.shots_total,
                "shots_on_target": stats.shots_on_target,
                "shots_off_target": stats.shots_off_target,
                "shot_accuracy_pct": round(stats.shot_accuracy_pct, 1),
                "goals": stats.goals,
                "tackles": stats.tackles,
                "interceptions": stats.interceptions,
                "headers": stats.headers,
                "corners": stats.corners,
                "free_kicks": stats.free_kicks,
                "throw_ins": stats.throw_ins,
                "goal_kicks": stats.goal_kicks,
            }

        return {
            "home": {
                "full_match": stats_to_dict(self.home_stats),
                "first_half": stats_to_dict(self.home_stats_h1),
                "second_half": stats_to_dict(self.home_stats_h2),
            },
            "away": {
                "full_match": stats_to_dict(self.away_stats),
                "first_half": stats_to_dict(self.away_stats_h1),
                "second_half": stats_to_dict(self.away_stats_h2),
            },
            "events_count": len(self.events),
            "current_period": self.current_period,
            "team_colors_detected": self.team_colors_initialized,
            "home_team_color": self.home_team_color.tolist() if self.home_team_color is not None else None,
            "away_team_color": self.away_team_color.tolist() if self.away_team_color is not None else None,
        }

    def get_all_events(self) -> List[Dict]:
        """Get all detected events as dictionaries."""
        return [
            {
                "event_id": e.event_id,
                "event_type": e.event_type.value,
                "timestamp_ms": e.timestamp_ms,
                "frame_number": e.frame_number,
                "position": {"x": e.position.x, "y": e.position.y} if e.position else None,
                "player_id": e.player_id,
                "team": e.team.value if e.team else None,
                "recipient_id": e.recipient_id,
                "success": e.success,
                "metadata": e.metadata,
            }
            for e in self.events
        ]

    def get_player_events(self, player_id: int) -> List[Dict]:
        """Get all events involving a specific player."""
        return [
            {
                "event_id": e.event_id,
                "event_type": e.event_type.value,
                "timestamp_ms": e.timestamp_ms,
                "frame_number": e.frame_number,
                "position": {"x": e.position.x, "y": e.position.y} if e.position else None,
                "success": e.success,
                "metadata": e.metadata,
            }
            for e in self.events
            if e.player_id == player_id or e.recipient_id == player_id
        ]


# Global instance for use in main.py
event_detection_service = EventDetectionService()
