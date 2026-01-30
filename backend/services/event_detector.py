"""
Automatic Event Detection Service

Detects match events from video analysis data:
- Goals (ball crossing goal line)
- Shots (ball moving toward goal at speed)
- Corners (ball out at goal line, awarded to attacking team)
- Free kicks (play stopped, ball placed)
- Goal kicks (ball out at goal line, awarded to defending team)
- Throw-ins (ball out at touchline)
- Penalties (shot from penalty spot)
- Kickoffs (ball at center, start of play)
- Offsides (player in offside position receiving ball)

Similar to VEO's automatic event tagging.
"""
import numpy as np
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import time


class DetectedEventType(Enum):
    """Types of events that can be auto-detected."""
    GOAL = "goal"
    SHOT = "shot"
    SHOT_ON_TARGET = "shot_on_target"
    SHOT_OFF_TARGET = "shot_off_target"
    SHOT_BLOCKED = "shot_blocked"
    CORNER = "corner"
    FREE_KICK = "free_kick"
    GOAL_KICK = "goal_kick"
    THROW_IN = "throw_in"
    PENALTY = "penalty"
    KICKOFF = "kickoff"
    HALF_TIME = "half_time"
    FULL_TIME = "full_time"
    OFFSIDE = "offside"
    PASS = "pass"
    LONG_BALL = "long_ball"
    CROSS = "cross"
    TACKLE = "tackle"
    INTERCEPTION = "interception"
    CLEARANCE = "clearance"
    HEADER = "header"
    DRIBBLE = "dribble"
    FOUL = "foul"
    POSSESSION_CHANGE = "possession_change"


@dataclass
class DetectedEvent:
    """An automatically detected match event."""
    event_type: DetectedEventType
    timestamp_ms: int
    frame_number: int
    team: Optional[str] = None
    player_jersey: Optional[int] = None
    position_x: Optional[float] = None  # Pitch coordinates (0-100)
    position_y: Optional[float] = None
    end_position_x: Optional[float] = None
    end_position_y: Optional[float] = None
    confidence: float = 0.8
    description: str = ""
    related_events: List[str] = field(default_factory=list)


@dataclass
class BallState:
    """Ball state for tracking."""
    x: float
    y: float
    vx: float = 0.0  # Velocity x
    vy: float = 0.0  # Velocity y
    speed: float = 0.0
    frame: int = 0
    in_play: bool = True
    possessing_team: Optional[str] = None
    possessing_player: Optional[int] = None


class EventDetector:
    """
    Automatic event detection from match analysis data.

    Uses ball position, player positions, and motion to detect
    key match events in real-time.
    """

    # Pitch dimensions (normalized 0-100)
    PITCH_LENGTH = 100.0
    PITCH_WIDTH = 100.0

    # Goal dimensions (scaled to pitch coordinates)
    GOAL_LINE_HOME = 0.0  # Home team defends x=0
    GOAL_LINE_AWAY = 100.0  # Away team defends x=100
    GOAL_Y_MIN = 45.0  # Goal posts (centered)
    GOAL_Y_MAX = 55.0

    # Detection thresholds
    SHOT_SPEED_THRESHOLD = 15.0  # Minimum ball speed for shot detection
    PASS_SPEED_THRESHOLD = 5.0  # Minimum speed for pass
    LONG_BALL_DISTANCE = 30.0  # Distance for long ball classification
    CROSS_X_THRESHOLD = 75.0  # Minimum x position for cross
    HEADER_HEIGHT_THRESHOLD = 1.8  # Meters (if height available)

    # Zone definitions
    PENALTY_AREA_X = 16.5  # 16.5m from goal line
    PENALTY_SPOT_X = 11.0  # 11m from goal
    SIX_YARD_BOX_X = 5.5

    # Timing thresholds
    POSSESSION_CHANGE_FRAMES = 10  # Frames before confirming possession change
    EVENT_COOLDOWN_FRAMES = 30  # Minimum frames between similar events

    def __init__(self):
        self.events: List[DetectedEvent] = []
        self.ball_history: deque = deque(maxlen=60)  # 2 seconds at 30fps

        # Current state
        self.current_ball_state: Optional[BallState] = None
        self.current_possession_team: Optional[str] = None
        self.possession_frame_count: int = 0

        # Event cooldowns (prevent duplicate detection)
        self.last_event_frame: Dict[DetectedEventType, int] = {}

        # Match state
        self.match_started: bool = False
        self.first_half: bool = True
        self.home_score: int = 0
        self.away_score: int = 0

        # Frame tracking
        self.current_frame: int = 0
        self.fps: float = 30.0

    def set_fps(self, fps: float):
        """Set video FPS for timing calculations."""
        self.fps = fps

    def set_video_info(self, fps: float, total_frames: int, frame_width: int = 1920, frame_height: int = 1080):
        """Set video metadata for event detection."""
        self.fps = fps
        self.total_frames = total_frames
        self.frame_width = frame_width
        self.frame_height = frame_height

    def _can_detect_event(self, event_type: DetectedEventType) -> bool:
        """Check if enough time has passed since last event of this type."""
        if event_type not in self.last_event_frame:
            return True
        return (self.current_frame - self.last_event_frame[event_type]) > self.EVENT_COOLDOWN_FRAMES

    def _record_event(self, event: DetectedEvent):
        """Record a detected event."""
        self.events.append(event)
        self.last_event_frame[event.event_type] = event.frame_number

    def process_frame(
        self,
        frame_number: int,
        timestamp_ms: int,
        ball_position: Optional[Tuple[float, float]] = None,
        player_positions: List[Dict] = None,  # [{team, jersey, x, y}, ...]
        ball_possessing_player: Optional[Tuple[str, int]] = None,  # (team, jersey)
        # Simplified parameters for direct x/y input
        ball_x: Optional[float] = None,
        ball_y: Optional[float] = None,
        possessing_team: Optional[str] = None
    ) -> List[DetectedEvent]:
        """
        Process a frame and detect events.

        Args:
            frame_number: Current frame number
            timestamp_ms: Current timestamp in milliseconds
            ball_position: Ball (x, y) in pitch coordinates (0-100)
            player_positions: List of player positions
            ball_possessing_player: (team, jersey) of player with ball
            ball_x, ball_y: Alternative way to pass ball position
            possessing_team: Team currently possessing the ball

        Returns:
            List of events detected in this frame
        """
        self.current_frame = frame_number
        detected_events = []

        # Handle simplified x/y parameters
        if ball_position is None and ball_x is not None and ball_y is not None:
            ball_position = (ball_x, ball_y)

        if ball_possessing_player is None and possessing_team is not None:
            ball_possessing_player = (possessing_team, None)

        if player_positions is None:
            player_positions = []

        # Update ball state
        if ball_position:
            self._update_ball_state(ball_position, frame_number, ball_possessing_player)

        # Check for match start
        if not self.match_started and ball_position:
            if self._is_kickoff_position(ball_position):
                self.match_started = True
                event = DetectedEvent(
                    event_type=DetectedEventType.KICKOFF,
                    timestamp_ms=timestamp_ms,
                    frame_number=frame_number,
                    position_x=ball_position[0],
                    position_y=ball_position[1],
                    confidence=0.9,
                    description="Match started"
                )
                self._record_event(event)
                detected_events.append(event)

        # Detect events based on ball movement and position
        if self.current_ball_state and len(self.ball_history) >= 2:
            # Detect shots
            shot_event = self._detect_shot(timestamp_ms)
            if shot_event:
                detected_events.append(shot_event)

            # Detect goals
            goal_event = self._detect_goal(timestamp_ms)
            if goal_event:
                detected_events.append(goal_event)

            # Detect ball out of play (corners, goal kicks, throw-ins)
            out_event = self._detect_ball_out(timestamp_ms)
            if out_event:
                detected_events.append(out_event)

            # Detect passes
            pass_event = self._detect_pass(timestamp_ms, player_positions)
            if pass_event:
                detected_events.append(pass_event)

            # Detect possession changes
            poss_event = self._detect_possession_change(timestamp_ms, ball_possessing_player)
            if poss_event:
                detected_events.append(poss_event)

        return detected_events

    def _update_ball_state(
        self,
        position: Tuple[float, float],
        frame: int,
        possessing_player: Optional[Tuple[str, int]]
    ):
        """Update ball state with new position."""
        x, y = position

        if self.current_ball_state:
            # Calculate velocity
            dt = 1.0 / self.fps
            dx = x - self.current_ball_state.x
            dy = y - self.current_ball_state.y
            vx = dx / dt
            vy = dy / dt
            speed = np.sqrt(vx**2 + vy**2)

            self.current_ball_state = BallState(
                x=x, y=y,
                vx=vx, vy=vy,
                speed=speed,
                frame=frame,
                in_play=self._is_in_play(x, y),
                possessing_team=possessing_player[0] if possessing_player else None,
                possessing_player=possessing_player[1] if possessing_player else None
            )
        else:
            self.current_ball_state = BallState(
                x=x, y=y,
                frame=frame,
                in_play=self._is_in_play(x, y),
                possessing_team=possessing_player[0] if possessing_player else None,
                possessing_player=possessing_player[1] if possessing_player else None
            )

        self.ball_history.append(self.current_ball_state)

    def _is_in_play(self, x: float, y: float) -> bool:
        """Check if ball position is within the pitch."""
        return 0 <= x <= 100 and 0 <= y <= 100

    def _is_kickoff_position(self, position: Tuple[float, float]) -> bool:
        """Check if ball is at kickoff position (center of pitch)."""
        x, y = position
        return 48 <= x <= 52 and 48 <= y <= 52

    def _detect_shot(self, timestamp_ms: int) -> Optional[DetectedEvent]:
        """Detect a shot on goal."""
        if not self._can_detect_event(DetectedEventType.SHOT):
            return None

        state = self.current_ball_state
        if not state or state.speed < self.SHOT_SPEED_THRESHOLD:
            return None

        # Check if ball is moving toward a goal
        attacking_goal = None
        if state.possessing_team == "home" and state.vx > 0:
            attacking_goal = "away"  # Home attacks away goal (x=100)
        elif state.possessing_team == "away" and state.vx < 0:
            attacking_goal = "home"  # Away attacks home goal (x=0)

        if not attacking_goal:
            return None

        # Check if in shooting position (attacking third)
        in_shooting_position = False
        if attacking_goal == "away" and state.x > 66:
            in_shooting_position = True
        elif attacking_goal == "home" and state.x < 34:
            in_shooting_position = True

        if not in_shooting_position:
            return None

        # Determine if on target
        target_y = state.y + (state.vy / abs(state.vx) if state.vx != 0 else 0) * (100 - state.x if attacking_goal == "away" else state.x)

        if self.GOAL_Y_MIN <= target_y <= self.GOAL_Y_MAX:
            event_type = DetectedEventType.SHOT_ON_TARGET
            description = "Shot on target"
        else:
            event_type = DetectedEventType.SHOT_OFF_TARGET
            description = "Shot off target"

        event = DetectedEvent(
            event_type=event_type,
            timestamp_ms=timestamp_ms,
            frame_number=self.current_frame,
            team=state.possessing_team,
            player_jersey=state.possessing_player,
            position_x=state.x,
            position_y=state.y,
            confidence=0.7,
            description=description
        )

        self._record_event(event)
        return event

    def _detect_goal(self, timestamp_ms: int) -> Optional[DetectedEvent]:
        """Detect a goal (ball crossing goal line within goal posts)."""
        if not self._can_detect_event(DetectedEventType.GOAL):
            return None

        state = self.current_ball_state
        if not state:
            return None

        # Check previous position
        if len(self.ball_history) < 2:
            return None

        prev_state = self.ball_history[-2]

        # Check if ball crossed goal line
        goal_scored = False
        scoring_team = None

        # Away goal (x=100)
        if prev_state.x < 100 and state.x >= 100:
            if self.GOAL_Y_MIN <= state.y <= self.GOAL_Y_MAX:
                goal_scored = True
                scoring_team = "home"
                self.home_score += 1

        # Home goal (x=0)
        elif prev_state.x > 0 and state.x <= 0:
            if self.GOAL_Y_MIN <= state.y <= self.GOAL_Y_MAX:
                goal_scored = True
                scoring_team = "away"
                self.away_score += 1

        if goal_scored:
            event = DetectedEvent(
                event_type=DetectedEventType.GOAL,
                timestamp_ms=timestamp_ms,
                frame_number=self.current_frame,
                team=scoring_team,
                player_jersey=prev_state.possessing_player if prev_state.possessing_team == scoring_team else None,
                position_x=prev_state.x,
                position_y=prev_state.y,
                confidence=0.95,
                description=f"GOAL! {self.home_score}-{self.away_score}"
            )
            self._record_event(event)
            return event

        return None

    def _detect_ball_out(self, timestamp_ms: int) -> Optional[DetectedEvent]:
        """Detect ball out of play (corner, goal kick, throw-in)."""
        state = self.current_ball_state
        if not state:
            return None

        if state.in_play:
            return None

        # Determine type of restart
        event_type = None
        description = ""
        team = None

        # Ball out at goal line
        if state.x <= 0 or state.x >= 100:
            # Determine which team touched it last
            last_touch_team = state.possessing_team

            if state.x >= 100:  # Away goal line
                if last_touch_team == "away":
                    event_type = DetectedEventType.CORNER
                    team = "home"
                    description = "Corner to home"
                else:
                    event_type = DetectedEventType.GOAL_KICK
                    team = "away"
                    description = "Goal kick to away"
            else:  # Home goal line
                if last_touch_team == "home":
                    event_type = DetectedEventType.CORNER
                    team = "away"
                    description = "Corner to away"
                else:
                    event_type = DetectedEventType.GOAL_KICK
                    team = "home"
                    description = "Goal kick to home"

        # Ball out at touchline
        elif state.y <= 0 or state.y >= 100:
            event_type = DetectedEventType.THROW_IN
            # Award to team that didn't touch last
            team = "away" if state.possessing_team == "home" else "home"
            description = f"Throw-in to {team}"

        if event_type and self._can_detect_event(event_type):
            event = DetectedEvent(
                event_type=event_type,
                timestamp_ms=timestamp_ms,
                frame_number=self.current_frame,
                team=team,
                position_x=state.x,
                position_y=state.y,
                confidence=0.75,
                description=description
            )
            self._record_event(event)
            return event

        return None

    def _detect_pass(
        self,
        timestamp_ms: int,
        player_positions: List[Dict]
    ) -> Optional[DetectedEvent]:
        """Detect a pass between players."""
        if not self._can_detect_event(DetectedEventType.PASS):
            return None

        state = self.current_ball_state
        if not state or state.speed < self.PASS_SPEED_THRESHOLD:
            return None

        # Need at least some history
        if len(self.ball_history) < 5:
            return None

        # Check for possession change between teammates
        old_state = self.ball_history[-5]

        if (old_state.possessing_player and state.possessing_player and
            old_state.possessing_team == state.possessing_team and
            old_state.possessing_player != state.possessing_player):

            # Calculate pass distance
            dx = state.x - old_state.x
            dy = state.y - old_state.y
            distance = np.sqrt(dx**2 + dy**2)

            # Determine pass type
            if distance > self.LONG_BALL_DISTANCE:
                event_type = DetectedEventType.LONG_BALL
                description = "Long ball"
            elif state.x > self.CROSS_X_THRESHOLD and abs(dy) > 20:
                event_type = DetectedEventType.CROSS
                description = "Cross"
            else:
                event_type = DetectedEventType.PASS
                description = "Pass"

            event = DetectedEvent(
                event_type=event_type,
                timestamp_ms=timestamp_ms,
                frame_number=self.current_frame,
                team=state.possessing_team,
                player_jersey=old_state.possessing_player,
                position_x=old_state.x,
                position_y=old_state.y,
                end_position_x=state.x,
                end_position_y=state.y,
                confidence=0.6,
                description=description
            )
            self._record_event(event)
            return event

        return None

    def _detect_possession_change(
        self,
        timestamp_ms: int,
        ball_possessing_player: Optional[Tuple[str, int]]
    ) -> Optional[DetectedEvent]:
        """Detect change of possession between teams."""
        if not ball_possessing_player:
            return None

        new_team = ball_possessing_player[0]

        if self.current_possession_team is None:
            self.current_possession_team = new_team
            self.possession_frame_count = 1
            return None

        if new_team != self.current_possession_team:
            self.possession_frame_count += 1

            if self.possession_frame_count >= self.POSSESSION_CHANGE_FRAMES:
                if self._can_detect_event(DetectedEventType.POSSESSION_CHANGE):
                    state = self.current_ball_state

                    # Determine how possession was won
                    description = f"Possession won by {new_team}"

                    event = DetectedEvent(
                        event_type=DetectedEventType.POSSESSION_CHANGE,
                        timestamp_ms=timestamp_ms,
                        frame_number=self.current_frame,
                        team=new_team,
                        player_jersey=ball_possessing_player[1],
                        position_x=state.x if state else None,
                        position_y=state.y if state else None,
                        confidence=0.7,
                        description=description
                    )

                    self.current_possession_team = new_team
                    self.possession_frame_count = 1

                    self._record_event(event)
                    return event
        else:
            self.possession_frame_count = 1

        return None

    def get_events(self) -> List[DetectedEvent]:
        """Get all detected events."""
        return self.events

    def get_events_by_type(self, event_type: DetectedEventType) -> List[DetectedEvent]:
        """Get all events of a specific type."""
        return [e for e in self.events if e.event_type == event_type]

    def get_events_in_range(
        self,
        start_frame: int,
        end_frame: int
    ) -> List[DetectedEvent]:
        """Get events within a frame range."""
        return [
            e for e in self.events
            if start_frame <= e.frame_number <= end_frame
        ]

    def get_events_timeline(self) -> List[Dict]:
        """Get all events as a timeline for display."""
        return [
            {
                "type": e.event_type.value,
                "timestamp_ms": e.timestamp_ms,
                "time_str": f"{e.timestamp_ms // 60000}:{(e.timestamp_ms // 1000) % 60:02d}",
                "frame": e.frame_number,
                "team": e.team,
                "player": e.player_jersey,
                "position": {"x": e.position_x, "y": e.position_y} if e.position_x else None,
                "confidence": e.confidence,
                "description": e.description
            }
            for e in sorted(self.events, key=lambda x: x.timestamp_ms)
        ]

    def get_score(self) -> Dict:
        """Get current score."""
        return {
            "home": self.home_score,
            "away": self.away_score
        }

    def get_event_counts(self) -> Dict:
        """Get counts of each event type."""
        counts = {}
        for event in self.events:
            key = event.event_type.value
            counts[key] = counts.get(key, 0) + 1
        return counts

    def reset(self):
        """Reset detector state."""
        self.events.clear()
        self.ball_history.clear()
        self.current_ball_state = None
        self.current_possession_team = None
        self.possession_frame_count = 0
        self.last_event_frame.clear()
        self.match_started = False
        self.first_half = True
        self.home_score = 0
        self.away_score = 0
        self.current_frame = 0


# Global instance
event_detector = EventDetector()
