"""
Tactical Event Detection Service

Detects key tactical events:
- Pressing triggers (opportunities to press)
- Dangerous attacks
- Counter-attack opportunities
- Defensive shape warnings
- High line opportunities
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from enum import Enum
from datetime import datetime


class EventType(Enum):
    PRESSING_TRIGGER = "pressing_trigger"
    PRESSING_OPPORTUNITY = "pressing_opportunity"
    DANGEROUS_ATTACK = "dangerous_attack"
    COUNTER_ATTACK = "counter_attack"
    SHAPE_WARNING = "shape_warning"
    HIGH_LINE_OPPORTUNITY = "high_line_opportunity"
    TRANSITION_MOMENT = "transition_moment"
    SET_PIECE = "set_piece"
    OVERLOAD = "overload"  # Numerical advantage in an area


class EventPriority(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class TacticalEvent:
    """A detected tactical event."""
    event_type: EventType
    priority: EventPriority
    timestamp: float
    frame_number: int
    team_affected: str  # 'home' or 'away' - who should react
    description: str
    position: Optional[Tuple[float, float]] = None  # Where on pitch
    details: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            'type': self.event_type.value,
            'priority': self.priority.value,
            'priority_name': self.priority.name,
            'timestamp': round(self.timestamp, 2),
            'frame': self.frame_number,
            'team': self.team_affected,
            'description': self.description,
            'position': self.position,
            'details': self.details
        }


class TacticalEventDetector:
    """
    Detects tactical events from match analysis data.

    Events detected:
    1. Pressing triggers - when opponent is under pressure and can be pressed
    2. Dangerous attacks - when a team is in a threatening position
    3. Counter attacks - quick transitions after winning ball
    4. Shape warnings - when team loses defensive shape
    5. High line - when opposition defense is high and can be exploited
    """

    def __init__(self):
        self.events: List[TacticalEvent] = []

        # Pitch zones (normalized 0-100)
        self.defensive_third = (0, 33)
        self.middle_third = (33, 66)
        self.attacking_third = (66, 100)

        # Detection thresholds
        self.pressing_player_threshold = 3  # Players within pressing distance
        self.pressing_distance = 100  # pixels
        self.dangerous_zone_y = 150  # pixels from goal line
        self.high_line_threshold = 40  # % of pitch from own goal
        self.shape_gap_threshold = 200  # pixels between defensive lines
        self.overload_threshold = 2  # Player advantage to trigger overload

        # State tracking
        self.last_possession_team = None
        self.last_possession_frame = 0
        self.ball_history: List[Tuple[int, float, float]] = []

    def reset(self):
        """Reset detector for new analysis."""
        self.events = []
        self.last_possession_team = None
        self.last_possession_frame = 0
        self.ball_history = []

    def process_frame(self, frame_data: dict) -> List[TacticalEvent]:
        """
        Process a frame and detect tactical events.

        Args:
            frame_data: Frame analysis data

        Returns:
            List of detected events in this frame
        """
        frame_num = frame_data.get('frame_number', 0)
        timestamp = frame_data.get('timestamp', 0.0)
        ball_pos = frame_data.get('ball_position')
        detections = frame_data.get('detections', [])

        frame_events = []

        # Track ball history
        if ball_pos:
            self.ball_history.append((frame_num, ball_pos[0], ball_pos[1]))
            if len(self.ball_history) > 90:  # 3 seconds at 30fps
                self.ball_history.pop(0)

        # Separate players by team
        home_players = []
        away_players = []

        for det in detections:
            bbox = det.get('bbox', [0, 0, 0, 0])
            pos = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)

            if det.get('team') == 'home':
                home_players.append(pos)
            elif det.get('team') == 'away':
                away_players.append(pos)

        # Detect various events
        if ball_pos:
            # Pressing opportunities
            press_event = self._detect_pressing_trigger(
                frame_num, timestamp, ball_pos, home_players, away_players
            )
            if press_event:
                frame_events.append(press_event)

            # Dangerous attacks
            danger_event = self._detect_dangerous_attack(
                frame_num, timestamp, ball_pos, home_players, away_players
            )
            if danger_event:
                frame_events.append(danger_event)

            # Counter attack opportunities
            counter_event = self._detect_counter_opportunity(
                frame_num, timestamp, ball_pos, home_players, away_players
            )
            if counter_event:
                frame_events.append(counter_event)

        # Shape warnings
        shape_events = self._detect_shape_issues(
            frame_num, timestamp, home_players, away_players
        )
        frame_events.extend(shape_events)

        # High line opportunities
        high_line_event = self._detect_high_line(
            frame_num, timestamp, home_players, away_players, ball_pos
        )
        if high_line_event:
            frame_events.append(high_line_event)

        # Overload situations
        overload_event = self._detect_overload(
            frame_num, timestamp, ball_pos, home_players, away_players
        )
        if overload_event:
            frame_events.append(overload_event)

        self.events.extend(frame_events)
        return frame_events

    def _detect_pressing_trigger(self, frame: int, timestamp: float,
                                  ball_pos: List[float],
                                  home_players: List[Tuple],
                                  away_players: List[Tuple]) -> Optional[TacticalEvent]:
        """Detect when opponent can be pressed effectively."""

        # Find ball carrier (nearest player to ball)
        ball_x, ball_y = ball_pos[0], ball_pos[1]

        # Check both teams
        for team_name, team_players, opponent_players in [
            ('away', away_players, home_players),  # Home should press away
            ('home', home_players, away_players)   # Away should press home
        ]:
            # Find nearest player to ball (likely ball carrier)
            min_dist = float('inf')
            carrier_pos = None
            for p in team_players:
                dist = np.sqrt((ball_x - p[0])**2 + (ball_y - p[1])**2)
                if dist < min_dist:
                    min_dist = dist
                    carrier_pos = p

            if carrier_pos is None or min_dist > 100:
                continue  # No clear ball carrier

            # Count pressing players nearby
            pressing_players = sum(
                1 for p in opponent_players
                if np.sqrt((carrier_pos[0] - p[0])**2 + (carrier_pos[1] - p[1])**2) < self.pressing_distance
            )

            # Count supporting teammates
            support_players = sum(
                1 for p in team_players
                if p != carrier_pos and
                np.sqrt((carrier_pos[0] - p[0])**2 + (carrier_pos[1] - p[1])**2) < 150
            )

            # Pressing trigger: many pressing players, few supporting teammates
            if pressing_players >= self.pressing_player_threshold and support_players <= 1:
                pressing_team = 'home' if team_name == 'away' else 'away'
                return TacticalEvent(
                    event_type=EventType.PRESSING_TRIGGER,
                    priority=EventPriority.HIGH,
                    timestamp=timestamp,
                    frame_number=frame,
                    team_affected=pressing_team,
                    description=f"Pressing opportunity - {team_name} player isolated",
                    position=carrier_pos,
                    details={
                        'pressing_players': pressing_players,
                        'support_players': support_players
                    }
                )

        return None

    def _detect_dangerous_attack(self, frame: int, timestamp: float,
                                  ball_pos: List[float],
                                  home_players: List[Tuple],
                                  away_players: List[Tuple]) -> Optional[TacticalEvent]:
        """Detect dangerous attacking situations."""

        ball_x = ball_pos[0]

        # Home attacking (ball in right third)
        if ball_x > 1920 * 0.75:
            # Count home players in attacking zone
            attackers = sum(1 for p in home_players if p[0] > 1920 * 0.66)
            defenders = sum(1 for p in away_players if p[0] > 1920 * 0.66)

            if attackers >= 3 or (attackers > defenders):
                return TacticalEvent(
                    event_type=EventType.DANGEROUS_ATTACK,
                    priority=EventPriority.HIGH,
                    timestamp=timestamp,
                    frame_number=frame,
                    team_affected='away',  # Away needs to defend
                    description=f"Home dangerous attack - {attackers} attackers vs {defenders} defenders",
                    position=(ball_x, ball_pos[1]),
                    details={
                        'attackers': attackers,
                        'defenders': defenders,
                        'attacking_team': 'home'
                    }
                )

        # Away attacking (ball in left third)
        elif ball_x < 1920 * 0.25:
            attackers = sum(1 for p in away_players if p[0] < 1920 * 0.33)
            defenders = sum(1 for p in home_players if p[0] < 1920 * 0.33)

            if attackers >= 3 or (attackers > defenders):
                return TacticalEvent(
                    event_type=EventType.DANGEROUS_ATTACK,
                    priority=EventPriority.HIGH,
                    timestamp=timestamp,
                    frame_number=frame,
                    team_affected='home',
                    description=f"Away dangerous attack - {attackers} attackers vs {defenders} defenders",
                    position=(ball_x, ball_pos[1]),
                    details={
                        'attackers': attackers,
                        'defenders': defenders,
                        'attacking_team': 'away'
                    }
                )

        return None

    def _detect_counter_opportunity(self, frame: int, timestamp: float,
                                      ball_pos: List[float],
                                      home_players: List[Tuple],
                                      away_players: List[Tuple]) -> Optional[TacticalEvent]:
        """Detect counter-attack opportunities after winning ball."""

        if len(self.ball_history) < 30:  # Need 1 second of history
            return None

        # Check if ball moved significantly and quickly
        old_frame, old_x, old_y = self.ball_history[0]
        new_x, new_y = ball_pos[0], ball_pos[1]

        ball_movement = new_x - old_x

        # Home counter (ball moving right rapidly)
        if ball_movement > 200:  # Moved 200+ pixels right in 1 second
            # Check if away team is caught high
            away_in_own_half = sum(1 for p in away_players if p[0] > 960)
            if away_in_own_half >= 6:  # Most of team caught forward
                return TacticalEvent(
                    event_type=EventType.COUNTER_ATTACK,
                    priority=EventPriority.CRITICAL,
                    timestamp=timestamp,
                    frame_number=frame,
                    team_affected='home',
                    description="Counter-attack opportunity - away team caught high",
                    position=(new_x, new_y),
                    details={
                        'away_players_forward': away_in_own_half,
                        'ball_speed': abs(ball_movement)
                    }
                )

        # Away counter (ball moving left rapidly)
        elif ball_movement < -200:
            home_in_own_half = sum(1 for p in home_players if p[0] < 960)
            if home_in_own_half >= 6:
                return TacticalEvent(
                    event_type=EventType.COUNTER_ATTACK,
                    priority=EventPriority.CRITICAL,
                    timestamp=timestamp,
                    frame_number=frame,
                    team_affected='away',
                    description="Counter-attack opportunity - home team caught high",
                    position=(new_x, new_y),
                    details={
                        'home_players_forward': home_in_own_half,
                        'ball_speed': abs(ball_movement)
                    }
                )

        return None

    def _detect_shape_issues(self, frame: int, timestamp: float,
                              home_players: List[Tuple],
                              away_players: List[Tuple]) -> List[TacticalEvent]:
        """Detect when a team loses defensive shape."""
        events = []

        for team_name, players in [('home', home_players), ('away', away_players)]:
            if len(players) < 6:
                continue

            # Sort by x position
            sorted_x = sorted(p[0] for p in players)

            # Check for large gaps between players
            max_gap = 0
            for i in range(len(sorted_x) - 1):
                gap = sorted_x[i+1] - sorted_x[i]
                max_gap = max(max_gap, gap)

            if max_gap > self.shape_gap_threshold:
                events.append(TacticalEvent(
                    event_type=EventType.SHAPE_WARNING,
                    priority=EventPriority.MEDIUM,
                    timestamp=timestamp,
                    frame_number=frame,
                    team_affected=team_name,
                    description=f"{team_name.capitalize()} team shape stretched - {int(max_gap)}px gap",
                    details={'max_gap': max_gap}
                ))

        return events

    def _detect_high_line(self, frame: int, timestamp: float,
                           home_players: List[Tuple],
                           away_players: List[Tuple],
                           ball_pos: Optional[List[float]]) -> Optional[TacticalEvent]:
        """Detect when defense is playing very high line."""

        # Check away team's defensive line (sorted by x, exclude GK)
        if len(away_players) >= 4:
            sorted_x = sorted(p[0] for p in away_players)
            defensive_line = sorted_x[1]  # Second-lowest x (exclude GK)
            defensive_line_pct = (defensive_line / 1920) * 100

            if defensive_line_pct > self.high_line_threshold:
                return TacticalEvent(
                    event_type=EventType.HIGH_LINE_OPPORTUNITY,
                    priority=EventPriority.MEDIUM,
                    timestamp=timestamp,
                    frame_number=frame,
                    team_affected='home',
                    description=f"Away playing high line - space behind",
                    details={
                        'defensive_line_pct': round(defensive_line_pct, 1),
                        'target_team': 'away'
                    }
                )

        # Check home team's defensive line
        if len(home_players) >= 4:
            sorted_x = sorted((p[0] for p in home_players), reverse=True)
            defensive_line = sorted_x[1]  # Second-highest x (exclude GK)
            defensive_line_pct = ((1920 - defensive_line) / 1920) * 100

            if defensive_line_pct > self.high_line_threshold:
                return TacticalEvent(
                    event_type=EventType.HIGH_LINE_OPPORTUNITY,
                    priority=EventPriority.MEDIUM,
                    timestamp=timestamp,
                    frame_number=frame,
                    team_affected='away',
                    description=f"Home playing high line - space behind",
                    details={
                        'defensive_line_pct': round(defensive_line_pct, 1),
                        'target_team': 'home'
                    }
                )

        return None

    def _detect_overload(self, frame: int, timestamp: float,
                          ball_pos: Optional[List[float]],
                          home_players: List[Tuple],
                          away_players: List[Tuple]) -> Optional[TacticalEvent]:
        """Detect numerical overloads in key areas."""

        if not ball_pos:
            return None

        ball_x, ball_y = ball_pos[0], ball_pos[1]

        # Define zone around ball (200px radius)
        zone_radius = 200

        home_in_zone = sum(
            1 for p in home_players
            if np.sqrt((p[0] - ball_x)**2 + (p[1] - ball_y)**2) < zone_radius
        )
        away_in_zone = sum(
            1 for p in away_players
            if np.sqrt((p[0] - ball_x)**2 + (p[1] - ball_y)**2) < zone_radius
        )

        advantage = home_in_zone - away_in_zone

        if advantage >= self.overload_threshold:
            return TacticalEvent(
                event_type=EventType.OVERLOAD,
                priority=EventPriority.MEDIUM,
                timestamp=timestamp,
                frame_number=frame,
                team_affected='home',
                description=f"Home overload - {home_in_zone} vs {away_in_zone} in zone",
                position=(ball_x, ball_y),
                details={
                    'home_players': home_in_zone,
                    'away_players': away_in_zone,
                    'advantage': advantage
                }
            )
        elif advantage <= -self.overload_threshold:
            return TacticalEvent(
                event_type=EventType.OVERLOAD,
                priority=EventPriority.MEDIUM,
                timestamp=timestamp,
                frame_number=frame,
                team_affected='away',
                description=f"Away overload - {away_in_zone} vs {home_in_zone} in zone",
                position=(ball_x, ball_y),
                details={
                    'home_players': home_in_zone,
                    'away_players': away_in_zone,
                    'advantage': -advantage
                }
            )

        return None

    def get_events_summary(self) -> Dict:
        """Get summary of all detected events."""
        event_counts = {}
        for e in self.events:
            key = e.event_type.value
            event_counts[key] = event_counts.get(key, 0) + 1

        priority_counts = {
            'critical': sum(1 for e in self.events if e.priority == EventPriority.CRITICAL),
            'high': sum(1 for e in self.events if e.priority == EventPriority.HIGH),
            'medium': sum(1 for e in self.events if e.priority == EventPriority.MEDIUM),
            'low': sum(1 for e in self.events if e.priority == EventPriority.LOW)
        }

        return {
            'total_events': len(self.events),
            'event_counts': event_counts,
            'priority_counts': priority_counts,
            'events': [e.to_dict() for e in self.events[-100:]]  # Last 100 events
        }

    def analyze_from_frames(self, frame_analyses: List[dict]) -> Dict:
        """Analyze tactical events from frame data."""
        self.reset()

        # Sample every 10th frame for performance
        sampled = frame_analyses[::10] if len(frame_analyses) > 500 else frame_analyses

        for frame_data in sampled:
            self.process_frame(frame_data)

        return self.get_events_summary()


# Singleton instance
tactical_detector = TacticalEventDetector()
