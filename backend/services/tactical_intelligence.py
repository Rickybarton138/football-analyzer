"""
Tactical Intelligence Service

Advanced real-time tactical analysis providing:
- Formation analysis and recommendations
- Defensive vulnerability detection
- Pressing effectiveness metrics
- Space exploitation opportunities
- Transition analysis (attack to defense, defense to attack)
- Set piece positioning analysis
- Player marking assignments

Similar to professional analysis tools used by elite clubs.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import math


class TacticalAlertType(Enum):
    """Types of real-time tactical alerts."""
    # Attacking opportunities
    SPACE_BEHIND_DEFENSE = "space_behind_defense"
    OVERLOAD_OPPORTUNITY = "overload_opportunity"
    SWITCH_PLAY = "switch_play"
    THROUGH_BALL_LANE = "through_ball_lane"
    THIRD_MAN_RUN = "third_man_run"
    CUTBACK_ZONE = "cutback_zone"

    # Defensive warnings
    DEFENSIVE_GAP = "defensive_gap"
    UNMARKED_RUNNER = "unmarked_runner"
    HIGH_LINE_VULNERABLE = "high_line_vulnerable"
    TRANSITION_DANGER = "transition_danger"
    WIDE_AREA_EXPOSED = "wide_area_exposed"
    PRESSING_TRAP = "pressing_trap"

    # Pressing
    PRESS_TRIGGER = "press_trigger"
    PRESS_RELEASE = "press_release"
    COUNTER_PRESS = "counter_press"

    # General
    FORMATION_SHIFT = "formation_shift"
    MOMENTUM_SHIFT = "momentum_shift"


class AlertPriority(Enum):
    """Alert priority levels."""
    CRITICAL = 4  # Immediate goalscoring/conceding opportunity
    HIGH = 3      # Significant tactical moment
    MEDIUM = 2    # Useful information
    LOW = 1       # General observation


@dataclass
class TacticalAlert:
    """A real-time tactical alert."""
    alert_type: TacticalAlertType
    priority: AlertPriority
    team: str  # Which team should act
    message: str
    recommendation: str
    frame_number: int
    timestamp_ms: int
    position: Optional[Tuple[float, float]] = None
    involved_players: List[int] = field(default_factory=list)
    expires_after_frames: int = 90  # Alert expires after 3 seconds

    def to_dict(self) -> Dict:
        return {
            "type": self.alert_type.value,
            "priority": self.priority.value,
            "priority_name": self.priority.name,
            "team": self.team,
            "message": self.message,
            "recommendation": self.recommendation,
            "frame": self.frame_number,
            "timestamp_ms": self.timestamp_ms,
            "time_str": f"{self.timestamp_ms // 60000}:{(self.timestamp_ms // 1000) % 60:02d}",
            "position": self.position,
            "involved_players": self.involved_players,
            "expires_at_frame": self.frame_number + self.expires_after_frames
        }


@dataclass
class DefensiveLine:
    """Analysis of a team's defensive line."""
    average_y: float  # Average position (0-100)
    spread: float     # How spread out
    highest_player_y: float
    lowest_player_y: float
    is_flat: bool     # Within 5 units
    players: List[int]


@dataclass
class PassingLane:
    """A detected passing lane."""
    start_pos: Tuple[float, float]
    end_pos: Tuple[float, float]
    distance: float
    angle: float
    is_blocked: bool
    blocking_players: List[int]
    danger_level: float  # 0-1, how dangerous this lane is


class TacticalIntelligenceService:
    """
    Real-time tactical intelligence and analysis.

    Provides coaching-quality insights during live match analysis.
    """

    # Pitch zones (normalized 0-100 coordinates)
    DEFENSIVE_THIRD = (0, 33.3)
    MIDDLE_THIRD = (33.3, 66.6)
    ATTACKING_THIRD = (66.6, 100)

    # Detection thresholds
    SPACE_THRESHOLD = 15.0  # Minimum space to be considered "open"
    PRESS_DISTANCE = 10.0   # Distance to consider "pressing"
    GAP_THRESHOLD = 20.0    # Gap in defensive line
    OFFSIDE_LINE_MARGIN = 2.0

    def __init__(self):
        self.alerts: List[TacticalAlert] = []
        self.active_alerts: List[TacticalAlert] = []
        self.dismissed_alerts: Set[int] = set()

        # State tracking
        self.current_frame = 0
        self.current_timestamp = 0
        self.possession_team: Optional[str] = None

        # Historical data for trend analysis
        self.defensive_line_history: Dict[str, List[float]] = {"home": [], "away": []}
        self.possession_history: List[Tuple[int, str]] = []  # (frame, team)
        self.pressure_events: List[Dict] = []

        # Cooldowns to prevent alert spam
        self.alert_cooldowns: Dict[TacticalAlertType, int] = {}
        self.COOLDOWN_FRAMES = 60  # 2 seconds at 30fps

    def reset(self):
        """Reset for new match."""
        self.alerts.clear()
        self.active_alerts.clear()
        self.dismissed_alerts.clear()
        self.current_frame = 0
        self.current_timestamp = 0
        self.possession_team = None
        self.defensive_line_history = {"home": [], "away": []}
        self.possession_history.clear()
        self.pressure_events.clear()
        self.alert_cooldowns.clear()

    def _can_create_alert(self, alert_type: TacticalAlertType) -> bool:
        """Check if alert can be created (respecting cooldown)."""
        if alert_type in self.alert_cooldowns:
            return self.current_frame > self.alert_cooldowns[alert_type]
        return True

    def _register_alert(self, alert: TacticalAlert):
        """Register a new alert."""
        self.alerts.append(alert)
        self.active_alerts.append(alert)
        self.alert_cooldowns[alert.alert_type] = self.current_frame + self.COOLDOWN_FRAMES

    def process_frame(
        self,
        frame_number: int,
        timestamp_ms: int,
        home_players: List[Dict],  # [{x, y, jersey_number, has_ball}, ...]
        away_players: List[Dict],
        ball_position: Optional[Tuple[float, float]] = None,
        possession_team: Optional[str] = None
    ) -> List[TacticalAlert]:
        """
        Process a frame and generate tactical alerts.

        Args:
            frame_number: Current frame number
            timestamp_ms: Timestamp in milliseconds
            home_players: List of home player positions
            away_players: List of away player positions
            ball_position: Ball (x, y) in pitch coordinates
            possession_team: Team currently with possession

        Returns:
            List of new tactical alerts
        """
        self.current_frame = frame_number
        self.current_timestamp = timestamp_ms
        self.possession_team = possession_team

        # Track possession history
        if possession_team:
            self.possession_history.append((frame_number, possession_team))
            if len(self.possession_history) > 300:
                self.possession_history = self.possession_history[-300:]

        # Clean up expired alerts
        self.active_alerts = [
            a for a in self.active_alerts
            if frame_number < a.frame_number + a.expires_after_frames
        ]

        new_alerts = []

        # Analyze defensive lines
        home_def_line = self._analyze_defensive_line(home_players, "home")
        away_def_line = self._analyze_defensive_line(away_players, "away")

        # Store defensive line history
        if home_def_line:
            self.defensive_line_history["home"].append(home_def_line.average_y)
        if away_def_line:
            self.defensive_line_history["away"].append(away_def_line.average_y)

        # Limit history length
        for team in ["home", "away"]:
            if len(self.defensive_line_history[team]) > 300:
                self.defensive_line_history[team] = self.defensive_line_history[team][-300:]

        # Run all detection algorithms
        alerts = []

        # Attacking opportunities
        if possession_team == "home":
            alerts.extend(self._detect_space_behind(away_def_line, home_players, ball_position, "home"))
            alerts.extend(self._detect_through_ball_lanes(home_players, away_players, ball_position, "home"))
            alerts.extend(self._detect_overload(home_players, away_players, ball_position, "home"))
        elif possession_team == "away":
            alerts.extend(self._detect_space_behind(home_def_line, away_players, ball_position, "away"))
            alerts.extend(self._detect_through_ball_lanes(away_players, home_players, ball_position, "away"))
            alerts.extend(self._detect_overload(away_players, home_players, ball_position, "away"))

        # Defensive warnings
        if possession_team == "away":
            alerts.extend(self._detect_defensive_gaps(home_players, away_players, ball_position, "home"))
            alerts.extend(self._detect_unmarked_runners(home_players, away_players, "home"))
        elif possession_team == "home":
            alerts.extend(self._detect_defensive_gaps(away_players, home_players, ball_position, "away"))
            alerts.extend(self._detect_unmarked_runners(away_players, home_players, "away"))

        # Pressing analysis
        alerts.extend(self._detect_pressing_opportunities(home_players, away_players, ball_position))

        # Transition detection
        alerts.extend(self._detect_transition_moments())

        # Register valid alerts
        for alert in alerts:
            if self._can_create_alert(alert.alert_type):
                self._register_alert(alert)
                new_alerts.append(alert)

        return new_alerts

    def _analyze_defensive_line(
        self,
        players: List[Dict],
        team: str
    ) -> Optional[DefensiveLine]:
        """Analyze the defensive line of a team."""
        # Get outfield players (exclude goalkeeper who would be at extreme position)
        outfield = [p for p in players if p.get('jersey_number', 0) != 1]

        if len(outfield) < 3:
            return None

        # Sort by x position and take the back 4-5 players
        if team == "home":
            # Home defends left (low x), so defenders have lowest x
            sorted_players = sorted(outfield, key=lambda p: p.get('x', 50))
        else:
            # Away defends right (high x), so defenders have highest x
            sorted_players = sorted(outfield, key=lambda p: -p.get('x', 50))

        # Take back 4 players
        defenders = sorted_players[:4]

        x_positions = [p.get('x', 50) for p in defenders]
        avg_x = np.mean(x_positions)
        spread = max(x_positions) - min(x_positions)

        return DefensiveLine(
            average_y=avg_x,  # Using x as the "line" position
            spread=spread,
            highest_player_y=max(x_positions),
            lowest_player_y=min(x_positions),
            is_flat=spread < 5,
            players=[p.get('jersey_number', 0) for p in defenders]
        )

    def _detect_space_behind(
        self,
        def_line: Optional[DefensiveLine],
        attacking_players: List[Dict],
        ball_pos: Optional[Tuple[float, float]],
        attacking_team: str
    ) -> List[TacticalAlert]:
        """Detect space behind the defensive line."""
        if not def_line or not ball_pos:
            return []

        alerts = []

        # Check if defensive line is high
        if attacking_team == "home":
            # Away defense is high if their line is closer to their own goal (high x)
            is_high_line = def_line.average_y > 60
            space_zone_x = def_line.average_y + 10
        else:
            # Home defense is high if their line is closer to away's attacking area (low x)
            is_high_line = def_line.average_y < 40
            space_zone_x = def_line.average_y - 10

        if is_high_line:
            # Check if attackers can exploit
            fast_attackers = [
                p for p in attacking_players
                if (attacking_team == "home" and p.get('x', 0) > 60) or
                   (attacking_team == "away" and p.get('x', 0) < 40)
            ]

            if fast_attackers:
                alert = TacticalAlert(
                    alert_type=TacticalAlertType.SPACE_BEHIND_DEFENSE,
                    priority=AlertPriority.HIGH,
                    team=attacking_team,
                    message=f"Space behind the defense! High line at {def_line.average_y:.0f}%",
                    recommendation="Play ball in behind or make runs into the channel. "
                                  "Timing is key - watch the offside line.",
                    frame_number=self.current_frame,
                    timestamp_ms=self.current_timestamp,
                    position=(space_zone_x, 50),
                    involved_players=[p.get('jersey_number', 0) for p in fast_attackers[:2]]
                )
                alerts.append(alert)

        return alerts

    def _detect_through_ball_lanes(
        self,
        attacking_players: List[Dict],
        defending_players: List[Dict],
        ball_pos: Optional[Tuple[float, float]],
        attacking_team: str
    ) -> List[TacticalAlert]:
        """Detect open through-ball lanes."""
        if not ball_pos:
            return []

        alerts = []

        # Find ball carrier (closest attacker to ball)
        ball_x, ball_y = ball_pos
        min_dist = float('inf')
        carrier = None

        for p in attacking_players:
            px, py = p.get('x', 0), p.get('y', 0)
            dist = math.sqrt((px - ball_x)**2 + (py - ball_y)**2)
            if dist < min_dist:
                min_dist = dist
                carrier = p

        if not carrier or min_dist > 15:
            return []

        # Find runners ahead of the ball
        if attacking_team == "home":
            runners = [p for p in attacking_players if p.get('x', 0) > ball_x + 10]
        else:
            runners = [p for p in attacking_players if p.get('x', 0) < ball_x - 10]

        for runner in runners:
            runner_x, runner_y = runner.get('x', 0), runner.get('y', 0)

            # Check if lane to runner is clear
            lane_blocked = False
            for defender in defending_players:
                def_x, def_y = defender.get('x', 0), defender.get('y', 0)

                # Check if defender is in the lane
                if self._point_in_lane(def_x, def_y, ball_x, ball_y, runner_x, runner_y, width=8):
                    lane_blocked = True
                    break

            if not lane_blocked:
                alert = TacticalAlert(
                    alert_type=TacticalAlertType.THROUGH_BALL_LANE,
                    priority=AlertPriority.MEDIUM,
                    team=attacking_team,
                    message=f"Through ball lane open to #{runner.get('jersey_number', '?')}",
                    recommendation="Weight the pass perfectly - play it into space, not to feet.",
                    frame_number=self.current_frame,
                    timestamp_ms=self.current_timestamp,
                    position=(runner_x, runner_y),
                    involved_players=[
                        carrier.get('jersey_number', 0),
                        runner.get('jersey_number', 0)
                    ]
                )
                alerts.append(alert)

        return alerts[:1]  # Only return best opportunity

    def _point_in_lane(
        self,
        px: float,
        py: float,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        width: float
    ) -> bool:
        """Check if point (px, py) is within a lane from (x1,y1) to (x2,y2)."""
        # Line length
        length = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        if length == 0:
            return False

        # Distance from point to line
        dist = abs((y2 - y1) * px - (x2 - x1) * py + x2 * y1 - y2 * x1) / length

        # Check if point is alongside the lane (not before start or after end)
        dot = ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / (length ** 2)

        return dist < width and 0.1 < dot < 0.9

    def _detect_overload(
        self,
        attacking_players: List[Dict],
        defending_players: List[Dict],
        ball_pos: Optional[Tuple[float, float]],
        attacking_team: str
    ) -> List[TacticalAlert]:
        """Detect numerical overloads in key areas."""
        if not ball_pos:
            return []

        alerts = []
        ball_x, ball_y = ball_pos

        # Define zones to check for overloads
        zones = [
            ("left_channel", 0, 35),
            ("central", 35, 65),
            ("right_channel", 65, 100)
        ]

        for zone_name, y_min, y_max in zones:
            # Count players in zone
            attackers_in_zone = sum(
                1 for p in attacking_players
                if y_min <= p.get('y', 50) <= y_max
            )
            defenders_in_zone = sum(
                1 for p in defending_players
                if y_min <= p.get('y', 50) <= y_max
            )

            advantage = attackers_in_zone - defenders_in_zone

            if advantage >= 2:
                alert = TacticalAlert(
                    alert_type=TacticalAlertType.OVERLOAD_OPPORTUNITY,
                    priority=AlertPriority.MEDIUM,
                    team=attacking_team,
                    message=f"Overload in {zone_name.replace('_', ' ')}! {attackers_in_zone}v{defenders_in_zone}",
                    recommendation=f"Exploit the {zone_name.replace('_', ' ')} - quick combination play or overlap.",
                    frame_number=self.current_frame,
                    timestamp_ms=self.current_timestamp,
                    position=(ball_x, (y_min + y_max) / 2)
                )
                alerts.append(alert)

        return alerts[:1]

    def _detect_defensive_gaps(
        self,
        defending_players: List[Dict],
        attacking_players: List[Dict],
        ball_pos: Optional[Tuple[float, float]],
        defending_team: str
    ) -> List[TacticalAlert]:
        """Detect gaps in the defensive structure."""
        if len(defending_players) < 4:
            return []

        alerts = []

        # Sort defenders by y position
        sorted_defenders = sorted(defending_players, key=lambda p: p.get('y', 50))

        # Check gaps between adjacent defenders
        for i in range(len(sorted_defenders) - 1):
            p1 = sorted_defenders[i]
            p2 = sorted_defenders[i + 1]

            gap = abs(p2.get('y', 50) - p1.get('y', 50))

            if gap > self.GAP_THRESHOLD:
                gap_y = (p1.get('y', 50) + p2.get('y', 50)) / 2

                # Check if attacker is exploiting or near the gap
                exploiting_attacker = None
                for attacker in attacking_players:
                    if abs(attacker.get('y', 0) - gap_y) < 10:
                        exploiting_attacker = attacker
                        break

                if exploiting_attacker:
                    alert = TacticalAlert(
                        alert_type=TacticalAlertType.DEFENSIVE_GAP,
                        priority=AlertPriority.HIGH,
                        team=defending_team,
                        message=f"Dangerous gap between #{p1.get('jersey_number', '?')} and #{p2.get('jersey_number', '?')}!",
                        recommendation="Close the gap immediately! One player step across, maintain communication.",
                        frame_number=self.current_frame,
                        timestamp_ms=self.current_timestamp,
                        position=((p1.get('x', 50) + p2.get('x', 50)) / 2, gap_y),
                        involved_players=[
                            p1.get('jersey_number', 0),
                            p2.get('jersey_number', 0)
                        ]
                    )
                    alerts.append(alert)

        return alerts[:1]

    def _detect_unmarked_runners(
        self,
        defending_players: List[Dict],
        attacking_players: List[Dict],
        defending_team: str
    ) -> List[TacticalAlert]:
        """Detect unmarked attacking players in dangerous positions."""
        alerts = []
        marking_distance = 12  # Distance to consider "marked"

        for attacker in attacking_players:
            ax, ay = attacker.get('x', 50), attacker.get('y', 50)

            # Only check attackers in dangerous positions
            if defending_team == "home" and ax < 35:  # Attacker in home's third
                is_dangerous = True
            elif defending_team == "away" and ax > 65:  # Attacker in away's third
                is_dangerous = True
            else:
                is_dangerous = False

            if not is_dangerous:
                continue

            # Check if any defender is marking this attacker
            is_marked = False
            for defender in defending_players:
                dx, dy = defender.get('x', 50), defender.get('y', 50)
                dist = math.sqrt((ax - dx)**2 + (ay - dy)**2)
                if dist < marking_distance:
                    is_marked = True
                    break

            if not is_marked:
                alert = TacticalAlert(
                    alert_type=TacticalAlertType.UNMARKED_RUNNER,
                    priority=AlertPriority.CRITICAL,
                    team=defending_team,
                    message=f"UNMARKED! #{attacker.get('jersey_number', '?')} in dangerous position",
                    recommendation="Someone pick up the runner NOW! Nearest player must engage.",
                    frame_number=self.current_frame,
                    timestamp_ms=self.current_timestamp,
                    position=(ax, ay),
                    involved_players=[attacker.get('jersey_number', 0)]
                )
                alerts.append(alert)

        return alerts[:1]  # Most critical unmarked runner only

    def _detect_pressing_opportunities(
        self,
        home_players: List[Dict],
        away_players: List[Dict],
        ball_pos: Optional[Tuple[float, float]]
    ) -> List[TacticalAlert]:
        """Detect optimal pressing triggers."""
        if not ball_pos:
            return []

        alerts = []
        ball_x, ball_y = ball_pos

        # Determine which team should press
        for pressing_team, opponents in [("home", away_players), ("away", home_players)]:
            # Find ball carrier among opponents
            carrier = None
            min_dist = float('inf')
            for p in opponents:
                dist = math.sqrt((p.get('x', 0) - ball_x)**2 + (p.get('y', 0) - ball_y)**2)
                if dist < min_dist:
                    min_dist = dist
                    carrier = p

            if not carrier or min_dist > 15:
                continue

            carrier_x, carrier_y = carrier.get('x', 50), carrier.get('y', 50)

            # Check if carrier is in a poor position (back to goal, near sideline)
            near_sideline = carrier_y < 15 or carrier_y > 85
            in_own_half = (pressing_team == "home" and carrier_x < 50) or \
                         (pressing_team == "away" and carrier_x > 50)

            # Count supporting teammates for carrier
            supporters = sum(
                1 for p in opponents
                if p != carrier and math.sqrt(
                    (p.get('x', 0) - carrier_x)**2 + (p.get('y', 0) - carrier_y)**2
                ) < 15
            )

            # Press trigger: carrier isolated, in poor position
            if supporters <= 1 and (near_sideline or in_own_half):
                pressing_players = home_players if pressing_team == "home" else away_players

                # Count potential pressing players nearby
                pressers = sum(
                    1 for p in pressing_players
                    if math.sqrt(
                        (p.get('x', 0) - carrier_x)**2 + (p.get('y', 0) - carrier_y)**2
                    ) < 20
                )

                if pressers >= 2:
                    alert = TacticalAlert(
                        alert_type=TacticalAlertType.PRESS_TRIGGER,
                        priority=AlertPriority.HIGH,
                        team=pressing_team,
                        message=f"PRESS NOW! #{carrier.get('jersey_number', '?')} isolated" +
                               (" near sideline" if near_sideline else ""),
                        recommendation="Aggressive press! Cut off passing lanes, force the error.",
                        frame_number=self.current_frame,
                        timestamp_ms=self.current_timestamp,
                        position=(carrier_x, carrier_y),
                        involved_players=[carrier.get('jersey_number', 0)]
                    )
                    alerts.append(alert)

        return alerts

    def _detect_transition_moments(self) -> List[TacticalAlert]:
        """Detect transition moments from possession changes."""
        alerts = []

        if len(self.possession_history) < 5:
            return []

        # Check for recent possession change
        recent = self.possession_history[-5:]
        current_team = recent[-1][1]
        prev_team = recent[0][1]

        if current_team != prev_team:
            # Possession just changed - transition moment
            winning_team = current_team
            losing_team = prev_team

            # Alert for team that won possession
            alert = TacticalAlert(
                alert_type=TacticalAlertType.TRANSITION_DANGER if winning_team == "away" else TacticalAlertType.TRANSITION_DANGER,
                priority=AlertPriority.HIGH,
                team=losing_team,
                message=f"TRANSITION! {winning_team.capitalize()} has won possession",
                recommendation="Quick recovery runs! Get goal side immediately. Delay the counter.",
                frame_number=self.current_frame,
                timestamp_ms=self.current_timestamp
            )
            alerts.append(alert)

        return alerts

    def get_active_alerts(self) -> List[Dict]:
        """Get currently active alerts."""
        return [
            a.to_dict() for a in self.active_alerts
            if a.frame_number + a.expires_after_frames > self.current_frame
        ]

    def get_alerts_by_team(self, team: str) -> List[Dict]:
        """Get active alerts for a specific team."""
        return [
            a.to_dict() for a in self.active_alerts
            if a.team == team and
            a.frame_number + a.expires_after_frames > self.current_frame
        ]

    def get_alerts_by_priority(self, min_priority: AlertPriority = AlertPriority.MEDIUM) -> List[Dict]:
        """Get active alerts above a priority threshold."""
        return [
            a.to_dict() for a in self.active_alerts
            if a.priority.value >= min_priority.value and
            a.frame_number + a.expires_after_frames > self.current_frame
        ]

    def dismiss_alert(self, frame_number: int):
        """Dismiss an alert by frame number."""
        self.dismissed_alerts.add(frame_number)
        self.active_alerts = [
            a for a in self.active_alerts
            if a.frame_number != frame_number
        ]

    def get_tactical_summary(self) -> Dict:
        """Get summary of tactical analysis."""
        total_alerts = len(self.alerts)

        alert_counts = defaultdict(int)
        team_alerts = {"home": 0, "away": 0}

        for alert in self.alerts:
            alert_counts[alert.alert_type.value] += 1
            team_alerts[alert.team] += 1

        return {
            "total_alerts": total_alerts,
            "alerts_by_type": dict(alert_counts),
            "alerts_by_team": team_alerts,
            "current_active": len(self.active_alerts),
            "high_priority_count": sum(
                1 for a in self.alerts
                if a.priority in [AlertPriority.HIGH, AlertPriority.CRITICAL]
            )
        }

    def get_full_analysis(self) -> Dict:
        """Get complete tactical analysis."""
        return {
            "summary": self.get_tactical_summary(),
            "active_alerts": self.get_active_alerts(),
            "all_alerts": [a.to_dict() for a in self.alerts[-50:]],  # Last 50 alerts
            "defensive_line_trend": {
                "home": self.defensive_line_history["home"][-30:] if self.defensive_line_history["home"] else [],
                "away": self.defensive_line_history["away"][-30:] if self.defensive_line_history["away"] else []
            }
        }


# Global instance
tactical_intelligence = TacticalIntelligenceService()
