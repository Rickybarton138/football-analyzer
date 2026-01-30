"""
Tactical Analyzer

Analyzes match situations and identifies tactical patterns,
pressing opportunities, space exploitation, and formation issues.
"""
import numpy as np
from typing import List, Dict, Optional, Tuple
from collections import deque
from dataclasses import dataclass

from config import settings
from models.schemas import (
    DetectedPlayer, DetectedBall, Position, TeamSide,
    TacticalAlert, AlertPriority
)


@dataclass
class TacticalSituation:
    """Represents current tactical situation."""
    pressing_opportunity: bool = False
    counter_attack: bool = False
    defensive_transition: bool = False
    space_available: Optional[Position] = None
    formation_broken: bool = False
    outnumbered_area: Optional[str] = None
    high_line_vulnerability: bool = False


@dataclass
class TeamShape:
    """Team's current shape metrics."""
    centroid: Position
    width: float  # Horizontal spread
    depth: float  # Vertical spread
    compactness: float  # How close together
    defensive_line: float  # X position of defensive line
    highest_player: float  # X position of most advanced player


class TacticalAnalyzer:
    """
    Analyzes tactical situations and generates alerts.

    Focuses on user priorities:
    1. Pressing/Defense
    2. Passing/Possession
    3. Physical metrics
    4. Set pieces/xG
    """

    def __init__(self):
        # History for pattern detection
        self.situation_history: deque = deque(maxlen=30)  # ~3 seconds
        self.shape_history: Dict[TeamSide, deque] = {
            TeamSide.HOME: deque(maxlen=30),
            TeamSide.AWAY: deque(maxlen=30)
        }

        # Thresholds
        self.pressing_trigger_distance = 10.0  # meters
        self.compact_threshold = 30.0  # meters - max distance for compact shape
        self.high_line_threshold = 35.0  # meters from goal line
        self.space_threshold = 15.0  # meters - clear space
        self.counter_speed_threshold = 6.0  # m/s

        # Pitch dimensions
        self.pitch_length = settings.PITCH_LENGTH
        self.pitch_width = settings.PITCH_WIDTH

    async def analyze(
        self,
        players: List[DetectedPlayer],
        ball: Optional[DetectedBall],
        frame_number: int = 0,
        timestamp_ms: int = 0
    ) -> List[TacticalAlert]:
        """
        Analyze current frame and generate tactical alerts.

        Args:
            players: All detected players
            ball: Detected ball
            frame_number: Current frame
            timestamp_ms: Current timestamp

        Returns:
            List of tactical alerts
        """
        alerts = []

        # Separate teams
        home_players = [p for p in players if p.team == TeamSide.HOME]
        away_players = [p for p in players if p.team == TeamSide.AWAY]

        # Calculate team shapes
        home_shape = self._calculate_team_shape(home_players)
        away_shape = self._calculate_team_shape(away_players)

        if home_shape:
            self.shape_history[TeamSide.HOME].append(home_shape)
        if away_shape:
            self.shape_history[TeamSide.AWAY].append(away_shape)

        # Determine possession
        possession_team = self._determine_possession(players, ball)

        # Analyze for each team
        for team, team_players, opponent_players, team_shape, opponent_shape in [
            (TeamSide.HOME, home_players, away_players, home_shape, away_shape),
            (TeamSide.AWAY, away_players, home_players, away_shape, home_shape)
        ]:
            if not team_shape:
                continue

            # Pressing opportunities (when opponent has ball)
            if possession_team and possession_team != team:
                pressing_alert = self._analyze_pressing(
                    team, team_players, opponent_players, ball,
                    team_shape, timestamp_ms
                )
                if pressing_alert:
                    alerts.append(pressing_alert)

            # Defensive vulnerabilities
            if possession_team and possession_team == team:
                # Check for counter-attack risk
                defensive_alert = self._analyze_defensive_risk(
                    team, team_players, opponent_players, ball,
                    team_shape, opponent_shape, timestamp_ms
                )
                if defensive_alert:
                    alerts.append(defensive_alert)

            # Formation drift
            formation_alert = self._analyze_formation(
                team, team_shape, timestamp_ms
            )
            if formation_alert:
                alerts.append(formation_alert)

            # Space to exploit (when team has ball)
            if possession_team == team:
                space_alert = self._analyze_space(
                    team, team_players, opponent_players,
                    opponent_shape, timestamp_ms
                )
                if space_alert:
                    alerts.append(space_alert)

        return alerts

    def _calculate_team_shape(
        self,
        players: List[DetectedPlayer]
    ) -> Optional[TeamShape]:
        """Calculate team's current shape metrics."""
        positions = [
            p.pitch_position for p in players
            if p.pitch_position and not p.is_goalkeeper
        ]

        if len(positions) < 3:
            return None

        # Calculate centroid
        x_coords = [p.x for p in positions]
        y_coords = [p.y for p in positions]

        centroid = Position(
            x=np.mean(x_coords),
            y=np.mean(y_coords)
        )

        # Calculate dimensions
        width = max(y_coords) - min(y_coords)
        depth = max(x_coords) - min(x_coords)

        # Calculate compactness (average distance to centroid)
        distances = []
        for pos in positions:
            dist = np.sqrt(
                (pos.x - centroid.x) ** 2 +
                (pos.y - centroid.y) ** 2
            )
            distances.append(dist)
        compactness = np.mean(distances)

        # Defensive line (4 deepest outfield players)
        sorted_by_x = sorted(x_coords)
        defensive_line = np.mean(sorted_by_x[:4])

        # Highest player
        highest_player = max(x_coords)

        return TeamShape(
            centroid=centroid,
            width=width,
            depth=depth,
            compactness=compactness,
            defensive_line=defensive_line,
            highest_player=highest_player
        )

    def _determine_possession(
        self,
        players: List[DetectedPlayer],
        ball: Optional[DetectedBall]
    ) -> Optional[TeamSide]:
        """Determine which team has possession."""
        if ball is None or ball.pitch_position is None:
            return None

        min_dist = float('inf')
        nearest_team = None

        for player in players:
            if player.pitch_position is None:
                continue

            dist = np.sqrt(
                (player.pitch_position.x - ball.pitch_position.x) ** 2 +
                (player.pitch_position.y - ball.pitch_position.y) ** 2
            )

            if dist < min_dist:
                min_dist = dist
                nearest_team = player.team

        # Only assign possession if player is close enough
        if min_dist <= 3.0:
            return nearest_team

        return None

    def _analyze_pressing(
        self,
        team: TeamSide,
        team_players: List[DetectedPlayer],
        opponent_players: List[DetectedPlayer],
        ball: Optional[DetectedBall],
        team_shape: TeamShape,
        timestamp_ms: int
    ) -> Optional[TacticalAlert]:
        """Analyze pressing opportunities."""
        if ball is None or ball.pitch_position is None:
            return None

        # Find opponent on ball
        opponent_on_ball = None
        min_dist = float('inf')
        for player in opponent_players:
            if player.pitch_position is None:
                continue
            dist = np.sqrt(
                (player.pitch_position.x - ball.pitch_position.x) ** 2 +
                (player.pitch_position.y - ball.pitch_position.y) ** 2
            )
            if dist < min_dist:
                min_dist = dist
                opponent_on_ball = player

        if opponent_on_ball is None:
            return None

        # Count pressing players (within trigger distance)
        pressing_count = 0
        pressing_players = []
        for player in team_players:
            if player.pitch_position is None:
                continue
            dist = np.sqrt(
                (player.pitch_position.x - ball.pitch_position.x) ** 2 +
                (player.pitch_position.y - ball.pitch_position.y) ** 2
            )
            if dist <= self.pressing_trigger_distance:
                pressing_count += 1
                pressing_players.append(player.track_id)

        # Check if opponent has limited passing options
        passing_options = 0
        for opponent in opponent_players:
            if opponent.pitch_position is None or opponent == opponent_on_ball:
                continue
            # Check if passing lane is blocked
            blocked = self._is_passing_lane_blocked(
                opponent_on_ball.pitch_position,
                opponent.pitch_position,
                team_players
            )
            if not blocked:
                passing_options += 1

        # Generate alert if good pressing opportunity
        if pressing_count >= 2 and passing_options <= 2:
            return TacticalAlert(
                alert_id=f"press_{timestamp_ms}",
                priority=AlertPriority.IMMEDIATE,
                timestamp_ms=timestamp_ms,
                message="PRESS NOW!",
                details=f"{pressing_count} players in position, opponent has limited options",
                suggested_action="Trigger coordinated press",
                related_players=pressing_players,
                position=ball.pitch_position,
                expires_at_ms=timestamp_ms + 3000  # Alert valid for 3 seconds
            )

        return None

    def _is_passing_lane_blocked(
        self,
        from_pos: Position,
        to_pos: Position,
        blockers: List[DetectedPlayer]
    ) -> bool:
        """Check if passing lane is blocked by defenders."""
        for blocker in blockers:
            if blocker.pitch_position is None:
                continue

            # Check if blocker is close to the line between positions
            dist = self._point_to_line_distance(
                blocker.pitch_position,
                from_pos,
                to_pos
            )

            if dist < 2.0:  # Within 2 meters of passing lane
                return True

        return False

    def _point_to_line_distance(
        self,
        point: Position,
        line_start: Position,
        line_end: Position
    ) -> float:
        """Calculate perpendicular distance from point to line."""
        # Line vector
        dx = line_end.x - line_start.x
        dy = line_end.y - line_start.y

        # Normalize
        length = np.sqrt(dx ** 2 + dy ** 2)
        if length == 0:
            return float('inf')

        dx /= length
        dy /= length

        # Vector from line start to point
        px = point.x - line_start.x
        py = point.y - line_start.y

        # Project onto line
        proj = px * dx + py * dy

        # Check if projection is on the segment
        if proj < 0 or proj > length:
            return float('inf')

        # Perpendicular distance
        return abs(px * (-dy) + py * dx)

    def _analyze_defensive_risk(
        self,
        team: TeamSide,
        team_players: List[DetectedPlayer],
        opponent_players: List[DetectedPlayer],
        ball: Optional[DetectedBall],
        team_shape: TeamShape,
        opponent_shape: Optional[TeamShape],
        timestamp_ms: int
    ) -> Optional[TacticalAlert]:
        """Analyze defensive vulnerabilities."""
        # Check high defensive line
        attacking_direction = 1 if team == TeamSide.HOME else -1

        if attacking_direction == 1:
            # Home attacks right, defends left
            defensive_line = team_shape.defensive_line
            risk_threshold = self.high_line_threshold
        else:
            # Away attacks left, defends right
            defensive_line = self.pitch_length - team_shape.defensive_line
            risk_threshold = self.high_line_threshold

        # Check for opponents behind defensive line
        opponents_behind = 0
        for opponent in opponent_players:
            if opponent.pitch_position is None:
                continue

            if attacking_direction == 1:
                if opponent.pitch_position.x < team_shape.defensive_line:
                    opponents_behind += 1
            else:
                if opponent.pitch_position.x > team_shape.defensive_line:
                    opponents_behind += 1

        # High line with space behind
        if defensive_line < risk_threshold and opponents_behind >= 1:
            return TacticalAlert(
                alert_id=f"def_risk_{timestamp_ms}",
                priority=AlertPriority.IMMEDIATE,
                timestamp_ms=timestamp_ms,
                message="High line risk!",
                details=f"Defensive line at {defensive_line:.0f}m, {opponents_behind} opponents behind",
                suggested_action="Drop deeper or press aggressively",
                expires_at_ms=timestamp_ms + 5000
            )

        # Check for exposed flanks (wide areas)
        return None

    def _analyze_formation(
        self,
        team: TeamSide,
        team_shape: TeamShape,
        timestamp_ms: int
    ) -> Optional[TacticalAlert]:
        """Analyze formation integrity."""
        # Check compactness
        if team_shape.compactness > self.compact_threshold:
            return TacticalAlert(
                alert_id=f"formation_{timestamp_ms}",
                priority=AlertPriority.TACTICAL,
                timestamp_ms=timestamp_ms,
                message="Shape too stretched",
                details=f"Team spread over {team_shape.compactness:.0f}m average",
                suggested_action="Compact the midfield",
                expires_at_ms=timestamp_ms + 10000
            )

        # Check depth (too flat)
        if team_shape.depth < 15:
            return TacticalAlert(
                alert_id=f"formation_flat_{timestamp_ms}",
                priority=AlertPriority.TACTICAL,
                timestamp_ms=timestamp_ms,
                message="Team too flat",
                details="Not enough depth between lines",
                suggested_action="Stagger positions vertically",
                expires_at_ms=timestamp_ms + 10000
            )

        return None

    def _analyze_space(
        self,
        team: TeamSide,
        team_players: List[DetectedPlayer],
        opponent_players: List[DetectedPlayer],
        opponent_shape: Optional[TeamShape],
        timestamp_ms: int
    ) -> Optional[TacticalAlert]:
        """Analyze exploitable space."""
        if opponent_shape is None:
            return None

        # Check for space behind opponent defensive line
        attacking_x = self.pitch_length if team == TeamSide.HOME else 0

        # Space between opponent line and goal
        space_behind = abs(
            attacking_x - opponent_shape.defensive_line
        )

        if space_behind > self.space_threshold:
            return TacticalAlert(
                alert_id=f"space_{timestamp_ms}",
                priority=AlertPriority.TACTICAL,
                timestamp_ms=timestamp_ms,
                message="Space behind!",
                details=f"{space_behind:.0f}m of space behind their line",
                suggested_action="Play through ball or switch play wide",
                position=Position(
                    x=opponent_shape.defensive_line,
                    y=self.pitch_width / 2
                ),
                expires_at_ms=timestamp_ms + 5000
            )

        # Check wide areas
        if opponent_shape.width < 50:
            # Opponent is narrow - exploit flanks
            return TacticalAlert(
                alert_id=f"wide_{timestamp_ms}",
                priority=AlertPriority.TACTICAL,
                timestamp_ms=timestamp_ms,
                message="Width available",
                details="Opponent narrow - switch play wide",
                suggested_action="Use full-backs or wingers",
                expires_at_ms=timestamp_ms + 8000
            )

        return None

    def get_situation_summary(self) -> Dict:
        """Get summary of current tactical situation."""
        return {
            "pressing_opportunities": sum(
                1 for s in self.situation_history
                if s.pressing_opportunity
            ),
            "counter_attacks": sum(
                1 for s in self.situation_history
                if s.counter_attack
            ),
            "formation_issues": sum(
                1 for s in self.situation_history
                if s.formation_broken
            )
        }

    def reset(self):
        """Reset analyzer state."""
        self.situation_history.clear()
        for key in self.shape_history:
            self.shape_history[key].clear()
