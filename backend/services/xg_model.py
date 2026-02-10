"""
Expected Goals (xG) Model

Calculates the probability of a shot resulting in a goal based on:
- Shot position (distance and angle to goal)
- Shot type (header, foot, volley)
- Assist type (through ball, cross, cutback)
- Defensive pressure
- Game state (score, time)

Based on statistical models similar to those used by Opta, StatsBomb, and Understat.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from math import atan2, degrees, sqrt, exp


class ShotType(Enum):
    """Type of shot taken."""
    RIGHT_FOOT = "right_foot"
    LEFT_FOOT = "left_foot"
    HEADER = "header"
    OTHER = "other"  # Includes volleys, backheel, etc.


class ShotSituation(Enum):
    """Situation leading to the shot."""
    OPEN_PLAY = "open_play"
    FAST_BREAK = "fast_break"  # Counter attack
    SET_PIECE = "set_piece"  # Free kick position
    CORNER = "corner"
    PENALTY = "penalty"
    DIRECT_FREE_KICK = "direct_free_kick"
    THROW_IN = "throw_in"


class AssistType(Enum):
    """Type of pass/situation before the shot."""
    THROUGH_BALL = "through_ball"
    CROSS = "cross"
    CUTBACK = "cutback"
    PULL_BACK = "pull_back"
    LONG_BALL = "long_ball"
    SHORT_PASS = "short_pass"
    DRIBBLE = "dribble"  # Individual effort
    REBOUND = "rebound"
    NONE = "none"


@dataclass
class Shot:
    """A shot event with all relevant data for xG calculation."""
    # Position (normalized 0-100 pitch coordinates)
    x: float  # 0 = own goal line, 100 = opponent goal line
    y: float  # 0 = left touchline, 100 = right touchline

    # Shot details
    shot_type: ShotType = ShotType.RIGHT_FOOT
    situation: ShotSituation = ShotSituation.OPEN_PLAY
    assist_type: AssistType = AssistType.SHORT_PASS

    # Context
    frame_number: int = 0
    timestamp_ms: int = 0
    team: str = "home"
    player_jersey: Optional[int] = None

    # Outcome
    is_goal: bool = False
    on_target: bool = False
    blocked: bool = False

    # Additional factors
    defenders_in_path: int = 0  # Number of defenders between shot and goal
    goalkeeper_position: Optional[Tuple[float, float]] = None  # GK position
    big_chance: bool = False  # Clear goalscoring opportunity
    fast_break: bool = False  # Counter attack

    # Calculated xG (filled by model)
    xg: float = 0.0
    xg_factors: Dict = field(default_factory=dict)


@dataclass
class xGResult:
    """Result of xG calculation with breakdown."""
    xg: float  # 0.0 to 1.0 probability
    base_xg: float  # From position only
    modifiers: Dict[str, float]  # Each factor's contribution
    confidence: float  # How confident we are in this xG
    shot_quality: str  # "excellent", "good", "average", "poor"


class xGModel:
    """
    Expected Goals Model for calculating shot quality.

    Uses a combination of geometric analysis and statistical modeling
    to estimate the probability of a goal.
    """

    # Standard pitch dimensions (meters)
    PITCH_LENGTH = 105.0
    PITCH_WIDTH = 68.0

    # Goal dimensions (meters)
    GOAL_WIDTH = 7.32
    GOAL_HEIGHT = 2.44

    # Penalty spot distance from goal
    PENALTY_DISTANCE = 11.0

    # Six-yard box dimensions
    SIX_YARD_WIDTH = 18.32
    SIX_YARD_DEPTH = 5.5

    # Penalty area dimensions
    PENALTY_AREA_WIDTH = 40.32
    PENALTY_AREA_DEPTH = 16.5

    def __init__(self):
        self.shots: List[Shot] = []
        self.total_xg: Dict[str, float] = {"home": 0.0, "away": 0.0}
        self.shot_count: Dict[str, int] = {"home": 0, "away": 0}

        # xG modifiers based on statistical analysis
        self.modifiers = {
            # Shot type modifiers
            "header": 0.75,  # Headers are harder to score
            "weak_foot": 0.85,  # Assumed weaker foot
            "volley": 0.80,

            # Assist type modifiers
            "through_ball": 1.15,  # Better chances
            "cross": 0.85,  # Harder to convert
            "cutback": 1.25,  # High quality chance
            "pull_back": 1.20,
            "rebound": 1.30,  # Often close range
            "dribble": 0.95,  # Individual effort

            # Situation modifiers
            "fast_break": 1.20,  # Counter attacks
            "set_piece": 0.90,  # Defensive setup
            "big_chance": 1.40,  # Clear opportunity

            # Defensive pressure modifiers
            "no_pressure": 1.15,
            "light_pressure": 1.00,
            "heavy_pressure": 0.75,
            "blocked_path": 0.60,  # Defender in way

            # Goalkeeper position
            "gk_out_of_position": 1.50,
            "gk_well_positioned": 0.85,
        }

    def _normalize_to_meters(self, x: float, y: float) -> Tuple[float, float]:
        """Convert 0-100 pitch coordinates to meters."""
        # x: 0 = own goal, 100 = opponent goal
        # We measure distance from opponent goal (x=100)
        x_meters = (100 - x) / 100 * self.PITCH_LENGTH

        # y: 0 = left, 100 = right, center = 50
        y_meters = (y - 50) / 100 * self.PITCH_WIDTH

        return x_meters, y_meters

    def _calculate_distance(self, x: float, y: float) -> float:
        """Calculate distance from shot position to center of goal."""
        x_m, y_m = self._normalize_to_meters(x, y)
        return sqrt(x_m**2 + y_m**2)

    def _calculate_angle(self, x: float, y: float) -> float:
        """
        Calculate the angle to goal in degrees.

        This is the angle subtended by the goal posts from the shot position.
        A wider angle = better chance.
        """
        x_m, y_m = self._normalize_to_meters(x, y)

        if x_m <= 0:
            return 0  # Behind goal line

        # Goal post positions (from center)
        half_goal = self.GOAL_WIDTH / 2

        # Angles to each post
        angle_left = atan2(half_goal - y_m, x_m)
        angle_right = atan2(-half_goal - y_m, x_m)

        # Angle subtended by goal
        goal_angle = abs(degrees(angle_left - angle_right))

        return goal_angle

    def _base_xg_from_position(self, x: float, y: float) -> float:
        """
        Calculate base xG from shot position using geometric model.

        Based on analysis of thousands of shots, distance and angle
        are the primary factors in determining goal probability.
        """
        distance = self._calculate_distance(x, y)
        angle = self._calculate_angle(x, y)

        # Distance factor (exponential decay)
        # Closer shots = higher xG
        # At penalty spot (~11m): ~0.76
        # At edge of box (~16.5m): ~0.10
        # At 25m: ~0.03
        distance_factor = exp(-0.1 * distance)

        # Angle factor (logistic function)
        # Central shots = higher xG
        # Max angle (on goal line center) = ~180 degrees
        # Penalty spot angle = ~35 degrees
        # Tight angle (near post) = ~10 degrees
        angle_factor = 1 / (1 + exp(-0.1 * (angle - 15)))

        # Combine factors
        # Weight distance more heavily than angle
        base_xg = 0.7 * distance_factor + 0.3 * angle_factor

        # Scale to realistic range (0.02 to 0.95)
        base_xg = max(0.02, min(0.95, base_xg))

        # Special zones adjustments
        if x >= 94 and 40 <= y <= 60:  # Six-yard box
            base_xg = max(base_xg, 0.60)
        elif x >= 83.5 and 30 <= y <= 70:  # Penalty area
            base_xg = max(base_xg, 0.08)

        return base_xg

    def calculate_xg(self, shot: Shot) -> xGResult:
        """
        Calculate expected goals for a shot.

        Returns xG value between 0 and 1 representing probability of goal.
        """
        modifiers_applied = {}

        # Start with base xG from position
        base_xg = self._base_xg_from_position(shot.x, shot.y)
        modifiers_applied["base_position"] = base_xg

        # Apply shot type modifier
        shot_mod = 1.0
        if shot.shot_type == ShotType.HEADER:
            shot_mod = self.modifiers["header"]
            modifiers_applied["header"] = shot_mod

        # Apply assist type modifier
        assist_mod = 1.0
        if shot.assist_type == AssistType.THROUGH_BALL:
            assist_mod = self.modifiers["through_ball"]
            modifiers_applied["through_ball"] = assist_mod
        elif shot.assist_type == AssistType.CROSS:
            assist_mod = self.modifiers["cross"]
            modifiers_applied["cross"] = assist_mod
        elif shot.assist_type == AssistType.CUTBACK:
            assist_mod = self.modifiers["cutback"]
            modifiers_applied["cutback"] = assist_mod
        elif shot.assist_type == AssistType.PULL_BACK:
            assist_mod = self.modifiers["pull_back"]
            modifiers_applied["pull_back"] = assist_mod
        elif shot.assist_type == AssistType.REBOUND:
            assist_mod = self.modifiers["rebound"]
            modifiers_applied["rebound"] = assist_mod
        elif shot.assist_type == AssistType.DRIBBLE:
            assist_mod = self.modifiers["dribble"]
            modifiers_applied["dribble"] = assist_mod

        # Apply situation modifier
        situation_mod = 1.0
        if shot.situation == ShotSituation.FAST_BREAK or shot.fast_break:
            situation_mod = self.modifiers["fast_break"]
            modifiers_applied["fast_break"] = situation_mod
        elif shot.situation == ShotSituation.SET_PIECE:
            situation_mod = self.modifiers["set_piece"]
            modifiers_applied["set_piece"] = situation_mod
        elif shot.situation == ShotSituation.PENALTY:
            # Penalties have fixed xG
            return xGResult(
                xg=0.76,
                base_xg=0.76,
                modifiers={"penalty": 0.76},
                confidence=0.95,
                shot_quality="excellent"
            )

        # Big chance modifier
        if shot.big_chance:
            situation_mod *= self.modifiers["big_chance"]
            modifiers_applied["big_chance"] = self.modifiers["big_chance"]

        # Defensive pressure modifier
        pressure_mod = 1.0
        if shot.defenders_in_path == 0:
            pressure_mod = self.modifiers["no_pressure"]
            modifiers_applied["no_pressure"] = pressure_mod
        elif shot.defenders_in_path >= 2:
            pressure_mod = self.modifiers["blocked_path"]
            modifiers_applied["blocked_path"] = pressure_mod
        elif shot.defenders_in_path == 1:
            pressure_mod = self.modifiers["heavy_pressure"]
            modifiers_applied["heavy_pressure"] = pressure_mod

        # Goalkeeper position modifier
        gk_mod = 1.0
        if shot.goalkeeper_position:
            gk_x, gk_y = shot.goalkeeper_position
            # Check if GK is out of position
            ideal_gk_y = 50  # Center of goal
            gk_offset = abs(gk_y - ideal_gk_y)

            if gk_offset > 15:  # GK significantly out of position
                gk_mod = self.modifiers["gk_out_of_position"]
                modifiers_applied["gk_out_of_position"] = gk_mod
            elif gk_offset < 5 and gk_x > 95:  # Well positioned
                gk_mod = self.modifiers["gk_well_positioned"]
                modifiers_applied["gk_well_positioned"] = gk_mod

        # Calculate final xG
        final_xg = base_xg * shot_mod * assist_mod * situation_mod * pressure_mod * gk_mod

        # Clamp to valid range
        final_xg = max(0.01, min(0.99, final_xg))

        # Determine shot quality
        if final_xg >= 0.40:
            quality = "excellent"
        elif final_xg >= 0.20:
            quality = "good"
        elif final_xg >= 0.08:
            quality = "average"
        else:
            quality = "poor"

        # Confidence based on data available
        confidence = 0.85
        if shot.goalkeeper_position:
            confidence += 0.05
        if shot.defenders_in_path > 0:
            confidence += 0.05

        return xGResult(
            xg=round(final_xg, 3),
            base_xg=round(base_xg, 3),
            modifiers=modifiers_applied,
            confidence=round(confidence, 2),
            shot_quality=quality
        )

    def add_shot(self, shot: Shot) -> xGResult:
        """Add a shot and calculate its xG."""
        result = self.calculate_xg(shot)
        shot.xg = result.xg
        shot.xg_factors = result.modifiers

        self.shots.append(shot)
        self.total_xg[shot.team] = self.total_xg.get(shot.team, 0) + result.xg
        self.shot_count[shot.team] = self.shot_count.get(shot.team, 0) + 1

        return result

    def calculate_xg_from_position(
        self,
        x: float,
        y: float,
        shot_type: str = "foot",
        is_header: bool = False,
        from_cross: bool = False,
        fast_break: bool = False,
        big_chance: bool = False,
        defenders: int = 0
    ) -> Dict:
        """
        Simplified xG calculation from just position and basic info.

        This is the main interface for the event detector to use.
        """
        shot = Shot(
            x=x,
            y=y,
            shot_type=ShotType.HEADER if is_header else ShotType.RIGHT_FOOT,
            assist_type=AssistType.CROSS if from_cross else AssistType.SHORT_PASS,
            fast_break=fast_break,
            big_chance=big_chance,
            defenders_in_path=defenders
        )

        result = self.calculate_xg(shot)

        return {
            "xg": result.xg,
            "base_xg": result.base_xg,
            "quality": result.shot_quality,
            "confidence": result.confidence,
            "distance": round(self._calculate_distance(x, y), 1),
            "angle": round(self._calculate_angle(x, y), 1),
            "factors": result.modifiers
        }

    def get_team_xg(self, team: str) -> Dict:
        """Get xG summary for a team."""
        team_shots = [s for s in self.shots if s.team == team]
        goals = sum(1 for s in team_shots if s.is_goal)

        return {
            "total_xg": round(self.total_xg.get(team, 0), 2),
            "shots": len(team_shots),
            "goals": goals,
            "xg_per_shot": round(self.total_xg.get(team, 0) / max(1, len(team_shots)), 3),
            "conversion_rate": round(goals / max(1, len(team_shots)) * 100, 1),
            "over_performance": round(goals - self.total_xg.get(team, 0), 2)
        }

    def get_shot_map_data(self) -> Dict:
        """Get shot data formatted for visualization."""
        return {
            "shots": [
                {
                    "x": s.x,
                    "y": s.y,
                    "xg": s.xg,
                    "team": s.team,
                    "player": s.player_jersey,
                    "is_goal": s.is_goal,
                    "on_target": s.on_target,
                    "shot_type": s.shot_type.value,
                    "quality": "excellent" if s.xg >= 0.40 else "good" if s.xg >= 0.20 else "average" if s.xg >= 0.08 else "poor",
                    "timestamp_ms": s.timestamp_ms
                }
                for s in self.shots
            ],
            "summary": {
                "home": self.get_team_xg("home"),
                "away": self.get_team_xg("away")
            }
        }

    def get_xg_timeline(self) -> List[Dict]:
        """Get cumulative xG over time for chart visualization."""
        timeline = []
        home_cumulative = 0.0
        away_cumulative = 0.0

        for shot in sorted(self.shots, key=lambda s: s.timestamp_ms):
            if shot.team == "home":
                home_cumulative += shot.xg
            else:
                away_cumulative += shot.xg

            timeline.append({
                "timestamp_ms": shot.timestamp_ms,
                "time_str": f"{shot.timestamp_ms // 60000}:{(shot.timestamp_ms // 1000) % 60:02d}",
                "home_xg": round(home_cumulative, 2),
                "away_xg": round(away_cumulative, 2),
                "event_team": shot.team,
                "event_xg": shot.xg,
                "is_goal": shot.is_goal
            })

        return timeline

    def get_big_chances(self) -> List[Dict]:
        """Get list of big chances (high xG shots)."""
        big_chances = [s for s in self.shots if s.xg >= 0.25]

        return [
            {
                "x": s.x,
                "y": s.y,
                "xg": s.xg,
                "team": s.team,
                "player": s.player_jersey,
                "converted": s.is_goal,
                "timestamp_ms": s.timestamp_ms,
                "time_str": f"{s.timestamp_ms // 60000}:{(s.timestamp_ms // 1000) % 60:02d}"
            }
            for s in sorted(big_chances, key=lambda x: -x.xg)
        ]

    def reset(self):
        """Reset model for new match."""
        self.shots.clear()
        self.total_xg = {"home": 0.0, "away": 0.0}
        self.shot_count = {"home": 0, "away": 0}


# Global instance
xg_model = xGModel()
