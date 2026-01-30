"""
Tactical Analysis System

Provides tactical intelligence for football matches:
1. Formation effectiveness scoring
2. Unit-based analysis (defense/midfield/attack)
3. Tactical pattern recognition
4. Phase-of-play analysis (based on UEFA Four Moments)

This integrates with the team profile system to provide
context-aware tactical recommendations.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
from datetime import datetime
import numpy as np
import json

from ai.team_profile import (
    FORMATION_TEMPLATES,
    PRINCIPLES_OF_PLAY,
    FOUR_MOMENTS,
    PlayingStyle
)


# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================

class TacticalPhase(str, Enum):
    """The four moments of the game (UEFA framework)"""
    ATTACKING_ORGANIZATION = "attacking_organization"
    DEFENSIVE_ORGANIZATION = "defensive_organization"
    ATTACKING_TRANSITION = "attacking_transition"
    DEFENSIVE_TRANSITION = "defensive_transition"


class TacticalUnit(str, Enum):
    """Team units for analysis"""
    DEFENSIVE = "defensive"  # GK + Back line
    MIDFIELD = "midfield"  # Central midfielders
    ATTACKING = "attacking"  # Forwards/wingers
    FULL_TEAM = "full_team"


class FormationZone(str, Enum):
    """Zones on the pitch"""
    DEFENSIVE_THIRD = "defensive_third"
    MIDDLE_THIRD = "middle_third"
    ATTACKING_THIRD = "attacking_third"
    LEFT_CHANNEL = "left_channel"
    CENTRAL_CHANNEL = "central_channel"
    RIGHT_CHANNEL = "right_channel"
    HALF_SPACE_LEFT = "half_space_left"
    HALF_SPACE_RIGHT = "half_space_right"


# Standard pitch dimensions (normalized to 100x100)
PITCH_LENGTH = 100
PITCH_WIDTH = 100

# Zone boundaries (normalized)
ZONE_BOUNDARIES = {
    "defensive_third": {"y_min": 0, "y_max": 33},
    "middle_third": {"y_min": 33, "y_max": 66},
    "attacking_third": {"y_min": 66, "y_max": 100},
    "left_channel": {"x_min": 0, "x_max": 25},
    "central_channel": {"x_min": 25, "x_max": 75},
    "right_channel": {"x_min": 75, "x_max": 100},
    "half_space_left": {"x_min": 15, "x_max": 35},
    "half_space_right": {"x_min": 65, "x_max": 85},
}

# Formation position templates (normalized coordinates)
FORMATION_POSITIONS = {
    "4-3-3": {
        "GK": (50, 5),
        "RB": (80, 25), "RCB": (65, 20), "LCB": (35, 20), "LB": (20, 25),
        "RCM": (65, 45), "CDM": (50, 40), "LCM": (35, 45),
        "RW": (85, 70), "ST": (50, 80), "LW": (15, 70)
    },
    "4-4-2": {
        "GK": (50, 5),
        "RB": (80, 25), "RCB": (65, 20), "LCB": (35, 20), "LB": (20, 25),
        "RM": (85, 50), "RCM": (60, 45), "LCM": (40, 45), "LM": (15, 50),
        "RST": (60, 75), "LST": (40, 75)
    },
    "4-2-3-1": {
        "GK": (50, 5),
        "RB": (80, 25), "RCB": (65, 20), "LCB": (35, 20), "LB": (20, 25),
        "RCDM": (60, 38), "LCDM": (40, 38),
        "RAM": (75, 55), "CAM": (50, 58), "LAM": (25, 55),
        "ST": (50, 78)
    },
    "3-5-2": {
        "GK": (50, 5),
        "RCB": (70, 20), "CB": (50, 18), "LCB": (30, 20),
        "RWB": (90, 45), "RCM": (65, 45), "CDM": (50, 38), "LCM": (35, 45), "LWB": (10, 45),
        "RST": (60, 75), "LST": (40, 75)
    },
    "3-4-3": {
        "GK": (50, 5),
        "RCB": (70, 20), "CB": (50, 18), "LCB": (30, 20),
        "RWB": (90, 50), "RCM": (60, 42), "LCM": (40, 42), "LWB": (10, 50),
        "RW": (80, 72), "ST": (50, 78), "LW": (20, 72)
    }
}

# Unit position groupings
UNIT_POSITIONS = {
    "4-3-3": {
        TacticalUnit.DEFENSIVE: ["GK", "RB", "RCB", "LCB", "LB"],
        TacticalUnit.MIDFIELD: ["RCM", "CDM", "LCM"],
        TacticalUnit.ATTACKING: ["RW", "ST", "LW"]
    },
    "4-4-2": {
        TacticalUnit.DEFENSIVE: ["GK", "RB", "RCB", "LCB", "LB"],
        TacticalUnit.MIDFIELD: ["RM", "RCM", "LCM", "LM"],
        TacticalUnit.ATTACKING: ["RST", "LST"]
    },
    "4-2-3-1": {
        TacticalUnit.DEFENSIVE: ["GK", "RB", "RCB", "LCB", "LB"],
        TacticalUnit.MIDFIELD: ["RCDM", "LCDM", "RAM", "CAM", "LAM"],
        TacticalUnit.ATTACKING: ["ST"]
    },
    "3-5-2": {
        TacticalUnit.DEFENSIVE: ["GK", "RCB", "CB", "LCB"],
        TacticalUnit.MIDFIELD: ["RWB", "RCM", "CDM", "LCM", "LWB"],
        TacticalUnit.ATTACKING: ["RST", "LST"]
    },
    "3-4-3": {
        TacticalUnit.DEFENSIVE: ["GK", "RCB", "CB", "LCB"],
        TacticalUnit.MIDFIELD: ["RWB", "RCM", "LCM", "LWB"],
        TacticalUnit.ATTACKING: ["RW", "ST", "LW"]
    }
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class PlayerPosition:
    """Player position on the pitch"""
    track_id: int
    jersey_number: Optional[int]
    x: float  # 0-100 (left to right)
    y: float  # 0-100 (own goal to opponent goal)
    team: str  # "home" or "away"
    assigned_role: Optional[str] = None  # e.g., "RB", "CDM"


@dataclass
class FormationSnapshot:
    """A snapshot of team formation at a specific moment"""
    timestamp_ms: int
    frame_number: int
    players: List[PlayerPosition]
    detected_formation: str
    formation_confidence: float
    phase: TacticalPhase
    ball_position: Optional[Tuple[float, float]] = None


@dataclass
class UnitMetrics:
    """Metrics for a tactical unit (defense/midfield/attack)"""
    unit: TacticalUnit

    # Compactness
    horizontal_compactness: float  # Distance between widest players
    vertical_compactness: float  # Distance between furthest players
    overall_compactness: float  # Combined metric

    # Shape
    average_height: float  # Average y-position (how high/deep)
    width_coverage: float  # What percentage of pitch width covered

    # Coordination
    line_alignment: float  # How well players form a line (0-1)
    spacing_quality: float  # How evenly spaced (0-1)

    # Effectiveness scores
    defensive_coverage: float  # How well positioned defensively
    attacking_support: float  # How well positioned to support attacks


@dataclass
class FormationEffectivenessScore:
    """Comprehensive formation effectiveness assessment"""
    formation: str
    timestamp_ms: int
    frame_number: int

    # Overall score
    overall_effectiveness: float  # 0-100

    # Component scores
    shape_maintenance: float  # How well formation shape is maintained
    compactness_score: float  # Team compactness
    width_balance: float  # Balance between wide and central
    depth_balance: float  # Balance between attack and defense

    # Unit scores
    unit_scores: Dict[str, UnitMetrics]

    # Phase-specific
    phase: TacticalPhase
    phase_appropriateness: float  # How appropriate formation is for current phase

    # Comparisons
    deviation_from_ideal: float  # How far from ideal positions

    # Issues identified
    vulnerabilities: List[str]
    strengths: List[str]
    recommendations: List[str]


@dataclass
class TacticalPeriodAnalysis:
    """Analysis of a period of play (e.g., 5-minute segment)"""
    start_time_ms: int
    end_time_ms: int

    # Phase breakdown
    phase_percentages: Dict[str, float]  # What % of time in each phase

    # Formation stability
    formation_changes: int
    avg_formation_score: float

    # Unit analysis
    defensive_unit_avg: UnitMetrics
    midfield_unit_avg: UnitMetrics
    attacking_unit_avg: UnitMetrics

    # Tactical patterns
    pressing_intensity: float  # Based on defensive line height
    possession_style: str  # "build_up", "direct", "mixed"
    width_usage: str  # "wide", "central", "balanced"

    # Key moments
    formation_breakdowns: List[int]  # Timestamps when formation broke

    # Recommendations
    tactical_observations: List[str]


# =============================================================================
# FORMATION DETECTION
# =============================================================================

class FormationDetector:
    """
    Detects team formation from player positions.

    Uses template matching against known formations to identify
    the most likely tactical setup.
    """

    def __init__(self):
        self.formation_templates = FORMATION_POSITIONS
        self.known_formations = list(FORMATION_TEMPLATES.keys())

    def detect_formation(
        self,
        players: List[PlayerPosition],
        team: str = "home"
    ) -> Tuple[str, float]:
        """
        Detect the most likely formation from player positions.

        Returns:
            Tuple of (formation_name, confidence)
        """
        # Filter to team players
        team_players = [p for p in players if p.team == team]

        if len(team_players) < 10:
            return "unknown", 0.0

        # Exclude goalkeeper (lowest y position)
        outfield_players = sorted(team_players, key=lambda p: p.y)[1:]  # Skip lowest

        if len(outfield_players) < 9:
            return "unknown", 0.0

        best_formation = "4-4-2"  # Default
        best_score = 0.0

        for formation_name, template in self.formation_templates.items():
            score = self._match_formation(outfield_players, template)
            if score > best_score:
                best_score = score
                best_formation = formation_name

        # Convert score to confidence (0-1)
        confidence = min(1.0, best_score / 100)

        return best_formation, confidence

    def _match_formation(
        self,
        players: List[PlayerPosition],
        template: Dict[str, Tuple[float, float]]
    ) -> float:
        """
        Calculate how well players match a formation template.

        Uses Hungarian algorithm style matching to find optimal assignment.
        """
        # Get template positions (excluding GK)
        template_positions = [(name, pos) for name, pos in template.items() if name != "GK"]

        if len(players) < len(template_positions):
            return 0.0

        # Calculate distance matrix
        distances = []
        for player in players[:len(template_positions)]:
            player_distances = []
            for _, (tx, ty) in template_positions:
                dist = np.sqrt((player.x - tx)**2 + (player.y - ty)**2)
                player_distances.append(dist)
            distances.append(player_distances)

        # Simple greedy matching (could use Hungarian for optimal)
        total_distance = 0
        used_positions = set()

        for player_dists in distances:
            best_dist = float('inf')
            best_idx = 0
            for idx, dist in enumerate(player_dists):
                if idx not in used_positions and dist < best_dist:
                    best_dist = dist
                    best_idx = idx
            used_positions.add(best_idx)
            total_distance += best_dist

        # Convert to score (lower distance = higher score)
        avg_distance = total_distance / len(template_positions)
        score = max(0, 100 - avg_distance * 2)  # Penalize distance

        return score

    def assign_roles(
        self,
        players: List[PlayerPosition],
        formation: str,
        team: str = "home"
    ) -> List[PlayerPosition]:
        """
        Assign tactical roles to players based on formation.
        """
        if formation not in self.formation_templates:
            return players

        template = self.formation_templates[formation]
        team_players = [p for p in players if p.team == team]

        # Find GK (lowest y)
        if team_players:
            gk = min(team_players, key=lambda p: p.y)
            gk.assigned_role = "GK"

        # Assign remaining players
        outfield = [p for p in team_players if p.assigned_role != "GK"]
        template_outfield = [(name, pos) for name, pos in template.items() if name != "GK"]

        # Match each player to nearest template position
        used_roles = {"GK"}
        for player in outfield:
            best_role = None
            best_dist = float('inf')

            for role, (tx, ty) in template_outfield:
                if role not in used_roles:
                    dist = np.sqrt((player.x - tx)**2 + (player.y - ty)**2)
                    if dist < best_dist:
                        best_dist = dist
                        best_role = role

            if best_role:
                player.assigned_role = best_role
                used_roles.add(best_role)

        return players


# =============================================================================
# TACTICAL ANALYSIS SERVICE
# =============================================================================

class TacticalAnalysisService:
    """
    Main service for tactical analysis.

    Provides:
    - Formation effectiveness scoring
    - Unit-based analysis
    - Phase detection
    - Tactical recommendations
    """

    def __init__(self):
        self.formation_detector = FormationDetector()
        self.snapshots: List[FormationSnapshot] = []
        self.effectiveness_scores: List[FormationEffectivenessScore] = []

        # Configuration
        self.ideal_compactness_horizontal = 35  # meters (normalized)
        self.ideal_compactness_vertical = 30
        self.ideal_unit_spacing = 12  # between units

    # =========================================================================
    # FORMATION EFFECTIVENESS SCORING
    # =========================================================================

    def calculate_formation_effectiveness(
        self,
        players: List[PlayerPosition],
        ball_position: Optional[Tuple[float, float]],
        team: str = "home",
        team_formation: str = "4-3-3",
        team_style: PlayingStyle = PlayingStyle.BALANCED
    ) -> FormationEffectivenessScore:
        """
        Calculate comprehensive formation effectiveness score.

        This is THE key metric - tells you how well the team is set up tactically.
        """
        timestamp_ms = int(datetime.now().timestamp() * 1000)

        # Detect current formation
        detected_formation, formation_confidence = self.formation_detector.detect_formation(
            players, team
        )

        # Assign roles
        players = self.formation_detector.assign_roles(players, detected_formation, team)
        team_players = [p for p in players if p.team == team]

        # Detect current phase
        phase = self._detect_phase(team_players, ball_position)

        # Calculate component scores
        shape_score = self._calculate_shape_maintenance(
            team_players, team_formation
        )

        compactness = self._calculate_compactness(team_players)

        width_balance = self._calculate_width_balance(team_players)
        depth_balance = self._calculate_depth_balance(team_players)

        # Calculate unit scores
        unit_scores = {}
        for unit in [TacticalUnit.DEFENSIVE, TacticalUnit.MIDFIELD, TacticalUnit.ATTACKING]:
            unit_scores[unit.value] = self._calculate_unit_metrics(
                team_players, unit, detected_formation
            )

        # Phase appropriateness
        phase_score = self._calculate_phase_appropriateness(
            team_players, phase, team_style, ball_position
        )

        # Deviation from ideal
        deviation = self._calculate_deviation_from_ideal(
            team_players, team_formation
        )

        # Identify vulnerabilities and strengths
        vulnerabilities = self._identify_vulnerabilities(
            team_players, unit_scores, phase
        )
        strengths = self._identify_strengths(
            team_players, unit_scores, phase
        )
        recommendations = self._generate_recommendations(
            vulnerabilities, phase, team_style
        )

        # Calculate overall score
        overall = (
            shape_score * 0.25 +
            compactness * 0.2 +
            width_balance * 0.15 +
            depth_balance * 0.15 +
            phase_score * 0.15 +
            (100 - deviation) * 0.1
        )

        score = FormationEffectivenessScore(
            formation=detected_formation,
            timestamp_ms=timestamp_ms,
            frame_number=0,
            overall_effectiveness=round(overall, 1),
            shape_maintenance=round(shape_score, 1),
            compactness_score=round(compactness, 1),
            width_balance=round(width_balance, 1),
            depth_balance=round(depth_balance, 1),
            unit_scores=unit_scores,
            phase=phase,
            phase_appropriateness=round(phase_score, 1),
            deviation_from_ideal=round(deviation, 1),
            vulnerabilities=vulnerabilities,
            strengths=strengths,
            recommendations=recommendations
        )

        self.effectiveness_scores.append(score)
        return score

    def _detect_phase(
        self,
        players: List[PlayerPosition],
        ball_position: Optional[Tuple[float, float]]
    ) -> TacticalPhase:
        """Detect current phase of play based on positions."""
        if not players:
            return TacticalPhase.DEFENSIVE_ORGANIZATION

        # Calculate team's average position
        avg_y = sum(p.y for p in players) / len(players)

        # Use ball position if available
        if ball_position:
            ball_y = ball_position[1]

            if ball_y > 66:  # Attacking third
                if avg_y > 50:
                    return TacticalPhase.ATTACKING_ORGANIZATION
                else:
                    return TacticalPhase.ATTACKING_TRANSITION
            elif ball_y < 33:  # Defensive third
                if avg_y < 50:
                    return TacticalPhase.DEFENSIVE_ORGANIZATION
                else:
                    return TacticalPhase.DEFENSIVE_TRANSITION
            else:
                # Middle third - use team shape
                if avg_y > 50:
                    return TacticalPhase.ATTACKING_ORGANIZATION
                else:
                    return TacticalPhase.DEFENSIVE_ORGANIZATION

        # Without ball, use average position
        if avg_y > 55:
            return TacticalPhase.ATTACKING_ORGANIZATION
        elif avg_y < 45:
            return TacticalPhase.DEFENSIVE_ORGANIZATION
        else:
            return TacticalPhase.ATTACKING_ORGANIZATION  # Default

    def _calculate_shape_maintenance(
        self,
        players: List[PlayerPosition],
        expected_formation: str
    ) -> float:
        """Calculate how well formation shape is maintained (0-100)."""
        if not players or expected_formation not in FORMATION_POSITIONS:
            return 50.0

        template = FORMATION_POSITIONS[expected_formation]

        # Match players to roles and calculate deviation
        total_deviation = 0
        matched = 0

        for player in players:
            if player.assigned_role and player.assigned_role in template:
                expected = template[player.assigned_role]
                deviation = np.sqrt(
                    (player.x - expected[0])**2 +
                    (player.y - expected[1])**2
                )
                total_deviation += deviation
                matched += 1

        if matched == 0:
            return 50.0

        avg_deviation = total_deviation / matched

        # Convert to 0-100 score (lower deviation = higher score)
        # 0 deviation = 100, 50+ deviation = 0
        score = max(0, 100 - avg_deviation * 2)

        return score

    def _calculate_compactness(self, players: List[PlayerPosition]) -> float:
        """Calculate team compactness score (0-100)."""
        if len(players) < 2:
            return 50.0

        # Horizontal compactness (x spread)
        x_positions = [p.x for p in players]
        horizontal_spread = max(x_positions) - min(x_positions)

        # Vertical compactness (y spread)
        y_positions = [p.y for p in players]
        vertical_spread = max(y_positions) - min(y_positions)

        # Ideal spreads
        ideal_horizontal = 45  # Allow width for attacking
        ideal_vertical = 35   # Compact but with depth

        # Score based on deviation from ideal
        h_score = 100 - abs(horizontal_spread - ideal_horizontal) * 2
        v_score = 100 - abs(vertical_spread - ideal_vertical) * 2

        return max(0, min(100, (h_score + v_score) / 2))

    def _calculate_width_balance(self, players: List[PlayerPosition]) -> float:
        """Calculate balance between left and right sides (0-100)."""
        if not players:
            return 50.0

        left_players = [p for p in players if p.x < 40]
        right_players = [p for p in players if p.x > 60]
        central_players = [p for p in players if 40 <= p.x <= 60]

        # Ideal distribution: 3 left, 4 central, 3 right (for 10 outfield)
        total = len(players)
        ideal_side = total * 0.3
        ideal_central = total * 0.4

        left_diff = abs(len(left_players) - ideal_side)
        right_diff = abs(len(right_players) - ideal_side)
        central_diff = abs(len(central_players) - ideal_central)

        total_diff = left_diff + right_diff + central_diff

        # Lower difference = higher score
        score = max(0, 100 - total_diff * 15)

        return score

    def _calculate_depth_balance(self, players: List[PlayerPosition]) -> float:
        """Calculate balance between defensive and attacking depth (0-100)."""
        if not players:
            return 50.0

        # Divide into thirds
        deep = [p for p in players if p.y < 35]
        middle = [p for p in players if 35 <= p.y <= 65]
        high = [p for p in players if p.y > 65]

        total = len(players)

        # Ideal: 4 deep, 4 middle, 2 high (defensive setup)
        # Varies by phase - this is general
        deep_ideal = total * 0.4
        middle_ideal = total * 0.4
        high_ideal = total * 0.2

        deep_diff = abs(len(deep) - deep_ideal)
        middle_diff = abs(len(middle) - middle_ideal)
        high_diff = abs(len(high) - high_ideal)

        total_diff = deep_diff + middle_diff + high_diff

        score = max(0, 100 - total_diff * 15)

        return score

    def _calculate_unit_metrics(
        self,
        players: List[PlayerPosition],
        unit: TacticalUnit,
        formation: str
    ) -> UnitMetrics:
        """Calculate detailed metrics for a tactical unit."""
        # Get players in this unit
        if formation not in UNIT_POSITIONS:
            formation = "4-3-3"  # Default

        unit_roles = UNIT_POSITIONS[formation].get(unit, [])
        unit_players = [p for p in players if p.assigned_role in unit_roles]

        if len(unit_players) < 2:
            return UnitMetrics(
                unit=unit,
                horizontal_compactness=50.0,
                vertical_compactness=50.0,
                overall_compactness=50.0,
                average_height=50.0,
                width_coverage=50.0,
                line_alignment=50.0,
                spacing_quality=50.0,
                defensive_coverage=50.0,
                attacking_support=50.0
            )

        x_positions = [p.x for p in unit_players]
        y_positions = [p.y for p in unit_players]

        # Compactness
        h_compact = max(x_positions) - min(x_positions)
        v_compact = max(y_positions) - min(y_positions)

        # Normalize compactness (smaller = better, but not too tight)
        h_score = 100 - abs(h_compact - 30) * 2  # Ideal around 30
        v_score = 100 - abs(v_compact - 8) * 2   # Ideal around 8 (tight line)

        # Average height
        avg_height = sum(y_positions) / len(y_positions)

        # Width coverage (percentage of pitch width covered)
        width_coverage = (max(x_positions) - min(x_positions)) / 100 * 100

        # Line alignment (how close y values are)
        y_std = np.std(y_positions) if len(y_positions) > 1 else 0
        line_alignment = max(0, 100 - y_std * 5)  # Lower std = better alignment

        # Spacing quality (even distribution)
        if len(x_positions) > 1:
            x_sorted = sorted(x_positions)
            gaps = [x_sorted[i+1] - x_sorted[i] for i in range(len(x_sorted)-1)]
            gap_std = np.std(gaps) if gaps else 0
            spacing_quality = max(0, 100 - gap_std * 3)
        else:
            spacing_quality = 50.0

        # Defensive coverage (based on position and spread)
        if unit == TacticalUnit.DEFENSIVE:
            defensive_coverage = min(100, width_coverage * 1.5 + line_alignment * 0.5)
        else:
            defensive_coverage = 50.0

        # Attacking support
        if unit == TacticalUnit.ATTACKING:
            attacking_support = min(100, avg_height + spacing_quality * 0.3)
        else:
            attacking_support = 50.0

        return UnitMetrics(
            unit=unit,
            horizontal_compactness=max(0, min(100, h_score)),
            vertical_compactness=max(0, min(100, v_score)),
            overall_compactness=max(0, min(100, (h_score + v_score) / 2)),
            average_height=avg_height,
            width_coverage=width_coverage,
            line_alignment=max(0, min(100, line_alignment)),
            spacing_quality=max(0, min(100, spacing_quality)),
            defensive_coverage=max(0, min(100, defensive_coverage)),
            attacking_support=max(0, min(100, attacking_support))
        )

    def _calculate_phase_appropriateness(
        self,
        players: List[PlayerPosition],
        phase: TacticalPhase,
        style: PlayingStyle,
        ball_position: Optional[Tuple[float, float]]
    ) -> float:
        """Calculate how appropriate current positions are for the phase."""
        if not players:
            return 50.0

        avg_y = sum(p.y for p in players) / len(players)
        y_spread = max(p.y for p in players) - min(p.y for p in players)

        score = 50.0  # Base score

        if phase == TacticalPhase.ATTACKING_ORGANIZATION:
            # Want high line, good width
            if avg_y > 55:
                score += 20
            if y_spread > 30:  # Good depth for attack
                score += 15
            if style == PlayingStyle.POSSESSION:
                # Extra points for spread out positions
                x_spread = max(p.x for p in players) - min(p.x for p in players)
                if x_spread > 50:
                    score += 15

        elif phase == TacticalPhase.DEFENSIVE_ORGANIZATION:
            # Want compact, deeper
            if avg_y < 45:
                score += 20
            if y_spread < 35:  # Compact
                score += 15
            if style == PlayingStyle.LOW_BLOCK:
                if avg_y < 35:
                    score += 15

        elif phase == TacticalPhase.ATTACKING_TRANSITION:
            # Want players pushing forward
            high_players = len([p for p in players if p.y > 60])
            if high_players >= 3:
                score += 20
            if style == PlayingStyle.COUNTER_ATTACK:
                score += 15

        elif phase == TacticalPhase.DEFENSIVE_TRANSITION:
            # Want quick recovery
            deep_players = len([p for p in players if p.y < 40])
            if deep_players >= 6:
                score += 20
            if style == PlayingStyle.HIGH_PRESS:
                # Want to press high
                if avg_y > 50:
                    score += 15

        return min(100, score)

    def _calculate_deviation_from_ideal(
        self,
        players: List[PlayerPosition],
        formation: str
    ) -> float:
        """Calculate average deviation from ideal formation positions."""
        if formation not in FORMATION_POSITIONS:
            return 50.0

        template = FORMATION_POSITIONS[formation]
        total_deviation = 0
        count = 0

        for player in players:
            if player.assigned_role and player.assigned_role in template:
                ideal = template[player.assigned_role]
                deviation = np.sqrt(
                    (player.x - ideal[0])**2 +
                    (player.y - ideal[1])**2
                )
                total_deviation += deviation
                count += 1

        if count == 0:
            return 50.0

        return total_deviation / count

    def _identify_vulnerabilities(
        self,
        players: List[PlayerPosition],
        unit_scores: Dict[str, UnitMetrics],
        phase: TacticalPhase
    ) -> List[str]:
        """Identify tactical vulnerabilities."""
        vulnerabilities = []

        # Check defensive unit
        def_metrics = unit_scores.get(TacticalUnit.DEFENSIVE.value)
        if def_metrics:
            if def_metrics.line_alignment < 60:
                vulnerabilities.append("Defensive line not aligned - vulnerable to through balls")
            if def_metrics.horizontal_compactness < 50:
                vulnerabilities.append("Defensive unit too spread - gaps between defenders")
            if def_metrics.width_coverage < 40:
                vulnerabilities.append("Defensive width insufficient - vulnerable on flanks")

        # Check midfield
        mid_metrics = unit_scores.get(TacticalUnit.MIDFIELD.value)
        if mid_metrics:
            if mid_metrics.overall_compactness < 50:
                vulnerabilities.append("Midfield too spread - losing second balls")
            if mid_metrics.spacing_quality < 50:
                vulnerabilities.append("Midfield spacing uneven - passing lanes blocked")

        # Check overall
        y_positions = [p.y for p in players]
        if y_positions:
            y_spread = max(y_positions) - min(y_positions)
            if y_spread > 60:
                vulnerabilities.append("Team stretched vertically - vulnerable to counter attacks")

        # Phase-specific
        if phase == TacticalPhase.DEFENSIVE_ORGANIZATION:
            high_players = len([p for p in players if p.y > 60])
            if high_players > 3:
                vulnerabilities.append("Too many players high when defending")

        return vulnerabilities[:5]  # Limit to top 5

    def _identify_strengths(
        self,
        players: List[PlayerPosition],
        unit_scores: Dict[str, UnitMetrics],
        phase: TacticalPhase
    ) -> List[str]:
        """Identify tactical strengths."""
        strengths = []

        # Check defensive unit
        def_metrics = unit_scores.get(TacticalUnit.DEFENSIVE.value)
        if def_metrics:
            if def_metrics.line_alignment > 75:
                strengths.append("Excellent defensive line discipline")
            if def_metrics.defensive_coverage > 75:
                strengths.append("Good defensive coverage of width")

        # Check midfield
        mid_metrics = unit_scores.get(TacticalUnit.MIDFIELD.value)
        if mid_metrics:
            if mid_metrics.overall_compactness > 70:
                strengths.append("Compact midfield unit")
            if mid_metrics.spacing_quality > 70:
                strengths.append("Well-spaced midfield for passing options")

        # Check attacking
        att_metrics = unit_scores.get(TacticalUnit.ATTACKING.value)
        if att_metrics:
            if att_metrics.attacking_support > 70:
                strengths.append("Good attacking positions for chances")

        return strengths[:5]

    def _generate_recommendations(
        self,
        vulnerabilities: List[str],
        phase: TacticalPhase,
        style: PlayingStyle
    ) -> List[str]:
        """Generate tactical recommendations based on analysis."""
        recommendations = []

        for vuln in vulnerabilities:
            if "defensive line" in vuln.lower():
                recommendations.append("Work on defensive line coordination - practice offside trap")
            elif "gaps between defenders" in vuln.lower():
                recommendations.append("Defenders need to communicate and shuffle across together")
            elif "midfield too spread" in vuln.lower():
                recommendations.append("Midfielders stay closer together - max 15 yards apart")
            elif "stretched vertically" in vuln.lower():
                recommendations.append("Compress team - reduce distance between units to 10-15 yards")
            elif "too many players high" in vuln.lower():
                recommendations.append("Get players behind the ball quickly when defending")

        # Phase-specific recommendations
        if phase == TacticalPhase.ATTACKING_TRANSITION and style == PlayingStyle.COUNTER_ATTACK:
            recommendations.append("Look to release wide players immediately on turnover")

        if phase == TacticalPhase.DEFENSIVE_TRANSITION and style == PlayingStyle.HIGH_PRESS:
            recommendations.append("Counter-press within 6 seconds of losing possession")

        return recommendations[:5]

    # =========================================================================
    # PERIOD ANALYSIS
    # =========================================================================

    def analyze_period(
        self,
        start_time_ms: int,
        end_time_ms: int
    ) -> Optional[TacticalPeriodAnalysis]:
        """
        Analyze a period of play from stored effectiveness scores.
        """
        # Filter scores in time range
        period_scores = [
            s for s in self.effectiveness_scores
            if start_time_ms <= s.timestamp_ms <= end_time_ms
        ]

        if not period_scores:
            return None

        # Phase breakdown
        phase_counts = {}
        for score in period_scores:
            phase = score.phase.value
            phase_counts[phase] = phase_counts.get(phase, 0) + 1

        total = len(period_scores)
        phase_percentages = {k: (v / total) * 100 for k, v in phase_counts.items()}

        # Formation changes
        formations = [s.formation for s in period_scores]
        formation_changes = sum(1 for i in range(1, len(formations)) if formations[i] != formations[i-1])

        # Average scores
        avg_effectiveness = sum(s.overall_effectiveness for s in period_scores) / len(period_scores)

        # Unit averages (simplified)
        def_metrics = self._average_unit_metrics(period_scores, TacticalUnit.DEFENSIVE)
        mid_metrics = self._average_unit_metrics(period_scores, TacticalUnit.MIDFIELD)
        att_metrics = self._average_unit_metrics(period_scores, TacticalUnit.ATTACKING)

        # Pressing intensity (based on defensive line height)
        avg_heights = []
        for score in period_scores:
            def_unit = score.unit_scores.get(TacticalUnit.DEFENSIVE.value)
            if def_unit:
                avg_heights.append(def_unit.average_height)
        pressing_intensity = sum(avg_heights) / len(avg_heights) if avg_heights else 40

        # Possession style
        if pressing_intensity > 50:
            possession_style = "high_press"
        elif pressing_intensity < 35:
            possession_style = "low_block"
        else:
            possession_style = "balanced"

        # Width usage
        width_scores = [s.width_balance for s in period_scores]
        avg_width = sum(width_scores) / len(width_scores) if width_scores else 50
        if avg_width > 70:
            width_usage = "wide"
        elif avg_width < 40:
            width_usage = "central"
        else:
            width_usage = "balanced"

        # Formation breakdowns (low scores)
        breakdowns = [
            s.timestamp_ms for s in period_scores
            if s.overall_effectiveness < 40
        ]

        # Observations
        observations = []
        if formation_changes > 3:
            observations.append(f"Formation instability - {formation_changes} changes detected")
        if pressing_intensity > 55:
            observations.append("High pressing approach maintained")
        if avg_effectiveness > 70:
            observations.append("Strong tactical discipline throughout period")
        elif avg_effectiveness < 50:
            observations.append("Tactical structure needs improvement")

        return TacticalPeriodAnalysis(
            start_time_ms=start_time_ms,
            end_time_ms=end_time_ms,
            phase_percentages=phase_percentages,
            formation_changes=formation_changes,
            avg_formation_score=round(avg_effectiveness, 1),
            defensive_unit_avg=def_metrics,
            midfield_unit_avg=mid_metrics,
            attacking_unit_avg=att_metrics,
            pressing_intensity=round(pressing_intensity, 1),
            possession_style=possession_style,
            width_usage=width_usage,
            formation_breakdowns=breakdowns[:10],
            tactical_observations=observations
        )

    def _average_unit_metrics(
        self,
        scores: List[FormationEffectivenessScore],
        unit: TacticalUnit
    ) -> UnitMetrics:
        """Calculate average metrics for a unit across multiple scores."""
        unit_data = [s.unit_scores.get(unit.value) for s in scores if unit.value in s.unit_scores]

        if not unit_data:
            return UnitMetrics(
                unit=unit,
                horizontal_compactness=50.0,
                vertical_compactness=50.0,
                overall_compactness=50.0,
                average_height=50.0,
                width_coverage=50.0,
                line_alignment=50.0,
                spacing_quality=50.0,
                defensive_coverage=50.0,
                attacking_support=50.0
            )

        n = len(unit_data)

        return UnitMetrics(
            unit=unit,
            horizontal_compactness=sum(u.horizontal_compactness for u in unit_data) / n,
            vertical_compactness=sum(u.vertical_compactness for u in unit_data) / n,
            overall_compactness=sum(u.overall_compactness for u in unit_data) / n,
            average_height=sum(u.average_height for u in unit_data) / n,
            width_coverage=sum(u.width_coverage for u in unit_data) / n,
            line_alignment=sum(u.line_alignment for u in unit_data) / n,
            spacing_quality=sum(u.spacing_quality for u in unit_data) / n,
            defensive_coverage=sum(u.defensive_coverage for u in unit_data) / n,
            attacking_support=sum(u.attacking_support for u in unit_data) / n
        )

    # =========================================================================
    # EXPORT AND REPORTING
    # =========================================================================

    def get_match_tactical_report(self) -> Dict:
        """Generate comprehensive tactical report for the match."""
        if not self.effectiveness_scores:
            return {"error": "No tactical data collected"}

        scores = self.effectiveness_scores

        # Overall statistics
        avg_effectiveness = sum(s.overall_effectiveness for s in scores) / len(scores)

        # Phase breakdown
        phase_counts = {}
        for s in scores:
            phase = s.phase.value
            phase_counts[phase] = phase_counts.get(phase, 0) + 1

        total = len(scores)
        phase_percentages = {k: round((v / total) * 100, 1) for k, v in phase_counts.items()}

        # Formation usage
        formation_counts = {}
        for s in scores:
            formation_counts[s.formation] = formation_counts.get(s.formation, 0) + 1

        main_formation = max(formation_counts, key=formation_counts.get)

        # Collect all vulnerabilities and strengths
        all_vulnerabilities = []
        all_strengths = []
        for s in scores:
            all_vulnerabilities.extend(s.vulnerabilities)
            all_strengths.extend(s.strengths)

        # Most common issues
        vuln_counts = {}
        for v in all_vulnerabilities:
            vuln_counts[v] = vuln_counts.get(v, 0) + 1

        strength_counts = {}
        for s in all_strengths:
            strength_counts[s] = strength_counts.get(s, 0) + 1

        top_vulnerabilities = sorted(vuln_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        top_strengths = sorted(strength_counts.items(), key=lambda x: x[1], reverse=True)[:5]

        return {
            "summary": {
                "avg_effectiveness": round(avg_effectiveness, 1),
                "main_formation": main_formation,
                "total_snapshots": len(scores)
            },
            "phase_breakdown": phase_percentages,
            "formation_usage": {k: round((v / total) * 100, 1) for k, v in formation_counts.items()},
            "recurring_vulnerabilities": [
                {"issue": v, "occurrences": c} for v, c in top_vulnerabilities
            ],
            "consistent_strengths": [
                {"strength": s, "occurrences": c} for s, c in top_strengths
            ],
            "unit_averages": {
                "defensive": self._average_unit_metrics(scores, TacticalUnit.DEFENSIVE).__dict__,
                "midfield": self._average_unit_metrics(scores, TacticalUnit.MIDFIELD).__dict__,
                "attacking": self._average_unit_metrics(scores, TacticalUnit.ATTACKING).__dict__
            }
        }

    def export_data(self, output_path: str) -> str:
        """Export all tactical data to JSON."""
        data = {
            "exported_at": datetime.now().isoformat(),
            "total_snapshots": len(self.effectiveness_scores),
            "match_report": self.get_match_tactical_report(),
            "effectiveness_scores": [
                {
                    "formation": s.formation,
                    "timestamp_ms": s.timestamp_ms,
                    "overall_effectiveness": s.overall_effectiveness,
                    "phase": s.phase.value,
                    "shape_maintenance": s.shape_maintenance,
                    "compactness_score": s.compactness_score,
                    "vulnerabilities": s.vulnerabilities,
                    "strengths": s.strengths
                }
                for s in self.effectiveness_scores
            ]
        }

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

        return output_path

    def reset(self):
        """Reset all analysis data."""
        self.snapshots.clear()
        self.effectiveness_scores.clear()


# Global instance
tactical_analysis = TacticalAnalysisService()
