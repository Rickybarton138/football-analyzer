"""
Formation Tracking Service

Ensures consistent tracking of all 22 players (11 per team) for formation analysis.

Features:
- Maintains 11 players per team at all times
- Interpolates positions for temporarily missing players
- Tracks formation patterns and trends over time
- Detects formation changes (e.g., 4-4-2 to 4-3-3)
- Monitors defensive/attacking line positions
- Calculates team shape metrics (width, depth, compactness)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum
import math


class FormationType(Enum):
    """Common football formations."""
    F_4_4_2 = "4-4-2"
    F_4_3_3 = "4-3-3"
    F_4_2_3_1 = "4-2-3-1"
    F_3_5_2 = "3-5-2"
    F_3_4_3 = "3-4-3"
    F_5_3_2 = "5-3-2"
    F_5_4_1 = "5-4-1"
    F_4_5_1 = "4-5-1"
    F_4_1_4_1 = "4-1-4-1"
    UNKNOWN = "unknown"


@dataclass
class PlayerSlot:
    """A slot for a player in the formation."""
    slot_id: int  # 1-11
    jersey_number: Optional[int] = None
    current_x: float = 50.0
    current_y: float = 50.0
    last_seen_frame: int = 0
    frames_interpolated: int = 0
    velocity_x: float = 0.0
    velocity_y: float = 0.0
    position_role: str = "unknown"  # GK, CB, LB, RB, CM, etc.
    is_interpolated: bool = False

    # Position history for smoothing
    position_history: List[Tuple[float, float, int]] = field(default_factory=list)

    def update_position(self, x: float, y: float, frame_number: int):
        """Update position with new detection."""
        if self.position_history:
            last_x, last_y, last_frame = self.position_history[-1]
            dt = max(1, frame_number - last_frame)
            self.velocity_x = (x - last_x) / dt
            self.velocity_y = (y - last_y) / dt

        self.current_x = x
        self.current_y = y
        self.last_seen_frame = frame_number
        self.frames_interpolated = 0
        self.is_interpolated = False

        self.position_history.append((x, y, frame_number))
        if len(self.position_history) > 30:  # Keep ~1 second at 30fps
            self.position_history = self.position_history[-30:]

    def interpolate_position(self, frame_number: int) -> Tuple[float, float]:
        """Interpolate position when not detected."""
        frames_since = frame_number - self.last_seen_frame
        self.frames_interpolated = frames_since
        self.is_interpolated = True

        # Simple linear interpolation with velocity decay
        decay = 0.9 ** frames_since
        pred_x = self.current_x + self.velocity_x * frames_since * decay
        pred_y = self.current_y + self.velocity_y * frames_since * decay

        # Clamp to pitch bounds
        pred_x = max(0, min(100, pred_x))
        pred_y = max(0, min(100, pred_y))

        return (pred_x, pred_y)


@dataclass
class TeamFormation:
    """Formation state for one team."""
    team: str  # 'home' or 'away'
    slots: Dict[int, PlayerSlot] = field(default_factory=dict)  # slot_id -> PlayerSlot
    jersey_to_slot: Dict[int, int] = field(default_factory=dict)  # jersey -> slot_id

    # Formation metrics
    current_formation: FormationType = FormationType.UNKNOWN
    defensive_line_x: float = 25.0
    midfield_line_x: float = 50.0
    attacking_line_x: float = 75.0
    team_width: float = 60.0
    team_depth: float = 50.0
    compactness: float = 0.0

    # Formation history for trend analysis
    formation_history: List[Tuple[int, FormationType]] = field(default_factory=list)
    line_positions_history: List[Tuple[int, float, float, float]] = field(default_factory=list)

    def __post_init__(self):
        # Initialize 11 slots with default positions based on 4-4-2
        default_positions = self._get_default_positions()
        for slot_id in range(1, 12):
            x, y, role = default_positions[slot_id]
            self.slots[slot_id] = PlayerSlot(
                slot_id=slot_id,
                current_x=x,
                current_y=y,
                position_role=role
            )

    def _get_default_positions(self) -> Dict[int, Tuple[float, float, str]]:
        """Get default positions for a 4-4-2 formation."""
        if self.team == "home":
            # Home attacks right (higher x)
            return {
                1: (5, 50, "GK"),      # Goalkeeper
                2: (25, 85, "RB"),     # Right back
                3: (25, 15, "LB"),     # Left back
                4: (25, 60, "CB"),     # Center back right
                5: (25, 40, "CB"),     # Center back left
                6: (45, 50, "CDM"),    # Central defensive mid
                7: (50, 85, "RM"),     # Right mid
                8: (50, 50, "CM"),     # Central mid
                9: (75, 50, "ST"),     # Striker
                10: (65, 40, "CAM"),   # Attacking mid
                11: (50, 15, "LM"),    # Left mid
            }
        else:
            # Away attacks left (lower x)
            return {
                1: (95, 50, "GK"),
                2: (75, 15, "RB"),
                3: (75, 85, "LB"),
                4: (75, 40, "CB"),
                5: (75, 60, "CB"),
                6: (55, 50, "CDM"),
                7: (50, 15, "RM"),
                8: (50, 50, "CM"),
                9: (25, 50, "ST"),
                10: (35, 60, "CAM"),
                11: (50, 85, "LM"),
            }


@dataclass
class FormationSnapshot:
    """A snapshot of both teams' formations at a point in time."""
    frame_number: int
    timestamp_ms: int
    home_formation: FormationType
    away_formation: FormationType
    home_players: List[Dict]  # 11 players with positions
    away_players: List[Dict]  # 11 players
    home_metrics: Dict
    away_metrics: Dict


class FormationTrackingService:
    """
    Service for tracking 22 players and monitoring formation trends.

    Ensures consistent player tracking even when detections are missed,
    and provides formation analysis over time.
    """

    def __init__(self):
        self.home_formation = TeamFormation(team="home")
        self.away_formation = TeamFormation(team="away")

        # Frame tracking
        self.current_frame = 0
        self.fps = 30.0

        # Formation snapshots for trend analysis
        self.snapshots: List[FormationSnapshot] = []
        self.snapshot_interval = 30  # Every 30 frames (1 second)

        # Player assignment tracking
        self.unassigned_detections: List[Dict] = []

    def set_video_info(self, fps: float, total_frames: int):
        """Set video metadata."""
        self.fps = fps

    def _assign_jersey_to_slot(self, team_formation: TeamFormation, jersey: int, x: float, y: float):
        """Assign a jersey number to the most appropriate slot."""
        if jersey in team_formation.jersey_to_slot:
            return team_formation.jersey_to_slot[jersey]

        # Find the closest unassigned slot
        best_slot = None
        best_dist = float('inf')

        for slot_id, slot in team_formation.slots.items():
            if slot.jersey_number is not None:
                continue

            dist = math.sqrt((x - slot.current_x)**2 + (y - slot.current_y)**2)
            if dist < best_dist:
                best_dist = dist
                best_slot = slot_id

        if best_slot:
            team_formation.slots[best_slot].jersey_number = jersey
            team_formation.jersey_to_slot[jersey] = best_slot
            return best_slot

        return None

    def process_frame(
        self,
        frame_number: int,
        timestamp_ms: int,
        home_detections: List[Dict],  # [{jersey_number, x, y}, ...]
        away_detections: List[Dict]
    ) -> Dict:
        """
        Process a frame and update all 22 player positions.

        Args:
            frame_number: Current frame number
            timestamp_ms: Timestamp in milliseconds
            home_detections: Detected home players with positions
            away_detections: Detected away players with positions

        Returns:
            Dictionary with all 22 player positions and formation analysis
        """
        self.current_frame = frame_number

        # Update home team
        self._update_team_positions(self.home_formation, home_detections, frame_number)

        # Update away team
        self._update_team_positions(self.away_formation, away_detections, frame_number)

        # Calculate formation metrics
        home_metrics = self._calculate_formation_metrics(self.home_formation)
        away_metrics = self._calculate_formation_metrics(self.away_formation)

        # Detect formations
        self.home_formation.current_formation = self._detect_formation(self.home_formation)
        self.away_formation.current_formation = self._detect_formation(self.away_formation)

        # Create snapshot periodically
        if frame_number % self.snapshot_interval == 0:
            self._create_snapshot(frame_number, timestamp_ms, home_metrics, away_metrics)

        return self.get_current_state(frame_number, timestamp_ms)

    def _update_team_positions(
        self,
        team_formation: TeamFormation,
        detections: List[Dict],
        frame_number: int
    ):
        """Update positions for a team, interpolating missing players."""
        # Track which slots were updated
        updated_slots = set()

        # First, update slots with detected players
        for det in detections:
            jersey = det.get('jersey_number', 0)
            x = det.get('x', 50)
            y = det.get('y', 50)

            if jersey <= 0:
                continue

            # Find or assign slot for this jersey
            if jersey in team_formation.jersey_to_slot:
                slot_id = team_formation.jersey_to_slot[jersey]
            else:
                slot_id = self._assign_jersey_to_slot(team_formation, jersey, x, y)

            if slot_id:
                team_formation.slots[slot_id].update_position(x, y, frame_number)
                updated_slots.add(slot_id)

        # Interpolate positions for undetected players
        for slot_id, slot in team_formation.slots.items():
            if slot_id not in updated_slots:
                # Player not detected - interpolate
                pred_x, pred_y = slot.interpolate_position(frame_number)
                slot.current_x = pred_x
                slot.current_y = pred_y

    def _calculate_formation_metrics(self, team_formation: TeamFormation) -> Dict:
        """Calculate formation metrics (lines, width, depth, compactness)."""
        positions = [(s.current_x, s.current_y) for s in team_formation.slots.values()]

        if not positions:
            return {}

        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]

        # Calculate lines (excluding goalkeeper)
        outfield_xs = sorted(xs)[1:]  # Exclude lowest x (goalkeeper)

        if len(outfield_xs) >= 4:
            # Defensive line (back 4)
            def_line = np.mean(outfield_xs[:4])
            # Midfield line (middle players)
            mid_line = np.mean(outfield_xs[4:7]) if len(outfield_xs) > 6 else np.mean(outfield_xs[2:5])
            # Attacking line (front players)
            att_line = np.mean(outfield_xs[-3:])
        else:
            def_line = mid_line = att_line = np.mean(xs)

        team_formation.defensive_line_x = def_line
        team_formation.midfield_line_x = mid_line
        team_formation.attacking_line_x = att_line

        # Width and depth
        team_formation.team_width = max(ys) - min(ys)
        team_formation.team_depth = max(xs) - min(xs)

        # Compactness (average distance between all pairs)
        total_dist = 0
        pairs = 0
        for i, p1 in enumerate(positions):
            for p2 in positions[i+1:]:
                total_dist += math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
                pairs += 1

        team_formation.compactness = total_dist / max(pairs, 1)

        return {
            "defensive_line": round(def_line, 1),
            "midfield_line": round(mid_line, 1),
            "attacking_line": round(att_line, 1),
            "width": round(team_formation.team_width, 1),
            "depth": round(team_formation.team_depth, 1),
            "compactness": round(team_formation.compactness, 1)
        }

    def _detect_formation(self, team_formation: TeamFormation) -> FormationType:
        """Detect the current formation based on player positions."""
        positions = [(s.current_x, s.current_y, s.slot_id) for s in team_formation.slots.values()]

        if len(positions) < 11:
            return FormationType.UNKNOWN

        # Sort by x position (excluding GK)
        sorted_by_x = sorted(positions, key=lambda p: p[0])

        # Count players in each third (defensive, midfield, attacking)
        gk_x = sorted_by_x[0][0]
        max_x = sorted_by_x[-1][0]
        third_size = (max_x - gk_x) / 3

        defensive = sum(1 for p in sorted_by_x[1:] if p[0] < gk_x + third_size)
        midfield = sum(1 for p in sorted_by_x[1:] if gk_x + third_size <= p[0] < gk_x + 2*third_size)
        attacking = sum(1 for p in sorted_by_x[1:] if p[0] >= gk_x + 2*third_size)

        # Match to known formations
        if defensive == 4 and midfield == 4 and attacking == 2:
            return FormationType.F_4_4_2
        elif defensive == 4 and midfield == 3 and attacking == 3:
            return FormationType.F_4_3_3
        elif defensive == 4 and (midfield == 5 or midfield == 4) and attacking <= 2:
            return FormationType.F_4_2_3_1
        elif defensive == 3 and midfield == 5 and attacking == 2:
            return FormationType.F_3_5_2
        elif defensive == 5 and midfield == 3 and attacking == 2:
            return FormationType.F_5_3_2
        elif defensive == 5 and midfield == 4 and attacking == 1:
            return FormationType.F_5_4_1

        return FormationType.UNKNOWN

    def _create_snapshot(
        self,
        frame_number: int,
        timestamp_ms: int,
        home_metrics: Dict,
        away_metrics: Dict
    ):
        """Create a formation snapshot for trend analysis."""
        home_players = [
            {
                "slot_id": slot.slot_id,
                "jersey_number": slot.jersey_number,
                "x": round(slot.current_x, 1),
                "y": round(slot.current_y, 1),
                "role": slot.position_role,
                "is_interpolated": slot.is_interpolated
            }
            for slot in self.home_formation.slots.values()
        ]

        away_players = [
            {
                "slot_id": slot.slot_id,
                "jersey_number": slot.jersey_number,
                "x": round(slot.current_x, 1),
                "y": round(slot.current_y, 1),
                "role": slot.position_role,
                "is_interpolated": slot.is_interpolated
            }
            for slot in self.away_formation.slots.values()
        ]

        snapshot = FormationSnapshot(
            frame_number=frame_number,
            timestamp_ms=timestamp_ms,
            home_formation=self.home_formation.current_formation,
            away_formation=self.away_formation.current_formation,
            home_players=home_players,
            away_players=away_players,
            home_metrics=home_metrics,
            away_metrics=away_metrics
        )

        self.snapshots.append(snapshot)

        # Keep last 5 minutes of snapshots
        max_snapshots = int(5 * 60 * self.fps / self.snapshot_interval)
        if len(self.snapshots) > max_snapshots:
            self.snapshots = self.snapshots[-max_snapshots:]

        # Track formation history
        self.home_formation.formation_history.append((frame_number, self.home_formation.current_formation))
        self.away_formation.formation_history.append((frame_number, self.away_formation.current_formation))

    def get_current_state(self, frame_number: int, timestamp_ms: int) -> Dict:
        """Get current state with all 22 players."""
        home_players = []
        away_players = []

        for slot in self.home_formation.slots.values():
            home_players.append({
                "slot_id": slot.slot_id,
                "jersey_number": slot.jersey_number or slot.slot_id,
                "x": round(slot.current_x, 1),
                "y": round(slot.current_y, 1),
                "role": slot.position_role,
                "is_interpolated": slot.is_interpolated,
                "frames_since_seen": slot.frames_interpolated
            })

        for slot in self.away_formation.slots.values():
            away_players.append({
                "slot_id": slot.slot_id,
                "jersey_number": slot.jersey_number or slot.slot_id,
                "x": round(slot.current_x, 1),
                "y": round(slot.current_y, 1),
                "role": slot.position_role,
                "is_interpolated": slot.is_interpolated,
                "frames_since_seen": slot.frames_interpolated
            })

        return {
            "frame_number": frame_number,
            "timestamp_ms": timestamp_ms,
            "players": {
                "home": sorted(home_players, key=lambda p: p["slot_id"]),
                "away": sorted(away_players, key=lambda p: p["slot_id"])
            },
            "formations": {
                "home": self.home_formation.current_formation.value,
                "away": self.away_formation.current_formation.value
            },
            "metrics": {
                "home": {
                    "defensive_line": round(self.home_formation.defensive_line_x, 1),
                    "midfield_line": round(self.home_formation.midfield_line_x, 1),
                    "attacking_line": round(self.home_formation.attacking_line_x, 1),
                    "width": round(self.home_formation.team_width, 1),
                    "depth": round(self.home_formation.team_depth, 1),
                    "compactness": round(self.home_formation.compactness, 1)
                },
                "away": {
                    "defensive_line": round(self.away_formation.defensive_line_x, 1),
                    "midfield_line": round(self.away_formation.midfield_line_x, 1),
                    "attacking_line": round(self.away_formation.attacking_line_x, 1),
                    "width": round(self.away_formation.team_width, 1),
                    "depth": round(self.away_formation.team_depth, 1),
                    "compactness": round(self.away_formation.compactness, 1)
                }
            },
            "player_count": {
                "home": 11,
                "away": 11,
                "home_interpolated": sum(1 for s in self.home_formation.slots.values() if s.is_interpolated),
                "away_interpolated": sum(1 for s in self.away_formation.slots.values() if s.is_interpolated)
            }
        }

    def get_formation_trends(self, last_n_seconds: int = 60) -> Dict:
        """Get formation trends over the last N seconds."""
        frames_to_check = int(last_n_seconds * self.fps / self.snapshot_interval)
        recent_snapshots = self.snapshots[-frames_to_check:] if self.snapshots else []

        if not recent_snapshots:
            return {"home": [], "away": [], "summary": {}}

        # Count formation occurrences
        home_formations = defaultdict(int)
        away_formations = defaultdict(int)

        for snap in recent_snapshots:
            home_formations[snap.home_formation.value] += 1
            away_formations[snap.away_formation.value] += 1

        # Calculate line position trends
        home_def_lines = [s.home_metrics.get("defensive_line", 25) for s in recent_snapshots]
        away_def_lines = [s.away_metrics.get("defensive_line", 75) for s in recent_snapshots]

        return {
            "home": {
                "formation_counts": dict(home_formations),
                "primary_formation": max(home_formations, key=home_formations.get) if home_formations else "unknown",
                "avg_defensive_line": round(np.mean(home_def_lines), 1) if home_def_lines else 25,
                "defensive_line_trend": "pushing_up" if len(home_def_lines) > 5 and home_def_lines[-1] > home_def_lines[0] + 3 else "dropping_back" if len(home_def_lines) > 5 and home_def_lines[-1] < home_def_lines[0] - 3 else "stable"
            },
            "away": {
                "formation_counts": dict(away_formations),
                "primary_formation": max(away_formations, key=away_formations.get) if away_formations else "unknown",
                "avg_defensive_line": round(np.mean(away_def_lines), 1) if away_def_lines else 75,
                "defensive_line_trend": "pushing_up" if len(away_def_lines) > 5 and away_def_lines[-1] < away_def_lines[0] - 3 else "dropping_back" if len(away_def_lines) > 5 and away_def_lines[-1] > away_def_lines[0] + 3 else "stable"
            },
            "snapshots_analyzed": len(recent_snapshots),
            "time_period_seconds": last_n_seconds
        }

    def get_all_22_positions(self, frame_number: Optional[int] = None) -> Dict:
        """
        Get positions of all 22 players, always returning exactly 11 per team.

        This is the main method for the 2D radar to use.
        """
        state = self.get_current_state(frame_number or self.current_frame, 0)

        return {
            "frame_number": state["frame_number"],
            "players": state["players"],
            "ball": None,  # Ball tracked separately
            "formations": state["formations"],
            "total_players": 22,
            "interpolated_count": state["player_count"]["home_interpolated"] + state["player_count"]["away_interpolated"]
        }

    def reset(self):
        """Reset tracker for new match."""
        self.home_formation = TeamFormation(team="home")
        self.away_formation = TeamFormation(team="away")
        self.current_frame = 0
        self.snapshots.clear()


# Global instance
formation_tracker = FormationTrackingService()
