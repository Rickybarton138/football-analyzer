"""
Player Movement Analytics Service

Advanced physical performance metrics:
- Distance covered (total, per half, per speed zone)
- Sprint detection and analysis
- Work rate metrics
- High-intensity running
- Heat maps of movement patterns
- Recovery tracking between sprints

Based on metrics used by GPS tracking systems like Catapult, STATSports, and Playertek.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import math


class SpeedZone(Enum):
    """Speed zones for football movement analysis."""
    STANDING = "standing"      # 0-0.5 m/s
    WALKING = "walking"        # 0.5-2 m/s
    JOGGING = "jogging"        # 2-4 m/s (7.2-14.4 km/h)
    RUNNING = "running"        # 4-5.5 m/s (14.4-19.8 km/h)
    HIGH_SPEED = "high_speed"  # 5.5-7 m/s (19.8-25.2 km/h)
    SPRINTING = "sprinting"    # 7+ m/s (25.2+ km/h)


@dataclass
class Sprint:
    """A detected sprint event."""
    start_frame: int
    end_frame: int
    start_time_ms: int
    end_time_ms: int
    max_speed_ms: float  # meters per second
    distance_m: float
    duration_ms: int
    start_position: Tuple[float, float]
    end_position: Tuple[float, float]
    player_id: int
    team: str

    @property
    def max_speed_kmh(self) -> float:
        return self.max_speed_ms * 3.6

    @property
    def duration_seconds(self) -> float:
        return self.duration_ms / 1000

    def to_dict(self) -> Dict:
        return {
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "start_time_ms": self.start_time_ms,
            "end_time_ms": self.end_time_ms,
            "max_speed_ms": round(self.max_speed_ms, 2),
            "max_speed_kmh": round(self.max_speed_kmh, 1),
            "distance_m": round(self.distance_m, 1),
            "duration_ms": self.duration_ms,
            "duration_seconds": round(self.duration_seconds, 2),
            "start_position": self.start_position,
            "end_position": self.end_position,
            "player_id": self.player_id,
            "team": self.team
        }


@dataclass
class PlayerMovementStats:
    """Movement statistics for a single player."""
    player_id: int
    team: str
    jersey_number: Optional[int] = None

    # Distance metrics (in meters)
    total_distance: float = 0.0
    distance_by_zone: Dict[SpeedZone, float] = field(default_factory=dict)

    # Speed metrics (m/s)
    max_speed: float = 0.0
    avg_speed: float = 0.0
    speed_samples: List[float] = field(default_factory=list)

    # Sprint metrics
    sprints: List[Sprint] = field(default_factory=list)
    total_sprint_distance: float = 0.0
    sprint_count: int = 0

    # High intensity running
    high_intensity_distance: float = 0.0  # >5.5 m/s
    high_intensity_count: int = 0

    # Work rate
    distance_per_minute: float = 0.0

    # Position history for heat map
    position_history: List[Tuple[float, float, int]] = field(default_factory=list)

    # Time tracking
    time_in_zones: Dict[SpeedZone, float] = field(default_factory=dict)  # seconds

    # Recovery
    avg_recovery_time_between_sprints: float = 0.0  # seconds

    def __post_init__(self):
        if not self.distance_by_zone:
            self.distance_by_zone = {zone: 0.0 for zone in SpeedZone}
        if not self.time_in_zones:
            self.time_in_zones = {zone: 0.0 for zone in SpeedZone}
        if not self.speed_samples:
            self.speed_samples = []
        if not self.sprints:
            self.sprints = []
        if not self.position_history:
            self.position_history = []

    def to_dict(self) -> Dict:
        return {
            "player_id": self.player_id,
            "team": self.team,
            "jersey_number": self.jersey_number,
            "total_distance_m": round(self.total_distance, 1),
            "total_distance_km": round(self.total_distance / 1000, 2),
            "distance_by_zone": {
                zone.value: round(dist, 1)
                for zone, dist in self.distance_by_zone.items()
            },
            "max_speed_ms": round(self.max_speed, 2),
            "max_speed_kmh": round(self.max_speed * 3.6, 1),
            "avg_speed_ms": round(self.avg_speed, 2),
            "avg_speed_kmh": round(self.avg_speed * 3.6, 1),
            "sprint_count": self.sprint_count,
            "total_sprint_distance_m": round(self.total_sprint_distance, 1),
            "sprints": [s.to_dict() for s in self.sprints[-10:]],  # Last 10 sprints
            "high_intensity_distance_m": round(self.high_intensity_distance, 1),
            "high_intensity_count": self.high_intensity_count,
            "distance_per_minute_m": round(self.distance_per_minute, 1),
            "time_in_zones_seconds": {
                zone.value: round(time, 1)
                for zone, time in self.time_in_zones.items()
            },
            "avg_recovery_between_sprints_s": round(self.avg_recovery_time_between_sprints, 1)
        }


class PlayerMovementAnalyzer:
    """
    Analyzes player movement from tracking data.

    Converts pixel-based tracking to real-world metrics
    and detects sprints, high-intensity runs, etc.
    """

    # Speed zone thresholds (m/s)
    SPEED_THRESHOLDS = {
        SpeedZone.STANDING: 0.5,
        SpeedZone.WALKING: 2.0,
        SpeedZone.JOGGING: 4.0,
        SpeedZone.RUNNING: 5.5,
        SpeedZone.HIGH_SPEED: 7.0,
        SpeedZone.SPRINTING: float('inf')
    }

    # Sprint detection thresholds
    SPRINT_SPEED_THRESHOLD = 7.0  # m/s to be considered sprinting
    SPRINT_MIN_DURATION = 1.0     # seconds
    SPRINT_END_SPEED = 5.0        # m/s - below this ends sprint

    # Pitch dimensions for coordinate conversion
    PITCH_LENGTH_M = 105.0
    PITCH_WIDTH_M = 68.0

    def __init__(self, fps: float = 30.0, pitch_pixels: Tuple[int, int] = (1920, 1080)):
        self.fps = fps
        self.dt = 1.0 / fps  # Time between frames
        self.pitch_pixels = pitch_pixels

        # Conversion factors (pixels to meters)
        self.px_to_m_x = self.PITCH_LENGTH_M / pitch_pixels[0]
        self.px_to_m_y = self.PITCH_WIDTH_M / pitch_pixels[1]

        # Player stats
        self.player_stats: Dict[int, PlayerMovementStats] = {}

        # Sprint detection state
        self.active_sprints: Dict[int, Dict] = {}  # player_id -> sprint data

        # Previous positions for velocity calculation
        self.prev_positions: Dict[int, Tuple[float, float]] = {}

        # Frame tracking
        self.current_frame = 0
        self.start_time_ms = 0
        self.total_time_ms = 0

    def set_video_info(self, fps: float, width: int, height: int):
        """Set video parameters."""
        self.fps = fps
        self.dt = 1.0 / fps
        self.pitch_pixels = (width, height)
        self.px_to_m_x = self.PITCH_LENGTH_M / width
        self.px_to_m_y = self.PITCH_WIDTH_M / height

    def _pixels_to_meters(self, px: float, py: float) -> Tuple[float, float]:
        """Convert pixel coordinates to meters."""
        return (px * self.px_to_m_x, py * self.px_to_m_y)

    def _calculate_distance_m(self, pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
        """Calculate distance in meters between two pixel positions."""
        dx = (pos2[0] - pos1[0]) * self.px_to_m_x
        dy = (pos2[1] - pos1[1]) * self.px_to_m_y
        return math.sqrt(dx**2 + dy**2)

    def _get_speed_zone(self, speed_ms: float) -> SpeedZone:
        """Determine speed zone from speed in m/s."""
        if speed_ms < self.SPEED_THRESHOLDS[SpeedZone.STANDING]:
            return SpeedZone.STANDING
        elif speed_ms < self.SPEED_THRESHOLDS[SpeedZone.WALKING]:
            return SpeedZone.WALKING
        elif speed_ms < self.SPEED_THRESHOLDS[SpeedZone.JOGGING]:
            return SpeedZone.JOGGING
        elif speed_ms < self.SPEED_THRESHOLDS[SpeedZone.RUNNING]:
            return SpeedZone.RUNNING
        elif speed_ms < self.SPEED_THRESHOLDS[SpeedZone.HIGH_SPEED]:
            return SpeedZone.HIGH_SPEED
        else:
            return SpeedZone.SPRINTING

    def _get_or_create_player(self, player_id: int, team: str) -> PlayerMovementStats:
        """Get or create player stats."""
        if player_id not in self.player_stats:
            self.player_stats[player_id] = PlayerMovementStats(
                player_id=player_id,
                team=team
            )
        return self.player_stats[player_id]

    def process_frame(
        self,
        frame_number: int,
        timestamp_ms: int,
        player_positions: List[Dict]
    ) -> Dict:
        """
        Process a single frame of player positions.

        Args:
            frame_number: Current frame number
            timestamp_ms: Timestamp in milliseconds
            player_positions: List of {player_id, team, x, y, jersey_number}

        Returns:
            Frame analysis results
        """
        self.current_frame = frame_number
        self.total_time_ms = timestamp_ms - self.start_time_ms

        frame_events = []

        for player in player_positions:
            player_id = player.get('track_id') or player.get('player_id', 0)
            team = player.get('team', 'unknown')
            x = player.get('x', 0)
            y = player.get('y', 0)
            jersey = player.get('jersey_number')

            # Get or create player stats
            stats = self._get_or_create_player(player_id, team)
            if jersey and not stats.jersey_number:
                stats.jersey_number = jersey

            # Record position
            stats.position_history.append((x, y, frame_number))
            if len(stats.position_history) > 5000:  # Keep last ~3 minutes at 30fps
                stats.position_history = stats.position_history[-5000:]

            # Calculate speed if we have previous position
            if player_id in self.prev_positions:
                prev_x, prev_y = self.prev_positions[player_id]

                # Distance in meters
                distance_m = self._calculate_distance_m((prev_x, prev_y), (x, y))

                # Speed in m/s
                speed_ms = distance_m / self.dt

                # Smooth speed (simple exponential smoothing)
                if stats.speed_samples:
                    speed_ms = 0.7 * speed_ms + 0.3 * stats.speed_samples[-1]

                # Update stats
                stats.total_distance += distance_m
                stats.speed_samples.append(speed_ms)
                if len(stats.speed_samples) > 1000:
                    stats.speed_samples = stats.speed_samples[-1000:]

                stats.max_speed = max(stats.max_speed, speed_ms)
                stats.avg_speed = np.mean(stats.speed_samples)

                # Update zone-specific metrics
                zone = self._get_speed_zone(speed_ms)
                stats.distance_by_zone[zone] += distance_m
                stats.time_in_zones[zone] += self.dt

                # High intensity tracking
                if speed_ms >= self.SPEED_THRESHOLDS[SpeedZone.HIGH_SPEED]:
                    stats.high_intensity_distance += distance_m

                # Sprint detection
                sprint_event = self._update_sprint_detection(
                    player_id, stats, speed_ms, x, y, frame_number, timestamp_ms
                )
                if sprint_event:
                    frame_events.append(sprint_event)

            # Store current position for next frame
            self.prev_positions[player_id] = (x, y)

        return {
            "frame_number": frame_number,
            "timestamp_ms": timestamp_ms,
            "events": frame_events,
            "player_count": len(player_positions)
        }

    def _update_sprint_detection(
        self,
        player_id: int,
        stats: PlayerMovementStats,
        speed_ms: float,
        x: float,
        y: float,
        frame_number: int,
        timestamp_ms: int
    ) -> Optional[Dict]:
        """
        Update sprint detection for a player.

        Returns sprint event dict if a sprint just ended.
        """
        is_sprinting = speed_ms >= self.SPRINT_SPEED_THRESHOLD

        if is_sprinting:
            if player_id not in self.active_sprints:
                # Start new sprint
                self.active_sprints[player_id] = {
                    "start_frame": frame_number,
                    "start_time_ms": timestamp_ms,
                    "start_position": (x, y),
                    "max_speed": speed_ms,
                    "distance": 0,
                    "positions": [(x, y)]
                }
            else:
                # Continue sprint
                sprint = self.active_sprints[player_id]
                sprint["max_speed"] = max(sprint["max_speed"], speed_ms)

                # Add distance from last position
                if sprint["positions"]:
                    last_pos = sprint["positions"][-1]
                    sprint["distance"] += self._calculate_distance_m(last_pos, (x, y))
                sprint["positions"].append((x, y))

        elif player_id in self.active_sprints:
            # Check if sprint should end (speed dropped below threshold)
            if speed_ms < self.SPRINT_END_SPEED:
                sprint_data = self.active_sprints.pop(player_id)

                duration_ms = timestamp_ms - sprint_data["start_time_ms"]

                # Only count if sprint was long enough
                if duration_ms >= self.SPRINT_MIN_DURATION * 1000:
                    sprint = Sprint(
                        start_frame=sprint_data["start_frame"],
                        end_frame=frame_number,
                        start_time_ms=sprint_data["start_time_ms"],
                        end_time_ms=timestamp_ms,
                        max_speed_ms=sprint_data["max_speed"],
                        distance_m=sprint_data["distance"],
                        duration_ms=duration_ms,
                        start_position=sprint_data["start_position"],
                        end_position=(x, y),
                        player_id=player_id,
                        team=stats.team
                    )

                    stats.sprints.append(sprint)
                    stats.sprint_count += 1
                    stats.total_sprint_distance += sprint.distance_m

                    # Calculate recovery time
                    if len(stats.sprints) > 1:
                        prev_sprint = stats.sprints[-2]
                        recovery = (sprint.start_time_ms - prev_sprint.end_time_ms) / 1000
                        # Update rolling average
                        n = len(stats.sprints) - 1
                        stats.avg_recovery_time_between_sprints = (
                            (stats.avg_recovery_time_between_sprints * (n - 1) + recovery) / n
                        )

                    return {
                        "type": "sprint_completed",
                        "player_id": player_id,
                        "team": stats.team,
                        "sprint": sprint.to_dict()
                    }

        return None

    def get_player_stats(self, player_id: int) -> Optional[Dict]:
        """Get stats for a specific player."""
        if player_id not in self.player_stats:
            return None
        return self.player_stats[player_id].to_dict()

    def get_team_stats(self, team: str) -> Dict:
        """Get aggregated stats for a team."""
        team_players = [
            stats for stats in self.player_stats.values()
            if stats.team == team
        ]

        if not team_players:
            return {"team": team, "player_count": 0}

        total_distance = sum(p.total_distance for p in team_players)
        total_sprints = sum(p.sprint_count for p in team_players)
        total_hi_distance = sum(p.high_intensity_distance for p in team_players)

        max_speeds = [p.max_speed for p in team_players if p.max_speed > 0]

        return {
            "team": team,
            "player_count": len(team_players),
            "total_distance_m": round(total_distance, 1),
            "total_distance_km": round(total_distance / 1000, 2),
            "avg_distance_per_player_m": round(total_distance / len(team_players), 1),
            "total_sprints": total_sprints,
            "avg_sprints_per_player": round(total_sprints / len(team_players), 1),
            "total_high_intensity_m": round(total_hi_distance, 1),
            "team_max_speed_ms": round(max(max_speeds) if max_speeds else 0, 2),
            "team_max_speed_kmh": round(max(max_speeds) * 3.6 if max_speeds else 0, 1),
            "players": [p.to_dict() for p in sorted(team_players, key=lambda x: -x.total_distance)]
        }

    def get_sprint_leaders(self, top_n: int = 5) -> List[Dict]:
        """Get players with most sprints."""
        sorted_players = sorted(
            self.player_stats.values(),
            key=lambda x: x.sprint_count,
            reverse=True
        )[:top_n]

        return [
            {
                "player_id": p.player_id,
                "team": p.team,
                "jersey_number": p.jersey_number,
                "sprint_count": p.sprint_count,
                "total_sprint_distance_m": round(p.total_sprint_distance, 1),
                "max_speed_kmh": round(p.max_speed * 3.6, 1)
            }
            for p in sorted_players
        ]

    def get_distance_leaders(self, top_n: int = 5) -> List[Dict]:
        """Get players with most distance covered."""
        sorted_players = sorted(
            self.player_stats.values(),
            key=lambda x: x.total_distance,
            reverse=True
        )[:top_n]

        return [
            {
                "player_id": p.player_id,
                "team": p.team,
                "jersey_number": p.jersey_number,
                "total_distance_m": round(p.total_distance, 1),
                "total_distance_km": round(p.total_distance / 1000, 2),
                "distance_per_minute": round(p.distance_per_minute, 1)
            }
            for p in sorted_players
        ]

    def get_speed_leaders(self, top_n: int = 5) -> List[Dict]:
        """Get players with highest max speeds."""
        sorted_players = sorted(
            self.player_stats.values(),
            key=lambda x: x.max_speed,
            reverse=True
        )[:top_n]

        return [
            {
                "player_id": p.player_id,
                "team": p.team,
                "jersey_number": p.jersey_number,
                "max_speed_ms": round(p.max_speed, 2),
                "max_speed_kmh": round(p.max_speed * 3.6, 1),
                "avg_speed_kmh": round(p.avg_speed * 3.6, 1)
            }
            for p in sorted_players
        ]

    def get_all_sprints(self) -> List[Dict]:
        """Get all sprints from all players."""
        all_sprints = []
        for stats in self.player_stats.values():
            for sprint in stats.sprints:
                sprint_dict = sprint.to_dict()
                sprint_dict["jersey_number"] = stats.jersey_number
                all_sprints.append(sprint_dict)

        return sorted(all_sprints, key=lambda x: x["start_time_ms"])

    def get_recent_sprints(self, last_n: int = 10) -> List[Dict]:
        """Get most recent sprints."""
        return self.get_all_sprints()[-last_n:]

    def get_full_analysis(self) -> Dict:
        """Get complete movement analysis."""
        total_time_s = self.total_time_ms / 1000

        # Update distance per minute for all players
        for stats in self.player_stats.values():
            if total_time_s > 0:
                stats.distance_per_minute = (stats.total_distance / total_time_s) * 60

        return {
            "summary": {
                "total_time_ms": self.total_time_ms,
                "total_time_minutes": round(total_time_s / 60, 1),
                "total_players_tracked": len(self.player_stats),
                "total_sprints_detected": sum(p.sprint_count for p in self.player_stats.values()),
                "total_distance_all_players_m": round(sum(p.total_distance for p in self.player_stats.values()), 1)
            },
            "teams": {
                "home": self.get_team_stats("home"),
                "away": self.get_team_stats("away")
            },
            "leaders": {
                "sprint_leaders": self.get_sprint_leaders(),
                "distance_leaders": self.get_distance_leaders(),
                "speed_leaders": self.get_speed_leaders()
            },
            "recent_sprints": self.get_recent_sprints()
        }

    def reset(self):
        """Reset analyzer for new match."""
        self.player_stats.clear()
        self.active_sprints.clear()
        self.prev_positions.clear()
        self.current_frame = 0
        self.start_time_ms = 0
        self.total_time_ms = 0


# Global instance
player_movement_analyzer = PlayerMovementAnalyzer()
