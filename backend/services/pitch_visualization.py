"""
Pitch Visualization Service

Generates professional pitch visualizations including:
- Heatmaps (team and individual player positioning)
- 2D Radar (real-time player/ball positions)
- Pass maps (pass locations and connections)
- Shot maps (shot positions with xG)

Similar to VEO Analytics visualizations.
"""
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import json


@dataclass
class PlayerPosition:
    """A player position sample."""
    team: str
    jersey_number: int
    x: float  # 0-100 pitch coordinates
    y: float  # 0-100 pitch coordinates
    frame_number: int
    timestamp_ms: int
    has_ball: bool = False


@dataclass
class BallPosition:
    """Ball position at a frame."""
    x: float
    y: float
    frame_number: int
    timestamp_ms: int


@dataclass
class HeatmapData:
    """Heatmap data for a team or player."""
    grid: np.ndarray  # 2D array of intensity values
    max_value: float
    total_samples: int


class PitchVisualizationService:
    """
    Service for generating pitch visualizations.

    Provides heatmaps, 2D radar data, and other visualizations
    for professional football analysis.
    """

    # Standard pitch dimensions (normalized to 0-100)
    PITCH_LENGTH = 100
    PITCH_WIDTH = 100

    # Heatmap grid resolution
    HEATMAP_GRID_X = 20  # Cells along length
    HEATMAP_GRID_Y = 15  # Cells along width

    # Goal dimensions (scaled)
    GOAL_WIDTH = 7.32
    GOAL_HEIGHT = 2.44

    def __init__(self):
        # Position tracking
        self.player_positions: List[PlayerPosition] = []
        self.ball_positions: List[BallPosition] = []

        # Real-time state (for 2D radar)
        self.current_player_positions: Dict[Tuple[str, int], Tuple[float, float]] = {}
        self.current_ball_position: Optional[Tuple[float, float]] = None
        self.current_frame: int = 0
        self.current_timestamp_ms: int = 0

        # Video info
        self.fps: float = 30.0
        self.total_frames: int = 0

    def set_video_info(self, fps: float, total_frames: int):
        """Set video metadata."""
        self.fps = fps
        self.total_frames = total_frames

    def record_player_position(
        self,
        team: str,
        jersey_number: int,
        x: float,
        y: float,
        frame_number: int,
        timestamp_ms: int,
        has_ball: bool = False
    ):
        """Record a player position sample."""
        # Normalize coordinates to 0-100 if needed
        x = max(0, min(100, x))
        y = max(0, min(100, y))

        pos = PlayerPosition(
            team=team,
            jersey_number=jersey_number,
            x=x,
            y=y,
            frame_number=frame_number,
            timestamp_ms=timestamp_ms,
            has_ball=has_ball
        )
        self.player_positions.append(pos)

        # Update current state for 2D radar
        self.current_player_positions[(team, jersey_number)] = (x, y)
        self.current_frame = frame_number
        self.current_timestamp_ms = timestamp_ms

    def record_ball_position(
        self,
        x: float,
        y: float,
        frame_number: int,
        timestamp_ms: int
    ):
        """Record ball position."""
        x = max(0, min(100, x))
        y = max(0, min(100, y))

        pos = BallPosition(
            x=x,
            y=y,
            frame_number=frame_number,
            timestamp_ms=timestamp_ms
        )
        self.ball_positions.append(pos)

        self.current_ball_position = (x, y)
        self.current_frame = frame_number
        self.current_timestamp_ms = timestamp_ms

    def generate_team_heatmap(
        self,
        team: str,
        start_frame: Optional[int] = None,
        end_frame: Optional[int] = None
    ) -> Dict:
        """
        Generate a heatmap for a team's positioning.

        Returns grid data and metadata for visualization.
        """
        # Filter positions by team and frame range
        positions = [
            p for p in self.player_positions
            if p.team == team
            and (start_frame is None or p.frame_number >= start_frame)
            and (end_frame is None or p.frame_number <= end_frame)
        ]

        if not positions:
            return {
                "grid": [[0] * self.HEATMAP_GRID_X for _ in range(self.HEATMAP_GRID_Y)],
                "max_value": 0,
                "total_samples": 0,
                "team": team
            }

        # Create grid
        grid = np.zeros((self.HEATMAP_GRID_Y, self.HEATMAP_GRID_X))

        cell_width = self.PITCH_LENGTH / self.HEATMAP_GRID_X
        cell_height = self.PITCH_WIDTH / self.HEATMAP_GRID_Y

        for pos in positions:
            # Map position to grid cell
            grid_x = int(pos.x / cell_width)
            grid_y = int(pos.y / cell_height)

            # Clamp to valid range
            grid_x = max(0, min(self.HEATMAP_GRID_X - 1, grid_x))
            grid_y = max(0, min(self.HEATMAP_GRID_Y - 1, grid_y))

            grid[grid_y, grid_x] += 1

        # Normalize to 0-1 range
        max_val = grid.max()
        if max_val > 0:
            normalized_grid = (grid / max_val).tolist()
        else:
            normalized_grid = grid.tolist()

        return {
            "grid": normalized_grid,
            "max_value": float(max_val),
            "total_samples": len(positions),
            "team": team,
            "grid_dimensions": {
                "x": self.HEATMAP_GRID_X,
                "y": self.HEATMAP_GRID_Y
            }
        }

    def generate_player_heatmap(
        self,
        team: str,
        jersey_number: int,
        start_frame: Optional[int] = None,
        end_frame: Optional[int] = None
    ) -> Dict:
        """Generate a heatmap for an individual player."""
        positions = [
            p for p in self.player_positions
            if p.team == team
            and p.jersey_number == jersey_number
            and (start_frame is None or p.frame_number >= start_frame)
            and (end_frame is None or p.frame_number <= end_frame)
        ]

        if not positions:
            return {
                "grid": [[0] * self.HEATMAP_GRID_X for _ in range(self.HEATMAP_GRID_Y)],
                "max_value": 0,
                "total_samples": 0,
                "team": team,
                "jersey_number": jersey_number
            }

        grid = np.zeros((self.HEATMAP_GRID_Y, self.HEATMAP_GRID_X))

        cell_width = self.PITCH_LENGTH / self.HEATMAP_GRID_X
        cell_height = self.PITCH_WIDTH / self.HEATMAP_GRID_Y

        for pos in positions:
            grid_x = int(pos.x / cell_width)
            grid_y = int(pos.y / cell_height)
            grid_x = max(0, min(self.HEATMAP_GRID_X - 1, grid_x))
            grid_y = max(0, min(self.HEATMAP_GRID_Y - 1, grid_y))
            grid[grid_y, grid_x] += 1

        max_val = grid.max()
        if max_val > 0:
            normalized_grid = (grid / max_val).tolist()
        else:
            normalized_grid = grid.tolist()

        return {
            "grid": normalized_grid,
            "max_value": float(max_val),
            "total_samples": len(positions),
            "team": team,
            "jersey_number": jersey_number,
            "grid_dimensions": {
                "x": self.HEATMAP_GRID_X,
                "y": self.HEATMAP_GRID_Y
            }
        }

    def get_2d_radar_state(self, frame_number: Optional[int] = None) -> Dict:
        """
        Get the 2D radar state at a specific frame or current state.

        Returns player positions, ball position, and connection lines.
        """
        if frame_number is not None:
            # Find positions at specific frame
            frame_positions = {}
            ball_pos = None

            for pos in self.player_positions:
                if pos.frame_number == frame_number:
                    frame_positions[(pos.team, pos.jersey_number)] = {
                        "x": pos.x,
                        "y": pos.y,
                        "has_ball": pos.has_ball
                    }

            for ball in self.ball_positions:
                if ball.frame_number == frame_number:
                    ball_pos = {"x": ball.x, "y": ball.y}
                    break

            players = {
                "home": [],
                "away": []
            }

            for (team, jersey), data in frame_positions.items():
                players[team].append({
                    "jersey_number": jersey,
                    "x": data["x"],
                    "y": data["y"],
                    "has_ball": data["has_ball"]
                })

            return {
                "frame_number": frame_number,
                "players": players,
                "ball": ball_pos
            }
        else:
            # Return current state
            players = {
                "home": [],
                "away": []
            }

            for (team, jersey), (x, y) in self.current_player_positions.items():
                players[team].append({
                    "jersey_number": jersey,
                    "x": x,
                    "y": y
                })

            return {
                "frame_number": self.current_frame,
                "timestamp_ms": self.current_timestamp_ms,
                "players": players,
                "ball": {"x": self.current_ball_position[0], "y": self.current_ball_position[1]}
                        if self.current_ball_position else None
            }

    def get_player_trail(
        self,
        team: str,
        jersey_number: int,
        start_frame: int,
        end_frame: int
    ) -> List[Dict]:
        """
        Get player movement trail between frames.

        Used for showing player runs on 2D radar.
        """
        positions = [
            {"x": p.x, "y": p.y, "frame": p.frame_number, "timestamp_ms": p.timestamp_ms}
            for p in self.player_positions
            if p.team == team
            and p.jersey_number == jersey_number
            and p.frame_number >= start_frame
            and p.frame_number <= end_frame
        ]

        return sorted(positions, key=lambda x: x["frame"])

    def get_possession_zones(
        self,
        team: str,
        start_frame: Optional[int] = None,
        end_frame: Optional[int] = None
    ) -> Dict:
        """
        Get possession zone breakdown.

        Returns percentage of possession in each third of the pitch.
        """
        positions = [
            p for p in self.player_positions
            if p.team == team
            and p.has_ball
            and (start_frame is None or p.frame_number >= start_frame)
            and (end_frame is None or p.frame_number <= end_frame)
        ]

        if not positions:
            return {
                "defensive_third": 0,
                "middle_third": 0,
                "attacking_third": 0,
                "total_samples": 0
            }

        defensive = 0
        middle = 0
        attacking = 0

        for pos in positions:
            if pos.x < 33.33:
                defensive += 1
            elif pos.x < 66.67:
                middle += 1
            else:
                attacking += 1

        total = len(positions)

        return {
            "defensive_third": round(defensive / total * 100, 1),
            "middle_third": round(middle / total * 100, 1),
            "attacking_third": round(attacking / total * 100, 1),
            "total_samples": total
        }

    def get_average_positions(self, team: str) -> List[Dict]:
        """
        Calculate average position for each player on a team.

        Used for formation display.
        """
        player_positions_sum = defaultdict(lambda: {"x": 0, "y": 0, "count": 0})

        for pos in self.player_positions:
            if pos.team == team:
                key = pos.jersey_number
                player_positions_sum[key]["x"] += pos.x
                player_positions_sum[key]["y"] += pos.y
                player_positions_sum[key]["count"] += 1

        result = []
        for jersey, data in player_positions_sum.items():
            if data["count"] > 0:
                result.append({
                    "jersey_number": jersey,
                    "x": round(data["x"] / data["count"], 1),
                    "y": round(data["y"] / data["count"], 1),
                    "samples": data["count"]
                })

        return sorted(result, key=lambda x: x["jersey_number"])

    def get_team_shape(self, team: str, frame_number: int) -> Dict:
        """
        Get team shape at a specific frame.

        Returns player positions with connection lines.
        """
        positions = [
            {"jersey": p.jersey_number, "x": p.x, "y": p.y}
            for p in self.player_positions
            if p.team == team and p.frame_number == frame_number
        ]

        if not positions:
            return {"players": [], "connections": [], "centroid": None}

        # Calculate team centroid
        avg_x = sum(p["x"] for p in positions) / len(positions)
        avg_y = sum(p["y"] for p in positions) / len(positions)

        # Calculate team width and depth
        xs = [p["x"] for p in positions]
        ys = [p["y"] for p in positions]

        return {
            "players": positions,
            "centroid": {"x": round(avg_x, 1), "y": round(avg_y, 1)},
            "width": round(max(ys) - min(ys), 1) if ys else 0,
            "depth": round(max(xs) - min(xs), 1) if xs else 0,
            "compactness": self._calculate_compactness(positions)
        }

    def _calculate_compactness(self, positions: List[Dict]) -> float:
        """Calculate team compactness (average distance between players)."""
        if len(positions) < 2:
            return 0

        total_distance = 0
        pairs = 0

        for i, p1 in enumerate(positions):
            for p2 in positions[i+1:]:
                dist = np.sqrt((p1["x"] - p2["x"])**2 + (p1["y"] - p2["y"])**2)
                total_distance += dist
                pairs += 1

        return round(total_distance / pairs, 1) if pairs > 0 else 0

    def get_distance_between_players(
        self,
        team: str,
        jersey1: int,
        jersey2: int,
        frame_number: int
    ) -> Optional[float]:
        """Get distance between two players at a frame (for connect players feature)."""
        p1 = None
        p2 = None

        for pos in self.player_positions:
            if pos.team == team and pos.frame_number == frame_number:
                if pos.jersey_number == jersey1:
                    p1 = (pos.x, pos.y)
                elif pos.jersey_number == jersey2:
                    p2 = (pos.x, pos.y)

        if p1 and p2:
            # Convert to yards (assuming 100 = 100 meters ~ 109 yards)
            dist = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
            return round(dist * 1.09, 1)  # Convert to yards

        return None

    def export_visualization_data(self) -> Dict:
        """Export all visualization data for frontend rendering."""
        return {
            "heatmaps": {
                "home": self.generate_team_heatmap("home"),
                "away": self.generate_team_heatmap("away")
            },
            "average_positions": {
                "home": self.get_average_positions("home"),
                "away": self.get_average_positions("away")
            },
            "possession_zones": {
                "home": self.get_possession_zones("home"),
                "away": self.get_possession_zones("away")
            },
            "total_frames": self.total_frames,
            "fps": self.fps
        }

    def reset(self):
        """Reset all visualization data."""
        self.player_positions.clear()
        self.ball_positions.clear()
        self.current_player_positions.clear()
        self.current_ball_position = None
        self.current_frame = 0
        self.current_timestamp_ms = 0


# Global instance
pitch_visualization_service = PitchVisualizationService()
