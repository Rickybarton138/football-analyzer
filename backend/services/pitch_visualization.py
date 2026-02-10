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

        # Goalkeeper tracking - identified by position near goal or jersey #1
        self.goalkeepers: Dict[str, int] = {}  # team -> jersey_number
        self.goalkeeper_positions: Dict[str, List[Tuple[float, float, int]]] = {
            "home": [],  # (x, y, frame_number)
            "away": []
        }

        # Continuous tracking for specific players by jersey number
        # Stores position history for each (team, jersey_number)
        self.tracked_positions: Dict[Tuple[str, int], List[Tuple[float, float, int]]] = {}

        # Standard football positions by jersey number (typical assignments)
        self.position_names = {
            1: "Goalkeeper",
            2: "Right Back",
            3: "Left Back",
            4: "Center Back",
            5: "Center Back",
            6: "Defensive Midfielder",
            7: "Right Winger",
            8: "Central Midfielder",
            9: "Striker",
            10: "Attacking Midfielder",
            11: "Left Winger"
        }

        # Video info
        self.fps: float = 30.0
        self.total_frames: int = 0

    def set_video_info(self, fps: float, total_frames: int):
        """Set video metadata."""
        self.fps = fps
        self.total_frames = total_frames

    def _is_goalkeeper_position(self, team: str, x: float, y: float) -> bool:
        """Check if position is in goalkeeper area (near goal)."""
        # Home goalkeeper typically on left (x < 15), away on right (x > 85)
        # Also check y is centered (goal area is roughly 30-70 on y axis)
        if team == "home":
            return x < 18 and 20 < y < 80
        else:  # away
            return x > 82 and 20 < y < 80

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

        # Continuous tracking for all players by jersey number
        key = (team, jersey_number)
        if key not in self.tracked_positions:
            self.tracked_positions[key] = []

        self.tracked_positions[key].append((x, y, frame_number))
        # Keep last 500 positions per player for movement analysis
        if len(self.tracked_positions[key]) > 500:
            self.tracked_positions[key] = self.tracked_positions[key][-500:]

        # Identify and track goalkeeper
        # Goalkeeper is jersey #1 OR player consistently in goal area
        is_gk_position = self._is_goalkeeper_position(team, x, y)

        if jersey_number == 1 or is_gk_position:
            # If no goalkeeper identified yet, or this is jersey #1, set as goalkeeper
            if team not in self.goalkeepers or jersey_number == 1:
                self.goalkeepers[team] = jersey_number

            # Track goalkeeper movement if this is the identified goalkeeper
            if self.goalkeepers.get(team) == jersey_number:
                self.goalkeeper_positions[team].append((x, y, frame_number))
                # Keep only last 100 positions for memory efficiency
                if len(self.goalkeeper_positions[team]) > 100:
                    self.goalkeeper_positions[team] = self.goalkeeper_positions[team][-100:]

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

    def _filter_on_pitch_players(self, player_list: List[Dict], max_players: int = 11) -> List[Dict]:
        """
        Filter players to only those on the pitch and limit to max per team.

        Players must be within pitch bounds (5-95 for x and y to exclude sidelines).
        Returns the most recently active players up to max_players.
        """
        # Filter to players within pitch bounds (exclude dugouts, sidelines, etc.)
        on_pitch = [
            p for p in player_list
            if 3 <= p.get("x", 0) <= 97 and 3 <= p.get("y", 0) <= 97
        ]

        # Sort by jersey number to get consistent ordering
        on_pitch.sort(key=lambda p: p.get("jersey_number", 99))

        # Limit to max players per team (11 for football)
        return on_pitch[:max_players]

    def get_2d_radar_state(self, frame_number: Optional[int] = None) -> Dict:
        """
        Get the 2D radar state at a specific frame or current state.

        Returns player positions (max 11 per team), ball position, and referee positions.
        Filters out players outside pitch bounds (coaches, subs, fans).
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
                if team in ["home", "away"]:
                    players[team].append({
                        "jersey_number": jersey,
                        "x": data["x"],
                        "y": data["y"],
                        "has_ball": data["has_ball"]
                    })

            # Filter and limit to 11 players per team
            players["home"] = self._filter_on_pitch_players(players["home"], max_players=11)
            players["away"] = self._filter_on_pitch_players(players["away"], max_players=11)

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
                if team in ["home", "away"]:
                    players[team].append({
                        "jersey_number": jersey,
                        "x": x,
                        "y": y,
                        "position": self.position_names.get(jersey, f"Player #{jersey}"),
                        "is_goalkeeper": self.goalkeepers.get(team) == jersey
                    })

            # Filter and limit to 11 players per team
            players["home"] = self._filter_on_pitch_players(players["home"], max_players=11)
            players["away"] = self._filter_on_pitch_players(players["away"], max_players=11)

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

    def get_player_continuous_tracking(self, team: str, jersey_number: int) -> Dict:
        """
        Get continuous tracking data for a specific player.

        Returns all recorded positions for the player, useful for
        tracking specific positions like goalkeeper (#1) or right back (#2).

        Args:
            team: 'home' or 'away'
            jersey_number: Player's jersey number

        Returns:
            Dictionary with player info and position history
        """
        key = (team, jersey_number)
        positions = self.tracked_positions.get(key, [])

        # Calculate stats from tracking data
        if positions:
            xs = [p[0] for p in positions]
            ys = [p[1] for p in positions]
            avg_x = sum(xs) / len(xs)
            avg_y = sum(ys) / len(ys)

            # Calculate distance covered (sum of distances between consecutive positions)
            total_distance = 0
            for i in range(1, len(positions)):
                dx = positions[i][0] - positions[i-1][0]
                dy = positions[i][1] - positions[i-1][1]
                total_distance += (dx**2 + dy**2) ** 0.5
        else:
            avg_x, avg_y, total_distance = 0, 0, 0

        return {
            "team": team,
            "jersey_number": jersey_number,
            "position_name": self.position_names.get(jersey_number, f"Player #{jersey_number}"),
            "is_goalkeeper": self.goalkeepers.get(team) == jersey_number,
            "total_samples": len(positions),
            "average_position": {"x": round(avg_x, 1), "y": round(avg_y, 1)},
            "distance_covered_units": round(total_distance, 1),
            "current_position": {"x": positions[-1][0], "y": positions[-1][1]} if positions else None,
            "position_history": [
                {"x": p[0], "y": p[1], "frame": p[2]}
                for p in positions[-100:]  # Return last 100 positions
            ]
        }

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

    def generate_pressure_map(
        self,
        team: str,
        start_frame: Optional[int] = None,
        end_frame: Optional[int] = None
    ) -> Dict:
        """
        Generate a pressure map showing where a team applies pressing.

        Pressure is calculated based on player density around the ball
        when the opposing team has possession.
        """
        grid = np.zeros((self.HEATMAP_GRID_Y, self.HEATMAP_GRID_X))

        cell_width = self.PITCH_LENGTH / self.HEATMAP_GRID_X
        cell_height = self.PITCH_WIDTH / self.HEATMAP_GRID_Y

        # Get ball positions and nearby pressing players
        for ball in self.ball_positions:
            if start_frame and ball.frame_number < start_frame:
                continue
            if end_frame and ball.frame_number > end_frame:
                continue

            # Count pressing players near the ball
            pressing_count = 0
            for player in self.player_positions:
                if player.frame_number == ball.frame_number and player.team == team:
                    dist = np.sqrt((player.x - ball.x)**2 + (player.y - ball.y)**2)
                    if dist < 15:  # Within pressing distance
                        pressing_count += 1

            if pressing_count >= 2:  # At least 2 players pressing
                grid_x = int(ball.x / cell_width)
                grid_y = int(ball.y / cell_height)
                grid_x = max(0, min(self.HEATMAP_GRID_X - 1, grid_x))
                grid_y = max(0, min(self.HEATMAP_GRID_Y - 1, grid_y))
                grid[grid_y, grid_x] += pressing_count

        max_val = grid.max()
        if max_val > 0:
            normalized_grid = (grid / max_val).tolist()
        else:
            normalized_grid = grid.tolist()

        return {
            "grid": normalized_grid,
            "max_value": float(max_val),
            "team": team,
            "type": "pressure_map",
            "grid_dimensions": {
                "x": self.HEATMAP_GRID_X,
                "y": self.HEATMAP_GRID_Y
            }
        }

    def generate_shot_map(self, shots: List[Dict]) -> Dict:
        """
        Generate shot map data for visualization.

        Args:
            shots: List of shot dictionaries with x, y, xg, is_goal, team, etc.

        Returns:
            Formatted shot map data for frontend
        """
        home_shots = []
        away_shots = []

        for shot in shots:
            shot_data = {
                "x": shot.get("x", 0),
                "y": shot.get("y", 0),
                "xg": shot.get("xg", 0),
                "is_goal": shot.get("is_goal", False),
                "on_target": shot.get("on_target", False),
                "player": shot.get("player_jersey") or shot.get("player"),
                "timestamp_ms": shot.get("timestamp_ms", 0),
                "shot_type": shot.get("shot_type", "foot"),
                # Size based on xG for visualization
                "size": max(8, min(30, shot.get("xg", 0.1) * 60))
            }

            if shot.get("team") == "home":
                home_shots.append(shot_data)
            else:
                away_shots.append(shot_data)

        # Calculate totals
        home_xg = sum(s["xg"] for s in home_shots)
        away_xg = sum(s["xg"] for s in away_shots)
        home_goals = sum(1 for s in home_shots if s["is_goal"])
        away_goals = sum(1 for s in away_shots if s["is_goal"])

        return {
            "home": {
                "shots": home_shots,
                "total_shots": len(home_shots),
                "total_xg": round(home_xg, 2),
                "goals": home_goals,
                "xg_per_shot": round(home_xg / max(1, len(home_shots)), 3)
            },
            "away": {
                "shots": away_shots,
                "total_shots": len(away_shots),
                "total_xg": round(away_xg, 2),
                "goals": away_goals,
                "xg_per_shot": round(away_xg / max(1, len(away_shots)), 3)
            },
            "comparison": {
                "xg_difference": round(home_xg - away_xg, 2),
                "shot_difference": len(home_shots) - len(away_shots),
                "home_conversion": round(home_goals / max(1, len(home_shots)) * 100, 1),
                "away_conversion": round(away_goals / max(1, len(away_shots)) * 100, 1)
            }
        }

    def get_defensive_line_data(
        self,
        team: str,
        frame_number: Optional[int] = None
    ) -> Dict:
        """
        Get defensive line position data for visualization.

        Returns the positions of the back line players.
        """
        if frame_number is not None:
            positions = [
                p for p in self.player_positions
                if p.team == team and p.frame_number == frame_number
            ]
        else:
            # Use current positions
            positions = [
                {"jersey": j, "x": pos[0], "y": pos[1]}
                for (t, j), pos in self.current_player_positions.items()
                if t == team
            ]

        if not positions:
            return {"team": team, "line": None, "players": []}

        # Sort by x position to find defensive line
        if team == "home":
            # Home defends left (lower x values)
            sorted_pos = sorted(positions, key=lambda p: p.x if hasattr(p, 'x') else p.get('x', 50))
        else:
            # Away defends right (higher x values)
            sorted_pos = sorted(positions, key=lambda p: -(p.x if hasattr(p, 'x') else p.get('x', 50)))

        # Get back 4 players (excluding obvious goalkeeper at extreme)
        defenders = sorted_pos[1:5] if len(sorted_pos) > 4 else sorted_pos

        if not defenders:
            return {"team": team, "line": None, "players": []}

        # Calculate line metrics
        x_positions = [p.x if hasattr(p, 'x') else p.get('x', 50) for p in defenders]
        y_positions = [p.y if hasattr(p, 'y') else p.get('y', 50) for p in defenders]

        return {
            "team": team,
            "line": {
                "avg_x": round(np.mean(x_positions), 1),
                "min_x": round(min(x_positions), 1),
                "max_x": round(max(x_positions), 1),
                "spread": round(max(x_positions) - min(x_positions), 1),
                "is_flat": max(x_positions) - min(x_positions) < 5
            },
            "players": [
                {
                    "jersey": p.jersey_number if hasattr(p, 'jersey_number') else p.get('jersey'),
                    "x": p.x if hasattr(p, 'x') else p.get('x'),
                    "y": p.y if hasattr(p, 'y') else p.get('y')
                }
                for p in defenders
            ],
            "offside_line": round(min(x_positions), 1) if team == "home" else round(max(x_positions), 1)
        }

    def get_passing_network_data(
        self,
        team: str,
        passes: List[Dict]
    ) -> Dict:
        """
        Generate passing network visualization data.

        Args:
            team: Team to generate network for
            passes: List of pass events with from_player, to_player, x, y, end_x, end_y

        Returns:
            Network data with nodes (players) and edges (passes)
        """
        from collections import defaultdict

        team_passes = [p for p in passes if p.get("team") == team]

        # Count passes between each pair
        pass_counts = defaultdict(int)
        player_passes = defaultdict(int)
        player_positions = defaultdict(list)

        for p in team_passes:
            from_player = p.get("from_player") or p.get("player_jersey")
            to_player = p.get("to_player")

            if from_player and to_player:
                key = (min(from_player, to_player), max(from_player, to_player))
                pass_counts[key] += 1
                player_passes[from_player] += 1

            # Track positions
            if from_player and p.get("x") is not None and p.get("y") is not None:
                player_positions[from_player].append((p["x"], p["y"]))

        # Calculate average positions for nodes
        nodes = []
        for player, positions in player_positions.items():
            if positions:
                avg_x = np.mean([p[0] for p in positions])
                avg_y = np.mean([p[1] for p in positions])
                nodes.append({
                    "id": player,
                    "jersey_number": player,
                    "x": round(avg_x, 1),
                    "y": round(avg_y, 1),
                    "passes": player_passes[player],
                    "size": max(20, min(50, player_passes[player] * 2))
                })

        # Create edges
        edges = []
        max_passes = max(pass_counts.values()) if pass_counts else 1
        for (p1, p2), count in pass_counts.items():
            if count >= 2:  # Only show connections with 2+ passes
                edges.append({
                    "source": p1,
                    "target": p2,
                    "weight": count,
                    "thickness": max(1, min(8, (count / max_passes) * 8))
                })

        return {
            "team": team,
            "nodes": nodes,
            "edges": edges,
            "total_passes": len(team_passes),
            "unique_connections": len(edges)
        }

    def get_sprint_map_data(self, sprints: List[Dict]) -> Dict:
        """
        Generate sprint visualization data.

        Args:
            sprints: List of sprint events with start/end positions

        Returns:
            Sprint map data for visualization
        """
        home_sprints = []
        away_sprints = []

        for sprint in sprints:
            sprint_data = {
                "start_x": sprint.get("start_position", (0, 0))[0] if isinstance(sprint.get("start_position"), tuple) else sprint.get("start_x", 0),
                "start_y": sprint.get("start_position", (0, 0))[1] if isinstance(sprint.get("start_position"), tuple) else sprint.get("start_y", 0),
                "end_x": sprint.get("end_position", (0, 0))[0] if isinstance(sprint.get("end_position"), tuple) else sprint.get("end_x", 0),
                "end_y": sprint.get("end_position", (0, 0))[1] if isinstance(sprint.get("end_position"), tuple) else sprint.get("end_y", 0),
                "max_speed_kmh": sprint.get("max_speed_kmh", 0),
                "distance_m": sprint.get("distance_m", 0),
                "player_id": sprint.get("player_id"),
                "jersey_number": sprint.get("jersey_number"),
                "timestamp_ms": sprint.get("start_time_ms", 0)
            }

            if sprint.get("team") == "home":
                home_sprints.append(sprint_data)
            else:
                away_sprints.append(sprint_data)

        return {
            "home": {
                "sprints": home_sprints,
                "total_sprints": len(home_sprints),
                "avg_speed": round(np.mean([s["max_speed_kmh"] for s in home_sprints]), 1) if home_sprints else 0,
                "total_distance": round(sum(s["distance_m"] for s in home_sprints), 1)
            },
            "away": {
                "sprints": away_sprints,
                "total_sprints": len(away_sprints),
                "avg_speed": round(np.mean([s["max_speed_kmh"] for s in away_sprints]), 1) if away_sprints else 0,
                "total_distance": round(sum(s["distance_m"] for s in away_sprints), 1)
            }
        }

    def reset(self):
        """Reset all visualization data."""
        self.player_positions.clear()
        self.ball_positions.clear()
        self.current_player_positions.clear()
        self.current_ball_position = None
        self.current_frame = 0
        self.current_timestamp_ms = 0
        self.goalkeepers.clear()
        self.goalkeeper_positions = {"home": [], "away": []}
        if hasattr(self, 'tracked_positions'):
            self.tracked_positions.clear()


# Global instance
pitch_visualization_service = PitchVisualizationService()
