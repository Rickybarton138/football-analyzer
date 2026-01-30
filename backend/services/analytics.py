"""
Analytics Engine

Calculates player and team metrics, generates heatmaps,
pass networks, and other analytical data.
"""
import numpy as np
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass
import json
import aiofiles

from config import settings
from models.schemas import (
    PlayerMetrics, TeamMetrics, HeatmapData, PassNetwork,
    Position, TeamSide, DetectedPlayer, DetectedBall,
    MatchEvent, EventType, MatchReport, MatchInfo,
    TeamImprovementReport, TeamImprovementArea, PlayerImprovementSuggestion,
    OpponentScoutReport, PlayerScoutReport, TeamWeakness, TeamPattern
)


@dataclass
class PlayerState:
    """Running state for a player during analysis."""
    track_id: int
    team: TeamSide
    positions: List[Tuple[int, Position]]  # (timestamp, position)
    speeds: List[float]
    touches: int = 0
    passes_attempted: int = 0
    passes_completed: int = 0
    shots: int = 0
    tackles: int = 0
    interceptions: int = 0


class AnalyticsEngine:
    """
    Engine for calculating match analytics and statistics.

    Provides:
    - Player metrics (distance, sprints, passes, etc.)
    - Team metrics (possession, pass completion, xG)
    - Heatmaps (player and team)
    - Pass networks
    - Formation detection
    """

    def __init__(self):
        self.player_states: Dict[int, PlayerState] = {}
        self.team_states: Dict[TeamSide, Dict] = {
            TeamSide.HOME: self._init_team_state(),
            TeamSide.AWAY: self._init_team_state()
        }
        self.events: List[MatchEvent] = []

        # Configuration
        self.sprint_threshold = settings.SPRINT_THRESHOLD_KMH
        self.high_intensity_threshold = settings.HIGH_INTENSITY_THRESHOLD_KMH
        self.heatmap_grid_size = (10, 7)  # Divisions along length x width

    def _init_team_state(self) -> Dict:
        """Initialize team state dictionary."""
        return {
            "possession_time_ms": 0,
            "passes_attempted": 0,
            "passes_completed": 0,
            "shots": 0,
            "shots_on_target": 0,
            "xg": 0.0,
            "formations": []
        }

    async def process_frame(
        self,
        players: List[DetectedPlayer],
        ball: Optional[DetectedBall],
        timestamp_ms: int,
        events: List[MatchEvent] = None
    ):
        """
        Process a frame and update analytics.

        Args:
            players: Detected players with pitch positions
            ball: Detected ball
            timestamp_ms: Current timestamp
            events: Events detected in this frame
        """
        # Update player states
        for player in players:
            if player.pitch_position is None:
                continue

            await self._update_player_state(player, timestamp_ms)

        # Process events
        if events:
            for event in events:
                self.events.append(event)
                await self._process_event(event)

    async def _update_player_state(
        self,
        player: DetectedPlayer,
        timestamp_ms: int
    ):
        """Update running statistics for a player."""
        track_id = player.track_id

        if track_id not in self.player_states:
            self.player_states[track_id] = PlayerState(
                track_id=track_id,
                team=player.team,
                positions=[],
                speeds=[]
            )

        state = self.player_states[track_id]

        # Calculate speed if we have previous position
        speed = 0.0
        if state.positions:
            prev_time, prev_pos = state.positions[-1]
            time_delta_s = (timestamp_ms - prev_time) / 1000

            if time_delta_s > 0:
                distance = self._calculate_distance(prev_pos, player.pitch_position)
                speed = (distance / time_delta_s) * 3.6  # km/h

        # Store position and speed
        state.positions.append((timestamp_ms, player.pitch_position))
        state.speeds.append(speed)

    async def _process_event(self, event: MatchEvent):
        """Process an event and update statistics."""
        if event.team is None:
            return

        team_state = self.team_states[event.team]

        if event.event_type == EventType.PASS:
            team_state["passes_attempted"] += 1
            if event.success:
                team_state["passes_completed"] += 1

            if event.player_id and event.player_id in self.player_states:
                self.player_states[event.player_id].passes_attempted += 1
                if event.success:
                    self.player_states[event.player_id].passes_completed += 1

        elif event.event_type == EventType.SHOT:
            team_state["shots"] += 1

            # Calculate basic xG based on position
            xg = self._calculate_xg(event.position)
            team_state["xg"] += xg

            if event.player_id and event.player_id in self.player_states:
                self.player_states[event.player_id].shots += 1

        elif event.event_type == EventType.TACKLE:
            if event.player_id and event.player_id in self.player_states:
                self.player_states[event.player_id].tackles += 1

        elif event.event_type == EventType.INTERCEPTION:
            if event.player_id and event.player_id in self.player_states:
                self.player_states[event.player_id].interceptions += 1

    def _calculate_xg(self, position: Position) -> float:
        """
        Calculate basic expected goals (xG) based on shot position.

        Simple model based on distance and angle to goal.
        """
        # Distance to goal center
        goal_x = settings.PITCH_LENGTH
        goal_y = settings.PITCH_WIDTH / 2

        distance = np.sqrt(
            (goal_x - position.x) ** 2 +
            (goal_y - position.y) ** 2
        )

        # Angle to goal (wider angle = better chance)
        goal_width = 7.32  # meters
        angle = np.arctan2(goal_width, distance)

        # Simple xG model (exponential decay with distance, boosted by angle)
        base_xg = np.exp(-distance / 15)  # Decay factor
        angle_factor = angle / (np.pi / 4)  # Normalize angle

        xg = base_xg * (0.5 + 0.5 * angle_factor)

        # Cap at reasonable values
        return min(max(xg, 0.01), 0.95)

    def _calculate_distance(self, pos1: Position, pos2: Position) -> float:
        """Calculate distance between two positions in meters."""
        return np.sqrt(
            (pos2.x - pos1.x) ** 2 +
            (pos2.y - pos1.y) ** 2
        )

    async def get_player_metrics(self, player_id: int) -> PlayerMetrics:
        """Calculate metrics for a specific player."""
        if player_id not in self.player_states:
            return PlayerMetrics(track_id=player_id, team=TeamSide.UNKNOWN)

        state = self.player_states[player_id]

        # Calculate distances
        total_distance = 0.0
        sprint_distance = 0.0
        high_intensity_distance = 0.0
        sprint_count = 0
        in_sprint = False

        for i in range(1, len(state.positions)):
            prev_time, prev_pos = state.positions[i - 1]
            curr_time, curr_pos = state.positions[i]

            segment_distance = self._calculate_distance(prev_pos, curr_pos)
            total_distance += segment_distance

            speed = state.speeds[i] if i < len(state.speeds) else 0

            if speed >= self.sprint_threshold:
                sprint_distance += segment_distance
                if not in_sprint:
                    sprint_count += 1
                    in_sprint = True
            else:
                in_sprint = False

            if speed >= self.high_intensity_threshold:
                high_intensity_distance += segment_distance

        # Calculate speed stats
        speeds = [s for s in state.speeds if s > 0]
        max_speed = max(speeds) if speeds else 0
        avg_speed = np.mean(speeds) if speeds else 0

        return PlayerMetrics(
            track_id=player_id,
            team=state.team,
            distance_covered_m=total_distance,
            sprint_count=sprint_count,
            sprint_distance_m=sprint_distance,
            high_intensity_distance_m=high_intensity_distance,
            max_speed_kmh=max_speed,
            avg_speed_kmh=avg_speed,
            passes_attempted=state.passes_attempted,
            passes_completed=state.passes_completed,
            touches=state.touches,
            tackles=state.tackles,
            interceptions=state.interceptions
        )

    async def get_team_metrics(self, team: TeamSide) -> TeamMetrics:
        """Calculate metrics for a team."""
        team_state = self.team_states[team]

        # Get all players for this team
        team_players = [
            pid for pid, state in self.player_states.items()
            if state.team == team
        ]

        # Calculate team totals
        total_distance = 0.0
        for pid in team_players:
            metrics = await self.get_player_metrics(pid)
            total_distance += metrics.distance_covered_m

        # Calculate possession
        total_possession = (
            self.team_states[TeamSide.HOME]["possession_time_ms"] +
            self.team_states[TeamSide.AWAY]["possession_time_ms"]
        )
        possession_pct = (
            (team_state["possession_time_ms"] / total_possession * 100)
            if total_possession > 0 else 50.0
        )

        # Pass completion
        pass_completion = (
            (team_state["passes_completed"] / team_state["passes_attempted"] * 100)
            if team_state["passes_attempted"] > 0 else 0.0
        )

        return TeamMetrics(
            team=team,
            possession_pct=possession_pct,
            total_passes=team_state["passes_attempted"],
            pass_completion_pct=pass_completion,
            shots=team_state["shots"],
            shots_on_target=team_state["shots_on_target"],
            xg=team_state["xg"],
            total_distance_km=total_distance / 1000
        )

    async def get_player_heatmap(
        self,
        match_id: str,
        player_id: int
    ) -> HeatmapData:
        """Generate heatmap for a player's positions."""
        grid_x, grid_y = self.heatmap_grid_size

        # Initialize grid
        grid = [[0.0 for _ in range(grid_y)] for _ in range(grid_x)]

        if player_id not in self.player_states:
            return HeatmapData(
                player_id=player_id,
                grid=grid,
                grid_size=self.heatmap_grid_size
            )

        state = self.player_states[player_id]

        # Map positions to grid
        for _, position in state.positions:
            # Calculate grid cell
            cell_x = int(position.x / settings.PITCH_LENGTH * grid_x)
            cell_y = int(position.y / settings.PITCH_WIDTH * grid_y)

            # Clamp to grid bounds
            cell_x = max(0, min(cell_x, grid_x - 1))
            cell_y = max(0, min(cell_y, grid_y - 1))

            grid[cell_x][cell_y] += 1

        # Normalize
        max_val = max(max(row) for row in grid)
        if max_val > 0:
            grid = [[cell / max_val for cell in row] for row in grid]

        return HeatmapData(
            player_id=player_id,
            team=state.team,
            grid=grid,
            grid_size=self.heatmap_grid_size
        )

    async def get_team_heatmap(
        self,
        match_id: str,
        team: TeamSide
    ) -> HeatmapData:
        """Generate combined heatmap for all players of a team."""
        grid_x, grid_y = self.heatmap_grid_size
        grid = [[0.0 for _ in range(grid_y)] for _ in range(grid_x)]

        # Aggregate all team players
        for player_id, state in self.player_states.items():
            if state.team != team:
                continue

            for _, position in state.positions:
                cell_x = int(position.x / settings.PITCH_LENGTH * grid_x)
                cell_y = int(position.y / settings.PITCH_WIDTH * grid_y)

                cell_x = max(0, min(cell_x, grid_x - 1))
                cell_y = max(0, min(cell_y, grid_y - 1))

                grid[cell_x][cell_y] += 1

        # Normalize
        max_val = max(max(row) for row in grid)
        if max_val > 0:
            grid = [[cell / max_val for cell in row] for row in grid]

        return HeatmapData(
            team=team,
            grid=grid,
            grid_size=self.heatmap_grid_size
        )

    async def get_pass_network(
        self,
        match_id: str,
        team: TeamSide
    ) -> PassNetwork:
        """Generate pass network visualization data."""
        # Get completed passes for team
        team_passes = [
            e for e in self.events
            if e.event_type == EventType.PASS
            and e.team == team
            and e.success
            and e.player_id is not None
            and e.recipient_id is not None
        ]

        # Build nodes (players with average positions)
        player_positions = defaultdict(list)
        for pid, state in self.player_states.items():
            if state.team != team:
                continue
            for _, pos in state.positions:
                player_positions[pid].append(pos)

        nodes = []
        for pid, positions in player_positions.items():
            if not positions:
                continue
            avg_x = np.mean([p.x for p in positions])
            avg_y = np.mean([p.y for p in positions])
            nodes.append({
                "id": pid,
                "x": avg_x,
                "y": avg_y,
                "passes": sum(1 for p in team_passes if p.player_id == pid)
            })

        # Build edges (pass connections)
        pass_counts = defaultdict(int)
        for pass_event in team_passes:
            key = (pass_event.player_id, pass_event.recipient_id)
            pass_counts[key] += 1

        edges = []
        for (source, target), count in pass_counts.items():
            edges.append({
                "source": source,
                "target": target,
                "weight": count
            })

        return PassNetwork(
            team=team,
            nodes=nodes,
            edges=edges
        )

    async def detect_formation(
        self,
        players: List[DetectedPlayer],
        team: TeamSide
    ) -> str:
        """
        Detect current formation from player positions.

        Returns formation string like "4-4-2", "4-3-3", etc.
        """
        team_players = [
            p for p in players
            if p.team == team
            and p.pitch_position is not None
            and not p.is_goalkeeper
        ]

        if len(team_players) < 10:
            return "unknown"

        # Sort players by x position (defense to attack)
        sorted_players = sorted(team_players, key=lambda p: p.pitch_position.x)

        # Cluster into lines (defense, midfield, attack)
        x_positions = [p.pitch_position.x for p in sorted_players]

        # Simple clustering by x-coordinate ranges
        lines = self._cluster_into_lines(x_positions)

        if len(lines) == 3:
            return f"{lines[0]}-{lines[1]}-{lines[2]}"
        elif len(lines) == 4:
            return f"{lines[0]}-{lines[1]}-{lines[2]}-{lines[3]}"
        else:
            return "unknown"

    def _cluster_into_lines(self, x_positions: List[float]) -> List[int]:
        """Cluster x positions into formation lines."""
        if len(x_positions) < 10:
            return []

        sorted_x = sorted(x_positions)

        # Find gaps in x positions to separate lines
        gaps = []
        for i in range(1, len(sorted_x)):
            gaps.append((sorted_x[i] - sorted_x[i-1], i))

        # Sort by gap size and take largest 2-3 gaps
        gaps.sort(reverse=True)
        split_indices = sorted([g[1] for g in gaps[:3]])

        # Count players in each line
        lines = []
        prev_idx = 0
        for idx in split_indices:
            lines.append(idx - prev_idx)
            prev_idx = idx
        lines.append(len(sorted_x) - prev_idx)

        # Ensure we have valid formation
        if sum(lines) == 10:
            return lines

        return []

    async def get_player_highlights(
        self,
        match_id: str,
        player_id: int
    ) -> List[Dict]:
        """Get highlight moments for a player."""
        highlights = []

        for event in self.events:
            if event.player_id != player_id:
                continue

            # Key events for highlights
            if event.event_type in [EventType.SHOT, EventType.GOAL]:
                highlights.append({
                    "type": event.event_type.value,
                    "timestamp_ms": event.timestamp_ms,
                    "frame_number": event.frame_number,
                    "position": {"x": event.position.x, "y": event.position.y}
                })
            elif event.event_type == EventType.TACKLE and event.success:
                highlights.append({
                    "type": "successful_tackle",
                    "timestamp_ms": event.timestamp_ms,
                    "frame_number": event.frame_number
                })

        return highlights

    async def generate_report(self, match_id: str) -> MatchReport:
        """Generate comprehensive post-match report."""
        # Get team metrics
        home_metrics = await self.get_team_metrics(TeamSide.HOME)
        away_metrics = await self.get_team_metrics(TeamSide.AWAY)

        # Get player metrics
        player_metrics = []
        for player_id in self.player_states.keys():
            metrics = await self.get_player_metrics(player_id)
            player_metrics.append(metrics)

        # Get key events
        key_events = [
            e for e in self.events
            if e.event_type in [EventType.GOAL, EventType.SHOT]
        ]

        # Generate tactical insights
        insights = await self._generate_insights()

        # Calculate player ratings (simple based on involvement)
        player_ratings = {}
        for pid, state in self.player_states.items():
            involvement = (
                state.passes_completed * 2 +
                state.tackles * 3 +
                state.interceptions * 2 +
                state.shots * 2
            )
            base_rating = 6.0
            rating = min(10.0, base_rating + involvement * 0.1)
            player_ratings[pid] = round(rating, 1)

        return MatchReport(
            match_info=MatchInfo(
                match_id=match_id,
                home_team="Home Team",
                away_team="Away Team",
                date=None,
            ),
            final_score={"home": 0, "away": 0},  # Would need goal tracking
            home_metrics=home_metrics,
            away_metrics=away_metrics,
            player_metrics=player_metrics,
            key_events=key_events,
            tactical_insights=insights,
            player_ratings=player_ratings
        )

    async def _generate_insights(self) -> List[str]:
        """Generate tactical insights from the match."""
        insights = []

        home_metrics = await self.get_team_metrics(TeamSide.HOME)
        away_metrics = await self.get_team_metrics(TeamSide.AWAY)

        # Possession insight
        if abs(home_metrics.possession_pct - 50) > 10:
            dominant = "Home" if home_metrics.possession_pct > 50 else "Away"
            insights.append(
                f"{dominant} team dominated possession "
                f"({max(home_metrics.possession_pct, away_metrics.possession_pct):.1f}%)"
            )

        # Passing insight
        if home_metrics.pass_completion_pct > 80 or away_metrics.pass_completion_pct > 80:
            better = "Home" if home_metrics.pass_completion_pct > away_metrics.pass_completion_pct else "Away"
            insights.append(
                f"{better} team showed excellent passing accuracy"
            )

        # xG insight
        if home_metrics.xg > 0 or away_metrics.xg > 0:
            insights.append(
                f"Expected goals: Home {home_metrics.xg:.2f} - Away {away_metrics.xg:.2f}"
            )

        return insights

    async def save_analytics(self, match_id: str, output_path: str):
        """Save analytics to JSON file."""
        report = await self.generate_report(match_id)

        async with aiofiles.open(output_path, 'w') as f:
            await f.write(report.model_dump_json())

    def reset(self):
        """Reset all analytics state."""
        self.player_states.clear()
        self.team_states = {
            TeamSide.HOME: self._init_team_state(),
            TeamSide.AWAY: self._init_team_state()
        }
        self.events.clear()

    # ============== Focused Analysis Methods ==============

    async def generate_team_improvement_report(
        self,
        match_id: str,
        team: TeamSide
    ) -> TeamImprovementReport:
        """
        Generate a detailed improvement report for a specific team.

        Analyzes the team's performance and provides actionable coaching insights:
        - Strengths to maintain
        - Areas needing improvement with drill suggestions
        - Player-specific recommendations
        - Training focus areas
        """
        team_metrics = await self.get_team_metrics(team)
        team_name = "Home" if team == TeamSide.HOME else "Away"

        # Get all players for this team
        team_player_ids = [
            pid for pid, state in self.player_states.items()
            if state.team == team
        ]

        # Analyze strengths
        strengths = await self._identify_team_strengths(team, team_metrics)

        # Analyze areas for improvement
        improvement_areas = await self._identify_improvement_areas(team, team_metrics)

        # Analyze individual players
        player_improvements = await self._analyze_player_improvements(team_player_ids)

        # Generate training recommendations
        training_focus, drills = await self._generate_training_recommendations(
            improvement_areas, player_improvements
        )

        # Analyze specific issues
        passing_issues = await self._analyze_passing_network_issues(team)
        defensive_issues = await self._analyze_defensive_vulnerabilities(team)
        attacking_patterns = await self._analyze_attacking_patterns(team)

        # Create summary
        summary = self._create_improvement_summary(strengths, improvement_areas)

        return TeamImprovementReport(
            team_name=team_name,
            match_id=match_id,
            analysis_summary=summary,
            team_metrics=team_metrics,
            strengths=strengths,
            improvement_areas=improvement_areas,
            player_improvements=player_improvements,
            training_focus=training_focus,
            recommended_drills=drills,
            passing_network_issues=passing_issues,
            defensive_vulnerabilities=defensive_issues,
            attacking_patterns=attacking_patterns
        )

    async def _identify_team_strengths(
        self,
        team: TeamSide,
        metrics: TeamMetrics
    ) -> List[str]:
        """Identify team strengths based on metrics."""
        strengths = []

        if metrics.possession_pct > 55:
            strengths.append(f"Strong possession control ({metrics.possession_pct:.0f}%)")

        if metrics.pass_completion_pct > 80:
            strengths.append(f"Excellent passing accuracy ({metrics.pass_completion_pct:.0f}%)")

        if metrics.shots_on_target > 0 and metrics.shots > 0:
            accuracy = (metrics.shots_on_target / metrics.shots) * 100
            if accuracy > 50:
                strengths.append(f"High shot accuracy ({accuracy:.0f}% on target)")

        if metrics.xg > 1.5:
            strengths.append(f"Creating quality chances (xG: {metrics.xg:.2f})")

        # Check pressing intensity
        pressing_events = [
            e for e in self.events
            if e.team == team and e.event_type in [EventType.TACKLE, EventType.INTERCEPTION]
        ]
        if len(pressing_events) > 10:
            strengths.append(f"Active pressing ({len(pressing_events)} defensive actions)")

        return strengths if strengths else ["Team showed solid overall performance"]

    async def _identify_improvement_areas(
        self,
        team: TeamSide,
        metrics: TeamMetrics
    ) -> List[TeamImprovementArea]:
        """Identify areas that need improvement."""
        areas = []

        # Possession issues
        if metrics.possession_pct < 45:
            areas.append(TeamImprovementArea(
                area="possession",
                observation=f"Struggled to retain possession ({metrics.possession_pct:.0f}%)",
                drill_suggestion="Rondo drills (4v2, 5v2) to improve ball retention under pressure",
                priority="high"
            ))

        # Passing issues
        if metrics.pass_completion_pct < 75:
            areas.append(TeamImprovementArea(
                area="passing",
                observation=f"Pass completion rate below par ({metrics.pass_completion_pct:.0f}%)",
                drill_suggestion="Passing circuits with movement, focus on body positioning and first touch",
                priority="high"
            ))

        # Shot quality
        if metrics.shots > 5 and metrics.xg / metrics.shots < 0.08:
            areas.append(TeamImprovementArea(
                area="shot_selection",
                observation="Taking shots from low-probability positions",
                drill_suggestion="Finishing drills from high-value positions, patience in build-up",
                priority="medium"
            ))

        # Team shape analysis from positions
        await self._check_team_shape_issues(team, areas)

        return areas

    async def _check_team_shape_issues(
        self,
        team: TeamSide,
        areas: List[TeamImprovementArea]
    ):
        """Check for team shape and compactness issues."""
        team_positions = []
        for pid, state in self.player_states.items():
            if state.team != team or not state.positions:
                continue
            for _, pos in state.positions[-100:]:  # Last 100 positions
                team_positions.append(pos)

        if not team_positions:
            return

        # Calculate team width and depth
        x_coords = [p.x for p in team_positions]
        y_coords = [p.y for p in team_positions]

        avg_width = max(y_coords) - min(y_coords) if y_coords else 0
        avg_depth = max(x_coords) - min(x_coords) if x_coords else 0

        if avg_width > 50:
            areas.append(TeamImprovementArea(
                area="compactness",
                observation=f"Team too stretched horizontally (avg width: {avg_width:.0f}m)",
                drill_suggestion="Defensive shape drills, practice compacting when pressing",
                priority="medium"
            ))

        if avg_depth > 40:
            areas.append(TeamImprovementArea(
                area="defensive_line",
                observation=f"Too much space between lines (avg depth: {avg_depth:.0f}m)",
                drill_suggestion="Unit drills - defense and midfield moving together",
                priority="high"
            ))

    async def _analyze_player_improvements(
        self,
        player_ids: List[int]
    ) -> List[PlayerImprovementSuggestion]:
        """Analyze individual player performance and suggest improvements."""
        suggestions = []

        for player_id in player_ids:
            if player_id not in self.player_states:
                continue

            state = self.player_states[player_id]

            # Check passing efficiency
            if state.passes_attempted > 5:
                completion_rate = (state.passes_completed / state.passes_attempted) * 100
                if completion_rate < 70:
                    suggestions.append(PlayerImprovementSuggestion(
                        player_id=player_id,
                        category="passing",
                        observation=f"Low pass completion rate ({completion_rate:.0f}%)",
                        suggestion="Focus on simpler, safer passing options and body positioning",
                        priority="high"
                    ))

            # Check ball retention (using events)
            player_losses = [
                e for e in self.events
                if e.player_id == player_id and e.event_type == EventType.PASS and not e.success
            ]
            if len(player_losses) > 5:
                suggestions.append(PlayerImprovementSuggestion(
                    player_id=player_id,
                    category="decision_making",
                    observation=f"Lost possession {len(player_losses)} times",
                    suggestion="Work on awareness - scan before receiving, simpler choices under pressure",
                    priority="medium"
                ))

            # Check defensive contribution
            if state.tackles == 0 and state.interceptions == 0:
                suggestions.append(PlayerImprovementSuggestion(
                    player_id=player_id,
                    category="defending",
                    observation="No defensive actions recorded",
                    suggestion="Increase defensive intensity, help team press and recover ball",
                    priority="low"
                ))

        return suggestions

    async def _generate_training_recommendations(
        self,
        improvement_areas: List[TeamImprovementArea],
        player_improvements: List[PlayerImprovementSuggestion]
    ) -> tuple:
        """Generate training focus areas and recommended drills."""
        training_focus = []
        drills = []

        # Aggregate improvement areas
        area_counts = {}
        for area in improvement_areas:
            area_counts[area.area] = area_counts.get(area.area, 0) + 1
            if area.drill_suggestion and area.drill_suggestion not in drills:
                drills.append(area.drill_suggestion)

        # Add to training focus based on frequency and priority
        for area_obj in improvement_areas:
            if area_obj.priority == "high":
                training_focus.append(f"Priority: {area_obj.area} - {area_obj.observation}")

        # Add player-specific training if patterns emerge
        category_counts = {}
        for suggestion in player_improvements:
            category_counts[suggestion.category] = category_counts.get(suggestion.category, 0) + 1

        for category, count in category_counts.items():
            if count >= 2:
                training_focus.append(f"Team-wide focus: {category} (affects {count} players)")

        return training_focus, drills

    async def _analyze_passing_network_issues(self, team: TeamSide) -> List[str]:
        """Analyze issues in the passing network."""
        issues = []

        # Get pass events for team
        passes = [
            e for e in self.events
            if e.team == team and e.event_type == EventType.PASS
        ]

        if not passes:
            return issues

        # Check for over-reliance on certain players
        passer_counts = {}
        for p in passes:
            if p.player_id:
                passer_counts[p.player_id] = passer_counts.get(p.player_id, 0) + 1

        if passer_counts:
            max_passer = max(passer_counts.values())
            total_passes = len(passes)
            if max_passer / total_passes > 0.3:
                issues.append(f"Over-reliance on one player for ball distribution ({max_passer}/{total_passes} passes)")

        # Check for lack of forward passes
        forward_passes = [
            p for p in passes
            if p.metadata.get("direction") == "forward" or
               (p.recipient_id and p.player_id and
                self._is_forward_pass(p.player_id, p.recipient_id))
        ]

        if len(forward_passes) / len(passes) < 0.3:
            issues.append("Too many sideways/backwards passes - need more progressive play")

        return issues

    def _is_forward_pass(self, passer_id: int, receiver_id: int) -> bool:
        """Check if a pass is forward based on average positions."""
        if passer_id not in self.player_states or receiver_id not in self.player_states:
            return False

        passer_state = self.player_states[passer_id]
        receiver_state = self.player_states[receiver_id]

        if not passer_state.positions or not receiver_state.positions:
            return False

        passer_avg_x = np.mean([p.x for _, p in passer_state.positions[-10:]])
        receiver_avg_x = np.mean([p.x for _, p in receiver_state.positions[-10:]])

        return receiver_avg_x > passer_avg_x + 5  # At least 5m forward

    async def _analyze_defensive_vulnerabilities(self, team: TeamSide) -> List[str]:
        """Analyze defensive vulnerabilities."""
        vulnerabilities = []

        # Check for goals/shots conceded from certain areas
        opponent = TeamSide.AWAY if team == TeamSide.HOME else TeamSide.HOME
        opponent_shots = [
            e for e in self.events
            if e.team == opponent and e.event_type == EventType.SHOT
        ]

        if not opponent_shots:
            return vulnerabilities

        # Analyze shot locations
        left_shots = [s for s in opponent_shots if s.position.y < settings.PITCH_WIDTH / 3]
        right_shots = [s for s in opponent_shots if s.position.y > settings.PITCH_WIDTH * 2 / 3]
        central_shots = [s for s in opponent_shots if settings.PITCH_WIDTH / 3 <= s.position.y <= settings.PITCH_WIDTH * 2 / 3]

        total = len(opponent_shots)
        if left_shots and len(left_shots) / total > 0.4:
            vulnerabilities.append("Vulnerable on the left side - opponents finding space there")
        if right_shots and len(right_shots) / total > 0.4:
            vulnerabilities.append("Vulnerable on the right side - opponents finding space there")
        if central_shots and len(central_shots) / total > 0.5:
            vulnerabilities.append("Conceding chances through the center - midfield protection needed")

        return vulnerabilities

    async def _analyze_attacking_patterns(self, team: TeamSide) -> List[str]:
        """Analyze attacking patterns of the team."""
        patterns = []

        shots = [
            e for e in self.events
            if e.team == team and e.event_type == EventType.SHOT
        ]

        if not shots:
            patterns.append("Limited shot creation - need more penetration in final third")
            return patterns

        # Analyze where shots come from
        central_shots = [s for s in shots if settings.PITCH_WIDTH / 3 <= s.position.y <= settings.PITCH_WIDTH * 2 / 3]
        wide_shots = len(shots) - len(central_shots)

        if len(central_shots) > wide_shots:
            patterns.append(f"Prefer central attacks ({len(central_shots)}/{len(shots)} shots from center)")
        else:
            patterns.append(f"Good width in attacks ({wide_shots}/{len(shots)} shots from wide areas)")

        return patterns

    def _create_improvement_summary(
        self,
        strengths: List[str],
        improvement_areas: List[TeamImprovementArea]
    ) -> str:
        """Create a summary of the improvement report."""
        high_priority = [a for a in improvement_areas if a.priority == "high"]

        if not high_priority:
            return f"Overall solid performance with {len(strengths)} key strengths. Focus on maintaining current level."
        else:
            return f"Found {len(high_priority)} high-priority areas to address. Key focus: {high_priority[0].area}."

    async def generate_opponent_scout_report(
        self,
        match_id: str,
        opponent_team: TeamSide
    ) -> OpponentScoutReport:
        """
        Generate a scouting report for the opponent team.

        Analyzes opponent's patterns, weaknesses, and provides tactical
        recommendations to beat them.
        """
        team_metrics = await self.get_team_metrics(opponent_team)
        team_name = "Home" if opponent_team == TeamSide.HOME else "Away"

        # Get opponent player IDs
        opponent_player_ids = [
            pid for pid, state in self.player_states.items()
            if state.team == opponent_team
        ]

        # Analyze patterns
        attacking_patterns = await self._scout_attacking_patterns(opponent_team)
        defensive_patterns = await self._scout_defensive_patterns(opponent_team)
        build_up_patterns = await self._scout_build_up_patterns(opponent_team)

        # Identify weaknesses
        weaknesses = await self._identify_opponent_weaknesses(opponent_team, team_metrics)

        # Identify strengths to be aware of
        strengths = await self._identify_opponent_strengths(opponent_team, team_metrics)

        # Scout key players
        key_player_reports, danger_players = await self._scout_key_players(opponent_player_ids)

        # Analyze set pieces
        set_piece_vulnerabilities = await self._analyze_set_piece_vulnerabilities(opponent_team)

        # Generate tactical recommendations
        recommendations, approach, formation = await self._generate_tactical_plan(
            weaknesses, strengths, attacking_patterns, defensive_patterns
        )

        # Identify key battles
        key_battles = await self._identify_key_battles(opponent_team, weaknesses)

        # Create summary
        summary = self._create_scout_summary(weaknesses, danger_players)

        return OpponentScoutReport(
            opponent_name=team_name,
            match_id=match_id,
            analysis_summary=summary,
            team_metrics=team_metrics,
            attacking_patterns=attacking_patterns,
            defensive_patterns=defensive_patterns,
            build_up_patterns=build_up_patterns,
            weaknesses=weaknesses,
            strengths=strengths,
            danger_players=danger_players,
            key_player_reports=key_player_reports,
            set_piece_vulnerabilities=set_piece_vulnerabilities,
            recommended_formation=formation,
            tactical_approach=approach,
            tactical_recommendations=recommendations,
            key_battles=key_battles
        )

    async def _scout_attacking_patterns(self, team: TeamSide) -> List[TeamPattern]:
        """Scout opponent's attacking patterns."""
        patterns = []

        shots = [e for e in self.events if e.team == team and e.event_type == EventType.SHOT]
        goals = [e for e in self.events if e.team == team and e.event_type == EventType.GOAL]

        if not shots:
            return patterns

        # Analyze preferred attacking side
        left_attacks = len([s for s in shots if s.position.y < settings.PITCH_WIDTH / 3])
        right_attacks = len([s for s in shots if s.position.y > settings.PITCH_WIDTH * 2 / 3])
        central_attacks = len(shots) - left_attacks - right_attacks

        if left_attacks > right_attacks and left_attacks > central_attacks:
            patterns.append(TeamPattern(
                pattern_type="attacking",
                description="Prefer attacks down the left side",
                frequency="often",
                counter_strategy="Strengthen your right side defense, don't let them get crosses in"
            ))
        elif right_attacks > left_attacks and right_attacks > central_attacks:
            patterns.append(TeamPattern(
                pattern_type="attacking",
                description="Prefer attacks down the right side",
                frequency="often",
                counter_strategy="Strengthen your left side defense, cut off supply to that wing"
            ))
        else:
            patterns.append(TeamPattern(
                pattern_type="attacking",
                description="Central attacking approach",
                frequency="often",
                counter_strategy="Pack the midfield, force them wide where they're less effective"
            ))

        return patterns

    async def _scout_defensive_patterns(self, team: TeamSide) -> List[TeamPattern]:
        """Scout opponent's defensive patterns."""
        patterns = []

        # Get opponent defensive actions
        defensive_events = [
            e for e in self.events
            if e.team == team and e.event_type in [EventType.TACKLE, EventType.INTERCEPTION]
        ]

        if not defensive_events:
            patterns.append(TeamPattern(
                pattern_type="defensive",
                description="Low pressing intensity",
                frequency="always",
                counter_strategy="You have time on the ball - play out from the back confidently"
            ))
            return patterns

        # Check where they defend
        high_def = [e for e in defensive_events if e.position.x > settings.PITCH_LENGTH * 0.6]
        mid_def = [e for e in defensive_events if settings.PITCH_LENGTH * 0.3 < e.position.x <= settings.PITCH_LENGTH * 0.6]
        low_def = [e for e in defensive_events if e.position.x <= settings.PITCH_LENGTH * 0.3]

        if len(high_def) > len(mid_def) and len(high_def) > len(low_def):
            patterns.append(TeamPattern(
                pattern_type="defensive",
                description="High pressing team - win ball in opponent's half",
                frequency="often",
                counter_strategy="Play long balls behind their high line, exploit space in behind"
            ))
        elif len(low_def) > len(high_def):
            patterns.append(TeamPattern(
                pattern_type="defensive",
                description="Deep defensive block",
                frequency="often",
                counter_strategy="Patient build-up, switch play to create gaps, shots from distance"
            ))

        return patterns

    async def _scout_build_up_patterns(self, team: TeamSide) -> List[TeamPattern]:
        """Scout opponent's build-up play patterns."""
        patterns = []

        passes = [e for e in self.events if e.team == team and e.event_type == EventType.PASS]

        if not passes:
            return patterns

        # Check build-up style
        successful_passes = [p for p in passes if p.success]
        completion_rate = len(successful_passes) / len(passes) if passes else 0

        if completion_rate > 0.8:
            patterns.append(TeamPattern(
                pattern_type="build_up",
                description="Confident in possession, short passing build-up",
                frequency="always",
                counter_strategy="Press high to disrupt rhythm, force long balls"
            ))
        else:
            patterns.append(TeamPattern(
                pattern_type="build_up",
                description="Direct style, bypass midfield with long balls",
                frequency="often",
                counter_strategy="Win second balls, stay compact in midfield"
            ))

        return patterns

    async def _identify_opponent_weaknesses(
        self,
        team: TeamSide,
        metrics: TeamMetrics
    ) -> List[TeamWeakness]:
        """Identify opponent weaknesses to exploit."""
        weaknesses = []

        # Possession weakness
        if metrics.possession_pct < 45:
            weaknesses.append(TeamWeakness(
                category="possession",
                description="Struggle to keep the ball under pressure",
                how_to_exploit="Press them high, force turnovers in dangerous areas",
                confidence=0.8,
                evidence=[f"Only {metrics.possession_pct:.0f}% possession"]
            ))

        # Passing weakness
        if metrics.pass_completion_pct < 75:
            weaknesses.append(TeamWeakness(
                category="passing",
                description=f"Poor passing accuracy ({metrics.pass_completion_pct:.0f}%)",
                how_to_exploit="Apply pressure to force misplaced passes, win possession in midfield",
                confidence=0.75,
                evidence=[f"{metrics.pass_completion_pct:.0f}% pass completion"]
            ))

        # Defensive weakness - check shots conceded
        my_team = TeamSide.HOME if team == TeamSide.AWAY else TeamSide.AWAY
        shots_against = [e for e in self.events if e.team == my_team and e.event_type == EventType.SHOT]
        if len(shots_against) > 10:
            weaknesses.append(TeamWeakness(
                category="defensive",
                description="Concede too many shots",
                how_to_exploit="Be direct, attack with pace, they can't defend well",
                confidence=0.7,
                evidence=[f"Conceded {len(shots_against)} shots"]
            ))

        # Check for weak areas using positions
        await self._find_positional_weaknesses(team, weaknesses)

        return weaknesses

    async def _find_positional_weaknesses(
        self,
        team: TeamSide,
        weaknesses: List[TeamWeakness]
    ):
        """Find positional/structural weaknesses."""
        # Analyze team shape
        player_positions = []
        for pid, state in self.player_states.items():
            if state.team != team or not state.positions:
                continue
            avg_y = np.mean([p.y for _, p in state.positions[-50:]])
            player_positions.append(avg_y)

        if not player_positions:
            return

        # Check if one side is undermanned
        left_players = len([y for y in player_positions if y < settings.PITCH_WIDTH / 3])
        right_players = len([y for y in player_positions if y > settings.PITCH_WIDTH * 2 / 3])

        if left_players < 2:
            weaknesses.append(TeamWeakness(
                category="structural",
                description="Weak on their left side - few players covering",
                how_to_exploit="Attack down your right side, overload that area",
                confidence=0.7,
                evidence=["Only few players covering left side"]
            ))
        elif right_players < 2:
            weaknesses.append(TeamWeakness(
                category="structural",
                description="Weak on their right side - few players covering",
                how_to_exploit="Attack down your left side, overload that area",
                confidence=0.7,
                evidence=["Only few players covering right side"]
            ))

    async def _identify_opponent_strengths(
        self,
        team: TeamSide,
        metrics: TeamMetrics
    ) -> List[str]:
        """Identify opponent strengths to neutralize."""
        strengths = []

        if metrics.possession_pct > 55:
            strengths.append(f"Good at keeping the ball ({metrics.possession_pct:.0f}% possession)")

        if metrics.pass_completion_pct > 82:
            strengths.append("Excellent passing team - hard to press")

        if metrics.xg > 1.5:
            strengths.append(f"Creating quality chances (xG: {metrics.xg:.2f})")

        if metrics.shots_on_target > 5:
            strengths.append(f"Clinical finishing ({metrics.shots_on_target} shots on target)")

        return strengths

    async def _scout_key_players(
        self,
        player_ids: List[int]
    ) -> tuple:
        """Scout key opponent players."""
        reports = []
        danger_players = []

        for player_id in player_ids:
            if player_id not in self.player_states:
                continue

            state = self.player_states[player_id]
            metrics = await self.get_player_metrics(player_id)

            # Determine threat level
            threat_level = 5
            tendencies = []
            strengths = []
            weaknesses = []

            # High involvement = key player
            involvement = (
                state.passes_completed +
                state.shots * 2 +
                state.tackles +
                state.interceptions
            )

            if involvement > 20:
                threat_level += 2
                danger_players.append(player_id)

            if state.shots > 2:
                threat_level += 1
                tendencies.append("Takes shots often")
                strengths.append("Goal threat")

            if state.passes_completed > 15:
                tendencies.append("Key passer, distributes play")
                strengths.append("Excellent passing range")

            # Check pass completion
            if state.passes_attempted > 5:
                completion = state.passes_completed / state.passes_attempted
                if completion < 0.7:
                    weaknesses.append(f"Loses ball under pressure ({completion*100:.0f}% pass accuracy)")
                    threat_level -= 1

            if state.tackles == 0 and state.interceptions == 0:
                weaknesses.append("Doesn't track back to defend")

            # Determine position from average location
            position = "Unknown"
            if state.positions:
                avg_x = np.mean([p.x for _, p in state.positions])
                if avg_x < settings.PITCH_LENGTH * 0.3:
                    position = "Defender"
                elif avg_x < settings.PITCH_LENGTH * 0.6:
                    position = "Midfielder"
                else:
                    position = "Forward"

            # Create advice
            advice = self._create_player_advice(threat_level, weaknesses, strengths)

            reports.append(PlayerScoutReport(
                player_id=player_id,
                position=position,
                tendencies=tendencies,
                strengths=strengths,
                weaknesses=weaknesses,
                threat_level=min(10, max(1, threat_level)),
                tactical_advice=advice
            ))

        # Sort by threat level
        reports.sort(key=lambda x: x.threat_level, reverse=True)

        return reports[:5], danger_players  # Return top 5 most threatening

    def _create_player_advice(
        self,
        threat_level: int,
        weaknesses: List[str],
        strengths: List[str]
    ) -> str:
        """Create tactical advice for dealing with a player."""
        if threat_level >= 7:
            if weaknesses:
                return f"Key danger - mark tightly. Exploit: {weaknesses[0]}"
            return "Key danger - man-mark, don't give space"
        elif threat_level >= 5:
            if weaknesses:
                return f"Monitor this player. Target when possible: {weaknesses[0]}"
            return "Solid player - stay alert"
        else:
            return "Limited threat - can be left to defensive structure"

    async def _analyze_set_piece_vulnerabilities(self, team: TeamSide) -> List[str]:
        """Analyze set piece vulnerabilities."""
        vulnerabilities = []

        corners = [e for e in self.events if e.team != team and e.event_type == EventType.CORNER]
        free_kicks = [e for e in self.events if e.team != team and e.event_type == EventType.FREE_KICK]

        if corners:
            vulnerabilities.append(f"Conceded {len(corners)} corners - test them from set pieces")

        # Would need more data about goals from set pieces
        vulnerabilities.append("Target the back post on corners - often leave space there")

        return vulnerabilities

    async def _generate_tactical_plan(
        self,
        weaknesses: List[TeamWeakness],
        strengths: List[str],
        attacking_patterns: List[TeamPattern],
        defensive_patterns: List[TeamPattern]
    ) -> tuple:
        """Generate tactical plan to beat opponent."""
        recommendations = []

        # Based on weaknesses
        for weakness in weaknesses:
            recommendations.append(f"Exploit {weakness.category}: {weakness.how_to_exploit}")

        # Based on their defensive patterns
        for pattern in defensive_patterns:
            recommendations.append(pattern.counter_strategy)

        # Determine approach
        approach = "balanced"
        if any(w.category == "defensive" for w in weaknesses):
            approach = "attacking"
            recommendations.append("Be aggressive - they struggle to defend")
        elif "Good at keeping the ball" in strengths:
            approach = "counter_attack"
            recommendations.append("Let them have the ball, strike on the counter")

        # Recommend formation
        formation = "4-3-3"  # Default
        if any(w.category == "structural" and "left" in w.description.lower() for w in weaknesses):
            formation = "4-3-3"
            recommendations.append("Use 4-3-3 to overload your right wing")
        elif approach == "counter_attack":
            formation = "4-5-1"
            recommendations.append("Use 4-5-1 for defensive solidity and quick counters")

        return recommendations, approach, formation

    async def _identify_key_battles(
        self,
        opponent_team: TeamSide,
        weaknesses: List[TeamWeakness]
    ) -> List[str]:
        """Identify key tactical battles."""
        battles = []

        for weakness in weaknesses:
            if weakness.category == "structural":
                battles.append(f"Your wingers vs their fullbacks on the weak side")
            elif weakness.category == "defensive":
                battles.append("Your forwards vs their center-backs - test them early")
            elif weakness.category == "passing":
                battles.append("Your midfield pressing vs their build-up")

        if not battles:
            battles.append("Midfield battle - control the center to control the game")
            battles.append("Wide areas - get crosses in and test their aerial ability")

        return battles

    def _create_scout_summary(
        self,
        weaknesses: List[TeamWeakness],
        danger_players: List[int]
    ) -> str:
        """Create a summary of the scouting report."""
        if weaknesses:
            main_weakness = weaknesses[0]
            return f"Key weakness: {main_weakness.description}. {len(danger_players)} players to watch closely."
        return f"Solid opponent. {len(danger_players)} key players identified."
