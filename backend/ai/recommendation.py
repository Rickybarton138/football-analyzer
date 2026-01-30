"""
Recommendation Engine

Generates coaching recommendations based on match analysis,
including substitution suggestions, tactical adjustments,
and strategic insights.
"""
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

from config import settings
from models.schemas import AnalysisMode
from models.schemas import (
    DetectedPlayer, TeamSide, PlayerMetrics, TeamMetrics,
    TacticalAlert, AlertPriority, SubstitutionSuggestion, Position
)


@dataclass
class PlayerFatigue:
    """Player fatigue metrics."""
    track_id: int
    total_distance_m: float
    sprint_count: int
    high_intensity_pct: float
    current_speed_vs_avg: float  # Ratio of current to average speed
    time_played_ms: int
    fatigue_score: float  # 0-1, higher = more tired


@dataclass
class TacticalRecommendation:
    """A tactical recommendation for the coach."""
    category: str  # "pressing", "possession", "defense", "attack"
    priority: AlertPriority
    message: str
    details: str
    action_items: List[str]
    confidence: float


class RecommendationEngine:
    """
    Generates coaching recommendations based on match data.

    Categories of recommendations:
    1. Substitutions (fatigue-based, tactical)
    2. Tactical adjustments (formation, pressing triggers)
    3. Strategic insights (opponent weaknesses, patterns)
    4. Set piece opportunities

    Now supports perspective-aware recommendations:
    - coach_team: Generate "how to improve" recommendations for your team
    - opponent analysis: Generate "how to exploit" recommendations against opponent
    """

    def __init__(self):
        self.player_fatigue: Dict[int, PlayerFatigue] = {}
        self.recommendations: List[TacticalRecommendation] = []
        self.match_phase = "first_half"  # first_half, second_half
        self.coach_team: Optional[TeamSide] = None  # Which team is the user coaching

        # Thresholds
        self.fatigue_threshold = 0.7  # Suggest substitution above this
        self.speed_drop_threshold = 0.75  # Speed dropped to 75% of average
        self.pressing_intensity_threshold = 5  # Minimum pressing triggers

    def set_coach_team(self, team: TeamSide):
        """Set which team the coach is managing for perspective-aware recommendations."""
        self.coach_team = team

    async def analyze_and_recommend(
        self,
        players: List[DetectedPlayer],
        player_metrics: Dict[int, PlayerMetrics],
        team_metrics: Dict[TeamSide, TeamMetrics],
        match_time_ms: int,
        recent_alerts: List[TacticalAlert]
    ) -> List[TacticalRecommendation]:
        """
        Generate recommendations based on current match state.

        Args:
            players: Current player detections
            player_metrics: Accumulated player metrics
            team_metrics: Team-level metrics
            match_time_ms: Current match time
            recent_alerts: Recent tactical alerts

        Returns:
            List of recommendations
        """
        recommendations = []

        # Update match phase
        if match_time_ms > 45 * 60 * 1000:
            self.match_phase = "second_half"

        # Analyze player fatigue
        fatigue_recommendations = await self._analyze_fatigue(
            player_metrics, match_time_ms
        )
        recommendations.extend(fatigue_recommendations)

        # Analyze tactical patterns
        tactical_recommendations = await self._analyze_tactics(
            team_metrics, recent_alerts, match_time_ms
        )
        recommendations.extend(tactical_recommendations)

        # Analyze pressing effectiveness
        pressing_recommendations = await self._analyze_pressing(
            recent_alerts, team_metrics
        )
        recommendations.extend(pressing_recommendations)

        # Strategic analysis
        strategic_recommendations = await self._analyze_strategy(
            team_metrics, match_time_ms
        )
        recommendations.extend(strategic_recommendations)

        # Store recommendations
        self.recommendations = recommendations

        return recommendations

    async def _analyze_fatigue(
        self,
        player_metrics: Dict[int, PlayerMetrics],
        match_time_ms: int
    ) -> List[TacticalRecommendation]:
        """Analyze player fatigue and suggest substitutions."""
        recommendations = []

        for player_id, metrics in player_metrics.items():
            # Calculate fatigue score
            fatigue = self._calculate_fatigue(metrics, match_time_ms)
            self.player_fatigue[player_id] = fatigue

            # Check if substitution needed
            if fatigue.fatigue_score > self.fatigue_threshold:
                # High fatigue - recommend substitution
                recommendations.append(TacticalRecommendation(
                    category="substitution",
                    priority=AlertPriority.TACTICAL,
                    message=f"Consider substituting player #{player_id}",
                    details=(
                        f"Fatigue score: {fatigue.fatigue_score:.0%}, "
                        f"Distance: {fatigue.total_distance_m/1000:.1f}km, "
                        f"Sprints: {fatigue.sprint_count}"
                    ),
                    action_items=[
                        f"Player has covered {fatigue.total_distance_m/1000:.1f}km",
                        f"Current speed at {fatigue.current_speed_vs_avg:.0%} of average",
                        "Consider fresh legs for final phase"
                    ],
                    confidence=fatigue.fatigue_score
                ))

            # Moderate fatigue warning
            elif fatigue.fatigue_score > 0.5 and self.match_phase == "second_half":
                recommendations.append(TacticalRecommendation(
                    category="fatigue_warning",
                    priority=AlertPriority.STRATEGIC,
                    message=f"Monitor player #{player_id}'s workload",
                    details=f"Fatigue building - {fatigue.fatigue_score:.0%}",
                    action_items=[
                        "Watch for decreased intensity",
                        "Have substitute ready"
                    ],
                    confidence=0.6
                ))

        return recommendations

    def _calculate_fatigue(
        self,
        metrics: PlayerMetrics,
        match_time_ms: int
    ) -> PlayerFatigue:
        """Calculate fatigue score for a player."""
        # Base fatigue from distance
        distance_factor = min(1.0, metrics.distance_covered_m / 12000)  # 12km is high

        # Sprint fatigue
        sprint_factor = min(1.0, metrics.sprint_count / 30)  # 30 sprints is high

        # High intensity work
        total_work = max(metrics.distance_covered_m, 1)
        high_intensity_pct = (
            (metrics.sprint_distance_m + metrics.high_intensity_distance_m) /
            total_work
        )

        # Time factor (fatigue increases over match)
        time_factor = min(1.0, match_time_ms / (90 * 60 * 1000))

        # Speed decline (if we have history)
        speed_ratio = 1.0
        if metrics.avg_speed_kmh > 0 and metrics.max_speed_kmh > 0:
            # Rough estimate - actual would need recent vs early match speed
            speed_ratio = metrics.avg_speed_kmh / (metrics.max_speed_kmh * 0.5)

        # Combined fatigue score
        fatigue_score = (
            0.3 * distance_factor +
            0.2 * sprint_factor +
            0.2 * high_intensity_pct +
            0.2 * time_factor +
            0.1 * (1 - min(1, speed_ratio))
        )

        return PlayerFatigue(
            track_id=metrics.track_id,
            total_distance_m=metrics.distance_covered_m,
            sprint_count=metrics.sprint_count,
            high_intensity_pct=high_intensity_pct,
            current_speed_vs_avg=speed_ratio,
            time_played_ms=match_time_ms,
            fatigue_score=min(1.0, fatigue_score)
        )

    async def _analyze_tactics(
        self,
        team_metrics: Dict[TeamSide, TeamMetrics],
        recent_alerts: List[TacticalAlert],
        match_time_ms: int
    ) -> List[TacticalRecommendation]:
        """Analyze tactical situation and recommend adjustments."""
        recommendations = []

        home_metrics = team_metrics.get(TeamSide.HOME)
        away_metrics = team_metrics.get(TeamSide.AWAY)

        if not home_metrics or not away_metrics:
            return recommendations

        # Possession analysis
        if home_metrics.possession_pct < 40:
            recommendations.append(TacticalRecommendation(
                category="possession",
                priority=AlertPriority.TACTICAL,
                message="Struggling to retain possession",
                details=f"Only {home_metrics.possession_pct:.0f}% possession",
                action_items=[
                    "Drop midfield deeper to receive",
                    "Use goalkeeper for circulation",
                    "Consider formation change to 3 midfielders"
                ],
                confidence=0.8
            ))
        elif home_metrics.possession_pct > 60:
            recommendations.append(TacticalRecommendation(
                category="possession",
                priority=AlertPriority.STRATEGIC,
                message="Dominating possession - convert to chances",
                details=f"{home_metrics.possession_pct:.0f}% possession",
                action_items=[
                    "Take more risks in final third",
                    "Make runs behind defense",
                    "Play quicker to catch them unorganized"
                ],
                confidence=0.7
            ))

        # Pass completion analysis
        if home_metrics.pass_completion_pct < 70:
            recommendations.append(TacticalRecommendation(
                category="passing",
                priority=AlertPriority.TACTICAL,
                message="Poor pass completion rate",
                details=f"Only {home_metrics.pass_completion_pct:.0f}% passes completed",
                action_items=[
                    "Play simpler, shorter passes",
                    "Better body positioning before receiving",
                    "Increase support angles"
                ],
                confidence=0.75
            ))

        return recommendations

    async def _analyze_pressing(
        self,
        recent_alerts: List[TacticalAlert],
        team_metrics: Dict[TeamSide, TeamMetrics]
    ) -> List[TacticalRecommendation]:
        """Analyze pressing effectiveness."""
        recommendations = []

        # Count pressing alerts
        pressing_alerts = [
            a for a in recent_alerts
            if "press" in a.message.lower()
        ]

        if len(pressing_alerts) > self.pressing_intensity_threshold:
            # High pressing - check effectiveness
            recommendations.append(TacticalRecommendation(
                category="pressing",
                priority=AlertPriority.STRATEGIC,
                message="High pressing activity detected",
                details=f"{len(pressing_alerts)} pressing triggers in recent play",
                action_items=[
                    "Ensure press is coordinated",
                    "Cover behind pressing players",
                    "Monitor energy expenditure"
                ],
                confidence=0.7
            ))
        elif len(pressing_alerts) < 2:
            recommendations.append(TacticalRecommendation(
                category="pressing",
                priority=AlertPriority.TACTICAL,
                message="Pressing triggers being missed",
                details="Few pressing opportunities taken",
                action_items=[
                    "React quicker to ball losses",
                    "Set pressing traps",
                    "Communicate triggers"
                ],
                confidence=0.6
            ))

        return recommendations

    async def _analyze_strategy(
        self,
        team_metrics: Dict[TeamSide, TeamMetrics],
        match_time_ms: int
    ) -> List[TacticalRecommendation]:
        """Generate strategic recommendations based on match phase."""
        recommendations = []

        home_metrics = team_metrics.get(TeamSide.HOME)
        if not home_metrics:
            return recommendations

        match_minutes = match_time_ms / (60 * 1000)

        # Late game adjustments
        if match_minutes > 70:
            if home_metrics.xg < 0.5:
                recommendations.append(TacticalRecommendation(
                    category="attack",
                    priority=AlertPriority.TACTICAL,
                    message="Low xG - need more threat",
                    details=f"Only {home_metrics.xg:.2f} xG with {90-match_minutes:.0f} mins left",
                    action_items=[
                        "Push more players forward",
                        "Take shots from distance",
                        "Attack crosses with more bodies"
                    ],
                    confidence=0.8
                ))

        # xG analysis
        if home_metrics.shots > 0 and home_metrics.xg / home_metrics.shots < 0.05:
            recommendations.append(TacticalRecommendation(
                category="shooting",
                priority=AlertPriority.TACTICAL,
                message="Shot quality is poor",
                details="Shooting from low-probability positions",
                action_items=[
                    "Work ball into better positions",
                    "Create overloads in the box",
                    "Be patient in build-up"
                ],
                confidence=0.7
            ))

        return recommendations

    async def get_substitution_suggestions(
        self,
        team: TeamSide,
        available_subs: List[int]
    ) -> List[SubstitutionSuggestion]:
        """
        Get specific substitution suggestions.

        Args:
            team: Team to analyze
            available_subs: Player IDs of available substitutes

        Returns:
            List of substitution suggestions
        """
        suggestions = []

        # Sort players by fatigue
        tired_players = [
            (pid, fatigue)
            for pid, fatigue in self.player_fatigue.items()
            if fatigue.fatigue_score > 0.5
        ]
        tired_players.sort(key=lambda x: x[1].fatigue_score, reverse=True)

        for player_id, fatigue in tired_players[:3]:  # Top 3 most tired
            suggestions.append(SubstitutionSuggestion(
                player_out_id=player_id,
                suggested_position="Same position",  # Would need position tracking
                reason=f"High fatigue: {fatigue.fatigue_score:.0%}",
                fatigue_score=fatigue.fatigue_score,
                current_performance_score=1.0 - fatigue.fatigue_score
            ))

        return suggestions

    def get_half_time_analysis(
        self,
        team_metrics: Dict[TeamSide, TeamMetrics]
    ) -> Dict:
        """Generate half-time analysis summary."""
        home = team_metrics.get(TeamSide.HOME)
        away = team_metrics.get(TeamSide.AWAY)

        analysis = {
            "summary": [],
            "key_points": [],
            "recommendations": []
        }

        if home and away:
            # Possession
            if home.possession_pct > 55:
                analysis["key_points"].append(
                    f"Good possession control ({home.possession_pct:.0f}%)"
                )
            elif home.possession_pct < 45:
                analysis["key_points"].append(
                    f"Need to improve ball retention ({home.possession_pct:.0f}%)"
                )

            # Chances
            if home.xg > away.xg:
                analysis["key_points"].append(
                    f"Creating better chances (xG: {home.xg:.2f} vs {away.xg:.2f})"
                )
            else:
                analysis["key_points"].append(
                    f"Opponent creating more threat (xG: {away.xg:.2f} vs {home.xg:.2f})"
                )

            # Distance
            analysis["summary"].append(
                f"Total distance covered: {home.total_distance_km:.1f}km"
            )

        # Add fatigue warnings
        high_fatigue = [
            pid for pid, f in self.player_fatigue.items()
            if f.fatigue_score > 0.6
        ]
        if high_fatigue:
            analysis["recommendations"].append(
                f"Consider resting players {high_fatigue} in second half"
            )

        return analysis

    def reset(self):
        """Reset engine state."""
        self.player_fatigue.clear()
        self.recommendations.clear()
        self.match_phase = "first_half"
        self.coach_team = None

    # ============== Perspective-Aware Recommendations ==============

    async def generate_improvement_recommendations(
        self,
        team: TeamSide,
        team_metrics: TeamMetrics,
        player_metrics: Dict[int, PlayerMetrics]
    ) -> List[TacticalRecommendation]:
        """
        Generate recommendations focused on improving YOUR team's performance.

        These are "how to get better" insights:
        - Weaknesses to address in training
        - Individual player development areas
        - Team-wide patterns to improve
        """
        recommendations = []

        # Analyze possession improvement
        if team_metrics.possession_pct < 50:
            recommendations.append(TacticalRecommendation(
                category="improvement",
                priority=AlertPriority.STRATEGIC,
                message="Work on ball retention",
                details=f"Only {team_metrics.possession_pct:.0f}% possession - team losing the ball too easily",
                action_items=[
                    "Practice rondo drills (4v2, 5v2) to improve keeping possession under pressure",
                    "Focus on body positioning when receiving - always have an escape option",
                    "Work on first touch to set up next action"
                ],
                confidence=0.8
            ))

        # Analyze passing improvement
        if team_metrics.pass_completion_pct < 78:
            recommendations.append(TacticalRecommendation(
                category="improvement",
                priority=AlertPriority.TACTICAL,
                message="Improve passing accuracy",
                details=f"Pass completion at {team_metrics.pass_completion_pct:.0f}% - needs improvement",
                action_items=[
                    "Passing circuits with movement and one-touch options",
                    "Weight of pass drills - appropriate pace for distance",
                    "Communication - call for ball earlier"
                ],
                confidence=0.75
            ))

        # Analyze shot quality improvement
        if team_metrics.shots > 0:
            shot_quality = team_metrics.xg / team_metrics.shots if team_metrics.shots > 0 else 0
            if shot_quality < 0.1:
                recommendations.append(TacticalRecommendation(
                    category="improvement",
                    priority=AlertPriority.TACTICAL,
                    message="Improve shot selection",
                    details="Taking shots from poor positions - low xG per shot",
                    action_items=[
                        "Practice patience in final third - extra pass often opens better angle",
                        "Finishing drills from high-value positions (6-yard box, penalty spot)",
                        "Work on getting shots off quicker when in good positions"
                    ],
                    confidence=0.7
                ))

        # Player-specific improvements
        for player_id, metrics in player_metrics.items():
            if metrics.passes_attempted > 5:
                completion = (metrics.passes_completed / metrics.passes_attempted) * 100
                if completion < 65:
                    recommendations.append(TacticalRecommendation(
                        category="player_development",
                        priority=AlertPriority.STRATEGIC,
                        message=f"Player #{player_id} needs passing work",
                        details=f"Only {completion:.0f}% pass completion",
                        action_items=[
                            "Individual passing practice",
                            "Decision-making drills - when to pass, when to carry",
                            "Work on playing simpler options under pressure"
                        ],
                        confidence=0.7
                    ))

        return recommendations

    async def generate_exploitation_recommendations(
        self,
        opponent_team: TeamSide,
        opponent_metrics: TeamMetrics,
        opponent_player_metrics: Dict[int, PlayerMetrics]
    ) -> List[TacticalRecommendation]:
        """
        Generate recommendations focused on EXPLOITING opponent weaknesses.

        These are "how to beat them" insights:
        - Weaknesses to target
        - Players to isolate or press
        - Tactical approaches to exploit their patterns
        """
        recommendations = []

        # Exploit poor possession
        if opponent_metrics.possession_pct < 45:
            recommendations.append(TacticalRecommendation(
                category="exploit",
                priority=AlertPriority.TACTICAL,
                message="Press them high - they can't keep the ball",
                details=f"Opponent only had {opponent_metrics.possession_pct:.0f}% possession",
                action_items=[
                    "High press from the start - they're uncomfortable on the ball",
                    "Force them to play long - win second balls",
                    "Target their build-up play, especially from goal kicks"
                ],
                confidence=0.8
            ))

        # Exploit poor passing
        if opponent_metrics.pass_completion_pct < 72:
            recommendations.append(TacticalRecommendation(
                category="exploit",
                priority=AlertPriority.TACTICAL,
                message="Force them to pass under pressure",
                details=f"Only {opponent_metrics.pass_completion_pct:.0f}% pass completion - error-prone",
                action_items=[
                    "Aggressive pressing will cause turnovers",
                    "Don't give them time on the ball",
                    "Look for quick counter-attacks after forced errors"
                ],
                confidence=0.75
            ))

        # Exploit defensive weakness
        if opponent_metrics.shots > 0:
            # If we got lots of shots against them, they defend poorly
            recommendations.append(TacticalRecommendation(
                category="exploit",
                priority=AlertPriority.TACTICAL,
                message="Attack them directly - defense is vulnerable",
                details="Opponent struggled to prevent shots",
                action_items=[
                    "Be direct in attack - shoot when possible",
                    "Run at their defenders - test them 1v1",
                    "Overload the box on crosses"
                ],
                confidence=0.7
            ))

        # Find weak individual players
        for player_id, metrics in opponent_player_metrics.items():
            if metrics.passes_attempted > 5:
                completion = (metrics.passes_completed / metrics.passes_attempted) * 100
                if completion < 60:
                    recommendations.append(TacticalRecommendation(
                        category="target_player",
                        priority=AlertPriority.TACTICAL,
                        message=f"Target opponent #{player_id} - weak on the ball",
                        details=f"Only {completion:.0f}% pass completion - loses possession often",
                        action_items=[
                            "Press this player specifically",
                            "Force play toward them to win possession",
                            "They're a liability - exploit mistakes"
                        ],
                        confidence=0.7
                    ))

        return recommendations

    async def get_perspective_recommendations(
        self,
        analysis_mode: AnalysisMode,
        my_team: TeamSide,
        my_metrics: TeamMetrics,
        opponent_metrics: TeamMetrics,
        my_player_metrics: Dict[int, PlayerMetrics],
        opponent_player_metrics: Dict[int, PlayerMetrics]
    ) -> List[TacticalRecommendation]:
        """
        Get recommendations based on analysis mode perspective.

        Args:
            analysis_mode: Which perspective to use
            my_team: Which team is the user's
            my_metrics: User's team metrics
            opponent_metrics: Opponent team metrics
            my_player_metrics: User's team player metrics
            opponent_player_metrics: Opponent player metrics

        Returns:
            Perspective-appropriate recommendations
        """
        if analysis_mode == AnalysisMode.MY_TEAM:
            # Focus on improvement recommendations
            return await self.generate_improvement_recommendations(
                my_team, my_metrics, my_player_metrics
            )
        elif analysis_mode == AnalysisMode.OPPONENT:
            # Focus on exploitation recommendations
            opponent_team = TeamSide.AWAY if my_team == TeamSide.HOME else TeamSide.HOME
            return await self.generate_exploitation_recommendations(
                opponent_team, opponent_metrics, opponent_player_metrics
            )
        else:
            # Full analysis - return both
            improvement = await self.generate_improvement_recommendations(
                my_team, my_metrics, my_player_metrics
            )
            opponent_team = TeamSide.AWAY if my_team == TeamSide.HOME else TeamSide.HOME
            exploitation = await self.generate_exploitation_recommendations(
                opponent_team, opponent_metrics, opponent_player_metrics
            )
            return improvement + exploitation
