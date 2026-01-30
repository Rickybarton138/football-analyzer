"""
Coach Assist AI Service

AI-powered coaching insights similar to VEO's Coach Assist.
Uses match data and AI to provide:
- Instant match summaries
- Tactical talking points
- Suggested clips for review
- Answers to tactical questions

Integrates with Claude/OpenAI for intelligent analysis.
"""
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum


class InsightCategory(Enum):
    """Categories of coaching insights."""
    POSSESSION = "possession"
    ATTACKING = "attacking"
    DEFENDING = "defending"
    TRANSITIONS = "transitions"
    SET_PIECES = "set_pieces"
    INDIVIDUAL = "individual"
    TACTICAL = "tactical"
    PHYSICAL = "physical"


@dataclass
class TalkingPoint:
    """A coaching talking point generated from match data."""
    category: InsightCategory
    title: str
    insight: str
    evidence: str  # Data backing this insight
    suggested_clips: List[str] = None  # Event IDs to review
    priority: int = 1  # 1=high, 2=medium, 3=low
    team: str = "home"  # Which team this applies to


@dataclass
class TacticalQuestion:
    """A tactical question and answer."""
    question: str
    answer: str
    relevant_data: Dict
    confidence: float
    suggested_clips: List[str] = None


class CoachAssistService:
    """
    AI-powered coaching assistant service.

    Analyzes match data to generate insights, talking points,
    and answers tactical questions.
    """

    def __init__(self):
        self.match_data: Optional[Dict] = None
        self.events_data: List[Dict] = []
        self.talking_points: List[TalkingPoint] = []
        self.questions_history: List[TacticalQuestion] = []

        # AI client (initialized externally)
        self.ai_client = None
        self.ai_provider: str = "none"

    def set_ai_client(self, client, provider: str = "claude"):
        """Set the AI client for generating insights."""
        self.ai_client = client
        self.ai_provider = provider

    def load_match_data(
        self,
        match_stats=None,
        events: List[Dict] = None,
        player_stats: Dict = None,
        formations: Optional[Dict] = None,
        pitch_viz=None,
        event_detector=None,
        player_highlights=None
    ):
        """
        Load match data for analysis.

        Can accept either raw dicts or service objects.
        """
        # If services are passed, extract data from them
        if match_stats is not None and hasattr(match_stats, 'get_match_summary'):
            # It's a service object, extract data
            summary = match_stats.get_match_summary()
            self.match_data = {
                "stats": summary,
                "player_stats": {},
                "formations": formations or {}
            }
        elif match_stats is not None:
            # It's already a dict
            self.match_data = {
                "stats": match_stats,
                "player_stats": player_stats or {},
                "formations": formations or {}
            }
        else:
            self.match_data = {
                "stats": {},
                "player_stats": player_stats or {},
                "formations": formations or {}
            }

        # Extract events from event detector if provided
        if event_detector is not None and hasattr(event_detector, 'get_events'):
            detected_events = event_detector.get_events()
            self.events_data = [
                {
                    "id": f"event_{i}",
                    "type": e.event_type.value,
                    "timestamp_ms": e.timestamp_ms,
                    "time_str": f"{e.timestamp_ms // 60000}:{(e.timestamp_ms // 1000) % 60:02d}",
                    "team": e.team,
                    "description": e.description
                }
                for i, e in enumerate(detected_events)
            ]
        elif events is not None:
            self.events_data = events
        else:
            self.events_data = []

        # Store pitch visualization service reference for heatmap/position queries
        self._pitch_viz = pitch_viz

        # Store player highlights for clip suggestions
        self._player_highlights = player_highlights

    def generate_match_summary(self) -> str:
        """
        Generate a quick match summary.

        Returns a brief overview of the match for coaches.
        """
        if not self.match_data:
            return "No match data available."

        stats = self.match_data["stats"]
        home = stats.get("home", {})
        away = stats.get("away", {})

        summary_parts = []

        # Score
        home_goals = home.get("goals", 0)
        away_goals = away.get("goals", 0)
        summary_parts.append(f"Final Score: Home {home_goals} - {away_goals} Away")

        # Possession
        home_poss = home.get("possession_pct", 50)
        away_poss = away.get("possession_pct", 50)
        if abs(home_poss - away_poss) > 10:
            dominant = "Home" if home_poss > away_poss else "Away"
            summary_parts.append(f"{dominant} dominated possession ({max(home_poss, away_poss):.0f}%)")

        # Shots
        home_shots = home.get("shots", {}).get("total", 0)
        away_shots = away.get("shots", {}).get("total", 0)
        home_on_target = home.get("shots", {}).get("on_target", 0)
        away_on_target = away.get("shots", {}).get("on_target", 0)
        summary_parts.append(f"Shots: Home {home_shots} ({home_on_target} on target) - Away {away_shots} ({away_on_target} on target)")

        # xG comparison
        home_xg = home.get("xG", 0)
        away_xg = away.get("xG", 0)
        if home_xg > 0 or away_xg > 0:
            summary_parts.append(f"Expected Goals: Home {home_xg:.2f} - Away {away_xg:.2f}")

        # Pass accuracy
        home_pass = home.get("pass_accuracy", 0)
        away_pass = away.get("pass_accuracy", 0)
        summary_parts.append(f"Pass Accuracy: Home {home_pass:.0f}% - Away {away_pass:.0f}%")

        return "\n".join(summary_parts)

    def generate_talking_points(self, team: str = "home") -> List[TalkingPoint]:
        """
        Generate coaching talking points for a team.

        Analyzes match data to identify key areas for discussion.
        """
        self.talking_points = []

        if not self.match_data:
            return []

        stats = self.match_data["stats"]
        team_stats = stats.get(team, {})
        opponent_stats = stats.get("away" if team == "home" else "home", {})

        # Possession analysis
        self._analyze_possession(team, team_stats, opponent_stats)

        # Attacking analysis
        self._analyze_attacking(team, team_stats, opponent_stats)

        # Defending analysis
        self._analyze_defending(team, team_stats, opponent_stats)

        # Set pieces analysis
        self._analyze_set_pieces(team, team_stats, opponent_stats)

        # Individual standouts
        self._analyze_individuals(team)

        # Sort by priority
        self.talking_points.sort(key=lambda x: x.priority)

        return self.talking_points

    def _analyze_possession(self, team: str, team_stats: Dict, opponent_stats: Dict):
        """Generate possession-related talking points."""
        possession = team_stats.get("possession_pct", 50)
        pass_accuracy = team_stats.get("pass_accuracy", 0)
        passes_by_third = team_stats.get("passes_by_third", {})

        # High/low possession
        if possession > 55:
            self.talking_points.append(TalkingPoint(
                category=InsightCategory.POSSESSION,
                title="Dominant Possession",
                insight=f"Strong ball retention with {possession:.0f}% possession. The team controlled the tempo well.",
                evidence=f"Pass accuracy: {pass_accuracy:.0f}%",
                priority=2,
                team=team
            ))
        elif possession < 45:
            self.talking_points.append(TalkingPoint(
                category=InsightCategory.POSSESSION,
                title="Limited Possession",
                insight=f"Only {possession:.0f}% possession. Consider ways to retain the ball better under pressure.",
                evidence=f"Pass accuracy: {pass_accuracy:.0f}%",
                priority=1,
                team=team
            ))

        # Pass distribution by thirds
        def_passes = passes_by_third.get("defensive", 0)
        mid_passes = passes_by_third.get("middle", 0)
        att_passes = passes_by_third.get("attacking", 0)
        total_passes = def_passes + mid_passes + att_passes

        if total_passes > 0:
            att_pct = (att_passes / total_passes) * 100
            if att_pct < 20:
                self.talking_points.append(TalkingPoint(
                    category=InsightCategory.POSSESSION,
                    title="Limited Final Third Presence",
                    insight=f"Only {att_pct:.0f}% of passes in the attacking third. Need to progress the ball more effectively.",
                    evidence=f"Defensive: {def_passes}, Middle: {mid_passes}, Attacking: {att_passes}",
                    priority=1,
                    team=team
                ))

    def _analyze_attacking(self, team: str, team_stats: Dict, opponent_stats: Dict):
        """Generate attacking-related talking points."""
        shots = team_stats.get("shots", {})
        total_shots = shots.get("total", 0)
        on_target = shots.get("on_target", 0)
        goals = team_stats.get("goals", 0)
        xg = team_stats.get("xG", 0)
        conversion = team_stats.get("shot_conversion", 0)

        # Shot efficiency
        if total_shots > 0:
            on_target_pct = (on_target / total_shots) * 100

            if on_target_pct < 30:
                self.talking_points.append(TalkingPoint(
                    category=InsightCategory.ATTACKING,
                    title="Poor Shot Accuracy",
                    insight=f"Only {on_target_pct:.0f}% of shots on target. Focus on composure in front of goal.",
                    evidence=f"{on_target} on target from {total_shots} shots",
                    priority=1,
                    team=team
                ))
            elif on_target_pct > 50:
                self.talking_points.append(TalkingPoint(
                    category=InsightCategory.ATTACKING,
                    title="Good Shot Selection",
                    insight=f"{on_target_pct:.0f}% of shots on target. Quality chances being created.",
                    evidence=f"{on_target} on target from {total_shots} shots",
                    priority=2,
                    team=team
                ))

        # xG vs actual goals
        if xg > 0:
            if goals > xg + 0.5:
                self.talking_points.append(TalkingPoint(
                    category=InsightCategory.ATTACKING,
                    title="Clinical Finishing",
                    insight=f"Scored {goals} goals from {xg:.2f} xG. Excellent finishing today.",
                    evidence=f"xG: {xg:.2f}, Actual: {goals}",
                    priority=2,
                    team=team
                ))
            elif goals < xg - 0.5:
                self.talking_points.append(TalkingPoint(
                    category=InsightCategory.ATTACKING,
                    title="Underperforming xG",
                    insight=f"Only {goals} goals from {xg:.2f} xG. Need to be more clinical with chances.",
                    evidence=f"xG: {xg:.2f}, Actual: {goals}",
                    priority=1,
                    team=team
                ))

    def _analyze_defending(self, team: str, team_stats: Dict, opponent_stats: Dict):
        """Generate defending-related talking points."""
        tackles = team_stats.get("tackles", 0)
        interceptions = team_stats.get("interceptions", 0)
        clearances = team_stats.get("clearances", 0)

        opp_shots = opponent_stats.get("shots", {}).get("total", 0)
        opp_on_target = opponent_stats.get("shots", {}).get("on_target", 0)
        opp_xg = opponent_stats.get("xG", 0)
        goals_conceded = opponent_stats.get("goals", 0)

        # Defensive actions
        total_defensive = tackles + interceptions
        if total_defensive > 20:
            self.talking_points.append(TalkingPoint(
                category=InsightCategory.DEFENDING,
                title="Active Defending",
                insight=f"High defensive activity with {tackles} tackles and {interceptions} interceptions.",
                evidence=f"Total defensive actions: {total_defensive}",
                priority=2,
                team=team
            ))

        # Opponent chances
        if opp_shots > 15:
            self.talking_points.append(TalkingPoint(
                category=InsightCategory.DEFENDING,
                title="Too Many Shots Conceded",
                insight=f"Allowed {opp_shots} shots. Need to close down shooting opportunities earlier.",
                evidence=f"Opponent shots on target: {opp_on_target}",
                priority=1,
                team=team
            ))

        # xG against
        if opp_xg > 0:
            if goals_conceded < opp_xg - 0.5:
                self.talking_points.append(TalkingPoint(
                    category=InsightCategory.DEFENDING,
                    title="Goalkeeper Performance",
                    insight=f"Conceded {goals_conceded} from {opp_xg:.2f} xG against. Excellent saves made.",
                    evidence=f"xG against: {opp_xg:.2f}",
                    priority=2,
                    team=team
                ))

    def _analyze_set_pieces(self, team: str, team_stats: Dict, opponent_stats: Dict):
        """Generate set piece talking points."""
        corners = team_stats.get("corners", 0)
        free_kicks = team_stats.get("free_kicks", 0)

        opp_corners = opponent_stats.get("corners", 0)
        opp_free_kicks = opponent_stats.get("free_kicks", 0)

        if corners > 7:
            self.talking_points.append(TalkingPoint(
                category=InsightCategory.SET_PIECES,
                title="Good Corner Frequency",
                insight=f"Won {corners} corners. Lots of attacking opportunities from set pieces.",
                evidence=f"Check conversion rate from corners",
                priority=2,
                team=team
            ))

        if opp_corners > 7:
            self.talking_points.append(TalkingPoint(
                category=InsightCategory.SET_PIECES,
                title="Defending Corners",
                insight=f"Defended {opp_corners} corners. Review marking and clearances at set pieces.",
                evidence=f"Opponent corners: {opp_corners}",
                priority=2,
                team=team
            ))

    def _analyze_individuals(self, team: str):
        """Analyze individual player performance."""
        player_stats = self.match_data.get("player_stats", {}).get(team, [])

        for player in player_stats:
            jersey = player.get("jersey_number")
            name = player.get("player_name", f"#{jersey}")

            # Top scorer
            goals = player.get("shooting", {}).get("goals", 0)
            if goals >= 2:
                self.talking_points.append(TalkingPoint(
                    category=InsightCategory.INDIVIDUAL,
                    title=f"{name} - Multiple Goals",
                    insight=f"Scored {goals} goals. Outstanding attacking contribution.",
                    evidence=f"xG: {player.get('shooting', {}).get('xG', 0):.2f}",
                    priority=1,
                    team=team
                ))

            # High pass accuracy
            passes = player.get("passes", {})
            if passes.get("attempted", 0) > 20:
                accuracy = passes.get("accuracy", 0)
                if accuracy > 90:
                    self.talking_points.append(TalkingPoint(
                        category=InsightCategory.INDIVIDUAL,
                        title=f"{name} - Excellent Passing",
                        insight=f"{accuracy:.0f}% pass accuracy. Reliable distribution.",
                        evidence=f"{passes.get('completed')}/{passes.get('attempted')} passes",
                        priority=2,
                        team=team
                    ))

            # Defensive standout
            defending = player.get("defending", {})
            total_def = defending.get("tackles", 0) + defending.get("interceptions", 0)
            if total_def >= 8:
                self.talking_points.append(TalkingPoint(
                    category=InsightCategory.INDIVIDUAL,
                    title=f"{name} - Defensive Standout",
                    insight=f"{total_def} defensive actions (tackles + interceptions).",
                    evidence=f"Tackles: {defending.get('tackles')}, Interceptions: {defending.get('interceptions')}",
                    priority=2,
                    team=team
                ))

    async def ask_question(self, question: str) -> TacticalQuestion:
        """
        Answer a tactical question using AI.

        Uses match data context to provide relevant answers.
        """
        if not self.ai_client:
            # Return rule-based answer without AI
            return self._answer_without_ai(question)

        # Build context from match data
        context = self._build_ai_context()

        # Create prompt for AI
        prompt = f"""You are an expert football coach assistant analyzing a match.

Match Data:
{json.dumps(context, indent=2)}

Coach's Question: {question}

Provide a concise, actionable answer based on the match data. Include specific statistics when relevant.
Focus on practical coaching insights that can be used in team discussions or training.
Keep the answer under 200 words."""

        try:
            if self.ai_provider == "claude":
                response = await self.ai_client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=500,
                    messages=[{"role": "user", "content": prompt}]
                )
                answer = response.content[0].text
            else:  # OpenAI
                response = await self.ai_client.chat.completions.create(
                    model="gpt-4",
                    max_tokens=500,
                    messages=[{"role": "user", "content": prompt}]
                )
                answer = response.choices[0].message.content

            qa = TacticalQuestion(
                question=question,
                answer=answer,
                relevant_data=context,
                confidence=0.85
            )

        except Exception as e:
            qa = TacticalQuestion(
                question=question,
                answer=f"Error generating AI response: {str(e)}. Please try again.",
                relevant_data={},
                confidence=0.0
            )

        self.questions_history.append(qa)
        return qa

    def _answer_without_ai(self, question: str) -> TacticalQuestion:
        """Provide rule-based answer when AI is not available."""
        question_lower = question.lower()

        if not self.match_data:
            return TacticalQuestion(
                question=question,
                answer="No match data available to analyze.",
                relevant_data={},
                confidence=0.0
            )

        stats = self.match_data["stats"]
        home = stats.get("home", {})
        away = stats.get("away", {})

        # Common question patterns
        if "possession" in question_lower:
            answer = f"Home had {home.get('possession_pct', 0):.0f}% possession, Away had {away.get('possession_pct', 0):.0f}%. "
            if home.get('possession_pct', 0) > 55:
                answer += "Home dominated possession and controlled the tempo."
            elif away.get('possession_pct', 0) > 55:
                answer += "Away dominated possession and controlled the tempo."
            else:
                answer += "Possession was fairly balanced."

        elif "shot" in question_lower or "scoring" in question_lower:
            h_shots = home.get('shots', {})
            a_shots = away.get('shots', {})
            answer = f"Home: {h_shots.get('total', 0)} shots ({h_shots.get('on_target', 0)} on target). "
            answer += f"Away: {a_shots.get('total', 0)} shots ({a_shots.get('on_target', 0)} on target). "
            answer += f"xG - Home: {home.get('xG', 0):.2f}, Away: {away.get('xG', 0):.2f}."

        elif "pass" in question_lower:
            answer = f"Pass accuracy - Home: {home.get('pass_accuracy', 0):.0f}%, Away: {away.get('pass_accuracy', 0):.0f}%. "
            h_seq = home.get('pass_sequences', {})
            a_seq = away.get('pass_sequences', {})
            answer += f"Longest passing sequence - Home: {h_seq.get('longest', 0)}, Away: {a_seq.get('longest', 0)}."

        elif "corner" in question_lower or "set piece" in question_lower:
            answer = f"Corners - Home: {home.get('corners', 0)}, Away: {away.get('corners', 0)}. "
            answer += f"Free kicks - Home: {home.get('free_kicks', 0)}, Away: {away.get('free_kicks', 0)}."

        elif "defend" in question_lower:
            answer = f"Home defensive actions: {home.get('tackles', 0)} tackles, {home.get('interceptions', 0)} interceptions, {home.get('clearances', 0)} clearances. "
            answer += f"Away defensive actions: {away.get('tackles', 0)} tackles, {away.get('interceptions', 0)} interceptions, {away.get('clearances', 0)} clearances."

        else:
            # General summary
            answer = self.generate_match_summary()

        return TacticalQuestion(
            question=question,
            answer=answer,
            relevant_data=stats,
            confidence=0.6
        )

    def _build_ai_context(self) -> Dict:
        """Build context dict for AI prompts."""
        if not self.match_data:
            return {}

        return {
            "match_stats": self.match_data["stats"],
            "key_events": self.events_data[:20] if self.events_data else [],
            "talking_points": [
                {"title": tp.title, "insight": tp.insight}
                for tp in self.talking_points[:5]
            ]
        }

    def get_suggested_clips(self, category: Optional[InsightCategory] = None) -> List[Dict]:
        """
        Get suggested clips for review based on analysis.

        Returns event IDs and descriptions for important moments.
        """
        suggestions = []

        # Key events to highlight
        key_types = ["goal", "shot_on_target", "shot", "corner", "free_kick", "possession_change"]

        for event in self.events_data:
            if event.get("type") in key_types:
                suggestions.append({
                    "event_id": event.get("id"),
                    "type": event.get("type"),
                    "timestamp_ms": event.get("timestamp_ms"),
                    "time_str": event.get("time_str"),
                    "team": event.get("team"),
                    "description": event.get("description"),
                    "reason": self._get_clip_reason(event)
                })

        return suggestions[:20]  # Limit to top 20

    def _get_clip_reason(self, event: Dict) -> str:
        """Get reason for suggesting a clip."""
        event_type = event.get("type", "")

        reasons = {
            "goal": "Scoring moment - analyze build-up and finish",
            "shot_on_target": "Shooting opportunity - review decision making",
            "shot": "Shooting attempt - discuss positioning and technique",
            "corner": "Set piece - review delivery and movement",
            "free_kick": "Set piece - analyze positioning and execution",
            "possession_change": "Turnover moment - discuss defensive transition"
        }

        return reasons.get(event_type, "Key moment for review")

    def export_analysis(self) -> Dict:
        """Export complete analysis for sharing/saving."""
        return {
            "generated_at": datetime.now().isoformat(),
            "match_summary": self.generate_match_summary(),
            "talking_points": [asdict(tp) for tp in self.talking_points],
            "suggested_clips": self.get_suggested_clips(),
            "questions_answered": [
                {"question": qa.question, "answer": qa.answer}
                for qa in self.questions_history
            ]
        }

    def reset(self):
        """Reset the service."""
        self.match_data = None
        self.events_data = []
        self.talking_points = []
        self.questions_history = []
        self._pitch_viz = None
        self._player_highlights = None


# Global instance
coach_assist_service = CoachAssistService()
