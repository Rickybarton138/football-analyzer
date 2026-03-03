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

        # Phase 1 analytics data
        self._pass_stats: Optional[Dict] = None
        self._formation_stats: Optional[Dict] = None
        self._xg_data: Optional[Dict] = None
        self._tactical_summary: Optional[Dict] = None

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
        player_highlights=None,
        pass_stats: Optional[Dict] = None,
        formation_stats: Optional[Dict] = None,
        xg_data: Optional[Dict] = None,
        tactical_summary: Optional[Dict] = None
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

        # Store Phase 1 analytics data
        self._pass_stats = pass_stats
        self._formation_stats = formation_stats
        self._xg_data = xg_data
        self._tactical_summary = tactical_summary

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
        prompt = f"""You are an expert FA-qualified football coach. Your job is NOT just to describe
what happened, but to tell the coach WHAT TO DO about it. Every answer should
include: (1) what the data shows, (2) why it matters, (3) what to work on in training.

Match Data:
{json.dumps(context, indent=2)}

Coach's Question: {question}

Provide a concise, actionable answer based on the match data. Include specific statistics when relevant.
Focus on practical coaching insights — drills, exercises, and tactical adjustments.
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
        """Build context dict for AI prompts, enriched with Phase 1 analytics."""
        if not self.match_data:
            return {}

        context = {
            "match_stats": self.match_data["stats"],
            "key_events": self.events_data[:20] if self.events_data else [],
            "talking_points": [
                {"title": tp.title, "insight": tp.insight}
                for tp in self.talking_points[:5]
            ]
        }

        # Enrich with Phase 1 pass analysis
        if self._pass_stats:
            home_p = self._pass_stats.get('home', {})
            away_p = self._pass_stats.get('away', {})
            context["pass_analysis"] = {
                "home_total": home_p.get('total', 0),
                "home_accuracy": home_p.get('accuracy', 0),
                "home_forward_ratio": home_p.get('forward_ratio', 0),
                "away_total": away_p.get('total', 0),
                "away_accuracy": away_p.get('accuracy', 0),
                "away_forward_ratio": away_p.get('forward_ratio', 0),
            }

        # Enrich with formation data
        if self._formation_stats:
            formations = {}
            for team in ['home', 'away']:
                team_f = self._formation_stats.get(team, {})
                if team_f:
                    formations[team] = {
                        "primary_formation": team_f.get('primary_formation', 'unknown'),
                        "formation_changes": team_f.get('formation_changes', 0),
                        "avg_defensive_line": team_f.get('avg_defensive_line', 0),
                        "avg_compactness": team_f.get('avg_compactness', 0),
                    }
            if formations:
                context["formations"] = formations

        # Enrich with xG data
        if self._xg_data:
            total_xg = self._xg_data.get('total_xg', {})
            context["expected_goals"] = {
                "home_xg": total_xg.get('home', 0),
                "away_xg": total_xg.get('away', 0),
                "total_shots": len(self._xg_data.get('shots', [])),
            }

        # Enrich with tactical events
        if self._tactical_summary:
            event_counts = self._tactical_summary.get('event_counts', {})
            top_events = sorted(event_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            context["tactical_events"] = {
                "total_events": self._tactical_summary.get('total_events', 0),
                "top_event_types": {k: v for k, v in top_events},
            }

        return context

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

    def generate_training_focus(self) -> Dict:
        """
        THE DIFFERENTIATOR: Convert match weaknesses into training recommendations.

        Analyzes Phase 1 data to identify priority areas and prescribe
        specific FA-style training drills and session plans.
        """
        priority_areas = []

        # --- Analyze pass stats ---
        if self._pass_stats:
            for team in ['home', 'away']:
                tp = self._pass_stats.get(team, {})
                accuracy = tp.get('accuracy', 100)
                forward_ratio = tp.get('forward_ratio', 0.5)

                if accuracy < 70:
                    priority_areas.append({
                        "area": "Pass Accuracy",
                        "team": team,
                        "severity": "high" if accuracy < 60 else "medium",
                        "metric": f"{accuracy:.0f}%",
                        "drill": "Rondo 4v2 / Directional possession",
                        "detail": "Focus on weight and timing of pass. Progress from 5v2 to 4v2 to increase pressure. Add directional element to simulate match build-up.",
                        "duration_mins": 15,
                    })

                if forward_ratio < 0.25:
                    priority_areas.append({
                        "area": "Progressive Passing",
                        "team": team,
                        "severity": "high" if forward_ratio < 0.15 else "medium",
                        "metric": f"{forward_ratio*100:.0f}% forward passes",
                        "drill": "Progressive passing gates exercise",
                        "detail": "Set up gates at 10m intervals. Players must pass through at least 2 gates before finishing. Reward forward passing with points.",
                        "duration_mins": 15,
                    })

        # --- Analyze formation stats ---
        if self._formation_stats:
            for team in ['home', 'away']:
                tf = self._formation_stats.get(team, {})
                changes = tf.get('formation_changes', 0)
                compactness = tf.get('avg_compactness', 0)

                if changes > 5:
                    priority_areas.append({
                        "area": "Formation Stability",
                        "team": team,
                        "severity": "high" if changes > 8 else "medium",
                        "metric": f"{changes} formation changes",
                        "drill": "Shadow play 11v0, freeze & check",
                        "detail": "Walk through shape in 11v0. Coach calls freeze — players check distances. Progress to shadow play vs 11 mannequins, then opposed.",
                        "duration_mins": 20,
                    })

                if compactness > 35:
                    priority_areas.append({
                        "area": "Team Compactness",
                        "team": team,
                        "severity": "high" if compactness > 45 else "medium",
                        "metric": f"avg {compactness:.0f}m between players",
                        "drill": "Small-sided games on reduced pitch",
                        "detail": "Play 7v7 on 40x30 yard pitch. Enforce maximum 2-touch. Team loses possession if any player is more than 15 yards from nearest teammate.",
                        "duration_mins": 20,
                    })

        # --- Analyze tactical events ---
        if self._tactical_summary:
            event_counts = self._tactical_summary.get('event_counts', {})

            pressing_events = event_counts.get('high_press', 0) + event_counts.get('pressing_trigger', 0)
            if pressing_events < 5:
                priority_areas.append({
                    "area": "Pressing Intensity",
                    "team": "home",
                    "severity": "medium",
                    "metric": f"{pressing_events} pressing actions",
                    "drill": "4v4+4 pressing rehearsal with triggers",
                    "detail": "4 attackers vs 4 defenders with 4 neutral players. Pressing team activates on trigger (bad touch, backward pass). 6-second rule to win ball back.",
                    "duration_mins": 15,
                })

            counter_attacks = event_counts.get('counter_attack', 0) + event_counts.get('counter_attack_conceded', 0)
            if counter_attacks > 3:
                priority_areas.append({
                    "area": "Counter-Press / Transition Defence",
                    "team": "home",
                    "severity": "high" if counter_attacks > 6 else "medium",
                    "metric": f"{counter_attacks} counter-attacks conceded",
                    "drill": "Transition game with counter-press",
                    "detail": "5v5+GKs — on turnover, losing team has 5 seconds to win ball back (counter-press). If they fail, opposition scores double points for transition goal.",
                    "duration_mins": 15,
                })

        # --- Analyze xG data ---
        if self._xg_data:
            total_xg = self._xg_data.get('total_xg', {})
            shots = self._xg_data.get('shots', [])

            for team in ['home', 'away']:
                team_xg = total_xg.get(team, 0)
                team_shots = [s for s in shots if s.get('team') == team]
                team_goals = sum(1 for s in team_shots if s.get('is_goal', False))

                if team_xg > 0 and team_goals < team_xg - 0.5:
                    priority_areas.append({
                        "area": "Finishing / Clinical Conversion",
                        "team": team,
                        "severity": "high",
                        "metric": f"{team_goals} goals from {team_xg:.2f} xG",
                        "drill": "Finishing circuit (3 stations)",
                        "detail": "Station 1: 1v1 vs GK from edge of box. Station 2: Cutback finish from byline cross. Station 3: Quick combination then strike. 2 mins per station, rotate.",
                        "duration_mins": 15,
                    })

            # High xG against — defensive block needed
            for team in ['home', 'away']:
                opp_team = 'away' if team == 'home' else 'home'
                opp_xg = total_xg.get(opp_team, 0)
                if opp_xg > 1.5:
                    priority_areas.append({
                        "area": "Defensive Block / Shot Prevention",
                        "team": team,
                        "severity": "high" if opp_xg > 2.0 else "medium",
                        "metric": f"{opp_xg:.2f} xG conceded",
                        "drill": "Defensive block work",
                        "detail": "11v11 on half pitch. Defending team sets block (2 banks of 4 + 2 strikers). Attacking team must score within 8 passes. Focus on compactness, no gaps between lines.",
                        "duration_mins": 20,
                    })

        # Sort by severity (high first)
        severity_order = {'high': 0, 'medium': 1, 'low': 2}
        priority_areas.sort(key=lambda x: severity_order.get(x.get('severity', 'low'), 2))

        # Build session plan from top priorities
        session_plan = self._build_session_plan(priority_areas)

        return {
            "priority_areas": priority_areas,
            "session_plan": session_plan,
            "generated_at": datetime.now().isoformat(),
        }

    def _build_session_plan(self, priority_areas: List[Dict]) -> Dict:
        """Build a structured training session plan from priority areas."""
        # Pick top priorities for main and secondary focus
        high_priorities = [p for p in priority_areas if p.get('severity') == 'high']
        medium_priorities = [p for p in priority_areas if p.get('severity') == 'medium']

        main_focus = high_priorities[0] if high_priorities else (medium_priorities[0] if medium_priorities else None)
        secondary_focus = high_priorities[1] if len(high_priorities) > 1 else (
            medium_priorities[0] if medium_priorities and main_focus not in medium_priorities else
            (medium_priorities[1] if len(medium_priorities) > 1 else None)
        )

        return {
            "warm_up": {
                "activity": "Dynamic warm-up with ball (passing pairs → movement patterns)",
                "duration_mins": 10,
            },
            "main_focus": {
                "area": main_focus['area'] if main_focus else "General play",
                "drill": main_focus['drill'] if main_focus else "Possession game",
                "detail": main_focus['detail'] if main_focus else "Keep ball in 6v4. Focus on quality.",
                "duration_mins": main_focus.get('duration_mins', 20) if main_focus else 20,
            },
            "secondary_focus": {
                "area": secondary_focus['area'] if secondary_focus else "Team shape",
                "drill": secondary_focus['drill'] if secondary_focus else "Shadow play",
                "detail": secondary_focus['detail'] if secondary_focus else "Walk through team shape.",
                "duration_mins": secondary_focus.get('duration_mins', 15) if secondary_focus else 15,
            },
            "game": {
                "activity": "Conditioned match — apply session focus in game context",
                "conditions": f"Bonus point for demonstrating {main_focus['area'].lower()}" if main_focus else "Free play",
                "duration_mins": 20,
            },
            "cool_down": {
                "activity": "Light jog, static stretches, group review of session focus",
                "duration_mins": 10,
            },
        }

    def reset(self):
        """Reset the service."""
        self.match_data = None
        self.events_data = []
        self.talking_points = []
        self.questions_history = []
        self._pitch_viz = None
        self._player_highlights = None
        self._pass_stats = None
        self._formation_stats = None
        self._xg_data = None
        self._tactical_summary = None


# Global instance
coach_assist_service = CoachAssistService()
