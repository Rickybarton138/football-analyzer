"""
AI Expert Coach System

The unified coaching intelligence system that differentiates us from VEO/HUDL.
Combines:
- Technical analysis framework (UEFA/FA methodology)
- Vision AI clip analysis
- Personalized player development plans
- Team tactical recommendations
- **CUSTOMIZABLE TEAM PHILOSOPHY** - learns your specific team's style

This is the "brain" that turns video data into actionable coaching intelligence.
It adapts to YOUR team's philosophy, formation, and individual player needs.
"""
import asyncio
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json

from ai.technical_analysis import (
    TECHNIQUE_LIBRARY,
    POSITION_REQUIREMENTS,
    SkillTechnique,
    SkillCategory,
    FourCornerDomain,
    get_technique_for_action,
    get_position_requirements,
    get_coaching_points_for_mistake,
    get_drills_for_weakness
)
from ai.vision_coach import VisionCoachService, ActionAnalysis, PlayerTechnicalProfile
from ai.team_profile import (
    TeamProfileService,
    TeamProfile,
    PlayerProfile as TeamPlayerProfile,
    PRINCIPLES_OF_PLAY,
    FOUR_MOMENTS,
    FORMATION_TEMPLATES,
    PlayingStyle,
    create_possession_team_profile,
    create_counter_attack_team_profile,
    create_high_press_team_profile
)


class DevelopmentPriority(str, Enum):
    """Priority levels for development areas."""
    CRITICAL = "critical"  # Fundamental issue affecting all play
    HIGH = "high"  # Significant weakness limiting effectiveness
    MEDIUM = "medium"  # Area for improvement
    LOW = "low"  # Minor refinement


@dataclass
class CoachingPoint:
    """A specific coaching intervention."""
    point: str  # The coaching instruction
    category: FourCornerDomain  # Which corner it addresses
    technique_area: str  # Specific skill area
    coaching_cue: str  # Short memorable phrase
    drill_to_address: str  # Recommended drill
    priority: DevelopmentPriority


@dataclass
class PlayerDevelopmentPlan:
    """Personalized development plan for a player."""
    player_id: int
    player_name: Optional[str]
    position: Optional[str]
    created_at: str

    # Assessment summary
    overall_rating: float  # 0-10
    four_corner_ratings: Dict[str, float]  # Domain -> rating

    # Strengths to maintain
    key_strengths: List[str]

    # Development areas
    development_priorities: List[CoachingPoint]

    # Weekly focus
    weekly_focus: str
    weekly_drills: List[str]

    # Long-term goals
    three_month_goals: List[str]
    six_month_goals: List[str]

    # Session notes
    recent_observations: List[str]


@dataclass
class TeamTacticalBrief:
    """Tactical brief for team/unit coaching."""
    team_name: str
    created_at: str

    # Formation analysis
    primary_formation: str
    formation_effectiveness: float
    formation_notes: List[str]

    # Unit assessments
    defensive_unit_rating: float
    midfield_unit_rating: float
    attacking_unit_rating: float

    # Key patterns identified
    attacking_patterns: List[str]
    defensive_patterns: List[str]
    transition_patterns: List[str]

    # Areas to develop
    team_weaknesses: List[str]
    unit_specific_work: Dict[str, List[str]]  # unit -> coaching points

    # Session plan
    recommended_session_focus: str
    drill_sequence: List[str]


class ExpertCoachService:
    """
    The AI Expert Coach - providing professional-level coaching intelligence.

    This service:
    1. Analyzes individual player technique using Vision AI
    2. Generates personalized development plans
    3. Provides UEFA-standard coaching feedback
    4. Creates team tactical recommendations
    5. Tracks progress over time
    6. **ADAPTS TO YOUR TEAM'S PHILOSOPHY** - customizable to your style

    The "secret weapon" - an AI coach that actually understands YOUR team.
    """

    def __init__(self, storage_path: str = "team_profiles"):
        self.vision_coach = VisionCoachService()
        self.player_plans: Dict[int, PlayerDevelopmentPlan] = {}
        self.team_briefs: List[TeamTacticalBrief] = []

        # Team profile system - THE KEY DIFFERENTIATOR
        self.team_profile_service = TeamProfileService(storage_path)

        # Analysis history for trends
        self.analysis_history: Dict[int, List[ActionAnalysis]] = {}

    async def initialize(self, api_key: str, provider: str = "claude"):
        """Initialize the coaching system with Vision AI."""
        await self.vision_coach.initialize(api_key, provider)

    # =========================================================================
    # TEAM PROFILE MANAGEMENT - Customization for YOUR team
    # =========================================================================

    def create_team_profile(
        self,
        team_name: str,
        playing_style: str = "balanced",
        formation: str = "4-3-3"
    ) -> TeamProfile:
        """
        Create a new team profile. This is where you define YOUR team's identity.

        Args:
            team_name: Your team's name
            playing_style: One of: possession, counter_attack, high_press, low_block,
                          direct_play, balanced, positional_play
            formation: e.g., "4-3-3", "4-4-2", "3-5-2"
        """
        return self.team_profile_service.create_profile(team_name, playing_style, formation)

    def create_from_template(self, team_name: str, template: str) -> TeamProfile:
        """
        Create team profile from a pre-built template.

        Templates:
        - "possession": Tiki-taka style, patient build-up, high pressing
        - "counter_attack": Compact defense, fast transitions, direct play
        - "high_press": Gegenpressing, win ball high, immediate pressure
        """
        if template == "possession":
            profile = create_possession_team_profile(team_name)
        elif template == "counter_attack":
            profile = create_counter_attack_team_profile(team_name)
        elif template == "high_press":
            profile = create_high_press_team_profile(team_name)
        else:
            profile = self.team_profile_service.create_profile(team_name)

        self.team_profile_service.current_profile = profile
        return profile

    def set_team_philosophy(
        self,
        philosophy_statement: str,
        team_motto: str = "",
        **style_settings
    ) -> None:
        """
        Define your coaching philosophy in your own words.

        Args:
            philosophy_statement: Your vision for how the team plays (free text)
            team_motto: A short inspiring phrase (e.g., "Play with courage")
            **style_settings: Any of:
                - playing_style: possession, counter_attack, high_press, etc.
                - build_up_style: short_from_gk, long_from_gk, mixed, etc.
                - pressing_intensity: ultra_high, high, medium, low, trigger_based
                - attacking_focus: central_overload, wide_overload, combination_play, etc.
                - defensive_shape: flat_four, high_line, mid_block, low_block, etc.
        """
        if not self.team_profile_service.current_profile:
            raise ValueError("Create a team profile first with create_team_profile()")

        self.team_profile_service.set_philosophy_statement(philosophy_statement)

        if team_motto:
            self.team_profile_service.current_profile.philosophy.team_motto = team_motto

        if style_settings:
            self.team_profile_service.update_philosophy(**style_settings)

    def set_principles_priority(
        self,
        attacking_principles: List[str],
        defending_principles: List[str]
    ) -> None:
        """
        Set your priority order for principles of play.

        Attacking options: penetration, depth, width, mobility, improvisation
        Defending options: pressure, cover, compactness, delay, control_restraint

        Order matters - first = highest priority in your philosophy
        """
        self.team_profile_service.set_attacking_principles_priority(attacking_principles)
        self.team_profile_service.set_defending_principles_priority(defending_principles)

    def add_team_player(
        self,
        jersey_number: int,
        name: str,
        position: str,
        characteristics: List[str] = None,
        development_focus: List[str] = None,
        notes: str = "",
        age_group: str = "senior"
    ) -> TeamPlayerProfile:
        """
        Add a player to your team profile for personalized coaching.

        Args:
            jersey_number: Shirt number (1-99)
            name: Player's name
            position: Primary position (GK, RB, CB, LB, CDM, CM, CAM, LW, RW, ST, etc.)
            characteristics: Player traits (e.g., ["left-footed", "quick", "good in air"])
            development_focus: Areas to prioritize (e.g., ["weak foot", "defensive awareness"])
            notes: Your coaching notes about this player
            age_group: u9, u12, u14, u16, u18, senior (affects coaching language)
        """
        return self.team_profile_service.add_player(
            jersey_number=jersey_number,
            name=name,
            position=position,
            characteristics=characteristics or [],
            development_focus=development_focus or [],
            notes=notes,
            age_group=age_group
        )

    def update_player_profile(
        self,
        jersey_number: int,
        **updates
    ) -> TeamPlayerProfile:
        """Update a player's profile with new information."""
        return self.team_profile_service.update_player(jersey_number, **updates)

    def set_player_four_corner_ratings(
        self,
        jersey_number: int,
        technical: float,
        tactical: float,
        physical: float,
        psychological: float
    ) -> None:
        """
        Set your assessment of a player's Four Corner ratings (1-10 scale).

        This helps the AI coach understand the player's current level and
        provide appropriate, targeted feedback.
        """
        self.team_profile_service.update_player_four_corner(
            jersey_number,
            {
                "technical": technical,
                "tactical": tactical,
                "physical": physical,
                "psychological": psychological
            }
        )

    def add_positional_instruction(
        self,
        position: str,
        instruction_key: str,
        value: str
    ) -> None:
        """
        Add specific instructions for a position in your system.

        Examples:
            add_positional_instruction("RB", "attacking_runs", "overlap")
            add_positional_instruction("CDM", "defensive_priority", "screen back four")
            add_positional_instruction("ST", "pressing_trigger", "ball to CB")
        """
        self.team_profile_service.set_positional_instruction(position, instruction_key, value)

    def add_key_partnership(
        self,
        positions: List[str],
        pattern: str,
        description: str = ""
    ) -> None:
        """
        Define key partnerships or combinations you want to develop.

        Examples:
            add_key_partnership(["CM", "ST"], "through ball runs", "CM looks for ST's runs")
            add_key_partnership(["RW", "RB"], "overlap combination", "RW holds, RB overlaps")
        """
        self.team_profile_service.add_key_partnership(positions, pattern, description)

    def set_season_objectives(self, objectives: List[str]) -> None:
        """Set your team's season objectives."""
        if not self.team_profile_service.current_profile:
            raise ValueError("Create a team profile first")
        self.team_profile_service.current_profile.season_objectives = objectives

    def set_development_themes(self, themes: List[str]) -> None:
        """Set development themes for the season (e.g., "Playing out from back", "Counter-pressing")."""
        if not self.team_profile_service.current_profile:
            raise ValueError("Create a team profile first")
        self.team_profile_service.current_profile.development_themes = themes

    def save_team_profile(self, filename: str = None) -> str:
        """Save the current team profile to disk."""
        return self.team_profile_service.save_profile(filename=filename)

    def load_team_profile(self, filename: str) -> TeamProfile:
        """Load a saved team profile."""
        return self.team_profile_service.load_profile(filename)

    def list_saved_profiles(self) -> List[str]:
        """List all saved team profiles."""
        return self.team_profile_service.list_profiles()

    def get_team_context(self) -> str:
        """Get the full team context for AI prompts."""
        return self.team_profile_service.generate_coaching_context()

    def get_player_context(self, jersey_number: int) -> str:
        """Get specific player context for AI prompts."""
        return self.team_profile_service.generate_player_context(jersey_number)

    def get_match_analysis_context(self) -> str:
        """Get context for match analysis based on team philosophy."""
        return self.team_profile_service.generate_match_analysis_context()

    # =========================================================================
    # ENHANCED ANALYSIS - Now using team context
    # =========================================================================

    async def analyze_player_action(
        self,
        video_path: str,
        start_frame: int,
        end_frame: int,
        action_type: str,
        player_jersey: int,
        player_name: Optional[str] = None,
        player_position: Optional[str] = None
    ) -> Dict:
        """
        Analyze a player action and return professional coaching feedback.

        This is the main entry point for individual player analysis.
        Now enhanced with team philosophy context for personalized coaching.
        """
        # Get team context if available
        team_context = self.get_team_context()
        player_context = self.get_player_context(player_jersey)

        # Get player profile data if available
        team_player = None
        if self.team_profile_service.current_profile:
            team_player = self.team_profile_service.current_profile.get_player(player_jersey)
            if team_player:
                player_name = player_name or team_player.name
                player_position = player_position or team_player.position

        # Build enhanced context for vision coach
        enhanced_context = ""
        if team_context and team_context != "No team profile configured.":
            enhanced_context = f"""
=== TEAM CONTEXT ===
{team_context}

=== PLAYER-SPECIFIC CONTEXT ===
{player_context}

IMPORTANT: Provide coaching feedback that aligns with the team's philosophy and
the specific development needs of this player. Reference the team's playing style
and principles when making recommendations.
"""

        # Run vision analysis with team context
        analysis = await self.vision_coach.analyze_action_clip(
            video_path=video_path,
            start_frame=start_frame,
            end_frame=end_frame,
            action_type=action_type,
            player_jersey=player_jersey,
            player_name=player_name,
            player_position=player_position,
            additional_context=enhanced_context
        )

        # Store for history
        if player_jersey not in self.analysis_history:
            self.analysis_history[player_jersey] = []
        self.analysis_history[player_jersey].append(analysis)

        # Generate coaching response with team context
        coaching_response = self._generate_coaching_response(
            analysis, player_position, team_player
        )

        return coaching_response

    def _generate_coaching_response(
        self,
        analysis: ActionAnalysis,
        player_position: Optional[str],
        team_player: Optional[TeamPlayerProfile] = None
    ) -> Dict:
        """Generate a professional coaching response from analysis, personalized to the team."""

        # Get technique framework for additional context
        technique = get_technique_for_action(analysis.action_type)

        # Build coaching points using UEFA methodology
        coaching_points = []
        for correction in analysis.specific_corrections[:3]:
            # Find relevant coaching cue from framework
            cue = self._find_coaching_cue(correction, technique)
            drill = self._recommend_drill(correction, technique)

            coaching_points.append({
                "intervention": correction,
                "coaching_cue": cue,
                "recommended_drill": drill,
                "methodology": "UEFA Positive Instruction"
            })

        # Build Four Corner assessment
        four_corner = self._build_four_corner_assessment(analysis.technique_scores)

        # Get position-specific context
        position_context = None
        if player_position:
            pos_req = get_position_requirements(player_position)
            if pos_req:
                position_context = {
                    "position": pos_req.position,
                    "key_skills_for_position": pos_req.primary_skills,
                    "relevance_of_action": self._assess_action_relevance(
                        analysis.action_type, pos_req
                    )
                }

        # Build team context section
        team_context_section = None
        if self.team_profile_service.current_profile:
            profile = self.team_profile_service.current_profile
            philosophy = profile.philosophy

            team_context_section = {
                "team_name": philosophy.team_name,
                "playing_style": philosophy.playing_style.value,
                "philosophy_alignment": self._assess_philosophy_alignment(analysis, philosophy),
                "principle_relevance": self._get_relevant_principles(analysis.action_type, philosophy)
            }

        # Build player-specific context
        player_profile_context = None
        if team_player:
            player_profile_context = {
                "name": team_player.name,
                "age_group": team_player.age_group,
                "development_focus": team_player.development_focus,
                "characteristics": team_player.characteristics,
                "coach_notes": team_player.notes,
                "four_corner_baseline": team_player.four_corner_ratings,
                "coaching_adaptations": self._get_age_appropriate_coaching(team_player.age_group)
            }

        return {
            "summary": {
                "action_type": analysis.action_type,
                "player": {
                    "jersey": analysis.player_jersey,
                    "name": analysis.player_name,
                    "position": player_position
                },
                "overall_score": round(analysis.overall_score * 10, 1),
                "confidence": analysis.confidence
            },
            "four_corner_assessment": four_corner,
            "technical_breakdown": {
                "scores": analysis.technique_scores,
                "strengths": analysis.strengths,
                "areas_for_development": analysis.weaknesses
            },
            "coaching_feedback": {
                "ai_observations": analysis.ai_observations,
                "coaching_points": coaching_points,
                "priority_focus": analysis.weaknesses[0] if analysis.weaknesses else None
            },
            "training_recommendations": {
                "drills": analysis.recommended_drills,
                "focus_areas": analysis.weaknesses[:2],
                "session_suggestions": self._suggest_session_plan(analysis)
            },
            "position_context": position_context,
            "team_context": team_context_section,
            "player_profile": player_profile_context,
            "methodology": {
                "framework": "UEFA Coaching License / FA Four Corner Model",
                "analysis_standard": "UEFA Pro License Assessment Criteria",
                "personalized": team_context_section is not None
            },
            "analyzed_at": analysis.analyzed_at
        }

    def _find_coaching_cue(
        self,
        correction: str,
        technique: Optional[SkillTechnique]
    ) -> str:
        """Find a relevant coaching cue from the technique framework."""
        if not technique:
            return "Focus on this aspect"

        correction_lower = correction.lower()

        for checkpoint in technique.checkpoints:
            # Check if any keywords match
            for what_to_look_for in checkpoint.what_to_look_for:
                if any(word in correction_lower for word in what_to_look_for.lower().split()[:3]):
                    return checkpoint.coaching_cues[0] if checkpoint.coaching_cues else "Focus on this"

            for mistake in checkpoint.common_mistakes:
                if any(word in correction_lower for word in mistake.lower().split()[:3]):
                    return checkpoint.coaching_cues[0] if checkpoint.coaching_cues else "Focus on this"

        return technique.checkpoints[0].coaching_cues[0] if technique.checkpoints else "Focus on this"

    def _recommend_drill(
        self,
        correction: str,
        technique: Optional[SkillTechnique]
    ) -> str:
        """Recommend a specific drill based on the correction needed."""
        if technique and technique.drills_to_improve:
            # Simple matching based on keywords
            correction_lower = correction.lower()

            for drill in technique.drills_to_improve:
                drill_words = drill.lower().split()
                if any(word in correction_lower for word in drill_words[:3]):
                    return drill

            return technique.drills_to_improve[0]

        return "Technique repetition with feedback"

    def _build_four_corner_assessment(self, scores: Dict[str, float]) -> Dict:
        """Build Four Corner Model assessment from scores."""
        # Map scores to corners
        technical_scores = []
        tactical_scores = []
        physical_scores = []
        psychological_scores = []

        score_mapping = {
            "body position": FourCornerDomain.PHYSICAL,
            "balance": FourCornerDomain.PHYSICAL,
            "coordination": FourCornerDomain.PHYSICAL,
            "execution": FourCornerDomain.TECHNICAL,
            "contact": FourCornerDomain.TECHNICAL,
            "surface": FourCornerDomain.TECHNICAL,
            "follow through": FourCornerDomain.TECHNICAL,
            "timing": FourCornerDomain.TACTICAL,
            "decision": FourCornerDomain.TACTICAL,
            "awareness": FourCornerDomain.TACTICAL,
            "scanning": FourCornerDomain.TACTICAL,
            "composure": FourCornerDomain.PSYCHOLOGICAL,
            "confidence": FourCornerDomain.PSYCHOLOGICAL,
        }

        for name, score in scores.items():
            name_lower = name.lower()
            assigned = False

            for key, domain in score_mapping.items():
                if key in name_lower:
                    if domain == FourCornerDomain.TECHNICAL:
                        technical_scores.append(score)
                    elif domain == FourCornerDomain.TACTICAL:
                        tactical_scores.append(score)
                    elif domain == FourCornerDomain.PHYSICAL:
                        physical_scores.append(score)
                    elif domain == FourCornerDomain.PSYCHOLOGICAL:
                        psychological_scores.append(score)
                    assigned = True
                    break

            if not assigned:
                technical_scores.append(score)

        def avg(lst):
            return sum(lst) / len(lst) if lst else 5.0

        return {
            "technical": {
                "score": round(avg(technical_scores), 1),
                "description": "Ball mastery, technique execution"
            },
            "tactical": {
                "score": round(avg(tactical_scores), 1),
                "description": "Decision making, awareness"
            },
            "physical": {
                "score": round(avg(physical_scores), 1),
                "description": "Balance, coordination, movement"
            },
            "psychological": {
                "score": round(avg(psychological_scores), 1),
                "description": "Composure, confidence"
            }
        }

    def _assess_action_relevance(self, action_type: str, pos_req) -> str:
        """Assess how relevant this action type is for the player's position."""
        action_lower = action_type.lower()

        if action_lower in [s.lower() for s in pos_req.primary_skills]:
            return "Core skill for this position - high priority"
        elif action_lower in [s.lower() for s in pos_req.secondary_skills]:
            return "Secondary skill for this position - moderate priority"
        else:
            return "Supplementary skill for this position"

    def _suggest_session_plan(self, analysis: ActionAnalysis) -> List[str]:
        """Suggest a mini session plan to address weaknesses."""
        technique = get_technique_for_action(analysis.action_type)

        if not analysis.weaknesses:
            return ["Continue current training program"]

        session = [
            f"Warm-up: Ball mastery exercises focusing on {analysis.action_type}",
            f"Technical: {analysis.recommended_drills[0] if analysis.recommended_drills else 'Technique repetition'}",
            f"Progression: Add passive then active defender",
            f"Game-related: Small-sided game emphasizing {analysis.action_type} opportunities",
            f"Cool-down: Review video clips with player, discuss coaching points"
        ]

        return session

    def _assess_philosophy_alignment(self, analysis: ActionAnalysis, philosophy) -> Dict:
        """Assess how well the action aligns with team philosophy."""
        action_lower = analysis.action_type.lower()

        # Determine relevance based on playing style
        style_relevance = {
            PlayingStyle.POSSESSION: {
                "high": ["pass", "first touch", "control", "receiving"],
                "medium": ["dribble", "turn"],
                "low": ["long ball", "clearance"]
            },
            PlayingStyle.COUNTER_ATTACK: {
                "high": ["through ball", "long pass", "sprint", "direct pass"],
                "medium": ["first touch", "pass"],
                "low": ["short pass", "keep ball"]
            },
            PlayingStyle.HIGH_PRESS: {
                "high": ["press", "tackle", "interception", "recovery"],
                "medium": ["first touch", "pass"],
                "low": []
            },
            PlayingStyle.DIRECT_PLAY: {
                "high": ["heading", "long ball", "cross", "aerial"],
                "medium": ["shooting", "first touch"],
                "low": ["short pass", "possession"]
            }
        }

        style_map = style_relevance.get(philosophy.playing_style, {})
        relevance = "medium"  # default

        for level, keywords in style_map.items():
            if any(kw in action_lower for kw in keywords):
                relevance = level
                break

        return {
            "relevance_to_style": relevance,
            "playing_style": philosophy.playing_style.value,
            "coaching_emphasis": self._get_style_emphasis(philosophy.playing_style, analysis.action_type)
        }

    def _get_style_emphasis(self, playing_style: PlayingStyle, action_type: str) -> str:
        """Get coaching emphasis based on team's playing style."""
        emphasis_map = {
            PlayingStyle.POSSESSION: f"Focus on clean execution to maintain possession. In our system, every {action_type} should create a passing option.",
            PlayingStyle.COUNTER_ATTACK: f"Speed and directness are key. Execute {action_type} quickly to exploit space before defense recovers.",
            PlayingStyle.HIGH_PRESS: f"Aggressiveness with control. After {action_type}, look to press immediately or support pressing teammate.",
            PlayingStyle.LOW_BLOCK: f"Safety first. Ensure {action_type} doesn't put team at risk. Security over ambition.",
            PlayingStyle.DIRECT_PLAY: f"Be decisive. {action_type.title()} should aim to progress ball quickly toward goal.",
            PlayingStyle.POSITIONAL_PLAY: f"Execute {action_type} to create positional superiority. Look for third man opportunities.",
            PlayingStyle.BALANCED: f"Read the game. Choose when to be direct and when to be patient with your {action_type}."
        }
        return emphasis_map.get(playing_style, f"Execute {action_type} with quality and purpose.")

    def _get_relevant_principles(self, action_type: str, philosophy) -> List[Dict]:
        """Get principles of play relevant to this action."""
        action_lower = action_type.lower()
        relevant = []

        # Map actions to principles
        action_principle_map = {
            "attacking": {
                "pass": ["penetration", "depth", "width"],
                "shot": ["penetration", "improvisation"],
                "dribble": ["penetration", "improvisation", "mobility"],
                "cross": ["width", "penetration"],
                "first touch": ["depth", "mobility"],
                "movement": ["mobility", "depth", "width"]
            },
            "defending": {
                "tackle": ["pressure", "control_restraint"],
                "interception": ["compactness", "cover"],
                "press": ["pressure", "compactness"],
                "block": ["delay", "cover"],
                "recovery": ["cover", "delay"]
            }
        }

        # Find matching principles
        for phase, mappings in action_principle_map.items():
            for action_key, principles in mappings.items():
                if action_key in action_lower:
                    for principle_key in principles:
                        principle_data = PRINCIPLES_OF_PLAY.get(phase, {}).get(principle_key)
                        if principle_data:
                            # Check if this principle is prioritized by the team
                            priority_list = (philosophy.attacking_principles_priority
                                           if phase == "attacking"
                                           else philosophy.defending_principles_priority)
                            priority_rank = (priority_list.index(principle_key) + 1
                                           if principle_key in priority_list
                                           else None)

                            relevant.append({
                                "principle": principle_data["name"],
                                "description": principle_data["description"],
                                "team_priority_rank": priority_rank,
                                "uefa_guidance": principle_data["uefa_guidance"]
                            })

        # Sort by team priority
        relevant.sort(key=lambda x: x.get("team_priority_rank") or 99)
        return relevant[:3]  # Top 3 most relevant

    def _get_age_appropriate_coaching(self, age_group: str) -> Dict:
        """Get age-appropriate coaching adaptations based on UEFA guidelines."""
        adaptations = {
            "u9": {
                "language_style": "Simple, positive, encouraging",
                "focus_areas": ["Fun and enjoyment", "Basic movement skills", "Ball familiarity"],
                "session_duration": "Short activities (5-10 mins each)",
                "feedback_approach": "Praise effort over outcome",
                "technical_expectations": "Process over product - let them explore",
                "tactical_depth": "Very basic - mostly let them play"
            },
            "u12": {
                "language_style": "Clear, visual demonstrations, questions to prompt thinking",
                "focus_areas": ["Technical foundations", "1v1 situations", "Small-sided games"],
                "session_duration": "15-20 minute activities",
                "feedback_approach": "Sandwich method - positive, correction, positive",
                "technical_expectations": "Building core techniques with repetition",
                "tactical_depth": "Basic principles in small groups"
            },
            "u14": {
                "language_style": "More detailed instruction, encourage problem-solving",
                "focus_areas": ["Technical refinement", "Introduction to tactics", "Physical development"],
                "session_duration": "20-25 minute activities",
                "feedback_approach": "Ask questions before telling, guided discovery",
                "technical_expectations": "Technique under moderate pressure",
                "tactical_depth": "Unit play, basic team shape"
            },
            "u16": {
                "language_style": "Tactical discussions, video analysis, peer feedback",
                "focus_areas": ["Position-specific skills", "Tactical understanding", "Physical conditioning"],
                "session_duration": "25-30 minute activities",
                "feedback_approach": "Detailed technical analysis with video",
                "technical_expectations": "Technique under pressure with consistency",
                "tactical_depth": "Full tactical understanding, game management"
            },
            "u18": {
                "language_style": "Professional, detailed, player-led learning",
                "focus_areas": ["Elite technical execution", "Advanced tactics", "Mental preparation"],
                "session_duration": "Full session blocks",
                "feedback_approach": "Analytical, data-driven where appropriate",
                "technical_expectations": "Match-level execution in training",
                "tactical_depth": "Complete game understanding, adaptability"
            },
            "senior": {
                "language_style": "Direct, professional, collaborative",
                "focus_areas": ["Maintenance and refinement", "Tactical flexibility", "Leadership"],
                "session_duration": "Varied based on periodization",
                "feedback_approach": "Honest, specific, actionable",
                "technical_expectations": "Consistent elite execution",
                "tactical_depth": "Full strategic awareness, in-game adjustments"
            }
        }
        return adaptations.get(age_group, adaptations["senior"])

    async def generate_development_plan(
        self,
        player_jersey: int,
        player_name: Optional[str] = None,
        player_position: Optional[str] = None
    ) -> PlayerDevelopmentPlan:
        """
        Generate a comprehensive development plan for a player.

        Based on all accumulated analysis data for this player.
        """
        # Get player profile from vision coach
        profile = self.vision_coach.player_profiles.get(player_jersey)

        if not profile:
            # Create minimal plan if no data
            return PlayerDevelopmentPlan(
                player_id=player_jersey,
                player_name=player_name,
                position=player_position,
                created_at=datetime.now().isoformat(),
                overall_rating=5.0,
                four_corner_ratings={
                    "technical": 5.0,
                    "tactical": 5.0,
                    "physical": 5.0,
                    "psychological": 5.0
                },
                key_strengths=["More data needed for assessment"],
                development_priorities=[],
                weekly_focus="Build baseline data through match/training analysis",
                weekly_drills=["Ball mastery", "Position-specific work"],
                three_month_goals=["Establish technical baseline"],
                six_month_goals=["Identify and address primary weakness"],
                recent_observations=[]
            )

        # Calculate Four Corner ratings from analyses
        four_corner_ratings = self._calculate_four_corner_from_profile(profile)

        # Identify key strengths
        key_strengths = profile.consistent_strengths[:5] if profile.consistent_strengths else []

        # Build development priorities
        development_priorities = self._build_development_priorities(profile)

        # Generate weekly focus
        weekly_focus = self._determine_weekly_focus(profile, development_priorities)

        # Get position-specific recommendations
        pos_req = get_position_requirements(player_position) if player_position else None

        plan = PlayerDevelopmentPlan(
            player_id=player_jersey,
            player_name=player_name or profile.player_name,
            position=player_position or profile.position,
            created_at=datetime.now().isoformat(),
            overall_rating=self._calculate_overall_rating(four_corner_ratings),
            four_corner_ratings=four_corner_ratings,
            key_strengths=key_strengths,
            development_priorities=development_priorities,
            weekly_focus=weekly_focus,
            weekly_drills=self._recommend_weekly_drills(development_priorities, pos_req),
            three_month_goals=self._set_short_term_goals(development_priorities),
            six_month_goals=self._set_medium_term_goals(development_priorities, pos_req),
            recent_observations=[
                a.coaching_feedback[0] if a.coaching_feedback else "No specific observation"
                for a in profile.action_analyses[-5:]
            ]
        )

        # Store plan
        self.player_plans[player_jersey] = plan

        return plan

    def _calculate_four_corner_from_profile(
        self,
        profile: PlayerTechnicalProfile
    ) -> Dict[str, float]:
        """Calculate Four Corner ratings from player profile."""
        technical_scores = []
        tactical_scores = []
        physical_scores = []
        psychological_scores = []

        for analysis in profile.action_analyses:
            if analysis.technique_scores:
                fc = self._build_four_corner_assessment(analysis.technique_scores)
                technical_scores.append(fc["technical"]["score"])
                tactical_scores.append(fc["tactical"]["score"])
                physical_scores.append(fc["physical"]["score"])
                psychological_scores.append(fc["psychological"]["score"])

        def avg(lst):
            return round(sum(lst) / len(lst), 1) if lst else 5.0

        return {
            "technical": avg(technical_scores),
            "tactical": avg(tactical_scores),
            "physical": avg(physical_scores),
            "psychological": avg(psychological_scores)
        }

    def _calculate_overall_rating(self, four_corner: Dict[str, float]) -> float:
        """Calculate overall rating from Four Corner scores."""
        scores = list(four_corner.values())
        return round(sum(scores) / len(scores), 1) if scores else 5.0

    def _build_development_priorities(
        self,
        profile: PlayerTechnicalProfile
    ) -> List[CoachingPoint]:
        """Build prioritized development points."""
        priorities = []

        for weakness in profile.recurring_weaknesses[:5]:
            # Determine category and priority
            category = self._categorize_weakness(weakness)
            priority = self._assess_priority(weakness, profile)

            # Find relevant technique and drill
            technique = None
            for analysis in profile.action_analyses:
                technique = get_technique_for_action(analysis.action_type)
                if technique:
                    break

            coaching_cue = self._find_coaching_cue(weakness, technique) if technique else "Focus on improvement"
            drill = self._recommend_drill(weakness, technique) if technique else "Technique repetition"

            priorities.append(CoachingPoint(
                point=weakness,
                category=category,
                technique_area=technique.skill_name if technique else "General",
                coaching_cue=coaching_cue,
                drill_to_address=drill,
                priority=priority
            ))

        return priorities

    def _categorize_weakness(self, weakness: str) -> FourCornerDomain:
        """Categorize a weakness into Four Corner domains."""
        weakness_lower = weakness.lower()

        tactical_keywords = ["decision", "awareness", "scan", "timing", "option", "choice"]
        physical_keywords = ["balance", "body", "position", "strength", "speed", "coordination"]
        psychological_keywords = ["composure", "confidence", "pressure", "concentration"]

        if any(kw in weakness_lower for kw in tactical_keywords):
            return FourCornerDomain.TACTICAL
        elif any(kw in weakness_lower for kw in physical_keywords):
            return FourCornerDomain.PHYSICAL
        elif any(kw in weakness_lower for kw in psychological_keywords):
            return FourCornerDomain.PSYCHOLOGICAL
        else:
            return FourCornerDomain.TECHNICAL

    def _assess_priority(
        self,
        weakness: str,
        profile: PlayerTechnicalProfile
    ) -> DevelopmentPriority:
        """Assess priority level of a weakness."""
        # Count how often this weakness appears
        count = sum(
            1 for a in profile.action_analyses
            if any(weakness.lower() in w.lower() for w in a.weaknesses)
        )

        if count >= len(profile.action_analyses) * 0.7:
            return DevelopmentPriority.CRITICAL
        elif count >= len(profile.action_analyses) * 0.5:
            return DevelopmentPriority.HIGH
        elif count >= len(profile.action_analyses) * 0.3:
            return DevelopmentPriority.MEDIUM
        else:
            return DevelopmentPriority.LOW

    def _determine_weekly_focus(
        self,
        profile: PlayerTechnicalProfile,
        priorities: List[CoachingPoint]
    ) -> str:
        """Determine the focus for this week's training."""
        if not priorities:
            return "Continue building technical foundation"

        # Focus on highest priority item
        critical = [p for p in priorities if p.priority == DevelopmentPriority.CRITICAL]
        if critical:
            return f"CRITICAL: {critical[0].point}"

        high = [p for p in priorities if p.priority == DevelopmentPriority.HIGH]
        if high:
            return f"Priority: {high[0].point}"

        return f"Focus: {priorities[0].point}"

    def _recommend_weekly_drills(
        self,
        priorities: List[CoachingPoint],
        pos_req
    ) -> List[str]:
        """Recommend drills for the week."""
        drills = []

        # Add drills from priorities
        for priority in priorities[:3]:
            drills.append(priority.drill_to_address)

        # Add position-specific drill if available
        if pos_req:
            for skill in pos_req.primary_skills[:2]:
                technique = get_technique_for_action(skill)
                if technique and technique.drills_to_improve:
                    drills.append(technique.drills_to_improve[0])

        return list(set(drills))[:5]  # Max 5 unique drills

    def _set_short_term_goals(self, priorities: List[CoachingPoint]) -> List[str]:
        """Set 3-month goals."""
        goals = []

        critical_high = [p for p in priorities if p.priority in [DevelopmentPriority.CRITICAL, DevelopmentPriority.HIGH]]

        for priority in critical_high[:3]:
            goals.append(f"Improve {priority.technique_area}: {priority.point}")

        if not goals:
            goals.append("Maintain current technical level while building consistency")

        return goals

    def _set_medium_term_goals(
        self,
        priorities: List[CoachingPoint],
        pos_req
    ) -> List[str]:
        """Set 6-month goals."""
        goals = []

        # All priority areas
        for priority in priorities[:2]:
            goals.append(f"Master: {priority.point}")

        # Position-specific goals
        if pos_req:
            goals.append(f"Develop primary skills for {pos_req.position} position")

        goals.append("Demonstrate consistent performance in match situations")

        return goals[:4]

    def get_player_report(self, player_jersey: int) -> Optional[Dict]:
        """Get a comprehensive report for a player."""
        plan = self.player_plans.get(player_jersey)
        vision_report = self.vision_coach.get_coaching_summary(player_jersey)

        if not plan and not vision_report:
            return None

        return {
            "development_plan": {
                "player_id": plan.player_id if plan else player_jersey,
                "player_name": plan.player_name if plan else None,
                "position": plan.position if plan else None,
                "overall_rating": plan.overall_rating if plan else None,
                "four_corner_ratings": plan.four_corner_ratings if plan else None,
                "key_strengths": plan.key_strengths if plan else [],
                "weekly_focus": plan.weekly_focus if plan else None,
                "weekly_drills": plan.weekly_drills if plan else [],
                "development_priorities": [
                    {
                        "point": p.point,
                        "category": p.category.value,
                        "priority": p.priority.value,
                        "coaching_cue": p.coaching_cue,
                        "drill": p.drill_to_address
                    }
                    for p in (plan.development_priorities if plan else [])
                ]
            },
            "coaching_summary": vision_report,
            "methodology": "UEFA Pro License / FA Four Corner Model"
        }

    def export_all_data(self, output_path: str) -> str:
        """Export all coaching data."""
        data = {
            "exported_at": datetime.now().isoformat(),
            "methodology": "UEFA Coaching License / FA Four Corner Model",
            "player_development_plans": {
                str(jersey): {
                    "name": plan.player_name,
                    "position": plan.position,
                    "overall_rating": plan.overall_rating,
                    "four_corner_ratings": plan.four_corner_ratings,
                    "weekly_focus": plan.weekly_focus,
                    "priorities": [p.point for p in plan.development_priorities]
                }
                for jersey, plan in self.player_plans.items()
            },
            "analysis_count": sum(len(analyses) for analyses in self.analysis_history.values())
        }

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

        return output_path


# Global instance
expert_coach = ExpertCoachService()
