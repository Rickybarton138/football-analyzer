"""
Team Profile & Philosophy System

This module enables coaches to customize the AI Expert Coach to understand
and adapt to their specific team's philosophy, formation, playing style,
and individual player profiles.

Based on UEFA Pro License concepts:
- Game Model (Modelo de Juego)
- Tactical Periodization (Vítor Frade methodology)
- Principles of Play (10 UEFA principles)
- Four Moments of the Game
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
from datetime import datetime
import json
import os


# =============================================================================
# ENUMS FOR TEAM PHILOSOPHY OPTIONS
# =============================================================================

class PlayingStyle(str, Enum):
    """Primary playing philosophy - how the team approaches the game"""
    POSSESSION = "possession"  # Tiki-taka, patient build-up
    COUNTER_ATTACK = "counter_attack"  # Direct, fast transitions
    HIGH_PRESS = "high_press"  # Gegenpressing, win ball high
    LOW_BLOCK = "low_block"  # Compact defense, hit on break
    DIRECT_PLAY = "direct_play"  # Long balls, second balls
    BALANCED = "balanced"  # Adaptable approach
    POSITIONAL_PLAY = "positional_play"  # Structured possession, positional superiority


class BuildUpStyle(str, Enum):
    """How the team builds from the back"""
    SHORT_FROM_GK = "short_from_gk"  # Play out from goalkeeper
    LONG_FROM_GK = "long_from_gk"  # Direct from goalkeeper
    MIXED = "mixed"  # Vary based on press
    WIDE_BUILD = "wide_build"  # Use full-backs early
    CENTRAL_BUILD = "central_build"  # Through central midfield


class PressingIntensity(str, Enum):
    """Defensive pressing approach"""
    ULTRA_HIGH = "ultra_high"  # Press from front constantly
    HIGH = "high"  # Press in opponent's half
    MEDIUM = "medium"  # Press in middle third
    LOW = "low"  # Sit deep, protect goal
    TRIGGER_BASED = "trigger_based"  # Press on specific triggers


class TransitionSpeed(str, Enum):
    """How quickly team transitions between phases"""
    IMMEDIATE = "immediate"  # Instant counter/recovery
    FAST = "fast"  # Quick transition
    CONTROLLED = "controlled"  # Secure possession first
    PATIENT = "patient"  # Reset and reorganize


class WingPlay(str, Enum):
    """Use of wide areas"""
    INVERTED_WINGERS = "inverted_wingers"  # Cut inside
    TRADITIONAL_WINGERS = "traditional_wingers"  # Stay wide, cross
    OVERLAPPING_FULLBACKS = "overlapping_fullbacks"  # Width from backs
    UNDERLAPPING_FULLBACKS = "underlapping_fullbacks"  # Inside runs
    HYBRID = "hybrid"  # Mix of approaches


class DefensiveShape(str, Enum):
    """Defensive organizational structure"""
    FLAT_FOUR = "flat_four"  # Traditional back four
    SWEEPER_COVER = "sweeper_cover"  # One deeper
    HIGH_LINE = "high_line"  # Aggressive offside trap
    MID_BLOCK = "mid_block"  # Defend middle third
    LOW_BLOCK = "low_block"  # Deep defensive block
    MAN_MARKING = "man_marking"  # Individual assignments
    ZONAL = "zonal"  # Space-based marking


class AttackingFocus(str, Enum):
    """Primary attacking threat"""
    CENTRAL_OVERLOAD = "central_overload"  # Through the middle
    WIDE_OVERLOAD = "wide_overload"  # Create width
    CROSSES = "crosses"  # Deliver into box
    CUTBACKS = "cutbacks"  # Pull back from byline
    THROUGH_BALLS = "through_balls"  # Behind defensive line
    COMBINATION_PLAY = "combination_play"  # Short passing moves
    INDIVIDUAL_BRILLIANCE = "individual_brilliance"  # Rely on key players


# =============================================================================
# FORMATIONS
# =============================================================================

FORMATION_TEMPLATES = {
    "4-3-3": {
        "positions": ["GK", "RB", "RCB", "LCB", "LB", "RCM", "CDM", "LCM", "RW", "ST", "LW"],
        "shape": "balanced",
        "strengths": ["Width in attack", "Pressing from front", "Counter-attack options"],
        "weaknesses": ["Central midfield can be overrun", "Relies on wingers tracking back"]
    },
    "4-4-2": {
        "positions": ["GK", "RB", "RCB", "LCB", "LB", "RM", "RCM", "LCM", "LM", "RST", "LST"],
        "shape": "compact",
        "strengths": ["Solid defensively", "Two banks of four", "Partnership up front"],
        "weaknesses": ["Can lack creativity", "Wingers must cover full flank"]
    },
    "4-2-3-1": {
        "positions": ["GK", "RB", "RCB", "LCB", "LB", "RCDM", "LCDM", "RAM", "CAM", "LAM", "ST"],
        "shape": "balanced",
        "strengths": ["Defensive solidity", "Creative #10", "Flexible attack"],
        "weaknesses": ["Striker can be isolated", "Double pivot must be disciplined"]
    },
    "3-5-2": {
        "positions": ["GK", "RCB", "CB", "LCB", "RWB", "RCM", "CDM", "LCM", "LWB", "RST", "LST"],
        "shape": "narrow_with_width",
        "strengths": ["Midfield control", "Wing-backs provide width", "Two strikers"],
        "weaknesses": ["Wing-backs must be fit", "Can be exposed wide if caught high"]
    },
    "3-4-3": {
        "positions": ["GK", "RCB", "CB", "LCB", "RWB", "RCM", "LCM", "LWB", "RW", "ST", "LW"],
        "shape": "aggressive",
        "strengths": ["High pressing", "Attacking width", "Overloads"],
        "weaknesses": ["Defensively vulnerable", "Requires exceptional fitness"]
    },
    "4-1-4-1": {
        "positions": ["GK", "RB", "RCB", "LCB", "LB", "CDM", "RM", "RCM", "LCM", "LM", "ST"],
        "shape": "defensive",
        "strengths": ["Defensive shield", "Counter-attacking", "Compact shape"],
        "weaknesses": ["Single striker isolated", "Can be negative"]
    },
    "4-3-2-1": {
        "positions": ["GK", "RB", "RCB", "LCB", "LB", "RCM", "CDM", "LCM", "RAM", "LAM", "ST"],
        "shape": "christmas_tree",
        "strengths": ["Midfield numbers", "Support for striker", "Defensive coverage"],
        "weaknesses": ["Lack of natural width", "Relies on full-backs for width"]
    },
    "5-3-2": {
        "positions": ["GK", "RWB", "RCB", "CB", "LCB", "LWB", "RCM", "CDM", "LCM", "RST", "LST"],
        "shape": "defensive",
        "strengths": ["Defensive solidity", "Counter-attack", "Difficult to break down"],
        "weaknesses": ["Can be overly defensive", "Wing-backs need to push high"]
    },
}


# =============================================================================
# UEFA PRINCIPLES OF PLAY
# =============================================================================

PRINCIPLES_OF_PLAY = {
    "attacking": {
        "penetration": {
            "name": "Penetration",
            "description": "Getting the ball behind the defensive line",
            "uefa_guidance": "Create and exploit space behind the defense through runs, passes, or dribbles",
            "key_factors": ["Through balls", "Runs in behind", "Penetrating passes", "Dribbling past players"]
        },
        "depth": {
            "name": "Depth & Support",
            "description": "Players at different distances to support ball carrier",
            "uefa_guidance": "Multiple passing options at various angles and distances",
            "key_factors": ["Checking movements", "Third man runs", "Layoff options", "Diagonal support"]
        },
        "width": {
            "name": "Width",
            "description": "Stretching the opposition horizontally",
            "uefa_guidance": "Use full width of pitch to create space centrally",
            "key_factors": ["Wide positioning", "Touchline play", "Crossing opportunities", "Isolating full-backs"]
        },
        "mobility": {
            "name": "Mobility & Movement",
            "description": "Dynamic movement to create and exploit space",
            "uefa_guidance": "Continuous intelligent movement to unbalance defense",
            "key_factors": ["Rotation", "Interchange", "Decoy runs", "Third man movement"]
        },
        "improvisation": {
            "name": "Improvisation & Creativity",
            "description": "Unpredictable actions to surprise opponents",
            "uefa_guidance": "Encourage creative solutions within team structure",
            "key_factors": ["1v1 ability", "Skill moves", "Unexpected passes", "Individual brilliance"]
        }
    },
    "defending": {
        "pressure": {
            "name": "Pressure",
            "description": "Closing down the player on the ball",
            "uefa_guidance": "Apply pressure to reduce time and space on the ball",
            "key_factors": ["Closing speed", "Body shape", "Forcing direction", "Winning ball"]
        },
        "cover": {
            "name": "Cover & Balance",
            "description": "Supporting the pressing player",
            "uefa_guidance": "Provide security behind pressing player and balance across the pitch",
            "key_factors": ["Covering angles", "Recovery positions", "Defensive shape", "Communication"]
        },
        "compactness": {
            "name": "Compactness",
            "description": "Reducing space between units",
            "uefa_guidance": "Maintain tight horizontal and vertical distances",
            "key_factors": ["Pressing triggers", "Defensive line", "Midfield squeeze", "Unit coordination"]
        },
        "delay": {
            "name": "Delay",
            "description": "Slowing down the attack",
            "uefa_guidance": "Buy time for teammates to recover and organize",
            "key_factors": ["Body position", "Jockeying", "Forcing wide", "No diving in"]
        },
        "control_restraint": {
            "name": "Control & Restraint",
            "description": "Disciplined defending without fouling",
            "uefa_guidance": "Stay on feet, avoid rash challenges, control emotions",
            "key_factors": ["Patience", "Timing of tackle", "Not over-committing", "Concentration"]
        }
    }
}


# =============================================================================
# FOUR MOMENTS OF THE GAME
# =============================================================================

FOUR_MOMENTS = {
    "attacking_organization": {
        "name": "Attacking Organization",
        "description": "In possession with time to organize",
        "sub_phases": ["Build-up", "Progression", "Final third", "Finishing"],
        "key_questions": [
            "How do we build from the back?",
            "What patterns do we use to progress?",
            "How do we create chances?",
            "What movements in the box?"
        ]
    },
    "defensive_organization": {
        "name": "Defensive Organization",
        "description": "Out of possession with time to organize",
        "sub_phases": ["High press", "Mid block", "Low block", "Box defending"],
        "key_questions": [
            "Where do we win the ball?",
            "What triggers our press?",
            "How compact are our units?",
            "How do we protect our goal?"
        ]
    },
    "attacking_transition": {
        "name": "Attacking Transition",
        "description": "Moment of winning possession",
        "sub_phases": ["Ball recovery", "Counter-attack", "Fast break", "Sustain possession"],
        "key_questions": [
            "Do we counter or secure?",
            "Who makes runs immediately?",
            "Where do we look first?",
            "How many players commit forward?"
        ]
    },
    "defensive_transition": {
        "name": "Defensive Transition",
        "description": "Moment of losing possession",
        "sub_phases": ["Counter-press", "Recovery runs", "Emergency defending", "Reset"],
        "key_questions": [
            "Do we press immediately?",
            "How many seconds to press?",
            "Who presses, who recovers?",
            "When do we reset shape?"
        ]
    }
}


# =============================================================================
# DATA CLASSES FOR TEAM PROFILE
# =============================================================================

@dataclass
class PlayerProfile:
    """Individual player profile for personalized coaching"""
    jersey_number: int
    name: str
    position: str  # Primary position
    alternative_positions: List[str] = field(default_factory=list)

    # Technical strengths/weaknesses (1-10 scale)
    technical_ratings: Dict[str, int] = field(default_factory=dict)
    # e.g., {"passing": 8, "first_touch": 7, "shooting": 6, "heading": 5}

    # Four Corner assessment
    four_corner_ratings: Dict[str, float] = field(default_factory=dict)
    # e.g., {"technical": 7.5, "tactical": 6.5, "physical": 8.0, "psychological": 7.0}

    # Development priorities
    development_focus: List[str] = field(default_factory=list)

    # Playing characteristics
    characteristics: List[str] = field(default_factory=list)
    # e.g., ["left-footed", "quick", "good in air", "leadership qualities"]

    # Coach's notes
    notes: str = ""

    # Age group (for age-appropriate coaching)
    age_group: str = "senior"  # u9, u12, u14, u16, u18, senior

    # Historical performance data
    match_appearances: int = 0
    goals: int = 0
    assists: int = 0

    def to_dict(self) -> dict:
        return {
            "jersey_number": self.jersey_number,
            "name": self.name,
            "position": self.position,
            "alternative_positions": self.alternative_positions,
            "technical_ratings": self.technical_ratings,
            "four_corner_ratings": self.four_corner_ratings,
            "development_focus": self.development_focus,
            "characteristics": self.characteristics,
            "notes": self.notes,
            "age_group": self.age_group,
            "match_appearances": self.match_appearances,
            "goals": self.goals,
            "assists": self.assists
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PlayerProfile":
        return cls(**data)


@dataclass
class TeamPhilosophy:
    """Core team playing philosophy and principles"""

    # Identity
    team_name: str
    team_motto: str = ""  # e.g., "Play with courage"
    philosophy_statement: str = ""  # Coach's vision

    # Primary Style
    playing_style: PlayingStyle = PlayingStyle.BALANCED
    build_up_style: BuildUpStyle = BuildUpStyle.MIXED
    pressing_intensity: PressingIntensity = PressingIntensity.MEDIUM

    # Transition preferences
    attacking_transition_speed: TransitionSpeed = TransitionSpeed.FAST
    defensive_transition_speed: TransitionSpeed = TransitionSpeed.IMMEDIATE

    # Attacking preferences
    wing_play: WingPlay = WingPlay.HYBRID
    attacking_focus: AttackingFocus = AttackingFocus.COMBINATION_PLAY

    # Defensive preferences
    defensive_shape: DefensiveShape = DefensiveShape.ZONAL
    defensive_line_height: str = "medium"  # low, medium, high

    # Set pieces
    set_piece_importance: str = "medium"  # low, medium, high

    # Priority principles (ranked)
    attacking_principles_priority: List[str] = field(default_factory=list)
    defending_principles_priority: List[str] = field(default_factory=list)

    # Four Moments preferences
    moment_preferences: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "team_name": self.team_name,
            "team_motto": self.team_motto,
            "philosophy_statement": self.philosophy_statement,
            "playing_style": self.playing_style.value,
            "build_up_style": self.build_up_style.value,
            "pressing_intensity": self.pressing_intensity.value,
            "attacking_transition_speed": self.attacking_transition_speed.value,
            "defensive_transition_speed": self.defensive_transition_speed.value,
            "wing_play": self.wing_play.value,
            "attacking_focus": self.attacking_focus.value,
            "defensive_shape": self.defensive_shape.value,
            "defensive_line_height": self.defensive_line_height,
            "set_piece_importance": self.set_piece_importance,
            "attacking_principles_priority": self.attacking_principles_priority,
            "defending_principles_priority": self.defending_principles_priority,
            "moment_preferences": self.moment_preferences
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TeamPhilosophy":
        data["playing_style"] = PlayingStyle(data.get("playing_style", "balanced"))
        data["build_up_style"] = BuildUpStyle(data.get("build_up_style", "mixed"))
        data["pressing_intensity"] = PressingIntensity(data.get("pressing_intensity", "medium"))
        data["attacking_transition_speed"] = TransitionSpeed(data.get("attacking_transition_speed", "fast"))
        data["defensive_transition_speed"] = TransitionSpeed(data.get("defensive_transition_speed", "immediate"))
        data["wing_play"] = WingPlay(data.get("wing_play", "hybrid"))
        data["attacking_focus"] = AttackingFocus(data.get("attacking_focus", "combination_play"))
        data["defensive_shape"] = DefensiveShape(data.get("defensive_shape", "zonal"))
        return cls(**data)


@dataclass
class FormationSetup:
    """Team formation and positional instructions"""

    primary_formation: str = "4-3-3"
    alternative_formations: List[str] = field(default_factory=list)

    # Position-specific instructions
    positional_instructions: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # e.g., {"RB": {"attacking_runs": "overlap", "defensive_priority": "stay"}}

    # Partnerships and combinations
    key_partnerships: List[Dict[str, Any]] = field(default_factory=list)
    # e.g., [{"players": ["CM", "ST"], "pattern": "through ball runs"}]

    # In-game variations
    situational_changes: Dict[str, str] = field(default_factory=dict)
    # e.g., {"chasing_game": "4-2-4", "protecting_lead": "5-4-1"}

    def to_dict(self) -> dict:
        return {
            "primary_formation": self.primary_formation,
            "alternative_formations": self.alternative_formations,
            "positional_instructions": self.positional_instructions,
            "key_partnerships": self.key_partnerships,
            "situational_changes": self.situational_changes
        }

    @classmethod
    def from_dict(cls, data: dict) -> "FormationSetup":
        return cls(**data)


@dataclass
class TeamProfile:
    """Complete team profile combining philosophy, formation, and players"""

    philosophy: TeamPhilosophy
    formation: FormationSetup
    players: Dict[int, PlayerProfile] = field(default_factory=dict)  # jersey_number -> profile

    # Season goals
    season_objectives: List[str] = field(default_factory=list)
    development_themes: List[str] = field(default_factory=list)

    # Match analysis preferences
    focus_areas: List[str] = field(default_factory=list)  # What coach wants highlighted

    # Created/updated timestamps
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def add_player(self, player: PlayerProfile):
        """Add or update a player profile"""
        self.players[player.jersey_number] = player
        self.updated_at = datetime.now().isoformat()

    def get_player(self, jersey_number: int) -> Optional[PlayerProfile]:
        """Get a player by jersey number"""
        return self.players.get(jersey_number)

    def get_players_by_position(self, position: str) -> List[PlayerProfile]:
        """Get all players who can play a position"""
        result = []
        for player in self.players.values():
            if player.position == position or position in player.alternative_positions:
                result.append(player)
        return result

    def to_dict(self) -> dict:
        return {
            "philosophy": self.philosophy.to_dict(),
            "formation": self.formation.to_dict(),
            "players": {k: v.to_dict() for k, v in self.players.items()},
            "season_objectives": self.season_objectives,
            "development_themes": self.development_themes,
            "focus_areas": self.focus_areas,
            "created_at": self.created_at,
            "updated_at": self.updated_at
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TeamProfile":
        philosophy = TeamPhilosophy.from_dict(data["philosophy"])
        formation = FormationSetup.from_dict(data["formation"])
        players = {int(k): PlayerProfile.from_dict(v) for k, v in data.get("players", {}).items()}

        return cls(
            philosophy=philosophy,
            formation=formation,
            players=players,
            season_objectives=data.get("season_objectives", []),
            development_themes=data.get("development_themes", []),
            focus_areas=data.get("focus_areas", []),
            created_at=data.get("created_at", datetime.now().isoformat()),
            updated_at=data.get("updated_at", datetime.now().isoformat())
        )


# =============================================================================
# TEAM PROFILE SERVICE
# =============================================================================

class TeamProfileService:
    """
    Service for managing team profiles and providing context to the AI Coach.

    This service allows coaches to:
    1. Create and customize their team's playing philosophy
    2. Define formation and positional instructions
    3. Build individual player profiles
    4. Generate context for AI coaching analysis
    """

    def __init__(self, storage_path: str = "team_profiles"):
        self.storage_path = storage_path
        self.current_profile: Optional[TeamProfile] = None

        # Ensure storage directory exists
        os.makedirs(storage_path, exist_ok=True)

    # =========================================================================
    # PROFILE MANAGEMENT
    # =========================================================================

    def create_profile(
        self,
        team_name: str,
        playing_style: str = "balanced",
        primary_formation: str = "4-3-3"
    ) -> TeamProfile:
        """Create a new team profile with defaults"""

        philosophy = TeamPhilosophy(
            team_name=team_name,
            playing_style=PlayingStyle(playing_style)
        )

        formation = FormationSetup(
            primary_formation=primary_formation
        )

        profile = TeamProfile(
            philosophy=philosophy,
            formation=formation
        )

        self.current_profile = profile
        return profile

    def save_profile(self, profile: Optional[TeamProfile] = None, filename: Optional[str] = None) -> str:
        """Save team profile to JSON file"""
        profile = profile or self.current_profile
        if not profile:
            raise ValueError("No profile to save")

        if not filename:
            # Create filename from team name
            safe_name = "".join(c if c.isalnum() else "_" for c in profile.philosophy.team_name)
            filename = f"{safe_name.lower()}_profile.json"

        filepath = os.path.join(self.storage_path, filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(profile.to_dict(), f, indent=2, ensure_ascii=False)

        return filepath

    def load_profile(self, filename: str) -> TeamProfile:
        """Load team profile from JSON file"""
        filepath = os.path.join(self.storage_path, filename)

        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.current_profile = TeamProfile.from_dict(data)
        return self.current_profile

    def list_profiles(self) -> List[str]:
        """List all saved team profiles"""
        profiles = []
        for filename in os.listdir(self.storage_path):
            if filename.endswith('_profile.json'):
                profiles.append(filename)
        return profiles

    # =========================================================================
    # PHILOSOPHY CONFIGURATION
    # =========================================================================

    def update_philosophy(self, **kwargs) -> TeamPhilosophy:
        """Update team philosophy settings"""
        if not self.current_profile:
            raise ValueError("No active profile")

        philosophy = self.current_profile.philosophy

        for key, value in kwargs.items():
            if hasattr(philosophy, key):
                # Handle enum conversions
                if key == "playing_style":
                    value = PlayingStyle(value)
                elif key == "build_up_style":
                    value = BuildUpStyle(value)
                elif key == "pressing_intensity":
                    value = PressingIntensity(value)
                elif key in ["attacking_transition_speed", "defensive_transition_speed"]:
                    value = TransitionSpeed(value)
                elif key == "wing_play":
                    value = WingPlay(value)
                elif key == "attacking_focus":
                    value = AttackingFocus(value)
                elif key == "defensive_shape":
                    value = DefensiveShape(value)

                setattr(philosophy, key, value)

        self.current_profile.updated_at = datetime.now().isoformat()
        return philosophy

    def set_philosophy_statement(self, statement: str) -> None:
        """Set the coach's philosophy statement"""
        if not self.current_profile:
            raise ValueError("No active profile")

        self.current_profile.philosophy.philosophy_statement = statement
        self.current_profile.updated_at = datetime.now().isoformat()

    def set_attacking_principles_priority(self, principles: List[str]) -> None:
        """Set priority order for attacking principles"""
        valid_principles = list(PRINCIPLES_OF_PLAY["attacking"].keys())
        for p in principles:
            if p not in valid_principles:
                raise ValueError(f"Invalid attacking principle: {p}. Valid: {valid_principles}")

        self.current_profile.philosophy.attacking_principles_priority = principles
        self.current_profile.updated_at = datetime.now().isoformat()

    def set_defending_principles_priority(self, principles: List[str]) -> None:
        """Set priority order for defending principles"""
        valid_principles = list(PRINCIPLES_OF_PLAY["defending"].keys())
        for p in principles:
            if p not in valid_principles:
                raise ValueError(f"Invalid defending principle: {p}. Valid: {valid_principles}")

        self.current_profile.philosophy.defending_principles_priority = principles
        self.current_profile.updated_at = datetime.now().isoformat()

    # =========================================================================
    # FORMATION CONFIGURATION
    # =========================================================================

    def set_formation(self, formation: str) -> None:
        """Set primary formation"""
        if formation not in FORMATION_TEMPLATES:
            raise ValueError(f"Unknown formation: {formation}. Valid: {list(FORMATION_TEMPLATES.keys())}")

        self.current_profile.formation.primary_formation = formation
        self.current_profile.updated_at = datetime.now().isoformat()

    def add_alternative_formation(self, formation: str) -> None:
        """Add an alternative formation"""
        if formation not in FORMATION_TEMPLATES:
            raise ValueError(f"Unknown formation: {formation}")

        if formation not in self.current_profile.formation.alternative_formations:
            self.current_profile.formation.alternative_formations.append(formation)
            self.current_profile.updated_at = datetime.now().isoformat()

    def set_positional_instruction(self, position: str, instruction_key: str, value: Any) -> None:
        """Set instruction for a specific position"""
        if not self.current_profile:
            raise ValueError("No active profile")

        if position not in self.current_profile.formation.positional_instructions:
            self.current_profile.formation.positional_instructions[position] = {}

        self.current_profile.formation.positional_instructions[position][instruction_key] = value
        self.current_profile.updated_at = datetime.now().isoformat()

    def add_key_partnership(self, players: List[str], pattern: str, description: str = "") -> None:
        """Define a key partnership or combination between positions"""
        partnership = {
            "players": players,
            "pattern": pattern,
            "description": description
        }
        self.current_profile.formation.key_partnerships.append(partnership)
        self.current_profile.updated_at = datetime.now().isoformat()

    # =========================================================================
    # PLAYER MANAGEMENT
    # =========================================================================

    def add_player(
        self,
        jersey_number: int,
        name: str,
        position: str,
        **kwargs
    ) -> PlayerProfile:
        """Add a new player to the team profile"""
        if not self.current_profile:
            raise ValueError("No active profile")

        player = PlayerProfile(
            jersey_number=jersey_number,
            name=name,
            position=position,
            **kwargs
        )

        self.current_profile.add_player(player)
        return player

    def update_player(self, jersey_number: int, **kwargs) -> PlayerProfile:
        """Update an existing player's profile"""
        if not self.current_profile:
            raise ValueError("No active profile")

        player = self.current_profile.get_player(jersey_number)
        if not player:
            raise ValueError(f"Player #{jersey_number} not found")

        for key, value in kwargs.items():
            if hasattr(player, key):
                setattr(player, key, value)

        self.current_profile.updated_at = datetime.now().isoformat()
        return player

    def set_player_development_focus(self, jersey_number: int, focus_areas: List[str]) -> None:
        """Set development priorities for a player"""
        player = self.current_profile.get_player(jersey_number)
        if not player:
            raise ValueError(f"Player #{jersey_number} not found")

        player.development_focus = focus_areas
        self.current_profile.updated_at = datetime.now().isoformat()

    def update_player_four_corner(self, jersey_number: int, ratings: Dict[str, float]) -> None:
        """Update a player's Four Corner ratings"""
        valid_domains = ["technical", "tactical", "physical", "psychological"]
        for domain in ratings.keys():
            if domain not in valid_domains:
                raise ValueError(f"Invalid domain: {domain}. Valid: {valid_domains}")

        player = self.current_profile.get_player(jersey_number)
        if not player:
            raise ValueError(f"Player #{jersey_number} not found")

        player.four_corner_ratings.update(ratings)
        self.current_profile.updated_at = datetime.now().isoformat()

    # =========================================================================
    # AI COACH CONTEXT GENERATION
    # =========================================================================

    def generate_coaching_context(self, include_players: bool = True) -> str:
        """
        Generate comprehensive context string for AI Coach.
        This context helps the AI understand the team's philosophy and provide
        tailored coaching feedback.
        """
        if not self.current_profile:
            return "No team profile configured."

        profile = self.current_profile
        philosophy = profile.philosophy
        formation = profile.formation

        context_parts = []

        # Team Identity
        context_parts.append(f"## Team: {philosophy.team_name}")
        if philosophy.team_motto:
            context_parts.append(f"**Motto:** {philosophy.team_motto}")
        if philosophy.philosophy_statement:
            context_parts.append(f"**Coach's Philosophy:** {philosophy.philosophy_statement}")

        # Playing Style
        context_parts.append("\n## Playing Style")
        context_parts.append(f"- **Primary Style:** {philosophy.playing_style.value.replace('_', ' ').title()}")
        context_parts.append(f"- **Build-Up:** {philosophy.build_up_style.value.replace('_', ' ').title()}")
        context_parts.append(f"- **Pressing:** {philosophy.pressing_intensity.value.replace('_', ' ').title()}")
        context_parts.append(f"- **Wing Play:** {philosophy.wing_play.value.replace('_', ' ').title()}")
        context_parts.append(f"- **Attacking Focus:** {philosophy.attacking_focus.value.replace('_', ' ').title()}")

        # Formation
        context_parts.append("\n## Formation")
        context_parts.append(f"- **Primary:** {formation.primary_formation}")
        if formation.alternative_formations:
            context_parts.append(f"- **Alternatives:** {', '.join(formation.alternative_formations)}")

        template = FORMATION_TEMPLATES.get(formation.primary_formation, {})
        if template:
            context_parts.append(f"- **Shape:** {template.get('shape', 'N/A')}")
            context_parts.append(f"- **Strengths:** {', '.join(template.get('strengths', []))}")
            context_parts.append(f"- **Weaknesses to manage:** {', '.join(template.get('weaknesses', []))}")

        # Defensive Setup
        context_parts.append("\n## Defensive Organization")
        context_parts.append(f"- **Shape:** {philosophy.defensive_shape.value.replace('_', ' ').title()}")
        context_parts.append(f"- **Line Height:** {philosophy.defensive_line_height.title()}")
        context_parts.append(f"- **Defensive Transition:** {philosophy.defensive_transition_speed.value.replace('_', ' ').title()}")

        # Principles Priority
        if philosophy.attacking_principles_priority:
            context_parts.append("\n## Attacking Principles (in priority order)")
            for i, principle in enumerate(philosophy.attacking_principles_priority, 1):
                principle_info = PRINCIPLES_OF_PLAY["attacking"].get(principle, {})
                context_parts.append(f"{i}. **{principle_info.get('name', principle)}** - {principle_info.get('description', '')}")

        if philosophy.defending_principles_priority:
            context_parts.append("\n## Defending Principles (in priority order)")
            for i, principle in enumerate(philosophy.defending_principles_priority, 1):
                principle_info = PRINCIPLES_OF_PLAY["defending"].get(principle, {})
                context_parts.append(f"{i}. **{principle_info.get('name', principle)}** - {principle_info.get('description', '')}")

        # Key Partnerships
        if formation.key_partnerships:
            context_parts.append("\n## Key Partnerships")
            for partnership in formation.key_partnerships:
                players = " + ".join(partnership["players"])
                context_parts.append(f"- **{players}:** {partnership['pattern']}")

        # Positional Instructions
        if formation.positional_instructions:
            context_parts.append("\n## Position-Specific Instructions")
            for position, instructions in formation.positional_instructions.items():
                context_parts.append(f"### {position}")
                for key, value in instructions.items():
                    context_parts.append(f"- {key.replace('_', ' ').title()}: {value}")

        # Season Objectives
        if profile.season_objectives:
            context_parts.append("\n## Season Objectives")
            for obj in profile.season_objectives:
                context_parts.append(f"- {obj}")

        # Development Themes
        if profile.development_themes:
            context_parts.append("\n## Development Themes")
            for theme in profile.development_themes:
                context_parts.append(f"- {theme}")

        # Players
        if include_players and profile.players:
            context_parts.append("\n## Squad Profiles")

            # Group by position
            positions_order = ["GK", "RB", "RCB", "CB", "LCB", "LB", "RWB", "LWB",
                            "CDM", "CM", "RCM", "LCM", "CAM", "RAM", "LAM",
                            "RM", "LM", "RW", "LW", "ST", "RST", "LST", "CF"]

            sorted_players = sorted(
                profile.players.values(),
                key=lambda p: positions_order.index(p.position) if p.position in positions_order else 99
            )

            for player in sorted_players:
                context_parts.append(f"\n### #{player.jersey_number} {player.name} ({player.position})")

                if player.characteristics:
                    context_parts.append(f"**Characteristics:** {', '.join(player.characteristics)}")

                if player.four_corner_ratings:
                    ratings_str = ", ".join([f"{k}: {v}/10" for k, v in player.four_corner_ratings.items()])
                    context_parts.append(f"**Four Corner Ratings:** {ratings_str}")

                if player.development_focus:
                    context_parts.append(f"**Development Focus:** {', '.join(player.development_focus)}")

                if player.notes:
                    context_parts.append(f"**Coach Notes:** {player.notes}")

        return "\n".join(context_parts)

    def generate_player_context(self, jersey_number: int) -> str:
        """Generate AI context for a specific player"""
        if not self.current_profile:
            return "No team profile configured."

        player = self.current_profile.get_player(jersey_number)
        if not player:
            return f"Player #{jersey_number} not found in team profile."

        philosophy = self.current_profile.philosophy

        context_parts = []

        context_parts.append(f"## Player: #{player.jersey_number} {player.name}")
        context_parts.append(f"**Position:** {player.position}")
        if player.alternative_positions:
            context_parts.append(f"**Can Also Play:** {', '.join(player.alternative_positions)}")
        context_parts.append(f"**Age Group:** {player.age_group}")

        # Four Corner Assessment
        if player.four_corner_ratings:
            context_parts.append("\n### Four Corner Assessment")
            for domain, rating in player.four_corner_ratings.items():
                context_parts.append(f"- {domain.title()}: {rating}/10")

        # Technical Ratings
        if player.technical_ratings:
            context_parts.append("\n### Technical Skills")
            for skill, rating in player.technical_ratings.items():
                context_parts.append(f"- {skill.replace('_', ' ').title()}: {rating}/10")

        # Characteristics
        if player.characteristics:
            context_parts.append(f"\n### Player Characteristics")
            for char in player.characteristics:
                context_parts.append(f"- {char}")

        # Development Focus
        if player.development_focus:
            context_parts.append("\n### Current Development Focus")
            for focus in player.development_focus:
                context_parts.append(f"- {focus}")

        # Coach Notes
        if player.notes:
            context_parts.append(f"\n### Coach's Notes")
            context_parts.append(player.notes)

        # Team Context
        context_parts.append(f"\n### Team Context")
        context_parts.append(f"**Team:** {philosophy.team_name}")
        context_parts.append(f"**Playing Style:** {philosophy.playing_style.value.replace('_', ' ').title()}")
        context_parts.append(f"**Formation:** {self.current_profile.formation.primary_formation}")

        # Position-specific instructions
        pos_instructions = self.current_profile.formation.positional_instructions.get(player.position, {})
        if pos_instructions:
            context_parts.append("\n### Position Instructions")
            for key, value in pos_instructions.items():
                context_parts.append(f"- {key.replace('_', ' ').title()}: {value}")

        return "\n".join(context_parts)

    def generate_match_analysis_context(self) -> str:
        """Generate context specifically for match analysis"""
        if not self.current_profile:
            return "No team profile configured."

        profile = self.current_profile
        philosophy = profile.philosophy

        context_parts = []

        context_parts.append(f"# Match Analysis Context: {philosophy.team_name}")

        # What to look for based on style
        context_parts.append("\n## Analysis Focus (Based on Team Philosophy)")

        style_focus = {
            PlayingStyle.POSSESSION: [
                "Ball retention percentage",
                "Passing accuracy in each third",
                "Time in possession per phase",
                "Successful combinations and triangles",
                "Build-up patterns from GK"
            ],
            PlayingStyle.COUNTER_ATTACK: [
                "Transition speed (seconds to reach final third)",
                "Direct passes in transition",
                "Runs behind the defense",
                "Ball recovery positions",
                "Counter-attack conversion rate"
            ],
            PlayingStyle.HIGH_PRESS: [
                "Pressing triggers executed",
                "PPDA (Passes per defensive action)",
                "Ball recoveries in attacking third",
                "Counter-press success rate",
                "Time opponent has on ball"
            ],
            PlayingStyle.LOW_BLOCK: [
                "Defensive shape maintenance",
                "Shot blocks and interceptions",
                "Clearances and aerial duels",
                "Counter-attack opportunities created",
                "Set piece defense"
            ],
            PlayingStyle.POSITIONAL_PLAY: [
                "Third man movements",
                "Position rotations",
                "Numerical superiorities created",
                "Free player identification",
                "Structured attacking patterns"
            ]
        }

        focus_areas = style_focus.get(philosophy.playing_style, [])
        for area in focus_areas:
            context_parts.append(f"- {area}")

        # User-defined focus areas
        if profile.focus_areas:
            context_parts.append("\n## Coach-Specified Focus Areas")
            for area in profile.focus_areas:
                context_parts.append(f"- {area}")

        # Formation-specific analysis
        context_parts.append(f"\n## Formation Analysis Points ({profile.formation.primary_formation})")
        template = FORMATION_TEMPLATES.get(profile.formation.primary_formation, {})
        if template:
            context_parts.append("**Monitor strengths:**")
            for strength in template.get("strengths", []):
                context_parts.append(f"- {strength}")
            context_parts.append("**Watch for weaknesses:**")
            for weakness in template.get("weaknesses", []):
                context_parts.append(f"- {weakness}")

        return "\n".join(context_parts)


# =============================================================================
# QUICK SETUP TEMPLATES
# =============================================================================

def create_possession_team_profile(team_name: str) -> TeamProfile:
    """Quick setup for a possession-based team"""
    service = TeamProfileService()

    profile = service.create_profile(team_name, "possession", "4-3-3")

    service.update_philosophy(
        build_up_style="short_from_gk",
        pressing_intensity="trigger_based",
        attacking_transition_speed="controlled",
        defensive_transition_speed="immediate",
        wing_play="inverted_wingers",
        attacking_focus="combination_play",
        defensive_shape="zonal",
        defensive_line_height="high"
    )

    service.set_attacking_principles_priority([
        "depth", "width", "mobility", "penetration", "improvisation"
    ])

    service.set_defending_principles_priority([
        "pressure", "compactness", "cover", "delay", "control_restraint"
    ])

    service.set_philosophy_statement(
        "We dominate games through intelligent possession. "
        "Every player is comfortable on the ball. "
        "We create numerical superiority through positional rotation. "
        "When we lose the ball, we counter-press immediately to regain possession."
    )

    return profile


def create_counter_attack_team_profile(team_name: str) -> TeamProfile:
    """Quick setup for a counter-attacking team"""
    service = TeamProfileService()

    profile = service.create_profile(team_name, "counter_attack", "4-4-2")

    service.update_philosophy(
        build_up_style="mixed",
        pressing_intensity="medium",
        attacking_transition_speed="immediate",
        defensive_transition_speed="fast",
        wing_play="traditional_wingers",
        attacking_focus="through_balls",
        defensive_shape="mid_block",
        defensive_line_height="medium"
    )

    service.set_attacking_principles_priority([
        "penetration", "mobility", "width", "depth", "improvisation"
    ])

    service.set_defending_principles_priority([
        "compactness", "delay", "cover", "pressure", "control_restraint"
    ])

    service.set_philosophy_statement(
        "We are disciplined without the ball and devastating with it. "
        "We invite pressure, remain compact, and strike with speed when we win possession. "
        "Our wide players are the key to our transitions. "
        "Every player knows their recovery position and counter-attack role."
    )

    return profile


def create_high_press_team_profile(team_name: str) -> TeamProfile:
    """Quick setup for a high-pressing team (Gegenpressing style)"""
    service = TeamProfileService()

    profile = service.create_profile(team_name, "high_press", "4-3-3")

    service.update_philosophy(
        build_up_style="short_from_gk",
        pressing_intensity="ultra_high",
        attacking_transition_speed="immediate",
        defensive_transition_speed="immediate",
        wing_play="inverted_wingers",
        attacking_focus="central_overload",
        defensive_shape="high_line",
        defensive_line_height="high"
    )

    service.set_attacking_principles_priority([
        "penetration", "mobility", "improvisation", "depth", "width"
    ])

    service.set_defending_principles_priority([
        "pressure", "compactness", "cover", "control_restraint", "delay"
    ])

    service.set_philosophy_statement(
        "We suffocate opponents with relentless pressing. "
        "The moment we lose the ball, we have 6 seconds to win it back. "
        "We win the ball high and attack with intensity. "
        "Our pressing is coordinated - everyone presses or no one presses."
    )

    return profile
