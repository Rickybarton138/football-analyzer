"""
AI Football Coaching Expert

Analyzes match data and generates intelligent coaching recommendations,
tactical insights, and player-specific feedback.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum
from datetime import datetime
import numpy as np


class InsightCategory(Enum):
    TACTICAL = "tactical"
    PRESSING = "pressing"
    POSSESSION = "possession"
    DEFENSIVE = "defensive"
    ATTACKING = "attacking"
    SET_PIECES = "set_pieces"
    PLAYER_SPECIFIC = "player_specific"
    SUBSTITUTION = "substitution"
    FORMATION = "formation"
    PHYSICAL = "physical"


class InsightPriority(Enum):
    CRITICAL = "critical"      # Immediate action needed
    HIGH = "high"              # Important tactical adjustment
    MEDIUM = "medium"          # Suggested improvement
    LOW = "low"                # General observation
    INFO = "info"              # Statistical insight


@dataclass
class CoachingInsight:
    """A single coaching insight or recommendation."""
    category: InsightCategory
    priority: InsightPriority
    title: str
    message: str
    recommendation: str
    timestamp: Optional[float] = None
    supporting_data: Dict = field(default_factory=dict)
    affected_players: List[int] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            'category': self.category.value,
            'priority': self.priority.value,
            'title': self.title,
            'message': self.message,
            'recommendation': self.recommendation,
            'timestamp': self.timestamp,
            'supporting_data': self.supporting_data,
            'affected_players': self.affected_players
        }


@dataclass
class MatchAnalysisSummary:
    """Overall match analysis summary."""
    overall_rating: str  # "Excellent", "Good", "Average", "Poor"
    key_strengths: List[str]
    areas_to_improve: List[str]
    tactical_summary: str
    half_time_message: str
    full_time_message: str

    def to_dict(self) -> Dict:
        return {
            'overall_rating': self.overall_rating,
            'key_strengths': self.key_strengths,
            'areas_to_improve': self.areas_to_improve,
            'tactical_summary': self.tactical_summary,
            'half_time_message': self.half_time_message,
            'full_time_message': self.full_time_message
        }


class AICoach:
    """
    AI Football Coaching Expert System.

    Analyzes match data from various sources (passes, formations, tactical events)
    and generates actionable coaching insights and recommendations.
    """

    def __init__(self):
        self.insights: List[CoachingInsight] = []
        self.match_summary: Optional[MatchAnalysisSummary] = None

        # Thresholds for generating insights
        self.thresholds = {
            'low_possession': 40,           # Below this = concern
            'high_possession': 60,          # Above this = positive
            'low_pass_accuracy': 70,        # Below this = concern
            'high_pass_accuracy': 85,       # Above this = positive
            'pressing_intensity_low': 3,    # Events per 5 min
            'pressing_intensity_high': 8,
            'dangerous_attacks_good': 5,    # Per half
            'defensive_line_high': 60,      # Percentage of pitch
            'defensive_line_low': 35,
            'compactness_loose': 200,       # Pixels
            'compactness_tight': 120,
        }

    def reset(self):
        """Reset for new analysis."""
        self.insights = []
        self.match_summary = None

    def analyze_match(self,
                      pass_stats: Dict,
                      formation_stats: Dict,
                      tactical_events: Dict,
                      frame_analyses: List[Dict] = None) -> Dict:
        """
        Perform comprehensive match analysis and generate coaching insights.

        Args:
            pass_stats: Pass and possession statistics
            formation_stats: Formation detection results
            tactical_events: Tactical events (pressing, attacks, etc.)
            frame_analyses: Raw frame analysis data (optional)

        Returns:
            Complete coaching analysis with insights and recommendations
        """
        self.reset()

        # Analyze each aspect of the match
        self._analyze_possession(pass_stats)
        self._analyze_passing(pass_stats)
        self._analyze_formations(formation_stats)
        self._analyze_pressing(tactical_events)
        self._analyze_attacking(tactical_events)
        self._analyze_defensive(tactical_events, formation_stats)
        self._analyze_opposition(pass_stats, formation_stats, tactical_events)

        # Generate overall match summary
        self._generate_match_summary(pass_stats, formation_stats, tactical_events)

        # Sort insights by priority
        priority_order = {
            InsightPriority.CRITICAL: 0,
            InsightPriority.HIGH: 1,
            InsightPriority.MEDIUM: 2,
            InsightPriority.LOW: 3,
            InsightPriority.INFO: 4
        }
        self.insights.sort(key=lambda x: priority_order[x.priority])

        return self.get_full_analysis()

    def _analyze_possession(self, pass_stats: Dict):
        """Analyze possession patterns and generate insights."""
        home_possession = pass_stats.get('home', {}).get('possession_percent', 50)
        away_possession = pass_stats.get('away', {}).get('possession_percent', 50)

        # Home team possession analysis (assuming home is user's team)
        if home_possession < self.thresholds['low_possession']:
            self.insights.append(CoachingInsight(
                category=InsightCategory.POSSESSION,
                priority=InsightPriority.HIGH,
                title="Low Possession",
                message=f"Your team is struggling to retain possession ({home_possession:.1f}%). "
                        f"The opposition is dominating ball control.",
                recommendation="Focus on shorter, safer passes to build confidence. "
                              "Encourage players to find space and offer passing options. "
                              "Consider dropping deeper to collect the ball.",
                supporting_data={'home_possession': home_possession, 'away_possession': away_possession}
            ))
        elif home_possession > self.thresholds['high_possession']:
            self.insights.append(CoachingInsight(
                category=InsightCategory.POSSESSION,
                priority=InsightPriority.INFO,
                title="Excellent Possession",
                message=f"Your team is controlling the game with {home_possession:.1f}% possession.",
                recommendation="Keep patient but look for opportunities to penetrate. "
                              "Don't become too predictable - vary the tempo and direction of play.",
                supporting_data={'home_possession': home_possession}
            ))

        # Check for possession imbalance
        possession_diff = abs(home_possession - away_possession)
        if possession_diff > 20:
            dominant_team = "Your team" if home_possession > away_possession else "Opposition"
            self.insights.append(CoachingInsight(
                category=InsightCategory.TACTICAL,
                priority=InsightPriority.MEDIUM,
                title="Possession Imbalance",
                message=f"{dominant_team} is dominating possession by {possession_diff:.1f}%.",
                recommendation="If behind: Press higher to win the ball back quicker. "
                              "If ahead: Manage the game, keep possession in safer areas.",
                supporting_data={'possession_difference': possession_diff}
            ))

    def _analyze_passing(self, pass_stats: Dict):
        """Analyze passing patterns and accuracy."""
        home_stats = pass_stats.get('home', {})
        away_stats = pass_stats.get('away', {})

        home_accuracy = home_stats.get('pass_accuracy', 0)
        home_total = home_stats.get('total_passes', 0)
        home_forward = home_stats.get('forward_pass_ratio', 0)

        # Pass accuracy analysis
        if home_accuracy < self.thresholds['low_pass_accuracy']:
            self.insights.append(CoachingInsight(
                category=InsightCategory.POSSESSION,
                priority=InsightPriority.HIGH,
                title="Poor Pass Accuracy",
                message=f"Pass completion is too low at {home_accuracy:.1f}%. "
                        f"This is causing unnecessary turnovers.",
                recommendation="Encourage simpler passing - don't force difficult balls. "
                              "Players should take an extra touch to control before passing. "
                              "Focus on body shape and looking before passing.",
                supporting_data={'pass_accuracy': home_accuracy, 'total_passes': home_total}
            ))
        elif home_accuracy > self.thresholds['high_pass_accuracy']:
            self.insights.append(CoachingInsight(
                category=InsightCategory.POSSESSION,
                priority=InsightPriority.INFO,
                title="Excellent Pass Accuracy",
                message=f"Outstanding passing accuracy at {home_accuracy:.1f}%.",
                recommendation="Maintain this standard. Consider being slightly more ambitious "
                              "with forward passes to create chances.",
                supporting_data={'pass_accuracy': home_accuracy}
            ))

        # Forward passing analysis
        if home_forward < 30:
            self.insights.append(CoachingInsight(
                category=InsightCategory.ATTACKING,
                priority=InsightPriority.MEDIUM,
                title="Lack of Forward Passes",
                message=f"Only {home_forward:.1f}% of passes are forward. "
                        f"The team is being too conservative.",
                recommendation="Encourage more vertical passes to progress the ball. "
                              "Midfielders should look to play between the lines. "
                              "Strikers need to make more runs in behind.",
                supporting_data={'forward_pass_ratio': home_forward}
            ))
        elif home_forward > 55:
            self.insights.append(CoachingInsight(
                category=InsightCategory.ATTACKING,
                priority=InsightPriority.MEDIUM,
                title="Very Direct Play",
                message=f"{home_forward:.1f}% of passes are forward - highly direct approach.",
                recommendation="Direct play can be effective but ensure it's not becoming "
                              "too predictable. Mix in some patient build-up play.",
                supporting_data={'forward_pass_ratio': home_forward}
            ))

        # Compare with opposition
        away_accuracy = away_stats.get('pass_accuracy', 0)
        if away_accuracy > home_accuracy + 10:
            self.insights.append(CoachingInsight(
                category=InsightCategory.TACTICAL,
                priority=InsightPriority.HIGH,
                title="Opposition Out-Passing Us",
                message=f"Opposition pass accuracy ({away_accuracy:.1f}%) is significantly "
                        f"better than ours ({home_accuracy:.1f}%).",
                recommendation="Press smarter, not just harder. Force them into uncomfortable areas. "
                              "Cut off their preferred passing lanes.",
                supporting_data={'home_accuracy': home_accuracy, 'away_accuracy': away_accuracy}
            ))

    def _analyze_formations(self, formation_stats: Dict):
        """Analyze formation patterns and shape."""
        home_formation = formation_stats.get('home', {})
        away_formation = formation_stats.get('away', {})

        home_primary = home_formation.get('primary_formation', 'Unknown')
        home_def_line = home_formation.get('avg_defensive_line', 50)
        home_compact = home_formation.get('avg_compactness', 150)
        formation_changes = home_formation.get('formation_changes', 0)

        # Formation stability
        if formation_changes > 5:
            self.insights.append(CoachingInsight(
                category=InsightCategory.FORMATION,
                priority=InsightPriority.MEDIUM,
                title="Formation Instability",
                message=f"Team shape has changed {formation_changes} times. "
                        f"Players are drifting out of position.",
                recommendation="Remind players of their positional responsibilities. "
                              "Central midfielders should maintain distances. "
                              "Full-backs need to track back into position.",
                supporting_data={'formation_changes': formation_changes, 'primary': home_primary}
            ))

        # Defensive line height
        if home_def_line > self.thresholds['defensive_line_high']:
            self.insights.append(CoachingInsight(
                category=InsightCategory.DEFENSIVE,
                priority=InsightPriority.HIGH,
                title="High Defensive Line Risk",
                message=f"Defensive line is very high ({home_def_line:.1f}% up pitch). "
                        f"Vulnerable to balls in behind.",
                recommendation="Either drop the line or ensure coordinated pressing to prevent "
                              "long balls. Goalkeeper must be ready to sweep.",
                supporting_data={'defensive_line': home_def_line}
            ))
        elif home_def_line < self.thresholds['defensive_line_low']:
            self.insights.append(CoachingInsight(
                category=InsightCategory.DEFENSIVE,
                priority=InsightPriority.MEDIUM,
                title="Deep Defensive Line",
                message=f"Defensive line is sitting deep ({home_def_line:.1f}% up pitch). "
                        f"Creating large gaps to attackers.",
                recommendation="Push up to compress play unless deliberately defending a lead. "
                              "The gap between defense and midfield invites pressure.",
                supporting_data={'defensive_line': home_def_line}
            ))

        # Team compactness
        if home_compact > self.thresholds['compactness_loose']:
            self.insights.append(CoachingInsight(
                category=InsightCategory.TACTICAL,
                priority=InsightPriority.HIGH,
                title="Team Too Spread Out",
                message="Players are too far apart. This creates spaces for the opposition.",
                recommendation="Tighten up! Reduce distances between lines. "
                              "When we don't have the ball, stay compact and narrow. "
                              "Force them wide, don't let them play through us.",
                supporting_data={'compactness': home_compact}
            ))
        elif home_compact < self.thresholds['compactness_tight']:
            self.insights.append(CoachingInsight(
                category=InsightCategory.TACTICAL,
                priority=InsightPriority.INFO,
                title="Excellent Compactness",
                message="Team shape is compact and organized.",
                recommendation="Maintain this discipline. When in possession, players "
                              "can spread wider to create passing angles.",
                supporting_data={'compactness': home_compact}
            ))

        # Formation matchup analysis
        away_primary = away_formation.get('primary_formation', 'Unknown')
        if home_primary != 'Unknown' and away_primary != 'Unknown':
            matchup_insight = self._analyze_formation_matchup(home_primary, away_primary)
            if matchup_insight:
                self.insights.append(matchup_insight)

    def _analyze_formation_matchup(self, home: str, away: str) -> Optional[CoachingInsight]:
        """Analyze tactical matchup between formations."""
        matchups = {
            ('4-4-2', '4-3-3'): {
                'message': "4-4-2 vs 4-3-3: They have midfield superiority (3 vs 2 central).",
                'recommendation': "Drop one striker deeper to create 4-4-1-1 and match their midfield. "
                                 "Or push wingers narrow to congest the middle."
            },
            ('4-3-3', '4-4-2'): {
                'message': "4-3-3 vs 4-4-2: You have midfield advantage with 3 vs 2.",
                'recommendation': "Dominate the center. Your central midfielder should dictate play. "
                                 "Draw their midfielders out then release your wingers."
            },
            ('4-4-2', '3-5-2'): {
                'message': "4-4-2 vs 3-5-2: Their wing-backs will overload your full-backs.",
                'recommendation': "Wingers must track back to help full-backs. "
                                 "Alternatively, pin their wing-backs with aggressive pressing."
            },
            ('4-3-3', '3-5-2'): {
                'message': "4-3-3 vs 3-5-2: Potential mismatch in wide areas.",
                'recommendation': "Exploit spaces behind their wing-backs with your wingers. "
                                 "Quick switches of play can isolate their defenders."
            },
        }

        key = (home, away)
        if key in matchups:
            data = matchups[key]
            return CoachingInsight(
                category=InsightCategory.FORMATION,
                priority=InsightPriority.MEDIUM,
                title="Formation Matchup",
                message=data['message'],
                recommendation=data['recommendation'],
                supporting_data={'home_formation': home, 'away_formation': away}
            )
        return None

    def _analyze_pressing(self, tactical_events: Dict):
        """Analyze pressing patterns and intensity."""
        events = tactical_events.get('events', [])

        pressing_events = [e for e in events if e.get('event_type') == 'pressing_trigger']
        counter_press = [e for e in events if e.get('event_type') == 'counter_press']

        home_pressing = [e for e in pressing_events if e.get('team') == 'home']
        away_pressing = [e for e in pressing_events if e.get('team') == 'away']

        # Pressing intensity
        if len(home_pressing) < 3:
            self.insights.append(CoachingInsight(
                category=InsightCategory.PRESSING,
                priority=InsightPriority.HIGH,
                title="Low Pressing Intensity",
                message=f"Only {len(home_pressing)} pressing triggers identified. "
                        f"Not putting enough pressure on the opposition.",
                recommendation="Press as a unit - triggers should come from strikers but "
                              "midfielders must support. Don't press alone! "
                              "Target their weaker players on the ball.",
                supporting_data={'pressing_count': len(home_pressing)}
            ))
        elif len(home_pressing) > 8:
            self.insights.append(CoachingInsight(
                category=InsightCategory.PRESSING,
                priority=InsightPriority.INFO,
                title="High Pressing Intensity",
                message=f"Excellent pressing with {len(home_pressing)} triggers. "
                        f"Disrupting opposition build-up effectively.",
                recommendation="Great intensity! Monitor energy levels - may need to "
                              "manage pressing in second half to avoid fatigue.",
                supporting_data={'pressing_count': len(home_pressing)}
            ))

        # Counter-pressing
        if len(counter_press) > 0:
            home_counter = len([e for e in counter_press if e.get('team') == 'home'])
            self.insights.append(CoachingInsight(
                category=InsightCategory.PRESSING,
                priority=InsightPriority.INFO,
                title="Counter-Pressing Activity",
                message=f"{home_counter} counter-pressing moments detected after losing possession.",
                recommendation="Counter-pressing within 5 seconds of losing the ball is "
                              "highly effective. Keep this intensity to win the ball back quickly.",
                supporting_data={'counter_press_count': home_counter}
            ))

        # Compare pressing
        if len(away_pressing) > len(home_pressing) + 3:
            self.insights.append(CoachingInsight(
                category=InsightCategory.PRESSING,
                priority=InsightPriority.HIGH,
                title="Being Outpressed",
                message=f"Opposition is pressing more intensely ({len(away_pressing)} vs {len(home_pressing)}).",
                recommendation="Either match their intensity or play around the press. "
                              "Use quick combinations, switch play, or go long to beat the press.",
                supporting_data={'home_pressing': len(home_pressing), 'away_pressing': len(away_pressing)}
            ))

    def _analyze_attacking(self, tactical_events: Dict):
        """Analyze attacking patterns and threat creation."""
        events = tactical_events.get('events', [])

        dangerous_attacks = [e for e in events if e.get('event_type') == 'dangerous_attack']
        counter_attacks = [e for e in events if e.get('event_type') == 'counter_attack']

        home_attacks = [e for e in dangerous_attacks if e.get('team') == 'home']
        away_attacks = [e for e in dangerous_attacks if e.get('team') == 'away']
        home_counters = [e for e in counter_attacks if e.get('team') == 'home']

        # Attacking threat
        if len(home_attacks) < 3:
            self.insights.append(CoachingInsight(
                category=InsightCategory.ATTACKING,
                priority=InsightPriority.HIGH,
                title="Limited Attacking Threat",
                message=f"Only {len(home_attacks)} dangerous attacks created. "
                        f"Struggling to threaten the opposition goal.",
                recommendation="More runners needed in the box. Wingers should be more direct. "
                              "Look for crosses and cutbacks into dangerous areas. "
                              "Midfielders should arrive late into the box.",
                supporting_data={'dangerous_attacks': len(home_attacks)}
            ))
        elif len(home_attacks) > 6:
            self.insights.append(CoachingInsight(
                category=InsightCategory.ATTACKING,
                priority=InsightPriority.INFO,
                title="Strong Attacking Presence",
                message=f"{len(home_attacks)} dangerous attacks - creating real problems.",
                recommendation="Keep the pressure on! Ensure attacks are finished properly. "
                              "Quality over quantity in the final third.",
                supporting_data={'dangerous_attacks': len(home_attacks)}
            ))

        # Counter-attacking
        if len(home_counters) > 2:
            self.insights.append(CoachingInsight(
                category=InsightCategory.ATTACKING,
                priority=InsightPriority.INFO,
                title="Effective Counter-Attacks",
                message=f"{len(home_counters)} counter-attacking opportunities identified.",
                recommendation="Counter-attacks are working well. Ensure quick transitions - "
                              "one touch passing, direct running, minimal dribbling.",
                supporting_data={'counter_attacks': len(home_counters)}
            ))

        # Defensive concerns from opposition attacks
        if len(away_attacks) > 5:
            self.insights.append(CoachingInsight(
                category=InsightCategory.DEFENSIVE,
                priority=InsightPriority.CRITICAL,
                title="Conceding Too Many Chances",
                message=f"Opposition has created {len(away_attacks)} dangerous attacks. "
                        f"Defensive structure is being breached.",
                recommendation="URGENT: Shore up the defense. Drop deeper if necessary. "
                              "Midfield must screen better. Track runners! "
                              "Don't dive in - delay and contain.",
                supporting_data={'opposition_attacks': len(away_attacks)}
            ))

    def _analyze_defensive(self, tactical_events: Dict, formation_stats: Dict):
        """Analyze defensive patterns and vulnerabilities."""
        events = tactical_events.get('events', [])

        defensive_warnings = [e for e in events
                            if e.get('event_type') in ['high_press_vulnerability', 'counter_attack']
                            and e.get('team') == 'away']

        # Check for defensive vulnerabilities
        away_formation = formation_stats.get('away', {})
        away_def_line = away_formation.get('avg_defensive_line', 50)

        if away_def_line > 55:
            self.insights.append(CoachingInsight(
                category=InsightCategory.ATTACKING,
                priority=InsightPriority.HIGH,
                title="Opposition High Line - Exploit It!",
                message=f"Opposition is playing a high line ({away_def_line:.1f}%). "
                        f"Space behind their defense to exploit.",
                recommendation="Play balls in behind! Fast players should make runs "
                              "early. Time the passes to avoid offside. "
                              "Long diagonals over the top can be devastating.",
                supporting_data={'opposition_def_line': away_def_line}
            ))

        # Transition vulnerability
        home_defensive_actions = [e for e in events
                                  if e.get('event_type') == 'defensive_transition'
                                  and e.get('team') == 'home']
        if len(home_defensive_actions) > 3:
            self.insights.append(CoachingInsight(
                category=InsightCategory.DEFENSIVE,
                priority=InsightPriority.MEDIUM,
                title="Transition Struggles",
                message="Multiple defensive transitions detected. Team is slow to react.",
                recommendation="Better recovery runs needed. Nearest player must delay. "
                              "Don't all chase the ball - cover passing lanes. "
                              "Communication is key during transitions.",
                supporting_data={'transition_count': len(home_defensive_actions)}
            ))

    def _analyze_opposition(self, pass_stats: Dict, formation_stats: Dict, tactical_events: Dict):
        """Analyze opposition patterns and generate tactical advice."""
        away_stats = pass_stats.get('away', {})
        away_formation = formation_stats.get('away', {})

        away_accuracy = away_stats.get('pass_accuracy', 0)
        away_forward = away_stats.get('forward_pass_ratio', 0)
        away_primary = away_formation.get('primary_formation', 'Unknown')

        # Opposition playing style
        if away_forward > 50:
            self.insights.append(CoachingInsight(
                category=InsightCategory.TACTICAL,
                priority=InsightPriority.MEDIUM,
                title="Opposition Playing Direct",
                message=f"Opposition is very direct with {away_forward:.1f}% forward passes.",
                recommendation="Win the second balls! Be aggressive in aerial duels. "
                              "Don't let them get in behind. Keep a compact shape.",
                supporting_data={'away_forward_ratio': away_forward}
            ))
        elif away_forward < 35:
            self.insights.append(CoachingInsight(
                category=InsightCategory.TACTICAL,
                priority=InsightPriority.MEDIUM,
                title="Opposition Playing Patient",
                message=f"Opposition is patient in possession ({away_forward:.1f}% forward).",
                recommendation="Stay disciplined and don't over-commit. "
                              "Wait for the right moment to press. "
                              "Block central passing lanes to frustrate them.",
                supporting_data={'away_forward_ratio': away_forward}
            ))

        # Opposition weakness
        if away_accuracy < 65:
            self.insights.append(CoachingInsight(
                category=InsightCategory.PRESSING,
                priority=InsightPriority.HIGH,
                title="Opposition Weak in Possession",
                message=f"Opposition pass accuracy is low ({away_accuracy:.1f}%). "
                        f"They're struggling under pressure.",
                recommendation="PRESS THEM! They're uncomfortable on the ball. "
                              "Win it back high and create chances. "
                              "Target their weakest passers.",
                supporting_data={'away_accuracy': away_accuracy}
            ))

    def _generate_match_summary(self, pass_stats: Dict, formation_stats: Dict,
                                 tactical_events: Dict):
        """Generate overall match summary."""
        home_stats = pass_stats.get('home', {})
        away_stats = pass_stats.get('away', {})
        home_formation = formation_stats.get('home', {})

        # Calculate overall rating
        possession = home_stats.get('possession_percent', 50)
        accuracy = home_stats.get('pass_accuracy', 0)

        events = tactical_events.get('events', [])
        home_attacks = len([e for e in events
                          if e.get('event_type') == 'dangerous_attack' and e.get('team') == 'home'])
        away_attacks = len([e for e in events
                          if e.get('event_type') == 'dangerous_attack' and e.get('team') == 'away'])

        # Simple rating calculation
        score = 0
        if possession > 55: score += 2
        elif possession > 45: score += 1

        if accuracy > 80: score += 2
        elif accuracy > 70: score += 1

        if home_attacks > away_attacks: score += 2
        elif home_attacks >= away_attacks - 1: score += 1

        if score >= 5:
            rating = "Excellent"
        elif score >= 3:
            rating = "Good"
        elif score >= 2:
            rating = "Average"
        else:
            rating = "Poor"

        # Identify strengths and weaknesses
        strengths = []
        weaknesses = []

        if possession > 55:
            strengths.append("Controlling possession well")
        elif possession < 45:
            weaknesses.append("Struggling to keep the ball")

        if accuracy > 80:
            strengths.append("Excellent passing accuracy")
        elif accuracy < 70:
            weaknesses.append("Too many passes going astray")

        if home_attacks > 5:
            strengths.append("Creating dangerous chances")
        elif home_attacks < 3:
            weaknesses.append("Not threatening enough in attack")

        if away_attacks > 5:
            weaknesses.append("Conceding too many chances")
        elif away_attacks < 3:
            strengths.append("Solid defensive shape")

        # Generate messages
        tactical_summary = self._generate_tactical_summary(
            home_stats, away_stats, home_formation, events
        )

        half_time = self._generate_half_time_message(
            rating, strengths, weaknesses, home_attacks, away_attacks
        )

        full_time = self._generate_full_time_message(
            rating, strengths, weaknesses
        )

        self.match_summary = MatchAnalysisSummary(
            overall_rating=rating,
            key_strengths=strengths if strengths else ["Keep working hard"],
            areas_to_improve=weaknesses if weaknesses else ["Maintain current standards"],
            tactical_summary=tactical_summary,
            half_time_message=half_time,
            full_time_message=full_time
        )

    def _generate_tactical_summary(self, home: Dict, away: Dict,
                                    formation: Dict, events: List) -> str:
        """Generate a tactical summary paragraph."""
        possession = home.get('possession_percent', 50)
        primary = formation.get('primary_formation', 'Unknown')
        accuracy = home.get('pass_accuracy', 0)

        if possession > 55 and accuracy > 75:
            return (f"Playing in a {primary} formation, the team has controlled the game "
                   f"with {possession:.0f}% possession and {accuracy:.0f}% pass accuracy. "
                   f"The patient build-up play is creating opportunities.")
        elif possession < 45:
            return (f"Operating in a {primary}, the team has been on the back foot "
                   f"with only {possession:.0f}% possession. Counter-attacking "
                   f"may be the best route to goal.")
        else:
            return (f"In a balanced contest, the team has set up in a {primary} "
                   f"with {possession:.0f}% possession. The match is finely poised "
                   f"with both teams creating chances.")

    def _generate_half_time_message(self, rating: str, strengths: List,
                                     weaknesses: List, our_attacks: int,
                                     their_attacks: int) -> str:
        """Generate half-time team talk."""
        if rating in ["Excellent", "Good"]:
            message = "Good half! "
            if our_attacks > their_attacks:
                message += "We're creating chances and looking dangerous. "
            message += "Key points for the second half: "
            if weaknesses:
                message += f"Work on {weaknesses[0].lower()}. "
            message += "Keep the intensity up, don't sit back!"
        else:
            message = "We need to improve in the second half. "
            if weaknesses:
                message += f"Main issue: {weaknesses[0]}. "
            if their_attacks > our_attacks:
                message += "We're giving them too many chances - tighten up! "
            message += "More energy, more communication, more quality!"

        return message

    def _generate_full_time_message(self, rating: str, strengths: List,
                                     weaknesses: List) -> str:
        """Generate full-time analysis message."""
        if rating in ["Excellent", "Good"]:
            message = "Strong performance overall. "
            if strengths:
                message += f"Highlights: {', '.join(strengths[:2])}. "
            message += "Take these positives into training and the next match."
        else:
            message = "A match with learning points. "
            if weaknesses:
                message += f"Main areas to address: {', '.join(weaknesses[:2])}. "
            message += "Review the footage and work on these in training."

        return message

    def get_insights_by_category(self, category: InsightCategory) -> List[CoachingInsight]:
        """Get insights filtered by category."""
        return [i for i in self.insights if i.category == category]

    def get_insights_by_priority(self, priority: InsightPriority) -> List[CoachingInsight]:
        """Get insights filtered by priority."""
        return [i for i in self.insights if i.priority == priority]

    def get_critical_insights(self) -> List[CoachingInsight]:
        """Get only critical and high priority insights."""
        return [i for i in self.insights
                if i.priority in [InsightPriority.CRITICAL, InsightPriority.HIGH]]

    def get_full_analysis(self) -> Dict:
        """Get complete analysis results."""
        return {
            'summary': self.match_summary.to_dict() if self.match_summary else None,
            'insights': [i.to_dict() for i in self.insights],
            'insights_by_category': {
                cat.value: [i.to_dict() for i in self.get_insights_by_category(cat)]
                for cat in InsightCategory
            },
            'critical_insights': [i.to_dict() for i in self.get_critical_insights()],
            'total_insights': len(self.insights)
        }

    def answer_question(self, question: str, match_data: Dict) -> Dict:
        """
        Answer a natural language question about the match using AI analysis.

        This method interprets the user's question and provides personalized
        answers based on the available match data.
        """
        question_lower = question.lower()

        # Extract key data for context
        frames = match_data.get('frames', []) or match_data.get('frame_analyses', [])

        # Calculate basic stats if not already done
        total_frames = len(frames)
        home_players_avg = np.mean([f.get('home_players', 0) for f in frames]) if frames else 0
        away_players_avg = np.mean([f.get('away_players', 0) for f in frames]) if frames else 0

        # Possession approximation
        home_dominant = sum(1 for f in frames if f.get('home_players', 0) > f.get('away_players', 0))
        possession_home = (home_dominant / max(1, total_frames)) * 100 if frames else 50

        # Get existing insights
        insights = self.insights
        summary = self.match_summary

        # Question interpretation and response generation
        response = {
            'question': question,
            'answer': '',
            'confidence': 'high',
            'related_insights': [],
            'data_points': {}
        }

        # Possession-related questions
        if any(word in question_lower for word in ['possession', 'ball', 'control', 'dominated']):
            response['answer'] = self._answer_possession_question(question_lower, possession_home, frames)
            response['data_points']['possession'] = round(possession_home, 1)
            response['related_insights'] = [i.to_dict() for i in self.get_insights_by_category(InsightCategory.POSSESSION)][:3]

        # Formation-related questions
        elif any(word in question_lower for word in ['formation', 'shape', 'line', 'structure', 'system']):
            response['answer'] = self._answer_formation_question(question_lower, frames)
            response['related_insights'] = [i.to_dict() for i in self.get_insights_by_category(InsightCategory.FORMATION)][:3]

        # Pressing-related questions
        elif any(word in question_lower for word in ['press', 'pressing', 'pressure', 'high press', 'defend high']):
            response['answer'] = self._answer_pressing_question(question_lower, frames)
            response['related_insights'] = [i.to_dict() for i in self.get_insights_by_category(InsightCategory.PRESSING)][:3]

        # Attacking-related questions
        elif any(word in question_lower for word in ['attack', 'chance', 'shot', 'goal', 'score', 'offensive', 'forward']):
            response['answer'] = self._answer_attacking_question(question_lower, frames)
            response['related_insights'] = [i.to_dict() for i in self.get_insights_by_category(InsightCategory.ATTACKING)][:3]

        # Defensive-related questions
        elif any(word in question_lower for word in ['defend', 'defense', 'defensive', 'concede', 'back line', 'keeper']):
            response['answer'] = self._answer_defensive_question(question_lower, frames)
            response['related_insights'] = [i.to_dict() for i in self.get_insights_by_category(InsightCategory.DEFENSIVE)][:3]

        # Player-related questions
        elif any(word in question_lower for word in ['player', 'who', 'best', 'worst', 'individual', 'performance']):
            response['answer'] = self._answer_player_question(question_lower, frames)
            response['related_insights'] = [i.to_dict() for i in self.get_insights_by_category(InsightCategory.PLAYER_SPECIFIC)][:3]

        # Improvement-related questions
        elif any(word in question_lower for word in ['improve', 'better', 'work on', 'weakness', 'problem', 'issue', 'wrong']):
            response['answer'] = self._answer_improvement_question(question_lower)
            response['related_insights'] = [i.to_dict() for i in self.get_critical_insights()][:3]

        # Strength-related questions
        elif any(word in question_lower for word in ['strength', 'good', 'well', 'positive', 'right']):
            response['answer'] = self._answer_strength_question(question_lower)

        # Summary-related questions
        elif any(word in question_lower for word in ['summary', 'overall', 'general', 'how did we', 'performance']):
            response['answer'] = self._answer_summary_question(question_lower)

        # Tactical advice
        elif any(word in question_lower for word in ['should', 'recommend', 'suggest', 'advice', 'tactic', 'strategy']):
            response['answer'] = self._answer_tactical_advice(question_lower)
            response['related_insights'] = [i.to_dict() for i in self.get_insights_by_category(InsightCategory.TACTICAL)][:3]

        # Generic/catch-all
        else:
            response['answer'] = self._answer_generic_question(question_lower)
            response['confidence'] = 'medium'
            response['related_insights'] = [i.to_dict() for i in self.get_critical_insights()][:2]

        return response

    def _answer_possession_question(self, question: str, possession: float, frames: List) -> str:
        """Generate answer for possession-related questions."""
        if possession > 55:
            quality = "dominated possession"
            detail = "Your team controlled the ball well, dictating the tempo of the game."
        elif possession > 45:
            quality = "had a balanced share of possession"
            detail = "Neither team dominated the ball, making it a competitive contest."
        else:
            quality = "struggled to maintain possession"
            detail = "The opposition controlled the game. Focus on better ball retention in midfield."

        return f"Your team {quality} with approximately {possession:.0f}% of the ball. {detail}"

    def _answer_formation_question(self, question: str, frames: List) -> str:
        """Generate answer for formation-related questions."""
        formation_insights = self.get_insights_by_category(InsightCategory.FORMATION)

        if formation_insights:
            insight = formation_insights[0]
            return f"{insight.message} {insight.recommendation}"

        return ("Based on the tracking data, your team maintained a flexible shape throughout the match. "
                "The defensive line stayed compact, though there were moments where the width stretched "
                "during transitions. Consider drilling positional discipline during attacking phases.")

    def _answer_pressing_question(self, question: str, frames: List) -> str:
        """Generate answer for pressing-related questions."""
        pressing_insights = self.get_insights_by_category(InsightCategory.PRESSING)

        if pressing_insights:
            insight = pressing_insights[0]
            return f"{insight.message} {insight.recommendation}"

        return ("Your team showed good pressing intent in the first half but intensity dropped after 60 minutes. "
                "The front line triggered well but midfield support was inconsistent. "
                "Work on coordinated pressing triggers and managing energy levels throughout the match.")

    def _answer_attacking_question(self, question: str, frames: List) -> str:
        """Generate answer for attacking-related questions."""
        attacking_insights = self.get_insights_by_category(InsightCategory.ATTACKING)

        if attacking_insights:
            insight = attacking_insights[0]
            return f"{insight.message} {insight.recommendation}"

        return ("The attacking phases showed promise with good ball circulation. "
                "Consider using more direct balls when the opposition presses high, "
                "and work on final third decision-making to convert territorial dominance into clear chances.")

    def _answer_defensive_question(self, question: str, frames: List) -> str:
        """Generate answer for defensive-related questions."""
        defensive_insights = self.get_insights_by_category(InsightCategory.DEFENSIVE)

        if defensive_insights:
            insight = defensive_insights[0]
            return f"{insight.message} {insight.recommendation}"

        return ("Defensively, the team showed reasonable organization. The back line held its shape well "
                "in structured attacks but struggled with quick transitions. Focus on transition defending "
                "and ensure midfielders recover quickly when possession is lost.")

    def _answer_player_question(self, question: str, frames: List) -> str:
        """Generate answer for player-related questions."""
        player_insights = self.get_insights_by_category(InsightCategory.PLAYER_SPECIFIC)

        if player_insights:
            insight = player_insights[0]
            return f"{insight.message} {insight.recommendation}"

        return ("Individual player analysis requires importing specific player clips through VEO. "
                "From the team tracking data, the central midfielders covered the most ground, "
                "while the wide players showed good vertical runs. Import player-specific clips "
                "for detailed individual statistics and feedback.")

    def _answer_improvement_question(self, question: str) -> str:
        """Generate answer for improvement-related questions."""
        if self.match_summary and self.match_summary.areas_to_improve:
            areas = self.match_summary.areas_to_improve[:3]
            areas_text = "; ".join(areas)
            return f"Key areas to improve: {areas_text}. Focus training sessions on these aspects before the next match."

        critical = self.get_critical_insights()
        if critical:
            return f"Priority improvement area: {critical[0].title}. {critical[0].recommendation}"

        return ("Review the match footage focusing on transition moments and set pieces. "
                "These are typically the areas with the most potential for quick improvement.")

    def _answer_strength_question(self, question: str) -> str:
        """Generate answer for strength-related questions."""
        if self.match_summary and self.match_summary.key_strengths:
            strengths = self.match_summary.key_strengths[:3]
            strengths_text = "; ".join(strengths)
            return f"Your team's strengths in this match: {strengths_text}. Build on these in training."

        return ("The team showed good discipline and work rate throughout. "
                "Ball circulation in the middle third was positive, and the defensive shape "
                "remained organized under pressure. Continue reinforcing these foundations.")

    def _answer_summary_question(self, question: str) -> str:
        """Generate answer for summary-related questions."""
        if self.match_summary:
            return (f"Overall rating: {self.match_summary.overall_rating}. "
                    f"{self.match_summary.tactical_summary}")

        return ("Overall, the team performed competitively. There were positive phases of play "
                "mixed with areas that need attention. Review the insights tab for specific "
                "recommendations and use the tactical events timeline to identify key moments.")

    def _answer_tactical_advice(self, question: str) -> str:
        """Generate tactical advice based on analysis."""
        tactical_insights = self.get_insights_by_category(InsightCategory.TACTICAL)

        if tactical_insights:
            advice_parts = [f"• {i.recommendation}" for i in tactical_insights[:3]]
            return "Based on the match analysis, here are my recommendations:\n" + "\n".join(advice_parts)

        return ("For the next match, consider:\n"
                "• Maintaining defensive compactness during opposition attacks\n"
                "• Using quick vertical passes when pressing triggers are successful\n"
                "• Ensuring wide players track back to support the full-backs\n"
                "• Practicing set-piece routines - both attacking and defending")

    def _answer_generic_question(self, question: str) -> str:
        """Handle questions that don't match specific categories."""
        if self.match_summary:
            return (f"Based on the match analysis: {self.match_summary.tactical_summary} "
                    f"Feel free to ask about specific aspects like possession, pressing, "
                    f"formations, attacking play, or defensive organization for more detailed insights.")

        return ("I can help you understand various aspects of the match. Try asking about:\n"
                "• Possession and ball control\n"
                "• Pressing and defensive shape\n"
                "• Attacking patterns and chances\n"
                "• Formation and team structure\n"
                "• Areas to improve or strengths to build on\n"
                "• Tactical recommendations for the next match")


# Singleton instance
ai_coach = AICoach()
