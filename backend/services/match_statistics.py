"""
Professional Match Statistics Service

Tracks and calculates match statistics similar to professional platforms like VEO.
Provides possession analysis, pass statistics, shot tracking, and event counting.
"""
import numpy as np
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from collections import defaultdict


class PitchZone(Enum):
    """Pitch divided into thirds for positional analysis."""
    DEFENSIVE_THIRD = "defensive"
    MIDDLE_THIRD = "middle"
    ATTACKING_THIRD = "attacking"

    # Also track left/center/right
    LEFT = "left"
    CENTER = "center"
    RIGHT = "right"


class EventType(Enum):
    """Match events that can be detected."""
    GOAL = "goal"
    SHOT_ON_TARGET = "shot_on_target"
    SHOT_OFF_TARGET = "shot_off_target"
    SHOT_BLOCKED = "shot_blocked"
    CORNER = "corner"
    FREE_KICK = "free_kick"
    GOAL_KICK = "goal_kick"
    THROW_IN = "throw_in"
    PENALTY = "penalty"
    KICKOFF = "kickoff"
    HALF_TIME = "half_time"
    FULL_TIME = "full_time"
    OFFSIDE = "offside"
    FOUL = "foul"
    YELLOW_CARD = "yellow_card"
    RED_CARD = "red_card"
    SUBSTITUTION = "substitution"
    PASS = "pass"
    CROSS = "cross"
    TACKLE = "tackle"
    INTERCEPTION = "interception"
    CLEARANCE = "clearance"
    DRIBBLE = "dribble"
    AERIAL_DUEL = "aerial_duel"


@dataclass
class MatchEvent:
    """A detected or manually tagged match event."""
    event_id: str
    event_type: EventType
    timestamp_ms: int
    frame_number: int
    team: str  # 'home' or 'away'
    player_jersey: Optional[int] = None
    player_name: Optional[str] = None
    position_x: Optional[float] = None  # Pitch coordinates (0-100)
    position_y: Optional[float] = None
    success: bool = True  # For passes, shots, etc.
    target_player_jersey: Optional[int] = None  # For passes
    end_position_x: Optional[float] = None  # For passes/shots
    end_position_y: Optional[float] = None
    description: str = ""
    confidence: float = 1.0  # AI detection confidence
    manually_tagged: bool = False


@dataclass
class PossessionPeriod:
    """A period of possession by one team."""
    team: str
    start_frame: int
    end_frame: int
    start_timestamp_ms: int
    end_timestamp_ms: int
    touches: int = 0
    passes: int = 0
    zone: Optional[PitchZone] = None


@dataclass
class PassSequence:
    """A sequence of passes (pass string)."""
    team: str
    passes: List[MatchEvent]
    start_frame: int
    end_frame: int
    length: int = 0  # Number of passes
    zones_touched: Set[str] = field(default_factory=set)
    ended_with_shot: bool = False
    ended_with_goal: bool = False


@dataclass
class ShotData:
    """Data for a shot attempt."""
    event: MatchEvent
    xG: float = 0.0  # Expected goals value
    distance_to_goal: float = 0.0
    angle_to_goal: float = 0.0
    body_part: str = "foot"  # foot, head, other
    situation: str = "open_play"  # open_play, corner, free_kick, penalty


@dataclass
class TeamStatistics:
    """Aggregated statistics for a team."""
    possession_pct: float = 0.0
    total_passes: int = 0
    successful_passes: int = 0
    pass_accuracy: float = 0.0

    passes_defensive_third: int = 0
    passes_middle_third: int = 0
    passes_attacking_third: int = 0

    total_shots: int = 0
    shots_on_target: int = 0
    shots_off_target: int = 0
    shots_blocked: int = 0
    goals: int = 0

    corners: int = 0
    free_kicks: int = 0
    penalties: int = 0

    tackles: int = 0
    interceptions: int = 0
    clearances: int = 0

    fouls_committed: int = 0
    fouls_won: int = 0
    yellow_cards: int = 0
    red_cards: int = 0
    offsides: int = 0

    # Derived metrics
    shot_conversion_rate: float = 0.0
    xG_total: float = 0.0

    # Pass sequences
    longest_pass_sequence: int = 0
    avg_pass_sequence_length: float = 0.0


@dataclass
class PlayerStatistics:
    """Individual player statistics."""
    jersey_number: int
    player_name: Optional[str] = None
    team: str = "home"

    minutes_played: float = 0.0
    touches: int = 0

    passes_attempted: int = 0
    passes_completed: int = 0
    pass_accuracy: float = 0.0
    key_passes: int = 0  # Passes leading to shots
    assists: int = 0

    shots: int = 0
    shots_on_target: int = 0
    goals: int = 0
    xG: float = 0.0

    tackles: int = 0
    tackles_won: int = 0
    interceptions: int = 0
    clearances: int = 0
    aerial_duels_won: int = 0
    aerial_duels_lost: int = 0

    fouls_committed: int = 0
    fouls_won: int = 0
    yellow_cards: int = 0
    red_cards: int = 0

    # Position data for heatmap
    position_samples: List[Tuple[float, float, int]] = field(default_factory=list)  # (x, y, frame)


class MatchStatisticsService:
    """
    Comprehensive match statistics tracking service.

    Tracks possession, passes, shots, and all match events to provide
    professional-level analytics similar to VEO Analytics.
    """

    # Pitch dimensions for coordinate normalization (0-100 scale)
    PITCH_LENGTH = 100.0
    PITCH_WIDTH = 100.0

    # Goal position (center of goal at attacking end)
    GOAL_X = 100.0
    GOAL_Y = 50.0
    GOAL_WIDTH = 7.32  # meters, scaled

    def __init__(self):
        self.events: List[MatchEvent] = []
        self.possession_periods: List[PossessionPeriod] = []
        self.pass_sequences: List[PassSequence] = []
        self.shots: List[ShotData] = []

        self.home_stats = TeamStatistics()
        self.away_stats = TeamStatistics()
        self.player_stats: Dict[Tuple[str, int], PlayerStatistics] = {}  # (team, jersey) -> stats

        # Tracking state
        self.current_possession_team: Optional[str] = None
        self.current_possession_start: int = 0
        self.current_pass_sequence: List[MatchEvent] = []
        self.last_touch_team: Optional[str] = None
        self.last_touch_frame: int = 0

        # Frame tracking
        self.total_frames: int = 0
        self.fps: float = 30.0
        self.home_possession_frames: int = 0
        self.away_possession_frames: int = 0

        # Event ID counter
        self._event_counter: int = 0

    def _generate_event_id(self) -> str:
        """Generate unique event ID."""
        self._event_counter += 1
        return f"evt_{self._event_counter:06d}"

    def set_video_info(self, total_frames: int, fps: float):
        """Set video metadata."""
        self.total_frames = total_frames
        self.fps = fps

    def _get_pitch_zone(self, x: float, team: str) -> PitchZone:
        """Determine which third of the pitch a position is in."""
        # Adjust for team attacking direction
        if team == "away":
            x = 100 - x

        if x < 33.33:
            return PitchZone.DEFENSIVE_THIRD
        elif x < 66.67:
            return PitchZone.MIDDLE_THIRD
        else:
            return PitchZone.ATTACKING_THIRD

    def _calculate_xg(self, shot: MatchEvent) -> float:
        """
        Calculate expected goals (xG) for a shot.

        Based on distance and angle to goal, plus situation context.
        """
        if shot.position_x is None or shot.position_y is None:
            return 0.1  # Default low xG

        # Distance to goal center
        dx = self.GOAL_X - shot.position_x
        dy = self.GOAL_Y - shot.position_y
        distance = np.sqrt(dx**2 + dy**2)

        # Angle to goal (wider angle = better chance)
        angle = np.arctan2(self.GOAL_WIDTH/2, distance) * 2
        angle_degrees = np.degrees(angle)

        # Base xG from distance and angle
        # Simplified model - production would use ML model
        base_xg = 0.0

        if distance < 6:  # Inside 6-yard box
            base_xg = 0.6
        elif distance < 12:  # Inside penalty area
            base_xg = 0.3
        elif distance < 20:  # Edge of box
            base_xg = 0.1
        else:  # Long range
            base_xg = 0.03

        # Adjust for angle
        angle_factor = min(1.0, angle_degrees / 40)
        base_xg *= (0.5 + 0.5 * angle_factor)

        # Penalty = high xG
        if shot.event_type == EventType.PENALTY:
            base_xg = 0.76

        return round(base_xg, 3)

    def record_event(
        self,
        event_type: EventType,
        timestamp_ms: int,
        frame_number: int,
        team: str,
        player_jersey: Optional[int] = None,
        position_x: Optional[float] = None,
        position_y: Optional[float] = None,
        success: bool = True,
        target_player_jersey: Optional[int] = None,
        end_position_x: Optional[float] = None,
        end_position_y: Optional[float] = None,
        description: str = "",
        confidence: float = 1.0,
        manually_tagged: bool = False
    ) -> MatchEvent:
        """Record a match event."""
        event = MatchEvent(
            event_id=self._generate_event_id(),
            event_type=event_type,
            timestamp_ms=timestamp_ms,
            frame_number=frame_number,
            team=team,
            player_jersey=player_jersey,
            position_x=position_x,
            position_y=position_y,
            success=success,
            target_player_jersey=target_player_jersey,
            end_position_x=end_position_x,
            end_position_y=end_position_y,
            description=description,
            confidence=confidence,
            manually_tagged=manually_tagged
        )

        self.events.append(event)
        self._process_event(event)

        return event

    def _process_event(self, event: MatchEvent):
        """Process event to update statistics."""
        team_stats = self.home_stats if event.team == "home" else self.away_stats

        # Update team stats based on event type
        if event.event_type == EventType.GOAL:
            team_stats.goals += 1
            self._end_pass_sequence(ended_with_goal=True)

        elif event.event_type in (EventType.SHOT_ON_TARGET, EventType.SHOT_OFF_TARGET, EventType.SHOT_BLOCKED):
            team_stats.total_shots += 1
            if event.event_type == EventType.SHOT_ON_TARGET:
                team_stats.shots_on_target += 1
            elif event.event_type == EventType.SHOT_OFF_TARGET:
                team_stats.shots_off_target += 1
            else:
                team_stats.shots_blocked += 1

            # Calculate xG
            xg = self._calculate_xg(event)
            team_stats.xG_total += xg

            shot_data = ShotData(
                event=event,
                xG=xg,
                distance_to_goal=np.sqrt((self.GOAL_X - (event.position_x or 50))**2 +
                                        (self.GOAL_Y - (event.position_y or 50))**2)
            )
            self.shots.append(shot_data)
            self._end_pass_sequence(ended_with_shot=True)

        elif event.event_type == EventType.PASS:
            team_stats.total_passes += 1
            if event.success:
                team_stats.successful_passes += 1
                self._add_to_pass_sequence(event)
            else:
                self._end_pass_sequence()

            # Track zone
            if event.position_x is not None:
                zone = self._get_pitch_zone(event.position_x, event.team)
                if zone == PitchZone.DEFENSIVE_THIRD:
                    team_stats.passes_defensive_third += 1
                elif zone == PitchZone.MIDDLE_THIRD:
                    team_stats.passes_middle_third += 1
                else:
                    team_stats.passes_attacking_third += 1

        elif event.event_type == EventType.CORNER:
            team_stats.corners += 1

        elif event.event_type == EventType.FREE_KICK:
            team_stats.free_kicks += 1

        elif event.event_type == EventType.PENALTY:
            team_stats.penalties += 1

        elif event.event_type == EventType.TACKLE:
            team_stats.tackles += 1

        elif event.event_type == EventType.INTERCEPTION:
            team_stats.interceptions += 1
            self._end_pass_sequence()  # Opponent's pass sequence ends

        elif event.event_type == EventType.CLEARANCE:
            team_stats.clearances += 1

        elif event.event_type == EventType.FOUL:
            team_stats.fouls_committed += 1
            other_stats = self.away_stats if event.team == "home" else self.home_stats
            other_stats.fouls_won += 1

        elif event.event_type == EventType.YELLOW_CARD:
            team_stats.yellow_cards += 1

        elif event.event_type == EventType.RED_CARD:
            team_stats.red_cards += 1

        elif event.event_type == EventType.OFFSIDE:
            team_stats.offsides += 1

        # Update player stats if jersey known
        if event.player_jersey is not None:
            self._update_player_stats(event)

    def _add_to_pass_sequence(self, pass_event: MatchEvent):
        """Add a pass to the current pass sequence."""
        if self.current_pass_sequence and self.current_pass_sequence[-1].team != pass_event.team:
            self._end_pass_sequence()

        self.current_pass_sequence.append(pass_event)

    def _end_pass_sequence(self, ended_with_shot: bool = False, ended_with_goal: bool = False):
        """End the current pass sequence and record it."""
        if not self.current_pass_sequence:
            return

        sequence = PassSequence(
            team=self.current_pass_sequence[0].team,
            passes=self.current_pass_sequence.copy(),
            start_frame=self.current_pass_sequence[0].frame_number,
            end_frame=self.current_pass_sequence[-1].frame_number,
            length=len(self.current_pass_sequence),
            ended_with_shot=ended_with_shot,
            ended_with_goal=ended_with_goal
        )

        # Track zones touched
        for p in self.current_pass_sequence:
            if p.position_x is not None:
                zone = self._get_pitch_zone(p.position_x, p.team)
                sequence.zones_touched.add(zone.value)

        self.pass_sequences.append(sequence)

        # Update team stats
        team_stats = self.home_stats if sequence.team == "home" else self.away_stats
        team_stats.longest_pass_sequence = max(team_stats.longest_pass_sequence, sequence.length)

        self.current_pass_sequence = []

    def _update_player_stats(self, event: MatchEvent):
        """Update individual player statistics."""
        key = (event.team, event.player_jersey)

        if key not in self.player_stats:
            self.player_stats[key] = PlayerStatistics(
                jersey_number=event.player_jersey,
                player_name=event.player_name,
                team=event.team
            )

        stats = self.player_stats[key]

        if event.event_type == EventType.PASS:
            stats.passes_attempted += 1
            if event.success:
                stats.passes_completed += 1

        elif event.event_type == EventType.GOAL:
            stats.goals += 1

        elif event.event_type in (EventType.SHOT_ON_TARGET, EventType.SHOT_OFF_TARGET, EventType.SHOT_BLOCKED):
            stats.shots += 1
            if event.event_type == EventType.SHOT_ON_TARGET:
                stats.shots_on_target += 1
            stats.xG += self._calculate_xg(event)

        elif event.event_type == EventType.TACKLE:
            stats.tackles += 1

        elif event.event_type == EventType.INTERCEPTION:
            stats.interceptions += 1

        elif event.event_type == EventType.CLEARANCE:
            stats.clearances += 1

        elif event.event_type == EventType.FOUL:
            stats.fouls_committed += 1

        elif event.event_type == EventType.YELLOW_CARD:
            stats.yellow_cards += 1

        elif event.event_type == EventType.RED_CARD:
            stats.red_cards += 1

    def record_possession(self, frame_number: int, team: str, timestamp_ms: int):
        """Record which team has possession at this frame."""
        if team != self.current_possession_team:
            # End previous possession period
            if self.current_possession_team is not None:
                period = PossessionPeriod(
                    team=self.current_possession_team,
                    start_frame=self.current_possession_start,
                    end_frame=frame_number - 1,
                    start_timestamp_ms=int(self.current_possession_start / self.fps * 1000),
                    end_timestamp_ms=timestamp_ms
                )
                self.possession_periods.append(period)

            self.current_possession_team = team
            self.current_possession_start = frame_number

        # Track possession frames
        if team == "home":
            self.home_possession_frames += 1
        elif team == "away":
            self.away_possession_frames += 1

    def record_player_position(
        self,
        team: str,
        jersey_number: int,
        x: float,
        y: float,
        frame_number: int
    ):
        """Record player position for heatmap generation."""
        key = (team, jersey_number)

        if key not in self.player_stats:
            self.player_stats[key] = PlayerStatistics(
                jersey_number=jersey_number,
                team=team
            )

        self.player_stats[key].position_samples.append((x, y, frame_number))
        self.player_stats[key].touches += 1

    def calculate_final_stats(self):
        """Calculate final aggregated statistics."""
        total_possession = self.home_possession_frames + self.away_possession_frames

        if total_possession > 0:
            self.home_stats.possession_pct = round(self.home_possession_frames / total_possession * 100, 1)
            self.away_stats.possession_pct = round(self.away_possession_frames / total_possession * 100, 1)

        # Calculate pass accuracy
        if self.home_stats.total_passes > 0:
            self.home_stats.pass_accuracy = round(
                self.home_stats.successful_passes / self.home_stats.total_passes * 100, 1
            )
        if self.away_stats.total_passes > 0:
            self.away_stats.pass_accuracy = round(
                self.away_stats.successful_passes / self.away_stats.total_passes * 100, 1
            )

        # Calculate shot conversion
        if self.home_stats.total_shots > 0:
            self.home_stats.shot_conversion_rate = round(
                self.home_stats.goals / self.home_stats.total_shots * 100, 1
            )
        if self.away_stats.total_shots > 0:
            self.away_stats.shot_conversion_rate = round(
                self.away_stats.goals / self.away_stats.total_shots * 100, 1
            )

        # Calculate average pass sequence length
        home_sequences = [s for s in self.pass_sequences if s.team == "home"]
        away_sequences = [s for s in self.pass_sequences if s.team == "away"]

        if home_sequences:
            self.home_stats.avg_pass_sequence_length = round(
                sum(s.length for s in home_sequences) / len(home_sequences), 1
            )
        if away_sequences:
            self.away_stats.avg_pass_sequence_length = round(
                sum(s.length for s in away_sequences) / len(away_sequences), 1
            )

        # Calculate player pass accuracy
        for stats in self.player_stats.values():
            if stats.passes_attempted > 0:
                stats.pass_accuracy = round(
                    stats.passes_completed / stats.passes_attempted * 100, 1
                )

    def get_match_summary(self) -> Dict:
        """Get comprehensive match summary."""
        self.calculate_final_stats()

        return {
            "home": {
                "possession_pct": self.home_stats.possession_pct,
                "total_passes": self.home_stats.total_passes,
                "pass_accuracy": self.home_stats.pass_accuracy,
                "passes_by_third": {
                    "defensive": self.home_stats.passes_defensive_third,
                    "middle": self.home_stats.passes_middle_third,
                    "attacking": self.home_stats.passes_attacking_third
                },
                "shots": {
                    "total": self.home_stats.total_shots,
                    "on_target": self.home_stats.shots_on_target,
                    "off_target": self.home_stats.shots_off_target,
                    "blocked": self.home_stats.shots_blocked
                },
                "goals": self.home_stats.goals,
                "xG": round(self.home_stats.xG_total, 2),
                "shot_conversion": self.home_stats.shot_conversion_rate,
                "corners": self.home_stats.corners,
                "free_kicks": self.home_stats.free_kicks,
                "penalties": self.home_stats.penalties,
                "tackles": self.home_stats.tackles,
                "interceptions": self.home_stats.interceptions,
                "clearances": self.home_stats.clearances,
                "fouls": self.home_stats.fouls_committed,
                "yellow_cards": self.home_stats.yellow_cards,
                "red_cards": self.home_stats.red_cards,
                "offsides": self.home_stats.offsides,
                "pass_sequences": {
                    "longest": self.home_stats.longest_pass_sequence,
                    "average": self.home_stats.avg_pass_sequence_length
                }
            },
            "away": {
                "possession_pct": self.away_stats.possession_pct,
                "total_passes": self.away_stats.total_passes,
                "pass_accuracy": self.away_stats.pass_accuracy,
                "passes_by_third": {
                    "defensive": self.away_stats.passes_defensive_third,
                    "middle": self.away_stats.passes_middle_third,
                    "attacking": self.away_stats.passes_attacking_third
                },
                "shots": {
                    "total": self.away_stats.total_shots,
                    "on_target": self.away_stats.shots_on_target,
                    "off_target": self.away_stats.shots_off_target,
                    "blocked": self.away_stats.shots_blocked
                },
                "goals": self.away_stats.goals,
                "xG": round(self.away_stats.xG_total, 2),
                "shot_conversion": self.away_stats.shot_conversion_rate,
                "corners": self.away_stats.corners,
                "free_kicks": self.away_stats.free_kicks,
                "penalties": self.away_stats.penalties,
                "tackles": self.away_stats.tackles,
                "interceptions": self.away_stats.interceptions,
                "clearances": self.away_stats.clearances,
                "fouls": self.away_stats.fouls_committed,
                "yellow_cards": self.away_stats.yellow_cards,
                "red_cards": self.away_stats.red_cards,
                "offsides": self.away_stats.offsides,
                "pass_sequences": {
                    "longest": self.away_stats.longest_pass_sequence,
                    "average": self.away_stats.avg_pass_sequence_length
                }
            }
        }

    def get_events_by_type(self, event_type: EventType) -> List[MatchEvent]:
        """Get all events of a specific type."""
        return [e for e in self.events if e.event_type == event_type]

    def get_events_timeline(self) -> List[Dict]:
        """Get all events as a timeline for display."""
        return [
            {
                "id": e.event_id,
                "type": e.event_type.value,
                "timestamp_ms": e.timestamp_ms,
                "time_str": f"{e.timestamp_ms // 60000}:{(e.timestamp_ms // 1000) % 60:02d}",
                "team": e.team,
                "player": e.player_jersey,
                "description": e.description or e.event_type.value.replace("_", " ").title(),
                "confidence": e.confidence
            }
            for e in sorted(self.events, key=lambda x: x.timestamp_ms)
        ]

    def get_shot_map_data(self) -> Dict:
        """Get shot map data for visualization."""
        return {
            "home": [
                {
                    "x": s.event.position_x,
                    "y": s.event.position_y,
                    "xG": s.xG,
                    "result": "goal" if s.event.event_type == EventType.GOAL else
                              "on_target" if s.event.event_type == EventType.SHOT_ON_TARGET else
                              "off_target" if s.event.event_type == EventType.SHOT_OFF_TARGET else "blocked",
                    "player": s.event.player_jersey,
                    "timestamp_ms": s.event.timestamp_ms
                }
                for s in self.shots if s.event.team == "home"
            ],
            "away": [
                {
                    "x": 100 - s.event.position_x if s.event.position_x else None,
                    "y": s.event.position_y,
                    "xG": s.xG,
                    "result": "goal" if s.event.event_type == EventType.GOAL else
                              "on_target" if s.event.event_type == EventType.SHOT_ON_TARGET else
                              "off_target" if s.event.event_type == EventType.SHOT_OFF_TARGET else "blocked",
                    "player": s.event.player_jersey,
                    "timestamp_ms": s.event.timestamp_ms
                }
                for s in self.shots if s.event.team == "away"
            ],
            "summary": {
                "home": {
                    "total": len([s for s in self.shots if s.event.team == "home"]),
                    "goals": self.home_stats.goals,
                    "xG": round(self.home_stats.xG_total, 2),
                    "conversion": self.home_stats.shot_conversion_rate
                },
                "away": {
                    "total": len([s for s in self.shots if s.event.team == "away"]),
                    "goals": self.away_stats.goals,
                    "xG": round(self.away_stats.xG_total, 2),
                    "conversion": self.away_stats.shot_conversion_rate
                }
            }
        }

    def get_player_stats(self, team: str, jersey_number: int) -> Optional[Dict]:
        """Get statistics for a specific player."""
        key = (team, jersey_number)
        if key not in self.player_stats:
            return None

        stats = self.player_stats[key]
        return {
            "jersey_number": stats.jersey_number,
            "player_name": stats.player_name,
            "team": stats.team,
            "touches": stats.touches,
            "passes": {
                "attempted": stats.passes_attempted,
                "completed": stats.passes_completed,
                "accuracy": stats.pass_accuracy,
                "key_passes": stats.key_passes,
                "assists": stats.assists
            },
            "shooting": {
                "shots": stats.shots,
                "on_target": stats.shots_on_target,
                "goals": stats.goals,
                "xG": round(stats.xG, 2)
            },
            "defending": {
                "tackles": stats.tackles,
                "interceptions": stats.interceptions,
                "clearances": stats.clearances,
                "aerial_duels_won": stats.aerial_duels_won,
                "aerial_duels_lost": stats.aerial_duels_lost
            },
            "discipline": {
                "fouls_committed": stats.fouls_committed,
                "fouls_won": stats.fouls_won,
                "yellow_cards": stats.yellow_cards,
                "red_cards": stats.red_cards
            }
        }

    def get_all_player_stats(self) -> Dict[str, List[Dict]]:
        """Get statistics for all players grouped by team."""
        result = {"home": [], "away": []}

        for (team, jersey), stats in self.player_stats.items():
            player_data = self.get_player_stats(team, jersey)
            if player_data:
                result[team].append(player_data)

        # Sort by jersey number
        result["home"].sort(key=lambda x: x["jersey_number"])
        result["away"].sort(key=lambda x: x["jersey_number"])

        return result

    def reset(self):
        """Reset all statistics."""
        self.events.clear()
        self.possession_periods.clear()
        self.pass_sequences.clear()
        self.shots.clear()
        self.player_stats.clear()

        self.home_stats = TeamStatistics()
        self.away_stats = TeamStatistics()

        self.current_possession_team = None
        self.current_possession_start = 0
        self.current_pass_sequence = []
        self.home_possession_frames = 0
        self.away_possession_frames = 0
        self._event_counter = 0


# Global instance
match_statistics_service = MatchStatisticsService()
