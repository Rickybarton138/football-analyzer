"""
Team-Specific Analysis Service

Provides focused analysis for:
1. Your Team - Performance tracking, improvement areas, tactical patterns
2. Opponent Scouting - Weaknesses, tendencies, game plan generation
"""
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict
import statistics

from services.training_data import training_data_service, MatchAnnotation


@dataclass
class TeamProfile:
    """Profile for a team based on historical data."""
    team_name: str
    matches_analyzed: int = 0

    # Results
    wins: int = 0
    draws: int = 0
    losses: int = 0
    goals_scored: int = 0
    goals_conceded: int = 0

    # Averages
    avg_possession: Optional[float] = None
    avg_shots: Optional[float] = None
    avg_shots_on_target: Optional[float] = None
    avg_corners: Optional[float] = None
    avg_fouls: Optional[float] = None
    avg_xg: Optional[float] = None
    avg_xg_against: Optional[float] = None

    # Patterns
    home_record: Dict = field(default_factory=dict)
    away_record: Dict = field(default_factory=dict)
    form: List[str] = field(default_factory=list)  # Last 5: W/D/L

    # Detailed stats
    scoring_patterns: Dict = field(default_factory=dict)
    conceding_patterns: Dict = field(default_factory=dict)


@dataclass
class TeamStrength:
    """Identified strength or weakness."""
    category: str  # attacking, defending, possession, set_pieces, etc.
    description: str
    confidence: float  # 0-1
    evidence: List[str] = field(default_factory=list)


@dataclass
class TacticalRecommendation:
    """Tactical recommendation for a match."""
    priority: str  # high, medium, low
    area: str  # attack, defense, midfield, set_pieces
    recommendation: str
    reasoning: str


@dataclass
class OpponentReport:
    """Scouting report for an opponent."""
    team_name: str
    profile: TeamProfile

    # Analysis
    strengths: List[TeamStrength] = field(default_factory=list)
    weaknesses: List[TeamStrength] = field(default_factory=list)
    key_patterns: List[str] = field(default_factory=list)
    danger_areas: List[str] = field(default_factory=list)

    # Game plan
    recommended_formation: Optional[str] = None
    tactical_approach: str = ""
    key_battles: List[str] = field(default_factory=list)
    recommendations: List[TacticalRecommendation] = field(default_factory=list)


@dataclass
class MyTeamReport:
    """Performance report for your own team."""
    team_name: str
    profile: TeamProfile

    # Self-analysis
    strengths: List[TeamStrength] = field(default_factory=list)
    weaknesses: List[TeamStrength] = field(default_factory=list)
    trends: List[str] = field(default_factory=list)

    # Improvement areas
    improvement_areas: List[TacticalRecommendation] = field(default_factory=list)
    training_focus: List[str] = field(default_factory=list)


class TeamAnalysisService:
    """Service for team-specific analysis."""

    def __init__(self):
        self.my_team: Optional[str] = None

    def set_my_team(self, team_name: str):
        """Set the user's team for focused analysis."""
        self.my_team = team_name

    async def build_team_profile(self, team_name: str) -> TeamProfile:
        """Build a comprehensive profile for a team from training data."""
        matches = await training_data_service.list_matches()

        profile = TeamProfile(team_name=team_name)

        possessions = []
        shots = []
        shots_on_target = []
        corners = []
        fouls = []
        xgs = []
        xgs_against = []

        for match in matches:
            is_home = match.home_team.lower() == team_name.lower()
            is_away = match.away_team.lower() == team_name.lower()

            if not is_home and not is_away:
                continue

            profile.matches_analyzed += 1

            # Get scores
            home_score = match.final_score.get('home', 0)
            away_score = match.final_score.get('away', 0)

            if is_home:
                team_score = home_score
                opp_score = away_score
                team_possession = match.home_possession
                team_shots = match.home_shots
                team_sot = match.home_shots_on_target
                team_corners = match.home_corners
                team_fouls = match.home_fouls
                team_xg = match.home_xg
                opp_xg = match.away_xg
            else:
                team_score = away_score
                opp_score = home_score
                team_possession = match.away_possession
                team_shots = match.away_shots
                team_sot = match.away_shots_on_target
                team_corners = match.away_corners
                team_fouls = match.away_fouls
                team_xg = match.away_xg
                opp_xg = match.home_xg

            # Results
            profile.goals_scored += team_score
            profile.goals_conceded += opp_score

            if team_score > opp_score:
                profile.wins += 1
                result = 'W'
            elif team_score < opp_score:
                profile.losses += 1
                result = 'L'
            else:
                profile.draws += 1
                result = 'D'

            profile.form.append(result)

            # Home/Away split
            record_key = 'home_record' if is_home else 'away_record'
            record = getattr(profile, record_key)
            record['matches'] = record.get('matches', 0) + 1
            record['wins'] = record.get('wins', 0) + (1 if result == 'W' else 0)
            record['draws'] = record.get('draws', 0) + (1 if result == 'D' else 0)
            record['losses'] = record.get('losses', 0) + (1 if result == 'L' else 0)
            record['goals_for'] = record.get('goals_for', 0) + team_score
            record['goals_against'] = record.get('goals_against', 0) + opp_score

            # Collect stats for averaging
            if team_possession is not None:
                possessions.append(team_possession)
            if team_shots is not None:
                shots.append(team_shots)
            if team_sot is not None:
                shots_on_target.append(team_sot)
            if team_corners is not None:
                corners.append(team_corners)
            if team_fouls is not None:
                fouls.append(team_fouls)
            if team_xg is not None:
                xgs.append(team_xg)
            if opp_xg is not None:
                xgs_against.append(opp_xg)

            # Scoring patterns
            if team_score > 0:
                profile.scoring_patterns[f'{team_score}_goals'] = \
                    profile.scoring_patterns.get(f'{team_score}_goals', 0) + 1

            if opp_score > 0:
                profile.conceding_patterns[f'{opp_score}_conceded'] = \
                    profile.conceding_patterns.get(f'{opp_score}_conceded', 0) + 1

        # Calculate averages
        if possessions:
            profile.avg_possession = round(statistics.mean(possessions), 1)
        if shots:
            profile.avg_shots = round(statistics.mean(shots), 1)
        if shots_on_target:
            profile.avg_shots_on_target = round(statistics.mean(shots_on_target), 1)
        if corners:
            profile.avg_corners = round(statistics.mean(corners), 1)
        if fouls:
            profile.avg_fouls = round(statistics.mean(fouls), 1)
        if xgs:
            profile.avg_xg = round(statistics.mean(xgs), 2)
        if xgs_against:
            profile.avg_xg_against = round(statistics.mean(xgs_against), 2)

        # Keep only last 5 form
        profile.form = profile.form[-5:]

        return profile

    def _analyze_strengths_weaknesses(
        self,
        profile: TeamProfile,
        league_averages: Dict
    ) -> tuple[List[TeamStrength], List[TeamStrength]]:
        """Analyze team strengths and weaknesses compared to league average."""
        strengths = []
        weaknesses = []

        if profile.matches_analyzed < 3:
            return strengths, weaknesses

        # Goal scoring analysis
        goals_per_game = profile.goals_scored / profile.matches_analyzed
        league_avg_goals = league_averages.get('goals_per_game', 1.3)

        if goals_per_game > league_avg_goals * 1.2:
            strengths.append(TeamStrength(
                category="attacking",
                description="Strong goal scoring ability",
                confidence=min(0.9, profile.matches_analyzed / 20),
                evidence=[f"Avg {goals_per_game:.1f} goals/game vs league avg {league_avg_goals:.1f}"]
            ))
        elif goals_per_game < league_avg_goals * 0.8:
            weaknesses.append(TeamStrength(
                category="attacking",
                description="Below average goal scoring",
                confidence=min(0.9, profile.matches_analyzed / 20),
                evidence=[f"Avg {goals_per_game:.1f} goals/game vs league avg {league_avg_goals:.1f}"]
            ))

        # Defensive analysis
        conceded_per_game = profile.goals_conceded / profile.matches_analyzed
        league_avg_conceded = league_averages.get('goals_conceded_per_game', 1.3)

        if conceded_per_game < league_avg_conceded * 0.8:
            strengths.append(TeamStrength(
                category="defending",
                description="Strong defensive record",
                confidence=min(0.9, profile.matches_analyzed / 20),
                evidence=[f"Avg {conceded_per_game:.1f} conceded/game vs league avg {league_avg_conceded:.1f}"]
            ))
        elif conceded_per_game > league_avg_conceded * 1.2:
            weaknesses.append(TeamStrength(
                category="defending",
                description="Vulnerable defensively",
                confidence=min(0.9, profile.matches_analyzed / 20),
                evidence=[f"Avg {conceded_per_game:.1f} conceded/game vs league avg {league_avg_conceded:.1f}"]
            ))

        # Possession analysis
        if profile.avg_possession:
            if profile.avg_possession > 55:
                strengths.append(TeamStrength(
                    category="possession",
                    description="Dominant possession style",
                    confidence=0.8,
                    evidence=[f"Avg {profile.avg_possession}% possession"]
                ))
            elif profile.avg_possession < 45:
                # Could be tactical choice, not necessarily weakness
                weaknesses.append(TeamStrength(
                    category="possession",
                    description="Low possession - counter-attacking style or struggle to keep ball",
                    confidence=0.6,
                    evidence=[f"Avg {profile.avg_possession}% possession"]
                ))

        # Shot conversion
        if profile.avg_shots and profile.avg_shots > 0:
            conversion = (profile.goals_scored / profile.matches_analyzed) / profile.avg_shots * 100
            if conversion > 12:
                strengths.append(TeamStrength(
                    category="attacking",
                    description="Efficient finishing",
                    confidence=0.7,
                    evidence=[f"{conversion:.1f}% shot conversion rate"]
                ))
            elif conversion < 8:
                weaknesses.append(TeamStrength(
                    category="attacking",
                    description="Poor shot conversion",
                    confidence=0.7,
                    evidence=[f"{conversion:.1f}% shot conversion rate"]
                ))

        # xG analysis
        if profile.avg_xg and profile.matches_analyzed > 5:
            actual_goals_pg = profile.goals_scored / profile.matches_analyzed
            if actual_goals_pg > profile.avg_xg * 1.15:
                strengths.append(TeamStrength(
                    category="attacking",
                    description="Outperforming xG - clinical finishing",
                    confidence=0.75,
                    evidence=[f"Scoring {actual_goals_pg:.2f}/game vs {profile.avg_xg:.2f} xG"]
                ))
            elif actual_goals_pg < profile.avg_xg * 0.85:
                weaknesses.append(TeamStrength(
                    category="attacking",
                    description="Underperforming xG - wasteful in front of goal",
                    confidence=0.75,
                    evidence=[f"Scoring {actual_goals_pg:.2f}/game vs {profile.avg_xg:.2f} xG"]
                ))

        # Home/Away analysis
        if profile.home_record.get('matches', 0) >= 3:
            home_ppg = (profile.home_record.get('wins', 0) * 3 +
                       profile.home_record.get('draws', 0)) / profile.home_record['matches']
            if home_ppg > 2.0:
                strengths.append(TeamStrength(
                    category="home",
                    description="Strong home form",
                    confidence=0.8,
                    evidence=[f"{home_ppg:.1f} points per game at home"]
                ))
            elif home_ppg < 1.0:
                weaknesses.append(TeamStrength(
                    category="home",
                    description="Poor home form",
                    confidence=0.8,
                    evidence=[f"{home_ppg:.1f} points per game at home"]
                ))

        if profile.away_record.get('matches', 0) >= 3:
            away_ppg = (profile.away_record.get('wins', 0) * 3 +
                       profile.away_record.get('draws', 0)) / profile.away_record['matches']
            if away_ppg > 1.5:
                strengths.append(TeamStrength(
                    category="away",
                    description="Good away form",
                    confidence=0.8,
                    evidence=[f"{away_ppg:.1f} points per game away"]
                ))
            elif away_ppg < 0.7:
                weaknesses.append(TeamStrength(
                    category="away",
                    description="Struggle away from home",
                    confidence=0.8,
                    evidence=[f"{away_ppg:.1f} points per game away"]
                ))

        return strengths, weaknesses

    async def _get_league_averages(self) -> Dict:
        """Calculate league averages from training data."""
        matches = await training_data_service.list_matches()

        if not matches:
            return {
                'goals_per_game': 1.3,
                'goals_conceded_per_game': 1.3,
                'possession': 50,
                'shots': 12,
            }

        total_goals = 0
        total_matches = len(matches)

        for match in matches:
            total_goals += match.final_score.get('home', 0)
            total_goals += match.final_score.get('away', 0)

        return {
            'goals_per_game': total_goals / (total_matches * 2) if total_matches > 0 else 1.3,
            'goals_conceded_per_game': total_goals / (total_matches * 2) if total_matches > 0 else 1.3,
            'possession': 50,
            'shots': 12,
        }

    async def analyze_my_team(self, team_name: str) -> MyTeamReport:
        """Generate a comprehensive report for your own team."""
        profile = await self.build_team_profile(team_name)
        league_avgs = await self._get_league_averages()

        strengths, weaknesses = self._analyze_strengths_weaknesses(profile, league_avgs)

        report = MyTeamReport(
            team_name=team_name,
            profile=profile,
            strengths=strengths,
            weaknesses=weaknesses,
        )

        # Identify trends
        if len(profile.form) >= 3:
            recent_form = profile.form[-3:]
            if recent_form.count('W') >= 2:
                report.trends.append("Team is in good form - won 2+ of last 3")
            elif recent_form.count('L') >= 2:
                report.trends.append("Team struggling - lost 2+ of last 3")

            if len(profile.form) >= 5:
                first_half = profile.form[:len(profile.form)//2]
                second_half = profile.form[len(profile.form)//2:]
                first_points = first_half.count('W') * 3 + first_half.count('D')
                second_points = second_half.count('W') * 3 + second_half.count('D')

                if second_points > first_points * 1.3:
                    report.trends.append("Form improving over recent matches")
                elif second_points < first_points * 0.7:
                    report.trends.append("Form declining - need to address issues")

        # Generate improvement recommendations
        for weakness in weaknesses:
            if weakness.category == "attacking":
                report.improvement_areas.append(TacticalRecommendation(
                    priority="high",
                    area="attack",
                    recommendation="Focus on creating higher quality chances",
                    reasoning=weakness.description
                ))
                report.training_focus.append("Finishing drills and shooting practice")
                report.training_focus.append("Movement in the final third")

            elif weakness.category == "defending":
                report.improvement_areas.append(TacticalRecommendation(
                    priority="high",
                    area="defense",
                    recommendation="Improve defensive organization",
                    reasoning=weakness.description
                ))
                report.training_focus.append("Defensive shape and positioning")
                report.training_focus.append("Set piece defending")

            elif weakness.category == "possession":
                report.improvement_areas.append(TacticalRecommendation(
                    priority="medium",
                    area="midfield",
                    recommendation="Work on ball retention and passing patterns",
                    reasoning=weakness.description
                ))
                report.training_focus.append("Passing combinations and rondos")
                report.training_focus.append("Press resistance")

            elif weakness.category == "away":
                report.improvement_areas.append(TacticalRecommendation(
                    priority="medium",
                    area="tactical",
                    recommendation="Develop better away game strategy",
                    reasoning=weakness.description
                ))
                report.training_focus.append("Defensive solidity exercises")
                report.training_focus.append("Counter-attacking patterns")

        return report

    async def scout_opponent(
        self,
        opponent_name: str,
        my_team_name: Optional[str] = None
    ) -> OpponentReport:
        """Generate a scouting report for an opponent with tactical recommendations."""
        profile = await self.build_team_profile(opponent_name)
        league_avgs = await self._get_league_averages()

        strengths, weaknesses = self._analyze_strengths_weaknesses(profile, league_avgs)

        report = OpponentReport(
            team_name=opponent_name,
            profile=profile,
            strengths=strengths,
            weaknesses=weaknesses,
        )

        # Identify key patterns
        if profile.matches_analyzed > 0:
            goals_per_game = profile.goals_scored / profile.matches_analyzed
            conceded_per_game = profile.goals_conceded / profile.matches_analyzed

            if goals_per_game > 2:
                report.key_patterns.append(f"High scoring team - avg {goals_per_game:.1f} goals/game")
                report.danger_areas.append("Potent attack - defensive concentration crucial")

            if conceded_per_game > 1.5:
                report.key_patterns.append(f"Concede frequently - avg {conceded_per_game:.1f}/game")

            if profile.avg_possession and profile.avg_possession > 55:
                report.key_patterns.append(f"Possession-based team - {profile.avg_possession}% avg")
                report.danger_areas.append("Will try to control the game")
            elif profile.avg_possession and profile.avg_possession < 45:
                report.key_patterns.append("Counter-attacking style - low possession")
                report.danger_areas.append("Dangerous on the break")

        # Generate tactical recommendations based on weaknesses
        for weakness in weaknesses:
            if weakness.category == "defending":
                report.recommendations.append(TacticalRecommendation(
                    priority="high",
                    area="attack",
                    recommendation="Press high and attack aggressively",
                    reasoning=f"Opponent vulnerable defensively: {weakness.description}"
                ))

            elif weakness.category == "attacking":
                report.recommendations.append(TacticalRecommendation(
                    priority="medium",
                    area="defense",
                    recommendation="Stay compact and limit space - they struggle to score",
                    reasoning=f"Opponent struggles in attack: {weakness.description}"
                ))

            elif weakness.category == "possession":
                report.recommendations.append(TacticalRecommendation(
                    priority="medium",
                    area="midfield",
                    recommendation="Dominate midfield and control possession",
                    reasoning=f"Can win the midfield battle: {weakness.description}"
                ))

            elif weakness.category == "away":
                report.recommendations.append(TacticalRecommendation(
                    priority="high",
                    area="tactical",
                    recommendation="Start fast at home, put them under pressure early",
                    reasoning=f"Poor away record: {weakness.description}"
                ))

            elif weakness.category == "home":
                report.recommendations.append(TacticalRecommendation(
                    priority="high",
                    area="tactical",
                    recommendation="Play with confidence - they don't dominate at home",
                    reasoning=f"Weak home form: {weakness.description}"
                ))

        # Counter their strengths
        for strength in strengths:
            if strength.category == "attacking":
                report.recommendations.append(TacticalRecommendation(
                    priority="high",
                    area="defense",
                    recommendation="Prioritize defensive organization against their attack",
                    reasoning=f"Must nullify: {strength.description}"
                ))

            elif strength.category == "possession":
                report.recommendations.append(TacticalRecommendation(
                    priority="medium",
                    area="tactical",
                    recommendation="Organized press or sit deep and counter",
                    reasoning=f"They will dominate ball: {strength.description}"
                ))

        # Suggest formation based on opponent profile
        if profile.avg_possession and profile.avg_possession > 55:
            report.recommended_formation = "4-5-1 or 5-4-1"
            report.tactical_approach = "Compact defensive shape, quick transitions"
        elif any(w.category == "defending" for w in weaknesses):
            report.recommended_formation = "4-3-3 or 3-4-3"
            report.tactical_approach = "Attack-minded, press high, exploit defensive weaknesses"
        else:
            report.recommended_formation = "4-4-2 or 4-2-3-1"
            report.tactical_approach = "Balanced approach, solid defensively, clinical in attack"

        # Key battles
        if profile.avg_shots and profile.avg_shots > 15:
            report.key_battles.append("Block shots and limit shooting opportunities")
        if profile.avg_corners and profile.avg_corners > 6:
            report.key_battles.append("Set piece defending - they win many corners")

        report.key_battles.append("Win second balls in midfield")
        report.key_battles.append("Win aerial duels")

        return report

    async def head_to_head(
        self,
        team1: str,
        team2: str
    ) -> Dict:
        """Analyze head-to-head record between two teams."""
        matches = await training_data_service.list_matches()

        h2h = {
            'matches': 0,
            'team1_wins': 0,
            'team2_wins': 0,
            'draws': 0,
            'team1_goals': 0,
            'team2_goals': 0,
            'results': []
        }

        for match in matches:
            is_t1_home = match.home_team.lower() == team1.lower()
            is_t1_away = match.away_team.lower() == team1.lower()
            is_t2_home = match.home_team.lower() == team2.lower()
            is_t2_away = match.away_team.lower() == team2.lower()

            if (is_t1_home and is_t2_away) or (is_t1_away and is_t2_home):
                h2h['matches'] += 1

                home_score = match.final_score.get('home', 0)
                away_score = match.final_score.get('away', 0)

                if is_t1_home:
                    t1_score, t2_score = home_score, away_score
                else:
                    t1_score, t2_score = away_score, home_score

                h2h['team1_goals'] += t1_score
                h2h['team2_goals'] += t2_score

                if t1_score > t2_score:
                    h2h['team1_wins'] += 1
                    result = f"{team1} {t1_score}-{t2_score}"
                elif t2_score > t1_score:
                    h2h['team2_wins'] += 1
                    result = f"{team2} {t2_score}-{t1_score}"
                else:
                    h2h['draws'] += 1
                    result = f"Draw {t1_score}-{t2_score}"

                h2h['results'].append({
                    'date': match.date,
                    'result': result,
                    'venue': 'Home' if is_t1_home else 'Away'
                })

        return h2h

    async def get_all_teams(self) -> List[str]:
        """Get list of all teams in the training data."""
        matches = await training_data_service.list_matches()

        teams = set()
        for match in matches:
            teams.add(match.home_team)
            teams.add(match.away_team)

        return sorted(list(teams))


# Global service instance
team_analysis_service = TeamAnalysisService()
