"""
AI Coaching Engine — Unified Claude + OpenAI Vision integration.

Claude (primary): coaching narratives, IDP, MDP, tactical analysis
OpenAI Vision (secondary): clip frame visual assessment
"""
import base64
import json
import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any

import cv2
import numpy as np

from config import settings
from services.player_clip_analyzer import ClipMetrics, POSITION_BENCHMARKS

logger = logging.getLogger(__name__)


# ─── Data Structures ────────────────────────────────────────────────────

@dataclass
class ClipFeedback:
    """AI-generated coaching feedback for a single clip."""
    clip_id: str = ""
    event_type: str = ""
    overall_rating: float = 0.0          # 1-10
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    coaching_points: List[str] = field(default_factory=list)
    drill_recommendations: List[str] = field(default_factory=list)
    raw_text: str = ""


@dataclass
class VisualAssessment:
    """OpenAI Vision analysis of clip key frames."""
    body_shape_score: float = 0.0        # 1-10
    positioning_score: float = 0.0       # 1-10
    technique_notes: List[str] = field(default_factory=list)
    observations: List[str] = field(default_factory=list)
    raw_text: str = ""


@dataclass
class FourCornerRating:
    """UEFA Four Corner Model assessment."""
    technical: float = 0.0
    tactical: float = 0.0
    physical: float = 0.0
    psychological: float = 0.0


@dataclass
class DevelopmentPriority:
    """A single development priority with drill prescription."""
    priority_level: str = "MEDIUM"       # HIGH / MEDIUM / LOW
    area: str = ""
    detail: str = ""
    metric_current: str = ""
    metric_benchmark: str = ""
    drill: str = ""
    drill_description: str = ""


@dataclass
class PlayerDevelopmentPlan:
    """Full Individual Development Plan (IDP) for a player."""
    jersey_number: int = 0
    player_name: str = ""
    position: str = ""
    team: str = ""
    overall_rating: float = 0.0
    four_corner: FourCornerRating = field(default_factory=FourCornerRating)
    key_strengths: List[str] = field(default_factory=list)
    development_priorities: List[DevelopmentPriority] = field(default_factory=list)
    weekly_focus: str = ""
    session_plan: str = ""
    three_month_goals: List[str] = field(default_factory=list)
    six_month_goals: List[str] = field(default_factory=list)
    raw_text: str = ""
    generated_at: str = ""


@dataclass
class ManagerDevelopmentPlan:
    """Manager Development Plan (MDP) — tactical assessment."""
    formation_management_score: float = 0.0
    pressing_strategy_score: float = 0.0
    transition_score: float = 0.0
    tactical_strengths: List[str] = field(default_factory=list)
    tactical_weaknesses: List[str] = field(default_factory=list)
    development_priorities: List[DevelopmentPriority] = field(default_factory=list)
    coaching_education: List[str] = field(default_factory=list)
    raw_text: str = ""
    generated_at: str = ""


# ─── System Prompts ──────────────────────────────────────────────────────

COACHING_SYSTEM_PROMPT = """You are an elite football coaching analyst with UEFA Pro Licence expertise.
Your role is to analyse player performance data from video analysis and provide actionable,
position-specific coaching feedback.

RULES:
- Be specific and data-driven — reference the metrics provided
- Use the UEFA Four Corner Model (Technical, Tactical, Physical, Psychological)
- Compare metrics to the position benchmarks when available
- Every weakness must come with a specific drill recommendation
- Drills should be FA-standard: describe setup, progression, coaching points
- Use professional coaching language but keep it accessible
- Be encouraging but honest — elite coaching is truthful
- Keep responses under the requested word limit"""

IDP_SYSTEM_PROMPT = """You are an elite football development coach generating an Individual Development Plan.
Use the UEFA Four Corner Model. Reference the clip metrics data provided.

OUTPUT FORMAT (use this exact structure):
OVERALL RATING: X.X/10

FOUR CORNER ASSESSMENT:
- Technical: X.X (brief justification)
- Tactical: X.X (brief justification)
- Physical: X.X (brief justification)
- Psychological: X.X (brief justification)

KEY STRENGTHS:
1. (strength with metric evidence)
2. (strength with metric evidence)
3. (strength with metric evidence)

DEVELOPMENT PRIORITIES:
1. [HIGH/MEDIUM/LOW] Area — current metric vs benchmark
   Drill: Name — setup and coaching points
2. ...
3. ...

WEEKLY FOCUS: (single priority area)
SESSION PLAN: (15min + 20min + 15min structure)

3-MONTH GOALS: (2-3 measurable targets)
6-MONTH GOALS: (2-3 measurable targets)"""

MDP_SYSTEM_PROMPT = """You are an elite football tactical analyst generating a Manager Development Plan.
Analyse the tactical data from video analysis and assess the manager's decisions.

OUTPUT FORMAT:
FORMATION MANAGEMENT: X.X/10
- (formation analysis)

PRESSING STRATEGY: X.X/10
- (pressing analysis)

TRANSITION MANAGEMENT: X.X/10
- (transition analysis)

TACTICAL STRENGTHS:
1. ...
2. ...

TACTICAL WEAKNESSES:
1. ...
2. ...

DEVELOPMENT PRIORITIES:
1. [HIGH/MEDIUM/LOW] Area — evidence
   Recommendation: ...
2. ...

COACHING EDUCATION:
- (recommended reading/study/session focus)"""


# ─── Engine ──────────────────────────────────────────────────────────────

class AICoachingEngine:
    """Unified AI coaching engine for clip feedback, IDP, MDP, and tactical narratives."""

    def __init__(self):
        self.claude_client = None
        self.openai_client = None
        self._initialized = False

    def initialize(self, claude_client=None, openai_client=None):
        """Initialize with AI clients."""
        self.claude_client = claude_client
        self.openai_client = openai_client
        self._initialized = True
        logger.info(
            f"AI Coaching Engine initialized — "
            f"Claude: {'yes' if claude_client else 'no'}, "
            f"OpenAI: {'yes' if openai_client else 'no'}"
        )

    @property
    def is_available(self) -> bool:
        return self.claude_client is not None

    # ── Clip-Level Feedback ──────────────────────────────────────────────

    async def analyze_player_clip(
        self,
        clip_metrics: ClipMetrics,
        player_position: str = "CM",
        player_name: str = "",
    ) -> ClipFeedback:
        """Generate coaching feedback for a single clip from its metrics."""
        if not self.claude_client:
            return self._rule_based_clip_feedback(clip_metrics)

        benchmarks = POSITION_BENCHMARKS.get(player_position, {})
        metrics_summary = self._format_metrics_for_prompt(clip_metrics, benchmarks)

        prompt = f"""Analyse this player clip and provide coaching feedback.

Player: {player_name or 'Unknown'} | Position: {player_position}
Clip Duration: {clip_metrics.duration_seconds:.1f}s
Event Type: {clip_metrics.events[0].get('event_type', 'general') if clip_metrics.events else 'general'}

METRICS:
{metrics_summary}

Provide:
1. Overall rating (1-10)
2. 2-3 strengths observed
3. 1-2 areas for improvement
4. 1-2 specific coaching points
5. 1 drill recommendation

Keep response under 200 words."""

        try:
            response = await self.claude_client.messages.create(
                model=settings.AI_COACHING_MODEL,
                max_tokens=settings.AI_COACHING_MAX_TOKENS,
                system=COACHING_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text

            return self._parse_clip_feedback(text, clip_metrics)
        except Exception as e:
            logger.error(f"Claude clip analysis failed: {e}")
            return self._rule_based_clip_feedback(clip_metrics)

    # ── Vision Frame Analysis ────────────────────────────────────────────

    async def analyze_clip_frames(
        self,
        frames: List[np.ndarray],
        context: str = "",
    ) -> VisualAssessment:
        """Analyse 3-5 key frames from a clip using OpenAI Vision."""
        if not self.openai_client:
            return VisualAssessment(observations=["Vision analysis unavailable — no OpenAI client"])

        # Encode frames as base64
        image_contents = []
        for i, frame in enumerate(frames[:5]):
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            b64 = base64.b64encode(buffer).decode('utf-8')
            image_contents.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "low"},
            })

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": (
                        f"Analyse these {len(frames)} frames from a football match clip. "
                        f"{context}\n\n"
                        "Assess: 1) Body shape/positioning (1-10), 2) Technique observations, "
                        "3) Key coaching points. Be concise (under 150 words)."
                    )},
                    *image_contents,
                ],
            }
        ]

        try:
            response = await self.openai_client.chat.completions.create(
                model=settings.AI_VISION_MODEL,
                max_tokens=settings.AI_VISION_MAX_TOKENS,
                messages=messages,
            )
            text = response.choices[0].message.content
            return VisualAssessment(
                body_shape_score=7.0,
                positioning_score=7.0,
                observations=text.split('\n') if text else [],
                raw_text=text or "",
            )
        except Exception as e:
            logger.error(f"OpenAI Vision analysis failed: {e}")
            return VisualAssessment(observations=[f"Vision analysis error: {str(e)}"])

    # ── IDP Generation ───────────────────────────────────────────────────

    async def generate_player_idp(
        self,
        jersey_number: int,
        player_name: str,
        team: str,
        position: str,
        clip_metrics_list: List[ClipMetrics],
        player_stats: Optional[Dict] = None,
    ) -> PlayerDevelopmentPlan:
        """Generate a full Individual Development Plan from aggregated clip data."""
        if not self.claude_client:
            return self._rule_based_idp(jersey_number, player_name, position, clip_metrics_list)

        benchmarks = POSITION_BENCHMARKS.get(position, POSITION_BENCHMARKS.get("CM", {}))

        # Aggregate metrics across all clips
        agg = self._aggregate_clip_metrics(clip_metrics_list)
        metrics_text = self._format_aggregated_metrics(agg, benchmarks, position)

        prompt = f"""Generate an Individual Development Plan for this player.

Player: #{jersey_number} {player_name}
Position: {position} | Team: {team}
Total Clips Analysed: {len(clip_metrics_list)}

AGGREGATED METRICS:
{metrics_text}

POSITION BENCHMARKS ({position}):
{json.dumps(benchmarks, indent=2)}

Generate the full IDP following the output format exactly."""

        try:
            response = await self.claude_client.messages.create(
                model=settings.AI_COACHING_MODEL,
                max_tokens=3000,
                system=IDP_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text
            return self._parse_idp(text, jersey_number, player_name, position, team)
        except Exception as e:
            logger.error(f"Claude IDP generation failed: {e}")
            return self._rule_based_idp(jersey_number, player_name, position, clip_metrics_list)

    # ── MDP Generation ───────────────────────────────────────────────────

    async def generate_manager_mdp(
        self,
        tactical_data: Dict,
        formation_data: Dict,
        event_summary: Optional[Dict] = None,
    ) -> ManagerDevelopmentPlan:
        """Generate a Manager Development Plan from tactical data."""
        if not self.claude_client:
            return self._rule_based_mdp(tactical_data, formation_data)

        prompt = f"""Generate a Manager Development Plan from this match data.

TACTICAL DATA:
{json.dumps(tactical_data, indent=2, default=str)}

FORMATION DATA:
{json.dumps(formation_data, indent=2, default=str)}

{f'EVENT SUMMARY: {json.dumps(event_summary, indent=2, default=str)}' if event_summary else ''}

Analyse the manager's tactical decisions and generate the MDP following the output format exactly."""

        try:
            response = await self.claude_client.messages.create(
                model=settings.AI_COACHING_MODEL,
                max_tokens=3000,
                system=MDP_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text
            return self._parse_mdp(text)
        except Exception as e:
            logger.error(f"Claude MDP generation failed: {e}")
            return self._rule_based_mdp(tactical_data, formation_data)

    # ── Tactical Narrative ───────────────────────────────────────────────

    async def generate_tactical_narrative(
        self,
        intelligence_data: Dict,
    ) -> str:
        """Generate an elite tactical analysis narrative from intelligence data."""
        if not self.claude_client:
            return self._rule_based_narrative(intelligence_data)

        prompt = f"""Write an elite tactical analysis narrative for this match.

TACTICAL INTELLIGENCE DATA:
{json.dumps(intelligence_data, indent=2, default=str)}

Write a 200-300 word professional tactical analysis covering:
1. Overall tactical picture
2. Key tactical moments and decisions
3. What worked and what didn't
4. Recommendations for next match

Write in the style of a Premier League analyst. Be specific with data references."""

        try:
            response = await self.claude_client.messages.create(
                model=settings.AI_COACHING_MODEL,
                max_tokens=settings.AI_COACHING_MAX_TOKENS,
                system=COACHING_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"Claude tactical narrative failed: {e}")
            return self._rule_based_narrative(intelligence_data)

    # ─── Helpers ─────────────────────────────────────────────────────────

    def _format_metrics_for_prompt(self, m: ClipMetrics, benchmarks: Dict) -> str:
        lines = []
        if m.touches > 0:
            lines.append(f"- Touches: {m.touches}")
        if m.passes_attempted > 0:
            bm = benchmarks.get('pass_accuracy', 0)
            lines.append(f"- Passes: {m.passes_completed}/{m.passes_attempted} ({m.pass_accuracy:.0f}%)"
                         + (f" [benchmark: {bm}%]" if bm else ""))
        if m.shots > 0:
            lines.append(f"- Shots: {m.shots} ({m.shots_on_target} on target), xG: {m.xg_total:.2f}")
        if m.tackles_attempted > 0:
            lines.append(f"- Tackles: {m.tackles_won}/{m.tackles_attempted}")
        if m.dribbles_attempted > 0:
            pct = m.dribbles_successful / m.dribbles_attempted * 100 if m.dribbles_attempted else 0
            lines.append(f"- Dribbles: {m.dribbles_successful}/{m.dribbles_attempted} ({pct:.0f}%)")
        if m.first_touch_quality > 0:
            lines.append(f"- First touch quality: {m.first_touch_quality:.1f}px displacement")
        if m.decision_speed_s > 0:
            bm = benchmarks.get('decision_speed_s', 0)
            lines.append(f"- Decision speed: {m.decision_speed_s:.2f}s"
                         + (f" [benchmark: {bm}s]" if bm else ""))
        if m.progressive_carries > 0:
            lines.append(f"- Progressive carries: {m.progressive_carries}")
        if m.line_breaking_passes > 0:
            lines.append(f"- Line-breaking passes: {m.line_breaking_passes}")
        if m.distance_covered_m > 0:
            lines.append(f"- Distance covered: {m.distance_covered_m:.0f}m")
        if m.off_ball_distance_m > 0:
            lines.append(f"- Off-ball movement: {m.off_ball_distance_m:.0f}m")
        return '\n'.join(lines) if lines else "No significant metrics detected"

    def _aggregate_clip_metrics(self, clips: List[ClipMetrics]) -> Dict:
        """Aggregate metrics across multiple clips."""
        agg = {
            "total_clips": len(clips),
            "total_duration_s": sum(c.duration_seconds for c in clips),
            "touches": sum(c.touches for c in clips),
            "passes_attempted": sum(c.passes_attempted for c in clips),
            "passes_completed": sum(c.passes_completed for c in clips),
            "shots": sum(c.shots for c in clips),
            "shots_on_target": sum(c.shots_on_target for c in clips),
            "xg_total": sum(c.xg_total for c in clips),
            "tackles_attempted": sum(c.tackles_attempted for c in clips),
            "tackles_won": sum(c.tackles_won for c in clips),
            "interceptions": sum(c.interceptions for c in clips),
            "dribbles_attempted": sum(c.dribbles_attempted for c in clips),
            "dribbles_successful": sum(c.dribbles_successful for c in clips),
            "headers": sum(c.headers for c in clips),
            "progressive_carries": sum(c.progressive_carries for c in clips),
            "line_breaking_passes": sum(c.line_breaking_passes for c in clips),
            "distance_m": sum(c.distance_covered_m for c in clips),
            "sprints": sum(c.sprints for c in clips),
        }
        # Averages
        ft = [c.first_touch_quality for c in clips if c.first_touch_quality > 0]
        agg["avg_first_touch_quality"] = round(float(np.mean(ft)), 1) if ft else 0
        ds = [c.decision_speed_s for c in clips if c.decision_speed_s > 0]
        agg["avg_decision_speed_s"] = round(float(np.mean(ds)), 2) if ds else 0
        if agg["passes_attempted"] > 0:
            agg["pass_accuracy"] = round(agg["passes_completed"] / agg["passes_attempted"] * 100, 1)
        else:
            agg["pass_accuracy"] = 0
        return agg

    def _format_aggregated_metrics(self, agg: Dict, benchmarks: Dict, position: str) -> str:
        lines = [
            f"Total clips: {agg['total_clips']} | Duration: {agg['total_duration_s'] / 60:.1f} min",
            f"Touches: {agg['touches']}",
            f"Pass accuracy: {agg['pass_accuracy']}% ({agg['passes_completed']}/{agg['passes_attempted']})"
            + (f" [benchmark: {benchmarks.get('pass_accuracy', '?')}%]" if 'pass_accuracy' in benchmarks else ""),
            f"Shots: {agg['shots']} ({agg['shots_on_target']} on target), xG: {agg['xg_total']:.2f}",
            f"Tackles: {agg['tackles_won']}/{agg['tackles_attempted']}",
            f"Dribbles: {agg['dribbles_successful']}/{agg['dribbles_attempted']}",
            f"Interceptions: {agg['interceptions']}, Headers: {agg['headers']}",
            f"Progressive carries: {agg['progressive_carries']}",
            f"Line-breaking passes: {agg['line_breaking_passes']}",
            f"Distance: {agg['distance_m']:.0f}m, Sprints: {agg['sprints']}",
            f"Avg first touch quality: {agg['avg_first_touch_quality']}px",
            f"Avg decision speed: {agg['avg_decision_speed_s']}s"
            + (f" [benchmark: {benchmarks.get('decision_speed_s', '?')}s]" if 'decision_speed_s' in benchmarks else ""),
        ]
        return '\n'.join(lines)

    # ─── Parsers ─────────────────────────────────────────────────────────

    def _parse_clip_feedback(self, text: str, metrics: ClipMetrics) -> ClipFeedback:
        """Parse Claude's clip feedback response."""
        feedback = ClipFeedback(
            clip_id=metrics.clip_id,
            event_type=metrics.events[0].get('event_type', '') if metrics.events else '',
            raw_text=text,
        )
        # Extract rating
        for line in text.split('\n'):
            lower = line.lower().strip()
            if 'rating' in lower or 'overall' in lower:
                for word in lower.replace('/', ' ').split():
                    try:
                        val = float(word)
                        if 1 <= val <= 10:
                            feedback.overall_rating = val
                            break
                    except ValueError:
                        continue

            if lower.startswith(('- ', '• ', '* ')):
                content = line.strip().lstrip('-•* ').strip()
                if any(kw in lower for kw in ['strength', 'positive', 'good', 'excellent']):
                    feedback.strengths.append(content)
                elif any(kw in lower for kw in ['improve', 'weak', 'work on', 'develop']):
                    feedback.weaknesses.append(content)
                elif any(kw in lower for kw in ['drill', 'exercise', 'practice']):
                    feedback.drill_recommendations.append(content)
                else:
                    feedback.coaching_points.append(content)

        return feedback

    def _parse_idp(
        self, text: str, jersey: int, name: str, position: str, team: str
    ) -> PlayerDevelopmentPlan:
        """Parse Claude's IDP response into structured plan."""
        from datetime import datetime
        plan = PlayerDevelopmentPlan(
            jersey_number=jersey,
            player_name=name,
            position=position,
            team=team,
            raw_text=text,
            generated_at=datetime.now().isoformat(),
        )

        lines = text.split('\n')
        section = ""
        for line in lines:
            stripped = line.strip()
            lower = stripped.lower()

            # Overall rating
            if 'overall rating' in lower:
                for word in lower.replace('/', ' ').split():
                    try:
                        val = float(word)
                        if 1 <= val <= 10:
                            plan.overall_rating = val
                            break
                    except ValueError:
                        continue

            # Four corner
            if 'technical:' in lower:
                plan.four_corner.technical = self._extract_score(lower)
            elif 'tactical:' in lower:
                plan.four_corner.tactical = self._extract_score(lower)
            elif 'physical:' in lower:
                plan.four_corner.physical = self._extract_score(lower)
            elif 'psychological:' in lower:
                plan.four_corner.psychological = self._extract_score(lower)

            # Section detection
            if 'key strength' in lower:
                section = "strengths"
                continue
            elif 'development priorit' in lower:
                section = "priorities"
                continue
            elif 'weekly focus' in lower:
                plan.weekly_focus = stripped.split(':', 1)[-1].strip() if ':' in stripped else ""
                section = ""
                continue
            elif 'session plan' in lower:
                plan.session_plan = stripped.split(':', 1)[-1].strip() if ':' in stripped else ""
                section = ""
                continue
            elif '3-month' in lower or 'three-month' in lower:
                section = "3month"
                continue
            elif '6-month' in lower or 'six-month' in lower:
                section = "6month"
                continue

            # Content
            if stripped.startswith(('1.', '2.', '3.', '4.', '5.', '-', '•')):
                content = stripped.lstrip('0123456789.-•* ').strip()
                if section == "strengths" and content:
                    plan.key_strengths.append(content)
                elif section == "priorities" and content:
                    level = "MEDIUM"
                    if '[HIGH]' in stripped.upper():
                        level = "HIGH"
                    elif '[LOW]' in stripped.upper():
                        level = "LOW"
                    plan.development_priorities.append(DevelopmentPriority(
                        priority_level=level,
                        area=content,
                    ))
                elif section == "3month" and content:
                    plan.three_month_goals.append(content)
                elif section == "6month" and content:
                    plan.six_month_goals.append(content)

        return plan

    def _parse_mdp(self, text: str) -> ManagerDevelopmentPlan:
        """Parse Claude's MDP response."""
        from datetime import datetime
        mdp = ManagerDevelopmentPlan(
            raw_text=text,
            generated_at=datetime.now().isoformat(),
        )

        lines = text.split('\n')
        section = ""
        for line in lines:
            stripped = line.strip()
            lower = stripped.lower()

            if 'formation management' in lower:
                mdp.formation_management_score = self._extract_score(lower)
            elif 'pressing strategy' in lower:
                mdp.pressing_strategy_score = self._extract_score(lower)
            elif 'transition' in lower and '/10' in lower:
                mdp.transition_score = self._extract_score(lower)

            if 'tactical strength' in lower:
                section = "strengths"
                continue
            elif 'tactical weakness' in lower:
                section = "weaknesses"
                continue
            elif 'development priorit' in lower:
                section = "priorities"
                continue
            elif 'coaching education' in lower:
                section = "education"
                continue

            if stripped.startswith(('1.', '2.', '3.', '-', '•')):
                content = stripped.lstrip('0123456789.-•* ').strip()
                if section == "strengths" and content:
                    mdp.tactical_strengths.append(content)
                elif section == "weaknesses" and content:
                    mdp.tactical_weaknesses.append(content)
                elif section == "priorities" and content:
                    level = "MEDIUM"
                    if '[HIGH]' in stripped.upper():
                        level = "HIGH"
                    elif '[LOW]' in stripped.upper():
                        level = "LOW"
                    mdp.development_priorities.append(DevelopmentPriority(
                        priority_level=level, area=content
                    ))
                elif section == "education" and content:
                    mdp.coaching_education.append(content)

        return mdp

    def _extract_score(self, text: str) -> float:
        """Extract a numeric score from text like 'Technical: 7.5/10'."""
        for word in text.replace('/', ' ').split():
            try:
                val = float(word)
                if 0 <= val <= 10:
                    return val
            except ValueError:
                continue
        return 0.0

    # ─── Rule-Based Fallbacks ────────────────────────────────────────────

    def _rule_based_clip_feedback(self, m: ClipMetrics) -> ClipFeedback:
        """Generate feedback without AI."""
        strengths, weaknesses, coaching = [], [], []

        if m.pass_accuracy >= 80:
            strengths.append(f"Good pass accuracy ({m.pass_accuracy:.0f}%)")
        elif m.passes_attempted > 0:
            weaknesses.append(f"Pass accuracy needs work ({m.pass_accuracy:.0f}%)")
            coaching.append("Focus on weight of pass in rondo exercises")

        if m.progressive_carries > 0:
            strengths.append(f"{m.progressive_carries} progressive carry(s) — good ball advancement")

        if m.decision_speed_s > 0.8:
            weaknesses.append(f"Decision speed {m.decision_speed_s:.2f}s — aim for <0.6s")
            coaching.append("One-touch rondo to improve decision speed under pressure")

        rating = 5.0
        if len(strengths) > len(weaknesses):
            rating = 7.0
        elif len(weaknesses) > len(strengths):
            rating = 4.5

        return ClipFeedback(
            clip_id=m.clip_id,
            overall_rating=rating,
            strengths=strengths or ["Clip analysed — limited data for detailed assessment"],
            weaknesses=weaknesses,
            coaching_points=coaching,
            drill_recommendations=["Rondo 4v2 for passing under pressure"],
        )

    def _rule_based_idp(
        self, jersey: int, name: str, position: str, clips: List[ClipMetrics]
    ) -> PlayerDevelopmentPlan:
        """Generate IDP without AI."""
        from datetime import datetime
        agg = self._aggregate_clip_metrics(clips)

        plan = PlayerDevelopmentPlan(
            jersey_number=jersey,
            player_name=name,
            position=position,
            overall_rating=6.0,
            four_corner=FourCornerRating(
                technical=6.5, tactical=6.0, physical=6.5, psychological=6.5
            ),
            key_strengths=["Data-driven assessment pending more clip data"],
            weekly_focus="General technical development",
            session_plan="15min rondo → 20min positional play → 15min small-sided game",
            three_month_goals=["Improve pass accuracy above position benchmark"],
            six_month_goals=["Consistent performance across all four corners"],
            generated_at=datetime.now().isoformat(),
        )

        if agg["pass_accuracy"] < 80:
            plan.development_priorities.append(DevelopmentPriority(
                priority_level="HIGH",
                area="Pass Accuracy",
                metric_current=f"{agg['pass_accuracy']:.0f}%",
                drill="Rondo 4v2 with directional constraint",
            ))

        return plan

    def _rule_based_mdp(self, tactical: Dict, formation: Dict) -> ManagerDevelopmentPlan:
        """Generate MDP without AI."""
        from datetime import datetime
        return ManagerDevelopmentPlan(
            formation_management_score=6.0,
            pressing_strategy_score=6.0,
            transition_score=6.0,
            tactical_strengths=["Assessment requires AI — rule-based fallback active"],
            tactical_weaknesses=["Connect Claude API for full tactical analysis"],
            coaching_education=["Review transition principles in 6v6 format"],
            generated_at=datetime.now().isoformat(),
        )

    def _rule_based_narrative(self, data: Dict) -> str:
        """Generate narrative without AI."""
        events = data.get('event_counts', {})
        total = data.get('total_events', 0)
        return (
            f"Match produced {total} tactical events. "
            f"Key patterns: {', '.join(f'{k}: {v}' for k, v in list(events.items())[:5])}. "
            "Connect Claude API for elite-level tactical narrative."
        )


# Global instance
ai_coaching_engine = AICoachingEngine()
