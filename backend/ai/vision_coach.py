"""
AI Vision Coach

The core differentiator from VEO/HUDL - uses Vision AI (Claude, GPT-4V, etc.)
to actually "watch" player clips and provide intelligent coaching feedback.

This module:
1. Extracts key frames from player action clips
2. Sends frames to Vision AI for technique analysis
3. Maps AI observations to our technical framework
4. Generates personalized, actionable coaching feedback
"""
import asyncio
import base64
import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
import json
import os
import httpx

from ai.technical_analysis import (
    TECHNIQUE_LIBRARY,
    SkillTechnique,
    TechniqueCheckpoint,
    get_technique_for_action,
    get_position_requirements,
    get_coaching_points_for_mistake,
    get_drills_for_weakness,
    SkillCategory
)


@dataclass
class FrameAnalysis:
    """Analysis of a single frame."""
    frame_number: int
    timestamp_ms: int
    observations: List[str]
    technique_scores: Dict[str, float]  # checkpoint_name -> score (0-1)
    issues_detected: List[str]
    frame_image_b64: Optional[str] = None


@dataclass
class ActionAnalysis:
    """Complete analysis of a player action."""
    action_id: str
    action_type: str
    player_jersey: int
    player_name: Optional[str]

    # Frame data
    frame_analyses: List[FrameAnalysis]
    key_frames: List[int]  # Most important frame numbers

    # Technical assessment
    technique_scores: Dict[str, float]  # checkpoint_name -> score
    overall_score: float
    strengths: List[str]
    weaknesses: List[str]

    # AI coaching
    ai_observations: str
    coaching_feedback: List[str]
    specific_corrections: List[str]
    recommended_drills: List[str]

    # Metadata
    video_path: str
    analyzed_at: str
    confidence: float


@dataclass
class PlayerTechnicalProfile:
    """Accumulated technical profile for a player over multiple actions."""
    jersey_number: int
    player_name: Optional[str]
    position: Optional[str]

    # Aggregated scores per skill category
    skill_scores: Dict[str, List[float]] = field(default_factory=dict)

    # Identified patterns
    consistent_strengths: List[str] = field(default_factory=list)
    recurring_weaknesses: List[str] = field(default_factory=list)

    # All analyses
    action_analyses: List[ActionAnalysis] = field(default_factory=list)

    # Training recommendations
    priority_improvements: List[str] = field(default_factory=list)
    recommended_drills: List[str] = field(default_factory=list)


class VisionCoachService:
    """
    AI-powered vision coach that analyzes player technique from video clips.

    This is the unique value proposition - turning video into coaching intelligence.
    """

    # Number of frames to extract per action for analysis
    FRAMES_PER_ACTION = 5

    # Vision AI providers
    PROVIDER_CLAUDE = "claude"
    PROVIDER_OPENAI = "openai"

    def __init__(self):
        self.api_key: Optional[str] = None
        self.provider: str = self.PROVIDER_CLAUDE
        self.model: str = "claude-sonnet-4-20250514"
        self.client: Optional[httpx.AsyncClient] = None

        # Player profiles accumulated over time
        self.player_profiles: Dict[int, PlayerTechnicalProfile] = {}

        # Analysis history
        self.analyses: List[ActionAnalysis] = []

    async def initialize(
        self,
        api_key: str,
        provider: str = "claude",
        model: Optional[str] = None
    ):
        """
        Initialize the Vision Coach with API credentials.

        Args:
            api_key: API key for the vision AI provider
            provider: "claude" or "openai"
            model: Specific model to use (optional)
        """
        self.api_key = api_key
        self.provider = provider

        if model:
            self.model = model
        elif provider == self.PROVIDER_CLAUDE:
            self.model = "claude-sonnet-4-20250514"
        elif provider == self.PROVIDER_OPENAI:
            self.model = "gpt-4o"

        # Initialize HTTP client
        if provider == self.PROVIDER_CLAUDE:
            self.client = httpx.AsyncClient(
                base_url="https://api.anthropic.com",
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json"
                },
                timeout=60.0
            )
        elif provider == self.PROVIDER_OPENAI:
            self.client = httpx.AsyncClient(
                base_url="https://api.openai.com/v1",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "content-type": "application/json"
                },
                timeout=60.0
            )

        return True

    async def analyze_action_clip(
        self,
        video_path: str,
        start_frame: int,
        end_frame: int,
        action_type: str,
        player_jersey: int,
        player_name: Optional[str] = None,
        player_position: Optional[str] = None,
        additional_context: Optional[str] = None
    ) -> ActionAnalysis:
        """
        Analyze a specific action clip using Vision AI.

        This is the core method that turns video into coaching intelligence.

        Args:
            video_path: Path to the video file
            start_frame: First frame of the action
            end_frame: Last frame of the action
            action_type: Type of action (pass, shot, tackle, etc.)
            player_jersey: Player's jersey number
            player_name: Player's name (optional)
            player_position: Player's position (optional)
            additional_context: Team/player context for personalized coaching (optional)

        Returns:
            Complete analysis of the action with coaching feedback
        """
        # Extract key frames from the clip
        frames = self._extract_key_frames(video_path, start_frame, end_frame)

        if not frames:
            raise ValueError(f"Could not extract frames from {video_path}")

        # Get the technique framework for this action
        technique = get_technique_for_action(action_type)

        # Build the analysis prompt
        prompt = self._build_analysis_prompt(
            action_type=action_type,
            technique=technique,
            player_position=player_position,
            additional_context=additional_context
        )

        # Send to Vision AI
        ai_response = await self._call_vision_ai(frames, prompt)

        # Parse AI response into structured analysis
        analysis = self._parse_ai_response(
            ai_response=ai_response,
            action_type=action_type,
            player_jersey=player_jersey,
            player_name=player_name,
            technique=technique,
            video_path=video_path,
            frames=frames
        )

        # Store analysis
        self.analyses.append(analysis)

        # Update player profile
        self._update_player_profile(analysis, player_position)

        return analysis

    def _extract_key_frames(
        self,
        video_path: str,
        start_frame: int,
        end_frame: int
    ) -> List[Tuple[int, np.ndarray]]:
        """Extract key frames from a video clip."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []

        total_frames = end_frame - start_frame
        if total_frames <= 0:
            return []

        # Calculate which frames to extract (evenly spaced)
        frame_indices = np.linspace(
            start_frame,
            end_frame,
            min(self.FRAMES_PER_ACTION, total_frames),
            dtype=int
        )

        frames = []
        fps = cap.get(cv2.CAP_PROP_FPS)

        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                timestamp_ms = int(frame_idx / fps * 1000) if fps > 0 else 0
                frames.append((frame_idx, timestamp_ms, frame))

        cap.release()
        return frames

    def _build_analysis_prompt(
        self,
        action_type: str,
        technique: Optional[SkillTechnique],
        player_position: Optional[str],
        additional_context: Optional[str] = None
    ) -> str:
        """
        Build the prompt for Vision AI analysis.

        Uses UEFA coaching methodology and FA Four Corner Model to provide
        professional-level coaching feedback that matches UEFA Pro/A/B standards.

        Now enhanced with team-specific context for personalized coaching.
        """

        base_prompt = f"""You are an elite UEFA Pro Licensed football coach analyzing video frames of a player performing a {action_type}.

Your analysis should meet UEFA coaching standards, using the FA Four Corner Model:
- TECHNICAL: Ball mastery, technique execution
- TACTICAL: Decision making, awareness, spatial understanding
- PHYSICAL: Balance, coordination, speed of execution
- PSYCHOLOGICAL: Composure, confidence, concentration

Analyze these sequential frames and provide professional coaching feedback.

"""

        if technique:
            # Add UEFA-aligned technique guidance
            base_prompt += f"""
## Technical Framework: {technique.skill_name}
{technique.description}

### UEFA Key Factors for this skill:
"""
            if hasattr(technique, 'uefa_key_factors') and technique.uefa_key_factors:
                for factor in technique.uefa_key_factors:
                    base_prompt += f"- {factor}\n"

            base_prompt += """
### Checkpoints to Evaluate:
"""
            for checkpoint in technique.checkpoints:
                domain = checkpoint.four_corner_domain.value if hasattr(checkpoint, 'four_corner_domain') else "technical"
                base_prompt += f"""
**{checkpoint.name}** [{domain.upper()}]
- Look for: {', '.join(checkpoint.what_to_look_for[:4])}
- Common faults: {', '.join(checkpoint.common_mistakes[:3])}
- Coaching cues: {', '.join(checkpoint.coaching_cues[:3])}
"""

            # Add Four Corner Model elements if available
            if hasattr(technique, 'technical_elements') and technique.technical_elements:
                base_prompt += f"""
### Four Corner Model Assessment:
- Technical elements: {', '.join(technique.technical_elements)}
- Tactical elements: {', '.join(technique.tactical_elements) if hasattr(technique, 'tactical_elements') else 'N/A'}
- Physical elements: {', '.join(technique.physical_elements) if hasattr(technique, 'physical_elements') else 'N/A'}
- Psychological elements: {', '.join(technique.psychological_elements) if hasattr(technique, 'psychological_elements') else 'N/A'}
"""

        if player_position:
            pos_req = get_position_requirements(player_position)
            if pos_req:
                base_prompt += f"""
## Position Context: {pos_req.position}
- Primary skills required: {', '.join(pos_req.primary_skills)}
- Key attributes: {', '.join(pos_req.key_attributes)}
- Common weaknesses to watch for: {', '.join(pos_req.common_weaknesses)}
"""

        base_prompt += """
## Your Analysis Task (UEFA Standard Assessment)

Analyze like a UEFA Pro Licensed coach observing a training session:

### 1. FRAME-BY-FRAME OBSERVATIONS
For each frame, describe precisely what you observe:
- Body position and shape
- Foot placement and angle
- Ball contact point (if applicable)
- Balance and weight distribution
- Head position and awareness indicators

### 2. FOUR CORNER ASSESSMENT (Score 1-10 each)

**TECHNICAL:**
- Execution Quality: X/10
- Surface Selection: X/10
- Contact Point: X/10

**TACTICAL:**
- Decision Making: X/10
- Awareness/Scanning: X/10
- Option Selection: X/10

**PHYSICAL:**
- Body Position: X/10
- Balance: X/10
- Coordination: X/10

**PSYCHOLOGICAL:**
- Composure: X/10
- Confidence (body language): X/10

### 3. STRENGTHS IDENTIFIED
What is the player executing well? Be specific about technique.

### 4. AREAS FOR DEVELOPMENT
What technical faults do you observe? Use coaching terminology:
- Describe the fault precisely
- Explain the consequence of the fault
- Reference which frame(s) show this issue

### 5. COACHING INTERVENTIONS (UEFA Standard)
Provide 2-3 specific coaching points using proper methodology:
- Use positive instruction ("Do this" not "Don't do that")
- Be precise and actionable
- Use coaching cues that players can remember
- Example: "Plant your standing foot beside the ball, pointing at your target, to improve accuracy"

### 6. TRAINING RECOMMENDATIONS
Suggest specific drills or exercises to address identified weaknesses:
- Name of drill
- How it addresses the weakness
- Key coaching points within the drill

### 7. DEVELOPMENT PRIORITY
If you could only fix ONE thing, what should this player focus on first and why?

Be specific, professional, and constructive. Your feedback should be immediately actionable.
"""

        # Add team-specific context if provided (THE SECRET WEAPON)
        if additional_context:
            base_prompt += f"""

## TEAM & PLAYER CONTEXT (PERSONALIZED COACHING)

{additional_context}

CRITICAL: Incorporate the above team/player context into your analysis:
- Reference the team's playing style when evaluating decisions
- Consider the player's development focus when prioritizing feedback
- Align coaching points with the team's philosophy
- Adapt language based on player's age group if specified
- Reference team's priority principles when relevant
"""

        return base_prompt

    async def _call_vision_ai(
        self,
        frames: List[Tuple[int, int, np.ndarray]],
        prompt: str
    ) -> str:
        """Call the Vision AI API with frames and prompt."""

        if not self.client:
            raise ValueError("Vision AI not initialized. Call initialize() first.")

        # Encode frames to base64
        encoded_frames = []
        for frame_idx, timestamp_ms, frame in frames:
            # Resize for API (reduce size/cost)
            resized = cv2.resize(frame, (640, 360))
            _, buffer = cv2.imencode('.jpg', resized, [cv2.IMWRITE_JPEG_QUALITY, 85])
            b64 = base64.b64encode(buffer).decode('utf-8')
            encoded_frames.append({
                "frame_idx": frame_idx,
                "timestamp_ms": timestamp_ms,
                "image_b64": b64
            })

        if self.provider == self.PROVIDER_CLAUDE:
            return await self._call_claude_vision(encoded_frames, prompt)
        elif self.provider == self.PROVIDER_OPENAI:
            return await self._call_openai_vision(encoded_frames, prompt)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    async def _call_claude_vision(
        self,
        encoded_frames: List[Dict],
        prompt: str
    ) -> str:
        """Call Claude Vision API."""

        # Build content with images
        content = []

        for i, frame_data in enumerate(encoded_frames):
            content.append({
                "type": "text",
                "text": f"Frame {i+1} (timestamp: {frame_data['timestamp_ms']}ms):"
            })
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": frame_data['image_b64']
                }
            })

        content.append({
            "type": "text",
            "text": prompt
        })

        response = await self.client.post(
            "/v1/messages",
            json={
                "model": self.model,
                "max_tokens": 2000,
                "messages": [
                    {"role": "user", "content": content}
                ]
            }
        )

        if response.status_code != 200:
            raise Exception(f"Claude API error: {response.text}")

        result = response.json()
        return result["content"][0]["text"]

    async def _call_openai_vision(
        self,
        encoded_frames: List[Dict],
        prompt: str
    ) -> str:
        """Call OpenAI Vision API."""

        # Build content with images
        content = [{"type": "text", "text": prompt}]

        for i, frame_data in enumerate(encoded_frames):
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{frame_data['image_b64']}",
                    "detail": "high"
                }
            })

        response = await self.client.post(
            "/chat/completions",
            json={
                "model": self.model,
                "max_tokens": 2000,
                "messages": [
                    {"role": "user", "content": content}
                ]
            }
        )

        if response.status_code != 200:
            raise Exception(f"OpenAI API error: {response.text}")

        result = response.json()
        return result["choices"][0]["message"]["content"]

    def _parse_ai_response(
        self,
        ai_response: str,
        action_type: str,
        player_jersey: int,
        player_name: Optional[str],
        technique: Optional[SkillTechnique],
        video_path: str,
        frames: List[Tuple[int, int, np.ndarray]]
    ) -> ActionAnalysis:
        """Parse the AI response into structured analysis."""

        # Extract scores from response
        technique_scores = self._extract_scores(ai_response)

        # Extract lists from response
        strengths = self._extract_section(ai_response, "STRENGTHS")
        weaknesses = self._extract_section(ai_response, "AREAS FOR IMPROVEMENT")
        corrections = self._extract_section(ai_response, "COACHING CORRECTIONS")
        drills = self._extract_section(ai_response, "DRILL RECOMMENDATIONS")

        # Calculate overall score
        if technique_scores:
            overall_score = sum(technique_scores.values()) / len(technique_scores) / 10
        else:
            overall_score = 0.5

        # Add drills from our framework if technique exists
        if technique and not drills:
            drills = technique.drills_to_improve[:3]

        analysis = ActionAnalysis(
            action_id=f"{player_jersey}_{action_type}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            action_type=action_type,
            player_jersey=player_jersey,
            player_name=player_name,
            frame_analyses=[],
            key_frames=[f[0] for f in frames],
            technique_scores=technique_scores,
            overall_score=overall_score,
            strengths=strengths,
            weaknesses=weaknesses,
            ai_observations=ai_response,
            coaching_feedback=corrections,
            specific_corrections=corrections,
            recommended_drills=drills,
            video_path=video_path,
            analyzed_at=datetime.now().isoformat(),
            confidence=0.8 if technique_scores else 0.5
        )

        return analysis

    def _extract_scores(self, response: str) -> Dict[str, float]:
        """Extract technique scores from AI response."""
        scores = {}

        # Look for patterns like "Body Position: 7/10" or "Timing: 8/10"
        import re
        pattern = r'(\w+(?:\s+\w+)?)\s*:\s*(\d+)\s*/\s*10'
        matches = re.findall(pattern, response, re.IGNORECASE)

        for name, score in matches:
            scores[name.strip().lower()] = float(score)

        return scores

    def _extract_section(self, response: str, section_name: str) -> List[str]:
        """Extract bullet points from a section of the response."""
        import re

        # Find the section
        pattern = rf'\*?\*?{section_name}\*?\*?:?\s*(.*?)(?=\n\s*\*?\*?[A-Z]|\Z)'
        match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)

        if not match:
            return []

        section_text = match.group(1)

        # Extract bullet points
        items = []
        for line in section_text.split('\n'):
            line = line.strip()
            # Remove bullet markers
            line = re.sub(r'^[-*•]\s*', '', line)
            line = re.sub(r'^\d+\.\s*', '', line)
            if line and len(line) > 10:  # Minimum length for meaningful content
                items.append(line)

        return items[:5]  # Limit to 5 items

    def _update_player_profile(
        self,
        analysis: ActionAnalysis,
        player_position: Optional[str]
    ):
        """Update the player's technical profile with new analysis."""

        jersey = analysis.player_jersey

        if jersey not in self.player_profiles:
            self.player_profiles[jersey] = PlayerTechnicalProfile(
                jersey_number=jersey,
                player_name=analysis.player_name,
                position=player_position
            )

        profile = self.player_profiles[jersey]
        profile.action_analyses.append(analysis)

        # Update skill scores
        category = analysis.action_type
        if category not in profile.skill_scores:
            profile.skill_scores[category] = []
        profile.skill_scores[category].append(analysis.overall_score)

        # Update patterns after enough data
        if len(profile.action_analyses) >= 3:
            self._analyze_patterns(profile)

    def _analyze_patterns(self, profile: PlayerTechnicalProfile):
        """Analyze patterns across multiple actions to identify consistent issues."""

        all_strengths = []
        all_weaknesses = []

        for analysis in profile.action_analyses:
            all_strengths.extend(analysis.strengths)
            all_weaknesses.extend(analysis.weaknesses)

        # Find recurring items (mentioned in >50% of analyses)
        threshold = len(profile.action_analyses) / 2

        # Count occurrences (simplified - could use NLP for better matching)
        from collections import Counter
        strength_counts = Counter(all_strengths)
        weakness_counts = Counter(all_weaknesses)

        profile.consistent_strengths = [
            s for s, count in strength_counts.items() if count >= threshold
        ][:5]

        profile.recurring_weaknesses = [
            w for w, count in weakness_counts.items() if count >= threshold
        ][:5]

        # Generate priority improvements
        profile.priority_improvements = profile.recurring_weaknesses[:3]

        # Get drills for weaknesses
        all_drills = []
        for analysis in profile.action_analyses:
            all_drills.extend(analysis.recommended_drills)
        profile.recommended_drills = list(set(all_drills))[:5]

    async def generate_player_report(
        self,
        jersey_number: int
    ) -> Optional[Dict]:
        """Generate a comprehensive technical report for a player."""

        if jersey_number not in self.player_profiles:
            return None

        profile = self.player_profiles[jersey_number]

        # Calculate average scores per skill
        avg_scores = {}
        for skill, scores in profile.skill_scores.items():
            avg_scores[skill] = sum(scores) / len(scores) if scores else 0

        return {
            "player": {
                "jersey_number": profile.jersey_number,
                "name": profile.player_name,
                "position": profile.position
            },
            "summary": {
                "actions_analyzed": len(profile.action_analyses),
                "overall_technical_score": sum(avg_scores.values()) / len(avg_scores) if avg_scores else 0,
                "skill_scores": avg_scores
            },
            "strengths": {
                "consistent": profile.consistent_strengths,
                "description": "These are technical aspects the player consistently executes well."
            },
            "areas_for_improvement": {
                "recurring_issues": profile.recurring_weaknesses,
                "priority_focus": profile.priority_improvements,
                "description": "These issues appear repeatedly and should be prioritized in training."
            },
            "training_recommendations": {
                "recommended_drills": profile.recommended_drills,
                "focus_areas": profile.priority_improvements
            },
            "detailed_analyses": [
                {
                    "action_type": a.action_type,
                    "score": a.overall_score,
                    "feedback": a.coaching_feedback[:2]
                }
                for a in profile.action_analyses[-10:]  # Last 10 analyses
            ],
            "generated_at": datetime.now().isoformat()
        }

    def get_coaching_summary(self, jersey_number: int) -> Optional[str]:
        """Get a human-readable coaching summary for a player."""

        if jersey_number not in self.player_profiles:
            return None

        profile = self.player_profiles[jersey_number]

        summary = f"""
## Technical Assessment: Player #{profile.jersey_number}
{"(" + profile.player_name + ")" if profile.player_name else ""}
{"Position: " + profile.position if profile.position else ""}

### Actions Analyzed: {len(profile.action_analyses)}

### Key Strengths
{chr(10).join("✓ " + s for s in profile.consistent_strengths) if profile.consistent_strengths else "More data needed"}

### Priority Improvements
{chr(10).join("→ " + w for w in profile.priority_improvements) if profile.priority_improvements else "More data needed"}

### Recommended Training Focus
{chr(10).join("• " + d for d in profile.recommended_drills[:3]) if profile.recommended_drills else "Continue current training"}

### Recent Feedback
"""
        for analysis in profile.action_analyses[-3:]:
            summary += f"\n**{analysis.action_type.title()}** (Score: {analysis.overall_score:.1f}/1.0)\n"
            for feedback in analysis.coaching_feedback[:2]:
                summary += f"  - {feedback}\n"

        return summary

    def export_analyses(self, output_path: str) -> str:
        """Export all analyses to JSON."""

        data = {
            "exported_at": datetime.now().isoformat(),
            "total_analyses": len(self.analyses),
            "players": {},
            "analyses": []
        }

        # Add player profiles
        for jersey, profile in self.player_profiles.items():
            data["players"][str(jersey)] = {
                "jersey_number": profile.jersey_number,
                "name": profile.player_name,
                "position": profile.position,
                "analyses_count": len(profile.action_analyses),
                "strengths": profile.consistent_strengths,
                "weaknesses": profile.recurring_weaknesses,
                "recommended_drills": profile.recommended_drills
            }

        # Add individual analyses (without images)
        for analysis in self.analyses:
            data["analyses"].append({
                "action_id": analysis.action_id,
                "action_type": analysis.action_type,
                "player_jersey": analysis.player_jersey,
                "overall_score": analysis.overall_score,
                "technique_scores": analysis.technique_scores,
                "strengths": analysis.strengths,
                "weaknesses": analysis.weaknesses,
                "coaching_feedback": analysis.coaching_feedback,
                "recommended_drills": analysis.recommended_drills,
                "analyzed_at": analysis.analyzed_at
            })

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

        return output_path


# Global instance
vision_coach = VisionCoachService()
