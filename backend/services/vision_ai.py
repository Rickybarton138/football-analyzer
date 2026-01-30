"""
Vision AI Service - Gemini Integration

Uses Google Gemini's vision capabilities to analyze football match footage
and provide expert coaching insights based on what it actually sees.
"""

import base64
import asyncio
import json
from pathlib import Path
from typing import List, Dict, Optional, Any
from datetime import datetime
import aiohttp

from config import settings


class VisionAIService:
    """
    Vision-based AI coaching using Google Gemini.

    Analyzes video frames to provide comprehensive tactical insights,
    similar to how ChatGPT analyzes uploaded videos.
    """

    # Football-specific analysis prompts
    TACTICAL_ANALYSIS_PROMPT = """You are an expert football (soccer) coach and analyst.
Analyze these frames from a football match and provide detailed coaching insights.

For each frame, identify and analyze:
1. **Player Positions**: Where are the players positioned? Is the formation clear?
2. **Team Shape**: Is the team compact or stretched? Defensive line height?
3. **Ball Position**: Where is the ball? Who has possession?
4. **Tactical Patterns**: Any pressing triggers, counter-attacks, overlapping runs?
5. **Space Usage**: Are there gaps being exploited? Overloaded areas?
6. **Movement Patterns**: Off-ball runs, defensive tracking, pressing coordination

Provide your analysis in this JSON format:
{
    "overall_assessment": "Brief overall tactical assessment",
    "formation_detected": "e.g., 4-4-2, 4-3-3, or 'unclear'",
    "possession_team": "home/away/contested",
    "tactical_phase": "attacking/defending/transition/set_piece",
    "key_observations": [
        {
            "category": "pressing/shape/space/movement/positioning",
            "observation": "What you observed",
            "recommendation": "Coaching point for improvement"
        }
    ],
    "strengths": ["List of positive aspects observed"],
    "areas_to_improve": ["List of areas needing work"],
    "coaching_points": [
        {
            "priority": "high/medium/low",
            "title": "Brief title",
            "message": "Detailed coaching message",
            "drill_suggestion": "Training drill to address this"
        }
    ],
    "half_time_talk": "What you'd say to the team at half-time based on this footage",
    "training_focus": ["Areas to focus on in next training session"]
}

Be specific and actionable. Reference actual positions and movements you see in the frames.
This is grassroots/amateur level football, so keep advice practical and achievable."""

    SPECIFIC_QUESTION_PROMPT = """You are an expert football coach analyzing match footage.
The user has asked: "{question}"

Analyze the provided frames and answer their question specifically.
Reference what you actually see in the frames.
Provide practical, actionable advice suitable for grassroots/amateur level.

Format your response as:
{{
    "answer": "Your detailed answer to the question",
    "observations": ["Specific things you noticed relevant to the question"],
    "recommendations": ["Actionable recommendations"],
    "related_coaching_points": ["Additional coaching points related to the question"]
}}"""

    def __init__(self):
        self.api_key: Optional[str] = None
        self.model = "gemini-2.0-flash"  # Fast and capable, good for video analysis
        self.api_url = "https://generativelanguage.googleapis.com/v1beta/models"
        self._session: Optional[aiohttp.ClientSession] = None

    async def initialize(self, api_key: Optional[str] = None):
        """Initialize the service with API key."""
        self.api_key = api_key or getattr(settings, 'GEMINI_API_KEY', None)

        if not self.api_key:
            print("[VISION AI] Warning: No Gemini API key configured")
            return False

        self._session = aiohttp.ClientSession()
        print("[VISION AI] Gemini service initialized")
        return True

    async def close(self):
        """Close the session."""
        if self._session:
            await self._session.close()
            self._session = None

    def _encode_image(self, image_path: str) -> Optional[str]:
        """Encode image to base64."""
        try:
            path = Path(image_path)
            if path.exists():
                with open(path, 'rb') as f:
                    return base64.b64encode(f.read()).decode('utf-8')
        except Exception as e:
            print(f"[VISION AI] Error encoding image {image_path}: {e}")
        return None

    def _encode_frame(self, frame) -> Optional[str]:
        """Encode numpy array frame to base64 JPEG."""
        try:
            import cv2
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            return base64.b64encode(buffer).decode('utf-8')
        except Exception as e:
            print(f"[VISION AI] Error encoding frame: {e}")
        return None

    async def analyze_frames(
        self,
        frames: List[Any],  # Can be file paths or numpy arrays
        custom_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze multiple frames using Gemini vision.

        Args:
            frames: List of frame paths or numpy arrays
            custom_prompt: Optional custom analysis prompt

        Returns:
            Analysis results from Gemini
        """
        if not self.api_key:
            return {
                "error": "Gemini API key not configured",
                "message": "Please add GEMINI_API_KEY to your .env file"
            }

        if not frames:
            return {"error": "No frames provided"}

        # Encode frames
        encoded_frames = []
        for frame in frames[:10]:  # Limit to 10 frames to stay within limits
            if isinstance(frame, str):
                encoded = self._encode_image(frame)
            else:
                encoded = self._encode_frame(frame)

            if encoded:
                encoded_frames.append(encoded)

        if not encoded_frames:
            return {"error": "Failed to encode any frames"}

        # Build the request
        prompt = custom_prompt or self.TACTICAL_ANALYSIS_PROMPT

        # Create content parts with images
        parts = []
        for i, encoded in enumerate(encoded_frames):
            parts.append({
                "inline_data": {
                    "mime_type": "image/jpeg",
                    "data": encoded
                }
            })
            parts.append({
                "text": f"Frame {i+1} of {len(encoded_frames)}"
            })

        parts.append({"text": prompt})

        # Make API request
        url = f"{self.api_url}/{self.model}:generateContent?key={self.api_key}"

        payload = {
            "contents": [{
                "parts": parts
            }],
            "generationConfig": {
                "temperature": 0.7,
                "maxOutputTokens": 4096,
                "responseMimeType": "application/json"
            }
        }

        max_retries = 3
        retry_delay = 5  # seconds

        for attempt in range(max_retries):
            try:
                async with self._session.post(url, json=payload) as response:
                    if response.status == 429:
                        # Rate limited - check retry delay from response
                        error_data = await response.json()
                        error_msg = error_data.get('error', {}).get('message', '')

                        # Extract retry time if provided
                        import re
                        retry_match = re.search(r'retry in (\d+(?:\.\d+)?)s', error_msg)
                        wait_time = float(retry_match.group(1)) if retry_match else retry_delay * (attempt + 1)

                        if attempt < max_retries - 1:
                            print(f"[VISION AI] Rate limited, waiting {wait_time:.1f}s before retry {attempt + 2}/{max_retries}")
                            await asyncio.sleep(min(wait_time + 1, 60))  # Cap at 60s
                            continue
                        else:
                            return {
                                "error": "Rate limit exceeded",
                                "message": "Gemini API rate limit reached. Please wait a minute and try again.",
                                "retry_after_seconds": wait_time
                            }

                    if response.status != 200:
                        error_text = await response.text()
                        print(f"[VISION AI] API error: {error_text}")
                        return {
                            "error": f"API request failed: {response.status}",
                            "details": error_text
                        }

                    result = await response.json()

                    # Extract the response text
                    if 'candidates' in result and result['candidates']:
                        content = result['candidates'][0].get('content', {})
                        parts = content.get('parts', [])
                        if parts:
                            text = parts[0].get('text', '')
                            try:
                                # Try to parse as JSON
                                return json.loads(text)
                            except json.JSONDecodeError:
                                # Return as plain text
                                return {"analysis": text}

                    return {"error": "No response from Gemini"}

            except Exception as e:
                print(f"[VISION AI] Request error (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay * (attempt + 1))
                else:
                    return {"error": str(e)}

    async def analyze_match(
        self,
        video_id: str,
        frame_paths: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Analyze a full match from saved frames.

        Args:
            video_id: ID of the processed video
            frame_paths: Optional list of specific frame paths to analyze

        Returns:
            Comprehensive match analysis
        """
        # Find frames for this video
        if not frame_paths:
            frames_dir = settings.DATA_DIR / "training" / "frames" / video_id
            if frames_dir.exists():
                frame_paths = sorted(frames_dir.glob("*.jpg"))[:10]
            else:
                # Try the regular frames directory
                frames_dir = settings.FRAMES_DIR
                frame_paths = []

        if not frame_paths:
            return {
                "error": "No frames found for analysis",
                "suggestion": "Process the video first to extract frames"
            }

        # Convert to strings if Path objects
        frame_paths = [str(p) for p in frame_paths]

        # Perform analysis
        analysis = await self.analyze_frames(frame_paths)

        # Add metadata
        analysis['video_id'] = video_id
        analysis['frames_analyzed'] = len(frame_paths)
        analysis['analyzed_at'] = datetime.now().isoformat()

        return analysis

    async def answer_question(
        self,
        question: str,
        frames: List[Any],
    ) -> Dict[str, Any]:
        """
        Answer a specific question about the match footage.

        Args:
            question: User's question
            frames: Frames to analyze

        Returns:
            Answer based on visual analysis
        """
        prompt = self.SPECIFIC_QUESTION_PROMPT.format(question=question)
        return await self.analyze_frames(frames, custom_prompt=prompt)

    async def get_coaching_summary(
        self,
        frames: List[Any],
        team_context: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Get a coaching-focused summary for half-time or post-match.

        Args:
            frames: Key frames from the match
            team_context: Optional context about the team (level, formation, etc.)

        Returns:
            Coaching summary with actionable points
        """
        context = ""
        if team_context:
            context = f"""
Team Context:
- Level: {team_context.get('level', 'grassroots')}
- Preferred Formation: {team_context.get('formation', 'unknown')}
- Key Focus Areas: {', '.join(team_context.get('focus_areas', ['general improvement']))}
"""

        prompt = f"""You are delivering a coaching talk based on match footage.
{context}

Analyze these frames and provide:

1. **What's Working Well** (2-3 points)
   - Specific positives to praise and reinforce

2. **Immediate Adjustments** (2-3 points)
   - Things to fix RIGHT NOW in the next half
   - Be specific about who needs to do what

3. **Key Message**
   - One clear, motivating message for the team

4. **Individual Notes** (if visible)
   - Any player-specific feedback based on what you see

Format as JSON:
{{
    "working_well": [
        {{"point": "description", "reinforce": "how to keep doing this"}}
    ],
    "adjustments": [
        {{"issue": "what's wrong", "fix": "specific action to take", "who": "position/player"}}
    ],
    "key_message": "Your main motivational/tactical message",
    "individual_notes": [
        {{"position": "e.g., left back", "feedback": "specific feedback"}}
    ],
    "energy_level": "high/medium/low - based on body language you see",
    "tactical_grade": "A/B/C/D - overall tactical execution"
}}

Be direct, specific, and motivating. This is for a real team talk."""

        return await self.analyze_frames(frames, custom_prompt=prompt)


    async def annotate_frame_for_training(
        self,
        frame,  # numpy array or path
        existing_detections: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """
        Use Gemini to help annotate a frame for training data.

        This leverages the LLM's football knowledge to improve annotations:
        - Verify if all players are detected
        - Identify missed players
        - Classify player roles (attacker, defender, goalkeeper)
        - Identify tactical actions (pressing, making run, etc.)

        Args:
            frame: Frame image to annotate
            existing_detections: Current detections from YOLO

        Returns:
            Suggested corrections and additional annotations
        """
        detection_context = ""
        if existing_detections:
            detection_context = f"""
Current detections from our system: {len(existing_detections)} players found.
Approximate positions: {[f"({d.get('x_center', 0):.2f}, {d.get('y_center', 0):.2f})" for d in existing_detections[:5]]}...
"""

        prompt = f"""You are helping to create training data for a football player detection AI.
{detection_context}

Analyze this football match frame and provide:

1. **Player Count**: How many players can you see? (Should be up to 22 + referees)
2. **Missed Detections**: Are there any players our system might have missed?
3. **Player Roles**: For visible players, identify their likely role:
   - Goalkeeper (GK)
   - Defender (DEF)
   - Midfielder (MID)
   - Attacker (ATT)
   - Referee (REF)

4. **Tactical State**: What is happening in this frame?
   - Attacking phase / Defensive phase / Transition
   - Any pressing happening?
   - Set piece?

5. **Annotation Quality**: Rate how good this frame would be for training (1-10)
   - Is the view clear?
   - Are players distinguishable?
   - Is it a typical game situation?

Return as JSON:
{{
    "visible_player_count": number,
    "detection_count_correct": true/false,
    "missed_players": [
        {{"approximate_location": "left side near goal", "likely_role": "GK"}}
    ],
    "player_roles_visible": {{
        "goalkeepers": number,
        "defenders": number,
        "midfielders": number,
        "attackers": number,
        "referees": number
    }},
    "tactical_state": {{
        "phase": "attacking/defending/transition/set_piece",
        "pressing_active": true/false,
        "description": "brief description"
    }},
    "training_quality_score": number (1-10),
    "training_quality_notes": "why this score",
    "suggestions": ["any suggestions for improving annotations"]
}}"""

        frames = [frame] if not isinstance(frame, list) else frame
        return await self.analyze_frames(frames, custom_prompt=prompt)


# Global instance
vision_ai_service = VisionAIService()
