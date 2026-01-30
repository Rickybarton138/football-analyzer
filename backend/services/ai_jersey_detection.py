"""
AI-Powered Jersey Number Detection Service

Uses GPT-4 Vision or Claude Vision to detect jersey numbers from player images.
Much more accurate than traditional OCR for handling motion blur, partial visibility,
and varying angles common in football footage.
"""
import asyncio
import base64
import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass, field
import json
import httpx

from models.schemas import DetectedPlayer, TeamSide
from config import settings


@dataclass
class JerseyObservation:
    """Single observation of a jersey number from AI vision."""
    number: int
    confidence: float
    frame_number: int
    source: str  # 'back', 'front', 'side'


@dataclass
class PlayerJerseyInfo:
    """Confirmed jersey information for a player."""
    track_id: int
    jersey_number: int
    team: TeamSide
    confidence: float
    observation_count: int
    confirmed: bool = False
    manually_corrected: bool = False


class AIJerseyDetectionService:
    """
    AI-powered jersey number detection using GPT-4V or Claude Vision.

    Key features:
    - Batch multiple players into single API calls (cost optimization)
    - Send full frame + individual crops for context
    - Cache confirmed numbers to avoid re-processing
    - Confirm via voting over 3+ observations
    - Support GPT-4V (primary) and Claude Vision (fallback)
    """

    # Minimum observations to confirm a jersey number
    MIN_OBSERVATIONS = 3
    # Minimum confidence from AI to accept a reading
    MIN_AI_CONFIDENCE = 0.6
    # Minimum agreement ratio for confirmation
    MIN_AGREEMENT_RATIO = 0.6
    # Maximum players per batch API call
    MAX_BATCH_SIZE = 6
    # Minimum bounding box height to process (skip very small/distant players)
    MIN_BBOX_HEIGHT = 50
    # Process every N frames (not every frame for cost savings)
    PROCESS_INTERVAL = 30

    # API providers
    PROVIDER_OPENAI = "openai"
    PROVIDER_CLAUDE = "claude"

    def __init__(self):
        self.provider: str = self.PROVIDER_OPENAI
        self.openai_api_key: Optional[str] = None
        self.anthropic_api_key: Optional[str] = None
        self.client: Optional[httpx.AsyncClient] = None

        # Observations per track: track_id -> list of observations
        self.observations: Dict[int, List[JerseyObservation]] = defaultdict(list)

        # Confirmed identities: track_id -> PlayerJerseyInfo
        self.confirmed_players: Dict[int, PlayerJerseyInfo] = {}

        # Jersey number to track_id mapping
        self.number_to_track: Dict[Tuple[int, TeamSide], int] = {}

        # Manual corrections: track_id -> jersey_number
        self.manual_corrections: Dict[int, int] = {}

        # Processing stats
        self.api_calls = 0
        self.total_players_processed = 0
        self.successful_detections = 0
        self.last_processed_frame = -1

    async def initialize(
        self,
        openai_api_key: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
        provider: str = "openai"
    ) -> bool:
        """
        Initialize the AI jersey detection service.

        Args:
            openai_api_key: OpenAI API key for GPT-4V
            anthropic_api_key: Anthropic API key for Claude Vision
            provider: Primary provider to use ("openai" or "claude")
        """
        self.openai_api_key = openai_api_key or settings.OPENAI_API_KEY
        self.anthropic_api_key = anthropic_api_key or getattr(settings, 'ANTHROPIC_API_KEY', None)
        self.provider = provider

        # Check if we have valid API keys
        if self.provider == self.PROVIDER_OPENAI and self.openai_api_key:
            self.client = httpx.AsyncClient(
                base_url="https://api.openai.com/v1",
                headers={
                    "Authorization": f"Bearer {self.openai_api_key}",
                    "Content-Type": "application/json"
                },
                timeout=60.0
            )
            print(f"AIJerseyDetection: Initialized with OpenAI GPT-4V")
            return True
        elif self.provider == self.PROVIDER_CLAUDE and self.anthropic_api_key:
            self.client = httpx.AsyncClient(
                base_url="https://api.anthropic.com",
                headers={
                    "x-api-key": self.anthropic_api_key,
                    "anthropic-version": "2023-06-01",
                    "Content-Type": "application/json"
                },
                timeout=60.0
            )
            print(f"AIJerseyDetection: Initialized with Claude Vision")
            return True

        # Try fallback provider
        if self.openai_api_key:
            self.provider = self.PROVIDER_OPENAI
            self.client = httpx.AsyncClient(
                base_url="https://api.openai.com/v1",
                headers={
                    "Authorization": f"Bearer {self.openai_api_key}",
                    "Content-Type": "application/json"
                },
                timeout=60.0
            )
            print(f"AIJerseyDetection: Initialized with OpenAI GPT-4V (fallback)")
            return True
        elif self.anthropic_api_key:
            self.provider = self.PROVIDER_CLAUDE
            self.client = httpx.AsyncClient(
                base_url="https://api.anthropic.com",
                headers={
                    "x-api-key": self.anthropic_api_key,
                    "anthropic-version": "2023-06-01",
                    "Content-Type": "application/json"
                },
                timeout=60.0
            )
            print(f"AIJerseyDetection: Initialized with Claude Vision (fallback)")
            return True

        print("AIJerseyDetection: No API keys available. Service disabled.")
        return False

    async def process_frame(
        self,
        frame: np.ndarray,
        players: List[DetectedPlayer],
        frame_number: int
    ) -> List[DetectedPlayer]:
        """
        Process a frame to detect jersey numbers using AI vision.

        Args:
            frame: Current video frame (BGR numpy array)
            players: List of detected players in this frame
            frame_number: Current frame number

        Returns:
            Players with jersey_number field populated where detected
        """
        if self.client is None:
            return players

        # Skip if not enough frames have passed since last processing
        if frame_number - self.last_processed_frame < self.PROCESS_INTERVAL:
            # Still apply confirmed numbers even if we skip processing
            return self._apply_confirmed_numbers(players)

        self.last_processed_frame = frame_number

        # Filter players to process
        players_to_process = []
        for player in players:
            # Skip if manually corrected
            if player.track_id in self.manual_corrections:
                player.jersey_number = self.manual_corrections[player.track_id]
                continue

            # Skip if already confirmed
            if player.track_id in self.confirmed_players:
                info = self.confirmed_players[player.track_id]
                if info.confirmed:
                    player.jersey_number = info.jersey_number
                    continue

            # Skip small bounding boxes
            bbox_height = player.bbox.y2 - player.bbox.y1
            if bbox_height < self.MIN_BBOX_HEIGHT:
                continue

            players_to_process.append(player)

        # Process in batches
        if players_to_process:
            for i in range(0, len(players_to_process), self.MAX_BATCH_SIZE):
                batch = players_to_process[i:i + self.MAX_BATCH_SIZE]
                await self._process_batch(frame, batch, frame_number)

        # Apply all confirmed numbers
        return self._apply_confirmed_numbers(players)

    async def _process_batch(
        self,
        frame: np.ndarray,
        players: List[DetectedPlayer],
        frame_number: int
    ):
        """Process a batch of players with a single API call."""
        if not players:
            return

        try:
            # Prepare images
            frame_resized = self._resize_for_api(frame, max_size=1024)
            frame_b64 = self._encode_image(frame_resized)

            # Extract and encode player crops
            player_crops = []
            for idx, player in enumerate(players):
                crop = self._extract_player_crop(frame, player)
                if crop is not None:
                    crop_resized = self._resize_for_api(crop, max_size=256)
                    crop_b64 = self._encode_image(crop_resized)
                    player_crops.append({
                        "index": idx + 1,
                        "track_id": player.track_id,
                        "team": player.team.value,
                        "image_b64": crop_b64
                    })

            if not player_crops:
                return

            # Build prompt
            prompt = self._build_detection_prompt(len(player_crops))

            # Call AI API
            if self.provider == self.PROVIDER_OPENAI:
                results = await self._call_openai_vision(frame_b64, player_crops, prompt)
            else:
                results = await self._call_claude_vision(frame_b64, player_crops, prompt)

            self.api_calls += 1
            self.total_players_processed += len(player_crops)

            # Process results
            if results:
                self._process_ai_results(results, players, frame_number)

        except Exception as e:
            print(f"AIJerseyDetection: Error processing batch: {e}")

    def _build_detection_prompt(self, num_players: int) -> str:
        """Build the prompt for jersey number detection."""
        return f"""You are analyzing football/soccer match footage to identify jersey numbers.

I'm showing you:
1. A full frame from the match for context
2. {num_players} individual player crops numbered 1-{num_players}

For each player crop, identify:
1. The jersey number visible (if any)
2. Your confidence level (0.0-1.0)
3. The view type: "back" (number clearly on back), "front" (smaller chest number), or "side" (partial view)

Return your analysis as a JSON array with this format:
```json
[
  {{"player_index": 1, "jersey_number": 10, "confidence": 0.9, "source": "back"}},
  {{"player_index": 2, "jersey_number": null, "confidence": 0.0, "source": "unclear"}},
  ...
]
```

Important rules:
- Jersey numbers are typically 1-99
- If you can't clearly see a number, set jersey_number to null
- Higher confidence (0.8+) for clear back numbers, lower (0.5-0.7) for partial/front views
- Be conservative - it's better to return null than guess wrong
- Consider that motion blur and angles can distort numbers (6/9, 1/7, etc.)

Analyze each player crop now:"""

    async def _call_openai_vision(
        self,
        frame_b64: str,
        player_crops: List[Dict],
        prompt: str
    ) -> Optional[List[Dict]]:
        """Call OpenAI GPT-4V API for jersey detection."""
        # Build content with images
        content = [
            {"type": "text", "text": "Full match frame for context:"},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{frame_b64}",
                    "detail": "low"  # Low detail for context frame
                }
            }
        ]

        # Add player crops
        for crop in player_crops:
            content.append({
                "type": "text",
                "text": f"Player {crop['index']} (Team: {crop['team']}):"
            })
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{crop['image_b64']}",
                    "detail": "high"  # High detail for player crops
                }
            })

        content.append({"type": "text", "text": prompt})

        try:
            response = await self.client.post(
                "/chat/completions",
                json={
                    "model": "gpt-4o",  # GPT-4o has good vision capabilities
                    "max_tokens": 500,
                    "messages": [
                        {"role": "user", "content": content}
                    ]
                }
            )

            if response.status_code != 200:
                print(f"AIJerseyDetection: OpenAI API error: {response.text}")
                return None

            result = response.json()
            response_text = result["choices"][0]["message"]["content"]

            # Parse JSON from response
            return self._parse_ai_response(response_text)

        except Exception as e:
            print(f"AIJerseyDetection: OpenAI call failed: {e}")
            return None

    async def _call_claude_vision(
        self,
        frame_b64: str,
        player_crops: List[Dict],
        prompt: str
    ) -> Optional[List[Dict]]:
        """Call Claude Vision API for jersey detection."""
        # Build content with images
        content = [
            {"type": "text", "text": "Full match frame for context:"},
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": frame_b64
                }
            }
        ]

        # Add player crops
        for crop in player_crops:
            content.append({
                "type": "text",
                "text": f"Player {crop['index']} (Team: {crop['team']}):"
            })
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": crop['image_b64']
                }
            })

        content.append({"type": "text", "text": prompt})

        try:
            response = await self.client.post(
                "/v1/messages",
                json={
                    "model": "claude-sonnet-4-20250514",
                    "max_tokens": 500,
                    "messages": [
                        {"role": "user", "content": content}
                    ]
                }
            )

            if response.status_code != 200:
                print(f"AIJerseyDetection: Claude API error: {response.text}")
                return None

            result = response.json()
            response_text = result["content"][0]["text"]

            # Parse JSON from response
            return self._parse_ai_response(response_text)

        except Exception as e:
            print(f"AIJerseyDetection: Claude call failed: {e}")
            return None

    def _parse_ai_response(self, response_text: str) -> Optional[List[Dict]]:
        """Parse the JSON response from AI."""
        try:
            # Find JSON array in response
            start = response_text.find('[')
            end = response_text.rfind(']') + 1
            if start == -1 or end == 0:
                return None

            json_str = response_text[start:end]
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"AIJerseyDetection: Failed to parse AI response: {e}")
            return None

    def _process_ai_results(
        self,
        results: List[Dict],
        players: List[DetectedPlayer],
        frame_number: int
    ):
        """Process AI detection results and update observations."""
        for result in results:
            try:
                player_idx = result.get("player_index", 0) - 1
                jersey_number = result.get("jersey_number")
                confidence = result.get("confidence", 0.0)
                source = result.get("source", "unknown")

                if player_idx < 0 or player_idx >= len(players):
                    continue

                if jersey_number is None:
                    continue

                if not isinstance(jersey_number, int) or jersey_number < 1 or jersey_number > 99:
                    continue

                if confidence < self.MIN_AI_CONFIDENCE:
                    continue

                player = players[player_idx]

                # Record observation
                obs = JerseyObservation(
                    number=jersey_number,
                    confidence=confidence,
                    frame_number=frame_number,
                    source=source
                )
                self.observations[player.track_id].append(obs)
                self.successful_detections += 1

                # Try to confirm identity
                self._try_confirm_identity(player.track_id, player.team)

            except Exception as e:
                print(f"AIJerseyDetection: Error processing result: {e}")

    def _try_confirm_identity(self, track_id: int, team: TeamSide):
        """Try to confirm a player's jersey number based on observations."""
        observations = self.observations[track_id]

        if len(observations) < self.MIN_OBSERVATIONS:
            return

        # Count weighted votes for each number
        number_votes: Dict[int, float] = defaultdict(float)
        for obs in observations:
            number_votes[obs.number] += obs.confidence

        if not number_votes:
            return

        # Find most voted number
        best_number = max(number_votes, key=number_votes.get)
        total_weight = sum(number_votes.values())
        agreement = number_votes[best_number] / total_weight

        # Confirm if enough agreement
        if agreement >= self.MIN_AGREEMENT_RATIO:
            self.confirmed_players[track_id] = PlayerJerseyInfo(
                track_id=track_id,
                jersey_number=best_number,
                team=team,
                confidence=agreement,
                observation_count=len(observations),
                confirmed=True
            )
            self.number_to_track[(best_number, team)] = track_id
            print(f"AIJerseyDetection: Confirmed player {track_id} = #{best_number} "
                  f"(team={team.value}, conf={agreement:.2f}, obs={len(observations)})")

    def _apply_confirmed_numbers(self, players: List[DetectedPlayer]) -> List[DetectedPlayer]:
        """Apply confirmed jersey numbers to player detections."""
        for player in players:
            # Check manual corrections first
            if player.track_id in self.manual_corrections:
                player.jersey_number = self.manual_corrections[player.track_id]
            # Then confirmed detections
            elif player.track_id in self.confirmed_players:
                info = self.confirmed_players[player.track_id]
                if info.confirmed:
                    player.jersey_number = info.jersey_number
        return players

    def _extract_player_crop(
        self,
        frame: np.ndarray,
        player: DetectedPlayer
    ) -> Optional[np.ndarray]:
        """Extract player crop from frame with some padding."""
        h, w = frame.shape[:2]

        # Get bbox with padding
        pad_x = int((player.bbox.x2 - player.bbox.x1) * 0.1)
        pad_y = int((player.bbox.y2 - player.bbox.y1) * 0.1)

        x1 = max(0, player.bbox.x1 - pad_x)
        y1 = max(0, player.bbox.y1 - pad_y)
        x2 = min(w, player.bbox.x2 + pad_x)
        y2 = min(h, player.bbox.y2 + pad_y)

        if x2 <= x1 or y2 <= y1:
            return None

        return frame[y1:y2, x1:x2].copy()

    def _resize_for_api(self, image: np.ndarray, max_size: int = 512) -> np.ndarray:
        """Resize image while maintaining aspect ratio."""
        h, w = image.shape[:2]
        if max(h, w) <= max_size:
            return image

        scale = max_size / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    def _encode_image(self, image: np.ndarray) -> str:
        """Encode image to base64 JPEG."""
        _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return base64.b64encode(buffer).decode('utf-8')

    def get_jersey_number(self, track_id: int) -> Optional[int]:
        """Get confirmed jersey number for a track ID."""
        # Check manual corrections first
        if track_id in self.manual_corrections:
            return self.manual_corrections[track_id]
        # Then confirmed detections
        if track_id in self.confirmed_players:
            info = self.confirmed_players[track_id]
            if info.confirmed:
                return info.jersey_number
        return None

    def get_player_by_number(self, jersey_number: int, team: TeamSide) -> Optional[int]:
        """Get track ID for a jersey number and team."""
        return self.number_to_track.get((jersey_number, team))

    def set_manual_correction(self, track_id: int, jersey_number: int):
        """Manually set/correct a player's jersey number."""
        self.manual_corrections[track_id] = jersey_number

        # Also update confirmed players
        if track_id in self.confirmed_players:
            self.confirmed_players[track_id].jersey_number = jersey_number
            self.confirmed_players[track_id].manually_corrected = True

        print(f"AIJerseyDetection: Manual correction - player {track_id} = #{jersey_number}")

    def get_all_detections(self) -> Dict[int, Dict]:
        """Get all jersey detections (confirmed and pending)."""
        result = {}

        # Add confirmed players
        for track_id, info in self.confirmed_players.items():
            result[track_id] = {
                "jersey_number": info.jersey_number,
                "team": info.team.value,
                "confidence": info.confidence,
                "observation_count": info.observation_count,
                "confirmed": info.confirmed,
                "manually_corrected": info.manually_corrected
            }

        # Add pending observations
        for track_id, observations in self.observations.items():
            if track_id not in result and observations:
                # Find most common number
                number_counts: Dict[int, int] = defaultdict(int)
                for obs in observations:
                    number_counts[obs.number] += 1

                if number_counts:
                    best_number = max(number_counts, key=number_counts.get)
                    result[track_id] = {
                        "jersey_number": best_number,
                        "team": "unknown",
                        "confidence": 0.0,
                        "observation_count": len(observations),
                        "confirmed": False,
                        "manually_corrected": False,
                        "pending": True
                    }

        return result

    def get_stats(self) -> Dict:
        """Get service statistics."""
        return {
            "api_calls": self.api_calls,
            "total_players_processed": self.total_players_processed,
            "successful_detections": self.successful_detections,
            "confirmed_players": len([p for p in self.confirmed_players.values() if p.confirmed]),
            "pending_observations": len(self.observations) - len(self.confirmed_players),
            "manual_corrections": len(self.manual_corrections),
            "provider": self.provider
        }

    def reset(self):
        """Reset all observations and detections."""
        self.observations.clear()
        self.confirmed_players.clear()
        self.number_to_track.clear()
        self.manual_corrections.clear()
        self.api_calls = 0
        self.total_players_processed = 0
        self.successful_detections = 0
        self.last_processed_frame = -1

    async def close(self):
        """Close the HTTP client."""
        if self.client:
            await self.client.aclose()
            self.client = None


# Global instance
ai_jersey_detection_service = AIJerseyDetectionService()
