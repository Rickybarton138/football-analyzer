"""
Jersey Number Recognition Service

Uses OCR (EasyOCR or PaddleOCR) to detect and recognize jersey numbers
from player bounding boxes. Builds confidence over multiple frames to
establish reliable track_id -> jersey_number mappings.
"""
import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass, field
import re

from models.schemas import DetectedPlayer, TeamSide


@dataclass
class JerseyNumberObservation:
    """Single observation of a jersey number."""
    number: int
    confidence: float
    frame_number: int


@dataclass
class PlayerIdentity:
    """Confirmed identity of a player."""
    track_id: int
    jersey_number: int
    player_name: Optional[str] = None
    team: TeamSide = TeamSide.UNKNOWN
    confidence: float = 0.0
    observation_count: int = 0


class JerseyOCRService:
    """
    Service for recognizing jersey numbers from player detections.

    Uses OCR to read numbers from the torso/back region of player bounding boxes.
    Builds confidence over multiple observations to handle OCR noise.
    """

    # Minimum observations needed to confirm a jersey number
    MIN_OBSERVATIONS = 5
    # Minimum confidence to accept an OCR result
    MIN_OCR_CONFIDENCE = 0.4
    # Minimum agreement ratio to confirm a number
    MIN_AGREEMENT_RATIO = 0.6

    def __init__(self):
        self.ocr_engine = None
        self.ocr_type = None  # 'easyocr' or 'paddleocr'

        # Observations per track: track_id -> list of observations
        self.observations: Dict[int, List[JerseyNumberObservation]] = defaultdict(list)

        # Confirmed identities: track_id -> PlayerIdentity
        self.confirmed_identities: Dict[int, PlayerIdentity] = {}

        # Jersey number to track_id mapping (for quick lookup)
        self.number_to_track: Dict[int, int] = {}

        # Player roster: jersey_number -> player_name
        self.roster: Dict[int, str] = {}

        # Team to track for OCR (usually own team only)
        self.target_team: Optional[TeamSide] = None

        # Processing stats
        self.total_ocr_attempts = 0
        self.successful_reads = 0

    async def initialize(self):
        """Initialize OCR engine. Tries EasyOCR first, falls back to PaddleOCR."""
        try:
            import easyocr
            self.ocr_engine = easyocr.Reader(['en'], gpu=False)
            self.ocr_type = 'easyocr'
            print("JerseyOCR: Initialized with EasyOCR")
            return True
        except ImportError:
            pass

        try:
            from paddleocr import PaddleOCR
            self.ocr_engine = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
            self.ocr_type = 'paddleocr'
            print("JerseyOCR: Initialized with PaddleOCR")
            return True
        except ImportError:
            pass

        print("JerseyOCR: No OCR engine available. Install easyocr or paddleocr.")
        return False

    def set_roster(self, roster: Dict[int, str]):
        """
        Set the team roster for name mapping.

        Args:
            roster: Dict mapping jersey number to player name
                   e.g., {1: "John Smith", 10: "Jane Doe", ...}
        """
        self.roster = roster

    def set_target_team(self, team: TeamSide):
        """Set which team to run OCR on (to save processing)."""
        self.target_team = team

    async def process_players(
        self,
        frame: np.ndarray,
        players: List[DetectedPlayer],
        frame_number: int
    ) -> List[DetectedPlayer]:
        """
        Process player detections to recognize jersey numbers.

        Args:
            frame: Current video frame
            players: List of detected players
            frame_number: Current frame number

        Returns:
            Players with jersey_number field populated where recognized
        """
        if self.ocr_engine is None:
            return players

        for player in players:
            # Skip if not target team (when set)
            if self.target_team and player.team != self.target_team:
                continue

            # Skip if already confidently identified
            if player.track_id in self.confirmed_identities:
                identity = self.confirmed_identities[player.track_id]
                player.jersey_number = identity.jersey_number
                continue

            # Try to read jersey number
            number, confidence = await self._read_jersey_number(frame, player)

            if number is not None:
                # Record observation
                obs = JerseyNumberObservation(
                    number=number,
                    confidence=confidence,
                    frame_number=frame_number
                )
                self.observations[player.track_id].append(obs)

                # Check if we can confirm identity
                self._try_confirm_identity(player.track_id, player.team)

                # Set jersey number if confirmed
                if player.track_id in self.confirmed_identities:
                    player.jersey_number = self.confirmed_identities[player.track_id].jersey_number

        return players

    async def _read_jersey_number(
        self,
        frame: np.ndarray,
        player: DetectedPlayer
    ) -> Tuple[Optional[int], float]:
        """
        Attempt to read jersey number from player bounding box.

        Focuses on the back/torso region where numbers typically appear.
        """
        self.total_ocr_attempts += 1

        # Extract player ROI
        x1, y1, x2, y2 = player.bbox.x1, player.bbox.y1, player.bbox.x2, player.bbox.y2

        # Ensure valid bounds
        h, w = frame.shape[:2]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        if x2 <= x1 or y2 <= y1:
            return None, 0.0

        player_roi = frame[y1:y2, x1:x2]

        # Focus on torso/back area (where numbers appear)
        roi_h, roi_w = player_roi.shape[:2]

        # Skip if too small (can't read numbers)
        if roi_h < 30 or roi_w < 20:
            return None, 0.0

        # Extract back region (upper-middle portion)
        back_y1 = int(roi_h * 0.15)
        back_y2 = int(roi_h * 0.65)
        back_x1 = int(roi_w * 0.15)
        back_x2 = int(roi_w * 0.85)

        back_roi = player_roi[back_y1:back_y2, back_x1:back_x2]

        if back_roi.size == 0:
            return None, 0.0

        # Preprocess for OCR
        processed = self._preprocess_for_ocr(back_roi)

        # Run OCR
        try:
            if self.ocr_type == 'easyocr':
                results = self.ocr_engine.readtext(processed, allowlist='0123456789')
                number, confidence = self._parse_easyocr_results(results)
            elif self.ocr_type == 'paddleocr':
                results = self.ocr_engine.ocr(processed, cls=True)
                number, confidence = self._parse_paddleocr_results(results)
            else:
                return None, 0.0

            if number is not None:
                self.successful_reads += 1

            return number, confidence

        except Exception as e:
            # OCR can fail on unusual images
            return None, 0.0

    def _preprocess_for_ocr(self, roi: np.ndarray) -> np.ndarray:
        """Preprocess image region for better OCR results."""
        # Convert to grayscale
        if len(roi.shape) == 3:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi

        # Resize for better OCR (larger is often better)
        scale = 3
        resized = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        # Apply CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(resized)

        # Threshold to make numbers stand out
        _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Also try inverted (numbers could be light on dark or dark on light)
        # Return whichever has more contrast
        return thresh

    def _parse_easyocr_results(self, results) -> Tuple[Optional[int], float]:
        """Parse EasyOCR results to extract jersey number."""
        if not results:
            return None, 0.0

        for bbox, text, confidence in results:
            if confidence < self.MIN_OCR_CONFIDENCE:
                continue

            # Extract digits only
            digits = re.sub(r'[^0-9]', '', text)

            if digits:
                try:
                    number = int(digits)
                    # Valid jersey numbers are typically 1-99
                    if 1 <= number <= 99:
                        return number, confidence
                except ValueError:
                    continue

        return None, 0.0

    def _parse_paddleocr_results(self, results) -> Tuple[Optional[int], float]:
        """Parse PaddleOCR results to extract jersey number."""
        if not results or not results[0]:
            return None, 0.0

        for line in results[0]:
            if not line:
                continue

            text = line[1][0]
            confidence = line[1][1]

            if confidence < self.MIN_OCR_CONFIDENCE:
                continue

            # Extract digits only
            digits = re.sub(r'[^0-9]', '', text)

            if digits:
                try:
                    number = int(digits)
                    if 1 <= number <= 99:
                        return number, confidence
                except ValueError:
                    continue

        return None, 0.0

    def _try_confirm_identity(self, track_id: int, team: TeamSide):
        """
        Try to confirm a player's jersey number based on observations.

        Uses voting/agreement to handle OCR noise.
        """
        observations = self.observations[track_id]

        if len(observations) < self.MIN_OBSERVATIONS:
            return

        # Count votes for each number
        number_votes: Dict[int, float] = defaultdict(float)

        for obs in observations:
            # Weight by confidence
            number_votes[obs.number] += obs.confidence

        if not number_votes:
            return

        # Find most voted number
        best_number = max(number_votes, key=number_votes.get)
        total_weight = sum(number_votes.values())
        agreement = number_votes[best_number] / total_weight

        # Confirm if enough agreement
        if agreement >= self.MIN_AGREEMENT_RATIO:
            player_name = self.roster.get(best_number)

            identity = PlayerIdentity(
                track_id=track_id,
                jersey_number=best_number,
                player_name=player_name,
                team=team,
                confidence=agreement,
                observation_count=len(observations)
            )

            self.confirmed_identities[track_id] = identity
            self.number_to_track[best_number] = track_id

    def get_player_by_number(self, jersey_number: int) -> Optional[PlayerIdentity]:
        """Get player identity by jersey number."""
        track_id = self.number_to_track.get(jersey_number)
        if track_id:
            return self.confirmed_identities.get(track_id)
        return None

    def get_player_by_track(self, track_id: int) -> Optional[PlayerIdentity]:
        """Get player identity by track ID."""
        return self.confirmed_identities.get(track_id)

    def get_all_identified_players(self) -> List[PlayerIdentity]:
        """Get all confirmed player identities."""
        return list(self.confirmed_identities.values())

    def get_jersey_number(self, track_id: int) -> Optional[int]:
        """Quick lookup of jersey number for a track ID."""
        identity = self.confirmed_identities.get(track_id)
        return identity.jersey_number if identity else None

    def get_stats(self) -> Dict:
        """Get OCR processing statistics."""
        return {
            "total_ocr_attempts": self.total_ocr_attempts,
            "successful_reads": self.successful_reads,
            "success_rate": self.successful_reads / max(1, self.total_ocr_attempts),
            "players_identified": len(self.confirmed_identities),
            "pending_observations": len(self.observations) - len(self.confirmed_identities)
        }

    def reset(self):
        """Reset all observations and identities."""
        self.observations.clear()
        self.confirmed_identities.clear()
        self.number_to_track.clear()
        self.total_ocr_attempts = 0
        self.successful_reads = 0


# Global instance
jersey_ocr_service = JerseyOCRService()
