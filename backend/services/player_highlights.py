"""
Player Highlights Generator Service

Generates individual highlight videos for each identified player based on
their jersey number. Tracks events and involvement throughout the match
and compiles clips into per-player highlight reels.
"""
import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field, asdict
from pathlib import Path
from datetime import datetime
import json
import os
import uuid

from models.schemas import (
    DetectedPlayer, MatchEvent, EventType, TeamSide, Position
)


@dataclass
class PlayerMoment:
    """A moment where a player is involved in the action."""
    frame_start: int
    frame_end: int
    timestamp_start_ms: int
    timestamp_end_ms: int
    event_type: Optional[str] = None  # 'touch', 'pass', 'shot', 'tackle', 'goal', etc.
    importance: float = 1.0  # For ranking clips
    description: str = ""
    position: Optional[Position] = None
    clip_path: Optional[str] = None  # Path to extracted clip file
    clip_id: Optional[str] = None  # Unique ID for this clip


@dataclass
class ClipMetadata:
    """Metadata for an extracted video clip."""
    clip_id: str
    jersey_number: int
    player_name: Optional[str]
    team: str
    event_type: str
    frame_start: int
    frame_end: int
    timestamp_start_ms: int
    timestamp_end_ms: int
    duration_seconds: float
    importance: float
    clip_path: str
    extracted_at: str


@dataclass
class PlayerHighlightData:
    """Collected data for a player's highlights."""
    jersey_number: int
    player_name: Optional[str]
    team: TeamSide
    track_ids: Set[int] = field(default_factory=set)  # May have multiple tracks if re-identified

    # Moments of involvement
    moments: List[PlayerMoment] = field(default_factory=list)

    # Event counts
    touches: int = 0
    passes_attempted: int = 0
    passes_completed: int = 0
    shots: int = 0
    goals: int = 0
    tackles: int = 0
    interceptions: int = 0

    # Movement data
    total_distance_m: float = 0.0
    sprints: int = 0
    max_speed_kmh: float = 0.0


class PlayerHighlightsService:
    """
    Service for generating per-player highlight videos.

    Tracks each identified player throughout the match and extracts
    clips of their key moments (touches, passes, shots, tackles, etc.)
    to compile individual highlight reels.
    """

    # Time padding around events (in seconds)
    # User requirement: clips should be 5-10 seconds long
    CLIP_PADDING_BEFORE = 4.0  # 4 seconds before the event
    CLIP_PADDING_AFTER = 4.0   # 4 seconds after = 8 second clips on average

    # Minimum and maximum clip duration (in seconds)
    MIN_CLIP_DURATION = 5.0  # User requirement: minimum 5 seconds
    MAX_CLIP_DURATION = 10.0  # User requirement: maximum 10 seconds

    # Minimum gap between clips to merge (in seconds)
    MERGE_GAP_THRESHOLD = 2.0

    # Ball proximity threshold for clip creation (pixels)
    BALL_PROXIMITY_THRESHOLD = 60  # Player must be within 60px of ball

    # Event importance weights for highlight ranking
    IMPORTANCE_WEIGHTS = {
        'goal': 10.0,
        'shot': 5.0,
        'assist': 7.0,
        'key_pass': 4.0,
        'pass': 1.0,
        'tackle': 3.0,
        'interception': 3.0,
        'touch': 0.5,
        'dribble': 2.0,
        'header': 2.0,
    }

    def __init__(self):
        self.players: Dict[int, PlayerHighlightData] = {}  # jersey_number -> data
        self.track_to_jersey: Dict[int, int] = {}  # track_id -> jersey_number

        # Video metadata
        self.video_path: Optional[str] = None
        self.fps: float = 30.0
        self.frame_width: int = 1920
        self.frame_height: int = 1080
        self.total_frames: int = 0

        # Current tracking state
        self.current_frame: int = 0
        self.last_touch_frame: Dict[int, int] = {}  # jersey_number -> last touch frame

        # Output settings
        self.output_dir: str = "data/highlights"
        self.clips_dir: str = "data/clips"  # Individual clips directory

        # Clips registry - all extracted clips indexed by clip_id
        self.clips_registry: Dict[str, ClipMetadata] = {}

    def set_video_info(
        self,
        video_path: str,
        fps: float,
        width: int,
        height: int,
        total_frames: int
    ):
        """Set video metadata for clip extraction."""
        self.video_path = video_path
        self.fps = fps
        self.frame_width = width
        self.frame_height = height
        self.total_frames = total_frames

    def register_player(
        self,
        jersey_number: int,
        player_name: Optional[str] = None,
        team: TeamSide = TeamSide.HOME,
        track_id: Optional[int] = None
    ):
        """Register a player for highlight tracking."""
        if jersey_number not in self.players:
            self.players[jersey_number] = PlayerHighlightData(
                jersey_number=jersey_number,
                player_name=player_name,
                team=team
            )

        if track_id is not None:
            self.players[jersey_number].track_ids.add(track_id)
            self.track_to_jersey[track_id] = jersey_number

    def link_track_to_player(self, track_id: int, jersey_number: int):
        """Link a track ID to a jersey number (from OCR)."""
        if jersey_number in self.players:
            self.players[jersey_number].track_ids.add(track_id)
            self.track_to_jersey[track_id] = jersey_number

    def process_frame(
        self,
        frame_number: int,
        players: List[DetectedPlayer],
        ball_possessed_by: Optional[int] = None,
        timestamp_ms: int = 0
    ):
        """
        Process a frame to track player involvement.

        Args:
            frame_number: Current frame number
            players: Detected players in this frame
            ball_possessed_by: Track ID of player with ball (if any)
            timestamp_ms: Frame timestamp in milliseconds
        """
        self.current_frame = frame_number

        for player in players:
            # Get jersey number from detection or track mapping
            jersey_number = self._get_jersey_number(player)

            # If no jersey number but valid track_id, use track_id as pseudo-jersey number
            # This allows tracking players even without OCR
            if jersey_number is None and player.track_id >= 0:
                # Use negative numbers for track-based IDs to distinguish from real jersey numbers
                # Track ID 123 becomes pseudo-jersey -123
                jersey_number = -player.track_id

            if jersey_number is None:
                continue

            # Auto-register player if not already registered
            if jersey_number not in self.players:
                if jersey_number < 0:
                    # Track-based registration (no OCR)
                    player_name = f"Track #{abs(jersey_number)}"
                else:
                    # Jersey number registration (OCR detected)
                    player_name = f"Player {jersey_number}"

                self.register_player(
                    jersey_number=jersey_number,
                    player_name=player_name,
                    team=player.team,
                    track_id=player.track_id if player.track_id >= 0 else None
                )

            # Update track association
            if player.track_id >= 0:
                self.link_track_to_player(player.track_id, jersey_number)

            # Check if player has the ball
            if ball_possessed_by == player.track_id:
                self._record_touch(jersey_number, frame_number, timestamp_ms)

    def _get_jersey_number(self, player: DetectedPlayer) -> Optional[int]:
        """Get jersey number for a player from detection or track mapping."""
        # Direct from detection (OCR result)
        if player.jersey_number is not None:
            return player.jersey_number

        # From track ID mapping
        if player.track_id in self.track_to_jersey:
            return self.track_to_jersey[player.track_id]

        return None

    def _record_touch(self, jersey_number: int, frame_number: int, timestamp_ms: int):
        """Record a ball touch for a player."""
        if jersey_number not in self.players:
            return

        player_data = self.players[jersey_number]
        player_data.touches += 1

        # Check if this extends an existing moment or starts a new one
        last_frame = self.last_touch_frame.get(jersey_number, -100)

        gap_frames = frame_number - last_frame
        gap_seconds = gap_frames / self.fps

        if gap_seconds <= self.MERGE_GAP_THRESHOLD and player_data.moments:
            # Extend the last moment
            player_data.moments[-1].frame_end = frame_number
            player_data.moments[-1].timestamp_end_ms = timestamp_ms
        else:
            # Start a new moment
            moment = PlayerMoment(
                frame_start=frame_number,
                frame_end=frame_number,
                timestamp_start_ms=timestamp_ms,
                timestamp_end_ms=timestamp_ms,
                event_type='touch',
                importance=self.IMPORTANCE_WEIGHTS.get('touch', 0.5)
            )
            player_data.moments.append(moment)

        self.last_touch_frame[jersey_number] = frame_number

    def record_event(
        self,
        event: MatchEvent,
        player_jersey: Optional[int] = None
    ):
        """
        Record a match event for a player's highlights.

        Args:
            event: The match event
            player_jersey: Jersey number of the involved player
        """
        # Try to get jersey number from event or parameter
        jersey_number = player_jersey

        if jersey_number is None and event.player_id is not None:
            jersey_number = self.track_to_jersey.get(event.player_id)

        if jersey_number is None or jersey_number not in self.players:
            return

        player_data = self.players[jersey_number]

        # Update stats based on event type
        event_name = event.event_type.value if hasattr(event.event_type, 'value') else str(event.event_type)

        if event_name == 'pass':
            player_data.passes_attempted += 1
            if event.success:
                player_data.passes_completed += 1
        elif event_name == 'shot':
            player_data.shots += 1
        elif event_name == 'goal':
            player_data.goals += 1
        elif event_name == 'tackle':
            player_data.tackles += 1
        elif event_name == 'interception':
            player_data.interceptions += 1

        # Calculate frame number from timestamp
        frame_number = int(event.timestamp_ms * self.fps / 1000)

        # Create moment with appropriate importance
        importance = self.IMPORTANCE_WEIGHTS.get(event_name, 1.0)

        # Boost importance for successful events
        if event.success:
            importance *= 1.5

        moment = PlayerMoment(
            frame_start=frame_number,
            frame_end=frame_number,
            timestamp_start_ms=event.timestamp_ms,
            timestamp_end_ms=event.timestamp_ms,
            event_type=event_name,
            importance=importance,
            description=f"{event_name.capitalize()}",
            position=event.position
        )

        player_data.moments.append(moment)

    def _merge_overlapping_moments(
        self,
        moments: List[PlayerMoment]
    ) -> List[PlayerMoment]:
        """Merge overlapping or close moments into single clips."""
        if not moments:
            return []

        # Sort by start frame
        sorted_moments = sorted(moments, key=lambda m: m.frame_start)

        merged = []
        current = sorted_moments[0]

        for next_moment in sorted_moments[1:]:
            # Check for overlap or small gap
            gap_frames = next_moment.frame_start - current.frame_end
            gap_seconds = gap_frames / self.fps

            if gap_seconds <= self.MERGE_GAP_THRESHOLD:
                # Merge: extend current moment
                current.frame_end = max(current.frame_end, next_moment.frame_end)
                current.timestamp_end_ms = max(
                    current.timestamp_end_ms,
                    next_moment.timestamp_end_ms
                )
                current.importance = max(current.importance, next_moment.importance)

                # Combine descriptions if different
                if next_moment.event_type and next_moment.event_type != current.event_type:
                    if current.description:
                        current.description += f", {next_moment.event_type}"
                    else:
                        current.description = next_moment.event_type
            else:
                # Gap too large, start new clip
                merged.append(current)
                current = next_moment

        merged.append(current)
        return merged

    async def generate_highlight_video(
        self,
        jersey_number: int,
        output_path: Optional[str] = None,
        max_duration_seconds: float = 120.0,
        min_importance: float = 0.5
    ) -> Optional[str]:
        """
        Generate a highlight video for a specific player.

        Args:
            jersey_number: Player's jersey number
            output_path: Output file path (auto-generated if None)
            max_duration_seconds: Maximum highlight reel duration
            min_importance: Minimum importance score for including a moment

        Returns:
            Path to generated video, or None if failed
        """
        if jersey_number not in self.players:
            print(f"No data for player #{jersey_number}")
            return None

        if not self.video_path or not os.path.exists(self.video_path):
            print(f"Source video not found: {self.video_path}")
            return None

        player_data = self.players[jersey_number]

        # Filter and merge moments
        filtered_moments = [
            m for m in player_data.moments
            if m.importance >= min_importance
        ]

        if not filtered_moments:
            print(f"No significant moments for player #{jersey_number}")
            return None

        merged_moments = self._merge_overlapping_moments(filtered_moments)

        # Sort by importance and limit to fit duration
        merged_moments.sort(key=lambda m: m.importance, reverse=True)

        # Calculate which moments to include
        selected_moments = []
        total_duration = 0.0

        for moment in merged_moments:
            # Add padding
            padded_start = max(0, moment.frame_start - int(self.CLIP_PADDING_BEFORE * self.fps))
            padded_end = min(
                self.total_frames,
                moment.frame_end + int(self.CLIP_PADDING_AFTER * self.fps)
            )

            clip_duration = (padded_end - padded_start) / self.fps

            # Enforce min/max clip duration (5-10 seconds per user requirement)
            if clip_duration < self.MIN_CLIP_DURATION:
                # Extend clip to meet minimum duration
                extra_frames_needed = int((self.MIN_CLIP_DURATION - clip_duration) * self.fps)
                extend_before = extra_frames_needed // 2
                extend_after = extra_frames_needed - extend_before
                padded_start = max(0, padded_start - extend_before)
                padded_end = min(self.total_frames, padded_end + extend_after)
                clip_duration = (padded_end - padded_start) / self.fps

            if clip_duration > self.MAX_CLIP_DURATION:
                # Trim clip to maximum duration, centered on the event
                max_frames = int(self.MAX_CLIP_DURATION * self.fps)
                event_center = (moment.frame_start + moment.frame_end) // 2
                padded_start = max(0, event_center - max_frames // 2)
                padded_end = min(self.total_frames, padded_start + max_frames)
                clip_duration = self.MAX_CLIP_DURATION

            if total_duration + clip_duration <= max_duration_seconds:
                selected_moments.append((padded_start, padded_end, moment))
                total_duration += clip_duration

        if not selected_moments:
            return None

        # Sort by time for chronological playback
        selected_moments.sort(key=lambda x: x[0])

        # Generate output path
        if output_path is None:
            os.makedirs(self.output_dir, exist_ok=True)
            player_name = player_data.player_name or f"player_{jersey_number}"
            safe_name = "".join(c if c.isalnum() else "_" for c in player_name)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(
                self.output_dir,
                f"{safe_name}_#{jersey_number}_{timestamp}.mp4"
            )

        # Open source video
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"Could not open video: {self.video_path}")
            return None

        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(
            output_path,
            fourcc,
            self.fps,
            (self.frame_width, self.frame_height)
        )

        # Write selected clips
        for start_frame, end_frame, moment in selected_moments:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            for frame_idx in range(start_frame, end_frame):
                ret, frame = cap.read()
                if not ret:
                    break

                # Optionally add overlay showing event type
                if moment.event_type:
                    self._add_overlay(frame, moment, frame_idx)

                out.write(frame)

        cap.release()
        out.release()

        print(f"Generated highlights for #{jersey_number}: {output_path}")
        print(f"  - {len(selected_moments)} clips, {total_duration:.1f}s total")

        return output_path

    def _add_overlay(
        self,
        frame: np.ndarray,
        moment: PlayerMoment,
        current_frame: int
    ):
        """Add event overlay to frame."""
        # Only show at start of moment
        if current_frame > moment.frame_start + int(1.5 * self.fps):
            return

        text = moment.description or moment.event_type or ""
        if not text:
            return

        # Draw text with background
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.2
        thickness = 2

        (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)

        x = 50
        y = self.frame_height - 100

        # Background rectangle
        cv2.rectangle(
            frame,
            (x - 10, y - text_h - 10),
            (x + text_w + 10, y + 10),
            (0, 0, 0),
            -1
        )

        # Text
        cv2.putText(
            frame,
            text.upper(),
            (x, y),
            font,
            font_scale,
            (255, 255, 255),
            thickness
        )

    async def generate_all_highlights(
        self,
        output_dir: Optional[str] = None,
        **kwargs
    ) -> Dict[int, str]:
        """
        Generate highlight videos for all tracked players.

        Returns:
            Dict mapping jersey_number to output video path
        """
        if output_dir:
            self.output_dir = output_dir

        results = {}

        for jersey_number in self.players:
            output_path = await self.generate_highlight_video(jersey_number, **kwargs)
            if output_path:
                results[jersey_number] = output_path

        return results

    def extract_individual_clip(
        self,
        jersey_number: int,
        moment: PlayerMoment,
        add_padding: bool = True
    ) -> Optional[ClipMetadata]:
        """
        Extract a single moment as an individual video clip file.

        Args:
            jersey_number: Player's jersey number
            moment: The moment to extract
            add_padding: Whether to add time padding before/after

        Returns:
            ClipMetadata for the extracted clip, or None if extraction failed
        """
        if not self.video_path or jersey_number not in self.players:
            return None

        player_data = self.players[jersey_number]

        # Calculate frame range with padding
        if add_padding:
            padding_before_frames = int(self.CLIP_PADDING_BEFORE * self.fps)
            padding_after_frames = int(self.CLIP_PADDING_AFTER * self.fps)
        else:
            padding_before_frames = 0
            padding_after_frames = 0

        start_frame = max(0, moment.frame_start - padding_before_frames)
        end_frame = min(self.total_frames, moment.frame_end + padding_after_frames)

        # Generate unique clip ID
        clip_id = str(uuid.uuid4())[:8]

        # Create clips directory structure: clips/{jersey_number}/
        player_clips_dir = os.path.join(self.clips_dir, f"player_{jersey_number}")
        os.makedirs(player_clips_dir, exist_ok=True)

        # Generate filename: {event_type}_{timestamp}_{clip_id}.mp4
        timestamp_sec = moment.timestamp_start_ms // 1000
        mins = timestamp_sec // 60
        secs = timestamp_sec % 60
        event_type = moment.event_type or "moment"
        clip_filename = f"{event_type}_{mins:02d}m{secs:02d}s_{clip_id}.mp4"
        clip_path = os.path.join(player_clips_dir, clip_filename)

        # Extract the clip
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"Could not open video: {self.video_path}")
            return None

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(
            clip_path,
            fourcc,
            self.fps,
            (self.frame_width, self.frame_height)
        )

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        for frame_idx in range(start_frame, end_frame):
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)

        cap.release()
        out.release()

        # Calculate duration
        duration_seconds = (end_frame - start_frame) / self.fps

        # Create metadata
        metadata = ClipMetadata(
            clip_id=clip_id,
            jersey_number=jersey_number,
            player_name=player_data.player_name,
            team=player_data.team.value,
            event_type=event_type,
            frame_start=start_frame,
            frame_end=end_frame,
            timestamp_start_ms=moment.timestamp_start_ms,
            timestamp_end_ms=moment.timestamp_end_ms,
            duration_seconds=duration_seconds,
            importance=moment.importance,
            clip_path=clip_path,
            extracted_at=datetime.now().isoformat()
        )

        # Update moment with clip info
        moment.clip_id = clip_id
        moment.clip_path = clip_path

        # Add to registry
        self.clips_registry[clip_id] = metadata

        return metadata

    async def extract_all_clips(
        self,
        min_importance: float = 0.0,
        event_types: Optional[List[str]] = None
    ) -> List[ClipMetadata]:
        """
        Extract individual clips for all moments across all players.

        Args:
            min_importance: Only extract clips with importance >= this value
            event_types: Only extract clips of these event types (None = all)

        Returns:
            List of ClipMetadata for all extracted clips
        """
        extracted_clips = []

        for jersey_number, player_data in self.players.items():
            for moment in player_data.moments:
                # Filter by importance
                if moment.importance < min_importance:
                    continue

                # Filter by event type
                if event_types and moment.event_type not in event_types:
                    continue

                # Skip if already extracted
                if moment.clip_path and os.path.exists(moment.clip_path):
                    continue

                # Extract the clip
                metadata = self.extract_individual_clip(jersey_number, moment)
                if metadata:
                    extracted_clips.append(metadata)

        # Save clips registry to JSON
        self._save_clips_registry()

        return extracted_clips

    async def extract_player_clips(
        self,
        jersey_number: int,
        min_importance: float = 0.0,
        event_types: Optional[List[str]] = None
    ) -> List[ClipMetadata]:
        """
        Extract all clips for a specific player.

        Args:
            jersey_number: Player's jersey number
            min_importance: Only extract clips with importance >= this value
            event_types: Only extract clips of these event types (None = all)

        Returns:
            List of ClipMetadata for extracted clips
        """
        if jersey_number not in self.players:
            return []

        extracted_clips = []
        player_data = self.players[jersey_number]

        for moment in player_data.moments:
            # Filter by importance
            if moment.importance < min_importance:
                continue

            # Filter by event type
            if event_types and moment.event_type not in event_types:
                continue

            # Skip if already extracted
            if moment.clip_path and os.path.exists(moment.clip_path):
                continue

            # Extract the clip
            metadata = self.extract_individual_clip(jersey_number, moment)
            if metadata:
                extracted_clips.append(metadata)

        # Save clips registry
        self._save_clips_registry()

        return extracted_clips

    def get_player_clips(
        self,
        jersey_number: int,
        event_type: Optional[str] = None
    ) -> List[ClipMetadata]:
        """
        Get all clips for a player (already extracted).

        Args:
            jersey_number: Player's jersey number
            event_type: Filter by event type (None = all)

        Returns:
            List of ClipMetadata for the player's clips
        """
        clips = [
            clip for clip in self.clips_registry.values()
            if clip.jersey_number == jersey_number
        ]

        if event_type:
            clips = [c for c in clips if c.event_type == event_type]

        # Sort by timestamp
        clips.sort(key=lambda c: c.timestamp_start_ms)

        return clips

    def get_all_clips(self, event_type: Optional[str] = None) -> List[ClipMetadata]:
        """
        Get all extracted clips.

        Args:
            event_type: Filter by event type (None = all)

        Returns:
            List of all ClipMetadata
        """
        clips = list(self.clips_registry.values())

        if event_type:
            clips = [c for c in clips if c.event_type == event_type]

        # Sort by timestamp
        clips.sort(key=lambda c: c.timestamp_start_ms)

        return clips

    def _save_clips_registry(self):
        """Save clips registry to JSON file."""
        os.makedirs(self.clips_dir, exist_ok=True)
        registry_path = os.path.join(self.clips_dir, "clips_registry.json")

        data = {
            "generated_at": datetime.now().isoformat(),
            "video_source": self.video_path,
            "total_clips": len(self.clips_registry),
            "clips": [asdict(clip) for clip in self.clips_registry.values()]
        }

        with open(registry_path, 'w') as f:
            json.dump(data, f, indent=2)

    def _load_clips_registry(self):
        """Load clips registry from JSON file."""
        registry_path = os.path.join(self.clips_dir, "clips_registry.json")

        if not os.path.exists(registry_path):
            return

        try:
            with open(registry_path, 'r') as f:
                data = json.load(f)

            for clip_data in data.get("clips", []):
                clip = ClipMetadata(**clip_data)
                self.clips_registry[clip.clip_id] = clip
        except Exception as e:
            print(f"Error loading clips registry: {e}")

    def get_player_summary(self, jersey_number: int) -> Optional[Dict]:
        """Get a summary of a player's involvement."""
        if jersey_number not in self.players:
            return None

        data = self.players[jersey_number]

        return {
            "jersey_number": data.jersey_number,
            "player_name": data.player_name,
            "team": data.team.value,
            "stats": {
                "touches": data.touches,
                "passes_attempted": data.passes_attempted,
                "passes_completed": data.passes_completed,
                "pass_accuracy": (
                    data.passes_completed / data.passes_attempted * 100
                    if data.passes_attempted > 0 else 0
                ),
                "shots": data.shots,
                "goals": data.goals,
                "tackles": data.tackles,
                "interceptions": data.interceptions,
            },
            "moments_count": len(data.moments),
            "highlight_duration_estimate": sum(
                (m.frame_end - m.frame_start) / self.fps
                for m in data.moments
            )
        }

    def get_all_summaries(self) -> List[Dict]:
        """Get summaries for all tracked players."""
        return [
            self.get_player_summary(jersey)
            for jersey in sorted(self.players.keys())
            if self.get_player_summary(jersey) is not None
        ]

    def export_data(self, output_path: str):
        """Export all player highlight data to JSON."""
        data = {
            "generated_at": datetime.now().isoformat(),
            "video_source": self.video_path,
            "fps": self.fps,
            "total_frames": self.total_frames,
            "players": self.get_all_summaries()
        }

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

        return output_path

    def reset(self):
        """Reset all tracking data."""
        self.players.clear()
        self.track_to_jersey.clear()
        self.last_touch_frame.clear()
        self.clips_registry.clear()
        self.current_frame = 0


# Global instance
player_highlights_service = PlayerHighlightsService()
