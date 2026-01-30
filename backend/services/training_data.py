"""
Training Data Collection Service

Manages collection, storage, and export of annotated match data
for training custom ML models.

Key features:
- Automatic frame capture during video processing
- Manual annotation interface support
- YOLO format export for training
- COCO format export for validation
- Dataset versioning and statistics
"""
import json
import uuid
import asyncio
import shutil
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict, field
import aiofiles
import csv
import random

from config import settings


# Detection class mappings for YOLO
DETECTION_CLASSES = {
    "player": 0,
    "ball": 1,
    "goalkeeper": 2,
    "referee": 3,
    "home_player": 4,
    "away_player": 5,
}

CLASS_NAMES = {v: k for k, v in DETECTION_CLASSES.items()}


@dataclass
class MatchAnnotation:
    """Annotation for a single match."""
    match_id: str
    video_path: str
    home_team: str
    away_team: str
    final_score: Dict[str, int]
    date: str
    competition: Optional[str] = None
    venue: Optional[str] = None

    # Match-level stats
    home_possession: Optional[float] = None
    away_possession: Optional[float] = None
    home_shots: Optional[int] = None
    away_shots: Optional[int] = None
    home_shots_on_target: Optional[int] = None
    away_shots_on_target: Optional[int] = None
    home_passes: Optional[int] = None
    away_passes: Optional[int] = None
    home_pass_accuracy: Optional[float] = None
    away_pass_accuracy: Optional[float] = None
    home_corners: Optional[int] = None
    away_corners: Optional[int] = None
    home_fouls: Optional[int] = None
    away_fouls: Optional[int] = None
    home_yellow_cards: Optional[int] = None
    away_yellow_cards: Optional[int] = None
    home_red_cards: Optional[int] = None
    away_red_cards: Optional[int] = None
    home_xg: Optional[float] = None
    away_xg: Optional[float] = None


@dataclass
class PlayerAnnotation:
    """Annotation for a player in a match."""
    player_id: str
    match_id: str
    name: str
    team: str  # 'home' or 'away'
    position: str  # GK, CB, LB, RB, CM, CAM, LW, RW, ST, etc.
    jersey_number: Optional[int] = None

    # Player stats
    minutes_played: Optional[int] = None
    goals: Optional[int] = None
    assists: Optional[int] = None
    shots: Optional[int] = None
    shots_on_target: Optional[int] = None
    passes: Optional[int] = None
    pass_accuracy: Optional[float] = None
    key_passes: Optional[int] = None
    crosses: Optional[int] = None
    dribbles_attempted: Optional[int] = None
    dribbles_successful: Optional[int] = None
    tackles: Optional[int] = None
    interceptions: Optional[int] = None
    clearances: Optional[int] = None
    blocks: Optional[int] = None
    aerial_duels_won: Optional[int] = None
    aerial_duels_lost: Optional[int] = None
    fouls_committed: Optional[int] = None
    fouls_won: Optional[int] = None
    yellow_cards: Optional[int] = None
    red_cards: Optional[int] = None
    distance_covered_km: Optional[float] = None
    sprints: Optional[int] = None
    top_speed_kmh: Optional[float] = None
    rating: Optional[float] = None  # 1-10 rating


@dataclass
class EventAnnotation:
    """Annotation for a specific event in a match."""
    event_id: str
    match_id: str
    timestamp_seconds: float
    event_type: str  # goal, shot, pass, tackle, foul, card, substitution, etc.
    team: str
    player_id: Optional[str] = None
    player_name: Optional[str] = None

    # Position data
    x_position: Optional[float] = None  # 0-100 (percentage of pitch)
    y_position: Optional[float] = None  # 0-100 (percentage of pitch)
    end_x: Optional[float] = None
    end_y: Optional[float] = None

    # Event-specific data
    outcome: Optional[str] = None  # success, fail, blocked, saved, etc.
    body_part: Optional[str] = None  # foot, head, etc.
    assist_player_id: Optional[str] = None
    recipient_player_id: Optional[str] = None
    xg_value: Optional[float] = None
    notes: Optional[str] = None


@dataclass
class FrameAnnotation:
    """Annotation for a video frame (for detection training)."""
    frame_id: str
    match_id: str
    frame_number: int
    timestamp_seconds: float
    image_path: str

    # Bounding boxes: list of {class, x_center, y_center, width, height} (YOLO format)
    annotations: List[Dict[str, Any]]


class TrainingDataService:
    """
    Service for collecting and managing training data.

    Supports:
    - Match-level statistics upload
    - Player-level statistics upload
    - Event-by-event annotation
    - Frame-level bounding box annotation (for YOLO training)
    - Export to various formats (JSON, CSV, YOLO)
    """

    def __init__(self):
        self.data_dir = settings.DATA_DIR / "training"
        self.matches_dir = self.data_dir / "matches"
        self.annotations_dir = self.data_dir / "annotations"
        self.frames_dir = self.data_dir / "frames"
        self.exports_dir = self.data_dir / "exports"

        # Ensure directories exist
        for dir_path in [self.data_dir, self.matches_dir, self.annotations_dir,
                         self.frames_dir, self.exports_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # In-memory index
        self.matches: Dict[str, MatchAnnotation] = {}
        self.players: Dict[str, List[PlayerAnnotation]] = {}  # match_id -> players
        self.events: Dict[str, List[EventAnnotation]] = {}  # match_id -> events
        self.frames: Dict[str, List[FrameAnnotation]] = {}  # match_id -> frames

    async def initialize(self):
        """Load existing data from disk."""
        await self._load_index()

    async def _load_index(self):
        """Load match index from disk."""
        index_path = self.data_dir / "index.json"
        if index_path.exists():
            async with aiofiles.open(index_path, 'r') as f:
                content = await f.read()
                data = json.loads(content)
                # Reconstruct match objects
                for match_data in data.get("matches", []):
                    match = MatchAnnotation(**match_data)
                    self.matches[match.match_id] = match

    async def _save_index(self):
        """Save match index to disk."""
        index_path = self.data_dir / "index.json"
        data = {
            "matches": [asdict(m) for m in self.matches.values()],
            "updated_at": datetime.now().isoformat()
        }
        async with aiofiles.open(index_path, 'w') as f:
            await f.write(json.dumps(data, indent=2))

    # ==================== Match Operations ====================

    async def add_match(self, match: MatchAnnotation) -> str:
        """Add a new match to the training dataset."""
        if not match.match_id:
            match.match_id = str(uuid.uuid4())

        self.matches[match.match_id] = match
        self.players[match.match_id] = []
        self.events[match.match_id] = []
        self.frames[match.match_id] = []

        # Save match data
        match_path = self.matches_dir / f"{match.match_id}.json"
        async with aiofiles.open(match_path, 'w') as f:
            await f.write(json.dumps(asdict(match), indent=2))

        await self._save_index()
        return match.match_id

    async def get_match(self, match_id: str) -> Optional[MatchAnnotation]:
        """Get a match by ID."""
        return self.matches.get(match_id)

    async def list_matches(self) -> List[MatchAnnotation]:
        """List all matches in the training dataset."""
        return list(self.matches.values())

    async def update_match_stats(self, match_id: str, stats: Dict[str, Any]) -> bool:
        """Update match statistics."""
        if match_id not in self.matches:
            return False

        match = self.matches[match_id]
        for key, value in stats.items():
            if hasattr(match, key):
                setattr(match, key, value)

        # Save updated match
        match_path = self.matches_dir / f"{match_id}.json"
        async with aiofiles.open(match_path, 'w') as f:
            await f.write(json.dumps(asdict(match), indent=2))

        await self._save_index()
        return True

    # ==================== Player Operations ====================

    async def add_player(self, player: PlayerAnnotation) -> str:
        """Add a player annotation to a match."""
        if not player.player_id:
            player.player_id = str(uuid.uuid4())

        if player.match_id not in self.players:
            self.players[player.match_id] = []

        self.players[player.match_id].append(player)

        # Save players for match
        await self._save_players(player.match_id)
        return player.player_id

    async def add_players_bulk(self, match_id: str, players: List[Dict]) -> int:
        """Add multiple players at once."""
        count = 0
        for player_data in players:
            player_data["match_id"] = match_id
            if "player_id" not in player_data:
                player_data["player_id"] = str(uuid.uuid4())
            player = PlayerAnnotation(**player_data)

            if match_id not in self.players:
                self.players[match_id] = []
            self.players[match_id].append(player)
            count += 1

        await self._save_players(match_id)
        return count

    async def get_players(self, match_id: str) -> List[PlayerAnnotation]:
        """Get all players for a match."""
        return self.players.get(match_id, [])

    async def _save_players(self, match_id: str):
        """Save players for a match to disk."""
        players_path = self.annotations_dir / f"{match_id}_players.json"
        data = [asdict(p) for p in self.players.get(match_id, [])]
        async with aiofiles.open(players_path, 'w') as f:
            await f.write(json.dumps(data, indent=2))

    # ==================== Event Operations ====================

    async def add_event(self, event: EventAnnotation) -> str:
        """Add an event annotation to a match."""
        if not event.event_id:
            event.event_id = str(uuid.uuid4())

        if event.match_id not in self.events:
            self.events[event.match_id] = []

        self.events[event.match_id].append(event)

        # Keep events sorted by timestamp
        self.events[event.match_id].sort(key=lambda e: e.timestamp_seconds)

        await self._save_events(event.match_id)
        return event.event_id

    async def add_events_bulk(self, match_id: str, events: List[Dict]) -> int:
        """Add multiple events at once."""
        count = 0
        for event_data in events:
            event_data["match_id"] = match_id
            if "event_id" not in event_data:
                event_data["event_id"] = str(uuid.uuid4())
            event = EventAnnotation(**event_data)

            if match_id not in self.events:
                self.events[match_id] = []
            self.events[match_id].append(event)
            count += 1

        # Sort by timestamp
        self.events[match_id].sort(key=lambda e: e.timestamp_seconds)
        await self._save_events(match_id)
        return count

    async def get_events(self, match_id: str) -> List[EventAnnotation]:
        """Get all events for a match."""
        return self.events.get(match_id, [])

    async def _save_events(self, match_id: str):
        """Save events for a match to disk."""
        events_path = self.annotations_dir / f"{match_id}_events.json"
        data = [asdict(e) for e in self.events.get(match_id, [])]
        async with aiofiles.open(events_path, 'w') as f:
            await f.write(json.dumps(data, indent=2))

    # ==================== Frame Annotation Operations ====================

    async def add_frame_annotation(self, frame: FrameAnnotation) -> str:
        """Add a frame annotation for detection training."""
        if not frame.frame_id:
            frame.frame_id = str(uuid.uuid4())

        if frame.match_id not in self.frames:
            self.frames[frame.match_id] = []

        self.frames[frame.match_id].append(frame)
        await self._save_frames(frame.match_id)
        return frame.frame_id

    async def _save_frames(self, match_id: str):
        """Save frame annotations to disk."""
        frames_path = self.annotations_dir / f"{match_id}_frames.json"
        data = [asdict(f) for f in self.frames.get(match_id, [])]
        async with aiofiles.open(frames_path, 'w') as f:
            await f.write(json.dumps(data, indent=2))

    # ==================== Export Operations ====================

    async def export_to_json(self, output_name: str = "training_data") -> str:
        """Export all training data to JSON."""
        output_path = self.exports_dir / f"{output_name}.json"

        data = {
            "metadata": {
                "exported_at": datetime.now().isoformat(),
                "total_matches": len(self.matches),
                "total_players": sum(len(p) for p in self.players.values()),
                "total_events": sum(len(e) for e in self.events.values()),
            },
            "matches": [asdict(m) for m in self.matches.values()],
            "players": {mid: [asdict(p) for p in players]
                       for mid, players in self.players.items()},
            "events": {mid: [asdict(e) for e in events]
                      for mid, events in self.events.items()},
        }

        async with aiofiles.open(output_path, 'w') as f:
            await f.write(json.dumps(data, indent=2))

        return str(output_path)

    async def export_matches_csv(self, output_name: str = "matches") -> str:
        """Export match data to CSV."""
        output_path = self.exports_dir / f"{output_name}.csv"

        if not self.matches:
            return str(output_path)

        # Get all fields from first match
        fieldnames = list(asdict(list(self.matches.values())[0]).keys())

        async with aiofiles.open(output_path, 'w', newline='') as f:
            # Write CSV manually since aiofiles doesn't support csv writer directly
            header = ','.join(fieldnames)
            await f.write(header + '\n')

            for match in self.matches.values():
                row = ','.join(str(v) if v is not None else ''
                              for v in asdict(match).values())
                await f.write(row + '\n')

        return str(output_path)

    async def export_players_csv(self, output_name: str = "players") -> str:
        """Export player data to CSV."""
        output_path = self.exports_dir / f"{output_name}.csv"

        all_players = []
        for players in self.players.values():
            all_players.extend(players)

        if not all_players:
            return str(output_path)

        fieldnames = list(asdict(all_players[0]).keys())

        async with aiofiles.open(output_path, 'w', newline='') as f:
            header = ','.join(fieldnames)
            await f.write(header + '\n')

            for player in all_players:
                row = ','.join(str(v) if v is not None else ''
                              for v in asdict(player).values())
                await f.write(row + '\n')

        return str(output_path)

    async def export_events_csv(self, output_name: str = "events") -> str:
        """Export event data to CSV."""
        output_path = self.exports_dir / f"{output_name}.csv"

        all_events = []
        for events in self.events.values():
            all_events.extend(events)

        if not all_events:
            return str(output_path)

        fieldnames = list(asdict(all_events[0]).keys())

        async with aiofiles.open(output_path, 'w', newline='') as f:
            header = ','.join(fieldnames)
            await f.write(header + '\n')

            for event in all_events:
                row = ','.join(str(v) if v is not None else ''
                              for v in asdict(event).values())
                await f.write(row + '\n')

        return str(output_path)

    async def export_yolo_format(self, output_name: str = "yolo_dataset") -> str:
        """Export frame annotations in YOLO format for training."""
        output_dir = self.exports_dir / output_name
        images_dir = output_dir / "images"
        labels_dir = output_dir / "labels"

        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        # Create dataset.yaml
        yaml_content = f"""
path: {output_dir}
train: images
val: images

names:
  0: player
  1: ball
  2: goalkeeper
  3: referee
"""
        async with aiofiles.open(output_dir / "dataset.yaml", 'w') as f:
            await f.write(yaml_content)

        # Export frame annotations
        for match_id, frames in self.frames.items():
            for frame in frames:
                # Copy image (if exists)
                # Create YOLO label file
                label_path = labels_dir / f"{frame.frame_id}.txt"
                async with aiofiles.open(label_path, 'w') as f:
                    for ann in frame.annotations:
                        # YOLO format: class x_center y_center width height
                        line = f"{ann['class']} {ann['x_center']} {ann['y_center']} {ann['width']} {ann['height']}\n"
                        await f.write(line)

        return str(output_dir)

    # ==================== Statistics ====================

    async def get_dataset_stats(self) -> Dict[str, Any]:
        """Get statistics about the training dataset."""
        total_goals = 0
        total_shots = 0
        total_passes = 0

        for events in self.events.values():
            for event in events:
                if event.event_type == 'goal':
                    total_goals += 1
                elif event.event_type == 'shot':
                    total_shots += 1
                elif event.event_type == 'pass':
                    total_passes += 1

        return {
            "total_matches": len(self.matches),
            "total_players": sum(len(p) for p in self.players.values()),
            "total_events": sum(len(e) for e in self.events.values()),
            "total_frames": sum(len(f) for f in self.frames.values()),
            "event_breakdown": {
                "goals": total_goals,
                "shots": total_shots,
                "passes": total_passes,
            },
            "matches_by_competition": self._count_by_field("competition"),
        }

    def _count_by_field(self, field: str) -> Dict[str, int]:
        """Count matches by a specific field."""
        counts = {}
        for match in self.matches.values():
            value = getattr(match, field, None) or "Unknown"
            counts[value] = counts.get(value, 0) + 1
        return counts

    # ==================== Frame Capture During Processing ====================

    async def capture_frame_for_training(
        self,
        video_id: str,
        frame_number: int,
        timestamp_ms: int,
        frame_image,  # numpy array
        detections: List[Dict],
        auto_annotate: bool = True
    ) -> Optional[str]:
        """
        Capture a frame during video processing for training data.

        Args:
            video_id: ID of the video being processed
            frame_number: Frame number in video
            timestamp_ms: Timestamp in milliseconds
            frame_image: numpy array of the frame (BGR)
            detections: List of detections from the model
            auto_annotate: Whether to auto-generate annotations from detections

        Returns:
            frame_id if saved, None if skipped
        """
        import cv2

        # Create frame storage directory
        video_frames_dir = self.frames_dir / video_id
        video_frames_dir.mkdir(parents=True, exist_ok=True)

        # Generate frame ID
        frame_id = f"{video_id}_{frame_number:06d}"

        # Save frame image
        image_path = video_frames_dir / f"{frame_id}.jpg"
        cv2.imwrite(str(image_path), frame_image, [cv2.IMWRITE_JPEG_QUALITY, 95])

        # Generate YOLO-format annotations if auto_annotate
        annotations = []
        if auto_annotate and detections:
            img_height, img_width = frame_image.shape[:2]

            for det in detections:
                bbox = det.get('bbox', {})
                if isinstance(bbox, dict):
                    x1, y1 = bbox.get('x1', 0), bbox.get('y1', 0)
                    x2, y2 = bbox.get('x2', 0), bbox.get('y2', 0)
                elif isinstance(bbox, list) and len(bbox) >= 4:
                    x1, y1, x2, y2 = bbox[:4]
                else:
                    continue

                # Convert to YOLO format (normalized center x, y, width, height)
                x_center = ((x1 + x2) / 2) / img_width
                y_center = ((y1 + y2) / 2) / img_height
                width = (x2 - x1) / img_width
                height = (y2 - y1) / img_height

                # Determine class
                team = det.get('team', 'unknown')
                is_goalkeeper = det.get('is_goalkeeper', False)

                if is_goalkeeper:
                    class_id = DETECTION_CLASSES['goalkeeper']
                elif team == 'home':
                    class_id = DETECTION_CLASSES['home_player']
                elif team == 'away':
                    class_id = DETECTION_CLASSES['away_player']
                else:
                    class_id = DETECTION_CLASSES['player']

                annotations.append({
                    'class': class_id,
                    'class_name': CLASS_NAMES[class_id],
                    'x_center': round(x_center, 6),
                    'y_center': round(y_center, 6),
                    'width': round(width, 6),
                    'height': round(height, 6),
                    'confidence': det.get('confidence', 0.0),
                    'track_id': det.get('track_id'),
                    'team': team,
                    'jersey_number': det.get('jersey_number'),
                })

        # Create frame annotation
        frame_annotation = FrameAnnotation(
            frame_id=frame_id,
            match_id=video_id,
            frame_number=frame_number,
            timestamp_seconds=timestamp_ms / 1000.0,
            image_path=str(image_path),
            annotations=annotations
        )

        # Save annotation
        if video_id not in self.frames:
            self.frames[video_id] = []
        self.frames[video_id].append(frame_annotation)

        # Save YOLO label file alongside image
        label_path = video_frames_dir / f"{frame_id}.txt"
        async with aiofiles.open(label_path, 'w') as f:
            for ann in annotations:
                line = f"{ann['class']} {ann['x_center']} {ann['y_center']} {ann['width']} {ann['height']}\n"
                await f.write(line)

        return frame_id

    async def update_frame_annotation(
        self,
        frame_id: str,
        annotations: List[Dict]
    ) -> bool:
        """
        Update annotations for a frame (manual correction).

        Args:
            frame_id: ID of the frame to update
            annotations: New list of annotations in YOLO format

        Returns:
            True if updated successfully
        """
        # Find the frame
        for video_id, frames in self.frames.items():
            for frame in frames:
                if frame.frame_id == frame_id:
                    # Update annotations
                    frame.annotations = annotations

                    # Update label file
                    video_frames_dir = self.frames_dir / video_id
                    label_path = video_frames_dir / f"{frame_id}.txt"

                    async with aiofiles.open(label_path, 'w') as f:
                        for ann in annotations:
                            line = f"{ann['class']} {ann['x_center']} {ann['y_center']} {ann['width']} {ann['height']}\n"
                            await f.write(line)

                    await self._save_frames(video_id)
                    return True

        return False

    async def get_frame_for_annotation(self, frame_id: str) -> Optional[Dict]:
        """Get frame data for annotation UI."""
        for video_id, frames in self.frames.items():
            for frame in frames:
                if frame.frame_id == frame_id:
                    return {
                        'frame_id': frame.frame_id,
                        'video_id': video_id,
                        'frame_number': frame.frame_number,
                        'timestamp_seconds': frame.timestamp_seconds,
                        'image_url': f"/api/training/frame/{frame_id}/image",
                        'annotations': frame.annotations,
                    }
        return None

    async def list_frames_for_video(self, video_id: str) -> List[Dict]:
        """List all captured frames for a video."""
        frames = self.frames.get(video_id, [])
        return [
            {
                'frame_id': f.frame_id,
                'frame_number': f.frame_number,
                'timestamp_seconds': f.timestamp_seconds,
                'annotation_count': len(f.annotations),
                'image_url': f"/api/training/frame/{f.frame_id}/image",
            }
            for f in frames
        ]

    async def export_yolo_dataset(
        self,
        output_name: str = "football_detection",
        train_split: float = 0.8,
        include_videos: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Export annotated frames as a YOLO training dataset.

        Args:
            output_name: Name for the dataset
            train_split: Fraction of data for training (rest for validation)
            include_videos: List of video IDs to include (None = all)

        Returns:
            Dict with export statistics
        """
        import shutil

        output_dir = self.exports_dir / output_name
        train_images = output_dir / "images" / "train"
        train_labels = output_dir / "labels" / "train"
        val_images = output_dir / "images" / "val"
        val_labels = output_dir / "labels" / "val"

        # Create directories
        for d in [train_images, train_labels, val_images, val_labels]:
            d.mkdir(parents=True, exist_ok=True)

        # Collect all frames
        all_frames = []
        for video_id, frames in self.frames.items():
            if include_videos is None or video_id in include_videos:
                all_frames.extend(frames)

        # Shuffle and split
        random.shuffle(all_frames)
        split_idx = int(len(all_frames) * train_split)
        train_frames = all_frames[:split_idx]
        val_frames = all_frames[split_idx:]

        # Export frames
        stats = {'train': 0, 'val': 0, 'total_annotations': 0}

        for frame in train_frames:
            if await self._export_frame(frame, train_images, train_labels):
                stats['train'] += 1
                stats['total_annotations'] += len(frame.annotations)

        for frame in val_frames:
            if await self._export_frame(frame, val_images, val_labels):
                stats['val'] += 1
                stats['total_annotations'] += len(frame.annotations)

        # Create dataset.yaml
        yaml_content = f"""# Football Detection Dataset
# Generated: {datetime.now().isoformat()}
# Total frames: {stats['train'] + stats['val']}
# Train: {stats['train']}, Val: {stats['val']}

path: {output_dir}
train: images/train
val: images/val

# Classes
names:
  0: player
  1: ball
  2: goalkeeper
  3: referee
  4: home_player
  5: away_player

# Recommended training command:
# yolo detect train data=dataset.yaml model=yolov8n.pt epochs=100 imgsz=1280
"""
        async with aiofiles.open(output_dir / "dataset.yaml", 'w') as f:
            await f.write(yaml_content)

        stats['output_path'] = str(output_dir)
        return stats

    async def _export_frame(self, frame: FrameAnnotation, images_dir: Path, labels_dir: Path) -> bool:
        """Export a single frame to YOLO format."""
        import shutil
        try:
            src_image = Path(frame.image_path)
            if src_image.exists():
                dst_image = images_dir / f"{frame.frame_id}.jpg"
                shutil.copy2(src_image, dst_image)

                label_path = labels_dir / f"{frame.frame_id}.txt"
                async with aiofiles.open(label_path, 'w') as f:
                    for ann in frame.annotations:
                        line = f"{ann['class']} {ann['x_center']} {ann['y_center']} {ann['width']} {ann['height']}\n"
                        await f.write(line)

                return True
        except Exception as e:
            print(f"Error exporting frame {frame.frame_id}: {e}")
        return False


# Global instance
training_data_service = TrainingDataService()
