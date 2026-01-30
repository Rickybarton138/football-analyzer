"""
Player Clip Analyzer Service

Analyzes individual player highlight clips (exported from VEO) to extract
player-specific statistics like passes, shots, tackles, and ball touches.
"""
import cv2
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from pathlib import Path
from collections import deque
import json
import os
from datetime import datetime


@dataclass
class PlayerEvent:
    """A detected event involving the player."""
    event_type: str  # 'touch', 'pass', 'shot', 'tackle', 'header', 'dribble'
    timestamp: float
    frame_number: int
    position: List[float]  # [x, y] on pitch
    success: Optional[bool] = None  # For passes/shots/tackles
    details: Dict = field(default_factory=dict)


@dataclass
class ClipAnalysis:
    """Analysis result for a single player clip."""
    clip_path: str
    duration_seconds: float
    player_visible_frames: int
    total_frames: int
    events: List[PlayerEvent] = field(default_factory=list)
    ball_touches: int = 0
    distance_covered_pixels: float = 0
    avg_position: List[float] = field(default_factory=list)
    heatmap_data: List[List[float]] = field(default_factory=list)


@dataclass
class PlayerStats:
    """Aggregated stats for a player across all clips."""
    player_name: str
    total_clips: int = 0
    total_play_time: float = 0  # seconds

    # Ball events
    ball_touches: int = 0
    passes_attempted: int = 0
    passes_completed: int = 0
    shots: int = 0
    shots_on_target: int = 0

    # Defensive events
    tackles_attempted: int = 0
    tackles_won: int = 0
    headers: int = 0
    interceptions: int = 0

    # Movement
    total_distance_pixels: float = 0
    sprints: int = 0

    # Position
    avg_position: List[float] = field(default_factory=list)
    heatmap_zones: Dict[str, int] = field(default_factory=dict)

    # Derived stats
    @property
    def pass_accuracy(self) -> float:
        if self.passes_attempted == 0:
            return 0
        return (self.passes_completed / self.passes_attempted) * 100

    @property
    def shot_accuracy(self) -> float:
        if self.shots == 0:
            return 0
        return (self.shots_on_target / self.shots) * 100

    @property
    def tackle_success_rate(self) -> float:
        if self.tackles_attempted == 0:
            return 0
        return (self.tackles_won / self.tackles_attempted) * 100


class PlayerClipAnalyzer:
    """Analyzes player highlight clips to extract individual statistics."""

    def __init__(self):
        self.model = None
        self.player_stats: Dict[str, PlayerStats] = {}
        self.clip_analyses: List[ClipAnalysis] = []

        # Tracking state
        self.prev_frame = None
        self.prev_ball_pos = None
        self.prev_player_pos = None
        self.player_positions = deque(maxlen=30)
        self.ball_positions = deque(maxlen=10)

        # Detection parameters
        self.VIDEO_WIDTH = 1920
        self.VIDEO_HEIGHT = 1080

    def _load_model(self):
        """Load YOLO model."""
        if self.model is None:
            try:
                from ultralytics import YOLO
                self.model = YOLO('yolov8n.pt')
                self.model.to('cpu')
                print("YOLOv8 loaded for player clip analysis")
            except ImportError:
                print("ultralytics not installed")
                self.model = None
        return self.model

    def analyze_clip(self, clip_path: str, player_name: str = "Player") -> ClipAnalysis:
        """
        Analyze a single player highlight clip.

        In VEO clips, the tagged player is typically:
        - Centered or prominent in frame
        - The largest/closest detected person
        - Consistently tracked throughout
        """
        model = self._load_model()

        cap = cv2.VideoCapture(clip_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open clip: {clip_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0

        analysis = ClipAnalysis(
            clip_path=clip_path,
            duration_seconds=duration,
            player_visible_frames=0,
            total_frames=total_frames
        )

        # Reset tracking state
        self.prev_frame = None
        self.prev_ball_pos = None
        self.prev_player_pos = None
        self.player_positions.clear()
        self.ball_positions.clear()

        frame_count = 0
        all_player_positions = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            timestamp = frame_count / fps if fps > 0 else 0

            # Detect people and ball
            player_box = None
            ball_pos = None

            if model is not None:
                results = model(frame, verbose=False)

                persons = []
                for r in results:
                    for box in r.boxes:
                        cls = int(box.cls[0])
                        conf = float(box.conf[0])
                        bbox = box.xyxy[0].tolist()

                        if cls == 0 and conf > 0.3:  # Person
                            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                            center_x = (bbox[0] + bbox[2]) / 2
                            center_y = (bbox[1] + bbox[3]) / 2
                            # Prefer center of frame (VEO focuses on tagged player)
                            dist_from_center = abs(center_x - self.VIDEO_WIDTH/2)
                            persons.append({
                                'bbox': bbox,
                                'area': area,
                                'center': [center_x, center_y],
                                'dist_from_center': dist_from_center
                            })

                        elif cls == 32 and conf > 0.15:  # Ball
                            ball_pos = [
                                (bbox[0] + bbox[2]) / 2,
                                (bbox[1] + bbox[3]) / 2
                            ]

                # Select the tagged player (largest person near center)
                if persons:
                    # Score by size and proximity to center
                    for p in persons:
                        p['score'] = p['area'] * 0.7 + (1 - p['dist_from_center'] / self.VIDEO_WIDTH) * 0.3 * 10000
                    persons.sort(key=lambda x: x['score'], reverse=True)
                    player_box = persons[0]

            # Track player
            if player_box:
                analysis.player_visible_frames += 1
                player_pos = player_box['center']
                all_player_positions.append(player_pos)
                self.player_positions.append(player_pos)

                # Calculate distance covered
                if self.prev_player_pos:
                    dist = np.sqrt(
                        (player_pos[0] - self.prev_player_pos[0])**2 +
                        (player_pos[1] - self.prev_player_pos[1])**2
                    )
                    analysis.distance_covered_pixels += dist

                    # Detect sprint (fast movement)
                    speed = dist * fps if fps > 0 else 0
                    if speed > 200:  # pixels per second threshold
                        # Could be a sprint
                        pass

                self.prev_player_pos = player_pos

            # Track ball and detect events
            if ball_pos:
                self.ball_positions.append(ball_pos)

                # Check for ball touch (ball near player)
                if player_box:
                    player_pos = player_box['center']
                    ball_dist = np.sqrt(
                        (ball_pos[0] - player_pos[0])**2 +
                        (ball_pos[1] - player_pos[1])**2
                    )

                    # Ball within ~50 pixels of player center = touch
                    if ball_dist < 80:
                        # Detect event type based on ball movement
                        event = self._detect_event(
                            frame, player_pos, ball_pos, timestamp, frame_count
                        )
                        if event:
                            analysis.events.append(event)
                            analysis.ball_touches += 1

                self.prev_ball_pos = ball_pos

            self.prev_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_count += 1

        cap.release()

        # Calculate average position
        if all_player_positions:
            analysis.avg_position = [
                np.mean([p[0] for p in all_player_positions]),
                np.mean([p[1] for p in all_player_positions])
            ]
            analysis.heatmap_data = all_player_positions

        # Store analysis
        self.clip_analyses.append(analysis)

        # Update player stats
        self._update_player_stats(player_name, analysis)

        return analysis

    def _detect_event(self, frame, player_pos, ball_pos, timestamp, frame_num) -> Optional[PlayerEvent]:
        """Detect what type of event is happening."""

        if len(self.ball_positions) < 2:
            return PlayerEvent(
                event_type='touch',
                timestamp=timestamp,
                frame_number=frame_num,
                position=ball_pos
            )

        # Calculate ball velocity
        prev_ball = self.ball_positions[-2]
        dx = ball_pos[0] - prev_ball[0]
        dy = ball_pos[1] - prev_ball[1]
        ball_speed = np.sqrt(dx*dx + dy*dy)

        # Determine event type based on ball movement
        event_type = 'touch'
        success = None
        details = {}

        # Shot detection: ball moving fast toward goal
        goal_left = self.VIDEO_WIDTH * 0.1
        goal_right = self.VIDEO_WIDTH * 0.9

        if ball_speed > 100:
            if (ball_pos[0] < goal_left and dx < -50) or (ball_pos[0] > goal_right and dx > 50):
                event_type = 'shot'
                # On target if ball is heading toward goal area vertically
                on_target = self.VIDEO_HEIGHT * 0.3 < ball_pos[1] < self.VIDEO_HEIGHT * 0.7
                success = on_target
                details['on_target'] = on_target
            else:
                # Fast ball movement away from goals = pass
                event_type = 'pass'
                # Success determined by whether ball reaches teammate (simplified)
                success = True  # Would need more context to determine
                details['direction'] = 'forward' if abs(dx) > abs(dy) else 'lateral'

        # Header detection: ball and player at similar height, ball moving up
        if dy < -30 and ball_pos[1] < self.VIDEO_HEIGHT * 0.4:
            event_type = 'header'

        return PlayerEvent(
            event_type=event_type,
            timestamp=timestamp,
            frame_number=frame_num,
            position=ball_pos,
            success=success,
            details=details
        )

    def _update_player_stats(self, player_name: str, analysis: ClipAnalysis):
        """Update aggregated player stats with new clip analysis."""

        if player_name not in self.player_stats:
            self.player_stats[player_name] = PlayerStats(player_name=player_name)

        stats = self.player_stats[player_name]
        stats.total_clips += 1
        stats.total_play_time += analysis.duration_seconds
        stats.ball_touches += analysis.ball_touches
        stats.total_distance_pixels += analysis.distance_covered_pixels

        # Count events
        for event in analysis.events:
            if event.event_type == 'pass':
                stats.passes_attempted += 1
                if event.success:
                    stats.passes_completed += 1
            elif event.event_type == 'shot':
                stats.shots += 1
                if event.success:
                    stats.shots_on_target += 1
            elif event.event_type == 'tackle':
                stats.tackles_attempted += 1
                if event.success:
                    stats.tackles_won += 1
            elif event.event_type == 'header':
                stats.headers += 1

    def analyze_clips_folder(self, folder_path: str, player_name: str = "Player") -> PlayerStats:
        """Analyze all video clips in a folder for a specific player."""

        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
        clips = []

        for ext in video_extensions:
            clips.extend(Path(folder_path).glob(f'*{ext}'))
            clips.extend(Path(folder_path).glob(f'*{ext.upper()}'))

        print(f"Found {len(clips)} clips to analyze for {player_name}")

        for i, clip_path in enumerate(clips):
            print(f"Analyzing clip {i+1}/{len(clips)}: {clip_path.name}")
            try:
                self.analyze_clip(str(clip_path), player_name)
            except Exception as e:
                print(f"Error analyzing {clip_path}: {e}")

        return self.player_stats.get(player_name)

    def get_player_report(self, player_name: str) -> Dict:
        """Generate a full report for a player."""

        if player_name not in self.player_stats:
            return {"error": f"No stats found for {player_name}"}

        stats = self.player_stats[player_name]

        return {
            "player_name": stats.player_name,
            "summary": {
                "total_clips_analyzed": stats.total_clips,
                "total_play_time_minutes": round(stats.total_play_time / 60, 1),
            },
            "attacking": {
                "ball_touches": stats.ball_touches,
                "passes_attempted": stats.passes_attempted,
                "passes_completed": stats.passes_completed,
                "pass_accuracy": round(stats.pass_accuracy, 1),
                "shots": stats.shots,
                "shots_on_target": stats.shots_on_target,
                "shot_accuracy": round(stats.shot_accuracy, 1),
            },
            "defensive": {
                "tackles_attempted": stats.tackles_attempted,
                "tackles_won": stats.tackles_won,
                "tackle_success_rate": round(stats.tackle_success_rate, 1),
                "headers": stats.headers,
                "interceptions": stats.interceptions,
            },
            "physical": {
                "distance_covered_pixels": round(stats.total_distance_pixels, 0),
                "distance_covered_meters_estimate": round(stats.total_distance_pixels * 0.05, 0),
                "sprints": stats.sprints,
            }
        }

    def export_stats(self, output_path: str):
        """Export all player stats to JSON."""

        data = {
            "generated_at": datetime.now().isoformat(),
            "players": {}
        }

        for player_name in self.player_stats:
            data["players"][player_name] = self.get_player_report(player_name)

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"Stats exported to: {output_path}")
        return data


# Global instance
player_analyzer = PlayerClipAnalyzer()
