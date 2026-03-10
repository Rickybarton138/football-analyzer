"""
Player Clip Analyzer Service

Analyzes individual player highlight clips to extract player-specific
statistics including elite-level metrics: first touch quality, decision
speed, progressive carries, line-breaking passes, and position benchmarks.
"""
import cv2
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from pathlib import Path
from collections import deque
import json
import os
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


# ─── Position Benchmarks (per-90 or percentage baselines) ───────────────
POSITION_BENCHMARKS: Dict[str, Dict[str, float]] = {
    "GK": {
        "pass_accuracy": 78.0, "passes_per_90": 25.0,
        "saves_per_90": 3.0,
    },
    "CB": {
        "pass_accuracy": 85.0, "tackles_per_90": 2.5, "interceptions_per_90": 2.0,
        "aerial_duel_win_pct": 60.0, "progressive_passes_per_90": 3.0,
        "recovery_runs_per_90": 2.0,
    },
    "FB": {
        "pass_accuracy": 80.0, "crosses_per_90": 3.0, "tackles_per_90": 2.0,
        "progressive_carries_per_90": 4.0, "distance_per_90_km": 10.5,
        "sprints_per_90": 25,
    },
    "CM": {
        "pass_accuracy": 83.0, "progressive_passes_per_90": 5.0,
        "interceptions_per_90": 1.5, "tackles_per_90": 2.0,
        "decision_speed_s": 0.6, "line_breaking_passes_per_90": 3.0,
        "distance_per_90_km": 11.0,
    },
    "CAM": {
        "pass_accuracy": 80.0, "key_passes_per_90": 2.5,
        "progressive_carries_per_90": 5.0, "shots_per_90": 2.0,
        "xg_per_shot": 0.12, "decision_speed_s": 0.5,
    },
    "WM": {  # Wide midfielder / winger
        "pass_accuracy": 77.0, "dribble_success_pct": 55.0,
        "crosses_per_90": 4.0, "progressive_carries_per_90": 6.0,
        "shots_per_90": 2.0, "sprints_per_90": 30,
    },
    "ST": {
        "pass_accuracy": 75.0, "shots_per_90": 3.0, "shots_on_target_pct": 45.0,
        "xg_per_shot": 0.15, "aerial_duel_win_pct": 50.0,
        "pressing_intensity_m_per_s": 3.0, "decision_speed_s": 0.5,
    },
}


@dataclass
class PlayerEvent:
    """A detected event involving the player."""
    event_type: str  # 'touch', 'pass', 'shot', 'tackle', 'header', 'dribble'
    timestamp: float
    frame_number: int
    position: List[float]  # [x, y] on pitch
    success: Optional[bool] = None
    details: Dict = field(default_factory=dict)


@dataclass
class ClipMetrics:
    """Full elite metrics extracted from a single clip."""
    clip_id: str = ""
    clip_path: str = ""
    duration_seconds: float = 0.0

    # ── Core metrics ──
    touches: int = 0
    passes_attempted: int = 0
    passes_completed: int = 0
    pass_accuracy: float = 0.0
    pass_directions: Dict[str, int] = field(default_factory=lambda: {
        "forward": 0, "lateral": 0, "backward": 0
    })
    shots: int = 0
    shots_on_target: int = 0
    xg_total: float = 0.0
    tackles_attempted: int = 0
    tackles_won: int = 0
    interceptions: int = 0
    dribbles_attempted: int = 0
    dribbles_successful: int = 0
    headers: int = 0

    # ── Advanced metrics ──
    first_touch_quality: float = 0.0   # Ball displacement after first contact (px, lower = better)
    body_orientation_score: float = 0.0  # 0-1 facing angle vs play direction
    decision_speed_s: float = 0.0      # Seconds between receiving ball and action
    pressing_intensity_m_per_s: float = 0.0  # Distance closed on opponent/sec
    off_ball_distance_m: float = 0.0   # Distance covered without ball
    space_creation_area: float = 0.0   # Convex hull area of movement

    # ── Elite metrics ──
    progressive_carries: int = 0       # Ball advancement >10m towards goal
    line_breaking_passes: int = 0      # Passes bypassing >=1 opponent line
    defensive_recovery_runs: int = 0   # Sprints back after losing possession
    positional_heat_zone: str = ""     # "defensive_third", "middle_third", "attacking_third"
    positional_channel: str = ""       # "left", "central", "right"

    # ── Movement ──
    distance_covered_m: float = 0.0
    sprints: int = 0
    max_speed_m_per_s: float = 0.0
    avg_position: List[float] = field(default_factory=list)

    # ── Benchmark comparison ──
    position: str = ""                 # Player position (CB, CM, ST, etc.)
    benchmark_comparison: Dict[str, Dict[str, float]] = field(default_factory=dict)
    # e.g. {"pass_accuracy": {"value": 87, "benchmark": 83, "delta": +4}}

    # ── Events list ──
    events: List[Dict] = field(default_factory=list)


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
    metrics: Optional[ClipMetrics] = None


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

    # ── Elite aggregates (NEW) ──
    total_distance_m: float = 0.0
    progressive_carries: int = 0
    line_breaking_passes: int = 0
    defensive_recovery_runs: int = 0
    dribbles_attempted: int = 0
    dribbles_successful: int = 0
    xg_total: float = 0.0
    avg_first_touch_quality: float = 0.0
    avg_decision_speed_s: float = 0.0
    position: str = ""
    clip_metrics: List[ClipMetrics] = field(default_factory=list)

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

    @property
    def dribble_success_rate(self) -> float:
        if self.dribbles_attempted == 0:
            return 0
        return (self.dribbles_successful / self.dribbles_attempted) * 100


class PlayerClipAnalyzer:
    """Analyzes player highlight clips to extract individual statistics."""

    # Pixels-to-meters rough estimate (VEO wide-angle ~105m across 1920px)
    PX_TO_M = 105.0 / 1920.0

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
                logger.info("YOLOv8 loaded for player clip analysis")
            except ImportError:
                logger.warning("ultralytics not installed")
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

        # Elite metric tracking state
        _first_touch_displacements = []
        _ball_receive_frame = None
        _action_frames = []  # frames between receive and action
        _off_ball_distance_px = 0.0
        _has_ball = False
        _progressive_carry_start = None
        _sprint_frames = 0
        _max_speed_px = 0.0
        _all_opponent_positions = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            timestamp = frame_count / fps if fps > 0 else 0

            player_box = None
            ball_pos = None
            opponent_boxes = []

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
                            dist_from_center = abs(center_x - self.VIDEO_WIDTH / 2)
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
                    for p in persons:
                        p['score'] = (
                            p['area'] * 0.7
                            + (1 - p['dist_from_center'] / self.VIDEO_WIDTH) * 0.3 * 10000
                        )
                    persons.sort(key=lambda x: x['score'], reverse=True)
                    player_box = persons[0]
                    # All other persons are "opponents" for metric calculations
                    opponent_boxes = persons[1:]

            # Track player
            if player_box:
                analysis.player_visible_frames += 1
                player_pos = player_box['center']
                all_player_positions.append(player_pos)
                self.player_positions.append(player_pos)

                if self.prev_player_pos:
                    dist_px = np.sqrt(
                        (player_pos[0] - self.prev_player_pos[0]) ** 2
                        + (player_pos[1] - self.prev_player_pos[1]) ** 2
                    )
                    analysis.distance_covered_pixels += dist_px

                    # Speed tracking
                    speed_px = dist_px * fps if fps > 0 else 0
                    if speed_px > _max_speed_px:
                        _max_speed_px = speed_px
                    if speed_px > 200:  # Sprint threshold (px/s)
                        _sprint_frames += 1

                    # Off-ball distance
                    if not _has_ball:
                        _off_ball_distance_px += dist_px

                self.prev_player_pos = player_pos

            # Collect opponent positions for line-break detection
            for opp in opponent_boxes:
                _all_opponent_positions.append(opp['center'])

            # Track ball and detect events
            if ball_pos:
                self.ball_positions.append(ball_pos)

                if player_box:
                    player_pos = player_box['center']
                    ball_dist = np.sqrt(
                        (ball_pos[0] - player_pos[0]) ** 2
                        + (ball_pos[1] - player_pos[1]) ** 2
                    )

                    was_on_ball = _has_ball
                    _has_ball = ball_dist < 80

                    # First touch: ball just arrived near player
                    if _has_ball and not was_on_ball:
                        _ball_receive_frame = frame_count
                        # Measure displacement next frame
                        if self.prev_ball_pos:
                            disp = np.sqrt(
                                (ball_pos[0] - self.prev_ball_pos[0]) ** 2
                                + (ball_pos[1] - self.prev_ball_pos[1]) ** 2
                            )
                            _first_touch_displacements.append(disp)

                        # Start progressive carry tracking
                        _progressive_carry_start = player_pos[0]

                    # Decision speed: frames between receive and action (pass/shot)
                    if _has_ball and _ball_receive_frame is not None:
                        _action_frames.append(frame_count - _ball_receive_frame)

                    # Lost ball — check progressive carry
                    if was_on_ball and not _has_ball and _progressive_carry_start is not None:
                        carry_dist_px = abs(player_pos[0] - _progressive_carry_start)
                        carry_dist_m = carry_dist_px * self.PX_TO_M
                        if carry_dist_m >= 10.0:
                            analysis.events.append(PlayerEvent(
                                event_type='progressive_carry',
                                timestamp=timestamp,
                                frame_number=frame_count,
                                position=player_pos,
                                details={"distance_m": round(carry_dist_m, 1)}
                            ))
                        _progressive_carry_start = None

                    if ball_dist < 80:
                        event = self._detect_event(
                            frame, player_pos, ball_pos, timestamp, frame_count,
                            opponent_boxes=opponent_boxes
                        )
                        if event:
                            analysis.events.append(event)
                            analysis.ball_touches += 1
                            # Reset decision timer on action
                            if event.event_type in ('pass', 'shot'):
                                if _ball_receive_frame is not None:
                                    _action_frames.append(
                                        frame_count - _ball_receive_frame
                                    )
                                _ball_receive_frame = None

                self.prev_ball_pos = ball_pos

            self.prev_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_count += 1

        cap.release()

        # Calculate average position
        if all_player_positions:
            analysis.avg_position = [
                float(np.mean([p[0] for p in all_player_positions])),
                float(np.mean([p[1] for p in all_player_positions]))
            ]
            analysis.heatmap_data = all_player_positions

        # Build ClipMetrics
        metrics = self._build_clip_metrics(
            analysis, fps,
            _first_touch_displacements, _action_frames,
            _off_ball_distance_px, _sprint_frames, _max_speed_px,
            all_player_positions
        )
        analysis.metrics = metrics

        # Store analysis
        self.clip_analyses.append(analysis)

        # Update player stats
        self._update_player_stats(player_name, analysis)

        return analysis

    def _detect_event(
        self, frame, player_pos, ball_pos, timestamp, frame_num,
        opponent_boxes: List[Dict] = None
    ) -> Optional[PlayerEvent]:
        """Detect what type of event is happening."""

        if len(self.ball_positions) < 2:
            return PlayerEvent(
                event_type='touch',
                timestamp=timestamp,
                frame_number=frame_num,
                position=ball_pos
            )

        prev_ball = self.ball_positions[-2]
        dx = ball_pos[0] - prev_ball[0]
        dy = ball_pos[1] - prev_ball[1]
        ball_speed = np.sqrt(dx * dx + dy * dy)

        event_type = 'touch'
        success = None
        details = {}

        goal_left = self.VIDEO_WIDTH * 0.1
        goal_right = self.VIDEO_WIDTH * 0.9

        if ball_speed > 100:
            if (ball_pos[0] < goal_left and dx < -50) or (ball_pos[0] > goal_right and dx > 50):
                event_type = 'shot'
                on_target = self.VIDEO_HEIGHT * 0.3 < ball_pos[1] < self.VIDEO_HEIGHT * 0.7
                success = on_target
                details['on_target'] = on_target
                # Simple xG estimate based on distance from goal
                goal_x = 0 if ball_pos[0] < self.VIDEO_WIDTH / 2 else self.VIDEO_WIDTH
                dist_to_goal_px = abs(ball_pos[0] - goal_x)
                dist_to_goal_m = dist_to_goal_px * self.PX_TO_M
                details['xg'] = round(max(0.02, min(0.9, 0.5 - dist_to_goal_m * 0.015)), 3)
            else:
                event_type = 'pass'
                success = True
                if abs(dx) > abs(dy):
                    details['direction'] = 'forward' if dx > 0 else 'backward'
                else:
                    details['direction'] = 'lateral'

                # Line-breaking pass detection
                if opponent_boxes:
                    line_broken = self._check_line_break(
                        player_pos, ball_pos, opponent_boxes
                    )
                    details['line_breaking'] = line_broken

        # Header detection
        if dy < -30 and ball_pos[1] < self.VIDEO_HEIGHT * 0.4:
            event_type = 'header'

        # Dribble detection: ball stays near player while player moves
        if event_type == 'touch' and len(self.player_positions) >= 3:
            recent_move = np.sqrt(
                (self.player_positions[-1][0] - self.player_positions[-3][0]) ** 2
                + (self.player_positions[-1][1] - self.player_positions[-3][1]) ** 2
            )
            if recent_move > 30:
                event_type = 'dribble'
                success = True

        # Tackle detection: ball near player, opponent very close
        if opponent_boxes and event_type == 'touch':
            for opp in opponent_boxes:
                opp_dist = np.sqrt(
                    (player_pos[0] - opp['center'][0]) ** 2
                    + (player_pos[1] - opp['center'][1]) ** 2
                )
                if opp_dist < 60:
                    event_type = 'tackle'
                    success = True
                    break

        return PlayerEvent(
            event_type=event_type,
            timestamp=timestamp,
            frame_number=frame_num,
            position=ball_pos,
            success=success,
            details=details
        )

    def _check_line_break(
        self,
        passer_pos: List[float],
        ball_dest: List[float],
        opponent_boxes: List[Dict]
    ) -> bool:
        """Check if a pass breaks through at least one opponent line."""
        if not opponent_boxes:
            return False

        # Sort opponents by x position (proxy for defensive lines)
        opp_xs = sorted([o['center'][0] for o in opponent_boxes])
        pass_min_x = min(passer_pos[0], ball_dest[0])
        pass_max_x = max(passer_pos[0], ball_dest[0])

        # Count opponents between passer and destination
        bypassed = sum(1 for ox in opp_xs if pass_min_x < ox < pass_max_x)
        return bypassed >= 1

    def _build_clip_metrics(
        self,
        analysis: ClipAnalysis,
        fps: float,
        first_touch_disps: List[float],
        action_frames: List[int],
        off_ball_dist_px: float,
        sprint_frames: int,
        max_speed_px: float,
        all_positions: List[List[float]],
    ) -> ClipMetrics:
        """Build full ClipMetrics from raw analysis data."""
        metrics = ClipMetrics(
            clip_path=analysis.clip_path,
            duration_seconds=analysis.duration_seconds,
        )

        # Count events by type
        for event in analysis.events:
            et = event.event_type
            if et == 'touch':
                metrics.touches += 1
            elif et == 'pass':
                metrics.passes_attempted += 1
                if event.success:
                    metrics.passes_completed += 1
                direction = event.details.get('direction', 'lateral')
                if direction in metrics.pass_directions:
                    metrics.pass_directions[direction] += 1
                if event.details.get('line_breaking'):
                    metrics.line_breaking_passes += 1
            elif et == 'shot':
                metrics.shots += 1
                if event.success:
                    metrics.shots_on_target += 1
                metrics.xg_total += event.details.get('xg', 0)
            elif et == 'tackle':
                metrics.tackles_attempted += 1
                if event.success:
                    metrics.tackles_won += 1
            elif et == 'header':
                metrics.headers += 1
            elif et == 'dribble':
                metrics.dribbles_attempted += 1
                if event.success:
                    metrics.dribbles_successful += 1
            elif et == 'progressive_carry':
                metrics.progressive_carries += 1
            elif et == 'interception':
                metrics.interceptions += 1

            metrics.events.append(asdict(event))

        # Pass accuracy
        if metrics.passes_attempted > 0:
            metrics.pass_accuracy = round(
                metrics.passes_completed / metrics.passes_attempted * 100, 1
            )

        # First touch quality (lower = better, avg pixel displacement)
        if first_touch_disps:
            metrics.first_touch_quality = round(float(np.mean(first_touch_disps)), 1)

        # Decision speed
        if action_frames:
            valid = [f for f in action_frames if f > 0 and fps > 0]
            if valid:
                metrics.decision_speed_s = round(float(np.mean(valid)) / fps, 2)

        # Off-ball movement
        metrics.off_ball_distance_m = round(off_ball_dist_px * self.PX_TO_M, 1)

        # Sprints and speed
        if fps > 0:
            metrics.sprints = max(1, sprint_frames // int(fps * 2)) if sprint_frames > fps else 0
            metrics.max_speed_m_per_s = round(max_speed_px * self.PX_TO_M, 1)

        # Total distance
        metrics.distance_covered_m = round(
            analysis.distance_covered_pixels * self.PX_TO_M, 1
        )

        # Average position
        metrics.avg_position = analysis.avg_position

        # Space creation (convex hull area of movement)
        if len(all_positions) >= 3:
            try:
                from scipy.spatial import ConvexHull
                pts = np.array(all_positions)
                hull = ConvexHull(pts)
                metrics.space_creation_area = round(
                    hull.volume * self.PX_TO_M * self.PX_TO_M, 1
                )
            except Exception:
                metrics.space_creation_area = 0.0

        # Positional heat zone
        if analysis.avg_position:
            avg_x = analysis.avg_position[0]
            avg_y = analysis.avg_position[1]
            # Thirds (x-axis)
            if avg_x < self.VIDEO_WIDTH / 3:
                metrics.positional_heat_zone = "defensive_third"
            elif avg_x < 2 * self.VIDEO_WIDTH / 3:
                metrics.positional_heat_zone = "middle_third"
            else:
                metrics.positional_heat_zone = "attacking_third"
            # Channels (y-axis)
            if avg_y < self.VIDEO_HEIGHT / 3:
                metrics.positional_channel = "left"
            elif avg_y < 2 * self.VIDEO_HEIGHT / 3:
                metrics.positional_channel = "central"
            else:
                metrics.positional_channel = "right"

        return metrics

    def compare_to_benchmark(
        self,
        metrics: ClipMetrics,
        position: str,
        duration_minutes: float = 0.0
    ) -> Dict[str, Dict[str, float]]:
        """Compare clip metrics to position benchmarks."""
        benchmarks = POSITION_BENCHMARKS.get(position, POSITION_BENCHMARKS.get("CM", {}))
        comparison = {}

        if "pass_accuracy" in benchmarks and metrics.passes_attempted > 0:
            comparison["pass_accuracy"] = {
                "value": metrics.pass_accuracy,
                "benchmark": benchmarks["pass_accuracy"],
                "delta": round(metrics.pass_accuracy - benchmarks["pass_accuracy"], 1),
            }

        if "decision_speed_s" in benchmarks and metrics.decision_speed_s > 0:
            comparison["decision_speed_s"] = {
                "value": metrics.decision_speed_s,
                "benchmark": benchmarks["decision_speed_s"],
                "delta": round(benchmarks["decision_speed_s"] - metrics.decision_speed_s, 2),
                # Positive delta = better than benchmark
            }

        if "xg_per_shot" in benchmarks and metrics.shots > 0:
            xg_per_shot = metrics.xg_total / metrics.shots
            comparison["xg_per_shot"] = {
                "value": round(xg_per_shot, 3),
                "benchmark": benchmarks["xg_per_shot"],
                "delta": round(xg_per_shot - benchmarks["xg_per_shot"], 3),
            }

        metrics.benchmark_comparison = comparison
        metrics.position = position
        return comparison

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
            elif event.event_type == 'dribble':
                stats.dribbles_attempted += 1
                if event.success:
                    stats.dribbles_successful += 1
            elif event.event_type == 'progressive_carry':
                stats.progressive_carries += 1
            elif event.event_type == 'interception':
                stats.interceptions += 1

        # Aggregate elite metrics
        if analysis.metrics:
            m = analysis.metrics
            stats.total_distance_m += m.distance_covered_m
            stats.line_breaking_passes += m.line_breaking_passes
            stats.defensive_recovery_runs += m.defensive_recovery_runs
            stats.xg_total += m.xg_total
            stats.clip_metrics.append(m)

            # Rolling averages
            all_ft = [cm.first_touch_quality for cm in stats.clip_metrics if cm.first_touch_quality > 0]
            if all_ft:
                stats.avg_first_touch_quality = round(float(np.mean(all_ft)), 1)
            all_ds = [cm.decision_speed_s for cm in stats.clip_metrics if cm.decision_speed_s > 0]
            if all_ds:
                stats.avg_decision_speed_s = round(float(np.mean(all_ds)), 2)

    def analyze_clips_folder(self, folder_path: str, player_name: str = "Player") -> PlayerStats:
        """Analyze all video clips in a folder for a specific player."""

        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
        clips = []

        for ext in video_extensions:
            clips.extend(Path(folder_path).glob(f'*{ext}'))
            clips.extend(Path(folder_path).glob(f'*{ext.upper()}'))

        logger.info(f"Found {len(clips)} clips to analyze for {player_name}")

        for i, clip_path in enumerate(clips):
            logger.info(f"Analyzing clip {i + 1}/{len(clips)}: {clip_path.name}")
            try:
                self.analyze_clip(str(clip_path), player_name)
            except Exception as e:
                logger.error(f"Error analyzing {clip_path}: {e}")

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
                "xg_total": round(stats.xg_total, 2),
                "dribbles_attempted": stats.dribbles_attempted,
                "dribbles_successful": stats.dribbles_successful,
                "dribble_success_rate": round(stats.dribble_success_rate, 1),
            },
            "defensive": {
                "tackles_attempted": stats.tackles_attempted,
                "tackles_won": stats.tackles_won,
                "tackle_success_rate": round(stats.tackle_success_rate, 1),
                "headers": stats.headers,
                "interceptions": stats.interceptions,
                "defensive_recovery_runs": stats.defensive_recovery_runs,
            },
            "physical": {
                "distance_covered_meters": round(stats.total_distance_m, 0),
                "sprints": stats.sprints,
            },
            "elite": {
                "avg_first_touch_quality": stats.avg_first_touch_quality,
                "avg_decision_speed_s": stats.avg_decision_speed_s,
                "progressive_carries": stats.progressive_carries,
                "line_breaking_passes": stats.line_breaking_passes,
            },
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

        logger.info(f"Stats exported to: {output_path}")
        return data


# Global instance
player_analyzer = PlayerClipAnalyzer()
