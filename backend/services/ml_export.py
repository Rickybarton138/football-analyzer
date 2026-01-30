"""
ML-Ready Dataset Export Pipeline

Exports analyzed football data in formats suitable for machine learning:
- Player action clips with labels
- Technical skill scores for supervised learning
- Tactical formation data
- Success/failure outcomes for classification

This enables:
1. Training custom models on YOUR team's data
2. Creating player performance prediction models
3. Building tactical pattern recognition systems
4. Fine-tuning vision models on football actions
"""

import json
import csv
import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from pathlib import Path
import numpy as np


@dataclass
class ActionDataPoint:
    """A single action data point for ML training"""
    # Identifiers
    action_id: str
    video_id: str
    player_jersey: int
    player_name: Optional[str]
    player_position: Optional[str]

    # Temporal
    start_frame: int
    end_frame: int
    timestamp_ms: int

    # Action info
    action_type: str  # pass, shot, tackle, dribble, etc.
    action_outcome: str  # success, failure, partial

    # Scores (labels for supervised learning)
    overall_score: float  # 0-1
    technical_score: float
    tactical_score: float
    physical_score: float
    psychological_score: float

    # Specific technique scores (for multi-task learning)
    technique_scores: Dict[str, float]

    # Metadata
    team: str
    formation: str
    phase_of_play: str
    match_context: str  # score state, time period

    # Raw features
    body_position_score: Optional[float] = None
    timing_score: Optional[float] = None
    decision_score: Optional[float] = None


@dataclass
class TacticalDataPoint:
    """Formation/tactical data point for ML"""
    # Identifiers
    snapshot_id: str
    video_id: str
    frame_number: int
    timestamp_ms: int

    # Formation
    formation: str
    formation_confidence: float

    # Team positions (normalized 0-100)
    player_positions: List[Dict[str, float]]  # [{x, y, role, team}]
    ball_position: Optional[Tuple[float, float]]

    # Labels
    effectiveness_score: float  # 0-100
    phase: str
    vulnerabilities_count: int

    # Component scores
    shape_score: float
    compactness_score: float
    width_balance: float
    depth_balance: float

    # Unit scores
    defensive_unit_score: float
    midfield_unit_score: float
    attacking_unit_score: float


@dataclass
class PlayerPerformanceRecord:
    """Aggregated player performance for prediction models"""
    player_jersey: int
    player_name: Optional[str]
    player_position: Optional[str]

    # Aggregated scores
    total_actions: int
    avg_overall_score: float
    avg_technical_score: float
    avg_tactical_score: float
    avg_physical_score: float
    avg_psychological_score: float

    # Action breakdown
    passes_attempted: int
    passes_successful: int
    pass_accuracy: float

    shots_attempted: int
    shots_on_target: int
    shot_accuracy: float

    dribbles_attempted: int
    dribbles_successful: int
    dribble_success_rate: float

    tackles_attempted: int
    tackles_won: int
    tackle_success_rate: float

    # Strengths/weaknesses
    top_strengths: List[str]
    top_weaknesses: List[str]

    # Development trend (if multiple matches)
    score_trend: str  # "improving", "stable", "declining"


class MLExportService:
    """
    Service for exporting analysis data in ML-ready formats.

    Supports:
    - CSV export for tabular ML (sklearn, XGBoost, etc.)
    - JSON export for deep learning
    - COCO format for object detection fine-tuning
    - Custom formats for specific frameworks
    """

    def __init__(self, output_dir: str = "ml_exports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Collected data
        self.action_data: List[ActionDataPoint] = []
        self.tactical_data: List[TacticalDataPoint] = []
        self.player_records: Dict[int, PlayerPerformanceRecord] = {}

    # =========================================================================
    # DATA COLLECTION
    # =========================================================================

    def add_action_analysis(
        self,
        action_id: str,
        video_id: str,
        player_jersey: int,
        action_type: str,
        start_frame: int,
        end_frame: int,
        scores: Dict[str, Any],
        outcome: str = "unknown",
        player_name: Optional[str] = None,
        player_position: Optional[str] = None,
        team: str = "home",
        formation: str = "4-3-3",
        phase: str = "attacking_organization",
        match_context: str = "0-0"
    ):
        """Add an action analysis to the dataset."""

        # Extract scores
        overall = scores.get("overall", 0.5)
        four_corner = scores.get("four_corner", {})
        technique_scores = scores.get("technique_scores", {})

        data_point = ActionDataPoint(
            action_id=action_id,
            video_id=video_id,
            player_jersey=player_jersey,
            player_name=player_name,
            player_position=player_position,
            start_frame=start_frame,
            end_frame=end_frame,
            timestamp_ms=int(start_frame * 33.33),  # Assuming 30fps
            action_type=action_type,
            action_outcome=outcome,
            overall_score=overall,
            technical_score=four_corner.get("technical", {}).get("score", 5.0) / 10,
            tactical_score=four_corner.get("tactical", {}).get("score", 5.0) / 10,
            physical_score=four_corner.get("physical", {}).get("score", 5.0) / 10,
            psychological_score=four_corner.get("psychological", {}).get("score", 5.0) / 10,
            technique_scores=technique_scores,
            team=team,
            formation=formation,
            phase_of_play=phase,
            match_context=match_context,
            body_position_score=technique_scores.get("body_position"),
            timing_score=technique_scores.get("timing"),
            decision_score=technique_scores.get("decision")
        )

        self.action_data.append(data_point)

        # Update player record
        self._update_player_record(data_point)

    def add_tactical_snapshot(
        self,
        video_id: str,
        frame_number: int,
        players: List[Dict],
        formation: str,
        effectiveness: float,
        phase: str,
        ball_position: Optional[Tuple[float, float]] = None,
        component_scores: Optional[Dict] = None,
        unit_scores: Optional[Dict] = None,
        vulnerabilities: Optional[List[str]] = None
    ):
        """Add a tactical snapshot to the dataset."""

        snapshot_id = f"{video_id}_{frame_number}"

        component = component_scores or {}
        units = unit_scores or {}

        data_point = TacticalDataPoint(
            snapshot_id=snapshot_id,
            video_id=video_id,
            frame_number=frame_number,
            timestamp_ms=int(frame_number * 33.33),
            formation=formation,
            formation_confidence=component.get("formation_confidence", 0.8),
            player_positions=players,
            ball_position=ball_position,
            effectiveness_score=effectiveness,
            phase=phase,
            vulnerabilities_count=len(vulnerabilities) if vulnerabilities else 0,
            shape_score=component.get("shape_maintenance", 50),
            compactness_score=component.get("compactness", 50),
            width_balance=component.get("width_balance", 50),
            depth_balance=component.get("depth_balance", 50),
            defensive_unit_score=units.get("defensive", {}).get("compactness", 50),
            midfield_unit_score=units.get("midfield", {}).get("compactness", 50),
            attacking_unit_score=units.get("attacking", {}).get("support", 50)
        )

        self.tactical_data.append(data_point)

    def _update_player_record(self, action: ActionDataPoint):
        """Update aggregated player record."""
        jersey = action.player_jersey

        if jersey not in self.player_records:
            self.player_records[jersey] = PlayerPerformanceRecord(
                player_jersey=jersey,
                player_name=action.player_name,
                player_position=action.player_position,
                total_actions=0,
                avg_overall_score=0,
                avg_technical_score=0,
                avg_tactical_score=0,
                avg_physical_score=0,
                avg_psychological_score=0,
                passes_attempted=0,
                passes_successful=0,
                pass_accuracy=0,
                shots_attempted=0,
                shots_on_target=0,
                shot_accuracy=0,
                dribbles_attempted=0,
                dribbles_successful=0,
                dribble_success_rate=0,
                tackles_attempted=0,
                tackles_won=0,
                tackle_success_rate=0,
                top_strengths=[],
                top_weaknesses=[],
                score_trend="stable"
            )

        record = self.player_records[jersey]
        n = record.total_actions

        # Update averages
        record.total_actions += 1
        record.avg_overall_score = (record.avg_overall_score * n + action.overall_score) / (n + 1)
        record.avg_technical_score = (record.avg_technical_score * n + action.technical_score) / (n + 1)
        record.avg_tactical_score = (record.avg_tactical_score * n + action.tactical_score) / (n + 1)
        record.avg_physical_score = (record.avg_physical_score * n + action.physical_score) / (n + 1)
        record.avg_psychological_score = (record.avg_psychological_score * n + action.psychological_score) / (n + 1)

        # Update action-specific counts
        action_type = action.action_type.lower()
        is_success = action.action_outcome == "success"

        if "pass" in action_type:
            record.passes_attempted += 1
            if is_success:
                record.passes_successful += 1
            record.pass_accuracy = record.passes_successful / record.passes_attempted

        elif "shot" in action_type:
            record.shots_attempted += 1
            if is_success or action.action_outcome == "on_target":
                record.shots_on_target += 1
            record.shot_accuracy = record.shots_on_target / record.shots_attempted

        elif "dribble" in action_type:
            record.dribbles_attempted += 1
            if is_success:
                record.dribbles_successful += 1
            record.dribble_success_rate = record.dribbles_successful / record.dribbles_attempted

        elif "tackle" in action_type:
            record.tackles_attempted += 1
            if is_success:
                record.tackles_won += 1
            record.tackle_success_rate = record.tackles_won / record.tackles_attempted

    # =========================================================================
    # CSV EXPORT (for sklearn, pandas, XGBoost)
    # =========================================================================

    def export_actions_csv(self, filename: str = "actions_dataset.csv") -> str:
        """Export action data to CSV for tabular ML."""
        filepath = self.output_dir / filename

        if not self.action_data:
            return str(filepath)

        # Define columns
        columns = [
            "action_id", "video_id", "player_jersey", "player_position",
            "start_frame", "end_frame", "action_type", "action_outcome",
            "overall_score", "technical_score", "tactical_score",
            "physical_score", "psychological_score",
            "body_position_score", "timing_score", "decision_score",
            "team", "formation", "phase_of_play", "match_context"
        ]

        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()

            for action in self.action_data:
                row = {
                    "action_id": action.action_id,
                    "video_id": action.video_id,
                    "player_jersey": action.player_jersey,
                    "player_position": action.player_position or "",
                    "start_frame": action.start_frame,
                    "end_frame": action.end_frame,
                    "action_type": action.action_type,
                    "action_outcome": action.action_outcome,
                    "overall_score": round(action.overall_score, 4),
                    "technical_score": round(action.technical_score, 4),
                    "tactical_score": round(action.tactical_score, 4),
                    "physical_score": round(action.physical_score, 4),
                    "psychological_score": round(action.psychological_score, 4),
                    "body_position_score": round(action.body_position_score, 4) if action.body_position_score else "",
                    "timing_score": round(action.timing_score, 4) if action.timing_score else "",
                    "decision_score": round(action.decision_score, 4) if action.decision_score else "",
                    "team": action.team,
                    "formation": action.formation,
                    "phase_of_play": action.phase_of_play,
                    "match_context": action.match_context
                }
                writer.writerow(row)

        return str(filepath)

    def export_tactical_csv(self, filename: str = "tactical_dataset.csv") -> str:
        """Export tactical data to CSV."""
        filepath = self.output_dir / filename

        if not self.tactical_data:
            return str(filepath)

        columns = [
            "snapshot_id", "video_id", "frame_number", "timestamp_ms",
            "formation", "effectiveness_score", "phase",
            "shape_score", "compactness_score", "width_balance", "depth_balance",
            "defensive_unit_score", "midfield_unit_score", "attacking_unit_score",
            "vulnerabilities_count"
        ]

        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()

            for snapshot in self.tactical_data:
                row = {
                    "snapshot_id": snapshot.snapshot_id,
                    "video_id": snapshot.video_id,
                    "frame_number": snapshot.frame_number,
                    "timestamp_ms": snapshot.timestamp_ms,
                    "formation": snapshot.formation,
                    "effectiveness_score": round(snapshot.effectiveness_score, 2),
                    "phase": snapshot.phase,
                    "shape_score": round(snapshot.shape_score, 2),
                    "compactness_score": round(snapshot.compactness_score, 2),
                    "width_balance": round(snapshot.width_balance, 2),
                    "depth_balance": round(snapshot.depth_balance, 2),
                    "defensive_unit_score": round(snapshot.defensive_unit_score, 2),
                    "midfield_unit_score": round(snapshot.midfield_unit_score, 2),
                    "attacking_unit_score": round(snapshot.attacking_unit_score, 2),
                    "vulnerabilities_count": snapshot.vulnerabilities_count
                }
                writer.writerow(row)

        return str(filepath)

    def export_player_records_csv(self, filename: str = "player_performance.csv") -> str:
        """Export player performance records to CSV."""
        filepath = self.output_dir / filename

        if not self.player_records:
            return str(filepath)

        columns = [
            "player_jersey", "player_name", "player_position",
            "total_actions", "avg_overall_score",
            "avg_technical_score", "avg_tactical_score",
            "avg_physical_score", "avg_psychological_score",
            "passes_attempted", "passes_successful", "pass_accuracy",
            "shots_attempted", "shots_on_target", "shot_accuracy",
            "dribbles_attempted", "dribbles_successful", "dribble_success_rate",
            "tackles_attempted", "tackles_won", "tackle_success_rate"
        ]

        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()

            for record in self.player_records.values():
                row = {
                    "player_jersey": record.player_jersey,
                    "player_name": record.player_name or "",
                    "player_position": record.player_position or "",
                    "total_actions": record.total_actions,
                    "avg_overall_score": round(record.avg_overall_score, 4),
                    "avg_technical_score": round(record.avg_technical_score, 4),
                    "avg_tactical_score": round(record.avg_tactical_score, 4),
                    "avg_physical_score": round(record.avg_physical_score, 4),
                    "avg_psychological_score": round(record.avg_psychological_score, 4),
                    "passes_attempted": record.passes_attempted,
                    "passes_successful": record.passes_successful,
                    "pass_accuracy": round(record.pass_accuracy, 4),
                    "shots_attempted": record.shots_attempted,
                    "shots_on_target": record.shots_on_target,
                    "shot_accuracy": round(record.shot_accuracy, 4),
                    "dribbles_attempted": record.dribbles_attempted,
                    "dribbles_successful": record.dribbles_successful,
                    "dribble_success_rate": round(record.dribble_success_rate, 4),
                    "tackles_attempted": record.tackles_attempted,
                    "tackles_won": record.tackles_won,
                    "tackle_success_rate": round(record.tackle_success_rate, 4)
                }
                writer.writerow(row)

        return str(filepath)

    # =========================================================================
    # JSON EXPORT (for deep learning)
    # =========================================================================

    def export_actions_json(self, filename: str = "actions_dataset.json") -> str:
        """Export action data to JSON for deep learning."""
        filepath = self.output_dir / filename

        data = {
            "metadata": {
                "exported_at": datetime.now().isoformat(),
                "total_samples": len(self.action_data),
                "action_types": list(set(a.action_type for a in self.action_data)),
                "schema_version": "1.0"
            },
            "data": [asdict(action) for action in self.action_data]
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

        return str(filepath)

    def export_tactical_json(self, filename: str = "tactical_dataset.json") -> str:
        """Export tactical data to JSON."""
        filepath = self.output_dir / filename

        data = {
            "metadata": {
                "exported_at": datetime.now().isoformat(),
                "total_snapshots": len(self.tactical_data),
                "formations": list(set(t.formation for t in self.tactical_data)),
                "schema_version": "1.0"
            },
            "data": [
                {
                    **asdict(snapshot),
                    "ball_position": list(snapshot.ball_position) if snapshot.ball_position else None
                }
                for snapshot in self.tactical_data
            ]
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

        return str(filepath)

    # =========================================================================
    # PYTORCH DATASET FORMAT
    # =========================================================================

    def export_pytorch_format(self, filename: str = "pytorch_dataset.json") -> str:
        """
        Export in format easily loadable by PyTorch Dataset.

        Structure:
        {
            "samples": [
                {
                    "id": ...,
                    "features": [...],  # Numerical features
                    "labels": {...},    # Target labels
                    "metadata": {...}   # Non-training info
                }
            ]
        }
        """
        filepath = self.output_dir / filename

        samples = []
        for action in self.action_data:
            # Extract numerical features
            features = [
                action.overall_score,
                action.technical_score,
                action.tactical_score,
                action.physical_score,
                action.psychological_score,
                action.body_position_score or 0.5,
                action.timing_score or 0.5,
                action.decision_score or 0.5,
                action.end_frame - action.start_frame,  # Duration
            ]

            # One-hot encode action type
            action_types = ["pass", "shot", "dribble", "tackle", "cross", "header", "save"]
            action_type_onehot = [1 if t in action.action_type.lower() else 0 for t in action_types]
            features.extend(action_type_onehot)

            # Labels
            labels = {
                "outcome": 1 if action.action_outcome == "success" else 0,
                "overall_score": action.overall_score,
                "four_corner": {
                    "technical": action.technical_score,
                    "tactical": action.tactical_score,
                    "physical": action.physical_score,
                    "psychological": action.psychological_score
                }
            }

            samples.append({
                "id": action.action_id,
                "features": features,
                "labels": labels,
                "metadata": {
                    "player_jersey": action.player_jersey,
                    "video_id": action.video_id,
                    "action_type": action.action_type
                }
            })

        data = {
            "metadata": {
                "exported_at": datetime.now().isoformat(),
                "num_samples": len(samples),
                "feature_dim": len(samples[0]["features"]) if samples else 0,
                "feature_names": [
                    "overall_score", "technical_score", "tactical_score",
                    "physical_score", "psychological_score",
                    "body_position_score", "timing_score", "decision_score",
                    "duration_frames",
                    "is_pass", "is_shot", "is_dribble", "is_tackle",
                    "is_cross", "is_header", "is_save"
                ]
            },
            "samples": samples
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

        return str(filepath)

    # =========================================================================
    # VIDEO CLIP MANIFEST (for video ML)
    # =========================================================================

    def export_clip_manifest(
        self,
        video_paths: Dict[str, str],
        filename: str = "clip_manifest.json"
    ) -> str:
        """
        Export manifest for video clip extraction.

        This allows training video models on specific action clips.

        Args:
            video_paths: Mapping of video_id to file path
        """
        filepath = self.output_dir / filename

        clips = []
        for action in self.action_data:
            if action.video_id not in video_paths:
                continue

            clip = {
                "clip_id": action.action_id,
                "video_path": video_paths[action.video_id],
                "start_frame": action.start_frame,
                "end_frame": action.end_frame,
                "action_class": action.action_type,
                "outcome": action.action_outcome,
                "labels": {
                    "overall_score": action.overall_score,
                    "technical_score": action.technical_score,
                    "tactical_score": action.tactical_score,
                    "physical_score": action.physical_score,
                    "psychological_score": action.psychological_score
                },
                "player_info": {
                    "jersey": action.player_jersey,
                    "position": action.player_position
                }
            }
            clips.append(clip)

        data = {
            "metadata": {
                "exported_at": datetime.now().isoformat(),
                "num_clips": len(clips),
                "action_classes": list(set(c["action_class"] for c in clips))
            },
            "clips": clips
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

        return str(filepath)

    # =========================================================================
    # COMPREHENSIVE EXPORT
    # =========================================================================

    def export_all(self, prefix: str = "") -> Dict[str, str]:
        """Export all data in all formats."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = f"{prefix}_" if prefix else ""

        exports = {}

        # CSV exports
        exports["actions_csv"] = self.export_actions_csv(
            f"{prefix}actions_{timestamp}.csv"
        )
        exports["tactical_csv"] = self.export_tactical_csv(
            f"{prefix}tactical_{timestamp}.csv"
        )
        exports["player_csv"] = self.export_player_records_csv(
            f"{prefix}players_{timestamp}.csv"
        )

        # JSON exports
        exports["actions_json"] = self.export_actions_json(
            f"{prefix}actions_{timestamp}.json"
        )
        exports["tactical_json"] = self.export_tactical_json(
            f"{prefix}tactical_{timestamp}.json"
        )
        exports["pytorch"] = self.export_pytorch_format(
            f"{prefix}pytorch_{timestamp}.json"
        )

        return exports

    def get_dataset_stats(self) -> Dict:
        """Get statistics about the collected dataset."""
        if not self.action_data:
            return {"error": "No data collected"}

        action_types = {}
        outcomes = {"success": 0, "failure": 0, "unknown": 0}
        scores = []

        for action in self.action_data:
            # Action types
            action_types[action.action_type] = action_types.get(action.action_type, 0) + 1

            # Outcomes
            outcome = action.action_outcome if action.action_outcome in outcomes else "unknown"
            outcomes[outcome] += 1

            scores.append(action.overall_score)

        return {
            "total_actions": len(self.action_data),
            "total_tactical_snapshots": len(self.tactical_data),
            "total_players": len(self.player_records),
            "action_type_distribution": action_types,
            "outcome_distribution": outcomes,
            "score_statistics": {
                "mean": round(np.mean(scores), 4),
                "std": round(np.std(scores), 4),
                "min": round(min(scores), 4),
                "max": round(max(scores), 4)
            },
            "unique_videos": len(set(a.video_id for a in self.action_data)),
            "unique_formations": len(set(t.formation for t in self.tactical_data)) if self.tactical_data else 0
        }

    def reset(self):
        """Clear all collected data."""
        self.action_data.clear()
        self.tactical_data.clear()
        self.player_records.clear()


# Global instance
ml_export = MLExportService()
