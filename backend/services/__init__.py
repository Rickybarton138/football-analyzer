"""Services package."""
from .video_ingestion import VideoIngestionService
from .detection import DetectionService
from .tracking import TrackingService
from .pitch_mapping import PitchMapper
from .ball_detection import BallDetectionService
from .event_detection import EventDetectionService
from .analytics import AnalyticsEngine

# Phase 0: New CV pipeline services
try:
    from .team_classifier import TeamClassifier
except ImportError:
    pass  # Optional: requires transformers + umap-learn

try:
    from .pitch_keypoint_detector import PitchKeypointDetector
except ImportError:
    pass  # Optional: requires trained pitch_keypoints.pt model

# Analytics services
from .xg_model import xGModel, xg_model
from .player_movement import PlayerMovementAnalyzer, player_movement_analyzer
from .tactical_intelligence import TacticalIntelligenceService, tactical_intelligence
from .pitch_visualization import PitchVisualizationService, pitch_visualization_service
from .formation_tracking import FormationTrackingService, formation_tracker
