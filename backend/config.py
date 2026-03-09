"""
Configuration settings for the Football Match Analyzer.
"""
from pathlib import Path
from pydantic_settings import BaseSettings
from typing import Optional
import os


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    # Application
    APP_NAME: str = "Football Match Analyzer"
    DEBUG: bool = True

    # Paths
    BASE_DIR: Path = Path(__file__).parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    UPLOAD_DIR: Path = DATA_DIR / "uploads"
    FRAMES_DIR: Path = DATA_DIR / "frames"
    MODELS_DIR: Path = DATA_DIR / "models"

    # Database
    DATABASE_URL: str = "sqlite:///./data/football_analyzer.db"

    # Video Processing
    LIVE_FPS: int = 10  # Frames per second for live processing
    ANALYSIS_FPS: int = 25  # Frames per second for post-match analysis
    MAX_UPLOAD_SIZE_MB: int = 5000  # 5GB max upload
    SUPPORTED_VIDEO_FORMATS: list = [".mp4", ".avi", ".mov", ".mkv"]

    # Analysis Mode FPS Settings
    # Different modes use different FPS for speed/accuracy tradeoff
    ANALYSIS_FPS_FULL: int = 3        # Full analysis - highest accuracy (~9 hrs CPU, ~2 hrs GPU for 90min)
    ANALYSIS_FPS_STANDARD: int = 1    # Standard analysis - good balance (~3 hrs CPU, ~45 min GPU)
    ANALYSIS_FPS_FOCUSED: int = 1     # My Team / Opponent - balanced
    ANALYSIS_FPS_QUICK: int = 1       # Quick Overview - faster
    ANALYSIS_FPS_PREVIEW: float = 0.033  # Quick preview - sample every 30 sec (~6 min CPU, ~2 min GPU)

    # Detection Settings
    DETECTION_CONFIDENCE: float = 0.3      # Lowered from 0.5 to catch distant/small players
    TRACKING_CONFIDENCE: float = 0.25      # Lowered to track more players consistently
    PLAYER_CLASS_ID: int = 0  # COCO class ID for person
    SPORTS_BALL_CLASS_ID: int = 32  # COCO class ID for sports ball

    # Detection Tuning
    DETECTION_ASPECT_RATIO_MIN: float = 0.8   # Min H/W ratio for valid player bbox
    DETECTION_ASPECT_RATIO_MAX: float = 3.0   # Max H/W ratio for valid player bbox
    DETECTION_MIN_CONFIDENCE_FLOOR: float = 0.15  # Absolute min confidence for retries
    DETECTION_RETRY_DECAY: float = 0.2        # Confidence decay per retry attempt
    DETECTION_MIN_BOX_AREA: int = 400         # Min bbox area (px) for jersey color extraction
    DETECTION_IMGSZ: int = 1280               # Native YOLO inference resolution

    # YOLO Model Fallback Chain (tried in order)
    YOLO_MODEL_CHAIN: list = [
        "data/models/football_best.pt",  # Fine-tuned football model (Step 3)
        "yolo11m.pt",                    # YOLO11 medium
        "yolov8m.pt",                    # YOLOv8 medium (fallback)
    ]
    YOLO_MODEL_CHAIN_CPU: list = [
        "data/models/football_best.pt",  # Fine-tuned football model
        "yolo11s.pt",                    # YOLO11 small
        "yolov8s.pt",                    # YOLOv8 small (fallback)
    ]

    # Tracking Tuning (BoT-SORT / supervision ByteTrack params)
    TRACKING_IOU_THRESHOLD: float = 0.8       # Match threshold (1 - min_iou)
    TRACKING_MAX_INTERPOLATION_FRAMES: int = 3   # Max frames to interpolate lost tracks (keep low for panning cameras)
    TRACKING_INTERPOLATION_CONFIDENCE_DECAY: float = 0.04  # Confidence decay per interpolated frame
    TRACK_ACTIVATION_THRESHOLD: float = 0.25  # Min confidence for new track
    TRACK_LOST_BUFFER: int = 60              # Keep lost tracks for 60 frames (~2s at 30fps)
    TRACK_MINIMUM_MATCHING_THRESHOLD: float = 0.85  # Min IoU for matching

    # Team Classification Tuning
    TEAM_COLOR_DISTANCE_THRESHOLD: float = 40.0  # LAB distance for referee/unknown
    TEAM_AMBIGUITY_RATIO: float = 0.85        # min_dist/max_dist ratio above which team is ambiguous
    TEAM_TEMPORAL_SMOOTHING_WINDOW: int = 30  # Frames for majority-vote team smoothing (upgraded)
    TEAM_CONSENSUS_THRESHOLD: float = 0.6     # Min fraction for majority vote

    # SigLIP Team Classification
    SIGLIP_MODEL_NAME: str = "google/siglip-base-patch16-224"
    SIGLIP_WARMUP_FRAMES: int = 50           # Frames of crops to collect before fitting
    SIGLIP_UMAP_COMPONENTS: int = 10         # UMAP dimensionality reduction target

    # SAHI Ball Detection
    SAHI_SLICE_SIZE: int = 640               # Slice dimensions for SAHI
    SAHI_OVERLAP_RATIO: float = 0.2          # Overlap between slices
    BALL_MAX_INTERPOLATION_GAP: int = 20     # Max frames to interpolate missing ball

    # SAHI Player Detection
    SAHI_PLAYER_DETECTION: bool = True        # Enable SAHI for player detection
    SAHI_PLAYER_MIN_TRIGGER: int = 18         # Only run SAHI if standard YOLO finds fewer than this
    SAHI_PLAYER_SLICE_SIZE: int = 640         # Tile size (can differ from ball detection)
    SAHI_PLAYER_OVERLAP: float = 0.2          # Tile overlap ratio
    SAHI_PLAYER_CONFIDENCE: float = 0.25      # Lower confidence for small players

    # Pitch Keypoint Detection
    PITCH_KEYPOINT_MODEL_PATH: str = "data/models/pitch_keypoints.pt"
    PITCH_NUM_KEYPOINTS: int = 32            # Standard pitch keypoints

    # Cloud Inference
    CLOUD_INFERENCE_ENABLED: bool = False
    CLOUD_INFERENCE_URL: Optional[str] = None
    CLOUD_API_KEY: Optional[str] = None
    CLOUD_TIMEOUT_MS: int = 500  # Max latency for cloud inference

    # Local Processing
    USE_GPU: bool = False  # Set to True if CUDA available
    CPU_THREADS: int = 4
    BATCH_SIZE: int = 1  # Frames to process at once

    # Pitch Dimensions (standard football pitch in meters)
    PITCH_LENGTH: float = 105.0
    PITCH_WIDTH: float = 68.0

    # Clip Analysis Settings
    CLIP_ANALYSIS_FPS: int = 10           # FPS for per-clip detection (lower = faster)
    CLIP_FIRST_TOUCH_THRESHOLD: float = 0.8  # Seconds — elite first touch < 0.5s
    CLIP_PROGRESSIVE_CARRY_DISTANCE: float = 10.0  # Meters forward to count as progressive carry
    CLIP_PRESSING_INTENSITY_WINDOW: float = 3.0     # Seconds to measure pressing distance closed
    CLIP_DECISION_SPEED_ELITE: float = 0.5           # Elite decision speed in seconds

    # Analytics
    SPRINT_THRESHOLD_KMH: float = 25.0  # Speed threshold for sprint detection
    HIGH_INTENSITY_THRESHOLD_KMH: float = 19.0
    PRESSING_DISTANCE_M: float = 5.0  # Distance to trigger pressing alert

    # WebSocket
    WS_HEARTBEAT_INTERVAL: int = 30

    # Redis (optional, for caching)
    REDIS_URL: Optional[str] = None

    # Vision AI (Gemini)
    GEMINI_API_KEY: Optional[str] = None
    GEMINI_MODEL: str = "gemini-2.0-flash"  # Fast and capable for video analysis

    # OpenAI (alternative to Gemini)
    OPENAI_API_KEY: Optional[str] = None

    # Anthropic (Claude Vision)
    ANTHROPIC_API_KEY: Optional[str] = None

    # AI Coaching Engine
    AI_COACHING_PROVIDER: str = "claude"
    AI_COACHING_MODEL: str = "claude-sonnet-4-20250514"
    AI_COACHING_MAX_TOKENS: int = 2000
    AI_VISION_PROVIDER: str = "openai"
    AI_VISION_MODEL: str = "gpt-4o"
    AI_VISION_MAX_TOKENS: int = 1500

    # Off-Screen Player Prediction
    PREDICTION_VELOCITY_DECAY_RATE: float = 0.02       # Velocity halves at ~25 frames, zero at ~50
    PREDICTION_FORMATION_BLEND_FRAMES: int = 90         # Frames to fully blend to formation template (~3s)
    PREDICTION_CONFIDENCE_DECAY: float = 0.5            # Exponential decay rate for confidence
    PREDICTION_MIN_CONFIDENCE: float = 0.1              # Floor confidence for long off-screen players
    PREDICTION_REENTRY_THRESHOLD: float = 20.0          # Max distance for re-entry matching (0-100 coords)
    PREDICTION_FRUSTUM_MARGIN: float = 5.0              # Margin added to detected player bounding box (%)

    # Chunked Upload
    UPLOAD_CHUNK_SIZE_MB: int = 50              # Size per chunk
    UPLOAD_MAX_FILE_SIZE_MB: int = 20000        # 20GB max (covers 4K VEO)
    UPLOAD_SESSION_TIMEOUT_HOURS: int = 24      # Stale session cleanup
    UPLOAD_TEMP_DIR: Path = DATA_DIR / "upload_chunks"

    # URL Import
    IMPORT_DOWNLOAD_TIMEOUT_S: int = 3600       # 1 hour max download

    # VEO API Integration
    VEO_API_BASE_URL: str = "https://api.veo.co.uk/api"
    VEO_API_TOKEN: Optional[str] = None  # User provides their VEO bearer token

    # AI Jersey Detection Settings
    AI_JERSEY_DETECTION_ENABLED: bool = True
    AI_JERSEY_PROVIDER: str = "openai"  # "openai" or "claude"
    AI_JERSEY_BATCH_SIZE: int = 6  # Players per API call
    AI_JERSEY_MIN_CONFIDENCE: float = 0.6  # Minimum confidence to accept
    AI_JERSEY_MIN_OBSERVATIONS: int = 3  # Observations needed to confirm
    AI_JERSEY_PROCESS_INTERVAL: int = 30  # Process every N frames
    AI_JERSEY_MIN_BBOX_HEIGHT: int = 50  # Skip small bounding boxes

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()

# Ensure directories exist
settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
settings.FRAMES_DIR.mkdir(parents=True, exist_ok=True)
settings.MODELS_DIR.mkdir(parents=True, exist_ok=True)
settings.UPLOAD_TEMP_DIR.mkdir(parents=True, exist_ok=True)
