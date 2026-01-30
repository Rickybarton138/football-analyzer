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
    DETECTION_CONFIDENCE: float = 0.5
    TRACKING_CONFIDENCE: float = 0.4
    PLAYER_CLASS_ID: int = 0  # COCO class ID for person
    SPORTS_BALL_CLASS_ID: int = 32  # COCO class ID for sports ball

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
