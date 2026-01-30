"""
Pydantic models for API request/response schemas.
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


# ============== Enums ==============

class TeamSide(str, Enum):
    HOME = "home"
    AWAY = "away"
    UNKNOWN = "unknown"


class EventType(str, Enum):
    PASS = "pass"
    SHOT = "shot"
    TACKLE = "tackle"
    INTERCEPTION = "interception"
    FOUL = "foul"
    GOAL = "goal"
    CORNER = "corner"
    THROW_IN = "throw_in"
    FREE_KICK = "free_kick"
    OFFSIDE = "offside"


class AlertPriority(str, Enum):
    IMMEDIATE = "immediate"  # < 5 sec delay needed
    TACTICAL = "tactical"  # 30-60 sec window
    STRATEGIC = "strategic"  # Half-time analysis


class ProcessingMode(str, Enum):
    LIVE = "live"
    POST_MATCH = "post_match"


class AnalysisMode(str, Enum):
    """Analysis mode determines what to focus on and processing speed."""
    FULL = "full"                    # Full analysis - highest accuracy (3 FPS)
    STANDARD = "standard"            # Standard analysis - good balance (1 FPS)
    MY_TEAM = "my_team"              # Focus on your team's performance (1 FPS)
    OPPONENT = "opponent"            # Scout opponent for weaknesses (1 FPS)
    QUICK_OVERVIEW = "quick_overview" # Fast summary (1 FPS, key stats only)
    QUICK_PREVIEW = "quick_preview"  # Ultra-fast preview (0.033 FPS - sample every 30 sec)


# ============== Base Models ==============

class Position(BaseModel):
    """2D position on the pitch in meters."""
    x: float = Field(..., description="X coordinate (0 = left goal line)")
    y: float = Field(..., description="Y coordinate (0 = bottom touchline)")


class PixelPosition(BaseModel):
    """2D position in pixel coordinates."""
    x: int
    y: int


class BoundingBox(BaseModel):
    """Bounding box for detected objects."""
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float = Field(ge=0, le=1)


class Velocity(BaseModel):
    """Velocity vector in m/s."""
    vx: float
    vy: float
    speed_kmh: float = Field(ge=0)


# ============== Detection Models ==============

class DetectedPlayer(BaseModel):
    """Single detected player in a frame."""
    track_id: int
    bbox: BoundingBox
    pixel_position: PixelPosition
    pitch_position: Optional[Position] = None
    team: TeamSide = TeamSide.UNKNOWN
    jersey_color: Optional[List[int]] = Field(None, description="RGB color")
    jersey_number: Optional[int] = Field(None, description="Recognized jersey number (1-99)")
    player_name: Optional[str] = Field(None, description="Player name from roster")
    is_goalkeeper: bool = False
    is_interpolated: bool = Field(False, description="True if position is predicted, not detected")


class DetectedBall(BaseModel):
    """Detected ball in a frame."""
    bbox: BoundingBox
    pixel_position: PixelPosition
    pitch_position: Optional[Position] = None
    velocity: Optional[Velocity] = None
    possessed_by: Optional[int] = Field(None, description="Track ID of possessing player")


class FrameDetection(BaseModel):
    """All detections for a single frame."""
    frame_number: int
    timestamp_ms: int
    players: List[DetectedPlayer]
    ball: Optional[DetectedBall] = None
    home_players: int = 0
    away_players: int = 0


# ============== Tracking Models ==============

class PlayerTrack(BaseModel):
    """Full track of a player across frames."""
    track_id: int
    team: TeamSide
    jersey_number: Optional[int] = None
    positions: List[Position]
    timestamps: List[int]
    is_goalkeeper: bool = False


# ============== Event Models ==============

class MatchEvent(BaseModel):
    """Detected match event."""
    event_id: str
    event_type: EventType
    timestamp_ms: int
    frame_number: int
    position: Position
    player_id: Optional[int] = None
    team: Optional[TeamSide] = None
    recipient_id: Optional[int] = None
    success: Optional[bool] = None
    metadata: Dict[str, Any] = {}


# ============== Analytics Models ==============

class PlayerMetrics(BaseModel):
    """Metrics for a single player."""
    track_id: int
    team: TeamSide
    distance_covered_m: float = 0.0
    sprint_count: int = 0
    sprint_distance_m: float = 0.0
    high_intensity_distance_m: float = 0.0
    max_speed_kmh: float = 0.0
    avg_speed_kmh: float = 0.0
    passes_attempted: int = 0
    passes_completed: int = 0
    touches: int = 0
    tackles: int = 0
    interceptions: int = 0


class TeamMetrics(BaseModel):
    """Aggregated metrics for a team."""
    team: TeamSide
    possession_pct: float = 0.0
    total_passes: int = 0
    pass_completion_pct: float = 0.0
    shots: int = 0
    shots_on_target: int = 0
    xg: float = 0.0
    total_distance_km: float = 0.0
    avg_formation: Optional[str] = None  # e.g., "4-3-3"


class HeatmapData(BaseModel):
    """Heatmap data for visualization."""
    player_id: Optional[int] = None
    team: Optional[TeamSide] = None
    grid: List[List[float]]  # 2D grid of position frequencies
    grid_size: tuple = (10, 7)  # Default grid divisions


class PassNetwork(BaseModel):
    """Pass connections between players."""
    team: TeamSide
    nodes: List[Dict[str, Any]]  # Player positions
    edges: List[Dict[str, Any]]  # Pass connections with weights


# ============== Coaching Models ==============

class TacticalAlert(BaseModel):
    """Real-time coaching alert."""
    alert_id: str
    priority: AlertPriority
    timestamp_ms: int
    message: str
    details: Optional[str] = None
    suggested_action: Optional[str] = None
    related_players: List[int] = []
    position: Optional[Position] = None
    expires_at_ms: Optional[int] = None


class FormationSnapshot(BaseModel):
    """Current team formation."""
    team: TeamSide
    formation: str  # e.g., "4-4-2"
    player_positions: Dict[int, Position]
    compactness: float  # Team compactness metric
    width: float  # Team width in meters
    depth: float  # Team depth in meters


# ============== Match Models ==============

class MatchInfo(BaseModel):
    """Match metadata."""
    match_id: str
    home_team: str
    away_team: str
    date: datetime
    venue: Optional[str] = None
    competition: Optional[str] = None


class MatchState(BaseModel):
    """Current match state."""
    match_id: str
    current_time_ms: int
    period: int = 1  # 1 = first half, 2 = second half
    home_score: int = 0
    away_score: int = 0
    home_metrics: TeamMetrics
    away_metrics: TeamMetrics
    recent_events: List[MatchEvent] = []
    active_alerts: List[TacticalAlert] = []


# ============== API Request/Response Models ==============

class VideoUploadResponse(BaseModel):
    """Response after video upload."""
    video_id: str
    filename: str
    duration_ms: int
    fps: float
    resolution: tuple
    status: str = "uploaded"


class ProcessingStatus(BaseModel):
    """Status of video processing job."""
    video_id: str
    status: str  # queued, processing, completed, failed
    progress_pct: float = 0.0
    current_frame: int = 0
    total_frames: int = 0
    estimated_remaining_s: Optional[int] = None
    error_message: Optional[str] = None


class FrameAnalysisRequest(BaseModel):
    """Request for single frame analysis (cloud inference)."""
    frame_data: str  # Base64 encoded image
    frame_number: int
    timestamp_ms: int
    include_ball: bool = True
    include_tracking: bool = True


class FrameAnalysisResponse(BaseModel):
    """Response from frame analysis."""
    frame_number: int
    detections: FrameDetection
    processing_time_ms: int


class CalibrationRequest(BaseModel):
    """Request to calibrate pitch mapping."""
    video_id: str
    pitch_corners: List[PixelPosition] = Field(..., min_length=4, max_length=4)
    additional_points: Optional[List[PixelPosition]] = None


class TeamSetupRequest(BaseModel):
    """Request to set up team identification."""
    video_id: str
    home_team_color: List[int] = Field(..., description="RGB color")
    away_team_color: List[int] = Field(..., description="RGB color")
    home_team_name: str
    away_team_name: str


class SubstitutionSuggestion(BaseModel):
    """AI-generated substitution suggestion."""
    player_out_id: int
    suggested_position: str
    reason: str
    fatigue_score: float
    current_performance_score: float


class MatchReport(BaseModel):
    """Post-match analysis report."""
    match_info: MatchInfo
    final_score: Dict[str, int]
    home_metrics: TeamMetrics
    away_metrics: TeamMetrics
    player_metrics: List[PlayerMetrics]
    key_events: List[MatchEvent]
    tactical_insights: List[str]
    player_ratings: Dict[int, float]


# ============== Focused Analysis Models ==============

class FocusedAnalysisRequest(BaseModel):
    """Request for focused analysis processing."""
    video_id: str
    analysis_mode: AnalysisMode = AnalysisMode.FULL
    my_team: TeamSide = TeamSide.HOME  # Which team is the user's team
    processing_mode: ProcessingMode = ProcessingMode.POST_MATCH


class PlayerImprovementSuggestion(BaseModel):
    """Improvement suggestion for a specific player."""
    player_id: int
    category: str  # "decision_making", "positioning", "passing", "defending", etc.
    observation: str  # What was observed
    suggestion: str  # How to improve
    priority: str = "medium"  # "high", "medium", "low"
    evidence_frames: List[int] = []  # Frame numbers showing the issue


class TeamImprovementArea(BaseModel):
    """Area for team improvement."""
    area: str  # "pressing", "width", "compactness", "transitions", etc.
    observation: str  # What was observed
    drill_suggestion: str  # Training drill to address it
    priority: str = "medium"


class TeamImprovementReport(BaseModel):
    """Detailed report for your own team's improvement."""
    team_name: str
    match_id: str
    analysis_summary: str

    # Team-level metrics
    team_metrics: TeamMetrics
    formation_analysis: Dict[str, Any] = {}

    # Strengths to maintain
    strengths: List[str] = []

    # Areas needing improvement
    improvement_areas: List[TeamImprovementArea] = []

    # Player-specific suggestions
    player_improvements: List[PlayerImprovementSuggestion] = []

    # Training recommendations
    training_focus: List[str] = []
    recommended_drills: List[str] = []

    # Key statistics
    passing_network_issues: List[str] = []
    defensive_vulnerabilities: List[str] = []
    attacking_patterns: List[str] = []


class PlayerScoutReport(BaseModel):
    """Scouting report for an opponent player."""
    player_id: int
    position: str

    # Tendencies
    tendencies: List[str] = []  # "cuts inside", "stays wide", "drops deep"

    # Strengths to neutralize
    strengths: List[str] = []

    # Weaknesses to exploit
    weaknesses: List[str] = []

    # Key stats
    avg_position: Optional[Position] = None
    touches_per_zone: Dict[str, int] = {}  # "left", "center", "right"
    duel_win_rate: Optional[float] = None

    # Threat level (1-10)
    threat_level: int = 5

    # How to handle this player
    tactical_advice: str = ""


class TeamWeakness(BaseModel):
    """A weakness identified in the opponent team."""
    category: str  # "defensive", "transition", "pressing", "set_pieces"
    description: str
    how_to_exploit: str
    confidence: float = Field(ge=0, le=1)
    evidence: List[str] = []


class TeamPattern(BaseModel):
    """A pattern identified in the opponent team."""
    pattern_type: str  # "attacking", "defensive", "build_up", "pressing"
    description: str
    frequency: str  # "always", "often", "sometimes"
    counter_strategy: str


class OpponentScoutReport(BaseModel):
    """Scouting report for the opponent team."""
    opponent_name: str
    match_id: str
    analysis_summary: str

    # Team-level analysis
    team_metrics: TeamMetrics
    formation: str = ""

    # Patterns identified
    attacking_patterns: List[TeamPattern] = []
    defensive_patterns: List[TeamPattern] = []
    build_up_patterns: List[TeamPattern] = []

    # Weaknesses to exploit
    weaknesses: List[TeamWeakness] = []

    # Strengths to be aware of
    strengths: List[str] = []
    danger_players: List[int] = []  # Player IDs to watch

    # Player-specific scouting
    key_player_reports: List[PlayerScoutReport] = []

    # Set piece analysis
    set_piece_vulnerabilities: List[str] = []

    # Tactical recommendations
    recommended_formation: str = ""
    tactical_approach: str = ""  # "high_press", "counter_attack", "possession"

    # Specific recommendations to beat them
    tactical_recommendations: List[str] = []

    # Key battles to focus on
    key_battles: List[str] = []
