"""
Football Match Analyzer - Main FastAPI Application

A computer vision-powered application for analyzing football matches,
providing both real-time coaching assistance and post-match analytics.
"""
import asyncio
import uuid
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import json

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
import aiofiles

from config import settings
from models.schemas import (
    VideoUploadResponse, ProcessingStatus, MatchState, MatchInfo,
    TeamSetupRequest, CalibrationRequest, FrameDetection, TacticalAlert,
    ProcessingMode, AnalysisMode, TeamSide, TeamImprovementReport,
    OpponentScoutReport, DetectedBall, Position
)
from models.player_identity import (
    PlayerIdentity, PlayerLabelRequest, player_identity_db
)
from services.video_ingestion import VideoIngestionService
from services.detection import DetectionService
from services.tracking import TrackingService
from services.pitch_mapping import PitchMapper
from services.ball_detection import BallDetectionService
from services.event_detection import EventDetectionService, event_detection_service
from services.ai_jersey_detection import ai_jersey_detection_service
from services.analytics import AnalyticsEngine
from ai.tactical_analyzer import TacticalAnalyzer
from ai.recommendation import RecommendationEngine
from ai.alerts import AlertManager

# Initialize FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    description="AI-powered football match analysis and coaching system",
    version="1.0.0"
)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "http://localhost:3002", "http://localhost:3003", "http://localhost:5173", "http://localhost:3004", "http://localhost:3005", "http://127.0.0.1:3000", "http://127.0.0.1:3001", "http://127.0.0.1:3002", "http://127.0.0.1:3003", "http://127.0.0.1:3004", "http://127.0.0.1:3005"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for video serving
app.mount("/static", StaticFiles(directory=str(settings.DATA_DIR)), name="static")
# Mount uploads directory for direct JSON access
app.mount("/uploads", StaticFiles(directory=str(settings.UPLOAD_DIR)), name="uploads")

# ============== Global State ==============

# Active WebSocket connections
active_connections: Dict[str, List[WebSocket]] = {}

# Processing jobs status
processing_jobs: Dict[str, ProcessingStatus] = {}

# Match metadata from upload form (team names, colors, formations)
match_metadata_store: Dict[str, Dict] = {}

# Active match states
match_states: Dict[str, MatchState] = {}

# Service instances (lazy loaded)
_services: Dict[str, any] = {}

# Analysis cache to avoid reprocessing on every request
_analysis_cache: Dict[str, Dict] = {}


def get_services():
    """Lazy load services to avoid initialization overhead."""
    if not _services:
        _services["video"] = VideoIngestionService()
        _services["detection"] = DetectionService()
        _services["tracking"] = TrackingService()
        _services["pitch_mapper"] = PitchMapper()
        _services["ball"] = BallDetectionService()
        _services["events"] = EventDetectionService()
        _services["analytics"] = AnalyticsEngine()
        _services["tactical"] = TacticalAnalyzer()
        _services["recommendations"] = RecommendationEngine()
        _services["alerts"] = AlertManager()
    return _services


# ============== WebSocket Manager ==============

class ConnectionManager:
    """Manage WebSocket connections for real-time updates."""

    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, match_id: str):
        await websocket.accept()
        if match_id not in self.active_connections:
            self.active_connections[match_id] = []
        self.active_connections[match_id].append(websocket)

    def disconnect(self, websocket: WebSocket, match_id: str):
        if match_id in self.active_connections:
            self.active_connections[match_id].remove(websocket)
            if not self.active_connections[match_id]:
                del self.active_connections[match_id]

    async def broadcast(self, match_id: str, message: dict):
        if match_id in self.active_connections:
            for connection in self.active_connections[match_id]:
                try:
                    await connection.send_json(message)
                except Exception:
                    pass

    async def send_personal(self, websocket: WebSocket, message: dict):
        await websocket.send_json(message)


manager = ConnectionManager()


# ============== API Routes ==============

@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "app": settings.APP_NAME,
        "version": "1.0.0"
    }


@app.get("/api/status")
async def get_status():
    """Get system status and capabilities including hardware detection."""
    import torch
    import platform
    import psutil

    # Detailed GPU detection
    gpu_info = {
        "available": False,
        "name": None,
        "memory_gb": None,
        "cuda_version": None
    }

    try:
        if torch.cuda.is_available():
            gpu_info["available"] = True
            gpu_info["name"] = torch.cuda.get_device_name(0)
            gpu_info["memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 1)
            gpu_info["cuda_version"] = torch.version.cuda
    except Exception:
        pass

    # CPU info
    cpu_info = {
        "cores": psutil.cpu_count(logical=False) or 1,
        "threads": psutil.cpu_count(logical=True) or 1,
        "name": platform.processor() or "Unknown"
    }

    # Memory info
    mem = psutil.virtual_memory()
    memory_info = {
        "total_gb": round(mem.total / (1024**3), 1),
        "available_gb": round(mem.available / (1024**3), 1)
    }

    # Processing time estimates (for 90-minute match)
    processing_estimates = {
        "quick_preview": {
            "description": "Quick preview - samples every 30 seconds",
            "frames_per_90min": 180,
            "cpu_minutes": 6,
            "gpu_minutes": 2
        },
        "standard": {
            "description": "Standard analysis at 1 FPS",
            "frames_per_90min": 5400,
            "cpu_minutes": 180,
            "gpu_minutes": 45
        },
        "full": {
            "description": "Full analysis at 3 FPS",
            "frames_per_90min": 16200,
            "cpu_minutes": 540,
            "gpu_minutes": 135
        }
    }

    # Recommend processing mode based on hardware
    if gpu_info["available"]:
        recommended_mode = "full"
        recommendation = f"GPU detected ({gpu_info['name']}) - full analysis recommended"
    else:
        recommended_mode = "quick_preview"
        recommendation = "No GPU detected - quick preview recommended for faster results. Full analysis may take several hours."

    return {
        "cloud_inference": settings.CLOUD_INFERENCE_ENABLED,
        "gpu": gpu_info,
        "cpu": cpu_info,
        "memory": memory_info,
        "processing_estimates": processing_estimates,
        "recommended_mode": recommended_mode,
        "recommendation": recommendation,
        "active_matches": len(match_states),
        "processing_jobs": len(processing_jobs)
    }


# ============== Video Upload & Processing ==============

@app.post("/api/video/upload", response_model=VideoUploadResponse)
async def upload_video(
    file: UploadFile = File(...),
    metadata: str = None,  # JSON string with match metadata
    background_tasks: BackgroundTasks = None
):
    """Upload a video file for analysis with optional match metadata."""
    # Validate file extension
    ext = Path(file.filename).suffix.lower()
    if ext not in settings.SUPPORTED_VIDEO_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format. Use: {settings.SUPPORTED_VIDEO_FORMATS}"
        )

    # Generate unique video ID
    video_id = str(uuid.uuid4())
    filename = f"{video_id}{ext}"
    filepath = settings.UPLOAD_DIR / filename

    # Save file asynchronously
    async with aiofiles.open(filepath, 'wb') as out_file:
        content = await file.read()
        await out_file.write(content)

    # Get video metadata
    services = get_services()
    video_meta = await services["video"].get_video_metadata(filepath)

    # Parse match metadata if provided
    match_metadata = {}
    if metadata:
        try:
            match_metadata = json.loads(metadata)
        except json.JSONDecodeError:
            pass

    # Store match metadata for use during processing
    if match_metadata:
        match_metadata_store[video_id] = match_metadata

    # Initialize processing status
    processing_jobs[video_id] = ProcessingStatus(
        video_id=video_id,
        status="uploaded",
        total_frames=video_meta.get("total_frames", 0)
    )

    return VideoUploadResponse(
        video_id=video_id,
        filename=file.filename,
        duration_ms=video_meta.get("duration_ms", 0),
        fps=video_meta.get("fps", 0),
        resolution=(video_meta.get("width", 0), video_meta.get("height", 0)),
        status="uploaded"
    )


@app.post("/api/video/{video_id}/register", response_model=VideoUploadResponse)
async def register_existing_video(video_id: str, filename: str = "video.mp4"):
    """Register a video that was manually copied to uploads folder."""
    # Find the video file
    video_path = None
    for ext in settings.SUPPORTED_VIDEO_FORMATS:
        path = settings.UPLOAD_DIR / f"{video_id}{ext}"
        if path.exists():
            video_path = path
            break

    if not video_path:
        raise HTTPException(status_code=404, detail="Video file not found in uploads folder")

    # Get video metadata
    services = get_services()
    metadata = await services["video"].get_video_metadata(video_path)

    # Initialize processing status
    processing_jobs[video_id] = ProcessingStatus(
        video_id=video_id,
        status="uploaded",
        total_frames=metadata.get("total_frames", 0)
    )

    return VideoUploadResponse(
        video_id=video_id,
        filename=filename,
        duration_ms=metadata.get("duration_ms", 0),
        fps=metadata.get("fps", 0),
        resolution=(metadata.get("width", 0), metadata.get("height", 0)),
        status="uploaded"
    )


@app.post("/api/video/{video_id}/process")
async def start_processing(
    video_id: str,
    background_tasks: BackgroundTasks,
    mode: ProcessingMode = ProcessingMode.POST_MATCH,
    analysis_mode: AnalysisMode = AnalysisMode.FULL,
    my_team: TeamSide = TeamSide.HOME
):
    """
    Start video processing job.

    Args:
        video_id: ID of the uploaded video
        mode: Processing mode (live or post_match)
        analysis_mode: Analysis focus mode
            - full: Analyze everything (slowest, most comprehensive)
            - my_team: Focus on your team's performance and improvements
            - opponent: Scout the opponent team for weaknesses
            - quick_overview: Fast summary with key stats only
        my_team: Which team is yours (home or away)
    """
    if video_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="Video not found")

    # Update status
    processing_jobs[video_id].status = "queued"
    print(f"[PROCESS] Starting background task for video {video_id}")

    # Start background processing with analysis mode
    background_tasks.add_task(process_video, video_id, mode, analysis_mode, my_team)

    return {
        "message": "Processing started",
        "video_id": video_id,
        "analysis_mode": analysis_mode.value,
        "my_team": my_team.value
    }


async def process_video(
    video_id: str,
    mode: ProcessingMode,
    analysis_mode: AnalysisMode = AnalysisMode.FULL,
    my_team: TeamSide = TeamSide.HOME
):
    """
    Background task to process video frames.

    Args:
        video_id: ID of the video to process
        mode: Processing mode (live or post_match)
        analysis_mode: Focus mode for analysis
        my_team: Which team to focus on
    """
    print(f"[PROCESS] Background task started for video {video_id}")
    services = get_services()
    job = processing_jobs[video_id]
    job.status = "processing"
    print(f"[PROCESS] Status set to 'processing', total_frames={job.total_frames}")

    # Initialize detection model if not already loaded
    if services["detection"].model is None:
        print("[PROCESS] Initializing detection model...")
        await services["detection"].initialize()
        print("[PROCESS] Detection model ready")

    try:
        # Find video file
        video_path = None
        for ext in settings.SUPPORTED_VIDEO_FORMATS:
            path = settings.UPLOAD_DIR / f"{video_id}{ext}"
            print(f"[PROCESS] Checking for video at: {path}")
            if path.exists():
                video_path = path
                break

        if not video_path:
            job.status = "failed"
            job.error_message = "Video file not found"
            print(f"[PROCESS] ERROR: Video file not found for {video_id}")
            return

        print(f"[PROCESS] Found video at: {video_path}")

        # Get source FPS to calculate expected frames
        video_metadata = await services["video"].get_video_metadata(video_path)
        source_fps = video_metadata.get("fps", 30)
        original_total_frames = video_metadata.get("total_frames", job.total_frames)
        print(f"[PROCESS] Video metadata: source_fps={source_fps}, total_frames={original_total_frames}")

        # Determine FPS based on mode and analysis_mode
        if mode == ProcessingMode.LIVE:
            target_fps = settings.LIVE_FPS
        else:
            # Use analysis mode-specific FPS for speed optimization
            if analysis_mode == AnalysisMode.FULL:
                target_fps = settings.ANALYSIS_FPS_FULL  # 3 FPS - highest accuracy
            elif analysis_mode == AnalysisMode.STANDARD:
                target_fps = settings.ANALYSIS_FPS_STANDARD  # 1 FPS - good balance
            elif analysis_mode in [AnalysisMode.MY_TEAM, AnalysisMode.OPPONENT]:
                target_fps = settings.ANALYSIS_FPS_FOCUSED  # 1 FPS
            elif analysis_mode == AnalysisMode.QUICK_PREVIEW:
                target_fps = settings.ANALYSIS_FPS_PREVIEW  # 0.033 FPS - sample every 30 sec
            else:  # QUICK_OVERVIEW
                target_fps = settings.ANALYSIS_FPS_QUICK

        # Calculate expected frames to process based on target FPS
        # This ensures progress bar works correctly for all processing modes
        if source_fps > 0 and target_fps > 0:
            frame_skip = max(1, int(source_fps / target_fps))
            expected_frames = max(1, original_total_frames // frame_skip)
        else:
            expected_frames = original_total_frames

        job.total_frames = expected_frames  # Update to expected frames for accurate progress
        print(f"[PROCESS] Target FPS: {target_fps}, frame_skip: {frame_skip if source_fps > 0 else 'N/A'}, expected_frames: {expected_frames}")

        # Extract and process frames
        frame_generator = services["video"].extract_frames(video_path, target_fps)

        all_detections = []
        frame_count = 0

        # Determine which team to focus on for detailed tracking
        focus_team = my_team if analysis_mode == AnalysisMode.MY_TEAM else (
            TeamSide.AWAY if my_team == TeamSide.HOME else TeamSide.HOME
        ) if analysis_mode == AnalysisMode.OPPONENT else None

        async for frame_data in frame_generator:
            frame_count += 1
            job.current_frame = frame_count
            job.progress_pct = (frame_count / job.total_frames) * 100 if job.total_frames > 0 else 0

            # Log progress periodically
            if frame_count % 10 == 0 or frame_count == 1:
                print(f"[PROCESS] Frame {frame_count}/{job.total_frames} ({job.progress_pct:.1f}%)")

            # Run detection
            detections = await services["detection"].detect(frame_data["frame"])

            # Run tracking
            tracked = await services["tracking"].update(detections, frame_data["frame_number"])

            # AI Jersey Detection - run on tracked players
            try:
                if settings.AI_JERSEY_DETECTION_ENABLED and settings.OPENAI_API_KEY:
                    # Initialize on first frame if needed
                    if frame_count == 1 and ai_jersey_detection_service.client is None:
                        await ai_jersey_detection_service.initialize(
                            openai_api_key=settings.OPENAI_API_KEY,
                            provider=settings.AI_JERSEY_PROVIDER
                        )
                        print("[JERSEY] AI Jersey Detection initialized")

                    # Process frame for jersey numbers
                    tracked = await ai_jersey_detection_service.process_frame(
                        frame=frame_data["frame"],
                        players=tracked,
                        frame_number=frame_data["frame_number"]
                    )
            except Exception as e:
                if frame_count <= 3:
                    print(f"[JERSEY] Warning: {e}")

            # Ball detection
            ball = await services["ball"].detect(frame_data["frame"])

            # Create frame detection object
            frame_detection = FrameDetection(
                frame_number=frame_data["frame_number"],
                timestamp_ms=frame_data["timestamp_ms"],
                players=tracked,
                ball=ball
            )

            all_detections.append(frame_detection)

            # Capture frame for training data (every processed frame)
            # This builds our ML training dataset automatically
            try:
                detection_dicts = [
                    {
                        'bbox': {'x1': p.bbox.x1, 'y1': p.bbox.y1, 'x2': p.bbox.x2, 'y2': p.bbox.y2},
                        'team': p.team,
                        'is_goalkeeper': p.is_goalkeeper,
                        'track_id': p.track_id,
                        'jersey_number': p.jersey_number,
                        'confidence': p.bbox.confidence if p.bbox else 0.85,
                    }
                    for p in tracked
                ]
                await training_data_service.capture_frame_for_training(
                    video_id=video_id,
                    frame_number=frame_data["frame_number"],
                    timestamp_ms=frame_data["timestamp_ms"],
                    frame_image=frame_data["frame"],
                    detections=detection_dicts,
                    auto_annotate=True
                )
            except Exception as e:
                print(f"[TRAINING] Warning: Failed to capture frame for training: {e}")

            # Track player moments for highlights
            try:
                ball_possessor = None
                if ball and hasattr(ball, 'possessed_by'):
                    ball_possessor = ball.possessed_by
                player_highlights_service.process_frame(
                    frame_number=frame_data["frame_number"],
                    players=tracked,
                    ball_possessed_by=ball_possessor,
                    timestamp_ms=frame_data["timestamp_ms"]
                )
            except Exception as e:
                if frame_count == 1:
                    print(f"[HIGHLIGHTS] Warning: Failed to track highlights: {e}")

            # Run event detection (passes, tackles, shots, set pieces)
            try:
                events = await event_detection_service.process_frame(
                    players=tracked,
                    ball=ball,
                    frame_number=frame_data["frame_number"],
                    timestamp_ms=frame_data["timestamp_ms"]
                )
                if events:
                    if frame_count <= 10:
                        print(f"[EVENTS] Detected {len(events)} events at frame {frame_data['frame_number']}")

                    # Record events in player highlights service for clip generation
                    # Note: player_highlights uses jersey_number as key, but we now use persistent player_ids
                    # For simplicity, we'll use track_id as a pseudo-jersey number since jersey OCR may not work
                    for event in events:
                        try:
                            # Event.player_id is the track_id
                            # Use negative track_id as pseudo-jersey to avoid collision with real jerseys
                            if event.player_id:
                                pseudo_jersey = -abs(event.player_id)
                                player_highlights_service.record_event(event, player_jersey=pseudo_jersey)
                        except Exception as e:
                            if frame_count <= 5:
                                print(f"[HIGHLIGHTS] Warning: Could not record event: {e}")
            except Exception as e:
                if frame_count == 1:
                    print(f"[EVENTS] Warning: Event detection failed: {e}")

            # Broadcast update if in live mode
            if mode == ProcessingMode.LIVE:
                await manager.broadcast(video_id, frame_detection.model_dump())

        # Set up video info for highlights service
        try:
            player_highlights_service.set_video_info(
                video_path=str(video_path),
                fps=source_fps,
                width=video_metadata.get("width", 1920),
                height=video_metadata.get("height", 1080),
                total_frames=original_total_frames
            )
            print(f"[HIGHLIGHTS] Tracked {len(player_highlights_service.players)} players during processing")
        except Exception as e:
            print(f"[HIGHLIGHTS] Warning: Could not set video info: {e}")

        # Collect events and statistics from event detection service
        try:
            all_events = event_detection_service.get_all_events() if hasattr(event_detection_service, 'get_all_events') else []
            team_stats = event_detection_service.get_team_stats() if hasattr(event_detection_service, 'get_team_stats') else {}
            print(f"[EVENTS] Collected {len(all_events)} events from processing")
        except Exception as e:
            print(f"[EVENTS] Could not collect events: {e}")
            all_events = []
            team_stats = {}

        # Save detections with metadata, events, and statistics
        output_data = {
            "detections": [d.model_dump() for d in all_detections],
            "events": all_events,
            "team_statistics": team_stats,
            "metadata": {
                "analysis_mode": analysis_mode.value,
                "my_team": my_team.value,
                "target_fps": target_fps,
                "total_frames": frame_count,
                "total_events": len(all_events)
            }
        }
        output_path = settings.FRAMES_DIR / f"{video_id}_detections.json"
        async with aiofiles.open(output_path, 'w') as f:
            await f.write(json.dumps(output_data))

        print(f"[SAVE] Saved {frame_count} frames, {len(all_events)} events to {output_path}")

        job.status = "completed"
        job.progress_pct = 100

    except Exception as e:
        import traceback
        job.status = "failed"
        job.error_message = str(e)
        print(f"Processing error: {e}")
        traceback.print_exc()


@app.get("/api/video/{video_id}/status", response_model=ProcessingStatus)
async def get_processing_status(video_id: str):
    """Get processing status for a video."""
    if video_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="Video not found")
    return processing_jobs[video_id]


@app.get("/api/jobs")
async def list_all_jobs():
    """Debug endpoint: List all processing jobs with their status."""
    return {
        "total": len(processing_jobs),
        "jobs": [
            {
                "video_id": job.video_id,
                "status": job.status,
                "progress_pct": job.progress_pct,
                "current_frame": job.current_frame,
                "total_frames": job.total_frames,
                "error_message": job.error_message
            }
            for job in processing_jobs.values()
        ]
    }


@app.get("/api/video/{video_id}/frame/{frame_number}")
async def get_frame(video_id: str, frame_number: int):
    """Get a specific frame from processed video."""
    frame_path = settings.FRAMES_DIR / video_id / f"frame_{frame_number:06d}.jpg"
    if not frame_path.exists():
        raise HTTPException(status_code=404, detail="Frame not found")
    return FileResponse(frame_path)


# ============== Match Management ==============

@app.post("/api/match/create")
async def create_match(
    home_team: str,
    away_team: str,
    video_id: Optional[str] = None,
    venue: Optional[str] = None,
    competition: Optional[str] = None
):
    """Create a new match for analysis."""
    match_id = str(uuid.uuid4())

    match_info = MatchInfo(
        match_id=match_id,
        home_team=home_team,
        away_team=away_team,
        date=datetime.now(),
        venue=venue,
        competition=competition
    )

    # Initialize match state
    from models.schemas import TeamMetrics, TeamSide
    match_states[match_id] = MatchState(
        match_id=match_id,
        current_time_ms=0,
        home_metrics=TeamMetrics(team=TeamSide.HOME),
        away_metrics=TeamMetrics(team=TeamSide.AWAY)
    )

    return {
        "match_id": match_id,
        "info": match_info.model_dump()
    }


@app.get("/api/match/{match_id}")
async def get_match(match_id: str):
    """Get current match state."""
    if match_id not in match_states:
        raise HTTPException(status_code=404, detail="Match not found")
    return match_states[match_id]


@app.post("/api/match/{match_id}/calibrate")
async def calibrate_pitch(match_id: str, request: CalibrationRequest):
    """Calibrate pitch mapping using corner points."""
    services = get_services()
    await services["pitch_mapper"].calibrate(request.pitch_corners)
    return {"message": "Pitch calibrated successfully"}


@app.post("/api/match/{match_id}/teams")
async def setup_teams(match_id: str, request: TeamSetupRequest):
    """Set up team identification by jersey colors."""
    services = get_services()
    await services["detection"].set_team_colors(
        request.home_team_color,
        request.away_team_color
    )
    return {"message": "Team colors configured"}


# ============== Analytics Endpoints ==============

@app.get("/api/match/{match_id}/analytics")
async def get_analytics(match_id: str):
    """Get comprehensive match analytics."""
    if match_id not in match_states:
        raise HTTPException(status_code=404, detail="Match not found")

    services = get_services()
    state = match_states[match_id]

    return {
        "home_metrics": state.home_metrics.model_dump(),
        "away_metrics": state.away_metrics.model_dump(),
        "match_time_ms": state.current_time_ms
    }


@app.get("/api/match/{match_id}/heatmap/{player_id}")
async def get_player_heatmap(match_id: str, player_id: int):
    """Get heatmap data for a specific player."""
    services = get_services()
    heatmap = await services["analytics"].get_player_heatmap(match_id, player_id)
    return heatmap.model_dump()


@app.get("/api/match/{match_id}/pass-network")
async def get_pass_network(match_id: str, team: str):
    """Get pass network visualization data."""
    services = get_services()
    from models.schemas import TeamSide
    team_side = TeamSide.HOME if team == "home" else TeamSide.AWAY
    network = await services["analytics"].get_pass_network(match_id, team_side)
    return network.model_dump()


@app.get("/api/match/{match_id}/events")
async def get_events(match_id: str, event_type: Optional[str] = None):
    """Get match events, optionally filtered by type."""
    if match_id not in match_states:
        raise HTTPException(status_code=404, detail="Match not found")

    state = match_states[match_id]
    events = state.recent_events

    if event_type:
        events = [e for e in events if e.event_type.value == event_type]

    return {"events": [e.model_dump() for e in events]}


# ============== Coaching & Alerts ==============

@app.get("/api/match/{match_id}/alerts")
async def get_alerts(match_id: str):
    """Get active coaching alerts."""
    if match_id not in match_states:
        raise HTTPException(status_code=404, detail="Match not found")

    state = match_states[match_id]
    return {"alerts": [a.model_dump() for a in state.active_alerts]}


@app.post("/api/match/{match_id}/alerts/{alert_id}/dismiss")
async def dismiss_alert(match_id: str, alert_id: str):
    """Dismiss a coaching alert."""
    if match_id not in match_states:
        raise HTTPException(status_code=404, detail="Match not found")

    state = match_states[match_id]
    state.active_alerts = [a for a in state.active_alerts if a.alert_id != alert_id]
    return {"message": "Alert dismissed"}


# ============== Jersey Detection Endpoints ==============

@app.get("/api/match/{match_id}/jersey-detections")
async def get_jersey_detections(match_id: str):
    """
    Get all detected jersey numbers for a match.

    Returns both confirmed and pending detections from the AI jersey detection service.
    """
    # Get all detections from the AI service
    detections = ai_jersey_detection_service.get_all_detections()
    stats = ai_jersey_detection_service.get_stats()

    return {
        "match_id": match_id,
        "detections": detections,
        "statistics": stats,
        "message": f"Found {len(detections)} player jersey detections"
    }


@app.post("/api/match/{match_id}/jersey-correction")
async def correct_jersey_number(
    match_id: str,
    track_id: int,
    jersey_number: int
):
    """
    Manually correct a player's jersey number.

    Use this endpoint when the AI detection is wrong or when you want to
    manually assign a jersey number to a player.

    Args:
        match_id: The match ID (for context)
        track_id: The player's tracking ID
        jersey_number: The correct jersey number (1-99)
    """
    # Validate jersey number
    if jersey_number < 1 or jersey_number > 99:
        raise HTTPException(
            status_code=400,
            detail="Jersey number must be between 1 and 99"
        )

    # Set the manual correction
    ai_jersey_detection_service.set_manual_correction(track_id, jersey_number)

    return {
        "match_id": match_id,
        "track_id": track_id,
        "jersey_number": jersey_number,
        "message": f"Jersey number for player {track_id} corrected to #{jersey_number}"
    }


@app.get("/api/match/{match_id}/jersey-stats")
async def get_jersey_detection_stats(match_id: str):
    """
    Get statistics about jersey number detection for a match.

    Returns information about API calls made, detection success rates,
    and the current state of player identification.
    """
    stats = ai_jersey_detection_service.get_stats()

    return {
        "match_id": match_id,
        "provider": stats.get("provider", "none"),
        "api_calls": stats.get("api_calls", 0),
        "total_players_processed": stats.get("total_players_processed", 0),
        "successful_detections": stats.get("successful_detections", 0),
        "confirmed_players": stats.get("confirmed_players", 0),
        "pending_observations": stats.get("pending_observations", 0),
        "manual_corrections": stats.get("manual_corrections", 0),
        "detection_rate": (
            stats.get("successful_detections", 0) /
            max(stats.get("total_players_processed", 1), 1) * 100
        )
    }


@app.post("/api/jersey-detection/initialize")
async def initialize_jersey_detection(
    openai_api_key: Optional[str] = None,
    anthropic_api_key: Optional[str] = None,
    provider: str = "openai"
):
    """
    Initialize or reinitialize the AI jersey detection service.

    Use this endpoint to set up jersey detection with API keys or
    to switch between providers (OpenAI/Claude).

    Args:
        openai_api_key: OpenAI API key for GPT-4V (optional, uses settings if not provided)
        anthropic_api_key: Anthropic API key for Claude Vision (optional)
        provider: Primary provider to use ("openai" or "claude")
    """
    success = await ai_jersey_detection_service.initialize(
        openai_api_key=openai_api_key,
        anthropic_api_key=anthropic_api_key,
        provider=provider
    )

    if success:
        return {
            "status": "initialized",
            "provider": ai_jersey_detection_service.provider,
            "message": f"AI jersey detection initialized with {ai_jersey_detection_service.provider}"
        }
    else:
        raise HTTPException(
            status_code=400,
            detail="Failed to initialize jersey detection. Please provide valid API keys."
        )


@app.post("/api/jersey-detection/reset")
async def reset_jersey_detection():
    """
    Reset all jersey detection data.

    Clears all observations, confirmed players, and manual corrections.
    Use this when starting analysis of a new match.
    """
    ai_jersey_detection_service.reset()

    return {
        "status": "reset",
        "message": "Jersey detection service reset successfully"
    }


# ============== WebSocket Endpoints ==============

@app.websocket("/ws/match/{match_id}")
async def websocket_match(websocket: WebSocket, match_id: str):
    """WebSocket connection for real-time match updates."""
    await manager.connect(websocket, match_id)
    try:
        while True:
            # Receive messages from client (e.g., control commands)
            data = await websocket.receive_json()

            if data.get("type") == "ping":
                await manager.send_personal(websocket, {"type": "pong"})

            elif data.get("type") == "subscribe":
                # Client wants specific data streams
                pass

    except WebSocketDisconnect:
        manager.disconnect(websocket, match_id)


@app.websocket("/ws/live")
async def websocket_live_stream(websocket: WebSocket):
    """WebSocket for live video stream processing."""
    await websocket.accept()
    services = get_services()

    # Initialize detection model if not already loaded
    if services["detection"].model is None:
        await services["detection"].initialize()

    try:
        while True:
            # Receive frame data
            data = await websocket.receive_bytes()

            # Process frame
            import numpy as np
            import cv2
            nparr = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            # Run detection pipeline
            detections = await services["detection"].detect(frame)
            tracked = await services["tracking"].update(detections, 0)
            ball = await services["ball"].detect(frame)

            # Generate alerts
            alerts = await services["tactical"].analyze(tracked, ball)

            # Send results back
            await websocket.send_json({
                "players": [p.model_dump() for p in tracked],
                "ball": ball.model_dump() if ball else None,
                "alerts": [a.model_dump() for a in alerts]
            })

    except WebSocketDisconnect:
        pass


# ============== Live Game Management (VEO Integration) ==============

from services.live_stream import live_game_manager, live_stream_service, LiveStreamConfig, StreamType

@app.post("/api/live/start")
async def start_live_session(
    stream_url: str,
    stream_type: str = "rtsp",
    target_fps: float = 2.0
):
    """
    Start a live game session from a VEO camera stream.

    Args:
        stream_url: RTSP or HLS stream URL from VEO camera
        stream_type: "rtsp", "hls", or "file" (for testing)
        target_fps: Processing FPS (1-3 recommended for real-time)

    Returns:
        Session ID and WebSocket URL for live updates
    """
    # Inject detection services
    services = get_services()
    live_game_manager.detection_service = services["detection"]
    live_game_manager.tracking_service = services["tracking"]
    live_game_manager.event_service = event_detection_service

    # Initialize detection if needed
    if services["detection"].model is None:
        await services["detection"].initialize()

    result = await live_game_manager.start_session(
        stream_url=stream_url,
        stream_type=stream_type,
        target_fps=target_fps
    )

    return result


@app.post("/api/live/stop")
async def stop_live_session():
    """Stop the current live game session."""
    await live_game_manager.stop_session()
    return {"status": "success", "message": "Live session stopped"}


@app.get("/api/live/status")
async def get_live_status():
    """Get current live stream status and metrics."""
    return {
        "is_active": live_game_manager.is_active,
        "session_id": live_game_manager.session_id,
        "stream": live_stream_service.get_status(),
        "stats": live_game_manager.get_live_stats()
    }


@app.get("/api/live/stats")
async def get_live_stats():
    """Get current live match statistics."""
    return live_game_manager.get_live_stats()


@app.post("/api/live/score")
async def update_live_score(home: int, away: int):
    """Manually update the score during a live game."""
    live_game_manager.update_score(home, away)
    return {
        "status": "success",
        "score": {"home": home, "away": away}
    }


@app.post("/api/live/period")
async def set_live_period(period: int):
    """Set the current match period (1 = first half, 2 = second half)."""
    if period not in [1, 2]:
        raise HTTPException(status_code=400, detail="Period must be 1 or 2")
    live_game_manager.set_period(period)
    return {"status": "success", "period": period}


@app.websocket("/ws/live-coaching/{session_id}")
async def websocket_live_coaching(websocket: WebSocket, session_id: str):
    """
    WebSocket for real-time coaching alerts and updates.

    Sends:
    - frame_update: Player positions and ball location
    - alert: Tactical coaching alerts
    - stats_update: Possession and other stats
    - event: Detected match events
    """
    if session_id != live_game_manager.session_id:
        await websocket.close(code=4004, reason="Invalid session ID")
        return

    await websocket.accept()
    live_game_manager.websocket_connections.append(websocket)

    try:
        # Send initial stats
        await websocket.send_json({
            "type": "stats_update",
            "stats": live_game_manager.get_live_stats()
        })

        while True:
            # Receive any client messages
            try:
                data = await asyncio.wait_for(websocket.receive_json(), timeout=0.1)

                # Handle client messages
                if data.get("type") == "dismiss_alert":
                    # Handle alert dismissal
                    pass
                elif data.get("type") == "update_score":
                    live_game_manager.update_score(
                        data.get("home", 0),
                        data.get("away", 0)
                    )
                elif data.get("type") == "add_note":
                    # Store coaching note
                    pass

            except asyncio.TimeoutError:
                pass

            # Get latest frame and process
            frame_data = live_stream_service.get_latest_frame()
            if frame_data:
                result = await live_game_manager.process_frame_async(frame_data)

                # Send frame update
                await websocket.send_json({
                    "type": "frame_update",
                    **result
                })

                # Send alerts
                for alert in result.get("alerts", []):
                    await websocket.send_json({
                        "type": "alert",
                        "alert": alert
                    })

                # Send events
                for event in result.get("events", []):
                    await websocket.send_json({
                        "type": "event",
                        "event": event
                    })

            # Send stats update periodically
            await websocket.send_json({
                "type": "stats_update",
                "stats": live_game_manager.get_live_stats()
            })

            await asyncio.sleep(0.5)  # 2 updates per second

    except WebSocketDisconnect:
        live_game_manager.websocket_connections.remove(websocket)


# ============== Post-Match Analysis ==============

@app.get("/api/match/{match_id}/report")
async def generate_report(match_id: str):
    """Generate post-match analysis report."""
    if match_id not in match_states:
        raise HTTPException(status_code=404, detail="Match not found")

    services = get_services()
    report = await services["analytics"].generate_report(match_id)
    return report.model_dump()


@app.get("/api/match/{match_id}/my-team-report")
async def get_my_team_report(
    match_id: str,
    my_team: TeamSide = TeamSide.HOME
):
    """
    Generate focused improvement report for your team.

    Returns detailed analysis of your team's performance including:
    - Strengths to maintain
    - Areas needing improvement with specific drill suggestions
    - Player-specific improvement recommendations
    - Training focus areas

    Args:
        match_id: ID of the match to analyze
        my_team: Which team is yours (home or away)
    """
    if match_id not in match_states:
        raise HTTPException(status_code=404, detail="Match not found")

    services = get_services()
    report = await services["analytics"].generate_team_improvement_report(match_id, my_team)
    return report.model_dump()


@app.get("/api/match/{match_id}/opponent-report")
async def get_opponent_report(
    match_id: str,
    my_team: TeamSide = TeamSide.HOME
):
    """
    Generate scouting report for the opponent team.

    Returns tactical analysis of the opponent including:
    - Their strengths to be aware of
    - Weaknesses to exploit with specific tactics
    - Key player analysis
    - Recommended formation and approach to beat them

    Args:
        match_id: ID of the match to analyze
        my_team: Which team is yours (opponent is the other team)
    """
    if match_id not in match_states:
        raise HTTPException(status_code=404, detail="Match not found")

    services = get_services()
    opponent_team = TeamSide.AWAY if my_team == TeamSide.HOME else TeamSide.HOME
    report = await services["analytics"].generate_opponent_scout_report(match_id, opponent_team)
    return report.model_dump()


@app.get("/api/match/{match_id}/highlights/{player_id}")
async def get_player_highlights(match_id: str, player_id: int):
    """Get highlight timestamps for a player."""
    services = get_services()
    highlights = await services["analytics"].get_player_highlights(match_id, player_id)
    return {"highlights": highlights}


# ============== RTMP Stream (for VEO) ==============

@app.post("/api/stream/start")
async def start_rtmp_listener(stream_url: str, match_id: str):
    """Start listening to RTMP stream from VEO."""
    services = get_services()
    await services["video"].start_rtmp_stream(stream_url, match_id)
    return {"message": "RTMP listener started", "match_id": match_id}


@app.post("/api/stream/stop")
async def stop_rtmp_listener(match_id: str):
    """Stop RTMP stream listener."""
    services = get_services()
    await services["video"].stop_rtmp_stream(match_id)
    return {"message": "RTMP listener stopped"}


# ============== Training Data Collection ==============

from services.training_data import (
    training_data_service,
    MatchAnnotation,
    PlayerAnnotation,
    EventAnnotation,
    FrameAnnotation
)


@app.get("/api/training/stats")
async def get_training_stats():
    """Get statistics about the training dataset."""
    return await training_data_service.get_dataset_stats()


@app.get("/api/training/matches")
async def list_training_matches():
    """List all matches in the training dataset."""
    matches = await training_data_service.list_matches()
    return {"matches": [m.__dict__ for m in matches]}


@app.post("/api/training/matches")
async def add_training_match(
    home_team: str,
    away_team: str,
    home_score: int,
    away_score: int,
    date: str,
    video_path: Optional[str] = None,
    competition: Optional[str] = None,
    venue: Optional[str] = None,
    home_possession: Optional[float] = None,
    away_possession: Optional[float] = None,
    home_shots: Optional[int] = None,
    away_shots: Optional[int] = None,
    home_shots_on_target: Optional[int] = None,
    away_shots_on_target: Optional[int] = None,
    home_passes: Optional[int] = None,
    away_passes: Optional[int] = None,
    home_pass_accuracy: Optional[float] = None,
    away_pass_accuracy: Optional[float] = None,
    home_corners: Optional[int] = None,
    away_corners: Optional[int] = None,
    home_fouls: Optional[int] = None,
    away_fouls: Optional[int] = None,
    home_xg: Optional[float] = None,
    away_xg: Optional[float] = None
):
    """Add a new match with stats to the training dataset."""
    match = MatchAnnotation(
        match_id=str(uuid.uuid4()),
        video_path=video_path or "",
        home_team=home_team,
        away_team=away_team,
        final_score={"home": home_score, "away": away_score},
        date=date,
        competition=competition,
        venue=venue,
        home_possession=home_possession,
        away_possession=away_possession,
        home_shots=home_shots,
        away_shots=away_shots,
        home_shots_on_target=home_shots_on_target,
        away_shots_on_target=away_shots_on_target,
        home_passes=home_passes,
        away_passes=away_passes,
        home_pass_accuracy=home_pass_accuracy,
        away_pass_accuracy=away_pass_accuracy,
        home_corners=home_corners,
        away_corners=away_corners,
        home_fouls=home_fouls,
        away_fouls=away_fouls,
        home_xg=home_xg,
        away_xg=away_xg
    )
    match_id = await training_data_service.add_match(match)
    return {"match_id": match_id, "message": "Match added to training dataset"}


@app.post("/api/training/matches/bulk")
async def add_training_matches_bulk(matches: List[Dict]):
    """Add multiple matches at once via JSON body."""
    added = []
    for match_data in matches:
        match = MatchAnnotation(
            match_id=match_data.get("match_id", str(uuid.uuid4())),
            video_path=match_data.get("video_path", ""),
            home_team=match_data["home_team"],
            away_team=match_data["away_team"],
            final_score=match_data.get("final_score", {"home": 0, "away": 0}),
            date=match_data["date"],
            competition=match_data.get("competition"),
            venue=match_data.get("venue"),
            home_possession=match_data.get("home_possession"),
            away_possession=match_data.get("away_possession"),
            home_shots=match_data.get("home_shots"),
            away_shots=match_data.get("away_shots"),
            home_shots_on_target=match_data.get("home_shots_on_target"),
            away_shots_on_target=match_data.get("away_shots_on_target"),
            home_passes=match_data.get("home_passes"),
            away_passes=match_data.get("away_passes"),
            home_pass_accuracy=match_data.get("home_pass_accuracy"),
            away_pass_accuracy=match_data.get("away_pass_accuracy"),
            home_corners=match_data.get("home_corners"),
            away_corners=match_data.get("away_corners"),
            home_fouls=match_data.get("home_fouls"),
            away_fouls=match_data.get("away_fouls"),
            home_xg=match_data.get("home_xg"),
            away_xg=match_data.get("away_xg")
        )
        match_id = await training_data_service.add_match(match)
        added.append(match_id)
    return {"added": len(added), "match_ids": added}


@app.get("/api/training/matches/{match_id}")
async def get_training_match(match_id: str):
    """Get a specific match from the training dataset."""
    match = await training_data_service.get_match(match_id)
    if not match:
        raise HTTPException(status_code=404, detail="Match not found")
    players = await training_data_service.get_players(match_id)
    events = await training_data_service.get_events(match_id)
    return {
        "match": match.__dict__,
        "players": [p.__dict__ for p in players],
        "events": [e.__dict__ for e in events]
    }


@app.post("/api/training/matches/{match_id}/players")
async def add_training_players(match_id: str, players: List[Dict]):
    """Add player stats to a match."""
    count = await training_data_service.add_players_bulk(match_id, players)
    return {"added": count}


@app.post("/api/training/matches/{match_id}/events")
async def add_training_events(match_id: str, events: List[Dict]):
    """Add events to a match."""
    count = await training_data_service.add_events_bulk(match_id, events)
    return {"added": count}


@app.post("/api/training/upload/csv")
async def upload_training_csv(
    file: UploadFile = File(...),
    data_type: str = "matches"  # matches, players, or events
):
    """Upload training data from CSV file."""
    import csv
    import io

    content = await file.read()
    text = content.decode('utf-8')
    reader = csv.DictReader(io.StringIO(text))

    rows = list(reader)
    count = 0

    if data_type == "matches":
        for row in rows:
            match = MatchAnnotation(
                match_id=row.get("match_id", str(uuid.uuid4())),
                video_path=row.get("video_path", ""),
                home_team=row["home_team"],
                away_team=row["away_team"],
                final_score={
                    "home": int(row.get("home_score", 0)),
                    "away": int(row.get("away_score", 0))
                },
                date=row["date"],
                competition=row.get("competition"),
                venue=row.get("venue"),
                home_possession=float(row["home_possession"]) if row.get("home_possession") else None,
                away_possession=float(row["away_possession"]) if row.get("away_possession") else None,
                home_shots=int(row["home_shots"]) if row.get("home_shots") else None,
                away_shots=int(row["away_shots"]) if row.get("away_shots") else None,
                home_xg=float(row["home_xg"]) if row.get("home_xg") else None,
                away_xg=float(row["away_xg"]) if row.get("away_xg") else None
            )
            await training_data_service.add_match(match)
            count += 1

    elif data_type == "players":
        # Group by match_id and add
        from collections import defaultdict
        by_match = defaultdict(list)
        for row in rows:
            by_match[row["match_id"]].append(row)
        for match_id, player_rows in by_match.items():
            count += await training_data_service.add_players_bulk(match_id, player_rows)

    elif data_type == "events":
        from collections import defaultdict
        by_match = defaultdict(list)
        for row in rows:
            by_match[row["match_id"]].append(row)
        for match_id, event_rows in by_match.items():
            count += await training_data_service.add_events_bulk(match_id, event_rows)

    return {"uploaded": count, "data_type": data_type}


@app.post("/api/training/upload/json")
async def upload_training_json(file: UploadFile = File(...)):
    """Upload training data from JSON file (full dataset format)."""
    content = await file.read()
    data = json.loads(content.decode('utf-8'))

    stats = {"matches": 0, "players": 0, "events": 0}

    # Process matches
    for match_data in data.get("matches", []):
        match = MatchAnnotation(**match_data)
        await training_data_service.add_match(match)
        stats["matches"] += 1

    # Process players
    for match_id, players in data.get("players", {}).items():
        stats["players"] += await training_data_service.add_players_bulk(match_id, players)

    # Process events
    for match_id, events in data.get("events", {}).items():
        stats["events"] += await training_data_service.add_events_bulk(match_id, events)

    return {"uploaded": stats}


@app.get("/api/training/export/json")
async def export_training_json():
    """Export all training data as JSON."""
    path = await training_data_service.export_to_json()
    return FileResponse(path, filename="training_data.json")


@app.get("/api/training/export/csv/{data_type}")
async def export_training_csv(data_type: str):
    """Export training data as CSV (matches, players, or events)."""
    if data_type == "matches":
        path = await training_data_service.export_matches_csv()
    elif data_type == "players":
        path = await training_data_service.export_players_csv()
    elif data_type == "events":
        path = await training_data_service.export_events_csv()
    else:
        raise HTTPException(status_code=400, detail="Invalid data type")

    return FileResponse(path, filename=f"{data_type}.csv")


@app.get("/api/training/export/yolo")
async def export_yolo_dataset(
    output_name: str = "football_detection",
    train_split: float = 0.8
):
    """Export frame annotations in YOLO format for model training."""
    stats = await training_data_service.export_yolo_dataset(
        output_name=output_name,
        train_split=train_split
    )
    return {
        "message": "YOLO dataset exported successfully",
        "stats": stats
    }


# ============== Frame Annotation Endpoints ==============

@app.get("/api/training/frames")
async def list_training_frames(video_id: Optional[str] = None):
    """List captured training frames."""
    if video_id:
        frames = await training_data_service.list_frames_for_video(video_id)
    else:
        # List all frames across all videos
        all_frames = []
        for vid, frames_list in training_data_service.frames.items():
            for f in frames_list:
                all_frames.append({
                    'frame_id': f.frame_id,
                    'video_id': vid,
                    'frame_number': f.frame_number,
                    'timestamp_seconds': f.timestamp_seconds,
                    'annotation_count': len(f.annotations),
                })
        frames = all_frames

    return {"frames": frames, "total": len(frames)}


@app.get("/api/training/frame/{frame_id}")
async def get_training_frame(frame_id: str):
    """Get frame annotation details."""
    frame = await training_data_service.get_frame_for_annotation(frame_id)
    if not frame:
        raise HTTPException(status_code=404, detail="Frame not found")
    return frame


@app.get("/api/training/frame/{frame_id}/image")
async def get_training_frame_image(frame_id: str):
    """Get the actual frame image."""
    # Find the frame
    for video_id, frames in training_data_service.frames.items():
        for frame in frames:
            if frame.frame_id == frame_id:
                image_path = Path(frame.image_path)
                if image_path.exists():
                    return FileResponse(
                        str(image_path),
                        media_type="image/jpeg",
                        filename=f"{frame_id}.jpg"
                    )
                raise HTTPException(status_code=404, detail="Image file not found")

    raise HTTPException(status_code=404, detail="Frame not found")


@app.put("/api/training/frame/{frame_id}/annotations")
async def update_frame_annotations(frame_id: str, annotations: List[Dict]):
    """Update annotations for a frame (manual correction)."""
    success = await training_data_service.update_frame_annotation(frame_id, annotations)
    if not success:
        raise HTTPException(status_code=404, detail="Frame not found")
    return {"message": "Annotations updated", "frame_id": frame_id}


@app.post("/api/training/capture-frame")
async def capture_frame_manually(
    video_id: str,
    frame_number: int,
    timestamp_ms: int
):
    """Manually capture a specific frame from a video for annotation."""
    import cv2

    # Find the video file
    video_path = None
    for ext in settings.SUPPORTED_VIDEO_FORMATS:
        path = settings.UPLOAD_DIR / f"{video_id}{ext}"
        if path.exists():
            video_path = path
            break

    if not video_path:
        raise HTTPException(status_code=404, detail="Video not found")

    # Extract the frame
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_MSEC, timestamp_ms)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise HTTPException(status_code=400, detail="Could not extract frame")

    # Capture for training (no auto-annotations - manual annotation expected)
    frame_id = await training_data_service.capture_frame_for_training(
        video_id=video_id,
        frame_number=frame_number,
        timestamp_ms=timestamp_ms,
        frame_image=frame,
        detections=[],  # Empty - manual annotation
        auto_annotate=False
    )

    return {
        "message": "Frame captured for annotation",
        "frame_id": frame_id,
        "image_url": f"/api/training/frame/{frame_id}/image"
    }


@app.post("/api/training/quick-extract")
async def quick_extract_frames(
    video_id: Optional[str] = None,
    frame_interval: int = 60,  # Extract every N frames (60 = ~2 sec at 30fps)
    max_frames: int = 300,
    pre_annotate: bool = True,  # Run YOLO to pre-fill boxes
    background_tasks: BackgroundTasks = None
):
    """
    Quick frame extraction for ML training - NO full analysis.

    This is much faster than full video processing:
    - Extracts frames at regular intervals
    - Optionally runs quick YOLO detection to pre-fill bounding boxes
    - Frames are saved for manual annotation/correction

    Args:
        video_id: Video to extract from (uses most recent if not specified)
        frame_interval: Extract every N frames (default 60 = ~2 sec at 30fps)
        max_frames: Maximum frames to extract (default 300)
        pre_annotate: Run YOLO detection to pre-fill boxes (default True)

    Returns:
        Extraction status and frame count
    """
    import cv2

    # Find video
    if not video_id:
        # Use most recent uploaded video
        videos = list(settings.UPLOAD_DIR.glob("*.*"))
        videos = [v for v in videos if v.suffix.lower() in settings.SUPPORTED_VIDEO_FORMATS]
        if not videos:
            raise HTTPException(status_code=404, detail="No videos found")
        video_path = max(videos, key=lambda p: p.stat().st_mtime)
        video_id = video_path.stem
    else:
        video_path = None
        for ext in settings.SUPPORTED_VIDEO_FORMATS:
            path = settings.UPLOAD_DIR / f"{video_id}{ext}"
            if path.exists():
                video_path = path
                break
        if not video_path:
            raise HTTPException(status_code=404, detail="Video not found")

    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise HTTPException(status_code=400, detail="Could not open video")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    # Calculate frame indices to extract
    frame_indices = list(range(0, total_frames, frame_interval))[:max_frames]

    print(f"[QUICK EXTRACT] Starting extraction of {len(frame_indices)} frames from {video_id}")
    print(f"[QUICK EXTRACT] Pre-annotate: {pre_annotate}")

    # Initialize detector if pre-annotating
    detector = None
    if pre_annotate:
        try:
            from services.detection import detection_service
            detector = detection_service
        except Exception as e:
            print(f"[QUICK EXTRACT] Warning: Could not load detector: {e}")
            pre_annotate = False

    extracted_count = 0

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()

        if not ret:
            continue

        timestamp_ms = int((idx / fps) * 1000)

        # Get pre-annotations if enabled
        detections = []
        if pre_annotate and detector:
            try:
                # Run quick detection
                det_result = detector.detect_frame(frame)
                if det_result and 'players' in det_result:
                    for player in det_result['players']:
                        bbox = player.get('bbox', {})
                        detections.append({
                            'class_name': 'goalkeeper' if player.get('is_goalkeeper') else 'player',
                            'bbox': {
                                'x1': bbox.get('x1', 0),
                                'y1': bbox.get('y1', 0),
                                'x2': bbox.get('x2', 0),
                                'y2': bbox.get('y2', 0),
                                'confidence': bbox.get('confidence', 0.5)
                            },
                            'team': player.get('team', 'unknown'),
                            'track_id': player.get('track_id')
                        })

                # Add ball if detected
                if det_result.get('ball'):
                    ball = det_result['ball']
                    bbox = ball.get('bbox', {})
                    detections.append({
                        'class_name': 'ball',
                        'bbox': {
                            'x1': bbox.get('x1', 0),
                            'y1': bbox.get('y1', 0),
                            'x2': bbox.get('x2', 0),
                            'y2': bbox.get('y2', 0),
                            'confidence': bbox.get('confidence', 0.5)
                        }
                    })
            except Exception as e:
                print(f"[QUICK EXTRACT] Detection error on frame {idx}: {e}")

        # Save frame for training
        try:
            frame_id = await training_data_service.capture_frame_for_training(
                video_id=video_id,
                frame_number=idx,
                timestamp_ms=timestamp_ms,
                frame_image=frame,
                detections=detections,
                auto_annotate=pre_annotate
            )
            extracted_count += 1

            if extracted_count % 50 == 0:
                print(f"[QUICK EXTRACT] Extracted {extracted_count}/{len(frame_indices)} frames...")

        except Exception as e:
            print(f"[QUICK EXTRACT] Error saving frame {idx}: {e}")

    cap.release()

    print(f"[QUICK EXTRACT] Complete! Extracted {extracted_count} frames")

    return {
        "status": "complete",
        "video_id": video_id,
        "frames_extracted": extracted_count,
        "pre_annotated": pre_annotate,
        "message": f"Extracted {extracted_count} frames. Go to Training Data tab to review and correct annotations."
    }


@app.get("/api/training/quick-extract/status")
async def get_extraction_status():
    """Get current extraction progress (for future async implementation)."""
    # For now, extraction is synchronous
    # Could be extended to track background task progress
    return {"status": "idle", "message": "No extraction in progress"}


# ============== Data Scraping Endpoints ==============

from services.data_scraper import scraper


@app.get("/api/scraper/sources")
async def list_scraper_sources():
    """List available data sources for scraping."""
    return {
        "sources": [
            {
                "name": "Football-Data.co.uk",
                "type": "csv",
                "description": "Historical match results with basic stats (free)",
                "leagues": {
                    "E0": "Premier League",
                    "E1": "Championship",
                    "SP1": "La Liga",
                    "D1": "Bundesliga",
                    "I1": "Serie A",
                    "F1": "Ligue 1"
                },
                "requires_api_key": False
            },
            {
                "name": "Understat",
                "type": "web",
                "description": "xG data and shot maps (free)",
                "leagues": ["EPL", "La_Liga", "Bundesliga", "Serie_A", "Ligue_1"],
                "requires_api_key": False
            },
            {
                "name": "FBref",
                "type": "web",
                "description": "Comprehensive match and player stats (free)",
                "requires_api_key": False
            },
            {
                "name": "API-Football",
                "type": "api",
                "description": "Real-time data, lineups, events (free tier: 100 req/day)",
                "requires_api_key": True
            }
        ],
        "note": "Commercial sources like Opta and StatsBomb require paid licenses"
    }


@app.post("/api/scraper/football-data")
async def scrape_football_data(
    league: str,
    season: str,
    import_to_training: bool = True
):
    """
    Scrape historical match data from Football-Data.co.uk.

    Args:
        league: League code (E0=EPL, SP1=La Liga, D1=Bundesliga, etc.)
        season: Season code (e.g., "2324" for 2023/24)
        import_to_training: Whether to import to training dataset
    """
    matches = await scraper.scrape_football_data_csv(league, season)

    if not matches:
        raise HTTPException(status_code=404, detail="No data found for this league/season")

    result = {"matches_found": len(matches)}

    if import_to_training:
        league_names = {
            "E0": "Premier League",
            "E1": "Championship",
            "SP1": "La Liga",
            "D1": "Bundesliga",
            "I1": "Serie A",
            "F1": "Ligue 1"
        }
        imported = await scraper.import_to_training_data(
            matches,
            competition=league_names.get(league, league)
        )
        result["imported"] = imported

    return result


@app.post("/api/scraper/understat")
async def scrape_understat(
    league: str,
    season: str,
    import_to_training: bool = True
):
    """
    Scrape xG data from Understat.

    Args:
        league: League name (EPL, La_Liga, Bundesliga, Serie_A, Ligue_1)
        season: Starting year (e.g., "2023" for 2023/24)
        import_to_training: Whether to import to training dataset
    """
    matches = await scraper.scrape_understat_league(league, season)

    if not matches:
        raise HTTPException(status_code=404, detail="No data found for this league/season")

    result = {"matches_found": len(matches)}

    if import_to_training:
        league_names = {
            "EPL": "Premier League",
            "La_Liga": "La Liga",
            "Bundesliga": "Bundesliga",
            "Serie_A": "Serie A",
            "Ligue_1": "Ligue 1"
        }
        imported = await scraper.import_to_training_data(
            matches,
            competition=league_names.get(league, league)
        )
        result["imported"] = imported

    return result


@app.post("/api/scraper/fbref")
async def scrape_fbref_match(
    match_url: str,
    import_to_training: bool = True
):
    """
    Scrape a single match from FBref.

    Args:
        match_url: Full FBref match URL
        import_to_training: Whether to import to training dataset
    """
    match = await scraper.scrape_fbref_match(match_url)

    if not match:
        raise HTTPException(status_code=404, detail="Could not scrape match data")

    result = {
        "home_team": match.home_team,
        "away_team": match.away_team,
        "score": f"{match.home_score}-{match.away_score}",
        "date": match.date,
        "players_found": len(match.players) if match.players else 0
    }

    if import_to_training:
        imported = await scraper.import_to_training_data(
            [{
                "home_team": match.home_team,
                "away_team": match.away_team,
                "home_score": match.home_score,
                "away_score": match.away_score,
                "date": match.date,
                "competition": match.competition,
                **match.stats
            }],
            competition=match.competition
        )
        result["imported"] = imported

    return result


@app.post("/api/scraper/api-football")
async def scrape_api_football(
    api_key: str,
    league_id: int,
    season: int,
    import_to_training: bool = True
):
    """
    Fetch data from API-Football.

    Get free API key at: https://www.api-football.com/

    Args:
        api_key: Your API-Football API key
        league_id: League ID (39=EPL, 140=La Liga, 78=Bundesliga, etc.)
        season: Season year (e.g., 2023)
        import_to_training: Whether to import to training dataset
    """
    matches = await scraper.fetch_league_fixtures(api_key, league_id, season)

    if not matches:
        raise HTTPException(status_code=404, detail="No data found")

    result = {"matches_found": len(matches)}

    if import_to_training:
        league_names = {
            39: "Premier League",
            140: "La Liga",
            78: "Bundesliga",
            135: "Serie A",
            61: "Ligue 1"
        }
        imported = await scraper.import_to_training_data(
            matches,
            competition=league_names.get(league_id, f"League {league_id}")
        )
        result["imported"] = imported

    return result


# ============== Team Analysis Endpoints ==============

from services.team_analysis import team_analysis_service


@app.get("/api/teams")
async def list_all_teams():
    """Get list of all teams in the training data."""
    teams = await team_analysis_service.get_all_teams()
    return {"teams": teams, "count": len(teams)}


@app.post("/api/teams/set-my-team")
async def set_my_team(team_name: str):
    """Set your team for focused analysis."""
    team_analysis_service.set_my_team(team_name)
    return {"message": f"Your team set to: {team_name}"}


@app.get("/api/teams/{team_name}/profile")
async def get_team_profile(team_name: str):
    """Get comprehensive profile for a team."""
    profile = await team_analysis_service.build_team_profile(team_name)
    return {
        "team_name": profile.team_name,
        "matches_analyzed": profile.matches_analyzed,
        "record": {
            "wins": profile.wins,
            "draws": profile.draws,
            "losses": profile.losses,
            "goals_scored": profile.goals_scored,
            "goals_conceded": profile.goals_conceded,
            "goal_difference": profile.goals_scored - profile.goals_conceded
        },
        "averages": {
            "possession": profile.avg_possession,
            "shots": profile.avg_shots,
            "shots_on_target": profile.avg_shots_on_target,
            "corners": profile.avg_corners,
            "fouls": profile.avg_fouls,
            "xg": profile.avg_xg,
            "xg_against": profile.avg_xg_against
        },
        "home_record": profile.home_record,
        "away_record": profile.away_record,
        "form": profile.form,
        "scoring_patterns": profile.scoring_patterns,
        "conceding_patterns": profile.conceding_patterns
    }


@app.get("/api/analysis/my-team/{team_name}")
async def analyze_my_team(team_name: str):
    """
    Analyze your own team's performance.

    Returns:
    - Team profile and stats
    - Identified strengths and weaknesses
    - Current trends
    - Areas for improvement
    - Training focus suggestions
    """
    report = await team_analysis_service.analyze_my_team(team_name)
    return {
        "team_name": report.team_name,
        "profile": {
            "matches_analyzed": report.profile.matches_analyzed,
            "record": f"{report.profile.wins}W-{report.profile.draws}D-{report.profile.losses}L",
            "goals": f"{report.profile.goals_scored} scored, {report.profile.goals_conceded} conceded",
            "form": report.profile.form,
            "avg_possession": report.profile.avg_possession,
            "avg_shots": report.profile.avg_shots,
            "avg_xg": report.profile.avg_xg
        },
        "strengths": [
            {
                "category": s.category,
                "description": s.description,
                "confidence": s.confidence,
                "evidence": s.evidence
            }
            for s in report.strengths
        ],
        "weaknesses": [
            {
                "category": w.category,
                "description": w.description,
                "confidence": w.confidence,
                "evidence": w.evidence
            }
            for w in report.weaknesses
        ],
        "trends": report.trends,
        "improvement_areas": [
            {
                "priority": r.priority,
                "area": r.area,
                "recommendation": r.recommendation,
                "reasoning": r.reasoning
            }
            for r in report.improvement_areas
        ],
        "training_focus": report.training_focus
    }


@app.get("/api/analysis/opponent/{opponent_name}")
async def scout_opponent(opponent_name: str, my_team: Optional[str] = None):
    """
    Scout an upcoming opponent.

    Returns:
    - Opponent profile and stats
    - Identified strengths to nullify
    - Weaknesses to exploit
    - Key patterns and danger areas
    - Tactical recommendations
    - Suggested formation and approach
    """
    report = await team_analysis_service.scout_opponent(opponent_name, my_team)
    return {
        "opponent": report.team_name,
        "profile": {
            "matches_analyzed": report.profile.matches_analyzed,
            "record": f"{report.profile.wins}W-{report.profile.draws}D-{report.profile.losses}L",
            "goals": f"{report.profile.goals_scored} scored, {report.profile.goals_conceded} conceded",
            "form": report.profile.form,
            "avg_possession": report.profile.avg_possession,
            "avg_shots": report.profile.avg_shots,
            "avg_xg": report.profile.avg_xg,
            "home_record": report.profile.home_record,
            "away_record": report.profile.away_record
        },
        "their_strengths": [
            {
                "category": s.category,
                "description": s.description,
                "how_to_nullify": f"Be aware: {s.description}",
                "evidence": s.evidence
            }
            for s in report.strengths
        ],
        "their_weaknesses": [
            {
                "category": w.category,
                "description": w.description,
                "how_to_exploit": f"Target this: {w.description}",
                "evidence": w.evidence
            }
            for w in report.weaknesses
        ],
        "key_patterns": report.key_patterns,
        "danger_areas": report.danger_areas,
        "game_plan": {
            "recommended_formation": report.recommended_formation,
            "tactical_approach": report.tactical_approach,
            "key_battles": report.key_battles
        },
        "tactical_recommendations": [
            {
                "priority": r.priority,
                "area": r.area,
                "recommendation": r.recommendation,
                "reasoning": r.reasoning
            }
            for r in report.recommendations
        ]
    }


@app.get("/api/analysis/head-to-head")
async def head_to_head(team1: str, team2: str):
    """Get head-to-head record between two teams."""
    h2h = await team_analysis_service.head_to_head(team1, team2)
    return {
        "team1": team1,
        "team2": team2,
        "matches": h2h['matches'],
        "team1_wins": h2h['team1_wins'],
        "team2_wins": h2h['team2_wins'],
        "draws": h2h['draws'],
        "team1_goals": h2h['team1_goals'],
        "team2_goals": h2h['team2_goals'],
        "recent_results": h2h['results'][-5:] if h2h['results'] else []
    }


@app.get("/api/analysis/match-preview")
async def match_preview(my_team: str, opponent: str, venue: str = "home"):
    """
    Generate a complete match preview with analysis of both teams.

    Args:
        my_team: Your team name
        opponent: Opponent team name
        venue: 'home' or 'away'
    """
    my_report = await team_analysis_service.analyze_my_team(my_team)
    opp_report = await team_analysis_service.scout_opponent(opponent, my_team)
    h2h = await team_analysis_service.head_to_head(my_team, opponent)

    # Generate match-specific recommendations
    match_recommendations = []

    # Combine insights
    for opp_weakness in opp_report.weaknesses:
        for my_strength in my_report.strengths:
            if opp_weakness.category == my_strength.category:
                match_recommendations.append({
                    "type": "exploit",
                    "insight": f"Your {my_strength.description} can exploit their {opp_weakness.description}"
                })

    for opp_strength in opp_report.strengths:
        for my_weakness in my_report.weaknesses:
            if opp_strength.category == my_weakness.category:
                match_recommendations.append({
                    "type": "concern",
                    "insight": f"Watch out: their {opp_strength.description} targets your {my_weakness.description}"
                })

    return {
        "match": f"{my_team} vs {opponent}",
        "venue": venue,
        "your_team": {
            "form": my_report.profile.form,
            "strengths": [s.description for s in my_report.strengths],
            "concerns": [w.description for w in my_report.weaknesses]
        },
        "opponent": {
            "form": opp_report.profile.form,
            "threats": [s.description for s in opp_report.strengths],
            "vulnerabilities": [w.description for w in opp_report.weaknesses]
        },
        "head_to_head": {
            "played": h2h['matches'],
            "your_wins": h2h['team1_wins'],
            "their_wins": h2h['team2_wins'],
            "draws": h2h['draws']
        },
        "match_insights": match_recommendations,
        "game_plan": {
            "formation": opp_report.recommended_formation,
            "approach": opp_report.tactical_approach,
            "key_focus": opp_report.key_battles
        },
        "tactical_priorities": [
            {
                "priority": r.priority,
                "recommendation": r.recommendation
            }
            for r in opp_report.recommendations[:5]
        ]
    }


# ============== Cloud GPU Endpoints ==============

from services.cloud_gpu import cloud_gpu_service, RUNPOD_SETUP_INSTRUCTIONS, LAMBDA_SETUP_INSTRUCTIONS


@app.post("/api/cloud-gpu/configure")
async def configure_cloud_gpu(
    provider: str,
    api_key: str,
    endpoint_url: Optional[str] = None
):
    """
    Configure cloud GPU processing.

    Providers: 'runpod', 'lambda', 'custom'
    """
    result = await cloud_gpu_service.configure(provider, api_key, endpoint_url)
    return result


@app.get("/api/cloud-gpu/status")
async def get_cloud_gpu_status():
    """Get cloud GPU connection and processing status."""
    return {
        "connected": cloud_gpu_service.is_connected,
        "provider": cloud_gpu_service.config.provider if cloud_gpu_service.config else None,
        "processing": cloud_gpu_service.get_status()
    }


@app.get("/api/cloud-gpu/setup-instructions")
async def get_setup_instructions(provider: str = "runpod"):
    """Get setup instructions for cloud GPU providers."""
    if provider == "runpod":
        return {"provider": "runpod", "instructions": RUNPOD_SETUP_INSTRUCTIONS}
    elif provider == "lambda":
        return {"provider": "lambda", "instructions": LAMBDA_SETUP_INSTRUCTIONS}
    else:
        return {"error": "Unknown provider. Use 'runpod' or 'lambda'"}


@app.post("/api/cloud-gpu/process-video")
async def process_video_cloud(video_path: str, fps: int = 15):
    """
    Start processing a video on cloud GPU.

    Args:
        video_path: Path to the video file
        fps: Frames per second to analyze (default 15)
    """
    if not cloud_gpu_service.is_connected:
        return {"error": "Cloud GPU not configured. Call /api/cloud-gpu/configure first."}

    # Start processing in background
    import asyncio
    asyncio.create_task(cloud_gpu_service.process_video(video_path, fps))

    return {
        "status": "started",
        "video_path": video_path,
        "fps": fps,
        "message": "Processing started. Check /api/cloud-gpu/status for progress."
    }


# ============== Video Analysis Endpoints ==============

from services.local_processor import local_processor
from services.player_clip_analyzer import player_analyzer
from services.pass_detector import pass_detector
from services.formation_detector import formation_detector
from services.tactical_events import tactical_detector
from services.ai_coach import ai_coach
from services.pdf_report import pdf_report_generator
from services.predictive_tracker import predictive_tracker
from services.vision_ai import vision_ai_service


@app.get("/api/video/extraction-status")
async def get_extraction_status():
    """Check video extraction status."""
    import os
    clip_path = "C:/Users/info/football-analyzer/backend/uploads/balti-away-30min.mp4"

    if os.path.exists(clip_path):
        size_mb = os.path.getsize(clip_path) / (1024 * 1024)
        return {
            "status": "complete",
            "path": clip_path,
            "size_mb": round(size_mb, 1)
        }
    else:
        return {
            "status": "in_progress",
            "message": "Video extraction is still running..."
        }


@app.post("/api/video/process-local")
async def process_video_local(
    video_path: str = "C:/Users/info/football-analyzer/backend/uploads/balti-away-30min.mp4",
    fps: int = 5
):
    """
    Start local CPU video processing.

    Args:
        video_path: Path to video file
        fps: Frames per second to analyze (default 5 for CPU)
    """
    import asyncio
    import os

    if not os.path.exists(video_path):
        return {"error": f"Video not found: {video_path}"}

    if local_processor.is_processing:
        return {"error": "Already processing a video", "status": local_processor.get_status()}

    # Start processing in background
    output_path = video_path.replace('.mp4', '_analysis.json')

    async def run_processing():
        await local_processor.process_video(video_path, fps, output_path)

    asyncio.create_task(run_processing())

    return {
        "status": "started",
        "video_path": video_path,
        "fps": fps,
        "output_path": output_path,
        "message": "Processing started. Check /api/video/process-status for progress."
    }


@app.get("/api/video/process-status")
async def get_process_status():
    """Get current video processing status."""
    return local_processor.get_status()


@app.post("/api/video/reprocess")
async def reprocess_video(
    background_tasks: BackgroundTasks,
    video_id: Optional[str] = None,
    confidence_threshold: float = 0.5,
    grid_dedup_size: int = 80,
    my_team_color: Optional[str] = None,
    away_team_color: Optional[str] = None
):
    """
    Reprocess a video with improved detection settings to fix player counting issues.

    This endpoint re-runs detection on an existing video with better settings:
    - Higher confidence threshold to reduce false positives
    - Spatial deduplication to merge duplicate detections
    - Optional manual team color specification

    Args:
        video_id: ID of video to reprocess (uses most recent if not provided)
        confidence_threshold: Minimum confidence for detections (0.0-1.0, default 0.5)
        grid_dedup_size: Pixel size for spatial deduplication grid (default 80)
        my_team_color: Your team's jersey color as hex (e.g., "#FF0000" for red)
        away_team_color: Opposition jersey color as hex (e.g., "#0000FF" for blue)

    Returns:
        Processing status and job ID
    """
    import asyncio

    # Find video to reprocess
    if video_id:
        video_path = None
        for ext in settings.SUPPORTED_VIDEO_FORMATS:
            path = settings.UPLOAD_DIR / f"{video_id}{ext}"
            if path.exists():
                video_path = path
                break
        if not video_path:
            raise HTTPException(status_code=404, detail=f"Video not found for ID: {video_id}")
    else:
        # Find most recent video
        detection_files = list(settings.FRAMES_DIR.glob("*_detections.json"))
        if not detection_files:
            raise HTTPException(status_code=404, detail="No processed videos found")
        most_recent = max(detection_files, key=lambda p: p.stat().st_mtime)
        video_id = most_recent.stem.replace("_detections", "")
        video_path = None
        for ext in settings.SUPPORTED_VIDEO_FORMATS:
            path = settings.UPLOAD_DIR / f"{video_id}{ext}"
            if path.exists():
                video_path = path
                break
        if not video_path:
            raise HTTPException(status_code=404, detail=f"Original video not found for {video_id}")

    # Parse team colors if provided
    home_color_rgb = None
    away_color_rgb = None
    if my_team_color:
        try:
            hex_color = my_team_color.lstrip('#')
            home_color_rgb = [int(hex_color[i:i+2], 16) for i in (0, 2, 4)]
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid my_team_color format. Use hex like #FF0000")
    if away_team_color:
        try:
            hex_color = away_team_color.lstrip('#')
            away_color_rgb = [int(hex_color[i:i+2], 16) for i in (0, 2, 4)]
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid away_team_color format. Use hex like #0000FF")

    # Store reprocess settings
    reprocess_settings = {
        "confidence_threshold": confidence_threshold,
        "grid_dedup_size": grid_dedup_size,
        "home_color_rgb": home_color_rgb,
        "away_color_rgb": away_color_rgb
    }

    # Create/update processing job
    if video_id not in processing_jobs:
        processing_jobs[video_id] = ProcessingStatus(
            video_id=video_id,
            status="reprocessing",
            total_frames=0
        )
    else:
        processing_jobs[video_id].status = "reprocessing"

    # Start reprocessing in background
    background_tasks.add_task(
        reprocess_video_task,
        video_id,
        str(video_path),
        reprocess_settings
    )

    return {
        "status": "started",
        "video_id": video_id,
        "message": "Reprocessing started with improved settings",
        "settings": {
            "confidence_threshold": confidence_threshold,
            "grid_dedup_size": grid_dedup_size,
            "my_team_color": my_team_color,
            "away_team_color": away_team_color
        }
    }


async def reprocess_video_task(
    video_id: str,
    video_path: str,
    settings_dict: dict
):
    """
    Background task to reprocess video with improved detection settings.
    """
    import json
    import aiofiles
    from pathlib import Path

    print(f"[REPROCESS] Starting reprocess for {video_id}")
    print(f"[REPROCESS] Settings: {settings_dict}")

    services = get_services()
    job = processing_jobs.get(video_id)

    try:
        # Initialize detection model if needed
        if services["detection"].model is None:
            print("[REPROCESS] Initializing detection model...")
            await services["detection"].initialize()

        # Update detection service settings
        original_medium_thresh = services["detection"].medium_confidence_thresh
        original_low_thresh = services["detection"].low_confidence_thresh

        # Apply improved thresholds
        conf_thresh = settings_dict.get("confidence_threshold", 0.5)
        services["detection"].medium_confidence_thresh = conf_thresh
        services["detection"].low_confidence_thresh = conf_thresh * 0.8

        # Set team colors if provided
        home_rgb = settings_dict.get("home_color_rgb")
        away_rgb = settings_dict.get("away_color_rgb")
        if home_rgb and away_rgb:
            # Convert RGB to BGR for OpenCV
            home_bgr = [home_rgb[2], home_rgb[1], home_rgb[0]]
            away_bgr = [away_rgb[2], away_rgb[1], away_rgb[0]]
            await services["detection"].set_team_colors(home_bgr, away_bgr)
            event_detection_service.set_team_colors(home_bgr, away_bgr)
            print(f"[REPROCESS] Set team colors: Home={home_rgb}, Away={away_rgb}")

        # Get video metadata
        video_metadata = await services["video"].get_video_metadata(video_path)
        source_fps = video_metadata.get("fps", 30)
        total_frames = video_metadata.get("total_frames", 0)

        # Use 3 FPS for reprocessing (good balance of speed and accuracy)
        target_fps = 3
        frame_skip = max(1, int(source_fps / target_fps))
        expected_frames = max(1, total_frames // frame_skip)

        if job:
            job.total_frames = expected_frames

        print(f"[REPROCESS] Processing {expected_frames} frames at {target_fps} FPS")

        # Process frames
        frame_generator = services["video"].extract_frames(video_path, target_fps)
        all_detections = []
        frame_count = 0
        grid_size = settings_dict.get("grid_dedup_size", 80)

        # Track colors for auto-detection if not manually set
        auto_detect_colors = not (home_rgb and away_rgb)
        collected_colors = []

        async for frame_data in frame_generator:
            frame_count += 1
            if job:
                job.current_frame = frame_count
                job.progress_pct = (frame_count / expected_frames) * 100 if expected_frames > 0 else 0

            if frame_count % 100 == 0:
                print(f"[REPROCESS] Frame {frame_count}/{expected_frames} ({job.progress_pct:.1f}%)")

            # Run detection
            detections = await services["detection"].detect(frame_data["frame"])

            # Apply spatial deduplication
            spatial_grid = {}
            for det in detections:
                center_x = det.pixel_position.x
                center_y = det.pixel_position.y
                grid_x = int(center_x / grid_size)
                grid_y = int(center_y / grid_size)
                grid_key = (grid_x, grid_y)

                confidence = det.bbox.confidence if det.bbox else 0.5
                if grid_key not in spatial_grid or confidence > spatial_grid[grid_key].bbox.confidence:
                    spatial_grid[grid_key] = det

            deduplicated = list(spatial_grid.values())

            # Collect colors for auto-detection (first 50 frames)
            if auto_detect_colors and frame_count <= 50:
                for det in deduplicated:
                    if det.jersey_color:
                        collected_colors.append(det.jersey_color)

            # Auto-detect team colors after collecting enough samples
            if auto_detect_colors and frame_count == 50 and len(collected_colors) >= 20:
                try:
                    await services["detection"].auto_detect_team_colors(deduplicated)
                    print(f"[REPROCESS] Auto-detected team colors")
                except Exception as e:
                    print(f"[REPROCESS] Team color detection failed: {e}")

            # Run tracking
            tracked = await services["tracking"].update(deduplicated, frame_data["frame_number"])

            # Ball detection
            ball = await services["ball"].detect(frame_data["frame"])

            # Create frame detection
            frame_detection = FrameDetection(
                frame_number=frame_data["frame_number"],
                timestamp_ms=frame_data["timestamp_ms"],
                players=tracked,
                ball=ball
            )
            all_detections.append(frame_detection)

        # Restore original thresholds
        services["detection"].medium_confidence_thresh = original_medium_thresh
        services["detection"].low_confidence_thresh = original_low_thresh

        # Save new detections
        output_data = {
            "detections": [d.model_dump() for d in all_detections],
            "metadata": {
                "reprocessed": True,
                "confidence_threshold": conf_thresh,
                "grid_dedup_size": grid_size,
                "target_fps": target_fps,
                "total_frames": frame_count,
                "home_color": home_rgb,
                "away_color": away_rgb
            }
        }

        output_path = settings.FRAMES_DIR / f"{video_id}_detections.json"
        async with aiofiles.open(output_path, 'w') as f:
            await f.write(json.dumps(output_data))

        print(f"[REPROCESS] Complete! Saved {frame_count} frames to {output_path}")

        if job:
            job.status = "complete"
            job.progress_pct = 100

    except Exception as e:
        import traceback
        print(f"[REPROCESS] Error: {e}")
        traceback.print_exc()
        if job:
            job.status = "failed"
            job.error_message = str(e)


@app.get("/api/video/stream")
async def stream_video(video_id: Optional[str] = None):
    """Stream the match video file."""
    import os

    # If video_id provided, stream that specific video
    if video_id:
        for ext in settings.SUPPORTED_VIDEO_FORMATS:
            video_path = settings.UPLOAD_DIR / f"{video_id}{ext}"
            if video_path.exists():
                print(f"[STREAM] Serving video: {video_path}")
                return FileResponse(
                    str(video_path),
                    media_type="video/mp4",
                    filename="match.mp4"
                )
        raise HTTPException(status_code=404, detail=f"Video not found for ID: {video_id}")

    # Find the most recently uploaded video (matching the most recent analysis)
    detection_files = list(settings.FRAMES_DIR.glob("*_detections.json"))
    if detection_files:
        # Get most recent detection file to find corresponding video
        most_recent = max(detection_files, key=lambda p: p.stat().st_mtime)
        vid_id = most_recent.stem.replace("_detections", "")
        for ext in settings.SUPPORTED_VIDEO_FORMATS:
            video_path = settings.UPLOAD_DIR / f"{vid_id}{ext}"
            if video_path.exists():
                print(f"[STREAM] Serving most recent video: {video_path}")
                return FileResponse(
                    str(video_path),
                    media_type="video/mp4",
                    filename="match.mp4"
                )

    # Fall back to old hardcoded video
    video_path = "C:/Users/info/football-analyzer/backend/uploads/balti-away-30min.mp4"
    if os.path.exists(video_path):
        print(f"[STREAM] Serving fallback video: {video_path}")
        return FileResponse(
            video_path,
            media_type="video/mp4",
            filename="match.mp4"
        )

    raise HTTPException(status_code=404, detail="No video file found")


@app.get("/api/video/info")
async def get_video_info(video_id: Optional[str] = None):
    """Get video file information."""
    import os

    video_path = None

    # If video_id provided, get info for that specific video
    if video_id:
        for ext in settings.SUPPORTED_VIDEO_FORMATS:
            path = settings.UPLOAD_DIR / f"{video_id}{ext}"
            if path.exists():
                video_path = str(path)
                break
    else:
        # Find most recent video matching analysis
        detection_files = list(settings.FRAMES_DIR.glob("*_detections.json"))
        if detection_files:
            most_recent = max(detection_files, key=lambda p: p.stat().st_mtime)
            vid = most_recent.stem.replace("_detections", "")
            for ext in settings.SUPPORTED_VIDEO_FORMATS:
                path = settings.UPLOAD_DIR / f"{vid}{ext}"
                if path.exists():
                    video_path = str(path)
                    break

    # Fall back to old hardcoded video
    if not video_path:
        video_path = "C:/Users/info/football-analyzer/backend/uploads/balti-away-30min.mp4"

    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="Video file not found")

    size_mb = os.path.getsize(video_path) / (1024 * 1024)

    return {
        "path": video_path,
        "size_mb": round(size_mb, 1),
        "url": "/api/video/stream",
        "exists": True
    }


@app.get("/api/video/full-analysis")
async def get_full_analysis(video_id: Optional[str] = None):
    """Get the complete analysis JSON file for the frontend with pitch-transformed coordinates."""
    import json
    import os
    import glob

    # If video_id provided, look for that specific analysis
    if video_id:
        json_path = settings.FRAMES_DIR / f"{video_id}_detections.json"
        if not json_path.exists():
            raise HTTPException(status_code=404, detail=f"Analysis not found for video {video_id}")
    else:
        # Find the most recent analysis file
        # First check for newly processed videos in FRAMES_DIR
        detection_files = list(settings.FRAMES_DIR.glob("*_detections.json"))
        if detection_files:
            # Get most recent
            json_path = max(detection_files, key=lambda p: p.stat().st_mtime)
            print(f"[ANALYSIS] Using most recent detection file: {json_path}")
        else:
            # Fall back to old analysis file
            json_path = Path("C:/Users/info/football-analyzer/backend/uploads/balti-away-30min_analysis.json")
            print(f"[ANALYSIS] Using fallback analysis file: {json_path}")

    if os.path.exists(json_path):
        try:
            # Check cache first
            cache_key = str(json_path)
            file_mtime = os.path.getmtime(json_path)

            if cache_key in _analysis_cache:
                cached_result, cached_mtime = _analysis_cache[cache_key]
                if cached_mtime == file_mtime:
                    print(f"[ANALYSIS] Using cached analysis for {json_path.name}")
                    return cached_result

            print(f"[ANALYSIS] Processing analysis for {json_path.name} (not in cache or file modified)")

            with open(json_path, 'r') as f:
                data = json.load(f)

            # Transform detections to pitch coordinates using homography
            services = get_services()
            pitch_mapper = services["pitch_mapper"]

            # Handle new format (detections array from process_video)
            if 'detections' in data and isinstance(data['detections'], list):
                # Check if file has already been fixed - skip re-classification if so
                metadata = data.get('metadata', {})
                file_already_fixed = metadata.get('fixed', False)

                # First, collect all jersey colors for re-classification (unless already fixed)
                # This fixes issues where team colors weren't properly detected during processing
                all_colors = []
                if not file_already_fixed:
                    for det in data['detections'][:100]:  # Sample first 100 frames
                        for player in det.get('players', []):
                            jersey_color = player.get('jersey_color')
                            if jersey_color:
                                # Filter out grass-like and non-jersey colors (BGR order)
                                b, g, r = jersey_color[0], jersey_color[1], jersey_color[2]
                                # Grass: green is highest or similar to other channels
                                is_grass = (g >= r and g >= b and g > 30) or (g > r * 0.9 and g > b * 0.9 and g > 50)
                                # Brown/mud: low saturation brownish colors
                                is_brown = (abs(r - g) < 40 and abs(g - b) < 40 and r < 120 and g < 120)
                                # Very dark (shadows)
                                is_dark = (r < 35 and g < 35 and b < 35)
                                # Very bright (sky/lines)
                                is_bright = (r > 200 and g > 200 and b > 200)
                                # Gray (low saturation)
                                max_channel = max(r, g, b)
                                min_channel = min(r, g, b)
                                saturation = (max_channel - min_channel) / max(max_channel, 1)
                                is_gray = saturation < 0.2 and max_channel > 50 and max_channel < 180

                                if not is_grass and not is_brown and not is_dark and not is_bright and not is_gray:
                                    all_colors.append(jersey_color)

                # Cluster to find team colors
                home_color = None
                away_color = None
                if len(all_colors) >= 20:
                    try:
                        from sklearn.cluster import KMeans
                        import numpy as np
                        colors_array = np.array(all_colors)
                        kmeans = KMeans(n_clusters=2, n_init=10, random_state=42)
                        kmeans.fit(colors_array)
                        home_color = kmeans.cluster_centers_[0]
                        away_color = kmeans.cluster_centers_[1]
                        print(f"[ANALYSIS] Re-classified team colors: Home={home_color.astype(int).tolist()}, Away={away_color.astype(int).tolist()}")
                    except Exception as e:
                        print(f"[ANALYSIS] Team color clustering failed: {e}")

                # Get video_id for player identity tracking
                video_id_for_tracking = video_id or str(json_path.stem).replace('_detections', '')

                # INITIALIZE FROM KICKOFF - Find frame with good player distribution
                print(f"[KICKOFF] Looking for initialization frame with well-distributed players...")
                kickoff_frame_idx = 0
                best_score = 0

                for idx, det in enumerate(data['detections'][:200]):  # Check first 200 frames (first ~7 seconds)
                    players = det.get('players', [])
                    if len(players) < 20:  # Need at least 20 players visible
                        continue

                    # Calculate player distribution score (higher = more spread out)
                    pitch_x_values = []
                    for p in players:
                        bbox_data = p.get('bbox', {})
                        if isinstance(bbox_data, dict):
                            bbox = [bbox_data.get('x1', 0), bbox_data.get('y1', 0),
                                   bbox_data.get('x2', 0), bbox_data.get('y2', 0)]
                        else:
                            bbox = bbox_data if isinstance(bbox_data, list) else [0, 0, 0, 0]
                        video_x = (bbox[0] + bbox[2]) / 2
                        video_y = bbox[3]
                        px, py = pitch_mapper.video_to_pitch_normalized(video_x, video_y)
                        pitch_x_values.append(px)

                    if len(pitch_x_values) > 0:
                        # Score = standard deviation of X positions (higher = more spread)
                        import statistics
                        spread_score = statistics.stdev(pitch_x_values) if len(pitch_x_values) > 1 else 0
                        # Multiply by player count to prefer frames with more players
                        score = spread_score * len(players)

                        if score > best_score:
                            best_score = score
                            kickoff_frame_idx = idx

                # Get kickoff frame and initialize player identities
                kickoff_frame = data['detections'][kickoff_frame_idx]
                kickoff_players = kickoff_frame.get('players', [])
                print(f"[KICKOFF] Using frame {kickoff_frame_idx} with {len(kickoff_players)} players (spread score: {best_score:.1f})")

                # Prepare kickoff detections with pitch coordinates
                kickoff_detections = []
                for player in kickoff_players:
                    bbox_data = player.get('bbox', {})
                    if isinstance(bbox_data, dict):
                        bbox = [bbox_data.get('x1', 0), bbox_data.get('y1', 0),
                               bbox_data.get('x2', 0), bbox_data.get('y2', 0)]
                    else:
                        bbox = bbox_data if isinstance(bbox_data, list) else [0, 0, 0, 0]

                    video_x = (bbox[0] + bbox[2]) / 2
                    video_y = bbox[3]
                    pitch_x, pitch_y = pitch_mapper.video_to_pitch_normalized(video_x, video_y)

                    kickoff_detections.append({
                        'track_id': player.get('track_id', 0),
                        'team': player.get('team', 'unknown'),
                        'pitch_x': pitch_x,
                        'pitch_y': pitch_y,
                        'frame_number': kickoff_frame.get('frame_number', 0),
                        'jersey_color': player.get('jersey_color')
                    })

                # Initialize the 22 master player identities from kickoff
                player_identity_db.initialize_from_kickoff(
                    video_id=video_id_for_tracking,
                    kickoff_detections=kickoff_detections,
                    formation_name="4-4-2"  # Use formation set via API
                )

                # Convert new format to frame_analyses format expected by frontend
                frames = []
                for det in data['detections']:
                    frame = {
                        'frame_number': det.get('frame_number', 0),
                        'timestamp_ms': det.get('timestamp_ms', 0),
                        'detections': [],
                        'ball_position': None,
                        'home_players': 0,
                        'away_players': 0
                    }

                    # Convert player detections
                    players = det.get('players', [])

                    # If file already fixed, just use detections as-is
                    if file_already_fixed:
                        for player in players:
                            bbox_data = player.get('bbox', {})
                            if isinstance(bbox_data, dict):
                                bbox = [bbox_data.get('x1', 0), bbox_data.get('y1', 0),
                                       bbox_data.get('x2', 0), bbox_data.get('y2', 0)]
                            else:
                                bbox = bbox_data if isinstance(bbox_data, list) else [0, 0, 0, 0]

                            team = player.get('team', 'unknown')

                            # Add pitch coordinates first (needed for player identity)
                            video_x = (bbox[0] + bbox[2]) / 2
                            video_y = bbox[3]
                            pitch_x, pitch_y = pitch_mapper.video_to_pitch_normalized(video_x, video_y)

                            # Get or create player identity
                            # NOTE: pitch_mapper returns 0-100, but formation system expects 0-1
                            track_id = player.get('track_id', 0)
                            jersey_color = player.get('jersey_color')

                            # Normalize coordinates to 0-1 range
                            norm_pitch_x = pitch_x / 100.0
                            norm_pitch_y = pitch_y / 100.0

                            # CRITICAL: Flip X coordinate for away team (they defend opposite goal)
                            if team == 'away':
                                norm_pitch_x = 1.0 - norm_pitch_x

                            player_identity = player_identity_db.get_or_create_identity(
                                video_id=video_id_for_tracking,
                                team=team,
                                track_id=track_id,
                                frame_number=frame['frame_number'],
                                pitch_x=norm_pitch_x,
                                pitch_y=norm_pitch_y,
                                jersey_color=jersey_color
                            )

                            detection = {
                                'track_id': track_id,
                                'player_id': player_identity.player_id,
                                'positional_role': player_identity.positional_role,
                                'bbox': bbox,
                                'team': team,
                                'jersey_number': player_identity.jersey_number or player.get('jersey_number'),
                                'confidence': player.get('confidence', 0.5),
                                'pitch_x': round(pitch_x, 2),
                                'pitch_y': round(pitch_y, 2)
                            }

                            frame['detections'].append(detection)
                            if team == 'home':
                                frame['home_players'] += 1
                            elif team == 'away':
                                frame['away_players'] += 1
                    else:
                        # File not fixed - apply spatial deduplication and re-classification
                        grid_size = 80
                        spatial_grid = {}

                        for player in players:
                            bbox_data = player.get('bbox', {})
                            if isinstance(bbox_data, dict):
                                bbox = [bbox_data.get('x1', 0), bbox_data.get('y1', 0),
                                       bbox_data.get('x2', 0), bbox_data.get('y2', 0)]
                                confidence = bbox_data.get('confidence', 0.85)
                            else:
                                bbox = bbox_data if isinstance(bbox_data, list) else [0, 0, 0, 0]
                                confidence = 0.85

                            if confidence < 0.45:
                                continue

                            center_x = (bbox[0] + bbox[2]) / 2
                            center_y = (bbox[1] + bbox[3]) / 2
                            grid_x = int(center_x / grid_size)
                            grid_y = int(center_y / grid_size)
                            grid_key = (grid_x, grid_y)

                            if grid_key not in spatial_grid or confidence > spatial_grid[grid_key]['confidence']:
                                spatial_grid[grid_key] = {'player': player, 'bbox': bbox, 'confidence': confidence}

                        # Process deduplicated detections
                        for grid_key, det_info in spatial_grid.items():
                            player = det_info['player']
                            bbox = det_info['bbox']
                            confidence = det_info['confidence']

                            team = player.get('team', 'unknown')
                            jersey_color = player.get('jersey_color')
                            if home_color is not None and away_color is not None and jersey_color:
                                import numpy as np
                                color = np.array(jersey_color)
                                home_dist = np.linalg.norm(color - home_color)
                                away_dist = np.linalg.norm(color - away_color)
                                min_dist = min(home_dist, away_dist)
                                if min_dist < 100:
                                    team = 'home' if home_dist < away_dist else 'away'

                            video_x = (bbox[0] + bbox[2]) / 2
                            video_y = bbox[3]
                            pitch_x, pitch_y = pitch_mapper.video_to_pitch_normalized(video_x, video_y)

                            # Get or create player identity
                            # NOTE: pitch_mapper returns 0-100, but formation system expects 0-1
                            track_id = player.get('track_id', 0)

                            # Normalize coordinates to 0-1 range
                            norm_pitch_x = pitch_x / 100.0
                            norm_pitch_y = pitch_y / 100.0

                            # CRITICAL: Flip X coordinate for away team (they defend opposite goal)
                            if team == 'away':
                                norm_pitch_x = 1.0 - norm_pitch_x

                            player_identity = player_identity_db.get_or_create_identity(
                                video_id=video_id_for_tracking,
                                team=team,
                                track_id=track_id,
                                frame_number=frame['frame_number'],
                                pitch_x=norm_pitch_x,
                                pitch_y=norm_pitch_y,
                                jersey_color=jersey_color
                            )

                            detection = {
                                'track_id': track_id,
                                'player_id': player_identity.player_id,
                                'positional_role': player_identity.positional_role,
                                'bbox': bbox,
                                'team': team,
                                'jersey_number': player_identity.jersey_number or player.get('jersey_number'),
                                'confidence': confidence,
                                'pitch_x': round(pitch_x, 2),
                                'pitch_y': round(pitch_y, 2)
                            }

                            frame['detections'].append(detection)
                            if team == 'home':
                                frame['home_players'] += 1
                            elif team == 'away':
                                frame['away_players'] += 1

                        # Cap player counts
                        frame['home_players'] = min(frame['home_players'], 14)
                        frame['away_players'] = min(frame['away_players'], 14)

                    # Handle ball position
                    ball = det.get('ball')
                    if ball:
                        ball_pos = ball.get('position') or ball.get('pixel_position')
                        if ball_pos:
                            if isinstance(ball_pos, dict):
                                frame['ball_position'] = [ball_pos.get('x', 0), ball_pos.get('y', 0)]
                            else:
                                frame['ball_position'] = ball_pos

                            if frame['ball_position']:
                                bx, by = frame['ball_position']
                                bpx, bpy = pitch_mapper.video_to_pitch_normalized(bx, by)
                                frame['ball_pitch_x'] = round(bpx, 2)
                                frame['ball_pitch_y'] = round(bpy, 2)

                    frames.append(frame)

                # BALL POSSESSION DETECTION - Track which player has the ball
                print(f"[ANALYSIS] Detecting ball possession across {len(frames)} frames...")
                previous_possessing_player_id = None
                possession_distance_threshold = 50  # pixels - player must be within this distance to possess ball

                for frame in frames:
                    ball_pos = frame.get('ball_position')
                    if not ball_pos or len(ball_pos) < 2:
                        continue

                    bx, by = ball_pos[0], ball_pos[1]
                    closest_player_id = None
                    closest_distance = float('inf')

                    # Find closest player to ball
                    for detection in frame.get('detections', []):
                        player_id = detection.get('player_id')
                        if not player_id:
                            continue

                        # Get player position (center bottom of bbox = feet)
                        bbox = detection.get('bbox', [0, 0, 0, 0])
                        px = (bbox[0] + bbox[2]) / 2
                        py = bbox[3]

                        # Calculate distance
                        import math
                        distance = math.sqrt((px - bx)**2 + (py - by)**2)

                        if distance < closest_distance:
                            closest_distance = distance
                            closest_player_id = player_id

                    # If a player is close enough, they possess the ball
                    if closest_player_id and closest_distance < possession_distance_threshold:
                        # Check if this is a new possession (touch)
                        is_new_touch = (previous_possessing_player_id != closest_player_id)

                        # Record possession
                        player_identity_db.record_ball_possession(
                            video_id=video_id_for_tracking,
                            player_id=closest_player_id,
                            is_new_touch=is_new_touch
                        )

                        previous_possessing_player_id = closest_player_id
                    else:
                        # No one possessing the ball
                        previous_possessing_player_id = None

                print(f"[ANALYSIS] Ball possession detection complete")

                # Get player statistics from player_identity_db
                identities = player_identity_db.get_all_identities(video_id_for_tracking)
                player_stats = []
                for identity in identities:
                    player_stats.append({
                        'player_id': identity.player_id,
                        'team': identity.team,
                        'jersey_number': identity.jersey_number,
                        'positional_role': identity.positional_role,
                        'ball_touches': identity.ball_touches,
                        'frames_with_ball': identity.frames_with_ball
                    })

                # Get events - first check if they're in the file, otherwise generate them
                events = data.get('events', [])
                team_stats = data.get('team_statistics', {})

                if not events:
                    # Events not in file - need to regenerate them
                    # This happens for files processed before event saving was added
                    print(f"[ANALYSIS] No events in file, regenerating from {len(frames)} frames...")

                    # Reset event detection service
                    event_detection_service.reset()

                    # Re-process all frames through event detection
                    from models.schemas import DetectedPlayer, DetectedBall, Position, BoundingBox, PixelPosition
                    for frame in frames:
                        try:
                            # Convert frame detections to DetectedPlayer objects
                            players_list = []
                            for det in frame.get('detections', []):
                                bbox_data = det.get('bbox', [0, 0, 0, 0])
                                # Calculate pixel position (center-bottom of bbox = feet)
                                pixel_x = int((bbox_data[0] + bbox_data[2]) / 2)
                                pixel_y = int(bbox_data[3])

                                player = DetectedPlayer(
                                    bbox=BoundingBox(
                                        x1=bbox_data[0],
                                        y1=bbox_data[1],
                                        x2=bbox_data[2],
                                        y2=bbox_data[3],
                                        confidence=det.get('confidence', 0.5)
                                    ),
                                    pixel_position=PixelPosition(x=pixel_x, y=pixel_y),
                                    track_id=det.get('track_id', -1),
                                    team=det.get('team', 'unknown'),
                                    jersey_number=det.get('jersey_number'),
                                    pitch_position=Position(
                                        x=det.get('pitch_x', 0),
                                        y=det.get('pitch_y', 0)
                                    ) if det.get('pitch_x') is not None else None,
                                    jersey_color=det.get('jersey_color')
                                )
                                players_list.append(player)

                            # Convert ball detection
                            ball = None
                            ball_pos = frame.get('ball_position')
                            if ball_pos and len(ball_pos) >= 2:
                                ball_pitch_x = frame.get('ball_pitch_x')
                                ball_pitch_y = frame.get('ball_pitch_y')
                                if ball_pitch_x is not None and ball_pitch_y is not None:
                                    ball = DetectedBall(
                                        bbox=BoundingBox(x1=ball_pos[0]-5, y1=ball_pos[1]-5,
                                                        x2=ball_pos[0]+5, y2=ball_pos[1]+5, confidence=0.9),
                                        pitch_position=Position(x=ball_pitch_x, y=ball_pitch_y),
                                        pixel_position=PixelPosition(x=int(ball_pos[0]), y=int(ball_pos[1]))
                                    )

                            # Process frame through event detection
                            frame_events = await event_detection_service.process_frame(
                                players=players_list,
                                ball=ball,
                                frame_number=frame.get('frame_number', 0),
                                timestamp_ms=frame.get('timestamp_ms', 0)
                            )

                        except Exception as e:
                            if frame.get('frame_number', 0) < 10:
                                print(f"[ANALYSIS] Warning: Event detection failed for frame {frame.get('frame_number')}: {e}")

                    # Get the regenerated events
                    events = event_detection_service.get_all_events()
                    team_stats = event_detection_service.get_team_stats()
                    print(f"[ANALYSIS] Regenerated {len(events)} events from frames")

                # Return in the format expected by frontend
                result = {
                    'frame_analyses': frames,
                    'total_frames': len(frames),
                    'analyzed_frames': len(frames),
                    'metadata': data.get('metadata', {}),
                    'events': events,
                    'team_statistics': team_stats,
                    'player_statistics': player_stats
                }
                print(f"[ANALYSIS] Converted {len(frames)} frames from new format, {len(events)} events, {len(player_stats)} players")

                # Cache the result
                _analysis_cache[cache_key] = (result, file_mtime)

                return result

            # Handle old format (frames or frame_analyses)
            frames = data.get('frames', []) or data.get('frame_analyses', [])
            for frame in frames:
                detections = frame.get('detections', [])
                # Add pitch coordinates to each detection
                for det in detections:
                    bbox = det.get('bbox', [0, 0, 0, 0])
                    # Use center-bottom of bbox (player's feet)
                    video_x = (bbox[0] + bbox[2]) / 2
                    video_y = bbox[3]
                    pitch_x, pitch_y = pitch_mapper.video_to_pitch_normalized(video_x, video_y)
                    det['pitch_x'] = round(pitch_x, 2)
                    det['pitch_y'] = round(pitch_y, 2)

                # Also transform ball position if present
                ball_pos = frame.get('ball_position')
                if ball_pos:
                    ball_pitch_x, ball_pitch_y = pitch_mapper.video_to_pitch_normalized(ball_pos[0], ball_pos[1])
                    frame['ball_pitch_x'] = round(ball_pitch_x, 2)
                    frame['ball_pitch_y'] = round(ball_pitch_y, 2)

            return data
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Failed to load analysis: {str(e)}")
    else:
        raise HTTPException(status_code=404, detail="Analysis file not found")


@app.get("/api/video/analysis-result")
async def get_analysis_result():
    """Get the analysis result after processing is complete."""
    import json
    import os

    # First check if we have in-memory analysis
    if local_processor.status == "complete" and local_processor.current_analysis:
        analysis = local_processor.current_analysis
    else:
        # Try to load from saved JSON file
        json_path = "C:/Users/info/football-analyzer/backend/uploads/balti-away-30min_analysis.json"
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r') as f:
                    saved_data = json.load(f)
                # Return the saved data directly
                return {
                    "status": "complete",
                    "video_path": saved_data.get("video_path", ""),
                    "duration_minutes": saved_data.get("duration_seconds", 0) / 60,
                    "frames_analyzed": saved_data.get("analyzed_frames", 0),
                    "avg_home_players": round(saved_data.get("avg_home_players", 0), 1),
                    "avg_away_players": round(saved_data.get("avg_away_players", 0), 1),
                    "total_events": len(saved_data.get("events", [])),
                    "events": saved_data.get("events", [])[:100],  # First 100 events
                    "sample_frames": saved_data.get("frame_analyses", [])[:20]
                }
            except Exception as e:
                return {"error": f"Failed to load saved analysis: {str(e)}"}
        else:
            return {
                "status": local_processor.status,
                "message": "Processing not complete yet",
                "progress": local_processor.progress
            }

    analysis = local_processor.current_analysis
    if not analysis:
        return {"error": "No analysis available"}

    return {
        "status": "complete",
        "video_path": analysis.video_path,
        "duration_minutes": analysis.duration_seconds / 60,
        "frames_analyzed": analysis.analyzed_frames,
        "avg_home_players": round(analysis.avg_home_players, 1),
        "avg_away_players": round(analysis.avg_away_players, 1),
        "sample_frames": [
            {
                "timestamp": f.timestamp,
                "player_count": f.player_count,
                "home": f.home_players,
                "away": f.away_players,
                "ball": f.ball_position
            }
            for f in analysis.frame_analyses[::50]  # Every 50th frame
        ][:20]  # Max 20 samples
    }


# ============== Advanced Analytics Endpoints ==============

@app.get("/api/analytics/passes")
async def get_pass_analysis():
    """
    Get pass analysis including pass accuracy, forward passes, and possession.
    Analyzes the saved match data.
    """
    import json
    import os

    json_path = "C:/Users/info/football-analyzer/backend/uploads/balti-away-30min_analysis.json"
    if not os.path.exists(json_path):
        raise HTTPException(status_code=404, detail="Analysis file not found")

    try:
        with open(json_path, 'r') as f:
            data = json.load(f)

        frame_analyses = data.get('frame_analyses', []) or data.get('frames', [])
        if not frame_analyses:
            return {"error": "No frame data available"}

        # Analyze passes
        pass_stats = pass_detector.analyze_from_frames(frame_analyses)

        return {
            "status": "success",
            "pass_stats": pass_stats,
            "total_passes_detected": pass_stats.get('total_passes', 0)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/analytics/formations")
async def get_formation_analysis():
    """
    Get formation analysis for both teams.
    Detects formations like 4-4-2, 4-3-3, etc.
    """
    import json
    import os

    json_path = "C:/Users/info/football-analyzer/backend/uploads/balti-away-30min_analysis.json"
    if not os.path.exists(json_path):
        raise HTTPException(status_code=404, detail="Analysis file not found")

    try:
        with open(json_path, 'r') as f:
            data = json.load(f)

        frame_analyses = data.get('frame_analyses', []) or data.get('frames', [])
        if not frame_analyses:
            return {"error": "No frame data available"}

        # Analyze formations
        formation_stats = formation_detector.analyze_from_frames(frame_analyses)

        return {
            "status": "success",
            "formations": formation_stats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/analytics/tactical-events")
async def get_tactical_events():
    """
    Get detected tactical events including:
    - Pressing triggers
    - Dangerous attacks
    - Counter-attack opportunities
    - Shape warnings
    - High line opportunities
    """
    import json
    import os

    json_path = "C:/Users/info/football-analyzer/backend/uploads/balti-away-30min_analysis.json"
    if not os.path.exists(json_path):
        raise HTTPException(status_code=404, detail="Analysis file not found")

    try:
        with open(json_path, 'r') as f:
            data = json.load(f)

        frame_analyses = data.get('frame_analyses', []) or data.get('frames', [])
        if not frame_analyses:
            return {"error": "No frame data available"}

        # Detect tactical events
        event_summary = tactical_detector.analyze_from_frames(frame_analyses)

        return {
            "status": "success",
            "tactical_events": event_summary
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/analytics/full-report")
async def get_full_analytics():
    """
    Get comprehensive analytics including passes, formations, and tactical events.
    """
    import json
    import os

    json_path = "C:/Users/info/football-analyzer/backend/uploads/balti-away-30min_analysis.json"
    if not os.path.exists(json_path):
        raise HTTPException(status_code=404, detail="Analysis file not found")

    try:
        with open(json_path, 'r') as f:
            data = json.load(f)

        frame_analyses = data.get('frame_analyses', []) or data.get('frames', [])
        if not frame_analyses:
            return {"error": "No frame data available"}

        # Run all analyses
        pass_stats = pass_detector.analyze_from_frames(frame_analyses)
        formation_stats = formation_detector.analyze_from_frames(frame_analyses)
        tactical_events = tactical_detector.analyze_from_frames(frame_analyses)

        return {
            "status": "success",
            "match_info": {
                "video_path": data.get('video_path', ''),
                "duration_minutes": round(data.get('duration_seconds', 0) / 60, 1),
                "frames_analyzed": data.get('analyzed_frames', 0)
            },
            "passes": pass_stats,
            "formations": formation_stats,
            "tactical_events": tactical_events
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============== Match Event Detection Endpoints ==============

@app.get("/api/match-events/team-stats")
async def get_team_statistics():
    """
    Get comprehensive team statistics from event detection.

    Returns possession %, passes (total/successful/direction/by third),
    shots, tackles, headers, and set pieces for both teams.
    Stats are broken down by full match, first half, and second half.
    """
    try:
        stats = event_detection_service.get_team_stats()
        return {
            "status": "success",
            "statistics": stats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/match-events/all")
async def get_all_match_events():
    """
    Get all detected match events (passes, tackles, shots, set pieces).
    """
    try:
        events = event_detection_service.get_all_events()
        return {
            "status": "success",
            "events": events,
            "total_count": len(events)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/match-events/player/{player_id}")
async def get_player_events(player_id: int):
    """
    Get all events involving a specific player.
    """
    try:
        events = event_detection_service.get_player_events(player_id)
        return {
            "status": "success",
            "player_id": player_id,
            "events": events,
            "total_count": len(events)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/match-events/by-type/{event_type}")
async def get_events_by_type(event_type: str):
    """
    Get events filtered by type (pass, tackle, shot, corner, free_kick, etc.)
    """
    try:
        all_events = event_detection_service.get_all_events()
        filtered = [e for e in all_events if e["event_type"] == event_type]
        return {
            "status": "success",
            "event_type": event_type,
            "events": filtered,
            "total_count": len(filtered)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/match-events/reset")
async def reset_event_detection():
    """
    Reset all event detection state for a new match.
    """
    try:
        event_detection_service.reset()
        return {
            "status": "success",
            "message": "Event detection state reset"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/match-events/set-team-colors")
async def set_team_colors(
    home_color: List[int],
    away_color: List[int]
):
    """
    Manually set team colors for classification.

    Args:
        home_color: RGB values for home team jerseys [R, G, B]
        away_color: RGB values for away team jerseys [R, G, B]
    """
    try:
        event_detection_service.set_team_colors(home_color, away_color)
        return {
            "status": "success",
            "message": "Team colors set",
            "home_color": home_color,
            "away_color": away_color
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============== Predictive Player Tracking Endpoints ==============

@app.get("/api/tracking/predictive-analysis")
async def get_predictive_tracking():
    """
    Analyze player tracking with predictions for out-of-frame players.

    Uses Kalman filtering to:
    - Track players across frames with consistent IDs
    - Predict positions when players leave the camera view
    - Estimate when/where players will re-enter the frame
    - Detect temporarily occluded players
    """
    import json
    import os

    json_path = "C:/Users/info/football-analyzer/backend/uploads/balti-away-30min_analysis.json"
    if not os.path.exists(json_path):
        raise HTTPException(status_code=404, detail="Analysis file not found")

    try:
        with open(json_path, 'r') as f:
            data = json.load(f)

        frame_analyses = data.get('frame_analyses', []) or data.get('frames', [])
        if not frame_analyses:
            return {"error": "No frame data available"}

        # Run predictive tracking analysis
        tracking_result = predictive_tracker.analyze_from_frames(frame_analyses)

        # Get all current trackers for player list
        players = []
        home_count = 0
        away_count = 0
        total_visibility = 0
        visibility_count = 0

        for track_id, tracker in predictive_tracker.trackers.items():
            player_data = tracker.to_dict()

            # Add trajectory
            trajectory = tracker.get_trajectory_prediction(15)
            player_data['trajectory'] = trajectory

            # Calculate visibility rate for this player
            if tracker.total_frames_tracked > 0:
                vis_rate = (tracker.total_frames_tracked - tracker.frames_since_seen) / tracker.total_frames_tracked
                total_visibility += vis_rate
                visibility_count += 1

            # Add reentry prediction if applicable
            reentry = tracker.predict_reentry()
            if reentry:
                player_data['reentry_prediction'] = {
                    'predicted_frame': reentry.get('frames_until_reentry', 0),
                    'predicted_position': reentry.get('reentry_position', (0, 0)),
                    'entry_side': reentry.get('reentry_side', 'unknown'),
                    'confidence': reentry.get('confidence', 0)
                }
            else:
                player_data['reentry_prediction'] = None

            # Format for frontend
            players.append({
                'track_id': player_data['track_id'],
                'team': player_data['team'],
                'state': player_data['state'],
                'last_seen_frame': player_data['last_seen_frame'],
                'frames_missing': player_data['frames_since_seen'],
                'current_position': player_data['position'] if player_data['state'] == 'visible' else None,
                'predicted_position': player_data['position'] if player_data['state'] != 'visible' else None,
                'velocity': player_data['velocity'],
                'confidence': 1.0 - (player_data['frames_since_seen'] / 90) if player_data['frames_since_seen'] < 90 else 0.1,
                'reentry_prediction': player_data['reentry_prediction'],
                'trajectory': trajectory
            })

            if player_data['team'] == 'home':
                home_count += 1
            elif player_data['team'] == 'away':
                away_count += 1

        # Sort by track_id
        players.sort(key=lambda x: x['track_id'])

        avg_visibility = total_visibility / visibility_count if visibility_count > 0 else 0.0

        return {
            "total_frames_processed": len(frame_analyses),
            "total_unique_players": len(predictive_tracker.trackers),
            "home_players_tracked": home_count,
            "away_players_tracked": away_count,
            "avg_visibility_rate": avg_visibility,
            "out_of_frame_predictions_made": tracking_result['summary']['total_predicted_positions'],
            "reentry_accuracy": 0.75,  # Placeholder - would need validation data
            "players": players
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/tracking/out-of-frame")
async def get_out_of_frame_players():
    """
    Get players currently predicted to be out of frame.

    Returns:
    - List of players who have left the camera view
    - Their exit direction (left, right, top, bottom)
    - Predicted current position
    - Estimated time until re-entry (if returning)
    """
    import json
    import os

    json_path = "C:/Users/info/football-analyzer/backend/uploads/balti-away-30min_analysis.json"
    if not os.path.exists(json_path):
        raise HTTPException(status_code=404, detail="Analysis file not found")

    try:
        with open(json_path, 'r') as f:
            data = json.load(f)

        frame_analyses = data.get('frame_analyses', []) or data.get('frames', [])
        if not frame_analyses:
            return {"error": "No frame data available"}

        # Process all frames to get current state
        predictive_tracker.reset()
        for frame_data in frame_analyses:
            frame_num = frame_data.get('frame_number', 0)
            timestamp = frame_data.get('timestamp', 0)
            detections = frame_data.get('detections', [])
            predictive_tracker.process_frame(frame_num, detections, timestamp)

        # Get out of frame players and format for frontend
        out_of_frame_raw = predictive_tracker.get_out_of_frame_players()

        out_of_frame_formatted = []
        for p in out_of_frame_raw:
            # Get the tracker for more details
            tracker = predictive_tracker.trackers.get(p['track_id'])
            if tracker:
                reentry = tracker.predict_reentry()
                trajectory = tracker.get_trajectory_prediction(15)

                out_of_frame_formatted.append({
                    'track_id': p['track_id'],
                    'team': p['team'],
                    'state': p['state'],
                    'last_seen_frame': p.get('last_seen_frame', 0),
                    'frames_missing': p.get('frames_since_seen', 0),
                    'current_position': None,
                    'predicted_position': p.get('predicted_position'),
                    'velocity': p.get('velocity', [0, 0]),
                    'confidence': 1.0 - (p.get('frames_since_seen', 0) / 90),
                    'reentry_prediction': {
                        'predicted_frame': reentry.get('frames_until_reentry', 0),
                        'predicted_position': reentry.get('reentry_position', (0, 0)),
                        'entry_side': reentry.get('reentry_side', 'unknown'),
                        'confidence': reentry.get('confidence', 0)
                    } if reentry else None,
                    'trajectory': trajectory
                })

        return {
            "status": "success",
            "out_of_frame_players": out_of_frame_formatted,
            "total_out_of_frame": len(out_of_frame_formatted),
            "by_team": {
                "home": [p for p in out_of_frame_formatted if p['team'] == 'home'],
                "away": [p for p in out_of_frame_formatted if p['team'] == 'away']
            }
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/tracking/trajectories")
async def get_player_trajectories(frames_ahead: int = 15):
    """
    Get predicted trajectories for all tracked players.

    Args:
        frames_ahead: Number of frames to predict into the future (default 15 = 0.5 sec)

    Returns:
        Predicted movement paths for home and away players
    """
    import json
    import os

    json_path = "C:/Users/info/football-analyzer/backend/uploads/balti-away-30min_analysis.json"
    if not os.path.exists(json_path):
        raise HTTPException(status_code=404, detail="Analysis file not found")

    try:
        with open(json_path, 'r') as f:
            data = json.load(f)

        frame_analyses = data.get('frame_analyses', []) or data.get('frames', [])
        if not frame_analyses:
            return {"error": "No frame data available"}

        # Process all frames
        predictive_tracker.reset()
        for frame_data in frame_analyses:
            frame_num = frame_data.get('frame_number', 0)
            timestamp = frame_data.get('timestamp', 0)
            detections = frame_data.get('detections', [])
            predictive_tracker.process_frame(frame_num, detections, timestamp)

        # Get trajectories
        trajectories = predictive_tracker.get_all_trajectories(frames_ahead)

        return {
            "status": "success",
            "frames_ahead": frames_ahead,
            "trajectories": trajectories
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/tracking/player/{track_id}")
async def get_single_player_tracking(track_id: int, frames_ahead: int = 30):
    """
    Get detailed tracking info for a specific player.

    Args:
        track_id: The player's tracking ID
        frames_ahead: Frames to predict ahead

    Returns:
        Complete tracking data for the player including trajectory prediction
    """
    import json
    import os

    json_path = "C:/Users/info/football-analyzer/backend/uploads/balti-away-30min_analysis.json"
    if not os.path.exists(json_path):
        raise HTTPException(status_code=404, detail="Analysis file not found")

    try:
        with open(json_path, 'r') as f:
            data = json.load(f)

        frame_analyses = data.get('frame_analyses', []) or data.get('frames', [])
        if not frame_analyses:
            return {"error": "No frame data available"}

        # Process all frames
        predictive_tracker.reset()
        for frame_data in frame_analyses:
            frame_num = frame_data.get('frame_number', 0)
            timestamp = frame_data.get('timestamp', 0)
            detections = frame_data.get('detections', [])
            predictive_tracker.process_frame(frame_num, detections, timestamp)

        # Get tracker for this player
        tracker = predictive_tracker.trackers.get(track_id)

        if not tracker:
            raise HTTPException(status_code=404, detail=f"Player with track_id {track_id} not found")

        # Get trajectory and reentry prediction
        trajectory = tracker.get_trajectory_prediction(frames_ahead)
        reentry = tracker.predict_reentry(frames_ahead)
        player_data = tracker.to_dict()

        # Return in frontend-expected format
        return {
            'track_id': player_data['track_id'],
            'team': player_data['team'],
            'state': player_data['state'],
            'last_seen_frame': player_data['last_seen_frame'],
            'frames_missing': player_data['frames_since_seen'],
            'current_position': player_data['position'] if player_data['state'] == 'visible' else None,
            'predicted_position': player_data['position'] if player_data['state'] != 'visible' else None,
            'velocity': player_data['velocity'],
            'confidence': 1.0 - (player_data['frames_since_seen'] / 90) if player_data['frames_since_seen'] < 90 else 0.1,
            'reentry_prediction': {
                'predicted_frame': reentry.get('frames_until_reentry', 0),
                'predicted_position': reentry.get('reentry_position', (0, 0)),
                'entry_side': reentry.get('reentry_side', 'unknown'),
                'confidence': reentry.get('confidence', 0)
            } if reentry else None,
            'trajectory': trajectory
        }
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ============== AI Coaching Expert Endpoints ==============

@app.get("/api/ai-coach/analysis")
async def get_ai_coaching_analysis():
    """
    Get comprehensive AI coaching analysis with tactical insights and recommendations.

    The AI Coach analyzes:
    - Possession and passing patterns
    - Formations and team shape
    - Pressing intensity and effectiveness
    - Attacking threat and defensive vulnerabilities
    - Opposition weaknesses to exploit

    Returns actionable coaching recommendations prioritized by importance.
    """
    import json
    import os

    json_path = "C:/Users/info/football-analyzer/backend/uploads/balti-away-30min_analysis.json"
    if not os.path.exists(json_path):
        raise HTTPException(status_code=404, detail="Analysis file not found")

    try:
        with open(json_path, 'r') as f:
            data = json.load(f)

        frame_analyses = data.get('frame_analyses', []) or data.get('frames', []) or data.get('frames', [])
        if not frame_analyses:
            return {"error": "No frame data available"}

        # Run all base analyses
        pass_stats = pass_detector.analyze_from_frames(frame_analyses)
        formation_stats = formation_detector.analyze_from_frames(frame_analyses)
        tactical_events = tactical_detector.analyze_from_frames(frame_analyses)

        # Run AI coaching analysis
        coaching_analysis = ai_coach.analyze_match(
            pass_stats=pass_stats,
            formation_stats=formation_stats,
            tactical_events=tactical_events,
            frame_analyses=frame_analyses
        )

        return {
            "status": "success",
            "coaching": coaching_analysis
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/ai-coach/insights")
async def get_coaching_insights(
    category: Optional[str] = None,
    priority: Optional[str] = None
):
    """
    Get filtered coaching insights.

    Args:
        category: Filter by category (tactical, pressing, possession, defensive,
                  attacking, set_pieces, player_specific, substitution, formation, physical)
        priority: Filter by priority (critical, high, medium, low, info)
    """
    import json
    import os

    json_path = "C:/Users/info/football-analyzer/backend/uploads/balti-away-30min_analysis.json"
    if not os.path.exists(json_path):
        raise HTTPException(status_code=404, detail="Analysis file not found")

    try:
        with open(json_path, 'r') as f:
            data = json.load(f)

        frame_analyses = data.get('frame_analyses', []) or data.get('frames', [])
        if not frame_analyses:
            return {"error": "No frame data available"}

        # Run analyses
        pass_stats = pass_detector.analyze_from_frames(frame_analyses)
        formation_stats = formation_detector.analyze_from_frames(frame_analyses)
        tactical_events = tactical_detector.analyze_from_frames(frame_analyses)

        # Get AI coaching analysis
        ai_coach.analyze_match(pass_stats, formation_stats, tactical_events, frame_analyses)

        # Filter insights
        insights = ai_coach.insights

        if category:
            from services.ai_coach import InsightCategory
            try:
                cat_enum = InsightCategory(category)
                insights = [i for i in insights if i.category == cat_enum]
            except ValueError:
                pass

        if priority:
            from services.ai_coach import InsightPriority
            try:
                pri_enum = InsightPriority(priority)
                insights = [i for i in insights if i.priority == pri_enum]
            except ValueError:
                pass

        return {
            "status": "success",
            "total_insights": len(insights),
            "insights": [i.to_dict() for i in insights]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/ai-coach/summary")
async def get_match_summary():
    """
    Get the AI Coach's match summary including:
    - Overall performance rating
    - Key strengths identified
    - Areas needing improvement
    - Half-time and full-time messages for the team
    """
    import json
    import os

    json_path = "C:/Users/info/football-analyzer/backend/uploads/balti-away-30min_analysis.json"
    if not os.path.exists(json_path):
        raise HTTPException(status_code=404, detail="Analysis file not found")

    try:
        with open(json_path, 'r') as f:
            data = json.load(f)

        frame_analyses = data.get('frame_analyses', []) or data.get('frames', [])
        if not frame_analyses:
            return {"error": "No frame data available"}

        # Run analyses
        pass_stats = pass_detector.analyze_from_frames(frame_analyses)
        formation_stats = formation_detector.analyze_from_frames(frame_analyses)
        tactical_events = tactical_detector.analyze_from_frames(frame_analyses)

        # Get AI coaching analysis
        ai_coach.analyze_match(pass_stats, formation_stats, tactical_events, frame_analyses)

        if ai_coach.match_summary:
            return {
                "status": "success",
                "summary": ai_coach.match_summary.to_dict()
            }
        else:
            return {"error": "Summary not available"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/ai-coach/critical")
async def get_critical_insights():
    """
    Get only critical and high-priority coaching insights.
    These require immediate attention during a match.
    """
    import json
    import os

    json_path = "C:/Users/info/football-analyzer/backend/uploads/balti-away-30min_analysis.json"
    if not os.path.exists(json_path):
        raise HTTPException(status_code=404, detail="Analysis file not found")

    try:
        with open(json_path, 'r') as f:
            data = json.load(f)

        frame_analyses = data.get('frame_analyses', []) or data.get('frames', [])
        if not frame_analyses:
            return {"error": "No frame data available"}

        # Run analyses
        pass_stats = pass_detector.analyze_from_frames(frame_analyses)
        formation_stats = formation_detector.analyze_from_frames(frame_analyses)
        tactical_events = tactical_detector.analyze_from_frames(frame_analyses)

        # Get AI coaching analysis
        ai_coach.analyze_match(pass_stats, formation_stats, tactical_events, frame_analyses)

        critical = ai_coach.get_critical_insights()

        return {
            "status": "success",
            "critical_count": len(critical),
            "insights": [i.to_dict() for i in critical]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/ai-coach/team-talk")
async def get_team_talk(moment: str = "half_time"):
    """
    Get an AI-generated team talk for specific moments.

    Args:
        moment: 'half_time' or 'full_time'
    """
    import json
    import os

    json_path = "C:/Users/info/football-analyzer/backend/uploads/balti-away-30min_analysis.json"
    if not os.path.exists(json_path):
        raise HTTPException(status_code=404, detail="Analysis file not found")

    try:
        with open(json_path, 'r') as f:
            data = json.load(f)

        frame_analyses = data.get('frame_analyses', []) or data.get('frames', [])
        if not frame_analyses:
            return {"error": "No frame data available"}

        # Run analyses
        pass_stats = pass_detector.analyze_from_frames(frame_analyses)
        formation_stats = formation_detector.analyze_from_frames(frame_analyses)
        tactical_events = tactical_detector.analyze_from_frames(frame_analyses)

        # Get AI coaching analysis
        ai_coach.analyze_match(pass_stats, formation_stats, tactical_events, frame_analyses)

        if not ai_coach.match_summary:
            return {"error": "Summary not available"}

        if moment == "half_time":
            message = ai_coach.match_summary.half_time_message
            title = "Half-Time Team Talk"
        else:
            message = ai_coach.match_summary.full_time_message
            title = "Full-Time Analysis"

        # Get key points to emphasize
        critical = ai_coach.get_critical_insights()
        key_points = [
            {
                "priority": i.priority.value,
                "point": i.recommendation
            }
            for i in critical[:3]
        ]

        return {
            "status": "success",
            "moment": moment,
            "title": title,
            "message": message,
            "overall_rating": ai_coach.match_summary.overall_rating,
            "key_points": key_points,
            "strengths": ai_coach.match_summary.key_strengths,
            "areas_to_improve": ai_coach.match_summary.areas_to_improve
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ai-coach/chat")
async def ai_coach_chat(request: dict):
    """
    Answer natural language questions about the match analysis.

    The user can ask questions like:
    - "How was our possession?"
    - "Who was our best player?"
    - "What should we work on?"
    - "How did our pressing look?"
    - "What formation did we play?"

    Returns AI-generated answers based on the match data.
    """
    import json
    import os

    question = request.get('question', '')
    if not question:
        raise HTTPException(status_code=400, detail="Question is required")

    json_path = "C:/Users/info/football-analyzer/backend/uploads/balti-away-30min_analysis.json"
    if not os.path.exists(json_path):
        raise HTTPException(status_code=404, detail="Analysis file not found")

    try:
        with open(json_path, 'r') as f:
            data = json.load(f)

        frame_analyses = data.get('frame_analyses', []) or data.get('frames', [])
        if not frame_analyses:
            return {
                "status": "error",
                "answer": "No match data available to analyze. Please process a video first.",
                "confidence": "low"
            }

        # Run all analyses to get comprehensive match data
        pass_stats = pass_detector.analyze_from_frames(frame_analyses)
        formation_stats = formation_detector.analyze_from_frames(frame_analyses)
        tactical_events = tactical_detector.analyze_from_frames(frame_analyses)

        # Run AI coach analysis
        ai_coach.analyze_match(pass_stats, formation_stats, tactical_events, frame_analyses)

        # Build match data context for the question answering
        match_data = {
            'pass_stats': pass_stats,
            'formation_stats': formation_stats,
            'tactical_events': tactical_events,
            'frame_analyses': frame_analyses,
            'insights': [insight.to_dict() for insight in ai_coach.insights] if ai_coach.insights else [],
            'summary': ai_coach.match_summary.to_dict() if ai_coach.match_summary else None
        }

        # Get AI-generated answer
        response = ai_coach.answer_question(question, match_data)

        return {
            "status": "success",
            **response
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ============== Vision AI Coaching Endpoints (Gemini) ==============

@app.get("/api/ai-coach/vision/status")
async def get_vision_ai_status():
    """Check if Vision AI (Gemini) is configured and available."""
    has_key = bool(settings.GEMINI_API_KEY)
    return {
        "enabled": has_key,
        "model": settings.GEMINI_MODEL if has_key else None,
        "message": "Gemini Vision AI is ready" if has_key else "No GEMINI_API_KEY configured in .env"
    }


@app.post("/api/ai-coach/vision/analyze")
async def vision_analyze_match(request: dict = None):
    """
    Analyze match footage using Gemini Vision AI.

    This provides ChatGPT-like analysis by actually looking at the video frames
    and providing intelligent, contextual coaching insights.

    Request body (optional):
    {
        "video_id": "optional video ID to analyze",
        "question": "optional specific question to answer"
    }
    """
    import cv2

    if not settings.GEMINI_API_KEY:
        raise HTTPException(
            status_code=400,
            detail="Gemini API key not configured. Add GEMINI_API_KEY to your .env file"
        )

    # Initialize vision service if needed
    if not vision_ai_service._session:
        await vision_ai_service.initialize(settings.GEMINI_API_KEY)

    video_id = request.get('video_id') if request else None
    question = request.get('question') if request else None

    # Find frames to analyze
    frames_to_analyze = []

    # First try training frames (higher quality, saved during processing)
    if video_id:
        training_frames_dir = settings.DATA_DIR / "training" / "frames" / video_id
        if training_frames_dir.exists():
            frame_files = sorted(training_frames_dir.glob("*.jpg"))[:8]
            frames_to_analyze = [str(f) for f in frame_files]

    # If no training frames, try to extract from the most recent video
    if not frames_to_analyze:
        # Find most recent detection file to get video ID
        detection_files = list(settings.FRAMES_DIR.glob("*_detections.json"))
        if detection_files:
            most_recent = max(detection_files, key=lambda p: p.stat().st_mtime)
            vid_id = most_recent.stem.replace("_detections", "")

            # Find video file
            for ext in settings.SUPPORTED_VIDEO_FORMATS:
                video_path = settings.UPLOAD_DIR / f"{vid_id}{ext}"
                if video_path.exists():
                    # Extract sample frames
                    cap = cv2.VideoCapture(str(video_path))
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    fps = cap.get(cv2.CAP_PROP_FPS)

                    # Get 8 evenly spaced frames
                    frame_indices = [int(total_frames * i / 8) for i in range(8)]
                    frames = []

                    for idx in frame_indices:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                        ret, frame = cap.read()
                        if ret:
                            frames.append(frame)

                    cap.release()
                    frames_to_analyze = frames
                    break

    if not frames_to_analyze:
        raise HTTPException(
            status_code=404,
            detail="No video frames found to analyze. Upload and process a video first."
        )

    print(f"[VISION AI] Analyzing {len(frames_to_analyze)} frames...")

    # Run analysis
    if question:
        result = await vision_ai_service.answer_question(question, frames_to_analyze)
    else:
        result = await vision_ai_service.analyze_frames(frames_to_analyze)

    if "error" in result:
        raise HTTPException(status_code=500, detail=result.get("error"))

    return {
        "status": "success",
        "frames_analyzed": len(frames_to_analyze),
        "analysis": result
    }


@app.post("/api/ai-coach/vision/team-talk")
async def vision_team_talk(request: dict = None):
    """
    Get a half-time or post-match team talk based on visual analysis.

    This simulates what a coach would say to the team after watching
    the footage, providing specific, actionable feedback.
    """
    import cv2

    if not settings.GEMINI_API_KEY:
        raise HTTPException(
            status_code=400,
            detail="Gemini API key not configured"
        )

    if not vision_ai_service._session:
        await vision_ai_service.initialize(settings.GEMINI_API_KEY)

    team_context = request.get('team_context') if request else None

    # Find frames (same logic as analyze endpoint)
    frames_to_analyze = []

    detection_files = list(settings.FRAMES_DIR.glob("*_detections.json"))
    if detection_files:
        most_recent = max(detection_files, key=lambda p: p.stat().st_mtime)
        vid_id = most_recent.stem.replace("_detections", "")

        # Check training frames first
        training_dir = settings.DATA_DIR / "training" / "frames" / vid_id
        if training_dir.exists():
            frames_to_analyze = [str(f) for f in sorted(training_dir.glob("*.jpg"))[:8]]
        else:
            # Extract from video
            for ext in settings.SUPPORTED_VIDEO_FORMATS:
                video_path = settings.UPLOAD_DIR / f"{vid_id}{ext}"
                if video_path.exists():
                    cap = cv2.VideoCapture(str(video_path))
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                    frame_indices = [int(total_frames * i / 8) for i in range(8)]
                    frames = []

                    for idx in frame_indices:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                        ret, frame = cap.read()
                        if ret:
                            frames.append(frame)

                    cap.release()
                    frames_to_analyze = frames
                    break

    if not frames_to_analyze:
        raise HTTPException(status_code=404, detail="No frames found to analyze")

    result = await vision_ai_service.get_coaching_summary(frames_to_analyze, team_context)

    if "error" in result:
        raise HTTPException(status_code=500, detail=result.get("error"))

    return {
        "status": "success",
        "team_talk": result
    }


@app.post("/api/ai-coach/vision/ask")
async def vision_ask_question(request: dict):
    """
    Ask a specific question about the match footage.

    Example questions:
    - "How was our defensive shape?"
    - "Were we pressing effectively?"
    - "What formation did we play?"
    - "Did we create good chances?"

    The AI will analyze the frames and answer based on what it sees.
    """
    import cv2

    question = request.get('question')
    if not question:
        raise HTTPException(status_code=400, detail="Question is required")

    if not settings.GEMINI_API_KEY:
        raise HTTPException(status_code=400, detail="Gemini API key not configured")

    if not vision_ai_service._session:
        await vision_ai_service.initialize(settings.GEMINI_API_KEY)

    # Find frames
    frames_to_analyze = []

    detection_files = list(settings.FRAMES_DIR.glob("*_detections.json"))
    if detection_files:
        most_recent = max(detection_files, key=lambda p: p.stat().st_mtime)
        vid_id = most_recent.stem.replace("_detections", "")

        training_dir = settings.DATA_DIR / "training" / "frames" / vid_id
        if training_dir.exists():
            frames_to_analyze = [str(f) for f in sorted(training_dir.glob("*.jpg"))[:8]]
        else:
            for ext in settings.SUPPORTED_VIDEO_FORMATS:
                video_path = settings.UPLOAD_DIR / f"{vid_id}{ext}"
                if video_path.exists():
                    cap = cv2.VideoCapture(str(video_path))
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                    frame_indices = [int(total_frames * i / 8) for i in range(8)]
                    frames = []

                    for idx in frame_indices:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                        ret, frame = cap.read()
                        if ret:
                            frames.append(frame)

                    cap.release()
                    frames_to_analyze = frames
                    break

    if not frames_to_analyze:
        raise HTTPException(status_code=404, detail="No frames found to analyze")

    result = await vision_ai_service.answer_question(question, frames_to_analyze)

    if "error" in result:
        raise HTTPException(status_code=500, detail=result.get("error"))

    return {
        "status": "success",
        "question": question,
        "response": result
    }


# ============== PDF Report Export Endpoints ==============

@app.get("/api/report/generate")
async def generate_pdf_report():
    """
    Generate a comprehensive HTML report that can be saved as PDF.

    The report includes:
    - Executive summary with overall rating
    - Key statistics (possession, pass accuracy, etc.)
    - Passing analysis comparison
    - Formation analysis
    - Tactical events summary
    - AI coaching insights
    - Team talk messages

    Returns the HTML file path for download.
    """
    import json
    import os
    from datetime import datetime

    json_path = "C:/Users/info/football-analyzer/backend/uploads/balti-away-30min_analysis.json"
    if not os.path.exists(json_path):
        raise HTTPException(status_code=404, detail="Analysis file not found")

    try:
        with open(json_path, 'r') as f:
            data = json.load(f)

        frame_analyses = data.get('frame_analyses', []) or data.get('frames', [])
        if not frame_analyses:
            return {"error": "No frame data available"}

        # Run all analyses
        pass_stats = pass_detector.analyze_from_frames(frame_analyses)
        formation_stats = formation_detector.analyze_from_frames(frame_analyses)
        tactical_events = tactical_detector.analyze_from_frames(frame_analyses)
        coaching_analysis = ai_coach.analyze_match(
            pass_stats, formation_stats, tactical_events, frame_analyses
        )

        # Match info
        match_info = {
            'video_path': data.get('video_path', 'Match Analysis'),
            'duration_minutes': data.get('duration_seconds', 0) / 60,
            'frames_analyzed': data.get('analyzed_frames', 0)
        }

        # Generate HTML report
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = f"C:/Users/info/football-analyzer/backend/uploads/match_report_{timestamp}.html"

        pdf_report_generator.generate_html_report(
            match_info=match_info,
            pass_stats=pass_stats,
            formation_stats=formation_stats,
            tactical_events=tactical_events,
            coaching_insights=coaching_analysis,
            output_path=output_path
        )

        return {
            "status": "success",
            "report_path": output_path,
            "download_url": f"/api/report/download/{timestamp}",
            "message": "Report generated successfully. Open the HTML file in your browser and use Print > Save as PDF."
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/report/download/{timestamp}")
async def download_report(timestamp: str):
    """Download a generated report."""
    import os

    report_path = f"C:/Users/info/football-analyzer/backend/uploads/match_report_{timestamp}.html"

    if not os.path.exists(report_path):
        raise HTTPException(status_code=404, detail="Report not found")

    return FileResponse(
        report_path,
        media_type="text/html",
        filename=f"match_report_{timestamp}.html"
    )


@app.get("/api/report/json")
async def get_report_json():
    """
    Get the report data as JSON for custom processing or display.
    """
    import json
    import os

    json_path = "C:/Users/info/football-analyzer/backend/uploads/balti-away-30min_analysis.json"
    if not os.path.exists(json_path):
        raise HTTPException(status_code=404, detail="Analysis file not found")

    try:
        with open(json_path, 'r') as f:
            data = json.load(f)

        frame_analyses = data.get('frame_analyses', []) or data.get('frames', [])
        if not frame_analyses:
            return {"error": "No frame data available"}

        # Run all analyses
        pass_stats = pass_detector.analyze_from_frames(frame_analyses)
        formation_stats = formation_detector.analyze_from_frames(frame_analyses)
        tactical_events = tactical_detector.analyze_from_frames(frame_analyses)
        coaching_analysis = ai_coach.analyze_match(
            pass_stats, formation_stats, tactical_events, frame_analyses
        )

        # Match info
        match_info = {
            'video_path': data.get('video_path', 'Match Analysis'),
            'duration_minutes': data.get('duration_seconds', 0) / 60,
            'frames_analyzed': data.get('analyzed_frames', 0)
        }

        # Generate structured report data
        report_data = pdf_report_generator.generate_report_data(
            match_info=match_info,
            pass_stats=pass_stats,
            formation_stats=formation_stats,
            tactical_events=tactical_events,
            coaching_insights=coaching_analysis
        )

        return {
            "status": "success",
            "report": report_data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============== Player Clip Analysis Endpoints ==============

@app.post("/api/player/analyze-clip")
async def analyze_player_clip(
    clip_path: str,
    player_name: str = "Player"
):
    """
    Analyze a single VEO-exported player highlight clip.

    Args:
        clip_path: Path to the video clip file
        player_name: Name of the player in the clip
    """
    import os

    if not os.path.exists(clip_path):
        raise HTTPException(status_code=404, detail=f"Clip not found: {clip_path}")

    try:
        analysis = player_analyzer.analyze_clip(clip_path, player_name)
        return {
            "status": "success",
            "player_name": player_name,
            "clip_path": analysis.clip_path,
            "duration_seconds": round(analysis.duration_seconds, 1),
            "player_visible_frames": analysis.player_visible_frames,
            "total_frames": analysis.total_frames,
            "ball_touches": analysis.ball_touches,
            "distance_covered_pixels": round(analysis.distance_covered_pixels, 0),
            "events": [
                {
                    "type": e.event_type,
                    "timestamp": round(e.timestamp, 2),
                    "frame": e.frame_number,
                    "position": e.position,
                    "success": e.success,
                    "details": e.details
                }
                for e in analysis.events
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/player/analyze-folder")
async def analyze_player_folder(
    folder_path: str,
    player_name: str = "Player",
    background_tasks: BackgroundTasks = None
):
    """
    Analyze all video clips in a folder for a specific player.

    This is useful when you have a folder of VEO-exported clips for one player.

    Args:
        folder_path: Path to folder containing video clips
        player_name: Name of the player
    """
    import os

    if not os.path.exists(folder_path):
        raise HTTPException(status_code=404, detail=f"Folder not found: {folder_path}")

    if not os.path.isdir(folder_path):
        raise HTTPException(status_code=400, detail=f"Not a directory: {folder_path}")

    # Run analysis in background for large folders
    async def run_folder_analysis():
        player_analyzer.analyze_clips_folder(folder_path, player_name)

    background_tasks.add_task(run_folder_analysis)

    return {
        "status": "started",
        "folder_path": folder_path,
        "player_name": player_name,
        "message": f"Analysis started for {player_name}. Check /api/player/{player_name}/stats for progress."
    }


@app.get("/api/player/{player_name}/stats")
async def get_player_stats(player_name: str):
    """
    Get aggregated statistics for a player.

    Returns stats compiled from all analyzed clips for this player.
    """
    if player_name not in player_analyzer.player_stats:
        raise HTTPException(
            status_code=404,
            detail=f"No stats found for {player_name}. Analyze some clips first."
        )

    stats = player_analyzer.player_stats[player_name]

    return {
        "player_name": stats.player_name,
        "total_clips": stats.total_clips,
        "total_play_time_seconds": round(stats.total_play_time, 1),
        "attacking": {
            "ball_touches": stats.ball_touches,
            "passes_attempted": stats.passes_attempted,
            "passes_completed": stats.passes_completed,
            "pass_accuracy": round(stats.pass_accuracy, 1),
            "shots": stats.shots,
            "shots_on_target": stats.shots_on_target,
            "shot_accuracy": round(stats.shot_accuracy, 1)
        },
        "defensive": {
            "tackles_attempted": stats.tackles_attempted,
            "tackles_won": stats.tackles_won,
            "tackle_success_rate": round(stats.tackle_success_rate, 1),
            "headers": stats.headers,
            "interceptions": stats.interceptions
        },
        "physical": {
            "distance_covered_pixels": round(stats.total_distance_pixels, 0),
            "distance_covered_meters_estimate": round(stats.total_distance_pixels * 0.05, 0),
            "sprints": stats.sprints
        }
    }


@app.get("/api/player/{player_name}/report")
async def get_player_report(player_name: str):
    """
    Generate a full report for a player.

    Returns comprehensive analysis including attacking, defensive, and physical metrics.
    """
    report = player_analyzer.get_player_report(player_name)

    if "error" in report:
        raise HTTPException(status_code=404, detail=report["error"])

    return report


@app.get("/api/player/list")
async def list_analyzed_players():
    """
    List all players that have been analyzed.
    """
    players = []
    for name, stats in player_analyzer.player_stats.items():
        players.append({
            "name": name,
            "clips_analyzed": stats.total_clips,
            "play_time_seconds": round(stats.total_play_time, 1),
            "ball_touches": stats.ball_touches
        })

    return {
        "players": players,
        "total_players": len(players),
        "total_clips_analyzed": len(player_analyzer.clip_analyses)
    }


@app.post("/api/player/export")
async def export_player_stats(output_path: str = None):
    """
    Export all player stats to JSON file.

    Args:
        output_path: Path for output file. Defaults to uploads/player_stats.json
    """
    if output_path is None:
        output_path = str(settings.UPLOAD_DIR / "player_stats.json")

    data = player_analyzer.export_stats(output_path)

    return {
        "status": "exported",
        "output_path": output_path,
        "players_exported": len(data.get("players", {}))
    }


@app.delete("/api/player/{player_name}/reset")
async def reset_player_stats(player_name: str):
    """
    Reset stats for a specific player.
    """
    if player_name in player_analyzer.player_stats:
        del player_analyzer.player_stats[player_name]
        # Remove related clip analyses
        player_analyzer.clip_analyses = [
            ca for ca in player_analyzer.clip_analyses
            if ca.clip_path not in [c.clip_path for c in player_analyzer.clip_analyses]
        ]
        return {"message": f"Stats reset for {player_name}"}
    else:
        raise HTTPException(status_code=404, detail=f"Player not found: {player_name}")


# ============== Jersey Number Recognition & Player Highlights ==============

from services.jersey_ocr import jersey_ocr_service
from services.player_highlights import player_highlights_service

# Professional Analytics Services (VEO-style)
from services.match_statistics import match_statistics_service, EventType as StatsEventType
from services.pitch_visualization import pitch_visualization_service
from services.event_detector import event_detector, DetectedEventType
from services.coach_assist import coach_assist_service


@app.post("/api/jersey-ocr/initialize")
async def initialize_jersey_ocr():
    """
    Initialize the jersey number OCR service.

    Requires easyocr or paddleocr to be installed.
    """
    success = await jersey_ocr_service.initialize()
    if success:
        return {"status": "initialized", "engine": jersey_ocr_service.ocr_type}
    else:
        raise HTTPException(
            status_code=500,
            detail="Failed to initialize OCR. Install easyocr or paddleocr."
        )


@app.post("/api/jersey-ocr/roster")
async def set_team_roster(roster: Dict[int, str]):
    """
    Set the team roster for jersey number to player name mapping.

    Args:
        roster: Dict mapping jersey number to player name
               e.g., {1: "John Smith", 7: "David Beckham", 10: "Lionel Messi"}
    """
    jersey_ocr_service.set_roster(roster)
    return {"status": "roster_set", "players": len(roster)}


@app.post("/api/jersey-ocr/target-team")
async def set_ocr_target_team(team: str):
    """
    Set which team to run OCR on (home or away).

    Running OCR only on your own team saves processing time.
    """
    team_side = TeamSide.HOME if team.lower() == "home" else TeamSide.AWAY
    jersey_ocr_service.set_target_team(team_side)
    return {"status": "target_set", "team": team_side.value}


@app.get("/api/jersey-ocr/identified-players")
async def get_identified_players():
    """Get all players that have been identified by jersey number."""
    players = jersey_ocr_service.get_all_identified_players()
    return {
        "players": [
            {
                "track_id": p.track_id,
                "jersey_number": p.jersey_number,
                "player_name": p.player_name,
                "team": p.team.value,
                "confidence": p.confidence,
                "observations": p.observation_count
            }
            for p in players
        ],
        "stats": jersey_ocr_service.get_stats()
    }


@app.get("/api/jersey-ocr/player/{jersey_number}")
async def get_player_by_jersey(jersey_number: int):
    """Get identified player info by jersey number."""
    player = jersey_ocr_service.get_player_by_number(jersey_number)
    if player:
        return {
            "track_id": player.track_id,
            "jersey_number": player.jersey_number,
            "player_name": player.player_name,
            "team": player.team.value,
            "confidence": player.confidence
        }
    else:
        raise HTTPException(status_code=404, detail=f"Player #{jersey_number} not identified yet")


@app.post("/api/highlights/register-player")
async def register_player_for_highlights(
    jersey_number: int,
    player_name: Optional[str] = None,
    team: str = "home"
):
    """
    Register a player to track for highlight generation.

    Args:
        jersey_number: Player's jersey number
        player_name: Optional player name
        team: 'home' or 'away'
    """
    team_side = TeamSide.HOME if team.lower() == "home" else TeamSide.AWAY
    player_highlights_service.register_player(jersey_number, player_name, team_side)
    return {"status": "registered", "jersey_number": jersey_number}


@app.post("/api/highlights/register-roster")
async def register_roster_for_highlights(
    roster: Dict[int, str],
    team: str = "home"
):
    """
    Register entire roster for highlight generation.

    Args:
        roster: Dict mapping jersey number to player name
        team: 'home' or 'away'
    """
    team_side = TeamSide.HOME if team.lower() == "home" else TeamSide.AWAY

    for jersey_number, player_name in roster.items():
        player_highlights_service.register_player(
            int(jersey_number),
            player_name,
            team_side
        )

    # Also set the OCR roster
    jersey_ocr_service.set_roster({int(k): v for k, v in roster.items()})

    return {"status": "roster_registered", "players": len(roster)}


@app.put("/api/highlights/player/{jersey_number}/name")
async def rename_player(
    jersey_number: int,
    new_name: str
):
    """
    Rename a player (change from "Player 9" to actual name).

    Args:
        jersey_number: Player's jersey number
        new_name: New name for the player
    """
    if jersey_number not in player_highlights_service.players:
        raise HTTPException(status_code=404, detail=f"Player #{jersey_number} not found")

    player_highlights_service.players[jersey_number].player_name = new_name
    return {
        "status": "renamed",
        "jersey_number": jersey_number,
        "new_name": new_name
    }


@app.post("/api/highlights/set-video")
async def set_highlights_video(
    video_id: str
):
    """
    Set the source video for highlight generation.

    This should be called after video upload.
    """
    services = get_services()
    video_service = services["video"]

    video_info = video_service.get_video_info(video_id)
    if not video_info:
        raise HTTPException(status_code=404, detail="Video not found")

    player_highlights_service.set_video_info(
        video_path=video_info["path"],
        fps=video_info["fps"],
        width=video_info["width"],
        height=video_info["height"],
        total_frames=video_info["total_frames"]
    )

    return {"status": "video_set", "video_id": video_id}


@app.get("/api/highlights/player/{jersey_number}/summary")
async def get_player_highlight_summary(jersey_number: int):
    """Get summary of a player's highlight data."""
    summary = player_highlights_service.get_player_summary(jersey_number)
    if summary:
        return summary
    else:
        raise HTTPException(status_code=404, detail=f"No data for player #{jersey_number}")


@app.get("/api/highlights/summaries")
async def get_all_highlight_summaries():
    """Get highlight summaries for all tracked players."""
    return {
        "players": player_highlights_service.get_all_summaries()
    }


@app.post("/api/highlights/generate/{jersey_number}")
async def generate_player_highlights(
    jersey_number: int,
    background_tasks: BackgroundTasks,
    max_duration: float = 120.0,
    min_importance: float = 0.5
):
    """
    Generate highlight video for a specific player.

    Args:
        jersey_number: Player's jersey number
        max_duration: Maximum video length in seconds
        min_importance: Minimum event importance to include (0-10)

    Returns:
        Task ID for background processing
    """
    task_id = str(uuid.uuid4())

    async def generate_task():
        output_path = await player_highlights_service.generate_highlight_video(
            jersey_number=jersey_number,
            max_duration_seconds=max_duration,
            min_importance=min_importance
        )
        return output_path

    background_tasks.add_task(generate_task)

    return {
        "status": "generating",
        "task_id": task_id,
        "jersey_number": jersey_number
    }


@app.post("/api/highlights/generate-all")
async def generate_all_highlights(
    background_tasks: BackgroundTasks,
    max_duration: float = 120.0,
    min_importance: float = 0.5
):
    """
    Generate highlight videos for all tracked players.

    Args:
        max_duration: Maximum video length per player in seconds
        min_importance: Minimum event importance to include
    """
    task_id = str(uuid.uuid4())

    async def generate_all_task():
        results = await player_highlights_service.generate_all_highlights(
            max_duration_seconds=max_duration,
            min_importance=min_importance
        )
        return results

    background_tasks.add_task(generate_all_task)

    return {
        "status": "generating",
        "task_id": task_id,
        "players": list(player_highlights_service.players.keys())
    }


@app.post("/api/highlights/export-data")
async def export_highlight_data(output_path: Optional[str] = None):
    """Export all highlight tracking data to JSON."""
    if output_path is None:
        output_path = str(settings.UPLOAD_DIR / "player_highlights_data.json")

    result_path = player_highlights_service.export_data(output_path)

    return {
        "status": "exported",
        "output_path": result_path,
        "players_exported": len(player_highlights_service.players)
    }


@app.get("/api/tracking/player-count-stats")
async def get_player_count_stats():
    """
    Get statistics about player detection counts.

    Useful for monitoring if all 22 players are being tracked consistently.
    """
    services = get_services()
    tracking_service = services["tracking"]

    return tracking_service.get_player_count_stats()


@app.post("/api/clips/extract/{jersey_number}")
async def extract_player_clips(
    jersey_number: int,
    min_importance: float = 0.0,
    event_types: Optional[str] = None
):
    """
    Extract individual video clips for a player's moments.

    Args:
        jersey_number: Player's jersey number
        min_importance: Only extract clips with importance >= this value
        event_types: Comma-separated list of event types to extract (e.g., "shot,pass,tackle")

    Returns:
        List of extracted clip metadata
    """
    event_type_list = event_types.split(",") if event_types else None
    clips = await player_highlights_service.extract_player_clips(
        jersey_number=jersey_number,
        min_importance=min_importance,
        event_types=event_type_list
    )
    return {
        "status": "extracted",
        "jersey_number": jersey_number,
        "clips_count": len(clips),
        "clips": [
            {
                "clip_id": c.clip_id,
                "event_type": c.event_type,
                "timestamp_ms": c.timestamp_start_ms,
                "duration_seconds": c.duration_seconds,
                "importance": c.importance,
                "clip_path": c.clip_path
            }
            for c in clips
        ]
    }


@app.post("/api/clips/extract-all")
async def extract_all_clips(
    min_importance: float = 0.0,
    event_types: Optional[str] = None
):
    """
    Extract individual video clips for all tracked players.

    Args:
        min_importance: Only extract clips with importance >= this value
        event_types: Comma-separated list of event types (e.g., "shot,pass,tackle")

    Returns:
        Summary of extracted clips
    """
    event_type_list = event_types.split(",") if event_types else None
    clips = await player_highlights_service.extract_all_clips(
        min_importance=min_importance,
        event_types=event_type_list
    )
    return {
        "status": "extracted",
        "total_clips": len(clips),
        "by_player": {
            jersey: len([c for c in clips if c.jersey_number == jersey])
            for jersey in set(c.jersey_number for c in clips)
        },
        "by_event_type": {
            event: len([c for c in clips if c.event_type == event])
            for event in set(c.event_type for c in clips)
        }
    }


@app.get("/api/clips/player/{jersey_number}")
async def get_player_clips(
    jersey_number: int,
    event_type: Optional[str] = None
):
    """
    Get all extracted clips for a player.

    Args:
        jersey_number: Player's jersey number
        event_type: Filter by event type (e.g., "shot", "pass")

    Returns:
        List of clip metadata
    """
    clips = player_highlights_service.get_player_clips(
        jersey_number=jersey_number,
        event_type=event_type
    )
    return {
        "jersey_number": jersey_number,
        "clips_count": len(clips),
        "clips": [
            {
                "clip_id": c.clip_id,
                "event_type": c.event_type,
                "timestamp_ms": c.timestamp_start_ms,
                "duration_seconds": c.duration_seconds,
                "importance": c.importance,
                "clip_path": c.clip_path,
                "player_name": c.player_name
            }
            for c in clips
        ]
    }


@app.get("/api/clips/all")
async def get_all_clips(event_type: Optional[str] = None):
    """
    Get all extracted clips across all players.

    Args:
        event_type: Filter by event type (e.g., "shot", "pass")

    Returns:
        List of all clip metadata
    """
    clips = player_highlights_service.get_all_clips(event_type=event_type)
    return {
        "total_clips": len(clips),
        "clips": [
            {
                "clip_id": c.clip_id,
                "jersey_number": c.jersey_number,
                "player_name": c.player_name,
                "team": c.team,
                "event_type": c.event_type,
                "timestamp_ms": c.timestamp_start_ms,
                "duration_seconds": c.duration_seconds,
                "importance": c.importance,
                "clip_path": c.clip_path
            }
            for c in clips
        ]
    }


@app.get("/api/clips/{clip_id}")
async def get_clip(clip_id: str):
    """
    Get metadata for a specific clip by ID.

    Args:
        clip_id: Unique clip identifier

    Returns:
        Clip metadata or 404 if not found
    """
    if clip_id not in player_highlights_service.clips_registry:
        raise HTTPException(status_code=404, detail=f"Clip {clip_id} not found")

    c = player_highlights_service.clips_registry[clip_id]
    return {
        "clip_id": c.clip_id,
        "jersey_number": c.jersey_number,
        "player_name": c.player_name,
        "team": c.team,
        "event_type": c.event_type,
        "frame_start": c.frame_start,
        "frame_end": c.frame_end,
        "timestamp_start_ms": c.timestamp_start_ms,
        "timestamp_end_ms": c.timestamp_end_ms,
        "duration_seconds": c.duration_seconds,
        "importance": c.importance,
        "clip_path": c.clip_path,
        "extracted_at": c.extracted_at
    }


@app.delete("/api/highlights/reset")
async def reset_highlights():
    """Reset all highlight tracking data."""
    player_highlights_service.reset()
    jersey_ocr_service.reset()
    return {"status": "reset"}


@app.post("/api/highlights/build-from-detections")
async def build_highlights_from_detections(
    video_id: Optional[str] = None,
    max_frames: int = 1000,
    background_tasks: BackgroundTasks = None
):
    """
    Build player highlights data from existing detection data.

    NOW USES PLAYER IDENTITY SYSTEM - Returns the 22 persistent player identities
    instead of creating thousands of duplicate track_id based players.

    Args:
        video_id: Optional video ID. Uses most recent if not specified.
        max_frames: Maximum frames to process (samples evenly if more)

    Returns:
        Summary of players tracked from player identity system
    """
    print(f"[HIGHLIGHTS BUILD] Starting build for video_id={video_id} (using player identity system)")

    # Find detection file
    detection_file = None
    video_path = None

    if video_id:
        detection_file = settings.FRAMES_DIR / f"{video_id}_detections.json"
        for ext in settings.SUPPORTED_VIDEO_FORMATS:
            path = settings.UPLOAD_DIR / f"{video_id}{ext}"
            if path.exists():
                video_path = path
                break
    else:
        # Find most recent detection file
        detection_files = list(settings.FRAMES_DIR.glob("*_detections.json"))
        if detection_files:
            detection_file = max(detection_files, key=lambda p: p.stat().st_mtime)
            video_id = detection_file.stem.replace("_detections", "")
            for ext in settings.SUPPORTED_VIDEO_FORMATS:
                path = settings.UPLOAD_DIR / f"{video_id}{ext}"
                if path.exists():
                    video_path = path
                    break

    if not detection_file or not detection_file.exists():
        print(f"[HIGHLIGHTS BUILD] Detection file not found: {detection_file}")
        raise HTTPException(status_code=404, detail="No detection data found")

    if not video_path:
        print(f"[HIGHLIGHTS BUILD] Video file not found for {video_id}")
        raise HTTPException(status_code=404, detail="Video file not found")

    print(f"[HIGHLIGHTS BUILD] Loading detection file: {detection_file}")

    # Load detection data - use standard file reading for large files
    try:
        with open(detection_file, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"[HIGHLIGHTS BUILD] Error loading detection file: {e}")
        raise HTTPException(status_code=500, detail=f"Error loading detection file: {e}")

    detections = data.get("detections", [])
    if not detections:
        raise HTTPException(status_code=400, detail="No detections in file")

    print(f"[HIGHLIGHTS BUILD] Loaded {len(detections)} frames")

    # Get video metadata
    services = get_services()
    video_metadata = await services["video"].get_video_metadata(video_path)

    # DO NOT reset highlights service - we want to use the player identity system instead
    # The player identity system already has the persistent player IDs
    # Resetting would create hundreds of duplicate track_id-based players

    # Set up video info only if not already set
    if player_highlights_service.video_path is None:
        player_highlights_service.set_video_info(
            video_path=str(video_path),
            fps=video_metadata.get("fps", 30),
            width=video_metadata.get("width", 1920),
            height=video_metadata.get("height", 1080),
            total_frames=video_metadata.get("total_frames", len(detections))
        )

    # SKIP THE OLD PROCESSING - Use player identity system instead
    # The old system creates a player for every track_id which results in hundreds of duplicates
    # The player identity system already has the 22 persistent players

    print(f"[HIGHLIGHTS BUILD] Using player identity system - skipping track_id processing")

    # Get player identities from the player identity database
    all_identities = player_identity_db.get_all_identities(video_id)

    # Filter to significant players (appeared in at least 100 frames)
    min_frames = 100
    significant_players = [
        identity for identity in all_identities
        if (identity.last_seen_frame - identity.first_seen_frame + 1) >= min_frames
    ]

    players_tracked = len(significant_players)
    moments_found = 0  # Will be populated from event detection later

    print(f"[HIGHLIGHTS BUILD] Complete: {players_tracked} players from identity system, {moments_found} moments")

    return {
        "status": "built",
        "video_id": video_id,
        "frames_processed": len(detections),
        "players_tracked": players_tracked,
        "total_moments": moments_found,
        "players": [
            {
                "player_id": p.player_id,
                "jersey_number": p.jersey_number,
                "positional_role": p.positional_role,
                "player_name": f"Player #{p.jersey_number}" if p.jersey_number else f"{p.positional_role or 'Unknown'}",
                "team": p.team,
                "touches": p.ball_touches,  # Use ball_touches from player identity
                "moments": 0,  # Will be populated from event detection
                "frames_appeared": p.last_seen_frame - p.first_seen_frame + 1
            }
            for p in significant_players
        ]
    }



@app.get("/api/highlights/match-players")
async def get_match_players(video_id: Optional[str] = None, min_frames: int = 100):
    """
    Get all players tracked from the match with their clip availability.

    Returns players with their stats and whether clips can be generated.
    Uses the new player identity system to properly track players across track_id changes.

    Args:
        video_id: Optional video ID to filter players
        min_frames: Minimum number of frames a player must appear in (default: 100)
    """
    players = []

    # Get the most recent video_id if not provided
    if not video_id:
        detection_files = list(settings.FRAMES_DIR.glob("*_detections.json"))
        if detection_files:
            json_path = max(detection_files, key=lambda p: p.stat().st_mtime)
            video_id = str(json_path.stem).replace('_detections', '')

    if video_id:
        # Get all player identities from the player identity database
        all_identities = player_identity_db.get_all_identities(video_id)

        # Filter to only players who appeared in enough frames
        for identity in all_identities:
            frames_appeared = identity.last_seen_frame - identity.first_seen_frame + 1

            # Skip players with too few appearances (likely noise/false detections)
            if frames_appeared < min_frames:
                continue

            # Get player moments/events from highlights service if available
            moments_count = 0
            touches = identity.ball_touches  # Use ball_touches from player identity
            if identity.jersey_number and identity.jersey_number in player_highlights_service.players:
                player_data = player_highlights_service.players[identity.jersey_number]
                moments_count = len(player_data.moments)

            # Count existing clips if any
            existing_clips = []
            if identity.jersey_number:
                existing_clips = player_highlights_service.get_player_clips(identity.jersey_number)

            players.append({
                "player_id": identity.player_id,
                "jersey_number": identity.jersey_number,
                "player_name": f"Player #{identity.jersey_number}" if identity.jersey_number else f"{identity.positional_role or 'Unknown'}",
                "team": identity.team,
                "positional_role": identity.positional_role,
                "track_ids": identity.track_id_history,
                "stats": {
                    "touches": touches,
                    "passes_attempted": 0,
                    "passes_completed": 0,
                    "shots": 0,
                    "goals": 0,
                    "tackles": 0,
                    "interceptions": 0,
                },
                "moments_count": moments_count,
                "clips_available": len(existing_clips),
                "frames_appeared": frames_appeared,
                "first_seen": identity.first_seen_frame,
                "last_seen": identity.last_seen_frame,
                "can_generate_clips": frames_appeared > min_frames
            })

        # Sort by frames appeared (most active first)
        players.sort(key=lambda p: p["frames_appeared"], reverse=True)
    else:
        # Fallback to old system if no video_id available
        for jersey_number, player_data in player_highlights_service.players.items():
            existing_clips = player_highlights_service.get_player_clips(jersey_number)

            players.append({
                "jersey_number": jersey_number,
                "player_name": player_data.player_name,
                "team": player_data.team.value if hasattr(player_data.team, 'value') else str(player_data.team),
                "track_ids": list(player_data.track_ids),
                "stats": {
                    "touches": player_data.touches,
                    "passes_attempted": player_data.passes_attempted,
                    "passes_completed": player_data.passes_completed,
                    "shots": player_data.shots,
                    "goals": player_data.goals,
                    "tackles": player_data.tackles,
                    "interceptions": player_data.interceptions,
                },
                "moments_count": len(player_data.moments),
                "clips_available": len(existing_clips),
                "can_generate_clips": len(player_data.moments) > 0 and player_highlights_service.video_path is not None
            })

        players.sort(key=lambda p: p["stats"]["touches"], reverse=True)

    return {
        "total_players": len(players),
        "video_configured": player_highlights_service.video_path is not None,
        "video_id": video_id,
        "players": players
    }


# ============== Professional Analytics (VEO-style) ==============

@app.get("/api/pro-analytics/match-stats")
async def get_match_statistics():
    """
    Get comprehensive match statistics.

    Returns possession, passes, shots, xG, and all team/player stats
    similar to VEO Analytics.
    """
    return match_statistics_service.get_match_summary()


@app.get("/api/pro-analytics/shot-map")
async def get_shot_map():
    """
    Get shot map data for visualization.

    Returns all shots with position, xG, and result for both teams.
    """
    return match_statistics_service.get_shot_map_data()


@app.get("/api/pro-analytics/events")
async def get_match_events(event_type: Optional[str] = None):
    """
    Get match events timeline.

    Args:
        event_type: Filter by event type (goal, shot, corner, etc.)
    """
    if event_type:
        try:
            etype = StatsEventType(event_type)
            events = match_statistics_service.get_events_by_type(etype)
            return {
                "event_type": event_type,
                "count": len(events),
                "events": [
                    {
                        "id": e.event_id,
                        "type": e.event_type.value,
                        "timestamp_ms": e.timestamp_ms,
                        "team": e.team,
                        "player": e.player_jersey,
                        "position": {"x": e.position_x, "y": e.position_y} if e.position_x else None
                    }
                    for e in events
                ]
            }
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid event type: {event_type}")

    return {
        "events": match_statistics_service.get_events_timeline(),
        "counts": match_statistics_service.get_event_counts() if hasattr(match_statistics_service, 'get_event_counts') else {}
    }


@app.get("/api/pro-analytics/player-stats/{team}/{jersey_number}")
async def get_player_statistics(team: str, jersey_number: int):
    """
    Get individual player statistics.

    Args:
        team: 'home' or 'away'
        jersey_number: Player's jersey number
    """
    stats = match_statistics_service.get_player_stats(team, jersey_number)
    if not stats:
        raise HTTPException(status_code=404, detail=f"Player #{jersey_number} not found in {team} team")
    return stats


@app.get("/api/pro-analytics/all-player-stats")
async def get_all_player_statistics():
    """Get statistics for all players grouped by team."""
    return match_statistics_service.get_all_player_stats()


# ============== Pitch Visualization (Heatmaps, 2D Radar) ==============

@app.get("/api/visualization/heatmap/team/{team}")
async def get_team_heatmap(
    team: str,
    start_frame: Optional[int] = None,
    end_frame: Optional[int] = None
):
    """
    Get team positioning heatmap.

    Args:
        team: 'home' or 'away'
        start_frame: Optional start frame for time filtering
        end_frame: Optional end frame for time filtering
    """
    if team not in ["home", "away"]:
        raise HTTPException(status_code=400, detail="Team must be 'home' or 'away'")

    return pitch_visualization_service.generate_team_heatmap(team, start_frame, end_frame)


@app.get("/api/visualization/heatmap/player/{team}/{jersey_number}")
async def get_player_heatmap(
    team: str,
    jersey_number: int,
    start_frame: Optional[int] = None,
    end_frame: Optional[int] = None
):
    """
    Get individual player positioning heatmap.

    Args:
        team: 'home' or 'away'
        jersey_number: Player's jersey number
    """
    return pitch_visualization_service.generate_player_heatmap(
        team, jersey_number, start_frame, end_frame
    )


@app.get("/api/visualization/2d-radar")
async def get_2d_radar(frame_number: Optional[int] = None):
    """
    Get 2D radar state (player and ball positions).

    Args:
        frame_number: Specific frame to get (None for current state)
    """
    return pitch_visualization_service.get_2d_radar_state(frame_number)


@app.get("/api/visualization/player-trail/{team}/{jersey_number}")
async def get_player_trail(
    team: str,
    jersey_number: int,
    start_frame: int,
    end_frame: int
):
    """
    Get player movement trail between frames.

    Used for showing player runs on 2D radar.
    """
    return pitch_visualization_service.get_player_trail(
        team, jersey_number, start_frame, end_frame
    )


@app.get("/api/visualization/team-shape/{team}")
async def get_team_shape(team: str, frame_number: int):
    """
    Get team shape at a specific frame.

    Returns player positions, centroid, width, depth, and compactness.
    """
    return pitch_visualization_service.get_team_shape(team, frame_number)


@app.get("/api/visualization/average-positions/{team}")
async def get_average_positions(team: str):
    """
    Get average positions for all players on a team.

    Used for formation display.
    """
    return pitch_visualization_service.get_average_positions(team)


@app.get("/api/visualization/possession-zones/{team}")
async def get_possession_zones(
    team: str,
    start_frame: Optional[int] = None,
    end_frame: Optional[int] = None
):
    """
    Get possession zone breakdown by thirds.

    Returns % of possession in defensive, middle, and attacking thirds.
    """
    return pitch_visualization_service.get_possession_zones(team, start_frame, end_frame)


@app.get("/api/visualization/player-distance")
async def get_player_distance(
    team: str,
    jersey1: int,
    jersey2: int,
    frame_number: int
):
    """
    Get distance between two players at a specific frame.

    Used for the 'connect players' feature (like VEO).
    Returns distance in yards.
    """
    distance = pitch_visualization_service.get_distance_between_players(
        team, jersey1, jersey2, frame_number
    )
    if distance is None:
        raise HTTPException(status_code=404, detail="Could not calculate distance - players not found")

    return {
        "team": team,
        "player1": jersey1,
        "player2": jersey2,
        "frame": frame_number,
        "distance_yards": distance
    }


@app.get("/api/visualization/export")
async def export_visualization_data():
    """Export all visualization data for frontend rendering."""
    return pitch_visualization_service.export_visualization_data()


# ============== Auto Event Detection ==============

@app.get("/api/events/detected")
async def get_detected_events(event_type: Optional[str] = None):
    """
    Get automatically detected match events.

    Args:
        event_type: Filter by type (goal, shot, corner, etc.)
    """
    if event_type:
        try:
            etype = DetectedEventType(event_type)
            events = event_detector.get_events_by_type(etype)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid event type: {event_type}")
    else:
        events = event_detector.events

    return {
        "total": len(events),
        "events": [
            {
                "type": e.event_type.value,
                "timestamp_ms": e.timestamp_ms,
                "time_str": f"{e.timestamp_ms // 60000}:{(e.timestamp_ms // 1000) % 60:02d}",
                "frame": e.frame_number,
                "team": e.team,
                "player": e.player_jersey,
                "position": {"x": e.position_x, "y": e.position_y} if e.position_x else None,
                "confidence": e.confidence,
                "description": e.description
            }
            for e in sorted(events, key=lambda x: x.timestamp_ms)
        ]
    }


@app.get("/api/events/timeline")
async def get_events_timeline():
    """Get all detected events as a timeline for display."""
    return {
        "timeline": event_detector.get_events_timeline(),
        "score": event_detector.get_score(),
        "counts": event_detector.get_event_counts()
    }


@app.get("/api/events/score")
async def get_match_score():
    """Get current match score from detected goals."""
    return event_detector.get_score()


# ============== Coach Assist AI ==============

@app.post("/api/coach-assist/load-data")
async def load_coach_assist_data():
    """
    Load match data into Coach Assist for analysis.

    Automatically pulls data from other services.
    """
    # Get data from services
    match_stats = match_statistics_service.get_match_summary()
    events = event_detector.get_events_timeline()
    player_stats = match_statistics_service.get_all_player_stats()

    coach_assist_service.load_match_data(
        match_stats=match_stats,
        events=events,
        player_stats=player_stats
    )

    return {
        "status": "loaded",
        "stats_available": bool(match_stats),
        "events_count": len(events),
        "players_tracked": {
            "home": len(player_stats.get("home", [])),
            "away": len(player_stats.get("away", []))
        }
    }


@app.get("/api/coach-assist/summary")
async def get_match_summary():
    """
    Get instant match summary.

    Quick overview with key statistics.
    """
    return {
        "summary": coach_assist_service.generate_match_summary()
    }


@app.get("/api/coach-assist/talking-points/{team}")
async def get_talking_points(team: str):
    """
    Get AI-generated talking points for team discussion.

    Args:
        team: 'home' or 'away'
    """
    if team not in ["home", "away"]:
        raise HTTPException(status_code=400, detail="Team must be 'home' or 'away'")

    points = coach_assist_service.generate_talking_points(team)

    return {
        "team": team,
        "talking_points": [
            {
                "category": p.category.value,
                "title": p.title,
                "insight": p.insight,
                "evidence": p.evidence,
                "priority": p.priority
            }
            for p in points
        ]
    }


@app.post("/api/coach-assist/ask")
async def ask_tactical_question(question: str):
    """
    Ask a tactical question about the match.

    Uses AI to provide answers based on match data.

    Args:
        question: The tactical question to ask
    """
    result = await coach_assist_service.ask_question(question)

    return {
        "question": result.question,
        "answer": result.answer,
        "confidence": result.confidence
    }


@app.get("/api/coach-assist/suggested-clips")
async def get_suggested_clips():
    """
    Get suggested clips for review.

    AI recommends key moments worth watching.
    """
    return {
        "clips": coach_assist_service.get_suggested_clips()
    }


@app.get("/api/coach-assist/export")
async def export_coach_analysis():
    """
    Export complete coaching analysis.

    Returns summary, talking points, and Q&A history.
    """
    return coach_assist_service.export_analysis()


@app.post("/api/coach-assist/set-ai")
async def set_coach_assist_ai(api_key: str, provider: str = "claude"):
    """
    Set AI client for Coach Assist.

    Args:
        api_key: API key for Claude or OpenAI
        provider: 'claude' or 'openai'
    """
    try:
        if provider == "claude":
            import anthropic
            client = anthropic.AsyncAnthropic(api_key=api_key)
        else:
            import openai
            client = openai.AsyncOpenAI(api_key=api_key)

        coach_assist_service.set_ai_client(client, provider)

        return {
            "status": "configured",
            "provider": provider
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to configure AI: {str(e)}")


@app.delete("/api/pro-analytics/reset")
async def reset_pro_analytics():
    """Reset all professional analytics services."""
    match_statistics_service.reset()
    pitch_visualization_service.reset()
    event_detector.reset()
    coach_assist_service.reset()

    return {"status": "reset", "services": ["match_statistics", "pitch_visualization", "event_detector", "coach_assist"]}


# ============== AI Expert Coach System ==============

from ai.expert_coach import expert_coach
from ai.technical_analysis import TECHNIQUE_LIBRARY, POSITION_REQUIREMENTS


@app.post("/api/expert-coach/initialize")
async def initialize_expert_coach(
    api_key: str,
    provider: str = "claude"
):
    """
    Initialize the AI Expert Coach with Vision AI capabilities.

    Args:
        api_key: API key for Claude or OpenAI
        provider: "claude" (recommended) or "openai"

    This enables UEFA Pro License level coaching analysis.
    """
    try:
        await expert_coach.initialize(api_key, provider)
        return {
            "status": "initialized",
            "provider": provider,
            "capabilities": [
                "Video clip analysis",
                "UEFA-standard technique assessment",
                "FA Four Corner Model evaluation",
                "Personalized development plans",
                "Professional coaching feedback"
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to initialize: {str(e)}")


@app.post("/api/expert-coach/analyze-action")
async def analyze_player_action(
    video_id: str,
    start_frame: int,
    end_frame: int,
    action_type: str,
    player_jersey: int,
    player_name: Optional[str] = None,
    player_position: Optional[str] = None
):
    """
    Analyze a specific player action using AI Vision.

    This is the core differentiator from VEO/HUDL - actual coaching intelligence.

    Args:
        video_id: ID of the uploaded video
        start_frame: Start frame of the action
        end_frame: End frame of the action
        action_type: Type of action (pass, shot, tackle, dribble, first_touch)
        player_jersey: Player's jersey number
        player_name: Optional player name
        player_position: Optional position (striker, midfielder, defender, etc.)

    Returns:
        Professional coaching analysis with:
        - Four Corner Model assessment
        - UEFA-standard technique breakdown
        - Specific coaching interventions
        - Recommended drills
    """
    services = get_services()
    video_service = services["video"]

    video_info = video_service.get_video_info(video_id)
    if not video_info:
        raise HTTPException(status_code=404, detail="Video not found")

    try:
        result = await expert_coach.analyze_player_action(
            video_path=video_info["path"],
            start_frame=start_frame,
            end_frame=end_frame,
            action_type=action_type,
            player_jersey=player_jersey,
            player_name=player_name,
            player_position=player_position
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.post("/api/expert-coach/development-plan/{jersey_number}")
async def generate_development_plan(
    jersey_number: int,
    player_name: Optional[str] = None,
    player_position: Optional[str] = None
):
    """
    Generate a personalized development plan for a player.

    Based on all accumulated analysis data, creates a comprehensive
    plan following UEFA coaching methodology.

    Returns:
        - Overall and Four Corner ratings
        - Key strengths to maintain
        - Prioritized development areas
        - Weekly focus and drills
        - Short and medium-term goals
    """
    plan = await expert_coach.generate_development_plan(
        player_jersey=jersey_number,
        player_name=player_name,
        player_position=player_position
    )

    return {
        "player_id": plan.player_id,
        "player_name": plan.player_name,
        "position": plan.position,
        "overall_rating": plan.overall_rating,
        "four_corner_ratings": plan.four_corner_ratings,
        "key_strengths": plan.key_strengths,
        "development_priorities": [
            {
                "issue": p.point,
                "category": p.category.value,
                "priority": p.priority.value,
                "coaching_cue": p.coaching_cue,
                "recommended_drill": p.drill_to_address
            }
            for p in plan.development_priorities
        ],
        "weekly_focus": plan.weekly_focus,
        "weekly_drills": plan.weekly_drills,
        "three_month_goals": plan.three_month_goals,
        "six_month_goals": plan.six_month_goals,
        "recent_observations": plan.recent_observations,
        "methodology": "UEFA Pro License / FA Four Corner Model"
    }


@app.get("/api/expert-coach/player-report/{jersey_number}")
async def get_player_coaching_report(jersey_number: int):
    """
    Get a comprehensive coaching report for a player.

    Combines development plan with vision analysis summary.
    """
    report = expert_coach.get_player_report(jersey_number)
    if not report:
        raise HTTPException(
            status_code=404,
            detail=f"No data for player #{jersey_number}. Analyze some actions first."
        )
    return report


@app.get("/api/expert-coach/technique-library")
async def get_technique_library():
    """
    Get the UEFA-aligned technique library.

    Returns all available techniques with coaching points,
    checkpoints, and drills.
    """
    library = {}
    for name, technique in TECHNIQUE_LIBRARY.items():
        library[name] = {
            "skill_name": technique.skill_name,
            "category": technique.category.value,
            "description": technique.description,
            "uefa_key_factors": technique.uefa_key_factors if hasattr(technique, 'uefa_key_factors') else [],
            "checkpoints": [
                {
                    "name": cp.name,
                    "what_to_look_for": cp.what_to_look_for,
                    "coaching_cues": cp.coaching_cues,
                    "four_corner_domain": cp.four_corner_domain.value if hasattr(cp, 'four_corner_domain') else "technical"
                }
                for cp in technique.checkpoints
            ],
            "drills": technique.drills_to_improve,
            "pro_examples": technique.pro_examples
        }
    return {"techniques": library, "methodology": "UEFA Coaching License Curriculum"}


@app.get("/api/expert-coach/position-requirements/{position}")
async def get_position_requirements_endpoint(position: str):
    """
    Get technical requirements for a specific position.

    Args:
        position: Position name (striker, midfielder, defender, goalkeeper, etc.)
    """
    # Try to find matching position
    position_lower = position.lower()
    for key, req in POSITION_REQUIREMENTS.items():
        if position_lower in key or key in position_lower:
            return {
                "position": req.position,
                "primary_skills": req.primary_skills,
                "secondary_skills": req.secondary_skills,
                "key_attributes": req.key_attributes,
                "common_weaknesses": req.common_weaknesses
            }

    raise HTTPException(
        status_code=404,
        detail=f"Position '{position}' not found. Try: goalkeeper, center_back, full_back, midfielder, winger, striker"
    )


@app.get("/api/expert-coach/four-corner-model")
async def get_four_corner_model_info():
    """
    Get information about the FA Four Corner Model used in our analysis.

    This is the holistic player development framework that underpins
    our coaching intelligence.
    """
    return {
        "name": "FA Four Corner Model",
        "description": "A holistic framework for player development considering technical, tactical, physical, and psychological aspects.",
        "source": "The Football Association (England)",
        "reference": "https://www.thefa.com/bootroom/resources/coaching/the-fas-4-corner-model",
        "corners": {
            "technical": {
                "description": "Ball mastery and technique execution",
                "elements": [
                    "Ball manipulation",
                    "Dribbling",
                    "Passing",
                    "First touch",
                    "Shooting",
                    "Heading",
                    "Crossing"
                ]
            },
            "tactical": {
                "description": "Decision making and game understanding",
                "elements": [
                    "Scanning and awareness",
                    "Decision making",
                    "Positioning",
                    "Movement off the ball",
                    "Understanding of game phases"
                ]
            },
            "physical": {
                "description": "Athletic capabilities",
                "elements": [
                    "Balance",
                    "Coordination",
                    "Speed",
                    "Strength",
                    "Endurance",
                    "Agility"
                ]
            },
            "psychological": {
                "description": "Mental attributes",
                "elements": [
                    "Composure under pressure",
                    "Confidence",
                    "Concentration",
                    "Resilience",
                    "Leadership",
                    "Communication"
                ]
            }
        },
        "key_principle": "No corner works in isolation - they are all interconnected in player development."
    }


@app.post("/api/expert-coach/export")
async def export_coaching_data(output_path: Optional[str] = None):
    """Export all coaching data to JSON."""
    if output_path is None:
        output_path = str(settings.UPLOAD_DIR / "expert_coach_data.json")

    result_path = expert_coach.export_all_data(output_path)

    return {
        "status": "exported",
        "output_path": result_path
    }


# ============== Team Profile API (THE SECRET WEAPON) ==============

@app.post("/api/team-profile/create")
async def create_team_profile(
    team_name: str,
    playing_style: str = "balanced",
    formation: str = "4-3-3"
):
    """
    Create a new customizable team profile.

    This is where you define YOUR team's identity. The AI Coach will learn
    and adapt to your specific philosophy, formation, and playing style.

    Args:
        team_name: Your team's name
        playing_style: One of: possession, counter_attack, high_press, low_block,
                      direct_play, balanced, positional_play
        formation: e.g., "4-3-3", "4-4-2", "3-5-2", "4-2-3-1"
    """
    try:
        profile = expert_coach.create_team_profile(team_name, playing_style, formation)
        return {
            "status": "created",
            "team_name": team_name,
            "playing_style": playing_style,
            "formation": formation,
            "message": f"Team profile created for {team_name}. Now customize your philosophy!"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/team-profile/create-from-template")
async def create_team_from_template(
    team_name: str,
    template: str
):
    """
    Create a team profile from a pre-built template.

    Templates available:
    - "possession": Tiki-taka style, patient build-up, high pressing
    - "counter_attack": Compact defense, fast transitions, direct play
    - "high_press": Gegenpressing, win ball high, immediate pressure
    """
    valid_templates = ["possession", "counter_attack", "high_press"]
    if template not in valid_templates:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid template. Choose from: {valid_templates}"
        )

    profile = expert_coach.create_from_template(team_name, template)

    return {
        "status": "created",
        "team_name": team_name,
        "template": template,
        "philosophy_statement": profile.philosophy.philosophy_statement,
        "message": f"Created {team_name} with {template} template. Customize further as needed!"
    }


@app.post("/api/team-profile/set-philosophy")
async def set_team_philosophy(
    philosophy_statement: str,
    team_motto: str = "",
    playing_style: Optional[str] = None,
    build_up_style: Optional[str] = None,
    pressing_intensity: Optional[str] = None,
    attacking_focus: Optional[str] = None,
    defensive_shape: Optional[str] = None,
    defensive_line_height: Optional[str] = None
):
    """
    Define your coaching philosophy in your own words.

    The AI Coach will use this to provide contextual, aligned feedback.
    """
    try:
        style_settings = {}
        if playing_style:
            style_settings["playing_style"] = playing_style
        if build_up_style:
            style_settings["build_up_style"] = build_up_style
        if pressing_intensity:
            style_settings["pressing_intensity"] = pressing_intensity
        if attacking_focus:
            style_settings["attacking_focus"] = attacking_focus
        if defensive_shape:
            style_settings["defensive_shape"] = defensive_shape
        if defensive_line_height:
            style_settings["defensive_line_height"] = defensive_line_height

        expert_coach.set_team_philosophy(
            philosophy_statement=philosophy_statement,
            team_motto=team_motto,
            **style_settings
        )

        return {
            "status": "updated",
            "philosophy_statement": philosophy_statement,
            "team_motto": team_motto,
            "style_settings": style_settings
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/team-profile/set-principles")
async def set_principles_priority(
    attacking_principles: List[str],
    defending_principles: List[str]
):
    """
    Set your priority order for principles of play.

    Attacking options: penetration, depth, width, mobility, improvisation
    Defending options: pressure, cover, compactness, delay, control_restraint

    Order matters - first = highest priority in your philosophy.
    """
    try:
        expert_coach.set_principles_priority(attacking_principles, defending_principles)
        return {
            "status": "updated",
            "attacking_principles": attacking_principles,
            "defending_principles": defending_principles,
            "message": "Principles priority updated. AI Coach will now prioritize these in analysis."
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/team-profile/add-player")
async def add_player_to_team(
    jersey_number: int,
    name: str,
    position: str,
    characteristics: Optional[List[str]] = None,
    development_focus: Optional[List[str]] = None,
    notes: str = "",
    age_group: str = "senior"
):
    """
    Add a player to your team profile for personalized coaching.

    The AI Coach will use this information to provide targeted feedback
    that considers the player's characteristics, development focus, and age group.

    Args:
        jersey_number: Shirt number (1-99)
        name: Player's name
        position: Primary position (GK, RB, CB, LB, CDM, CM, CAM, LW, RW, ST, etc.)
        characteristics: Player traits (e.g., ["left-footed", "quick", "good in air"])
        development_focus: Areas to prioritize (e.g., ["weak foot", "defensive awareness"])
        notes: Your coaching notes about this player
        age_group: u9, u12, u14, u16, u18, senior (affects coaching language)
    """
    try:
        player = expert_coach.add_team_player(
            jersey_number=jersey_number,
            name=name,
            position=position,
            characteristics=characteristics,
            development_focus=development_focus,
            notes=notes,
            age_group=age_group
        )

        return {
            "status": "added",
            "player": {
                "jersey_number": jersey_number,
                "name": name,
                "position": position,
                "characteristics": characteristics,
                "development_focus": development_focus,
                "age_group": age_group
            },
            "message": f"#{jersey_number} {name} added to team profile."
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.put("/api/team-profile/update-player/{jersey_number}")
async def update_player_profile(
    jersey_number: int,
    name: Optional[str] = None,
    position: Optional[str] = None,
    characteristics: Optional[List[str]] = None,
    development_focus: Optional[List[str]] = None,
    notes: Optional[str] = None,
    age_group: Optional[str] = None
):
    """Update an existing player's profile."""
    try:
        updates = {}
        if name is not None:
            updates["name"] = name
        if position is not None:
            updates["position"] = position
        if characteristics is not None:
            updates["characteristics"] = characteristics
        if development_focus is not None:
            updates["development_focus"] = development_focus
        if notes is not None:
            updates["notes"] = notes
        if age_group is not None:
            updates["age_group"] = age_group

        player = expert_coach.update_player_profile(jersey_number, **updates)

        return {
            "status": "updated",
            "jersey_number": jersey_number,
            "updates": updates
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/api/team-profile/player-four-corner/{jersey_number}")
async def set_player_four_corner(
    jersey_number: int,
    technical: float,
    tactical: float,
    physical: float,
    psychological: float
):
    """
    Set your assessment of a player's Four Corner ratings (1-10 scale).

    This helps the AI coach understand the player's current level and
    provide appropriate, targeted feedback.
    """
    try:
        expert_coach.set_player_four_corner_ratings(
            jersey_number=jersey_number,
            technical=technical,
            tactical=tactical,
            physical=physical,
            psychological=psychological
        )

        return {
            "status": "updated",
            "jersey_number": jersey_number,
            "four_corner_ratings": {
                "technical": technical,
                "tactical": tactical,
                "physical": physical,
                "psychological": psychological
            }
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/api/team-profile/positional-instruction")
async def add_positional_instruction(
    position: str,
    instruction_key: str,
    value: str
):
    """
    Add specific instructions for a position in your system.

    Examples:
        - position: "RB", instruction_key: "attacking_runs", value: "overlap"
        - position: "CDM", instruction_key: "defensive_priority", value: "screen back four"
        - position: "ST", instruction_key: "pressing_trigger", value: "ball to CB"
    """
    try:
        expert_coach.add_positional_instruction(position, instruction_key, value)

        return {
            "status": "added",
            "position": position,
            "instruction": {instruction_key: value}
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/team-profile/key-partnership")
async def add_key_partnership(
    positions: List[str],
    pattern: str,
    description: str = ""
):
    """
    Define key partnerships or combinations you want to develop.

    Examples:
        - positions: ["CM", "ST"], pattern: "through ball runs"
        - positions: ["RW", "RB"], pattern: "overlap combination"
    """
    try:
        expert_coach.add_key_partnership(positions, pattern, description)

        return {
            "status": "added",
            "partnership": {
                "positions": positions,
                "pattern": pattern,
                "description": description
            }
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/team-profile/season-objectives")
async def set_season_objectives(objectives: List[str]):
    """Set your team's season objectives."""
    try:
        expert_coach.set_season_objectives(objectives)
        return {
            "status": "updated",
            "objectives": objectives
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/team-profile/development-themes")
async def set_development_themes(themes: List[str]):
    """
    Set development themes for the season.

    Examples: "Playing out from back", "Counter-pressing", "1v1 defending"
    """
    try:
        expert_coach.set_development_themes(themes)
        return {
            "status": "updated",
            "themes": themes
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/team-profile/save")
async def save_team_profile(filename: Optional[str] = None):
    """Save the current team profile to disk."""
    try:
        filepath = expert_coach.save_team_profile(filename)
        return {
            "status": "saved",
            "filepath": filepath
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/team-profile/load")
async def load_team_profile(filename: str):
    """Load a saved team profile."""
    try:
        profile = expert_coach.load_team_profile(filename)
        return {
            "status": "loaded",
            "team_name": profile.philosophy.team_name,
            "playing_style": profile.philosophy.playing_style.value,
            "formation": profile.formation.primary_formation,
            "player_count": len(profile.players)
        }
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Profile not found: {filename}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/team-profile/list")
async def list_team_profiles():
    """List all saved team profiles."""
    profiles = expert_coach.list_saved_profiles()
    return {
        "profiles": profiles,
        "count": len(profiles)
    }


@app.get("/api/team-profile/context")
async def get_team_context():
    """
    Get the full team context that the AI Coach uses.

    This shows exactly what the AI understands about your team.
    """
    context = expert_coach.get_team_context()
    return {
        "context": context,
        "message": "This is the context the AI Coach uses for personalized feedback."
    }


@app.get("/api/team-profile/player-context/{jersey_number}")
async def get_player_context(jersey_number: int):
    """Get the context the AI Coach has for a specific player."""
    context = expert_coach.get_player_context(jersey_number)
    return {
        "jersey_number": jersey_number,
        "context": context
    }


@app.get("/api/team-profile/match-analysis-context")
async def get_match_analysis_context():
    """Get the context used for match analysis based on team philosophy."""
    context = expert_coach.get_match_analysis_context()
    return {
        "context": context,
        "message": "This context guides what the AI focuses on during match analysis."
    }


@app.get("/api/team-profile/philosophy-options")
async def get_philosophy_options():
    """
    Get all available options for customizing team philosophy.

    Returns the enums and formations you can use when setting up your team.
    """
    from ai.team_profile import (
        PlayingStyle, BuildUpStyle, PressingIntensity, TransitionSpeed,
        WingPlay, DefensiveShape, AttackingFocus, FORMATION_TEMPLATES,
        PRINCIPLES_OF_PLAY, FOUR_MOMENTS
    )

    return {
        "playing_styles": [e.value for e in PlayingStyle],
        "build_up_styles": [e.value for e in BuildUpStyle],
        "pressing_intensities": [e.value for e in PressingIntensity],
        "transition_speeds": [e.value for e in TransitionSpeed],
        "wing_play_options": [e.value for e in WingPlay],
        "defensive_shapes": [e.value for e in DefensiveShape],
        "attacking_focus_options": [e.value for e in AttackingFocus],
        "formations": list(FORMATION_TEMPLATES.keys()),
        "attacking_principles": list(PRINCIPLES_OF_PLAY["attacking"].keys()),
        "defending_principles": list(PRINCIPLES_OF_PLAY["defending"].keys()),
        "four_moments": list(FOUR_MOMENTS.keys())
    }


@app.get("/api/team-profile/principles-of-play")
async def get_principles_of_play():
    """
    Get detailed information about UEFA Principles of Play.

    Use this to understand what each principle means when setting priorities.
    """
    from ai.team_profile import PRINCIPLES_OF_PLAY
    return PRINCIPLES_OF_PLAY


@app.get("/api/team-profile/four-moments")
async def get_four_moments():
    """
    Get information about the Four Moments of the Game (UEFA framework).

    Understanding these helps you define your team's approach to each phase.
    """
    from ai.team_profile import FOUR_MOMENTS
    return FOUR_MOMENTS


@app.get("/api/team-profile/formation-info/{formation}")
async def get_formation_info(formation: str):
    """
    Get detailed information about a specific formation.

    Includes positions, strengths, weaknesses, and tactical shape.
    """
    from ai.team_profile import FORMATION_TEMPLATES

    if formation not in FORMATION_TEMPLATES:
        raise HTTPException(
            status_code=404,
            detail=f"Formation not found. Available: {list(FORMATION_TEMPLATES.keys())}"
        )

    return {
        "formation": formation,
        **FORMATION_TEMPLATES[formation]
    }


# ============== Tactical Analysis API ==============

from ai.tactical_analysis import (
    TacticalAnalysisService,
    PlayerPosition,
    TacticalPhase,
    TacticalUnit,
    tactical_analysis
)


@app.post("/api/tactical/analyze-formation")
async def analyze_formation_effectiveness(
    players: List[Dict],
    ball_position: Optional[List[float]] = None,
    team: str = "home",
    formation: str = "4-3-3",
    playing_style: str = "balanced"
):
    """
    Analyze current formation effectiveness.

    This is the KEY tactical metric - tells you how well the team is set up.

    Args:
        players: List of player positions with x, y (0-100 normalized), track_id, team
        ball_position: Optional [x, y] of ball
        team: "home" or "away"
        formation: Expected formation (e.g., "4-3-3")
        playing_style: Team's playing style for context
    """
    try:
        # Convert to PlayerPosition objects
        player_positions = [
            PlayerPosition(
                track_id=p.get("track_id", i),
                jersey_number=p.get("jersey_number"),
                x=p.get("x", 50),
                y=p.get("y", 50),
                team=p.get("team", team)
            )
            for i, p in enumerate(players)
        ]

        ball_pos = tuple(ball_position) if ball_position else None

        # Get playing style enum
        from ai.team_profile import PlayingStyle
        try:
            style = PlayingStyle(playing_style)
        except ValueError:
            style = PlayingStyle.BALANCED

        # Calculate effectiveness
        score = tactical_analysis.calculate_formation_effectiveness(
            players=player_positions,
            ball_position=ball_pos,
            team=team,
            team_formation=formation,
            team_style=style
        )

        return {
            "formation_detected": score.formation,
            "overall_effectiveness": score.overall_effectiveness,
            "component_scores": {
                "shape_maintenance": score.shape_maintenance,
                "compactness": score.compactness_score,
                "width_balance": score.width_balance,
                "depth_balance": score.depth_balance,
                "phase_appropriateness": score.phase_appropriateness
            },
            "current_phase": score.phase.value,
            "unit_scores": {
                unit: {
                    "compactness": metrics.overall_compactness,
                    "line_alignment": metrics.line_alignment,
                    "spacing_quality": metrics.spacing_quality,
                    "average_height": metrics.average_height
                }
                for unit, metrics in score.unit_scores.items()
            },
            "deviation_from_ideal": score.deviation_from_ideal,
            "vulnerabilities": score.vulnerabilities,
            "strengths": score.strengths,
            "recommendations": score.recommendations
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/tactical/detect-formation")
async def detect_formation(
    players: List[Dict],
    team: str = "home"
):
    """
    Detect team formation from player positions.

    Returns the most likely formation and confidence level.
    """
    try:
        player_positions = [
            PlayerPosition(
                track_id=p.get("track_id", i),
                jersey_number=p.get("jersey_number"),
                x=p.get("x", 50),
                y=p.get("y", 50),
                team=p.get("team", team)
            )
            for i, p in enumerate(players)
        ]

        formation, confidence = tactical_analysis.formation_detector.detect_formation(
            player_positions, team
        )

        # Assign roles
        players_with_roles = tactical_analysis.formation_detector.assign_roles(
            player_positions, formation, team
        )

        return {
            "detected_formation": formation,
            "confidence": round(confidence, 2),
            "player_roles": [
                {
                    "track_id": p.track_id,
                    "jersey_number": p.jersey_number,
                    "position": {"x": p.x, "y": p.y},
                    "assigned_role": p.assigned_role
                }
                for p in players_with_roles if p.team == team
            ]
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/tactical/unit-analysis/{unit}")
async def get_unit_analysis(unit: str):
    """
    Get analysis for a specific tactical unit (defensive/midfield/attacking).

    Based on the most recent effectiveness score.
    """
    try:
        unit_enum = TacticalUnit(unit)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid unit. Choose from: defensive, midfield, attacking"
        )

    if not tactical_analysis.effectiveness_scores:
        raise HTTPException(status_code=404, detail="No tactical data available yet")

    # Get most recent score
    latest = tactical_analysis.effectiveness_scores[-1]
    unit_metrics = latest.unit_scores.get(unit_enum.value)

    if not unit_metrics:
        raise HTTPException(status_code=404, detail=f"No data for {unit} unit")

    return {
        "unit": unit,
        "metrics": {
            "horizontal_compactness": unit_metrics.horizontal_compactness,
            "vertical_compactness": unit_metrics.vertical_compactness,
            "overall_compactness": unit_metrics.overall_compactness,
            "average_height": unit_metrics.average_height,
            "width_coverage": unit_metrics.width_coverage,
            "line_alignment": unit_metrics.line_alignment,
            "spacing_quality": unit_metrics.spacing_quality,
            "defensive_coverage": unit_metrics.defensive_coverage,
            "attacking_support": unit_metrics.attacking_support
        },
        "interpretation": _interpret_unit_metrics(unit_enum, unit_metrics)
    }


def _interpret_unit_metrics(unit: TacticalUnit, metrics) -> Dict:
    """Generate human-readable interpretation of unit metrics."""
    interpretation = {}

    if unit == TacticalUnit.DEFENSIVE:
        if metrics.line_alignment > 75:
            interpretation["line"] = "Excellent defensive line discipline"
        elif metrics.line_alignment > 50:
            interpretation["line"] = "Reasonable line shape, some inconsistency"
        else:
            interpretation["line"] = "Poor line shape - vulnerable to through balls"

        if metrics.overall_compactness > 70:
            interpretation["compactness"] = "Good compact shape"
        else:
            interpretation["compactness"] = "Too stretched - gaps between defenders"

    elif unit == TacticalUnit.MIDFIELD:
        if metrics.spacing_quality > 70:
            interpretation["spacing"] = "Well spaced for passing options"
        else:
            interpretation["spacing"] = "Uneven spacing - limiting passing lanes"

        if metrics.overall_compactness > 65:
            interpretation["compactness"] = "Compact midfield unit"
        else:
            interpretation["compactness"] = "Midfield too spread out"

    elif unit == TacticalUnit.ATTACKING:
        if metrics.attacking_support > 70:
            interpretation["support"] = "Good attacking positions"
        else:
            interpretation["support"] = "Need better attacking movement"

    return interpretation


@app.post("/api/tactical/analyze-period")
async def analyze_tactical_period(
    start_time_ms: int,
    end_time_ms: int
):
    """
    Analyze a period of play (e.g., first 15 minutes, second half).

    Returns comprehensive tactical breakdown for the period.
    """
    analysis = tactical_analysis.analyze_period(start_time_ms, end_time_ms)

    if not analysis:
        raise HTTPException(
            status_code=404,
            detail="No tactical data in specified time range"
        )

    return {
        "period": {
            "start_ms": analysis.start_time_ms,
            "end_ms": analysis.end_time_ms
        },
        "phase_breakdown": analysis.phase_percentages,
        "formation": {
            "changes": analysis.formation_changes,
            "avg_effectiveness": analysis.avg_formation_score
        },
        "pressing_intensity": analysis.pressing_intensity,
        "possession_style": analysis.possession_style,
        "width_usage": analysis.width_usage,
        "unit_averages": {
            "defensive": {
                "compactness": analysis.defensive_unit_avg.overall_compactness,
                "line_alignment": analysis.defensive_unit_avg.line_alignment
            },
            "midfield": {
                "compactness": analysis.midfield_unit_avg.overall_compactness,
                "spacing_quality": analysis.midfield_unit_avg.spacing_quality
            },
            "attacking": {
                "support": analysis.attacking_unit_avg.attacking_support,
                "height": analysis.attacking_unit_avg.average_height
            }
        },
        "formation_breakdowns": len(analysis.formation_breakdowns),
        "observations": analysis.tactical_observations
    }


@app.get("/api/tactical/match-report")
async def get_tactical_match_report():
    """
    Get comprehensive tactical report for the match.

    Aggregates all tactical analysis into a single report.
    """
    report = tactical_analysis.get_match_tactical_report()

    if "error" in report:
        raise HTTPException(status_code=404, detail=report["error"])

    return report


@app.get("/api/tactical/current-phase")
async def get_current_tactical_phase():
    """
    Get the current phase of play based on most recent analysis.

    Returns one of: attacking_organization, defensive_organization,
    attacking_transition, defensive_transition
    """
    if not tactical_analysis.effectiveness_scores:
        return {"phase": "unknown", "message": "No tactical data yet"}

    latest = tactical_analysis.effectiveness_scores[-1]

    phase_info = {
        TacticalPhase.ATTACKING_ORGANIZATION: {
            "description": "Team is in possession with time to organize",
            "focus": "Build-up, progression, creating chances"
        },
        TacticalPhase.DEFENSIVE_ORGANIZATION: {
            "description": "Team is defending with time to organize",
            "focus": "Shape, compactness, denying space"
        },
        TacticalPhase.ATTACKING_TRANSITION: {
            "description": "Team just won possession",
            "focus": "Quick decision - counter or secure?"
        },
        TacticalPhase.DEFENSIVE_TRANSITION: {
            "description": "Team just lost possession",
            "focus": "Counter-press or recover?"
        }
    }

    return {
        "phase": latest.phase.value,
        "info": phase_info.get(latest.phase, {}),
        "formation_effectiveness": latest.overall_effectiveness,
        "phase_appropriateness": latest.phase_appropriateness
    }


@app.get("/api/tactical/formation-positions/{formation}")
async def get_formation_positions(formation: str):
    """
    Get ideal position coordinates for a formation.

    Useful for visualization and comparison.
    """
    from ai.tactical_analysis import FORMATION_POSITIONS, UNIT_POSITIONS

    if formation not in FORMATION_POSITIONS:
        raise HTTPException(
            status_code=404,
            detail=f"Formation not found. Available: {list(FORMATION_POSITIONS.keys())}"
        )

    positions = FORMATION_POSITIONS[formation]
    units = UNIT_POSITIONS.get(formation, {})

    return {
        "formation": formation,
        "positions": {
            role: {"x": pos[0], "y": pos[1]}
            for role, pos in positions.items()
        },
        "units": {
            unit.value: roles
            for unit, roles in units.items()
        }
    }


@app.post("/api/tactical/export")
async def export_tactical_data(output_path: Optional[str] = None):
    """Export all tactical analysis data to JSON."""
    if output_path is None:
        output_path = str(settings.UPLOAD_DIR / "tactical_analysis.json")

    filepath = tactical_analysis.export_data(output_path)

    return {
        "status": "exported",
        "filepath": filepath,
        "total_snapshots": len(tactical_analysis.effectiveness_scores)
    }


@app.post("/api/tactical/reset")
async def reset_tactical_analysis():
    """Reset all tactical analysis data."""
    tactical_analysis.reset()
    return {"status": "reset", "message": "Tactical analysis data cleared"}


# ============== ML Dataset Export API ==============

from services.ml_export import ml_export, MLExportService


@app.post("/api/ml-export/add-action")
async def add_action_to_ml_dataset(
    action_id: str,
    video_id: str,
    player_jersey: int,
    action_type: str,
    start_frame: int,
    end_frame: int,
    overall_score: float,
    outcome: str = "unknown",
    player_name: Optional[str] = None,
    player_position: Optional[str] = None,
    technical_score: float = 0.5,
    tactical_score: float = 0.5,
    physical_score: float = 0.5,
    psychological_score: float = 0.5,
    team: str = "home",
    formation: str = "4-3-3",
    phase: str = "attacking_organization"
):
    """
    Add an action analysis to the ML dataset.

    This builds up training data for machine learning models.
    """
    scores = {
        "overall": overall_score,
        "four_corner": {
            "technical": {"score": technical_score * 10},
            "tactical": {"score": tactical_score * 10},
            "physical": {"score": physical_score * 10},
            "psychological": {"score": psychological_score * 10}
        },
        "technique_scores": {}
    }

    ml_export.add_action_analysis(
        action_id=action_id,
        video_id=video_id,
        player_jersey=player_jersey,
        action_type=action_type,
        start_frame=start_frame,
        end_frame=end_frame,
        scores=scores,
        outcome=outcome,
        player_name=player_name,
        player_position=player_position,
        team=team,
        formation=formation,
        phase=phase
    )

    return {
        "status": "added",
        "action_id": action_id,
        "total_actions": len(ml_export.action_data)
    }


@app.post("/api/ml-export/add-tactical")
async def add_tactical_to_ml_dataset(
    video_id: str,
    frame_number: int,
    players: List[Dict],
    formation: str,
    effectiveness: float,
    phase: str,
    ball_position: Optional[List[float]] = None
):
    """Add a tactical snapshot to the ML dataset."""
    ml_export.add_tactical_snapshot(
        video_id=video_id,
        frame_number=frame_number,
        players=players,
        formation=formation,
        effectiveness=effectiveness,
        phase=phase,
        ball_position=tuple(ball_position) if ball_position else None
    )

    return {
        "status": "added",
        "total_tactical": len(ml_export.tactical_data)
    }


@app.post("/api/ml-export/export-csv")
async def export_ml_csv():
    """
    Export all collected data to CSV format.

    Perfect for sklearn, pandas, XGBoost, etc.
    """
    exports = {
        "actions": ml_export.export_actions_csv(),
        "tactical": ml_export.export_tactical_csv(),
        "players": ml_export.export_player_records_csv()
    }

    return {
        "status": "exported",
        "format": "csv",
        "files": exports
    }


@app.post("/api/ml-export/export-json")
async def export_ml_json():
    """
    Export all collected data to JSON format.

    Perfect for deep learning and custom data loading.
    """
    exports = {
        "actions": ml_export.export_actions_json(),
        "tactical": ml_export.export_tactical_json()
    }

    return {
        "status": "exported",
        "format": "json",
        "files": exports
    }


@app.post("/api/ml-export/export-pytorch")
async def export_pytorch_dataset():
    """
    Export data in PyTorch-friendly format.

    Ready to use with torch.utils.data.Dataset.
    """
    filepath = ml_export.export_pytorch_format()

    return {
        "status": "exported",
        "format": "pytorch",
        "filepath": filepath,
        "usage_example": """
from torch.utils.data import Dataset
import json

class FootballActionDataset(Dataset):
    def __init__(self, json_path):
        with open(json_path) as f:
            data = json.load(f)
        self.samples = data['samples']

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        features = torch.tensor(sample['features'], dtype=torch.float32)
        label = sample['labels']['outcome']
        return features, label
"""
    }


@app.post("/api/ml-export/export-all")
async def export_all_ml_formats(prefix: str = ""):
    """
    Export data in ALL available formats.

    Returns paths to all exported files.
    """
    exports = ml_export.export_all(prefix)

    return {
        "status": "exported",
        "files": exports,
        "stats": ml_export.get_dataset_stats()
    }


@app.get("/api/ml-export/stats")
async def get_ml_dataset_stats():
    """
    Get statistics about the collected ML dataset.

    Shows distributions, counts, and data quality metrics.
    """
    return ml_export.get_dataset_stats()


@app.get("/api/ml-export/player-records")
async def get_ml_player_records():
    """Get aggregated player performance records."""
    records = []
    for jersey, record in ml_export.player_records.items():
        records.append({
            "jersey": record.player_jersey,
            "name": record.player_name,
            "position": record.player_position,
            "total_actions": record.total_actions,
            "avg_scores": {
                "overall": round(record.avg_overall_score, 3),
                "technical": round(record.avg_technical_score, 3),
                "tactical": round(record.avg_tactical_score, 3),
                "physical": round(record.avg_physical_score, 3),
                "psychological": round(record.avg_psychological_score, 3)
            },
            "action_stats": {
                "passes": {
                    "attempted": record.passes_attempted,
                    "successful": record.passes_successful,
                    "accuracy": round(record.pass_accuracy, 3)
                },
                "shots": {
                    "attempted": record.shots_attempted,
                    "on_target": record.shots_on_target,
                    "accuracy": round(record.shot_accuracy, 3)
                },
                "dribbles": {
                    "attempted": record.dribbles_attempted,
                    "successful": record.dribbles_successful,
                    "success_rate": round(record.dribble_success_rate, 3)
                },
                "tackles": {
                    "attempted": record.tackles_attempted,
                    "won": record.tackles_won,
                    "success_rate": round(record.tackle_success_rate, 3)
                }
            }
        })

    return {
        "total_players": len(records),
        "records": records
    }


@app.post("/api/ml-export/reset")
async def reset_ml_dataset():
    """Clear all collected ML data."""
    ml_export.reset()
    return {"status": "reset", "message": "ML dataset cleared"}


# ============== Unified Coaching Intelligence API ==============

@app.get("/api/coaching-intelligence/full-report")
async def get_full_coaching_report():
    """
    Get comprehensive coaching intelligence report for all platform tabs.

    This is the MASTER endpoint that combines all AI coaching services:
    - Match Video tab: Detection data with coaching annotations
    - Match Analysis tab: Stats, heatmaps, formations with expert insights
    - AI Coach tab: Professional coaching recommendations
    - Tactical Events tab: Key moments with coaching context
    - Players tab: Individual player assessments

    This is what differentiates us from VEO/HUDL - actual coaching intelligence!
    """
    import json
    import os
    from datetime import datetime

    json_path = "C:/Users/info/football-analyzer/backend/uploads/balti-away-30min_analysis.json"
    if not os.path.exists(json_path):
        raise HTTPException(status_code=404, detail="Analysis file not found. Please process a video first.")

    try:
        with open(json_path, 'r') as f:
            data = json.load(f)

        frame_analyses = data.get('frames', [])
        if not frame_analyses:
            return {"error": "No frame data available"}

        # Get base analysis
        pass_stats = pass_detector.analyze_from_frames(frame_analyses)
        formation_stats = formation_detector.analyze_from_frames(frame_analyses)
        tactical_events = tactical_detector.analyze_from_frames(frame_analyses)

        # Get AI coaching analysis
        coaching_analysis = ai_coach.analyze_match(
            pass_stats=pass_stats,
            formation_stats=formation_stats,
            tactical_events=tactical_events,
            frame_analyses=frame_analyses
        )

        # Build formation analysis from frames
        home_formations = []
        away_formations = []
        for frame in frame_analyses[::100]:  # Sample every 100 frames
            home_players = [d for d in frame.get("detections", []) if d.get("team") == "home"]
            away_players = [d for d in frame.get("detections", []) if d.get("team") == "away"]

            if len(home_players) >= 7:
                home_formations.append(len(home_players))
            if len(away_players) >= 7:
                away_formations.append(len(away_players))

        # Calculate match statistics
        total_home = sum(f.get("home_players", 0) for f in frame_analyses)
        total_away = sum(f.get("away_players", 0) for f in frame_analyses)
        total_frames = len(frame_analyses)

        avg_home = total_home / total_frames if total_frames > 0 else 0
        avg_away = total_away / total_frames if total_frames > 0 else 0

        # Build comprehensive report
        report = {
            "generated_at": datetime.now().isoformat(),
            "video_info": {
                "path": data.get("video_path"),
                "duration_seconds": data.get("duration_seconds"),
                "total_frames": data.get("total_frames"),
                "analyzed_frames": data.get("analyzed_frames"),
                "fps_analyzed": data.get("fps_analyzed")
            },

            # TAB 1: Match Video - frame data is served separately via /api/video/full-analysis
            "match_video": {
                "total_frames_analyzed": len(frame_analyses),
                "tracking_quality": "good" if avg_home > 5 and avg_away > 5 else "partial",
                "avg_players_tracked": {
                    "home": round(avg_home, 1),
                    "away": round(avg_away, 1)
                }
            },

            # TAB 2: Match Analysis - enhanced with coaching context
            "match_analysis": {
                "possession": pass_stats.get("possession", {"home": 50, "away": 50}),
                "passing": {
                    "home": {
                        "total": pass_stats.get("home_passes", 0),
                        "completed": pass_stats.get("home_completed", 0),
                        "accuracy": pass_stats.get("home_accuracy", 0)
                    },
                    "away": {
                        "total": pass_stats.get("away_passes", 0),
                        "completed": pass_stats.get("away_completed", 0),
                        "accuracy": pass_stats.get("away_accuracy", 0)
                    }
                },
                "formations_detected": formation_stats,
                "territorial_control": _calculate_territorial_control(frame_analyses),
                "coaching_notes": _generate_analysis_coaching_notes(pass_stats, formation_stats)
            },

            # TAB 3: AI Coach - the main coaching intelligence
            "ai_coach": {
                "summary": coaching_analysis.get("summary", {}),
                "insights": coaching_analysis.get("insights", []),
                "critical_insights": [i for i in coaching_analysis.get("insights", [])
                                      if i.get("priority") in ["critical", "high"]],
                "team_talks": {
                    "half_time": coaching_analysis.get("summary", {}).get("half_time_message", ""),
                    "full_time": coaching_analysis.get("summary", {}).get("full_time_message", "")
                },
                "total_insights": len(coaching_analysis.get("insights", []))
            },

            # TAB 4: Tactical Events - key moments with coaching context
            "tactical_events": {
                "events": tactical_events.get("events", [])[:50],  # Top 50 events
                "total_events": tactical_events.get("total_events", 0),
                "events_by_type": tactical_events.get("event_counts", {}),
                "momentum_periods": _calculate_momentum_periods(frame_analyses),
                "key_phases": _identify_key_phases(tactical_events.get("events", []))
            },

            # TAB 5: Players - individual assessments
            "players": {
                "home_team": _build_player_summary(frame_analyses, "home"),
                "away_team": _build_player_summary(frame_analyses, "away"),
                "coaching_focus": _identify_player_coaching_focus(frame_analyses)
            },

            # Coaching Intelligence Summary
            "coaching_intelligence": {
                "overall_performance": coaching_analysis.get("summary", {}).get("overall_rating", "Average"),
                "key_strengths": coaching_analysis.get("summary", {}).get("key_strengths", []),
                "improvement_areas": coaching_analysis.get("summary", {}).get("areas_to_improve", []),
                "tactical_summary": coaching_analysis.get("summary", {}).get("tactical_summary", ""),
                "expert_coach_available": expert_coach.is_initialized if hasattr(expert_coach, 'is_initialized') else False,
                "team_profile_loaded": False,  # Will be True if team profile is set
                "services_active": {
                    "ai_coach": True,
                    "expert_coach": expert_coach.is_initialized if hasattr(expert_coach, 'is_initialized') else False,
                    "tactical_analysis": True,
                    "ml_export": True
                }
            }
        }

        return report

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


def _calculate_territorial_control(frames: list) -> dict:
    """Calculate territorial control from frame data."""
    home_third = 0
    mid_third = 0
    away_third = 0

    for frame in frames:
        ball_x = frame.get("ball_pitch_x")
        if ball_x is not None:
            if ball_x < 33:
                home_third += 1
            elif ball_x < 66:
                mid_third += 1
            else:
                away_third += 1

    total = home_third + mid_third + away_third
    if total == 0:
        return {"defensive_third": 33, "middle_third": 34, "attacking_third": 33}

    return {
        "defensive_third": round(home_third / total * 100, 1),
        "middle_third": round(mid_third / total * 100, 1),
        "attacking_third": round(away_third / total * 100, 1)
    }


def _generate_analysis_coaching_notes(pass_stats: dict, formation_stats: dict) -> list:
    """Generate coaching notes from analysis."""
    notes = []

    possession = pass_stats.get("possession", {})
    if possession.get("home", 50) > 60:
        notes.append({
            "type": "positive",
            "message": "Strong ball retention - controlling the tempo well"
        })
    elif possession.get("home", 50) < 40:
        notes.append({
            "type": "warning",
            "message": "Low possession - consider pressing higher to win the ball back"
        })

    home_accuracy = pass_stats.get("home_accuracy", 0)
    if home_accuracy > 85:
        notes.append({
            "type": "positive",
            "message": f"Excellent passing accuracy ({home_accuracy}%) - keep playing quick combinations"
        })
    elif home_accuracy < 70:
        notes.append({
            "type": "improvement",
            "message": f"Passing accuracy ({home_accuracy}%) needs work - focus on weight of pass"
        })

    return notes


def _count_events_by_type(events: list) -> dict:
    """Count tactical events by type."""
    counts = {}
    for event in events:
        event_type = event.get("type", "unknown")
        counts[event_type] = counts.get(event_type, 0) + 1
    return counts


def _calculate_momentum_periods(frames: list) -> list:
    """Calculate momentum periods from frame data."""
    periods = []
    window_size = 50
    current_period_start = 0

    for i in range(0, len(frames), window_size):
        window = frames[i:i + window_size]
        home_count = sum(f.get("home_players", 0) for f in window)
        away_count = sum(f.get("away_players", 0) for f in window)

        if home_count > away_count * 1.2:
            momentum = "home"
        elif away_count > home_count * 1.2:
            momentum = "away"
        else:
            momentum = "neutral"

        if window:
            periods.append({
                "start_frame": i,
                "end_frame": min(i + window_size, len(frames)),
                "start_time": window[0].get("timestamp", 0),
                "end_time": window[-1].get("timestamp", 0),
                "momentum": momentum
            })

    return periods[:20]  # Limit to 20 periods


def _identify_key_phases(events: list) -> list:
    """Identify key tactical phases from events."""
    key_phases = []
    high_priority_events = [e for e in events if e.get("priority", 0) >= 3]

    for event in high_priority_events[:10]:
        key_phases.append({
            "timestamp": event.get("timestamp", 0),
            "type": event.get("type", "unknown"),
            "description": event.get("description", ""),
            "coaching_point": _get_coaching_point_for_event(event)
        })

    return key_phases


def _get_coaching_point_for_event(event: dict) -> str:
    """Get a coaching point for a tactical event."""
    event_type = event.get("type", "")

    coaching_points = {
        "pressing_trigger": "Good pressing trigger - ensure supporting players recover spaces",
        "dangerous_attack": "Quality attacking move - look to replicate the build-up pattern",
        "counter_attack": "Counter-attack opportunity - practice quick transitions in training",
        "shape_warning": "Team shape compromised - work on maintaining distances between lines",
        "high_line_opportunity": "Space behind the line - practice long balls over the top",
        "overload": "Numerical advantage created - exploit with quick combinations"
    }

    return coaching_points.get(event_type, "Review this moment in detail with the team")


def _build_player_summary(frames: list, team: str) -> list:
    """Build summary of players detected for a team."""
    player_appearances = {}

    for frame in frames:
        for detection in frame.get("detections", []):
            if detection.get("team") == team:
                track_id = detection.get("track_id") or "unknown"
                if track_id not in player_appearances:
                    player_appearances[track_id] = {
                        "track_id": track_id,
                        "appearances": 0,
                        "positions": []
                    }
                player_appearances[track_id]["appearances"] += 1

                bbox = detection.get("bbox", [])
                if bbox:
                    center_x = (bbox[0] + bbox[2]) / 2
                    center_y = (bbox[1] + bbox[3]) / 2
                    player_appearances[track_id]["positions"].append((center_x, center_y))

    # Convert to list and calculate average positions
    players = []
    for track_id, data in player_appearances.items():
        if data["appearances"] > 10:  # Only include players with significant presence
            avg_x = sum(p[0] for p in data["positions"]) / len(data["positions"]) if data["positions"] else 0
            avg_y = sum(p[1] for p in data["positions"]) / len(data["positions"]) if data["positions"] else 0
            players.append({
                "track_id": track_id,
                "appearances": data["appearances"],
                "avg_position": {"x": round(avg_x, 1), "y": round(avg_y, 1)},
                "play_time_estimate": f"{data['appearances'] / 3:.0f}s"  # Based on 3 fps
            })

    return sorted(players, key=lambda x: x["appearances"], reverse=True)[:15]


def _identify_player_coaching_focus(frames: list) -> list:
    """Identify players that need coaching focus."""
    focus_points = []

    # This would be enhanced with actual performance data
    focus_points.append({
        "area": "Defensive Shape",
        "description": "Work on maintaining compact lines when opponent has ball",
        "priority": "high"
    })
    focus_points.append({
        "area": "Transition Speed",
        "description": "Quicker ball movement when winning possession",
        "priority": "medium"
    })
    focus_points.append({
        "area": "Pressing Triggers",
        "description": "Practice coordinated pressing from front to back",
        "priority": "medium"
    })

    return focus_points


@app.get("/api/coaching-intelligence/tab/{tab_name}")
async def get_coaching_for_tab(tab_name: str):
    """
    Get coaching intelligence specific to a frontend tab.

    Args:
        tab_name: One of "video", "analysis", "aicoach", "tactical", "players"
    """
    report = await get_full_coaching_report()

    tab_mapping = {
        "video": "match_video",
        "analysis": "match_analysis",
        "aicoach": "ai_coach",
        "tactical": "tactical_events",
        "players": "players"
    }

    if tab_name not in tab_mapping:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown tab. Available: {list(tab_mapping.keys())}"
        )

    return {
        "tab": tab_name,
        "data": report.get(tab_mapping[tab_name], {}),
        "coaching_intelligence": report.get("coaching_intelligence", {})
    }


@app.get("/api/coaching-intelligence/status")
async def get_coaching_status():
    """Get status of all coaching intelligence services."""
    return {
        "services": {
            "ai_coach": {
                "active": True,
                "description": "Basic AI coaching insights and recommendations"
            },
            "expert_coach": {
                "active": expert_coach.is_initialized if hasattr(expert_coach, 'is_initialized') else False,
                "description": "UEFA Pro License level analysis with Vision AI"
            },
            "tactical_analysis": {
                "active": True,
                "snapshots": len(tactical_analysis.effectiveness_scores),
                "description": "Formation effectiveness and unit analysis"
            },
            "ml_export": {
                "active": True,
                "actions_recorded": len(ml_export.action_data),
                "description": "ML-ready dataset generation"
            }
        },
        "capabilities": [
            "Match video overlay with tracking",
            "Comprehensive match statistics",
            "AI-powered coaching insights",
            "Tactical event detection",
            "Player performance analysis",
            "Formation detection and scoring",
            "Unit-based tactical analysis",
            "Professional coaching feedback",
            "Team philosophy customization",
            "ML dataset export"
        ]
    }


# ============== Player Identity Management ==============

@app.post("/api/player/label")
async def label_player(request: PlayerLabelRequest):
    """
    Manually label a player with jersey number and/or positional role.
    This creates a persistent player identity that survives track_id changes.
    """
    player_identity = player_identity_db.manual_label_player(
        video_id=request.video_id,
        track_id=request.track_id,
        jersey_number=request.jersey_number,
        positional_role=request.positional_role,
        team=request.team,
        frame_number=request.frame_number
    )

    return {
        "success": True,
        "player_identity": player_identity.model_dump()
    }


@app.post("/api/player/set-formation")
async def set_formation(video_id: str, team: str, formation: str):
    """
    Set formation for a team to enable automatic positional role assignment.

    Args:
        video_id: Match/video ID
        team: 'home' or 'away'
        formation: Formation name like '4-4-2', '4-3-3', etc.
    """
    # Store formation for this team
    player_identity_db.set_formation(f"{video_id}_{team}", formation)

    return {
        "success": True,
        "video_id": video_id,
        "team": team,
        "formation": formation,
        "message": f"Formation {formation} set for {team} team. Players will now be assigned positional roles."
    }


@app.get("/api/player/identities/{video_id}")
async def get_player_identities(video_id: str, team: Optional[str] = None):
    """
    Get all player identities for a match.

    Args:
        video_id: Match/video ID
        team: Optional filter by team ('home' or 'away')
    """
    all_identities = player_identity_db.get_all_identities(video_id)

    # Filter by team if specified
    if team:
        all_identities = [p for p in all_identities if p.team == team]

    # Sort by positional role
    role_order = ['GK', 'LB', 'CB_L', 'CB_R', 'RB', 'LM', 'CM_L', 'CM_R', 'RM', 'CAM', 'LW', 'ST_L', 'ST', 'ST_R', 'RW']

    def role_sort_key(p: PlayerIdentity):
        role = p.positional_role or 'ZZZ'  # Put unknown roles at end
        try:
            return role_order.index(role)
        except ValueError:
            return 999

    all_identities.sort(key=role_sort_key)

    return {
        "video_id": video_id,
        "team": team,
        "total_players": len(all_identities),
        "players": [p.model_dump() for p in all_identities]
    }


@app.get("/api/player/identity/{video_id}/{track_id}")
async def get_player_by_track_id(video_id: str, track_id: int):
    """
    Get player identity by track_id.
    """
    player_identity = player_identity_db.get_identity_by_track_id(video_id, track_id)

    if not player_identity:
        raise HTTPException(status_code=404, detail=f"No player found with track_id {track_id}")

    return {
        "success": True,
        "player_identity": player_identity.model_dump()
    }


@app.get("/api/player/timeline/{video_id}/{player_id}")
async def get_player_timeline(video_id: str, player_id: str):
    """
    Get timeline of all frames where a specific player appears.
    Useful for creating individual player highlight clips.

    Returns:
        List of frame numbers where the player was detected
    """
    # Find the player identity
    all_identities = player_identity_db.get_all_identities(video_id)
    player_identity = None
    for identity in all_identities:
        if identity.player_id == player_id:
            player_identity = identity
            break

    if not player_identity:
        raise HTTPException(status_code=404, detail=f"Player {player_id} not found")

    # Load detection file to find all frames with this player
    json_path = settings.FRAMES_DIR / f"{video_id}_detections.json"
    if not json_path.exists():
        raise HTTPException(status_code=404, detail=f"Detection file not found for video {video_id}")

    import json
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Find all frames where any of this player's track_ids appear
    frames_with_player = []
    track_ids = set(player_identity.track_id_history)

    for det in data.get('detections', []):
        frame_number = det.get('frame_number', 0)
        for player in det.get('players', []):
            if player.get('track_id') in track_ids:
                frames_with_player.append({
                    'frame_number': frame_number,
                    'timestamp_ms': det.get('timestamp_ms', 0),
                    'track_id': player.get('track_id'),
                    'bbox': player.get('bbox')
                })
                break  # Only add frame once

    return {
        "player_id": player_id,
        "jersey_number": player_identity.jersey_number,
        "positional_role": player_identity.positional_role,
        "team": player_identity.team,
        "total_frames": len(frames_with_player),
        "first_frame": player_identity.first_seen_frame,
        "last_frame": player_identity.last_seen_frame,
        "frames": frames_with_player
    }


# ============== Startup/Shutdown Events ==============

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    print(f"Starting {settings.APP_NAME}...")
    print(f"Cloud inference: {'enabled' if settings.CLOUD_INFERENCE_ENABLED else 'disabled'}")
    print(f"GPU available: {settings.USE_GPU}")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    print("Shutting down...")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
