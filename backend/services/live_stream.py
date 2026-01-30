"""
Live Stream Processing Service

Handles real-time video stream input from VEO cameras (RTSP/HLS)
and provides low-latency frame processing for live game management.
"""
import asyncio
import cv2
import numpy as np
from typing import Optional, Dict, List, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from collections import deque
import threading
import time


class StreamType(str, Enum):
    RTSP = "rtsp"
    HLS = "hls"
    FILE = "file"  # For testing with recorded video


class StreamStatus(str, Enum):
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    ERROR = "error"


@dataclass
class LiveStreamConfig:
    """Configuration for live stream input."""
    stream_url: str
    stream_type: StreamType = StreamType.RTSP
    target_fps: float = 2.0  # Processing FPS
    buffer_seconds: float = 2.0
    reconnect_attempts: int = 5
    reconnect_delay_ms: int = 2000
    timeout_ms: int = 10000


@dataclass
class StreamMetrics:
    """Real-time metrics for the stream."""
    actual_fps: float = 0.0
    latency_ms: float = 0.0
    frames_processed: int = 0
    frames_dropped: int = 0
    reconnect_count: int = 0
    last_frame_time: float = 0.0
    stream_start_time: float = 0.0
    errors: List[str] = field(default_factory=list)


@dataclass
class FrameData:
    """Data for a single processed frame."""
    frame: np.ndarray
    frame_number: int
    timestamp_ms: int
    capture_time: float
    width: int
    height: int


class LiveStreamService:
    """
    Service for handling live video streams from VEO cameras.

    Supports RTSP and HLS streams with automatic reconnection,
    frame buffering, and rate limiting for consistent processing.
    """

    def __init__(self):
        self.config: Optional[LiveStreamConfig] = None
        self.status = StreamStatus.DISCONNECTED
        self.metrics = StreamMetrics()

        # Video capture
        self._capture: Optional[cv2.VideoCapture] = None
        self._is_running = False
        self._capture_thread: Optional[threading.Thread] = None

        # Frame buffer (thread-safe)
        self._frame_buffer: deque = deque(maxlen=30)
        self._buffer_lock = threading.Lock()

        # Callbacks for frame processing
        self._frame_callbacks: List[Callable[[FrameData], Any]] = []

        # Rate limiting
        self._last_process_time = 0.0
        self._frame_interval = 0.5  # Default 2 FPS

        # Stream metadata
        self.source_fps: float = 30.0
        self.frame_width: int = 1920
        self.frame_height: int = 1080

    def configure(self, config: LiveStreamConfig):
        """Configure the stream settings."""
        self.config = config
        self._frame_interval = 1.0 / config.target_fps

    def add_frame_callback(self, callback: Callable[[FrameData], Any]):
        """Add a callback to be called for each processed frame."""
        self._frame_callbacks.append(callback)

    def remove_frame_callback(self, callback: Callable[[FrameData], Any]):
        """Remove a frame callback."""
        if callback in self._frame_callbacks:
            self._frame_callbacks.remove(callback)

    async def start(self) -> bool:
        """Start capturing from the configured stream."""
        if not self.config:
            print("[LIVE_STREAM] No configuration set")
            return False

        if self._is_running:
            print("[LIVE_STREAM] Already running")
            return True

        self.status = StreamStatus.CONNECTING
        print(f"[LIVE_STREAM] Connecting to {self.config.stream_url}")

        # Connect to stream
        success = await self._connect()

        if success:
            self._is_running = True
            self.metrics.stream_start_time = time.time()

            # Start capture thread
            self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
            self._capture_thread.start()

            self.status = StreamStatus.CONNECTED
            print("[LIVE_STREAM] Connected and capturing")
            return True
        else:
            self.status = StreamStatus.ERROR
            return False

    async def stop(self):
        """Stop capturing."""
        self._is_running = False

        if self._capture_thread:
            self._capture_thread.join(timeout=2.0)
            self._capture_thread = None

        if self._capture:
            self._capture.release()
            self._capture = None

        self.status = StreamStatus.DISCONNECTED
        print("[LIVE_STREAM] Stopped")

    async def _connect(self) -> bool:
        """Connect to the video stream."""
        try:
            if self.config.stream_type == StreamType.RTSP:
                # RTSP with TCP for reliability
                self._capture = cv2.VideoCapture(
                    self.config.stream_url,
                    cv2.CAP_FFMPEG
                )
                # Set RTSP over TCP
                self._capture.set(cv2.CAP_PROP_BUFFERSIZE, 3)

            elif self.config.stream_type == StreamType.HLS:
                # HLS stream
                self._capture = cv2.VideoCapture(self.config.stream_url)

            else:  # FILE - for testing
                self._capture = cv2.VideoCapture(self.config.stream_url)

            if not self._capture.isOpened():
                print(f"[LIVE_STREAM] Failed to open stream: {self.config.stream_url}")
                return False

            # Get stream properties
            self.source_fps = self._capture.get(cv2.CAP_PROP_FPS) or 30.0
            self.frame_width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
            self.frame_height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080

            print(f"[LIVE_STREAM] Stream properties: {self.frame_width}x{self.frame_height} @ {self.source_fps} FPS")
            return True

        except Exception as e:
            print(f"[LIVE_STREAM] Connection error: {e}")
            self.metrics.errors.append(f"Connection: {str(e)}")
            return False

    def _capture_loop(self):
        """Background thread for continuous frame capture."""
        frame_count = 0
        last_fps_time = time.time()
        fps_frame_count = 0

        while self._is_running:
            if not self._capture or not self._capture.isOpened():
                # Try to reconnect
                self._handle_disconnect()
                continue

            ret, frame = self._capture.read()

            if not ret:
                self._handle_disconnect()
                continue

            capture_time = time.time()
            frame_count += 1
            fps_frame_count += 1

            # Calculate actual FPS every second
            if capture_time - last_fps_time >= 1.0:
                self.metrics.actual_fps = fps_frame_count / (capture_time - last_fps_time)
                fps_frame_count = 0
                last_fps_time = capture_time

            # Rate limiting - only process at target FPS
            if capture_time - self._last_process_time < self._frame_interval:
                continue

            self._last_process_time = capture_time

            # Create frame data
            timestamp_ms = int((capture_time - self.metrics.stream_start_time) * 1000)

            frame_data = FrameData(
                frame=frame,
                frame_number=self.metrics.frames_processed,
                timestamp_ms=timestamp_ms,
                capture_time=capture_time,
                width=self.frame_width,
                height=self.frame_height
            )

            # Add to buffer
            with self._buffer_lock:
                self._frame_buffer.append(frame_data)

            self.metrics.frames_processed += 1
            self.metrics.last_frame_time = capture_time

            # Call callbacks
            for callback in self._frame_callbacks:
                try:
                    callback(frame_data)
                except Exception as e:
                    print(f"[LIVE_STREAM] Callback error: {e}")

    def _handle_disconnect(self):
        """Handle stream disconnection with reconnection logic."""
        if self.status == StreamStatus.RECONNECTING:
            return

        self.status = StreamStatus.RECONNECTING
        self.metrics.reconnect_count += 1

        print(f"[LIVE_STREAM] Disconnected, attempting reconnect ({self.metrics.reconnect_count})")

        # Release current capture
        if self._capture:
            self._capture.release()
            self._capture = None

        # Wait before reconnecting
        time.sleep(self.config.reconnect_delay_ms / 1000)

        # Try to reconnect
        for attempt in range(self.config.reconnect_attempts):
            try:
                if self.config.stream_type == StreamType.RTSP:
                    self._capture = cv2.VideoCapture(self.config.stream_url, cv2.CAP_FFMPEG)
                else:
                    self._capture = cv2.VideoCapture(self.config.stream_url)

                if self._capture.isOpened():
                    self.status = StreamStatus.CONNECTED
                    print(f"[LIVE_STREAM] Reconnected on attempt {attempt + 1}")
                    return
            except Exception as e:
                print(f"[LIVE_STREAM] Reconnect attempt {attempt + 1} failed: {e}")

            time.sleep(self.config.reconnect_delay_ms / 1000)

        # All attempts failed
        self.status = StreamStatus.ERROR
        self._is_running = False
        print("[LIVE_STREAM] Failed to reconnect after all attempts")

    def get_latest_frame(self) -> Optional[FrameData]:
        """Get the most recent frame from the buffer."""
        with self._buffer_lock:
            if self._frame_buffer:
                return self._frame_buffer[-1]
        return None

    def get_frame_buffer(self) -> List[FrameData]:
        """Get all frames currently in the buffer."""
        with self._buffer_lock:
            return list(self._frame_buffer)

    def get_status(self) -> Dict:
        """Get current stream status and metrics."""
        return {
            "status": self.status.value,
            "stream_url": self.config.stream_url if self.config else None,
            "stream_type": self.config.stream_type.value if self.config else None,
            "is_running": self._is_running,
            "metrics": {
                "actual_fps": round(self.metrics.actual_fps, 1),
                "latency_ms": round(self.metrics.latency_ms, 0),
                "frames_processed": self.metrics.frames_processed,
                "frames_dropped": self.metrics.frames_dropped,
                "reconnect_count": self.metrics.reconnect_count,
                "uptime_seconds": time.time() - self.metrics.stream_start_time if self.metrics.stream_start_time else 0,
            },
            "stream_info": {
                "source_fps": self.source_fps,
                "width": self.frame_width,
                "height": self.frame_height,
                "target_fps": self.config.target_fps if self.config else None,
            }
        }


class LiveGameManager:
    """
    Manages a live game session with real-time analysis.

    Coordinates stream input, detection, tracking, event detection,
    and alert generation for live coaching assistance.
    """

    def __init__(self):
        self.stream_service = LiveStreamService()
        self.session_id: Optional[str] = None
        self.is_active = False

        # Live statistics
        self.live_stats = {
            "home_possession_frames": 0,
            "away_possession_frames": 0,
            "total_frames": 0,
            "home_score": 0,
            "away_score": 0,
            "match_time_ms": 0,
            "period": 1,
        }

        # Alert queue
        self.pending_alerts: deque = deque(maxlen=50)
        self.alert_history: List[Dict] = []

        # WebSocket connections for live updates
        self.websocket_connections: List[Any] = []

        # Detection services (will be injected)
        self.detection_service = None
        self.tracking_service = None
        self.event_service = None

    async def start_session(
        self,
        stream_url: str,
        stream_type: str = "rtsp",
        target_fps: float = 2.0
    ) -> Dict:
        """Start a new live game session."""
        import uuid

        self.session_id = str(uuid.uuid4())[:8]

        # Configure stream
        config = LiveStreamConfig(
            stream_url=stream_url,
            stream_type=StreamType(stream_type),
            target_fps=target_fps
        )
        self.stream_service.configure(config)

        # Add frame processing callback
        self.stream_service.add_frame_callback(self._process_live_frame)

        # Start stream
        success = await self.stream_service.start()

        if success:
            self.is_active = True
            return {
                "status": "success",
                "session_id": self.session_id,
                "websocket_url": f"/ws/live-coaching/{self.session_id}",
                "stream_info": self.stream_service.get_status()["stream_info"]
            }
        else:
            return {
                "status": "error",
                "message": "Failed to connect to stream"
            }

    async def stop_session(self):
        """Stop the current live session."""
        self.is_active = False
        await self.stream_service.stop()

        # Notify all WebSocket connections
        for ws in self.websocket_connections:
            try:
                await ws.send_json({"type": "session_ended"})
            except Exception:
                pass

        self.websocket_connections.clear()
        self.session_id = None

    def _process_live_frame(self, frame_data: FrameData):
        """Process a frame for live analysis (called from capture thread)."""
        # This runs in the capture thread, so we need to be careful
        # Queue work for the async event loop instead of blocking

        # Update match time
        self.live_stats["match_time_ms"] = frame_data.timestamp_ms
        self.live_stats["total_frames"] += 1

        # The actual detection/tracking will be done asynchronously
        # This callback just updates basic stats and triggers async processing

    async def process_frame_async(self, frame_data: FrameData) -> Dict:
        """
        Async frame processing with full detection pipeline.

        Called from the main event loop to process frames
        with detection, tracking, and event detection.
        """
        result = {
            "frame_number": frame_data.frame_number,
            "timestamp_ms": frame_data.timestamp_ms,
            "players": [],
            "ball": None,
            "events": [],
            "alerts": []
        }

        if not self.detection_service:
            return result

        try:
            # Run detection
            players = await self.detection_service.detect(frame_data.frame)

            # Run tracking
            if self.tracking_service:
                players = await self.tracking_service.update(players, frame_data.frame_number)

            result["players"] = [p.model_dump() for p in players]

            # Ball detection
            if hasattr(self.detection_service, 'detect_ball'):
                ball = await self.detection_service.detect_ball(frame_data.frame)
                if ball:
                    result["ball"] = ball.model_dump()

            # Event detection
            if self.event_service:
                events = await self.event_service.process_frame(
                    players=players,
                    ball=ball,
                    frame_number=frame_data.frame_number,
                    timestamp_ms=frame_data.timestamp_ms
                )
                result["events"] = [e.model_dump() if hasattr(e, 'model_dump') else e for e in events]

                # Generate alerts from events
                alerts = self._generate_alerts(players, ball, events, frame_data)
                result["alerts"] = alerts

            # Update possession stats
            self._update_live_stats(players, ball)

        except Exception as e:
            print(f"[LIVE_GAME] Frame processing error: {e}")

        return result

    def _generate_alerts(
        self,
        players: List,
        ball: Optional[Any],
        events: List,
        frame_data: FrameData
    ) -> List[Dict]:
        """Generate coaching alerts from current frame analysis."""
        alerts = []

        # Check for pressing opportunity
        pressing_alert = self._check_pressing_opportunity(players, ball)
        if pressing_alert:
            alerts.append(pressing_alert)

        # Check for defensive gap
        gap_alert = self._check_defensive_gap(players)
        if gap_alert:
            alerts.append(gap_alert)

        # Check for counter-attack opportunity
        for event in events:
            event_type = event.event_type if hasattr(event, 'event_type') else event.get('event_type')
            if event_type in ['tackle', 'interception']:
                counter_alert = self._check_counter_opportunity(players, ball, event)
                if counter_alert:
                    alerts.append(counter_alert)

        return alerts

    def _check_pressing_opportunity(self, players: List, ball: Optional[Any]) -> Optional[Dict]:
        """Check if there's a good pressing opportunity."""
        if not ball or not hasattr(ball, 'possessed_by') or ball.possessed_by is None:
            return None

        # Find ball carrier
        ball_carrier = next((p for p in players if p.track_id == ball.possessed_by), None)
        if not ball_carrier:
            return None

        # Count nearby defenders (opposite team)
        nearby_defenders = 0
        for player in players:
            if player.team != ball_carrier.team and player.team.value != "unknown":
                if player.pixel_position and ball_carrier.pixel_position:
                    dist = np.sqrt(
                        (player.pixel_position.x - ball_carrier.pixel_position.x)**2 +
                        (player.pixel_position.y - ball_carrier.pixel_position.y)**2
                    )
                    if dist < 100:  # Within pressing range
                        nearby_defenders += 1

        # Pressing opportunity if 2+ defenders nearby
        if nearby_defenders >= 2:
            return {
                "alert_id": f"press_{int(time.time()*1000)}",
                "priority": "immediate",
                "category": "pressing",
                "title": "PRESSING OPPORTUNITY",
                "message": f"Press #{ball_carrier.track_id} - {nearby_defenders} defenders nearby",
                "action": "Trigger high press",
                "highlight_players": [ball_carrier.track_id],
                "duration_seconds": 5,
                "play_sound": True
            }

        return None

    def _check_defensive_gap(self, players: List) -> Optional[Dict]:
        """Check for gaps in the defensive line."""
        # Simplified gap detection - look for large spaces between defenders
        # Full implementation would analyze the back line specifically
        return None

    def _check_counter_opportunity(
        self,
        players: List,
        ball: Optional[Any],
        event: Any
    ) -> Optional[Dict]:
        """Check for counter-attack opportunity after turnover."""
        # Full implementation would analyze space ahead
        return None

    def _update_live_stats(self, players: List, ball: Optional[Any]):
        """Update live match statistics."""
        if ball and hasattr(ball, 'possessed_by') and ball.possessed_by is not None:
            # Find who has possession
            possessor = next((p for p in players if p.track_id == ball.possessed_by), None)
            if possessor and possessor.team:
                if possessor.team.value == "home":
                    self.live_stats["home_possession_frames"] += 1
                elif possessor.team.value == "away":
                    self.live_stats["away_possession_frames"] += 1

    def get_live_stats(self) -> Dict:
        """Get current live statistics."""
        total = self.live_stats["home_possession_frames"] + self.live_stats["away_possession_frames"]
        home_poss = (self.live_stats["home_possession_frames"] / total * 100) if total > 0 else 50
        away_poss = (self.live_stats["away_possession_frames"] / total * 100) if total > 0 else 50

        return {
            "possession": {
                "home": round(home_poss, 1),
                "away": round(away_poss, 1)
            },
            "score": {
                "home": self.live_stats["home_score"],
                "away": self.live_stats["away_score"]
            },
            "match_time_ms": self.live_stats["match_time_ms"],
            "period": self.live_stats["period"],
            "total_frames_analyzed": self.live_stats["total_frames"]
        }

    def update_score(self, home: int, away: int):
        """Manually update the score."""
        self.live_stats["home_score"] = home
        self.live_stats["away_score"] = away

    def set_period(self, period: int):
        """Set the current match period (1 or 2)."""
        self.live_stats["period"] = period


# Global instances
live_stream_service = LiveStreamService()
live_game_manager = LiveGameManager()
