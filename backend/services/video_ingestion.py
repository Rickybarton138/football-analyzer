"""
Video Ingestion Service

Handles video file uploads, RTMP stream ingestion from VEO Cam 3,
and frame extraction for processing.
"""
import asyncio
import subprocess
import tempfile
from pathlib import Path
from typing import AsyncGenerator, Dict, Optional, List
import cv2
import numpy as np
import aiofiles

from config import settings


class VideoIngestionService:
    """Service for video ingestion and frame extraction."""

    def __init__(self):
        self.active_streams: Dict[str, subprocess.Popen] = {}
        self._ffmpeg_path = self._find_ffmpeg()

    def _find_ffmpeg(self) -> str:
        """Find FFmpeg executable path."""
        # Try common locations
        common_paths = [
            "ffmpeg",  # In PATH
            "/usr/bin/ffmpeg",
            "/usr/local/bin/ffmpeg",
            "C:\\ffmpeg\\bin\\ffmpeg.exe",
        ]
        for path in common_paths:
            try:
                result = subprocess.run(
                    [path, "-version"],
                    capture_output=True,
                    timeout=5
                )
                if result.returncode == 0:
                    return path
            except (subprocess.SubprocessError, FileNotFoundError):
                continue
        return "ffmpeg"  # Default, hope it's in PATH

    async def get_video_metadata(self, video_path: Path) -> Dict:
        """
        Extract metadata from video file using OpenCV.

        Args:
            video_path: Path to video file

        Returns:
            Dictionary with video metadata
        """
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration_ms = int((total_frames / fps) * 1000) if fps > 0 else 0

        cap.release()

        return {
            "fps": fps,
            "total_frames": total_frames,
            "width": width,
            "height": height,
            "duration_ms": duration_ms
        }

    async def extract_frames(
        self,
        video_path: Path,
        target_fps: int = 10,
        start_time_ms: int = 0,
        end_time_ms: Optional[int] = None
    ) -> AsyncGenerator[Dict, None]:
        """
        Extract frames from video at specified FPS.

        Args:
            video_path: Path to video file
            target_fps: Target frames per second to extract
            start_time_ms: Start time in milliseconds
            end_time_ms: End time in milliseconds (None for entire video)

        Yields:
            Dictionary with frame data:
                - frame: numpy array (BGR)
                - frame_number: int
                - timestamp_ms: int
        """
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")

        source_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_skip = max(1, int(source_fps / target_fps))

        # Set start position
        if start_time_ms > 0:
            cap.set(cv2.CAP_PROP_POS_MSEC, start_time_ms)

        frame_count = 0
        processed_count = 0

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            current_ms = cap.get(cv2.CAP_PROP_POS_MSEC)

            # Check end time
            if end_time_ms and current_ms > end_time_ms:
                break

            # Skip frames to match target FPS
            if frame_count % frame_skip == 0:
                yield {
                    "frame": frame,
                    "frame_number": processed_count,
                    "timestamp_ms": int(current_ms),
                    "original_frame_number": frame_count
                }
                processed_count += 1

                # Yield control to event loop periodically
                if processed_count % 10 == 0:
                    await asyncio.sleep(0)

            frame_count += 1

        cap.release()

    async def extract_frame_at_time(
        self,
        video_path: Path,
        timestamp_ms: int
    ) -> Optional[np.ndarray]:
        """
        Extract a single frame at specific timestamp.

        Args:
            video_path: Path to video file
            timestamp_ms: Timestamp in milliseconds

        Returns:
            Frame as numpy array or None if failed
        """
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            return None

        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp_ms)
        ret, frame = cap.read()
        cap.release()

        return frame if ret else None

    async def save_frame(
        self,
        frame: np.ndarray,
        output_path: Path,
        quality: int = 95
    ) -> bool:
        """
        Save frame to disk as JPEG.

        Args:
            frame: Frame as numpy array
            output_path: Path to save frame
            quality: JPEG quality (0-100)

        Returns:
            True if successful
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        params = [cv2.IMWRITE_JPEG_QUALITY, quality]
        return cv2.imwrite(str(output_path), frame, params)

    async def start_rtmp_stream(
        self,
        stream_url: str,
        match_id: str,
        output_format: str = "frames"
    ) -> bool:
        """
        Start listening to RTMP stream from VEO Cam 3.

        Args:
            stream_url: RTMP stream URL (e.g., rtmp://...)
            match_id: Match identifier
            output_format: "frames" for individual frames, "hls" for HLS stream

        Returns:
            True if stream started successfully
        """
        if match_id in self.active_streams:
            return False  # Already streaming

        output_dir = settings.FRAMES_DIR / match_id
        output_dir.mkdir(parents=True, exist_ok=True)

        if output_format == "frames":
            # Extract frames at configured FPS
            cmd = [
                self._ffmpeg_path,
                "-i", stream_url,
                "-vf", f"fps={settings.LIVE_FPS}",
                "-frame_pts", "1",
                str(output_dir / "frame_%06d.jpg"),
                "-y"
            ]
        else:
            # Convert to HLS for web playback
            cmd = [
                self._ffmpeg_path,
                "-i", stream_url,
                "-c:v", "libx264",
                "-preset", "ultrafast",
                "-tune", "zerolatency",
                "-hls_time", "2",
                "-hls_list_size", "5",
                "-hls_flags", "delete_segments",
                str(output_dir / "stream.m3u8"),
                "-y"
            ]

        # Start FFmpeg process
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        self.active_streams[match_id] = process
        return True

    async def stop_rtmp_stream(self, match_id: str) -> bool:
        """
        Stop RTMP stream listener.

        Args:
            match_id: Match identifier

        Returns:
            True if stopped successfully
        """
        if match_id not in self.active_streams:
            return False

        process = self.active_streams[match_id]
        process.terminate()

        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()

        del self.active_streams[match_id]
        return True

    async def get_stream_frame(self, match_id: str) -> Optional[np.ndarray]:
        """
        Get the latest frame from an active stream.

        Args:
            match_id: Match identifier

        Returns:
            Latest frame as numpy array or None
        """
        if match_id not in self.active_streams:
            return None

        frame_dir = settings.FRAMES_DIR / match_id
        frames = sorted(frame_dir.glob("frame_*.jpg"))

        if not frames:
            return None

        # Get latest frame
        latest_frame = frames[-1]
        return cv2.imread(str(latest_frame))

    async def create_video_clip(
        self,
        video_path: Path,
        output_path: Path,
        start_ms: int,
        end_ms: int,
        with_audio: bool = True
    ) -> bool:
        """
        Extract a clip from video file.

        Args:
            video_path: Source video path
            output_path: Output clip path
            start_ms: Start time in milliseconds
            end_ms: End time in milliseconds
            with_audio: Include audio in clip

        Returns:
            True if successful
        """
        start_sec = start_ms / 1000
        duration_sec = (end_ms - start_ms) / 1000

        cmd = [
            self._ffmpeg_path,
            "-i", str(video_path),
            "-ss", str(start_sec),
            "-t", str(duration_sec),
            "-c:v", "libx264",
            "-preset", "fast",
        ]

        if with_audio:
            cmd.extend(["-c:a", "aac"])
        else:
            cmd.extend(["-an"])

        cmd.extend([str(output_path), "-y"])

        result = subprocess.run(cmd, capture_output=True)
        return result.returncode == 0

    def encode_frame_base64(self, frame: np.ndarray) -> str:
        """
        Encode frame as base64 string for transmission.

        Args:
            frame: Frame as numpy array

        Returns:
            Base64 encoded JPEG string
        """
        import base64
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return base64.b64encode(buffer).decode('utf-8')

    def decode_frame_base64(self, data: str) -> np.ndarray:
        """
        Decode base64 string to frame.

        Args:
            data: Base64 encoded image string

        Returns:
            Frame as numpy array
        """
        import base64
        buffer = base64.b64decode(data)
        nparr = np.frombuffer(buffer, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
