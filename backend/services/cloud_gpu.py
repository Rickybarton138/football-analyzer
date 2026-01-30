"""
Cloud GPU Processing Service

Handles video analysis offloading to cloud GPU providers:
- RunPod (recommended for cost/performance)
- Lambda Labs
- Custom inference endpoint
"""
import aiohttp
import asyncio
import base64
import json
from typing import Optional, Dict, List, Any
from dataclasses import dataclass
from pathlib import Path
import cv2
import numpy as np

from config import settings


@dataclass
class CloudGPUConfig:
    """Configuration for cloud GPU processing."""
    provider: str  # 'runpod', 'lambda', 'custom'
    api_key: str
    endpoint_url: Optional[str] = None
    model_id: str = "yolov8x"  # Model to use for detection
    gpu_type: str = "RTX 4090"  # Preferred GPU type


@dataclass
class FrameResult:
    """Result from processing a single frame."""
    frame_number: int
    detections: List[Dict]
    tracks: List[Dict]
    ball_position: Optional[Dict]
    processing_time_ms: float


class CloudGPUService:
    """Service for cloud GPU video processing."""

    def __init__(self):
        self.config: Optional[CloudGPUConfig] = None
        self.session: Optional[aiohttp.ClientSession] = None
        self.is_connected = False
        self.processing_status = {
            'is_processing': False,
            'video_path': None,
            'total_frames': 0,
            'processed_frames': 0,
            'current_fps': 0,
            'estimated_remaining': None,
            'status': 'idle'
        }

    async def configure(self, provider: str, api_key: str, endpoint_url: Optional[str] = None):
        """Configure cloud GPU connection."""
        self.config = CloudGPUConfig(
            provider=provider,
            api_key=api_key,
            endpoint_url=endpoint_url
        )

        # Set default endpoints based on provider
        if provider == 'runpod' and not endpoint_url:
            self.config.endpoint_url = "https://api.runpod.ai/v2"
        elif provider == 'lambda' and not endpoint_url:
            self.config.endpoint_url = "https://cloud.lambdalabs.com/api/v1"

        self.session = aiohttp.ClientSession(
            headers={
                'Authorization': f'Bearer {api_key}',
                'Content-Type': 'application/json'
            }
        )

        # Test connection
        try:
            await self._test_connection()
            self.is_connected = True
            return {"status": "connected", "provider": provider}
        except Exception as e:
            self.is_connected = False
            return {"status": "error", "message": str(e)}

    async def _test_connection(self):
        """Test the cloud GPU connection."""
        if self.config.provider == 'runpod':
            # Test with RunPod user endpoint to verify API key
            async with self.session.get("https://api.runpod.io/graphql?query={myself{id}}") as resp:
                if resp.status != 200:
                    raise Exception(f"RunPod connection failed: {resp.status}")
                data = await resp.json()
                if 'errors' in data:
                    raise Exception(f"RunPod API key invalid: {data['errors']}")
        elif self.config.provider == 'lambda':
            async with self.session.get(f"{self.config.endpoint_url}/instances") as resp:
                if resp.status != 200:
                    raise Exception(f"Lambda Labs connection failed: {resp.status}")

    async def process_video(
        self,
        video_path: str,
        fps: int = 15,
        callback: Optional[callable] = None
    ) -> Dict:
        """
        Process a video file using cloud GPU.

        Args:
            video_path: Path to the video file
            fps: Frames per second to analyze
            callback: Optional callback for progress updates
        """
        if not self.is_connected:
            return {"error": "Not connected to cloud GPU. Call configure() first."}

        self.processing_status['is_processing'] = True
        self.processing_status['video_path'] = video_path
        self.processing_status['status'] = 'starting'

        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {"error": f"Could not open video: {video_path}"}

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / video_fps

        # Calculate frame skip to achieve target FPS
        frame_skip = max(1, int(video_fps / fps))
        frames_to_process = total_frames // frame_skip

        self.processing_status['total_frames'] = frames_to_process
        self.processing_status['status'] = 'processing'

        results = []
        frame_count = 0
        processed_count = 0
        batch_size = 10  # Process 10 frames at a time
        batch_frames = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_skip == 0:
                # Encode frame for transmission
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                frame_b64 = base64.b64encode(buffer).decode('utf-8')
                batch_frames.append({
                    'frame_number': frame_count,
                    'data': frame_b64
                })

                # Process batch when full
                if len(batch_frames) >= batch_size:
                    batch_results = await self._process_batch(batch_frames)
                    results.extend(batch_results)
                    processed_count += len(batch_frames)
                    batch_frames = []

                    self.processing_status['processed_frames'] = processed_count
                    self.processing_status['current_fps'] = fps

                    if callback:
                        await callback({
                            'processed': processed_count,
                            'total': frames_to_process,
                            'progress': processed_count / frames_to_process * 100
                        })

            frame_count += 1

        # Process remaining frames
        if batch_frames:
            batch_results = await self._process_batch(batch_frames)
            results.extend(batch_results)

        cap.release()

        self.processing_status['is_processing'] = False
        self.processing_status['status'] = 'complete'

        return {
            'video_path': video_path,
            'total_frames': total_frames,
            'frames_analyzed': len(results),
            'duration_seconds': duration,
            'results': results
        }

    async def _process_batch(self, frames: List[Dict]) -> List[FrameResult]:
        """Process a batch of frames on cloud GPU."""
        if self.config.provider == 'runpod':
            return await self._process_runpod(frames)
        elif self.config.provider == 'lambda':
            return await self._process_lambda(frames)
        else:
            return await self._process_custom(frames)

    async def _process_runpod(self, frames: List[Dict]) -> List[FrameResult]:
        """Process frames using RunPod serverless."""
        payload = {
            'input': {
                'frames': frames,
                'model': self.config.model_id,
                'detect_ball': True,
                'track_players': True
            }
        }

        try:
            async with self.session.post(
                f"{self.config.endpoint_url}/{self.config.model_id}/run",
                json=payload
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return self._parse_results(data.get('output', []))
                else:
                    print(f"RunPod error: {resp.status}")
                    return []
        except Exception as e:
            print(f"RunPod processing error: {e}")
            return []

    async def _process_lambda(self, frames: List[Dict]) -> List[FrameResult]:
        """Process frames using Lambda Labs."""
        # Similar to RunPod but with Lambda's API format
        payload = {
            'frames': frames,
            'model': self.config.model_id
        }

        try:
            async with self.session.post(
                f"{self.config.endpoint_url}/inference",
                json=payload
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return self._parse_results(data.get('results', []))
                else:
                    return []
        except Exception as e:
            print(f"Lambda processing error: {e}")
            return []

    async def _process_custom(self, frames: List[Dict]) -> List[FrameResult]:
        """Process frames using custom endpoint."""
        payload = {'frames': frames}

        try:
            async with self.session.post(
                self.config.endpoint_url,
                json=payload
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return self._parse_results(data)
                else:
                    return []
        except Exception as e:
            print(f"Custom endpoint error: {e}")
            return []

    def _parse_results(self, raw_results: List[Dict]) -> List[FrameResult]:
        """Parse raw API results into FrameResult objects."""
        results = []
        for r in raw_results:
            results.append(FrameResult(
                frame_number=r.get('frame_number', 0),
                detections=r.get('detections', []),
                tracks=r.get('tracks', []),
                ball_position=r.get('ball'),
                processing_time_ms=r.get('time_ms', 0)
            ))
        return results

    def get_status(self) -> Dict:
        """Get current processing status."""
        return self.processing_status.copy()

    async def close(self):
        """Close the cloud connection."""
        if self.session:
            await self.session.close()
            self.session = None
        self.is_connected = False


# Global service instance
cloud_gpu_service = CloudGPUService()


# Deployment instructions
RUNPOD_SETUP_INSTRUCTIONS = """
# RunPod Cloud GPU Setup

1. Create a RunPod account at https://runpod.io
2. Add credits ($10 minimum recommended for testing)
3. Go to Settings > API Keys > Create API Key
4. Copy your API key

## Deploy the Inference Endpoint

Option A: Use RunPod Serverless (Recommended)
- Go to Serverless > Deploy
- Select "Custom" template
- Use Docker image: ultralytics/ultralytics:latest-python
- Set GPU: RTX 4090 (fastest) or RTX 3090 (cheaper)
- Copy the endpoint ID

Option B: Use a Pre-built YOLOv8 endpoint
- Search for "yolov8" in the marketplace
- Deploy and copy the endpoint URL

## Configure in the app
POST /api/cloud-gpu/configure
{
    "provider": "runpod",
    "api_key": "YOUR_API_KEY",
    "endpoint_id": "YOUR_ENDPOINT_ID"
}
"""

LAMBDA_SETUP_INSTRUCTIONS = """
# Lambda Labs Cloud GPU Setup

1. Create account at https://lambdalabs.com/cloud
2. Add payment method
3. Go to API > Create API Key
4. Copy your API key

## Launch an Instance
- Select GPU: A100 (fastest) or A10 (cheaper)
- Launch instance and note the IP address

## Configure in the app
POST /api/cloud-gpu/configure
{
    "provider": "lambda",
    "api_key": "YOUR_API_KEY",
    "endpoint_url": "http://INSTANCE_IP:8000"
}
"""
