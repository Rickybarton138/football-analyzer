"""Mux video hosting service — upload, transcode, clip, stream."""

import httpx
from app.core.config import get_settings


class MuxService:
    def __init__(self):
        s = get_settings()
        self.token_id = s.mux_token_id
        self.token_secret = s.mux_token_secret
        self.base_url = "https://api.mux.com"

    @property
    def auth(self) -> tuple[str, str]:
        return (self.token_id, self.token_secret)

    async def create_upload(self) -> dict:
        """Create a direct upload URL for the frontend."""
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{self.base_url}/video/v1/uploads",
                json={
                    "new_asset_settings": {
                        "playback_policy": ["public"],
                        "encoding_tier": "baseline",
                    },
                    "cors_origin": "*",
                },
                auth=self.auth,
            )
            resp.raise_for_status()
            data = resp.json()["data"]
            return {
                "upload_id": data["id"],
                "upload_url": data["url"],
                "asset_id": data.get("asset_id"),
            }

    async def create_asset_from_url(self, video_url: str) -> dict:
        """Create a Mux asset directly from a video URL (no upload needed)."""
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                f"{self.base_url}/video/v1/assets",
                json={
                    "input": [{"url": video_url}],
                    "playback_policy": ["public"],
                    "encoding_tier": "baseline",
                },
                auth=self.auth,
            )
            resp.raise_for_status()
            asset = resp.json()["data"]
            playback_ids = asset.get("playback_ids", [])
            return {
                "asset_id": asset["id"],
                "status": asset["status"],
                "playback_id": playback_ids[0]["id"] if playback_ids else None,
                "duration": asset.get("duration"),
            }

    async def get_asset(self, asset_id: str) -> dict:
        """Get asset details including playback IDs and status."""
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{self.base_url}/video/v1/assets/{asset_id}",
                auth=self.auth,
            )
            resp.raise_for_status()
            asset = resp.json()["data"]
            playback_ids = asset.get("playback_ids", [])
            playback_id = playback_ids[0]["id"] if playback_ids else None
            # Get MP4 download URL for TwelveLabs indexing
            static_renditions = asset.get("static_renditions", {})
            mp4_url = None
            if static_renditions.get("status") == "ready":
                files = static_renditions.get("files", [])
                if files:
                    # Pick highest quality MP4
                    best = sorted(files, key=lambda f: f.get("width", 0), reverse=True)[0]
                    mp4_url = f"https://stream.mux.com/{playback_id}/{best['name']}"

            return {
                "asset_id": asset["id"],
                "status": asset["status"],
                "duration": asset.get("duration"),
                "playback_id": playback_id,
                "stream_url": f"https://stream.mux.com/{playback_id}.m3u8" if playback_id else None,
                "mp4_url": mp4_url,
                "thumbnail_url": f"https://image.mux.com/{playback_id}/thumbnail.jpg" if playback_id else None,
            }

    async def get_asset_from_upload(self, upload_id: str) -> str | None:
        """Get asset ID from an upload."""
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{self.base_url}/video/v1/uploads/{upload_id}",
                auth=self.auth,
            )
            resp.raise_for_status()
            return resp.json()["data"].get("asset_id")

    def get_clip_url(self, playback_id: str, start: float, end: float) -> str:
        """Generate an HLS clip URL for a specific time range."""
        return (
            f"https://stream.mux.com/{playback_id}.m3u8"
            f"?asset_start_time={start}&asset_end_time={end}"
        )

    def get_thumbnail_at(self, playback_id: str, time: float) -> str:
        """Get thumbnail at a specific timestamp."""
        return f"https://image.mux.com/{playback_id}/thumbnail.jpg?time={time}"

    def get_gif(self, playback_id: str, start: float, end: float) -> str:
        """Get animated GIF for a time range."""
        return (
            f"https://image.mux.com/{playback_id}/animated.gif"
            f"?start={start}&end={end}&width=320&fps=10"
        )
