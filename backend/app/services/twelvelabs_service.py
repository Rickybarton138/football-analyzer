"""TwelveLabs video intelligence service — index, analyse, search."""

import httpx
from app.core.config import get_settings


class TwelveLabsService:
    def __init__(self):
        s = get_settings()
        self.api_key = s.twelvelabs_api_key
        self.index_id = s.twelvelabs_index_id
        self.base_url = "https://api.twelvelabs.io/v1.3"

    @property
    def headers(self) -> dict:
        return {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
        }

    async def index_video_from_file(self, file_path: str, title: str = "", content_type: str = "video/mp4") -> dict:
        """Upload a local video file for indexing. Returns task info."""
        import os
        headers = {"x-api-key": self.api_key}
        async with httpx.AsyncClient(timeout=600) as client:
            with open(file_path, "rb") as f:
                resp = await client.post(
                    f"{self.base_url}/tasks",
                    data={"index_id": self.index_id},
                    files={"video_file": (os.path.basename(file_path), f, content_type)},
                    headers=headers,
                )
            resp.raise_for_status()
            data = resp.json()
            return {
                "task_id": data["_id"],
                "video_id": data.get("video_id"),
                "status": data.get("status", "pending"),
            }

    async def index_video_from_url(self, video_url: str, title: str = "") -> dict:
        """Submit a video URL for indexing via multipart form. Returns task info."""
        headers = {"x-api-key": self.api_key}
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(
                f"{self.base_url}/tasks",
                data={"index_id": self.index_id, "video_url": video_url},
                files={"_": ("", b"")},  # forces multipart/form-data encoding
                headers=headers,
            )
            resp.raise_for_status()
            data = resp.json()
            return {
                "task_id": data["_id"],
                "video_id": data.get("video_id"),
                "status": data.get("status", "pending"),
            }

    async def index_video(self, video_url: str, title: str = "") -> dict:
        """Submit a video URL for indexing. Returns task info."""
        headers = {"x-api-key": self.api_key}
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(
                f"{self.base_url}/tasks",
                data={"index_id": self.index_id, "video_url": video_url},
                headers=headers,
            )
            resp.raise_for_status()
            data = resp.json()
            return {
                "task_id": data["_id"],
                "video_id": data.get("video_id"),
                "status": data.get("status", "pending"),
            }

    async def get_task_status(self, task_id: str) -> dict:
        """Check indexing task status."""
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{self.base_url}/tasks/{task_id}",
                headers=self.headers,
            )
            resp.raise_for_status()
            data = resp.json()
            return {
                "task_id": data["_id"],
                "video_id": data.get("video_id"),
                "status": data["status"],
            }

    async def analyse_video(self, video_id: str, prompt: str) -> str:
        """Analyse a video with a custom prompt. Returns text analysis."""
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(
                f"{self.base_url}/generate",
                json={
                    "video_id": video_id,
                    "prompt": prompt,
                },
                headers=self.headers,
            )
            resp.raise_for_status()
            return resp.json().get("data", "")

    async def get_highlights(self, video_id: str) -> list[dict]:
        """Get auto-generated highlights with timestamps."""
        prompt = (
            "Identify the key moments in this football match. For each moment, provide: "
            "1) What happened (goal, shot, tackle, key pass, defensive error, counter-attack) "
            "2) The approximate timestamp "
            "3) Why it matters tactically. "
            "Focus on moments a grassroots football coach would want to review with their team."
        )
        analysis = await self.analyse_video(video_id, prompt)
        return [{"raw_analysis": analysis}]

    async def get_tactical_analysis(self, video_id: str, context: str = "") -> str:
        """Get detailed tactical analysis of the match."""
        coach_line = f"\nAdditional context from the coach: {context}" if context else ""
        prompt = (
            "You are an experienced football coach analysing match footage. Provide a detailed "
            "tactical analysis covering:\n"
            "1. Formation and shape - how the team sets up in and out of possession\n"
            "2. Build-up play - how they progress the ball from defence to attack\n"
            "3. Pressing and defensive transitions - intensity, triggers, shape\n"
            "4. Attacking patterns - width, combinations, final third entries\n"
            "5. Set pieces - any notable patterns\n"
            "6. Key weaknesses to address in training\n"
            "7. Key strengths to reinforce\n"
            + coach_line
        )
        return await self.analyse_video(video_id, prompt)

    async def search_moments(self, query: str, match_video_id: str | None = None) -> list[dict]:
        """Search for specific moments across indexed videos."""
        payload = {
            "index_id": self.index_id,
            "query_text": query,
            "search_options": ["visual", "conversation"],
        }
        if match_video_id:
            payload["filter"] = {"id": [match_video_id]}

        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                f"{self.base_url}/search",
                json=payload,
                headers=self.headers,
            )
            resp.raise_for_status()
            data = resp.json()
            results = []
            for item in data.get("data", []):
                results.append({
                    "video_id": item["video_id"],
                    "start": item["start"],
                    "end": item["end"],
                    "confidence": item["confidence"],
                    "metadata": item.get("metadata", {}),
                })
            return results

    async def create_entity_collection(self, name: str) -> str:
        """Create an entity collection (e.g., a squad)."""
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{self.base_url}/entity-collections",
                json={"name": name},
                headers=self.headers,
            )
            resp.raise_for_status()
            return resp.json()["_id"]

    async def add_player_entity(
        self, collection_id: str, name: str, image_urls: list[str]
    ) -> str:
        """Add a player to an entity collection with reference images."""
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{self.base_url}/entity-collections/{collection_id}/entities",
                json={
                    "name": name,
                    "image_urls": image_urls,
                },
                headers=self.headers,
            )
            resp.raise_for_status()
            return resp.json()["_id"]
