"""Gemini video analysis service — analyses match footage directly.

Uses Gemini 2.5 Flash for fast, cheap video analysis with no rate limits.
Requires video to be uploaded to Gemini File API first (max 2GB per file).
"""

import logging
import time
from google import genai
from app.core.config import get_settings

logger = logging.getLogger(__name__)


class GeminiService:
    def __init__(self):
        s = get_settings()
        self.client = genai.Client(
            api_key=s.gemini_api_key,
            http_options={"timeout": 600_000},  # 10 min timeout for long video analysis
        )
        self.model = "gemini-2.5-flash"

    def upload_and_wait(self, file_path: str, display_name: str = "") -> str:
        """Upload a video file to Gemini File API and wait until active.

        Returns the file URI (e.g. https://generativelanguage.googleapis.com/v1beta/files/xxx).
        """
        logger.info("Uploading %s to Gemini File API...", file_path)
        uploaded = self.client.files.upload(
            file=file_path,
            config={"display_name": display_name} if display_name else None,
        )
        logger.info("Gemini file: %s (state: %s)", uploaded.name, uploaded.state)

        while uploaded.state.name == "PROCESSING":
            time.sleep(5)
            uploaded = self.client.files.get(name=uploaded.name)

        if uploaded.state.name != "ACTIVE":
            raise Exception(f"Gemini file failed: {uploaded.state.name}")

        logger.info("Gemini file active: %s", uploaded.uri)
        return uploaded.uri

    def get_active_file_uri(self, display_name: str = "") -> str | None:
        """Find an existing active file by display name."""
        for f in self.client.files.list():
            if f.state.name == "ACTIVE":
                if not display_name or (f.display_name and display_name in f.display_name):
                    return f.uri
        return None

    async def analyse_with_file_uri(self, file_uri: str, prompt: str) -> str:
        """Analyse a video that's already uploaded to Gemini File API."""
        logger.info("Gemini analysing with prompt: %s...", prompt[:80])

        response = self.client.models.generate_content(
            model=self.model,
            contents=[
                genai.types.Part.from_uri(file_uri=file_uri, mime_type="video/mp4"),
                prompt,  # Pass as raw string, not Part.from_text()
            ],
        )

        text = response.text or ""
        logger.info("Gemini returned %d chars", len(text))
        return text

    async def analyse_video_file(self, file_path: str, prompt: str, display_name: str = "") -> str:
        """Upload a video file and analyse it. Full pipeline."""
        file_uri = self.upload_and_wait(file_path, display_name)
        return await self.analyse_with_file_uri(file_uri, prompt)
