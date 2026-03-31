"""Gemini video analysis service — analyses match footage directly.

Uses Gemini 2.5 Flash for fast, cheap video analysis with no rate limits.
TwelveLabs is kept for semantic search; Gemini handles all analysis prompts.
"""

import logging
from google import genai
from app.core.config import get_settings

logger = logging.getLogger(__name__)


class GeminiService:
    def __init__(self):
        s = get_settings()
        self.client = genai.Client(api_key=s.gemini_api_key)
        self.model = "gemini-2.5-flash"

    async def analyse_video_url(self, video_url: str, prompt: str) -> str:
        """Analyse a video from a Mux stream URL using Gemini."""
        logger.info("Gemini analysing video with prompt: %s...", prompt[:80])

        response = self.client.models.generate_content(
            model=self.model,
            contents=[
                genai.types.Content(
                    parts=[
                        genai.types.Part.from_uri(
                            file_uri=video_url,
                            mime_type="video/mp4",
                        ),
                        genai.types.Part.from_text(prompt),
                    ]
                )
            ],
        )

        text = response.text or ""
        logger.info("Gemini returned %d chars", len(text))
        return text

    async def analyse_video_file(self, file_path: str, prompt: str) -> str:
        """Analyse a local video file using Gemini.

        Uploads the file to Gemini's File API first, then analyses it.
        """
        import os
        logger.info("Uploading %s to Gemini File API...", os.path.basename(file_path))

        # Upload file to Gemini
        uploaded = self.client.files.upload(file=file_path)
        logger.info("Uploaded to Gemini: %s (state: %s)", uploaded.name, uploaded.state)

        # Wait for processing if needed
        import time
        while uploaded.state.name == "PROCESSING":
            time.sleep(5)
            uploaded = self.client.files.get(name=uploaded.name)
            logger.info("Gemini file state: %s", uploaded.state)

        if uploaded.state.name == "FAILED":
            raise Exception(f"Gemini file processing failed: {uploaded.name}")

        # Analyse
        logger.info("Gemini analysing with prompt: %s...", prompt[:80])
        response = self.client.models.generate_content(
            model=self.model,
            contents=[
                genai.types.Content(
                    parts=[
                        genai.types.Part.from_uri(
                            file_uri=uploaded.uri,
                            mime_type=uploaded.mime_type,
                        ),
                        genai.types.Part.from_text(prompt),
                    ]
                )
            ],
        )

        text = response.text or ""
        logger.info("Gemini returned %d chars", len(text))
        return text

    async def analyse_match(self, mux_playback_id: str, prompt: str) -> str:
        """Analyse a match using its Mux HLS stream URL.

        Gemini can process HLS streams directly — no download needed.
        """
        stream_url = f"https://stream.mux.com/{mux_playback_id}.m3u8"
        return await self.analyse_video_url(stream_url, prompt)
