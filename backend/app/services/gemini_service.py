"""Gemini video analysis service — analyses match footage directly.

Uses Gemini 2.5 Flash for fast video analysis. Hourly-tier rate limit of
~250K input tokens/min requires a ~65s pause between quarter analyses.
Gemini File API caps at 2GB per file and files expire after 48 hours.
"""

import asyncio
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from google import genai
from app.core.config import get_settings

logger = logging.getLogger(__name__)

# Prompt used for per-segment match analysis. Deliberately short — Gemini
# refuses overly long prompts, and hallucinates disciplinary events, so we
# explicitly rule those out.
MATCH_ANALYSIS_PROMPT = (
    "You are a professional football analyst watching a match video segment ({label}). "
    "Provide a detailed tactical analysis covering:\n"
    "- Formations and shape of both teams (in and out of possession)\n"
    "- Build-up play and attacking patterns\n"
    "- Pressing triggers, defensive transitions, and shape\n"
    "- Key tactical moments with approximate timestamps\n"
    "- Standout individual performances — reference players by kit colour and shirt number\n"
    "- Set piece routines\n"
    "Reference the teams by their kit colour throughout. Be specific with timestamps. "
    "Do NOT report yellow cards, red cards, sendings off, or exact scorelines — focus on "
    "tactical patterns, positioning, and style of play."
)


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
        """Analyse a video already uploaded to Gemini File API. Uses the async client."""
        logger.info("Gemini analysing with prompt: %s...", prompt[:80])
        response = await self.client.aio.models.generate_content(
            model=self.model,
            contents=[
                genai.types.Part.from_uri(file_uri=file_uri, mime_type="video/mp4"),
                prompt,
            ],
        )
        text = response.text or ""
        logger.info("Gemini returned %d chars", len(text))
        return text

    async def analyse_video_file(self, file_path: str, prompt: str, display_name: str = "") -> str:
        """Upload a video file and analyse it. Full pipeline."""
        file_uri = await asyncio.to_thread(self.upload_and_wait, file_path, display_name)
        return await self.analyse_with_file_uri(file_uri, prompt)

    # --- Pipeline helper ---

    def _upload_one_sync(self, file_path: str, label: str, max_retries: int = 3) -> dict | None:
        """Sync: upload one segment to Gemini and wait until ACTIVE. Retries on failure."""
        for attempt in range(1, max_retries + 1):
            try:
                logger.info("[%s] Gemini upload attempt %d/%d", label, attempt, max_retries)
                f = self.client.files.upload(file=file_path)
                logger.info("[%s] uploaded: %s (state=%s)", label, f.name, f.state.name)
                while f.state.name == "PROCESSING":
                    time.sleep(10)
                    f = self.client.files.get(name=f.name)
                if f.state.name == "ACTIVE":
                    return {"file": f, "label": label}
                logger.warning("[%s] non-ACTIVE state: %s (attempt %d)", label, f.state.name, attempt)
            except Exception as e:
                logger.warning("[%s] upload error attempt %d: %s", label, attempt, e)
                if attempt < max_retries:
                    time.sleep(10 * attempt)
        return None

    async def analyse_segments(
        self,
        segments: list[dict],
        match_id: str,
        db,
        match_title: str = "",
    ) -> dict:
        """Upload and analyse match segments with Gemini, save tactical_raw to analyses table.

        Uploads in parallel threads (Gemini SDK is sync), analyses sequentially with
        a 65s pause between segments to respect the hourly token rate limit. Deletes
        Gemini files afterwards so the 48h File API storage doesn't accumulate.

        Args:
            segments: list of {"path": str, "label": str, "start": float, "end": float}
            match_id: the Supabase matches.id to link analyses rows to
            db: SupabaseService instance for inserting rows
            match_title: for logging only

        Returns:
            {"analyses_created": int, "errors": list[str]}
        """
        errors: list[str] = []

        # Parallel uploads — Gemini SDK is blocking so run in a thread pool
        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor(max_workers=max(1, len(segments))) as pool:
            upload_futures = [
                loop.run_in_executor(pool, self._upload_one_sync, s["path"], s["label"])
                for s in segments
            ]
            upload_results = await asyncio.gather(*upload_futures)

        gemini_files = [r for r in upload_results if r is not None]
        gemini_files.sort(key=lambda x: x["label"])
        failed_uploads = len(segments) - len(gemini_files)
        if failed_uploads:
            errors.append(f"{failed_uploads} segment(s) failed to upload to Gemini")
        logger.info("Gemini files ready: %d/%d for match %s", len(gemini_files), len(segments), match_id)

        # Sequential analysis with 65s pacing
        analyses_created = 0
        for i, gf in enumerate(gemini_files):
            if i > 0:
                logger.info("Pausing 65s for Gemini rate limit before %s...", gf["label"])
                await asyncio.sleep(65)

            prompt = MATCH_ANALYSIS_PROMPT.format(label=gf["label"])
            analysis_text = None
            for attempt in range(1, 4):
                try:
                    logger.info("Analysing %s with Gemini (attempt %d/3)", gf["label"], attempt)
                    response = await self.client.aio.models.generate_content(
                        model=self.model,
                        contents=[gf["file"], prompt],
                    )
                    analysis_text = response.text or ""
                    logger.info("%s analysis: %d chars", gf["label"], len(analysis_text))
                    break
                except Exception as e:
                    logger.warning("Analysis error %s attempt %d: %s", gf["label"], attempt, e)
                    if attempt < 3:
                        await asyncio.sleep(30 * attempt)
                    else:
                        errors.append(f"{gf['label']}: {e}")

            if analysis_text:
                try:
                    await db.insert("analyses", {
                        "id": str(uuid.uuid4()),
                        "match_id": match_id,
                        "analysis_type": "full",
                        "status": "complete",
                        "prompt": f"Gemini analysis of {gf['label']}",
                        "tactical_raw": analysis_text,
                    })
                    analyses_created += 1
                except Exception as e:
                    logger.exception("Failed to save %s analysis to DB", gf["label"])
                    errors.append(f"{gf['label']} save: {e}")

            # Cleanup Gemini file — 48h TTL otherwise, and quota can fill up
            try:
                await asyncio.to_thread(self.client.files.delete, name=gf["file"].name)
            except Exception as e:
                logger.warning("Gemini file cleanup failed for %s: %s", gf["label"], e)

        return {"analyses_created": analyses_created, "errors": errors}
