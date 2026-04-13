"""Process a local match video through the full pipeline.

Usage: python process_local.py <file_path> <title> [opponent]

Pipeline:
1. Create match record in Supabase
2. Upload to Mux via curl (reliable for large files)
3. Split into halves with ffmpeg
4. Upload halves to Gemini for analysis
5. Index first half with TwelveLabs for semantic search
"""

import asyncio
import os
import sys
import uuid
import logging

# Add parent to path so we can import app modules
sys.path.insert(0, os.path.dirname(__file__))

from app.services.mux_service import MuxService
from app.services.twelvelabs_service import TwelveLabsService
from app.services.supabase_service import SupabaseService
from app.services.video_splitter import split_match, get_video_duration

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

mux = MuxService()
twelvelabs = TwelveLabsService()
db = SupabaseService()


async def main():
    if len(sys.argv) < 3:
        print("Usage: python process_local.py <file_path> <title> [opponent]")
        sys.exit(1)

    file_path = sys.argv[1]
    title = sys.argv[2]
    opponent = sys.argv[3] if len(sys.argv) > 3 else None

    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        sys.exit(1)

    file_size = os.path.getsize(file_path)
    duration = await get_video_duration(file_path)
    match_id = str(uuid.uuid4())

    logger.info("=== PROCESSING: %s ===", title)
    logger.info("File: %s (%.1f GB, %.0f min)", file_path, file_size / 1e9, duration / 60)
    logger.info("Match ID: %s", match_id)

    # Step 1: Create match record
    logger.info("Step 1: Creating match record...")
    await db.insert("matches", {
        "id": match_id,
        "title": title,
        "opponent": opponent,
        "duration_seconds": duration,
        "status": "processing",
    })
    logger.info("Match record created: %s", match_id)

    # Step 2: Upload to Mux (with retry)
    logger.info("Step 2: Uploading to Mux (%.1f GB)...", file_size / 1e9)
    mux_ok = False
    for mux_attempt in range(1, 4):
        upload = await mux.create_upload()
        logger.info("  Mux upload attempt %d/3...", mux_attempt)
        proc = await asyncio.create_subprocess_exec(
            "curl", "-s", "-o", "/dev/null", "-w", "%{http_code}",
            "--upload-file", file_path,
            "-H", "Content-Type: video/mp4",
            upload["upload_url"],
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()
        http_code = stdout.decode().strip()[-3:]
        logger.info("  Mux upload HTTP %s", http_code)
        if http_code.startswith("2"):
            mux_ok = True
            break
        logger.warning("  Mux upload failed (attempt %d), retrying in %ds...", mux_attempt, 15 * mux_attempt)
        await asyncio.sleep(15 * mux_attempt)

    if not mux_ok:
        logger.error("Mux upload failed after 3 attempts! Continuing anyway...")

    # Poll for Mux asset
    logger.info("Waiting for Mux asset...")
    asset_id = None
    for i in range(90):
        asset_id = await mux.get_asset_from_upload(upload["upload_id"])
        if asset_id:
            break
        await asyncio.sleep(5)
        if i % 6 == 0:
            logger.info("  Still waiting for Mux asset... (%ds)", i * 5)

    if asset_id:
        for i in range(120):
            asset = await mux.get_asset(asset_id)
            if asset["status"] == "ready":
                await db.update("matches", match_id, {
                    "mux_asset_id": asset["asset_id"],
                    "mux_playback_id": asset["playback_id"],
                    "thumbnail_url": asset.get("thumbnail_url"),
                })
                logger.info("Mux READY — playback ID: %s", asset["playback_id"])
                break
            if asset["status"] == "errored":
                logger.error("Mux asset errored!")
                break
            await asyncio.sleep(5)
            if i % 12 == 0:
                logger.info("  Mux processing... (%ds)", i * 5)
    else:
        logger.warning("Mux asset not found after polling")

    # Step 3: Split into halves
    logger.info("Step 3: Splitting video into halves...")
    halves = await split_match(file_path, match_id)
    logger.info("Split into %d parts", len(halves))
    for h in halves:
        size_mb = os.path.getsize(h["path"]) / (1024 * 1024) if h["path"] != file_path else file_size / (1024 * 1024)
        logger.info("  %s: %.0f MB (%.0f-%.0f sec)", h["label"], size_mb, h["start"], h["end"])

    # Step 4: Upload to Gemini for analysis (parallel with retry)
    logger.info("Step 4: Uploading halves to Gemini (parallel)...")
    try:
        import time
        import google.genai as genai
        from app.core.config import get_settings
        settings = get_settings()
        client = genai.Client(api_key=settings.gemini_api_key, http_options={"timeout": 600_000})

        MAX_RETRIES = 3

        def upload_and_wait(half):
            """Upload a single quarter to Gemini with retry, then wait until ACTIVE."""
            label = half["label"]
            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    logger.info("  [%s] Uploading (attempt %d/%d)...", label, attempt, MAX_RETRIES)
                    f = client.files.upload(file=half["path"])
                    logger.info("  [%s] Uploaded: %s (state: %s)", label, f.name, f.state.name)

                    while f.state.name == "PROCESSING":
                        time.sleep(10)
                        f = client.files.get(name=f.name)
                        logger.info("  [%s] Processing... (%s)", label, f.state.name)

                    if f.state.name == "ACTIVE":
                        logger.info("  [%s] READY → %s", label, f.name)
                        return {"file": f, "label": label}
                    else:
                        logger.warning("  [%s] State: %s (attempt %d)", label, f.state.name, attempt)
                except Exception as e:
                    logger.warning("  [%s] Upload error (attempt %d): %s", label, attempt, e)
                    if attempt < MAX_RETRIES:
                        wait = 10 * attempt
                        logger.info("  [%s] Retrying in %ds...", label, wait)
                        time.sleep(wait)
            logger.error("  [%s] FAILED after %d attempts", label, MAX_RETRIES)
            return None

        # Upload all quarters in parallel using threads
        from concurrent.futures import ThreadPoolExecutor, as_completed
        gemini_files = []
        with ThreadPoolExecutor(max_workers=len(halves)) as pool:
            futures = {pool.submit(upload_and_wait, h): h["label"] for h in halves}
            for future in as_completed(futures):
                result = future.result()
                if result:
                    gemini_files.append(result)

        # Sort by label so analysis order is consistent (Q1, Q2, Q3, Q4)
        gemini_files.sort(key=lambda x: x["label"])
        logger.info("Gemini files ready: %d/%d", len(gemini_files), len(halves))

        # Run analysis on each quarter (sequential — respects rate limits)
        for i, gf in enumerate(gemini_files):
            if i > 0:
                logger.info("  Pausing 65s for Gemini rate limit...")
                await asyncio.sleep(65)

            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    logger.info("  Analysing %s with Gemini (attempt %d)...", gf["label"], attempt)
                    response = client.models.generate_content(
                        model="gemini-2.5-flash",
                        contents=[
                            gf["file"],
                            f"Analyse this football match video ({gf['label']}). "
                            f"Cover: formations, tactical patterns, key moments with timestamps, "
                            f"standout players, defensive and attacking analysis, set pieces. "
                            f"Be specific with timestamps and player descriptions.",
                        ],
                    )
                    analysis_text = response.text
                    logger.info("  %s analysis: %d chars", gf["label"], len(analysis_text))

                    analysis_id = str(uuid.uuid4())
                    await db.insert("analyses", {
                        "id": analysis_id,
                        "match_id": match_id,
                        "analysis_type": "full",
                        "status": "complete",
                        "prompt": f"Gemini analysis of {gf['label']}",
                        "tactical_raw": analysis_text,
                    })
                    logger.info("  Saved analysis %s for %s", analysis_id, gf["label"])
                    break
                except Exception as e:
                    logger.warning("  Analysis error for %s (attempt %d): %s", gf["label"], attempt, e)
                    if attempt < MAX_RETRIES:
                        wait = 30 * attempt
                        logger.info("  Retrying analysis in %ds...", wait)
                        await asyncio.sleep(wait)
                    else:
                        logger.error("  FAILED analysis for %s after %d attempts", gf["label"], MAX_RETRIES)

    except ImportError:
        logger.warning("google-genai not installed — skipping Gemini analysis")
    except Exception as e:
        logger.error("Gemini analysis failed: %s", e)

    # Step 5: Index first half with TwelveLabs (if rate limit allows)
    logger.info("Step 5: Indexing first half with TwelveLabs...")
    try:
        first_half = halves[0]
        result = await twelvelabs.index_video_from_file(
            first_half["path"],
            f"{title} - {first_half['label']}",
            "video/mp4",
        )
        logger.info("TwelveLabs task submitted: %s", result["task_id"])

        for i in range(180):
            await asyncio.sleep(5)
            task = await twelvelabs.get_task_status(result["task_id"])
            if task["status"] == "ready":
                await db.update("matches", match_id, {
                    "twelvelabs_video_id": task["video_id"],
                    "twelvelabs_task_id": result["task_id"],
                })
                logger.info("TwelveLabs READY: %s", task["video_id"])
                break
            elif task["status"] == "failed":
                logger.error("TwelveLabs indexing failed")
                break
            if i % 12 == 0:
                logger.info("  TwelveLabs indexing... (%ds)", i * 5)
    except Exception as e:
        logger.warning("TwelveLabs indexing failed (rate limit?): %s", e)

    # Mark match as ready
    await db.update("matches", match_id, {"status": "ready"})
    logger.info("=== DONE: %s ===", title)
    logger.info("Match ID: %s", match_id)
    logger.info("View at: http://localhost:5176/match/%s", match_id)

    # Cleanup split files (not the original)
    for h in halves:
        if h["path"] != file_path and os.path.exists(h["path"]):
            os.remove(h["path"])
            logger.info("Cleaned up: %s", h["path"])


if __name__ == "__main__":
    asyncio.run(main())
