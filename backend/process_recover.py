"""Recover Cobham match — split into quarters, upload to Gemini, analyse.

Mux is already done. This script handles:
1. Split into ~15-min quarters (~1GB each)
2. Upload each quarter to Gemini
3. Run analysis on each
4. Save results to DB
5. Index first quarter with TwelveLabs
"""

import asyncio
import os
import sys
import uuid
import time
import logging

sys.path.insert(0, os.path.dirname(__file__))

from app.services.twelvelabs_service import TwelveLabsService
from app.services.supabase_service import SupabaseService
from app.services.video_splitter import get_video_duration

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

MATCH_ID = "b10b8f07-0426-4fe9-beea-51b0272b5daa"
TITLE = "VTFC vs Cobham"

twelvelabs = TwelveLabsService()
db = SupabaseService()


async def split_into_quarters(file_path: str, match_id: str) -> list[dict]:
    """Split a match video into ~15-min quarters, skipping halftime."""
    duration = await get_video_duration(file_path)
    logger.info("Total duration: %.0f sec (%.0f min)", duration, duration / 60)

    # For a ~114 min VEO recording:
    # 0:00 - ~50:00 = First half
    # ~50:00 - ~63:00 = Halftime (skip)
    # ~63:00 - ~114:00 = Second half
    #
    # Split into 4 playing quarters:
    # Q1: 0:00 - 25:00 (first half, first 25 min)
    # Q2: 25:00 - 50:00 (first half, second 25 min)
    # Q3: 63:00 - 88:00 (second half, first 25 min)
    # Q4: 88:00 - end (second half, remaining)

    # Detect halftime roughly
    ht_start = duration * 0.44  # ~50 min
    ht_end = duration * 0.55    # ~63 min
    h1_mid = ht_start / 2       # ~25 min
    h2_mid = ht_end + (duration - ht_end) / 2  # ~88 min

    segments = [
        {"label": "Q1 - First Half (0-25min)", "start": 0, "end": h1_mid},
        {"label": "Q2 - First Half (25-50min)", "start": h1_mid, "end": ht_start},
        {"label": "Q3 - Second Half (63-88min)", "start": ht_end, "end": h2_mid},
        {"label": "Q4 - Second Half (88min-end)", "start": h2_mid, "end": duration},
    ]

    results = []
    ext = os.path.splitext(file_path)[1]

    for i, seg in enumerate(segments):
        out_path = os.path.join(UPLOAD_DIR, f"{match_id}_q{i+1}{ext}")
        seg_duration = seg["end"] - seg["start"]
        logger.info("Extracting %s: %.0f-%.0f sec (%.0f min)", seg["label"], seg["start"], seg["end"], seg_duration / 60)

        proc = await asyncio.create_subprocess_exec(
            "ffmpeg", "-y", "-i", file_path,
            "-ss", str(int(seg["start"])),
            "-t", str(int(seg_duration)),
            "-c", "copy",
            out_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await proc.communicate()

        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            size_mb = os.path.getsize(out_path) / (1024 * 1024)
            logger.info("  Created: %s (%.0f MB)", out_path, size_mb)
            results.append({**seg, "path": out_path})
        else:
            logger.error("  Failed to create %s", out_path)

    return results


async def upload_to_gemini(file_path: str, label: str, max_retries: int = 3):
    """Upload a file to Gemini with retry logic."""
    import google.genai as genai
    from app.core.config import get_settings
    settings = get_settings()
    client = genai.Client(api_key=settings.gemini_api_key, http_options={"timeout": 600_000})

    for attempt in range(max_retries):
        try:
            logger.info("  Uploading %s to Gemini (attempt %d/%d)...", label, attempt + 1, max_retries)
            f = client.files.upload(file=file_path)
            logger.info("  Upload started: %s (state: %s)", f.name, f.state.name)

            while f.state.name == "PROCESSING":
                time.sleep(10)
                f = client.files.get(name=f.name)
                logger.info("  Processing %s... (%s)", label, f.state.name)

            if f.state.name == "ACTIVE":
                logger.info("  Gemini READY: %s -> %s", label, f.name)
                return f
            else:
                logger.error("  Gemini failed for %s: %s", label, f.state.name)
                if attempt < max_retries - 1:
                    logger.info("  Retrying in 30s...")
                    time.sleep(30)

        except Exception as e:
            logger.error("  Upload error for %s: %s", label, e)
            if attempt < max_retries - 1:
                logger.info("  Retrying in 30s...")
                time.sleep(30)

    return None


async def analyse_with_gemini(gemini_file, label: str) -> str | None:
    """Run Gemini analysis on an uploaded video segment."""
    import google.genai as genai
    from app.core.config import get_settings
    settings = get_settings()
    client = genai.Client(api_key=settings.gemini_api_key, http_options={"timeout": 600_000})

    try:
        logger.info("  Analysing %s...", label)
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                gemini_file,
                f"Analyse this football match video segment ({label}). "
                f"Cover: formations, tactical patterns, key moments with timestamps, "
                f"standout players, defensive and attacking analysis, set pieces. "
                f"Be specific with timestamps and player descriptions.",
            ],
        )
        text = response.text
        logger.info("  Analysis complete: %d chars", len(text))
        return text
    except Exception as e:
        logger.error("  Analysis failed for %s: %s", label, e)
        return None


async def main():
    file_path = sys.argv[1] if len(sys.argv) > 1 else "C:/Users/info/Downloads/first-team-vs-cobham-2025-08-26.mp4"

    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        sys.exit(1)

    logger.info("=== RECOVERING: %s ===", TITLE)
    logger.info("Match ID: %s", MATCH_ID)

    # Step 1: Split into quarters
    logger.info("Step 1: Splitting into quarters...")
    quarters = await split_into_quarters(file_path, MATCH_ID)
    logger.info("Created %d quarters", len(quarters))

    # Step 2: Upload each quarter to Gemini and analyse
    logger.info("Step 2: Uploading quarters to Gemini + analysing...")
    all_analysis = []

    for q in quarters:
        gemini_file = await upload_to_gemini(q["path"], q["label"])
        if not gemini_file:
            logger.error("Skipping %s — upload failed", q["label"])
            continue

        analysis_text = await analyse_with_gemini(gemini_file, q["label"])
        if analysis_text:
            analysis_id = str(uuid.uuid4())
            await db.insert("analyses", {
                "id": analysis_id,
                "match_id": MATCH_ID,
                "analysis_type": "full",
                "status": "complete",
                "prompt": f"Gemini analysis of {q['label']}",
                "tactical_raw": analysis_text,
            })
            all_analysis.append(analysis_text)
            logger.info("  Saved analysis for %s (%s)", q["label"], analysis_id)

    logger.info("Gemini: %d/%d quarters analysed", len(all_analysis), len(quarters))

    # Step 3: Generate combined coaching advice with Claude
    if all_analysis:
        logger.info("Step 3: Generating coaching advice...")
        try:
            import anthropic
            from app.core.config import get_settings
            settings = get_settings()
            client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

            combined = "\n\n---\n\n".join(all_analysis)
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=4096,
                messages=[{
                    "role": "user",
                    "content": f"You are Manager Mentor, an elite football coaching AI. "
                    f"Below is a detailed tactical analysis of a match: {TITLE}. "
                    f"Provide coaching advice covering:\n"
                    f"1. Overall tactical assessment\n"
                    f"2. Key strengths shown\n"
                    f"3. Areas for improvement\n"
                    f"4. Specific training drill recommendations\n"
                    f"5. Set piece observations\n"
                    f"6. Individual player highlights\n\n"
                    f"ANALYSIS:\n{combined}"
                }],
            )
            coaching = response.content[0].text
            logger.info("Coaching advice: %d chars", len(coaching))

            advice_id = str(uuid.uuid4())
            await db.insert("analyses", {
                "id": advice_id,
                "match_id": MATCH_ID,
                "analysis_type": "full",
                "status": "complete",
                "prompt": "Combined coaching advice",
                "coaching_advice": coaching,
                "tactical_raw": combined[:10000],  # Store truncated raw for reference
            })
            logger.info("Saved coaching advice: %s", advice_id)
        except Exception as e:
            logger.error("Coaching advice failed: %s", e)

    # Step 4: Index first quarter with TwelveLabs
    logger.info("Step 4: Indexing Q1 with TwelveLabs...")
    try:
        q1 = quarters[0]
        result = await twelvelabs.index_video_from_file(
            q1["path"], f"{TITLE} - Q1", "video/mp4",
        )
        logger.info("TwelveLabs task: %s", result["task_id"])

        for i in range(180):
            await asyncio.sleep(5)
            task = await twelvelabs.get_task_status(result["task_id"])
            if task["status"] == "ready":
                await db.update("matches", MATCH_ID, {
                    "twelvelabs_video_id": task["video_id"],
                    "twelvelabs_task_id": result["task_id"],
                })
                logger.info("TwelveLabs READY: %s", task["video_id"])
                break
            elif task["status"] == "failed":
                logger.error("TwelveLabs failed")
                break
            if i % 12 == 0 and i > 0:
                logger.info("  TwelveLabs indexing... (%ds)", i * 5)
    except Exception as e:
        logger.warning("TwelveLabs failed: %s", e)

    # Mark match ready
    await db.update("matches", MATCH_ID, {"status": "ready"})

    # Cleanup quarter files
    for q in quarters:
        if os.path.exists(q["path"]):
            os.remove(q["path"])
            logger.info("Cleaned up: %s", os.path.basename(q["path"]))

    logger.info("=== DONE: %s ===", TITLE)
    logger.info("View at: http://localhost:5176/match/%s", MATCH_ID)


if __name__ == "__main__":
    asyncio.run(main())
