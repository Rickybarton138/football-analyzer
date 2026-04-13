"""Run Gemini analysis on an existing match — re-split, upload, analyse, coaching advice.

Usage: python process_gemini.py <match_id> <file_path> [title]
"""

import asyncio
import os
import sys
import uuid
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(__file__))

from app.services.supabase_service import SupabaseService
from app.services.video_splitter import split_match, get_video_duration

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODEL = "gemini-2.5-flash"
MAX_RETRIES = 3

db = SupabaseService()


def get_gemini_client():
    import google.genai as genai
    from app.core.config import get_settings
    settings = get_settings()
    return genai.Client(api_key=settings.gemini_api_key, http_options={"timeout": 600_000})


def upload_and_wait(client, file_path: str, label: str):
    """Upload a single quarter to Gemini with retry."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logger.info("  [%s] Uploading (attempt %d/%d)...", label, attempt, MAX_RETRIES)
            f = client.files.upload(file=file_path)
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
                time.sleep(10 * attempt)
    logger.error("  [%s] FAILED after %d attempts", label, MAX_RETRIES)
    return None


async def main():
    if len(sys.argv) < 3:
        print("Usage: python process_gemini.py <match_id> <file_path> [title]")
        sys.exit(1)

    match_id = sys.argv[1]
    file_path = sys.argv[2]
    title = sys.argv[3] if len(sys.argv) > 3 else "Match"

    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        sys.exit(1)

    logger.info("=== GEMINI ANALYSIS: %s ===", title)
    logger.info("Match ID: %s", match_id)

    # Step 1: Split into quarters
    logger.info("Step 1: Splitting into quarters...")
    quarters = await split_match(file_path, match_id)
    logger.info("Split into %d parts", len(quarters))
    for q in quarters:
        size_mb = os.path.getsize(q["path"]) / (1024 * 1024) if q["path"] != file_path else os.path.getsize(file_path) / (1024 * 1024)
        logger.info("  %s: %.0f MB (%.0f-%.0f sec)", q["label"], size_mb, q["start"], q["end"])

    # Step 2: Parallel upload to Gemini
    logger.info("Step 2: Uploading %d quarters to Gemini (parallel)...", len(quarters))
    client = get_gemini_client()

    gemini_files = []
    with ThreadPoolExecutor(max_workers=len(quarters)) as pool:
        futures = {pool.submit(upload_and_wait, client, q["path"], q["label"]): q["label"] for q in quarters}
        for future in as_completed(futures):
            result = future.result()
            if result:
                gemini_files.append(result)

    gemini_files.sort(key=lambda x: x["label"])
    logger.info("Gemini files ready: %d/%d", len(gemini_files), len(quarters))

    if not gemini_files:
        logger.error("No files uploaded — aborting")
        sys.exit(1)

    # Step 3: Analyse each quarter (sequential for rate limits)
    logger.info("Step 3: Analysing quarters...")
    all_analysis = []
    for i, gf in enumerate(gemini_files):
        if i > 0:
            logger.info("  Pausing 65s for Gemini rate limit...")
            await asyncio.sleep(65)

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                logger.info("  Analysing %s (attempt %d)...", gf["label"], attempt)
                response = client.models.generate_content(
                    model=MODEL,
                    contents=[
                        gf["file"],
                        f"Analyse this football match video segment ({gf['label']}). "
                        f"Cover: formations, tactical patterns, key moments with timestamps, "
                        f"standout players, defensive and attacking analysis, set pieces. "
                        f"Be specific with timestamps and player descriptions.",
                    ],
                )
                analysis_text = response.text
                logger.info("  %s: %d chars", gf["label"], len(analysis_text))

                analysis_id = str(uuid.uuid4())
                await db.insert("analyses", {
                    "id": analysis_id,
                    "match_id": match_id,
                    "analysis_type": "full",
                    "status": "complete",
                    "prompt": f"Gemini analysis of {gf['label']}",
                    "tactical_raw": analysis_text,
                })
                all_analysis.append(analysis_text)
                logger.info("  Saved analysis for %s", gf["label"])
                break
            except Exception as e:
                logger.warning("  Analysis error for %s (attempt %d): %s", gf["label"], attempt, e)
                if attempt < MAX_RETRIES:
                    await asyncio.sleep(30 * attempt)
                else:
                    logger.error("  FAILED analysis for %s", gf["label"])

    logger.info("Analysed %d/%d quarters", len(all_analysis), len(gemini_files))

    # Step 4: Generate coaching advice with Claude
    if all_analysis:
        logger.info("Step 4: Generating coaching advice with Claude...")
        try:
            import anthropic
            from app.core.config import get_settings
            settings = get_settings()
            aclient = anthropic.Anthropic(api_key=settings.anthropic_api_key)

            combined = "\n\n---\n\n".join(all_analysis)
            response = aclient.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=4096,
                messages=[{
                    "role": "user",
                    "content": f"You are Manager Mentor, an elite grassroots football coaching AI. "
                    f"Below is a detailed tactical analysis of: {title}.\n\n"
                    f"Provide coaching advice covering:\n"
                    f"1. Overall tactical assessment\n"
                    f"2. Key strengths shown\n"
                    f"3. Areas for improvement with specific drills\n"
                    f"4. Set piece observations\n"
                    f"5. Individual player highlights\n"
                    f"6. Suggested training focus for next week\n\n"
                    f"ANALYSIS:\n{combined}"
                }],
            )
            coaching = response.content[0].text
            logger.info("Coaching advice: %d chars", len(coaching))

            advice_id = str(uuid.uuid4())
            await db.insert("analyses", {
                "id": advice_id,
                "match_id": match_id,
                "analysis_type": "full",
                "status": "complete",
                "prompt": "Combined coaching advice from Manager Mentor",
                "coaching_advice": coaching,
                "tactical_raw": combined[:10000],
            })
            logger.info("Saved coaching advice")
        except Exception as e:
            logger.error("Coaching advice failed: %s", e)

    # Cleanup split files
    for q in quarters:
        if q["path"] != file_path and os.path.exists(q["path"]):
            os.remove(q["path"])
            logger.info("Cleaned up: %s", q["path"])

    logger.info("=== DONE: %s ===", title)
    logger.info("View at: http://localhost:5176/match/%s", match_id)


if __name__ == "__main__":
    asyncio.run(main())
