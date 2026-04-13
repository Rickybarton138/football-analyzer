"""Re-analyse Cobham match — use existing Gemini files, upload Q4, run analysis.

Gemini files Q1-Q3 already uploaded. Just need:
1. Re-upload Q4 (failed due to network)
2. Run analysis on all 4 with correct model name
3. Generate coaching advice with Claude
"""

import asyncio
import os
import sys
import uuid
import time
import logging

sys.path.insert(0, os.path.dirname(__file__))

from app.services.supabase_service import SupabaseService

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
MATCH_ID = "b10b8f07-0426-4fe9-beea-51b0272b5daa"
TITLE = "VTFC vs Cobham"
MODEL = "gemini-2.5-flash"

db = SupabaseService()


def get_gemini_client():
    import google.genai as genai
    from app.core.config import get_settings
    settings = get_settings()
    return genai.Client(api_key=settings.gemini_api_key, http_options={"timeout": 600_000})


def upload_file(client, file_path: str, label: str, max_retries: int = 3):
    """Upload with retry."""
    for attempt in range(max_retries):
        try:
            logger.info("Uploading %s (attempt %d/%d)...", label, attempt + 1, max_retries)
            f = client.files.upload(file=file_path)
            while f.state.name == "PROCESSING":
                time.sleep(10)
                f = client.files.get(name=f.name)
                logger.info("  Processing %s... (%s)", label, f.state.name)
            if f.state.name == "ACTIVE":
                logger.info("  READY: %s -> %s", label, f.name)
                return f
            logger.error("  Failed: %s", f.state.name)
        except Exception as e:
            logger.error("  Error: %s", e)
            if attempt < max_retries - 1:
                time.sleep(30)
    return None


def analyse(client, gemini_file, label: str) -> str | None:
    """Run analysis with correct model."""
    try:
        logger.info("Analysing %s with %s...", label, MODEL)
        response = client.models.generate_content(
            model=MODEL,
            contents=[
                gemini_file,
                f"Analyse this football match video segment ({label}). "
                f"Cover: formations, tactical patterns, key moments with timestamps, "
                f"standout players, defensive and attacking analysis, set pieces. "
                f"Be specific with timestamps and player descriptions.",
            ],
        )
        text = response.text
        logger.info("  %s: %d chars", label, len(text))
        return text
    except Exception as e:
        logger.error("  Analysis failed: %s", e)
        return None


async def main():
    client = get_gemini_client()

    # Check if existing Gemini files are still valid
    existing_files = {
        "Q1 - First Half (0-25min)": "files/fl61iqy16z69",
        "Q2 - First Half (25-50min)": "files/mskvufxc4wjc",
        "Q3 - Second Half (63-88min)": "files/2e9en5x2yw1i",
    }

    gemini_files = {}
    for label, file_name in existing_files.items():
        try:
            f = client.files.get(name=file_name)
            if f.state.name == "ACTIVE":
                logger.info("Existing file OK: %s -> %s", label, file_name)
                gemini_files[label] = f
            else:
                logger.warning("File expired or not active: %s (%s)", file_name, f.state.name)
        except Exception as e:
            logger.warning("File not found: %s (%s)", file_name, e)

    # Upload Q4 if needed
    q4_label = "Q4 - Second Half (88min-end)"
    q4_path = os.path.join(UPLOAD_DIR, f"{MATCH_ID}_q4.mp4")
    if not os.path.exists(q4_path):
        # Re-split Q4 from original
        source = "C:/Users/info/Downloads/first-team-vs-cobham-2025-08-26.mp4"
        if os.path.exists(source):
            logger.info("Re-extracting Q4 from original...")
            proc = await asyncio.create_subprocess_exec(
                "ffmpeg", "-y", "-i", source,
                "-ss", "5323", "-c", "copy", q4_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await proc.communicate()
            logger.info("Q4 extracted: %.0f MB", os.path.getsize(q4_path) / (1024 * 1024))
        else:
            logger.error("Original file not found: %s", source)

    if os.path.exists(q4_path):
        f = upload_file(client, q4_path, q4_label)
        if f:
            gemini_files[q4_label] = f

    logger.info("Ready to analyse %d/%d quarters", len(gemini_files), 4)

    # Run analysis on all available files
    all_analysis = []
    for label in ["Q1 - First Half (0-25min)", "Q2 - First Half (25-50min)",
                   "Q3 - Second Half (63-88min)", "Q4 - Second Half (88min-end)"]:
        if label not in gemini_files:
            logger.warning("Skipping %s — no file", label)
            continue

        text = analyse(client, gemini_files[label], label)
        if text:
            analysis_id = str(uuid.uuid4())
            await db.insert("analyses", {
                "id": analysis_id,
                "match_id": MATCH_ID,
                "analysis_type": "full",
                "status": "complete",
                "prompt": f"Gemini analysis of {label}",
                "tactical_raw": text,
            })
            all_analysis.append(text)
            logger.info("Saved analysis for %s", label)

    logger.info("Analysed %d quarters", len(all_analysis))

    # Generate coaching advice with Claude
    if all_analysis:
        logger.info("Generating coaching advice with Claude...")
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
                    f"Below is a detailed tactical analysis of: {TITLE}.\n\n"
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
                "match_id": MATCH_ID,
                "analysis_type": "full",
                "status": "complete",
                "prompt": "Combined coaching advice from Manager Mentor",
                "coaching_advice": coaching,
                "tactical_raw": combined[:10000],
            })
            logger.info("Saved coaching advice")
        except Exception as e:
            logger.error("Coaching advice failed: %s", e)

    # Mark match ready
    await db.update("matches", MATCH_ID, {"status": "ready"})
    logger.info("=== DONE: %s ===", TITLE)
    logger.info("View at: http://localhost:5176/match/%s", MATCH_ID)


if __name__ == "__main__":
    asyncio.run(main())
