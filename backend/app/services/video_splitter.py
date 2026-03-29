"""Video splitter — extracts match halves, skipping halftime.

For a typical VEO recording (~2 hours):
  0:00 - ~50:00  = First half
  ~50:00 - ~65:00 = Halftime (skip)
  ~65:00 - end    = Second half

Uses ffmpeg to split without re-encoding (fast, lossless).
"""

import asyncio
import logging
import os
import shutil

logger = logging.getLogger(__name__)

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


async def download_video(url: str, output_path: str) -> None:
    """Download a video from URL to a local file."""
    import httpx
    logger.info("Downloading video to %s", output_path)
    async with httpx.AsyncClient(timeout=httpx.Timeout(1800.0, connect=30.0)) as client:
        async with client.stream("GET", url) as resp:
            resp.raise_for_status()
            with open(output_path, "wb") as f:
                async for chunk in resp.aiter_bytes(chunk_size=4 * 1024 * 1024):
                    f.write(chunk)
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    logger.info("Downloaded %.0f MB to %s", size_mb, output_path)


async def get_video_duration(file_path: str) -> float:
    """Get video duration in seconds using ffprobe."""
    proc = await asyncio.create_subprocess_exec(
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        file_path,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, _ = await proc.communicate()
    return float(stdout.decode().strip())


async def detect_halftime(file_path: str, duration: float) -> tuple[float, float]:
    """Detect halftime by finding a quiet/static period in the middle of the video.

    Returns (halftime_start, halftime_end) in seconds.

    Strategy: For VEO footage, halftime typically starts around 45-55 min
    and lasts 10-20 min. We look for a section with minimal motion/audio
    in that window. Falls back to a simple split if detection fails.
    """
    # Simple heuristic: assume halftime is at roughly 42-48% of total duration
    # VEO recordings typically: ~50min half, ~15min break, ~50min half = ~115min
    # So halftime starts around 43% and ends around 56%
    mid_point = duration * 0.44
    halftime_duration = duration * 0.13  # ~15 min for a 2-hour recording

    # Try audio-based detection: find silence in the expected halftime window
    search_start = duration * 0.38
    search_end = duration * 0.58

    try:
        proc = await asyncio.create_subprocess_exec(
            "ffmpeg", "-i", file_path,
            "-ss", str(int(search_start)),
            "-t", str(int(search_end - search_start)),
            "-af", "silencedetect=noise=-35dB:d=30",
            "-f", "null", "-",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()
        output = stderr.decode()

        # Parse silence detection output
        silence_starts = []
        silence_ends = []
        for line in output.split("\n"):
            if "silence_start:" in line:
                try:
                    val = float(line.split("silence_start:")[1].strip().split()[0])
                    silence_starts.append(val + search_start)
                except (ValueError, IndexError):
                    pass
            if "silence_end:" in line:
                try:
                    val = float(line.split("silence_end:")[1].strip().split()[0])
                    silence_ends.append(val + search_start)
                except (ValueError, IndexError):
                    pass

        # Find the longest silence period (likely halftime)
        if silence_starts and silence_ends:
            longest_silence = 0
            best_start = mid_point
            best_end = mid_point + halftime_duration
            for i in range(min(len(silence_starts), len(silence_ends))):
                length = silence_ends[i] - silence_starts[i]
                if length > longest_silence:
                    longest_silence = length
                    best_start = silence_starts[i]
                    best_end = silence_ends[i]

            if longest_silence > 60:  # At least 1 min of silence = likely halftime
                logger.info(
                    "Detected halftime via silence: %.0f-%.0f (%.0f min)",
                    best_start, best_end, (best_end - best_start) / 60,
                )
                return (best_start, best_end)

    except Exception as e:
        logger.warning("Silence detection failed: %s", e)

    # Fallback: heuristic split
    ht_start = mid_point
    ht_end = mid_point + halftime_duration
    logger.info(
        "Using heuristic halftime: %.0f-%.0f (%.0f min)",
        ht_start, ht_end, halftime_duration / 60,
    )
    return (ht_start, ht_end)


async def split_match(
    file_path: str,
    match_id: str,
) -> list[dict]:
    """Split a match video into halves, skipping halftime.

    Returns list of dicts: [{"path": str, "label": str, "start": float, "end": float}, ...]
    """
    duration = await get_video_duration(file_path)
    logger.info("Video duration: %.0f sec (%.0f min)", duration, duration / 60)

    # If under 55 min, no split needed
    if duration <= 3300:
        logger.info("Video under 55 min — no split needed")
        return [{"path": file_path, "label": "Full Match", "start": 0, "end": duration}]

    # Detect halftime
    ht_start, ht_end = await detect_halftime(file_path, duration)

    halves = []
    base_dir = os.path.dirname(file_path)
    ext = os.path.splitext(file_path)[1]

    # First half: 0 to halftime_start
    h1_path = os.path.join(base_dir, f"{match_id}_h1{ext}")
    logger.info("Extracting first half: 0 - %.0f sec", ht_start)
    proc = await asyncio.create_subprocess_exec(
        "ffmpeg", "-y", "-i", file_path,
        "-t", str(int(ht_start)),
        "-c", "copy",  # No re-encoding = fast
        h1_path,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    await proc.communicate()
    if os.path.exists(h1_path) and os.path.getsize(h1_path) > 0:
        halves.append({
            "path": h1_path,
            "label": "First Half",
            "start": 0,
            "end": ht_start,
        })

    # Second half: halftime_end to end
    h2_path = os.path.join(base_dir, f"{match_id}_h2{ext}")
    logger.info("Extracting second half: %.0f - %.0f sec", ht_end, duration)
    proc = await asyncio.create_subprocess_exec(
        "ffmpeg", "-y", "-i", file_path,
        "-ss", str(int(ht_end)),
        "-c", "copy",
        h2_path,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    await proc.communicate()
    if os.path.exists(h2_path) and os.path.getsize(h2_path) > 0:
        halves.append({
            "path": h2_path,
            "label": "Second Half",
            "start": ht_end,
            "end": duration,
        })

    logger.info("Split into %d halves", len(halves))
    return halves


def cleanup_files(*paths: str) -> None:
    """Delete temporary files."""
    for p in paths:
        try:
            if os.path.exists(p):
                os.remove(p)
        except OSError:
            pass
