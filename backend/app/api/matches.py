"""Match upload, listing, and management endpoints."""

import logging
import mimetypes
import os
import uuid
import asyncio
import aiofiles
import httpx
from pydantic import BaseModel
from fastapi import APIRouter, HTTPException, BackgroundTasks, UploadFile, File, Form
from app.models.schemas import MatchCreate, MatchResponse, MatchStatus
from app.services.mux_service import MuxService
from app.services.gemini_service import GeminiService
from app.services.supabase_service import SupabaseService

logger = logging.getLogger(__name__)

router = APIRouter()
mux = MuxService()
gemini = GeminiService()
db = SupabaseService()

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

MAX_UPLOAD_SIZE = 30 * 1024 * 1024 * 1024  # 30GB — fits 4K Xbot Go / VEO full matches


@router.post("/upload")
async def upload_match(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    title: str = Form(...),
    opponent: str = Form(""),
    formation: str = Form(""),
    notes: str = Form(""),
):
    """Upload a match video file. Handles Mux + Gemini analysis in background."""
    match_id = str(uuid.uuid4())

    # Save file locally using chunked streaming (handles large files)
    file_ext = os.path.splitext(video.filename or "video.mp4")[1]
    content_type = video.content_type or mimetypes.guess_type(video.filename or "")[0] or "video/mp4"
    local_path = os.path.join(UPLOAD_DIR, f"{match_id}{file_ext}")

    total_size = 0
    too_large = False
    async with aiofiles.open(local_path, "wb") as f:
        while chunk := await video.read(1024 * 1024):  # 1MB chunks
            total_size += len(chunk)
            if total_size > MAX_UPLOAD_SIZE:
                too_large = True
                break
            await f.write(chunk)
    if too_large:
        try:
            os.remove(local_path)
        except OSError:
            pass
        raise HTTPException(413, f"File too large. Maximum {MAX_UPLOAD_SIZE // (1024**3)}GB.")

    logger.info("Saved %s (%d MB) to %s", video.filename, total_size // (1024 * 1024), local_path)

    record = await db.insert("matches", {
        "id": match_id,
        "title": title,
        "opponent": opponent or None,
        "formation": formation or None,
        "notes": notes or None,
        "status": MatchStatus.processing,
    })

    background_tasks.add_task(process_video_pipeline, match_id, local_path, title, content_type)
    return record


class UrlUploadRequest(BaseModel):
    video_url: str
    title: str
    opponent: str | None = None
    formation: str | None = None
    notes: str | None = None


@router.post("/upload-url")
async def upload_match_from_url(request: UrlUploadRequest, background_tasks: BackgroundTasks):
    """Upload a match from a direct video URL (e.g. VEO download link).
    Downloads to disk, pushes to Mux for playback, then analyses with Gemini."""
    url = request.video_url.strip()
    if not url.startswith("http://") and not url.startswith("https://"):
        raise HTTPException(400, "URL must start with http:// or https://")
    match_id = str(uuid.uuid4())

    record = await db.insert("matches", {
        "id": match_id,
        "title": request.title,
        "opponent": request.opponent,
        "formation": request.formation,
        "notes": request.notes,
        "status": MatchStatus.processing,
    })

    background_tasks.add_task(
        process_url_pipeline, match_id, url, request.title
    )
    return record


def _friendly_error(e: Exception) -> str:
    """Turn an exception into a user-facing error message.

    For httpx.HTTPStatusError, pull the upstream JSON `message`/`detail` body
    instead of the generic URL-only string FastAPI logs by default.
    """
    if isinstance(e, httpx.HTTPStatusError):
        try:
            body = e.response.json()
            msg = body.get("message") or body.get("detail") or body.get("error") or ""
            code = body.get("code") or ""
            if msg:
                return f"{code}: {msg}" if code else msg
        except Exception:
            pass
        return f"{e.response.status_code} from {e.request.url.host}: {e.response.text[:200]}"
    msg = str(e) or e.__class__.__name__
    return msg


async def process_video_pipeline(match_id: str, local_path: str, title: str, content_type: str):
    """Background: upload to Mux for streaming + analyse with Gemini 2.5 Flash.

    Long matches (>55 min) are quarter-split before analysis so each segment
    fits within Gemini's 2GB File API limit and respects its ~250K tokens/min
    rate window (handled inside GeminiService.analyse_segments).
    """
    from app.services.video_splitter import split_match, cleanup_files, get_video_duration

    files_to_clean = [local_path]

    try:
        # Step 1: Upload to Mux via streaming (no full-file memory load)
        upload = await mux.create_upload()
        file_size = os.path.getsize(local_path)
        logger.info("Uploading %d MB to Mux for match %s", file_size // (1024 * 1024), match_id)

        async with httpx.AsyncClient(timeout=httpx.Timeout(3600.0, connect=30.0)) as client:
            async def stream_file():
                async with aiofiles.open(local_path, "rb") as f:
                    while chunk := await f.read(4 * 1024 * 1024):  # 4MB chunks
                        yield chunk

            resp = await client.put(
                upload["upload_url"],
                content=stream_file(),
                headers={
                    "Content-Type": content_type,
                    "Content-Length": str(file_size),
                },
            )
            resp.raise_for_status()

        logger.info("Mux upload complete for match %s, waiting for asset", match_id)

        # Read duration locally — don't depend on Mux for it
        duration = await get_video_duration(local_path)
        logger.info("Video duration: %.0f sec (%.0f min) for match %s", duration, duration / 60, match_id)

        # Wait for Mux to process the upload into an asset
        asset_id = None
        for _ in range(60):
            asset_id = await mux.get_asset_from_upload(upload["upload_id"])
            if asset_id:
                break
            await asyncio.sleep(3)

        if asset_id:
            for _ in range(120):
                asset = await mux.get_asset(asset_id)
                if asset["status"] == "ready":
                    await db.update("matches", match_id, {
                        "mux_asset_id": asset["asset_id"],
                        "mux_playback_id": asset["playback_id"],
                        "duration_seconds": duration,
                        "thumbnail_url": asset.get("thumbnail_url"),
                    })
                    logger.info("Mux asset ready for match %s: %s", match_id, asset["asset_id"])
                    break
                if asset["status"] == "errored":
                    logger.error("Mux asset errored for match %s", match_id)
                    break
                await asyncio.sleep(3)
            else:
                logger.warning("Mux asset polling timed out for match %s", match_id)
                await db.update("matches", match_id, {"duration_seconds": duration})
        else:
            logger.warning("Mux upload->asset resolution timed out for match %s", match_id)
            await db.update("matches", match_id, {"duration_seconds": duration})

        # Step 2: Analyse with Gemini — split first if the match exceeds the single-video limit
        await db.update("matches", match_id, {"status": MatchStatus.analysing})
        logger.info("Starting Gemini analysis for match %s", match_id)

        # 55-minute ceiling per segment so file size stays under Gemini's 2GB cap
        # and token counts stay under the hourly window.
        segments = await split_match(local_path, match_id)
        files_to_clean.extend(s["path"] for s in segments if s["path"] != local_path)
        logger.info("Match %s split into %d segment(s)", match_id, len(segments))

        result = await gemini.analyse_segments(
            segments=segments,
            match_id=match_id,
            db=db,
            match_title=title,
        )

        if result["analyses_created"] > 0:
            await db.update("matches", match_id, {"status": MatchStatus.ready})
            logger.info(
                "Match %s READY — %d Gemini analyses saved (%d errors)",
                match_id, result["analyses_created"], len(result["errors"]),
            )
        else:
            await db.update("matches", match_id, {
                "status": MatchStatus.failed,
                "error_message": "Gemini analysis produced no results: " + "; ".join(result["errors"][:3]),
            })
            logger.error("Match %s FAILED — no analyses saved. Errors: %s", match_id, result["errors"])

    except Exception as e:
        logger.exception("Pipeline failed for match %s", match_id)
        await db.update("matches", match_id, {
            "status": MatchStatus.failed,
            "error_message": _friendly_error(e),
        })
    finally:
        files_to_clean.extend(
            os.path.join(UPLOAD_DIR, f)
            for f in os.listdir(UPLOAD_DIR)
            if f.startswith(match_id)
        )
        cleanup_files(*files_to_clean)


async def process_url_pipeline(match_id: str, video_url: str, title: str):
    """Background: download video, upload to Mux, split and analyse with Gemini."""
    from app.services.video_splitter import download_video, split_match, cleanup_files, get_video_duration

    local_path = os.path.join(UPLOAD_DIR, f"{match_id}_full.mp4")
    files_to_clean = [local_path]

    try:
        # Step 1: Download video locally
        logger.info("Downloading video for match %s", match_id)
        await download_video(video_url, local_path)
        file_size = os.path.getsize(local_path)
        logger.info("Downloaded %.0f MB for match %s", file_size / (1024 * 1024), match_id)

        # Get duration from the file directly (don't rely on Mux for this)
        duration = await get_video_duration(local_path)
        logger.info("Video duration: %.0f sec (%.0f min) for match %s", duration, duration / 60, match_id)

        # Step 2: Upload to Mux from local file using curl (reliable for large files)
        logger.info("Uploading %.0f MB to Mux for match %s", file_size / (1024*1024), match_id)
        upload = await mux.create_upload()

        proc = await asyncio.create_subprocess_exec(
            "curl", "-s", "-o", "/dev/null", "-w", "%{http_code}",
            "--upload-file", local_path,
            "-H", "Content-Type: video/mp4",
            upload["upload_url"],
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        http_code = stdout.decode().strip()[-3:]  # Last 3 chars are the HTTP code
        logger.info("Mux curl upload returned HTTP %s for match %s", http_code, match_id)

        if not http_code.startswith("2"):
            logger.error("Mux upload failed with HTTP %s", http_code)
            # Don't abort — continue with Gemini analysis, Mux can be retried later

        # Poll until Mux asset is ready
        mux_ready = False
        asset_id = None
        for _ in range(60):
            asset_id = await mux.get_asset_from_upload(upload["upload_id"])
            if asset_id:
                break
            await asyncio.sleep(5)

        if asset_id:
            for _ in range(120):
                asset = await mux.get_asset(asset_id)
                if asset["status"] == "ready":
                    mux_ready = True
                    await db.update("matches", match_id, {
                        "mux_asset_id": asset["asset_id"],
                        "mux_playback_id": asset["playback_id"],
                        "duration_seconds": duration,
                        "thumbnail_url": asset.get("thumbnail_url"),
                    })
                    logger.info("Mux asset ready for match %s", match_id)
                    break
                if asset["status"] == "errored":
                    logger.error("Mux asset errored for match %s", match_id)
                    break
                await asyncio.sleep(5)

        if not mux_ready:
            logger.warning("Mux not ready for match %s — continuing with Gemini anyway", match_id)
            await db.update("matches", match_id, {"duration_seconds": duration})

        # Step 3: Analyse with Gemini
        await db.update("matches", match_id, {"status": MatchStatus.analysing})
        logger.info("Starting Gemini analysis for match %s", match_id)

        segments = await split_match(local_path, match_id)
        files_to_clean.extend(s["path"] for s in segments if s["path"] != local_path)
        logger.info("Match %s split into %d segment(s)", match_id, len(segments))

        result = await gemini.analyse_segments(
            segments=segments,
            match_id=match_id,
            db=db,
            match_title=title,
        )

        if result["analyses_created"] > 0:
            await db.update("matches", match_id, {"status": MatchStatus.ready})
            logger.info(
                "Match %s READY — %d Gemini analyses saved (%d errors)",
                match_id, result["analyses_created"], len(result["errors"]),
            )
        else:
            await db.update("matches", match_id, {
                "status": MatchStatus.failed,
                "error_message": "Gemini analysis produced no results: " + "; ".join(result["errors"][:3]),
            })
            logger.error("Match %s FAILED — no analyses saved. Errors: %s", match_id, result["errors"])

    except Exception as e:
        logger.exception("URL pipeline failed for match %s", match_id)
        await db.update("matches", match_id, {
            "status": MatchStatus.failed,
            "error_message": _friendly_error(e),
        })
    finally:
        files_to_clean.extend(
            os.path.join(UPLOAD_DIR, f)
            for f in os.listdir(UPLOAD_DIR)
            if f.startswith(match_id)
        )
        cleanup_files(*files_to_clean)


@router.post("", response_model=MatchResponse)
async def create_match(match: MatchCreate, background_tasks: BackgroundTasks):
    """Create a new match record (metadata only, no video)."""
    match_id = str(uuid.uuid4())
    record = await db.insert("matches", {
        "id": match_id,
        "title": match.title,
        "opponent": match.opponent,
        "match_date": match.match_date.isoformat() if match.match_date else None,
        "formation": match.formation,
        "notes": match.notes,
        "status": MatchStatus.uploading,
    })
    return record


@router.post("/{match_id}/process")
async def process_match(match_id: str, mux_upload_id: str, background_tasks: BackgroundTasks):
    """After Mux direct upload completes, start processing pipeline."""
    match = await db.select_one("matches", match_id)
    if not match:
        raise HTTPException(404, "Match not found")

    asset_id = await mux.get_asset_from_upload(mux_upload_id)
    if not asset_id:
        raise HTTPException(400, "Upload not yet complete")

    asset = await mux.get_asset(asset_id)

    await db.update("matches", match_id, {
        "mux_asset_id": asset["asset_id"],
        "mux_playback_id": asset["playback_id"],
        "duration_seconds": asset.get("duration"),
        "thumbnail_url": asset.get("thumbnail_url"),
        "status": MatchStatus.processing,
    })

    return {"status": "processing", "match_id": match_id, "asset": asset}


@router.get("/{match_id}/status")
async def get_match_status(match_id: str):
    """Return current match processing status — the background pipeline drives transitions."""
    match = await db.select_one("matches", match_id)
    if not match:
        raise HTTPException(404, "Match not found")
    return match


@router.get("")
async def list_matches(include_archived: bool = False):
    """List matches. Archived (legacy TwelveLabs-only) matches are hidden by default."""
    all_matches = await db.select("matches")
    if include_archived:
        return all_matches
    return [m for m in all_matches if m.get("status") != MatchStatus.archived]


@router.get("/{match_id}/report")
async def download_match_report(match_id: str):
    """Generate and download a PDF match report."""
    from fastapi.responses import Response
    from app.services.pdf_report import generate_pdf

    match = await db.select_one("matches", match_id)
    if not match:
        raise HTTPException(404, "Match not found")

    analyses = await db.select("analyses", f"match_id=eq.{match_id}&status=eq.complete")
    if not analyses:
        raise HTTPException(400, "No completed analysis found. Run an analysis first.")

    pdf_bytes = await generate_pdf(match, analyses)
    filename = f"match-report-{match.get('title', 'report').replace(' ', '-').lower()}.pdf"

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.patch("/{match_id}")
async def update_match(match_id: str, updates: dict):
    """Update match fields (team_color, opponent_color, formation, notes, etc.)."""
    allowed = {"team_color", "opponent_color", "formation", "notes", "opponent", "title"}
    filtered = {k: v for k, v in updates.items() if k in allowed}
    if not filtered:
        raise HTTPException(400, "No valid fields to update")
    match = await db.select_one("matches", match_id)
    if not match:
        raise HTTPException(404, "Match not found")
    await db.update("matches", match_id, filtered)
    return {**match, **filtered}


@router.get("/{match_id}")
async def get_match(match_id: str):
    """Get a single match."""
    match = await db.select_one("matches", match_id)
    if not match:
        raise HTTPException(404, "Match not found")
    return match
