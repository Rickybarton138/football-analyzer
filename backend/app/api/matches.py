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
from app.services.twelvelabs_service import TwelveLabsService
from app.services.supabase_service import SupabaseService

logger = logging.getLogger(__name__)

router = APIRouter()
mux = MuxService()
twelvelabs = TwelveLabsService()
db = SupabaseService()

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

MAX_UPLOAD_SIZE = 2 * 1024 * 1024 * 1024  # 2GB


@router.post("/upload")
async def upload_match(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    title: str = Form(...),
    opponent: str = Form(""),
    formation: str = Form(""),
    notes: str = Form(""),
):
    """Upload a match video file. Handles Mux + TwelveLabs in background."""
    match_id = str(uuid.uuid4())

    # Save file locally using chunked streaming (handles large files)
    file_ext = os.path.splitext(video.filename or "video.mp4")[1]
    content_type = video.content_type or mimetypes.guess_type(video.filename or "")[0] or "video/mp4"
    local_path = os.path.join(UPLOAD_DIR, f"{match_id}{file_ext}")

    total_size = 0
    async with aiofiles.open(local_path, "wb") as f:
        while chunk := await video.read(1024 * 1024):  # 1MB chunks
            total_size += len(chunk)
            if total_size > MAX_UPLOAD_SIZE:
                os.remove(local_path)
                raise HTTPException(413, "File too large. Maximum 2GB.")
            await f.write(chunk)

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
    Sends URL directly to Mux + TwelveLabs — no local file needed."""
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


async def process_video_pipeline(match_id: str, local_path: str, title: str, content_type: str):
    """Background: upload to Mux for streaming + index with TwelveLabs for analysis."""
    try:
        # Step 1: Upload to Mux via streaming (no full-file memory load)
        upload = await mux.create_upload()
        file_size = os.path.getsize(local_path)
        logger.info("Uploading %d MB to Mux for match %s", file_size // (1024 * 1024), match_id)

        async with httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=30.0)) as client:
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

        # Wait for Mux to process the upload into an asset
        asset_id = None
        for _ in range(60):
            asset_id = await mux.get_asset_from_upload(upload["upload_id"])
            if asset_id:
                break
            await asyncio.sleep(3)

        if asset_id:
            for _ in range(60):
                asset = await mux.get_asset(asset_id)
                if asset["status"] == "ready":
                    await db.update("matches", match_id, {
                        "mux_asset_id": asset["asset_id"],
                        "mux_playback_id": asset["playback_id"],
                        "duration_seconds": asset.get("duration"),
                        "thumbnail_url": asset.get("thumbnail_url"),
                    })
                    logger.info("Mux asset ready for match %s: %s", match_id, asset["asset_id"])
                    break
                await asyncio.sleep(3)
            else:
                logger.warning("Mux asset polling timed out for match %s", match_id)
        else:
            logger.warning("Mux upload->asset resolution timed out for match %s", match_id)

        # Step 2: Index with TwelveLabs from local file
        await db.update("matches", match_id, {"status": MatchStatus.indexing})
        logger.info("Starting TwelveLabs indexing for match %s", match_id)
        result = await twelvelabs.index_video_from_file(local_path, title, content_type)
        await db.update("matches", match_id, {
            "twelvelabs_task_id": result["task_id"],
            "twelvelabs_video_id": result.get("video_id"),
        })
        logger.info("TwelveLabs task submitted for match %s: %s", match_id, result["task_id"])

        # Step 3: Poll TwelveLabs until ready (so status transitions autonomously)
        for _ in range(120):  # up to ~10 minutes
            await asyncio.sleep(5)
            task = await twelvelabs.get_task_status(result["task_id"])
            if task["status"] == "ready":
                await db.update("matches", match_id, {
                    "status": MatchStatus.ready,
                    "twelvelabs_video_id": task["video_id"],
                })
                logger.info("Match %s is READY", match_id)
                return
            elif task["status"] == "failed":
                await db.update("matches", match_id, {
                    "status": MatchStatus.failed,
                    "error_message": "TwelveLabs indexing failed",
                })
                logger.error("TwelveLabs indexing failed for match %s", match_id)
                return

        # If we get here, polling timed out
        await db.update("matches", match_id, {
            "status": MatchStatus.failed,
            "error_message": "TwelveLabs indexing timed out after 10 minutes",
        })
        logger.error("TwelveLabs polling timed out for match %s", match_id)

    except Exception as e:
        logger.exception("Pipeline failed for match %s", match_id)
        await db.update("matches", match_id, {
            "status": MatchStatus.failed,
            "error_message": str(e),
        })
    finally:
        try:
            os.remove(local_path)
            logger.info("Cleaned up local file for match %s", match_id)
        except OSError:
            pass


async def process_url_pipeline(match_id: str, video_url: str, title: str):
    """Background: download video, upload to Mux, split for TwelveLabs."""
    from app.services.video_splitter import download_video, split_match, cleanup_files

    local_path = os.path.join(UPLOAD_DIR, f"{match_id}_full.mp4")
    files_to_clean = [local_path]

    try:
        # Step 1: Download video locally
        logger.info("Downloading video for match %s", match_id)
        await download_video(video_url, local_path)
        file_size = os.path.getsize(local_path)
        logger.info("Downloaded %.0f MB for match %s", file_size / (1024 * 1024), match_id)

        # Step 2: Upload to Mux from local file (proven reliable)
        logger.info("Uploading to Mux for match %s", match_id)
        upload = await mux.create_upload()

        async with httpx.AsyncClient(timeout=httpx.Timeout(1800.0, connect=30.0)) as client:
            async def stream_file():
                async with aiofiles.open(local_path, "rb") as f:
                    while chunk := await f.read(4 * 1024 * 1024):
                        yield chunk

            resp = await client.put(
                upload["upload_url"],
                content=stream_file(),
                headers={
                    "Content-Type": "video/mp4",
                    "Content-Length": str(file_size),
                },
            )
            resp.raise_for_status()

        logger.info("Mux upload complete for match %s, polling for asset", match_id)

        # Poll until Mux asset is ready
        asset_id = None
        duration = None
        for _ in range(180):
            asset_id = await mux.get_asset_from_upload(upload["upload_id"])
            if asset_id:
                break
            await asyncio.sleep(3)

        if asset_id:
            for _ in range(180):
                asset = await mux.get_asset(asset_id)
                if asset["status"] == "ready":
                    duration = asset.get("duration")
                    await db.update("matches", match_id, {
                        "mux_asset_id": asset["asset_id"],
                        "mux_playback_id": asset["playback_id"],
                        "duration_seconds": duration,
                        "thumbnail_url": asset.get("thumbnail_url"),
                    })
                    logger.info("Mux asset ready for match %s (%.0f min)", match_id, (duration or 0) / 60)
                    break
                if asset["status"] == "errored":
                    logger.error("Mux asset errored for match %s", match_id)
                    break
                await asyncio.sleep(3)

        # Step 3: Index with TwelveLabs
        await db.update("matches", match_id, {"status": MatchStatus.indexing})

        max_tl_duration = 3300  # 55 min

        if duration and duration > max_tl_duration:
            # Split into halves, skip halftime
            logger.info("Video is %d min — splitting for TwelveLabs", int(duration // 60))
            halves = await split_match(local_path, match_id)
            files_to_clean.extend(h["path"] for h in halves if h["path"] != local_path)

            video_ids = []
            for half in halves:
                logger.info("Indexing %s for match %s", half["label"], match_id)
                result = await twelvelabs.index_video_from_file(
                    half["path"],
                    f"{title} - {half['label']}",
                    "video/mp4",
                )
                for _ in range(180):
                    await asyncio.sleep(5)
                    task = await twelvelabs.get_task_status(result["task_id"])
                    if task["status"] == "ready":
                        video_ids.append(task["video_id"])
                        logger.info("%s indexed: %s", half["label"], task["video_id"])
                        break
                    elif task["status"] == "failed":
                        logger.error("%s indexing failed", half["label"])
                        break

            if video_ids:
                await db.update("matches", match_id, {
                    "status": MatchStatus.ready,
                    "twelvelabs_video_id": video_ids[0],
                    "twelvelabs_task_id": ",".join(video_ids),
                })
                logger.info("Match %s READY — %d halves indexed", match_id, len(video_ids))
            else:
                await db.update("matches", match_id, {
                    "status": MatchStatus.failed,
                    "error_message": "Failed to index match halves with TwelveLabs",
                })
        else:
            # Short video — index the whole file
            result = await twelvelabs.index_video_from_file(local_path, title, "video/mp4")
            await db.update("matches", match_id, {
                "twelvelabs_task_id": result["task_id"],
                "twelvelabs_video_id": result.get("video_id"),
            })

            for _ in range(180):
                await asyncio.sleep(5)
                task = await twelvelabs.get_task_status(result["task_id"])
                if task["status"] == "ready":
                    await db.update("matches", match_id, {
                        "status": MatchStatus.ready,
                        "twelvelabs_video_id": task["video_id"],
                    })
                    logger.info("Match %s is READY", match_id)
                    return
                elif task["status"] == "failed":
                    await db.update("matches", match_id, {
                        "status": MatchStatus.failed,
                        "error_message": "TwelveLabs indexing failed",
                    })
                    return

            await db.update("matches", match_id, {
                "status": MatchStatus.failed,
                "error_message": "TwelveLabs indexing timed out",
            })

    except Exception as e:
        logger.exception("URL pipeline failed for match %s", match_id)
        await db.update("matches", match_id, {
            "status": MatchStatus.failed,
            "error_message": str(e),
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
    """Check match processing status, including TwelveLabs indexing."""
    match = await db.select_one("matches", match_id)
    if not match:
        raise HTTPException(404, "Match not found")

    if match.get("status") == MatchStatus.indexing and match.get("twelvelabs_task_id"):
        task = await twelvelabs.get_task_status(match["twelvelabs_task_id"])
        if task["status"] == "ready":
            await db.update("matches", match_id, {
                "status": MatchStatus.ready,
                "twelvelabs_video_id": task["video_id"],
            })
            match["status"] = MatchStatus.ready
            match["twelvelabs_video_id"] = task["video_id"]
        elif task["status"] == "failed":
            await db.update("matches", match_id, {"status": MatchStatus.failed})
            match["status"] = MatchStatus.failed

    return match


@router.get("")
async def list_matches():
    """List all matches."""
    return await db.select("matches")


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


@router.get("/{match_id}")
async def get_match(match_id: str):
    """Get a single match."""
    match = await db.select_one("matches", match_id)
    if not match:
        raise HTTPException(404, "Match not found")
    return match
