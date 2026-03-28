"""Match upload, listing, and management endpoints."""

import os
import uuid
import asyncio
import aiofiles
from fastapi import APIRouter, HTTPException, BackgroundTasks, UploadFile, File, Form
from app.models.schemas import MatchCreate, MatchResponse, MatchStatus
from app.services.mux_service import MuxService
from app.services.twelvelabs_service import TwelveLabsService
from app.services.supabase_service import SupabaseService

router = APIRouter()
mux = MuxService()
twelvelabs = TwelveLabsService()
db = SupabaseService()

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


@router.post("/upload-url")
async def get_upload_url():
    """Get a Mux direct upload URL for the frontend."""
    upload = await mux.create_upload()
    return upload


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
    local_path = os.path.join(UPLOAD_DIR, f"{match_id}{file_ext}")
    async with aiofiles.open(local_path, "wb") as f:
        while chunk := await video.read(1024 * 1024):  # 1MB chunks
            await f.write(chunk)

    # Create match record
    record = await db.insert("matches", {
        "id": match_id,
        "title": title,
        "opponent": opponent or None,
        "formation": formation or None,
        "notes": notes or None,
        "status": MatchStatus.processing,
    })

    # Process in background: Mux upload + TwelveLabs indexing
    background_tasks.add_task(process_video_pipeline, match_id, local_path, title)

    return record


async def process_video_pipeline(match_id: str, local_path: str, title: str):
    """Background: upload to Mux for streaming + index with TwelveLabs for analysis."""
    try:
        # Step 1: Upload to Mux
        upload = await mux.create_upload()
        import httpx
        async with httpx.AsyncClient(timeout=600) as client:
            with open(local_path, "rb") as f:
                resp = await client.put(
                    upload["upload_url"],
                    content=f,
                    headers={"Content-Type": "video/mp4"},
                )

        # Wait for Mux to process the upload into an asset
        asset_id = None
        for _ in range(60):
            asset_id = await mux.get_asset_from_upload(upload["upload_id"])
            if asset_id:
                break
            await asyncio.sleep(3)

        if asset_id:
            # Wait for asset to be ready
            for _ in range(60):
                asset = await mux.get_asset(asset_id)
                if asset["status"] == "ready":
                    await db.update("matches", match_id, {
                        "mux_asset_id": asset["asset_id"],
                        "mux_playback_id": asset["playback_id"],
                        "duration_seconds": asset.get("duration"),
                        "thumbnail_url": asset.get("thumbnail_url"),
                    })
                    break
                await asyncio.sleep(3)

        # Step 2: Index with TwelveLabs from local file
        await db.update("matches", match_id, {"status": MatchStatus.indexing})
        result = await twelvelabs.index_video_from_file(local_path, title)
        await db.update("matches", match_id, {
            "twelvelabs_task_id": result["task_id"],
            "twelvelabs_video_id": result.get("video_id"),
        })

    except Exception as e:
        await db.update("matches", match_id, {
            "status": MatchStatus.failed,
            "error_message": str(e),
        })
    finally:
        # Clean up local file
        try:
            os.remove(local_path)
        except OSError:
            pass


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


@router.get("/{match_id}")
async def get_match(match_id: str):
    """Get a single match."""
    match = await db.select_one("matches", match_id)
    if not match:
        raise HTTPException(404, "Match not found")
    return match
