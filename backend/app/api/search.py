"""Search endpoints — natural language search across match footage."""

from fastapi import APIRouter
from app.models.schemas import SearchRequest
from app.services.twelvelabs_service import TwelveLabsService
from app.services.mux_service import MuxService
from app.services.supabase_service import SupabaseService

router = APIRouter()
twelvelabs = TwelveLabsService()
mux = MuxService()
db = SupabaseService()


@router.post("")
async def search_footage(request: SearchRequest):
    """Search across all indexed match footage with natural language."""
    # Get video_id filter if match_id provided
    video_id = None
    match_data = None
    if request.match_id:
        match_data = await db.select_one("matches", request.match_id)
        if match_data:
            video_id = match_data.get("twelvelabs_video_id")

    results = await twelvelabs.search_moments(request.query, video_id)

    # Enrich results with Mux clip URLs and match info
    enriched = []
    for r in results:
        # Find the match for this video
        if not match_data or match_data.get("twelvelabs_video_id") != r["video_id"]:
            matches = await db.select("matches", f"twelvelabs_video_id=eq.{r['video_id']}", limit=1)
            match_data = matches[0] if matches else None

        playback_id = match_data.get("mux_playback_id") if match_data else None

        enriched.append({
            "match_id": match_data["id"] if match_data else None,
            "match_title": match_data["title"] if match_data else "Unknown",
            "start_time": r["start"],
            "end_time": r["end"],
            "confidence": r["confidence"],
            "clip_url": mux.get_clip_url(playback_id, r["start"], r["end"]) if playback_id else None,
            "thumbnail_url": mux.get_thumbnail_at(playback_id, r["start"]) if playback_id else None,
            "gif_url": mux.get_gif(playback_id, r["start"], r["end"]) if playback_id else None,
        })

    return {"query": request.query, "results": enriched, "total": len(enriched)}
