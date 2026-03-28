"""Player management — squad registration and entity recognition."""

from fastapi import APIRouter, HTTPException
from app.models.schemas import PlayerCreate, PlayerResponse
from app.services.twelvelabs_service import TwelveLabsService
from app.services.supabase_service import SupabaseService
import uuid

router = APIRouter()
twelvelabs = TwelveLabsService()
db = SupabaseService()


@router.post("")
async def create_player(player: PlayerCreate):
    """Register a player in the squad."""
    player_id = str(uuid.uuid4())
    record = await db.insert("players", {
        "id": player_id,
        "name": player.name,
        "squad_number": player.squad_number,
        "position": player.position,
    })
    return record


@router.get("")
async def list_players():
    """List all players in the squad."""
    return await db.select("players", order="squad_number.asc.nullslast")


@router.post("/{player_id}/register-face")
async def register_player_face(player_id: str, image_urls: list[str]):
    """Register player reference images for TwelveLabs entity recognition."""
    player = await db.select_one("players", player_id)
    if not player:
        raise HTTPException(404, "Player not found")

    # Ensure entity collection exists
    collection_id = await get_or_create_squad_collection()

    # Add player as entity
    entity_id = await twelvelabs.add_player_entity(
        collection_id, player["name"], image_urls
    )

    await db.update("players", player_id, {
        "twelvelabs_entity_id": entity_id,
    })

    return {"player_id": player_id, "entity_id": entity_id, "images_registered": len(image_urls)}


@router.get("/{player_id}/moments")
async def get_player_moments(player_id: str, match_id: str | None = None):
    """Find all moments featuring a specific player."""
    player = await db.select_one("players", player_id)
    if not player:
        raise HTTPException(404, "Player not found")

    if not player.get("twelvelabs_entity_id"):
        raise HTTPException(400, "Player has no registered face. Upload reference images first.")

    # Search for this player's appearances
    query = f"player {player['name']}"
    video_id = None
    if match_id:
        match = await db.select_one("matches", match_id)
        video_id = match.get("twelvelabs_video_id") if match else None

    results = await twelvelabs.search_moments(query, video_id)
    return {"player": player["name"], "moments": results}


async def get_or_create_squad_collection() -> str:
    """Get or create the squad entity collection."""
    settings = await db.select("settings", "key=eq.squad_collection_id", limit=1)
    if settings:
        return settings[0]["value"]

    collection_id = await twelvelabs.create_entity_collection("Squad")
    await db.insert("settings", {"key": "squad_collection_id", "value": collection_id})
    return collection_id
