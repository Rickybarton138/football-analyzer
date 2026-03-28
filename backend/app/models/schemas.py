"""Pydantic schemas for Manager Mentor v2."""

from datetime import datetime
from pydantic import BaseModel
from enum import Enum


class MatchStatus(str, Enum):
    uploading = "uploading"
    processing = "processing"
    indexing = "indexing"
    analysing = "analysing"
    ready = "ready"
    failed = "failed"


class AnalysisType(str, Enum):
    full = "full"
    highlights = "highlights"
    tactical = "tactical"
    player_spotlight = "player_spotlight"


# --- Match ---
class MatchCreate(BaseModel):
    title: str
    opponent: str | None = None
    match_date: datetime | None = None
    formation: str | None = None
    notes: str | None = None


class MatchResponse(BaseModel):
    id: str
    title: str
    opponent: str | None
    match_date: datetime | None
    formation: str | None
    status: MatchStatus
    mux_asset_id: str | None
    mux_playback_id: str | None
    twelvelabs_video_id: str | None
    duration_seconds: float | None
    thumbnail_url: str | None
    created_at: datetime


# --- Analysis ---
class AnalysisRequest(BaseModel):
    match_id: str
    analysis_type: AnalysisType = AnalysisType.full
    prompt: str | None = None


class AnalysisResponse(BaseModel):
    id: str
    match_id: str
    analysis_type: AnalysisType
    status: str
    summary: str | None
    highlights: list[dict] | None
    tactical_insights: list[dict] | None
    coaching_advice: str | None
    created_at: datetime


# --- Search ---
class SearchRequest(BaseModel):
    query: str
    match_id: str | None = None


class SearchResult(BaseModel):
    match_id: str
    match_title: str
    start_time: float
    end_time: float
    confidence: float
    thumbnail_url: str | None
    clip_url: str | None


# --- Player ---
class PlayerCreate(BaseModel):
    name: str
    squad_number: int | None = None
    position: str | None = None


class PlayerResponse(BaseModel):
    id: str
    name: str
    squad_number: int | None
    position: str | None
    twelvelabs_entity_id: str | None
    created_at: datetime


# --- Highlight ---
class Highlight(BaseModel):
    title: str
    start_time: float
    end_time: float
    event_type: str
    description: str | None
    clip_url: str | None
    thumbnail_url: str | None
