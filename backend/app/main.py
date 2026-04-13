"""Manager Mentor v2 — FastAPI Application"""

import sys
import asyncio
import logging

# Windows + Python 3.14 defaults to SelectorEventLoop which cannot spawn
# subprocesses (ffprobe/ffmpeg). Force the Proactor policy before any loop
# is created so the video pipeline can run.
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

# Route app loggers (app.api.matches, app.services.gemini_service, etc.) to
# stdout at INFO level so background-task progress is visible in uvicorn's log.
# Without this the root logger has no handler and every logger.info() is dropped.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    stream=sys.stdout,
    force=True,
)

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import get_settings
from app.api.matches import router as matches_router
from app.api.analysis import router as analysis_router
from app.api.search import router as search_router
from app.api.players import router as players_router
from app.api.health import router as health_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    settings = get_settings()
    print(f"Starting {settings.app_name} v{settings.app_version}")
    yield
    print("Shutting down...")


app = FastAPI(
    title="Manager Mentor",
    version="2.0.0",
    description="AI-powered football video analysis for grassroots coaches",
    lifespan=lifespan,
)

# CORS — permissive for local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
app.include_router(health_router, prefix="/api", tags=["health"])
app.include_router(matches_router, prefix="/api/matches", tags=["matches"])
app.include_router(analysis_router, prefix="/api/analysis", tags=["analysis"])
app.include_router(search_router, prefix="/api/search", tags=["search"])
app.include_router(players_router, prefix="/api/players", tags=["players"])
