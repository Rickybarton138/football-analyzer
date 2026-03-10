"""
SQLite persistence layer for Manager Mentor.
Uses raw aiosqlite — no ORM.
"""
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Any

import aiosqlite

from config import settings

_db_path: Path = settings.DATABASE_PATH


async def _get_db() -> aiosqlite.Connection:
    db = await aiosqlite.connect(str(_db_path))
    db.row_factory = aiosqlite.Row
    await db.execute("PRAGMA journal_mode=WAL")
    await db.execute("PRAGMA foreign_keys=ON")
    return db


async def init_db():
    """Create tables if they don't exist."""
    _db_path.parent.mkdir(parents=True, exist_ok=True)
    db = await _get_db()
    try:
        await db.executescript("""
            CREATE TABLE IF NOT EXISTS videos (
                video_id    TEXT PRIMARY KEY,
                filename    TEXT,
                status      TEXT NOT NULL DEFAULT 'uploaded',
                progress_pct INTEGER NOT NULL DEFAULT 0,
                current_frame INTEGER NOT NULL DEFAULT 0,
                total_frames INTEGER NOT NULL DEFAULT 0,
                match_id    TEXT,
                match_half  TEXT,
                duration_ms INTEGER DEFAULT 0,
                fps         REAL DEFAULT 0,
                resolution  TEXT,
                error_message TEXT,
                created_at  TEXT NOT NULL DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS matches (
                match_id    TEXT PRIMARY KEY,
                home_team   TEXT,
                away_team   TEXT,
                metadata    TEXT,
                created_at  TEXT NOT NULL DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS import_jobs (
                import_id   TEXT PRIMARY KEY,
                url         TEXT,
                video_id    TEXT,
                status      TEXT NOT NULL DEFAULT 'downloading',
                progress_pct INTEGER NOT NULL DEFAULT 0,
                error       TEXT,
                extra       TEXT,
                created_at  TEXT NOT NULL DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS settings (
                key   TEXT PRIMARY KEY,
                value TEXT
            );
        """)
        await db.commit()
    finally:
        await db.close()


# ── Videos ──

async def upsert_video(
    video_id: str,
    *,
    filename: str = None,
    status: str = "uploaded",
    progress_pct: int = 0,
    current_frame: int = 0,
    total_frames: int = 0,
    match_id: str = None,
    match_half: str = None,
    duration_ms: int = 0,
    fps: float = 0,
    resolution: str = None,
    error_message: str = None,
):
    db = await _get_db()
    try:
        await db.execute(
            """INSERT INTO videos
                (video_id, filename, status, progress_pct, current_frame, total_frames,
                 match_id, match_half, duration_ms, fps, resolution, error_message)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(video_id) DO UPDATE SET
                 filename = COALESCE(excluded.filename, filename),
                 status = excluded.status,
                 progress_pct = excluded.progress_pct,
                 current_frame = excluded.current_frame,
                 total_frames = excluded.total_frames,
                 match_id = COALESCE(excluded.match_id, match_id),
                 match_half = COALESCE(excluded.match_half, match_half),
                 duration_ms = CASE WHEN excluded.duration_ms > 0 THEN excluded.duration_ms ELSE duration_ms END,
                 fps = CASE WHEN excluded.fps > 0 THEN excluded.fps ELSE fps END,
                 resolution = COALESCE(excluded.resolution, resolution),
                 error_message = excluded.error_message
            """,
            (video_id, filename, status, progress_pct, current_frame, total_frames,
             match_id, match_half, duration_ms, fps, resolution, error_message),
        )
        await db.commit()
    finally:
        await db.close()


async def get_video(video_id: str) -> Optional[Dict]:
    db = await _get_db()
    try:
        cursor = await db.execute("SELECT * FROM videos WHERE video_id = ?", (video_id,))
        row = await cursor.fetchone()
        return dict(row) if row else None
    finally:
        await db.close()


async def update_video_status(
    video_id: str,
    *,
    status: str = None,
    progress_pct: int = None,
    current_frame: int = None,
    total_frames: int = None,
    error_message: str = None,
):
    parts = []
    params = []
    if status is not None:
        parts.append("status = ?")
        params.append(status)
    if progress_pct is not None:
        parts.append("progress_pct = ?")
        params.append(progress_pct)
    if current_frame is not None:
        parts.append("current_frame = ?")
        params.append(current_frame)
    if total_frames is not None:
        parts.append("total_frames = ?")
        params.append(total_frames)
    if error_message is not None:
        parts.append("error_message = ?")
        params.append(error_message)
    if not parts:
        return
    params.append(video_id)
    db = await _get_db()
    try:
        await db.execute(f"UPDATE videos SET {', '.join(parts)} WHERE video_id = ?", params)
        await db.commit()
    finally:
        await db.close()


async def get_all_videos() -> List[Dict]:
    db = await _get_db()
    try:
        cursor = await db.execute("SELECT * FROM videos ORDER BY created_at DESC")
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]
    finally:
        await db.close()


async def video_exists(video_id: str) -> bool:
    db = await _get_db()
    try:
        cursor = await db.execute("SELECT 1 FROM videos WHERE video_id = ?", (video_id,))
        return (await cursor.fetchone()) is not None
    finally:
        await db.close()


# ── Matches ──

async def upsert_match(
    match_id: str,
    *,
    home_team: str = None,
    away_team: str = None,
    metadata: dict = None,
):
    meta_json = json.dumps(metadata) if metadata else None
    db = await _get_db()
    try:
        await db.execute(
            """INSERT INTO matches (match_id, home_team, away_team, metadata)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(match_id) DO UPDATE SET
                 home_team = COALESCE(excluded.home_team, home_team),
                 away_team = COALESCE(excluded.away_team, away_team),
                 metadata = COALESCE(excluded.metadata, metadata)
            """,
            (match_id, home_team, away_team, meta_json),
        )
        await db.commit()
    finally:
        await db.close()


async def get_match(match_id: str) -> Optional[Dict]:
    db = await _get_db()
    try:
        cursor = await db.execute("SELECT * FROM matches WHERE match_id = ?", (match_id,))
        row = await cursor.fetchone()
        if not row:
            return None
        d = dict(row)
        if d.get("metadata"):
            d["metadata"] = json.loads(d["metadata"])
        return d
    finally:
        await db.close()


async def get_videos_for_match(match_id: str) -> Dict[str, Dict]:
    """Return {"first": {...}, "second": {...}} for a match's half videos."""
    db = await _get_db()
    try:
        cursor = await db.execute(
            "SELECT * FROM videos WHERE match_id = ? AND match_half IN ('first', 'second')",
            (match_id,),
        )
        rows = await cursor.fetchall()
        result = {}
        for row in rows:
            d = dict(row)
            result[d["match_half"]] = d
        return result
    finally:
        await db.close()


# ── Import Jobs ──

async def upsert_import_job(
    import_id: str,
    *,
    url: str = None,
    video_id: str = None,
    status: str = "downloading",
    progress_pct: int = 0,
    error: str = None,
    extra: dict = None,
):
    extra_json = json.dumps(extra) if extra else None
    db = await _get_db()
    try:
        await db.execute(
            """INSERT INTO import_jobs (import_id, url, video_id, status, progress_pct, error, extra)
               VALUES (?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(import_id) DO UPDATE SET
                 url = COALESCE(excluded.url, url),
                 video_id = COALESCE(excluded.video_id, video_id),
                 status = excluded.status,
                 progress_pct = excluded.progress_pct,
                 error = excluded.error,
                 extra = COALESCE(excluded.extra, extra)
            """,
            (import_id, url, video_id, status, progress_pct, error, extra_json),
        )
        await db.commit()
    finally:
        await db.close()


async def get_import_job(import_id: str) -> Optional[Dict]:
    db = await _get_db()
    try:
        cursor = await db.execute("SELECT * FROM import_jobs WHERE import_id = ?", (import_id,))
        row = await cursor.fetchone()
        if not row:
            return None
        d = dict(row)
        if d.get("extra"):
            d["extra"] = json.loads(d["extra"])
        return d
    finally:
        await db.close()


# ── Settings KV ──

async def get_setting(key: str) -> Optional[str]:
    db = await _get_db()
    try:
        cursor = await db.execute("SELECT value FROM settings WHERE key = ?", (key,))
        row = await cursor.fetchone()
        return row["value"] if row else None
    finally:
        await db.close()


async def set_setting(key: str, value: str):
    db = await _get_db()
    try:
        await db.execute(
            "INSERT INTO settings (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )
        await db.commit()
    finally:
        await db.close()
