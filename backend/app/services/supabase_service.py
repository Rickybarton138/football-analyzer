"""Supabase database service."""

import httpx
from app.core.config import get_settings


class SupabaseService:
    def __init__(self):
        s = get_settings()
        self.url = s.supabase_url
        self.key = s.supabase_service_role_key
        self.anon_key = s.supabase_anon_key

    @property
    def headers(self) -> dict:
        return {
            "apikey": self.key,
            "Authorization": f"Bearer {self.key}",
            "Content-Type": "application/json",
            "Prefer": "return=representation",
        }

    async def insert(self, table: str, data: dict) -> dict:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{self.url}/rest/v1/{table}",
                json=data,
                headers=self.headers,
            )
            resp.raise_for_status()
            return resp.json()[0]

    async def select(self, table: str, filters: str = "", order: str = "created_at.desc", limit: int = 50) -> list:
        params = {"order": order, "limit": str(limit)}
        url = f"{self.url}/rest/v1/{table}?{filters}" if filters else f"{self.url}/rest/v1/{table}"
        async with httpx.AsyncClient() as client:
            resp = await client.get(url, params=params, headers=self.headers)
            resp.raise_for_status()
            return resp.json()

    async def select_one(self, table: str, id: str) -> dict | None:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{self.url}/rest/v1/{table}?id=eq.{id}",
                headers={**self.headers, "Accept": "application/vnd.pgrst.object+json"},
            )
            if resp.status_code == 406:
                return None
            resp.raise_for_status()
            return resp.json()

    async def update(self, table: str, id: str, data: dict) -> dict:
        async with httpx.AsyncClient() as client:
            resp = await client.patch(
                f"{self.url}/rest/v1/{table}?id=eq.{id}",
                json=data,
                headers=self.headers,
            )
            resp.raise_for_status()
            result = resp.json()
            return result[0] if result else data

    async def delete(self, table: str, id: str) -> None:
        async with httpx.AsyncClient() as client:
            resp = await client.delete(
                f"{self.url}/rest/v1/{table}?id=eq.{id}",
                headers=self.headers,
            )
            resp.raise_for_status()
