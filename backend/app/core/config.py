"""Manager Mentor v2 — Configuration"""

from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # App
    app_name: str = "Manager Mentor"
    app_version: str = "2.0.0"
    debug: bool = False

    # Supabase
    supabase_url: str = ""
    supabase_anon_key: str = ""
    supabase_service_role_key: str = ""

    # TwelveLabs
    twelvelabs_api_key: str = ""
    twelvelabs_index_id: str = "69c70b5374e8033fe643609e"

    # Mux
    mux_token_id: str = ""
    mux_token_secret: str = ""

    # Claude AI
    anthropic_api_key: str = ""
    ai_model: str = "claude-sonnet-4-20250514"

    # Stripe
    stripe_secret_key: str = ""
    stripe_webhook_secret: str = ""

    # CORS
    cors_origins: list[str] = [
        "http://localhost:5173",
        "http://localhost:5174",
        "http://localhost:3000",
    ]

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


@lru_cache
def get_settings() -> Settings:
    return Settings()
