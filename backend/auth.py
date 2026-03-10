"""
Simple JWT authentication for Manager Mentor.
Single admin password — same pattern as Astra/PrimeHaul.
"""
from datetime import datetime, timedelta, timezone
from typing import Optional

from jose import jwt, JWTError
from pydantic import BaseModel

from config import settings

ALGORITHM = "HS256"
TOKEN_EXPIRY_HOURS = 72


class LoginRequest(BaseModel):
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


def create_token(sub: str = "admin") -> str:
    expire = datetime.now(timezone.utc) + timedelta(hours=TOKEN_EXPIRY_HOURS)
    payload = {"sub": sub, "exp": expire}
    return jwt.encode(payload, settings.JWT_SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> Optional[str]:
    """Decode and validate a JWT. Returns the subject or None if invalid."""
    try:
        payload = jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=[ALGORITHM])
        return payload.get("sub")
    except JWTError:
        return None


def login(password: str) -> Optional[str]:
    """Validate password and return a JWT, or None if invalid."""
    if password == settings.ADMIN_PASSWORD:
        return create_token()
    return None
