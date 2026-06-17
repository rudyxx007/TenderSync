"""Supabase JWT authentication for FastAPI."""

import os
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from supabase.client import Client

_bearer = HTTPBearer(auto_error=False)

ALLOW_DEV_BYPASS = os.environ.get("ALLOW_DEV_BYPASS", "false").lower() == "true"
DEVELOPMENT_USER_ID = os.environ.get(
    "DEVELOPMENT_USER_ID", "77498816-5535-4406-8756-5b91f53161b9"
)


def get_token(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
) -> Optional[str]:
    if credentials and credentials.scheme.lower() == "bearer":
        return credentials.credentials
    return None


def resolve_user_id(supabase_client: Client, token: Optional[str]) -> str:
    """
    Resolve authenticated user ID from Supabase JWT.
    Falls back to DEVELOPMENT_USER_ID only when ALLOW_DEV_BYPASS=true.
    """
    if token:
        try:
            response = supabase_client.auth.get_user(token)
            if response and response.user:
                return response.user.id
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Invalid or expired token: {exc}",
            ) from exc

    if ALLOW_DEV_BYPASS and DEVELOPMENT_USER_ID:
        return DEVELOPMENT_USER_ID

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication required. Sign in and send Authorization: Bearer <token>.",
    )


def get_current_user_id(
    supabase_client: Client,
    token: Optional[str] = Depends(get_token),
) -> str:
    return resolve_user_id(supabase_client, token)
