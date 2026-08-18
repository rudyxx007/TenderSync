"""Supabase JWT authentication for FastAPI."""

import os
from typing import Optional, Dict

from fastapi import Depends, HTTPException, status, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from supabase.client import Client
from dotenv import load_dotenv

load_dotenv()

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


def resolve_user_org(supabase_client: Client, user_id: str) -> Dict[str, str]:
    """
    Resolve the organization ID and role for a user.
    """
    if ALLOW_DEV_BYPASS and user_id == DEVELOPMENT_USER_ID:
        # For dev bypass, just fetch the first org or fail gracefully if none exists
        result = supabase_client.table("org_members").select("org_id, role").eq("user_id", user_id).limit(1).execute()
        if result.data:
            return result.data[0]
        # Return dummy org for dev if not set up
        return {"org_id": "00000000-0000-0000-0000-000000000000", "role": "owner"}

    result = supabase_client.table("org_members").select("org_id, role").eq("user_id", user_id).limit(1).execute()
    if not result.data:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User does not belong to any organization. Please create or join an organization.",
        )
    return result.data[0]


def require_role(allowed_roles: list[str]):
    """
    Dependency to enforce RBAC.
    Usage: Depends(require_role(["owner", "admin"]))
    """
    def role_checker(request: Request):
        # We assume org_info is attached to request.state by a higher-level dependency
        if not hasattr(request.state, "org_info"):
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="org_info not found in request state. Ensure get_current_user_org was called.",
            )
        user_role = request.state.org_info.get("role")
        if user_role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions. Requires one of: {allowed_roles}. You are a {user_role}.",
            )
        return True
    return role_checker
