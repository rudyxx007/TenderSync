from typing import Optional
from fastapi import APIRouter, Depends
import schemas
from auth import get_token
import db

router = APIRouter(tags=["Authentication"])


@router.get("/api/me", response_model=schemas.AuthUserResponse)
@router.get("/api/auth/me", response_model=schemas.AuthUserResponse)
def get_me(
    user_id: str = Depends(db.get_current_user_id),
    token: Optional[str] = Depends(get_token),
):
    email = None
    if token:
        try:
            user_resp = db.supabase_client.auth.get_user(token)
            if user_resp and user_resp.user:
                email = user_resp.user.email
        except Exception:
            pass
    return schemas.AuthUserResponse(user_id=user_id, email=email)
