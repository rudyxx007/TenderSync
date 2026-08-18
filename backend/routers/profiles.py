from fastapi import APIRouter, Depends, HTTPException, status
import schemas
import db
from profile_service import get_profile_status, upsert_profile

router = APIRouter(prefix="/api/profile", tags=["Company Profile"])


@router.get("/status", response_model=schemas.ProfileStatusResponse)
def profile_status(org_info: dict = Depends(db.get_current_user_org)):
    """If can_use_app=false, frontend must redirect to /onboarding."""
    return get_profile_status(db.supabase_client, org_info["org_id"])


@router.get("", response_model=schemas.CompanyProfileResponse)
def get_profile(org_info: dict = Depends(db.get_current_user_org)):
    status_resp = get_profile_status(db.supabase_client, org_info["org_id"])
    if not status_resp.exists or not status_resp.profile:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Company profile not found. Complete onboarding first.",
        )
    return status_resp.profile


@router.put("", response_model=schemas.CompanyProfileResponse)
def create_or_update_profile(
    payload: schemas.CompanyProfileInput,
    org_info: dict = Depends(db.get_current_user_org),
):
    if org_info["role"] not in ["owner", "admin"]:
        raise HTTPException(status_code=403, detail="Only owner or admin can edit the company profile.")
    return upsert_profile(db.supabase_client, org_info["org_id"], payload)
