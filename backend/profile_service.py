"""Company profile CRUD and completeness validation."""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from bid_engine import normalize_company_profile
from schemas import CompanyProfileInput, CompanyProfileResponse, ProfileStatusResponse

# Minimum fields required before a user can upload tenders or run evaluations.
REQUIRED_PROFILE_FIELDS: Dict[str, str] = {
    "organization_name": "Organization name",
    "min_contract_value": "Minimum contract value",
    "active_certifications": "At least one certification",
    "core_capabilities": "At least one core capability",
}


def _is_non_empty_list(value: Any) -> bool:
    return isinstance(value, list) and len(value) > 0


def compute_missing_fields(profile: Dict[str, Any]) -> List[str]:
    missing: List[str] = []
    if not profile.get("organization_name") or not str(profile["organization_name"]).strip():
        missing.append("organization_name")
    min_val = profile.get("min_contract_value")
    if min_val is None or float(min_val) <= 0:
        missing.append("min_contract_value")
    if not _is_non_empty_list(profile.get("active_certifications")):
        missing.append("active_certifications")
    if not _is_non_empty_list(profile.get("core_capabilities")):
        missing.append("core_capabilities")
    return missing


def is_profile_complete(profile: Dict[str, Any]) -> bool:
    return len(compute_missing_fields(profile)) == 0


def compute_completion_percentage(profile: Dict[str, Any]) -> int:
    """Calculate profile completion percentage across all standard profile fields."""
    tracked = [
        bool(profile.get("organization_name") and str(profile["organization_name"]).strip()),
        bool(profile.get("min_contract_value") and float(profile.get("min_contract_value", 0)) > 0),
        bool(profile.get("max_contract_value") and float(profile.get("max_contract_value", 0)) > 0),
        _is_non_empty_list(profile.get("active_certifications")),
        _is_non_empty_list(profile.get("core_capabilities")),
        _is_non_empty_list(profile.get("strategic_focus_areas")),
        _is_non_empty_list(profile.get("past_performance_sectors")),
        _is_non_empty_list(profile.get("geographic_coverage")),
        bool(isinstance(profile.get("insurance_coverage"), dict) and len(profile.get("insurance_coverage", {})) > 0),
        bool(profile.get("min_bid_lead_time_days")),
        bool(profile.get("team_capacity_score")),
        bool(profile.get("relationship_strength_score")),
    ]
    return round((sum(1 for t in tracked if t) / len(tracked)) * 100)


def profile_to_response(raw: Dict[str, Any]) -> CompanyProfileResponse:
    normalized = normalize_company_profile(raw)
    missing = compute_missing_fields(normalized)
    pct = compute_completion_percentage(normalized)
    return CompanyProfileResponse(
        org_id=str(normalized.get("org_id", "")),
        organization_name=normalized.get("organization_name") or "",
        min_contract_value=float(normalized.get("min_contract_value") or 0),
        max_contract_value=normalized.get("max_contract_value"),
        active_certifications=normalized.get("active_certifications") or [],
        core_capabilities=normalized.get("core_capabilities") or [],
        strategic_focus_areas=normalized.get("strategic_focus_areas") or [],
        past_performance_sectors=normalized.get("past_performance_sectors") or [],
        geographic_coverage=normalized.get("geographic_coverage") or [],
        insurance_coverage=normalized.get("insurance_coverage") or {},
        min_bid_lead_time_days=normalized.get("min_bid_lead_time_days", 14),
        team_capacity_score=normalized.get("team_capacity_score", 3),
        relationship_strength_score=normalized.get("relationship_strength_score", 2),
        is_complete=len(missing) == 0,
        completion_percentage=pct,
        missing_fields=missing,
        created_at=normalized.get("created_at"),
        updated_at=normalized.get("updated_at"),
    )


def get_profile_status(supabase_client, org_id: str) -> ProfileStatusResponse:
    response = (
        supabase_client.table("company_profiles")
        .select("*")
        .eq("org_id", org_id)
        .execute()
    )
    if not response.data:
        all_missing = list(REQUIRED_PROFILE_FIELDS.keys())
        return ProfileStatusResponse(
            exists=False,
            is_complete=False,
            completion_percentage=0,
            missing_fields=all_missing,
            can_use_app=False,
            profile=None,
        )

    profile_resp = profile_to_response(response.data[0])
    return ProfileStatusResponse(
        exists=True,
        is_complete=profile_resp.is_complete,
        completion_percentage=profile_resp.completion_percentage,
        missing_fields=profile_resp.missing_fields,
        can_use_app=profile_resp.is_complete,
        profile=profile_resp,
    )


def require_complete_profile(supabase_client, org_id: str) -> Dict[str, Any]:
    """Return normalized profile or raise ValueError with actionable message."""
    response = (
        supabase_client.table("company_profiles")
        .select("*")
        .eq("org_id", org_id)
        .execute()
    )
    if not response.data:
        raise ValueError(
            "Company profile not set up. Complete onboarding at /api/profile before using TenderSync."
        )
    normalized = normalize_company_profile(response.data[0])
    missing = compute_missing_fields(normalized)
    if missing:
        labels = [REQUIRED_PROFILE_FIELDS[field] for field in missing]
        raise ValueError(
            f"Company profile incomplete. Missing: {', '.join(labels)}. "
            "Update your profile before uploading tenders."
        )
    return normalized


def upsert_profile(
    supabase_client,
    org_id: str,
    payload: CompanyProfileInput,
) -> CompanyProfileResponse:
    now = datetime.now(timezone.utc).isoformat()
    row = payload.model_dump()
    row["org_id"] = org_id
    row["updated_at"] = now

    existing = (
        supabase_client.table("company_profiles")
        .select("org_id, created_at")
        .eq("org_id", org_id)
        .execute()
    )
    if existing.data:
        supabase_client.table("company_profiles").update(row).eq("org_id", org_id).execute()
    else:
        row["created_at"] = now
        supabase_client.table("company_profiles").insert(row).execute()

    saved = (
        supabase_client.table("company_profiles")
        .select("*")
        .eq("org_id", org_id)
        .execute()
    )
    return profile_to_response(saved.data[0])
