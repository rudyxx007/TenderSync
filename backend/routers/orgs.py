import secrets
import string
from typing import List
from fastapi import APIRouter, Depends, HTTPException

import schemas
import db

router = APIRouter(prefix="/api/orgs", tags=["Organizations"])


def generate_invite_code() -> str:
    alphabet = string.ascii_uppercase + string.digits
    return ''.join(secrets.choice(alphabet) for _ in range(8))


@router.post("", response_model=schemas.OrgResponse)
def create_organization(
    request: schemas.CreateOrgRequest,
    user_id: str = Depends(db.get_current_user_id),
):
    # Check if user already in an org
    existing = db.supabase_client.table("org_members").select("org_id").eq("user_id", user_id).execute()
    if existing.data:
        raise HTTPException(status_code=400, detail="User already belongs to an organization.")

    code = generate_invite_code()
    org_res = db.supabase_client.table("organizations").insert({
        "name": request.name,
        "invite_code": code,
    }).execute()

    org_id = org_res.data[0]["id"]
    db.supabase_client.table("org_members").insert({
        "org_id": org_id,
        "user_id": user_id,
        "role": "owner",
    }).execute()

    return org_res.data[0]


@router.post("/join", response_model=schemas.OrgResponse)
def join_organization(
    request: schemas.JoinOrgRequest,
    user_id: str = Depends(db.get_current_user_id),
):
    existing = db.supabase_client.table("org_members").select("org_id").eq("user_id", user_id).execute()
    if existing.data:
        raise HTTPException(status_code=400, detail="User already belongs to an organization.")

    org_res = db.supabase_client.table("organizations").select("*").eq("invite_code", request.invite_code).execute()
    if not org_res.data:
        raise HTTPException(status_code=404, detail="Invalid invite code.")

    org = org_res.data[0]
    db.supabase_client.table("org_members").insert({
        "org_id": org["id"],
        "user_id": user_id,
        "role": "member",
    }).execute()

    return org


@router.get("/members", response_model=List[schemas.OrgMemberResponse])
def list_org_members(org_info: dict = Depends(db.get_current_user_org)):
    res = db.supabase_client.table("org_members").select("*").eq("org_id", org_info["org_id"]).execute()
    return res.data


@router.post("/invite/regenerate", response_model=schemas.OrgResponse)
def regenerate_invite_code(org_info: dict = Depends(db.get_current_user_org)):
    if org_info["role"] not in ["owner", "admin"]:
        raise HTTPException(status_code=403, detail="Insufficient permissions to regenerate invite code.")

    new_code = generate_invite_code()
    res = db.supabase_client.table("organizations").update({"invite_code": new_code}).eq("id", org_info["org_id"]).execute()
    return res.data[0]
