from typing import List
from fastapi import APIRouter, Depends, HTTPException
import db

router = APIRouter(prefix="/api/batches", tags=["Batch Processing"])


@router.get("")
def list_batches(org_info: dict = Depends(db.get_current_user_org)):
    res = (
        db.supabase_client.table("batch_jobs")
        .select("*")
        .eq("org_id", org_info["org_id"])
        .order("created_at", desc=True)
        .execute()
    )
    return res.data


@router.get("/{batch_id}")
def get_batch_status(batch_id: str, org_info: dict = Depends(db.get_current_user_org)):
    res = (
        db.supabase_client.table("batch_jobs")
        .select("*")
        .eq("id", batch_id)
        .eq("org_id", org_info["org_id"])
        .execute()
    )
    if not res.data:
        raise HTTPException(status_code=404, detail="Batch not found")
    return res.data[0]
