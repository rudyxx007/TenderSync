from fastapi import APIRouter, Depends, HTTPException
from auth import get_token
import db
import profile_service
from tender_discovery import sync_market_tenders

router = APIRouter(tags=["Market Tenders Discovery"])


@router.post("/api/internal/sync-tenders")
def trigger_sync_tenders(authorization: str = Depends(get_token)):
    try:
        sync_market_tenders(db.supabase_client)
        return {"status": "success", "message": "Synced market tenders"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/market-tenders")
def list_market_tenders(
    limit: int = 50,
    offset: int = 0,
    org_info: dict = Depends(db.get_current_user_org),
):
    res = db.supabase_client.table("market_tenders").select("*").order("posted_date", desc=True).range(offset, offset + limit - 1).execute()
    return res.data


@router.post("/api/market-tenders/{market_id}/evaluate")
async def evaluate_market_tender(
    market_id: str,
    org_info: dict = Depends(db.get_current_user_org),
):
    org_id = org_info["org_id"]
    try:
        profile_service.require_complete_profile(db.supabase_client, org_id)
    except Exception as exc:
        raise HTTPException(status_code=403, detail=str(exc))

    res = db.supabase_client.table("market_tenders").select("*").eq("id", market_id).execute()
    if not res.data:
        raise HTTPException(status_code=404, detail="Market tender not found")

    raise HTTPException(status_code=501, detail="Evaluate market tender is partially implemented")
