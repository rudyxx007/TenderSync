from typing import List
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status, BackgroundTasks

import schemas
import db
from document_extractor import is_supported_document
from pipeline_service import run_single_tender_pipeline, run_batch_processing_task
import profile_service
import comparison_engine

router = APIRouter(tags=["Tender Processing & Analysis"])


@router.post("/api/process-tender")
@router.post("/api/upload-tender")
@router.post("/api/tenders/process")
async def process_tender(
    file: UploadFile = File(...),
    org_info: dict = Depends(db.get_current_user_org),
):
    if not is_supported_document(file.filename):
        raise HTTPException(
            status_code=400,
            detail="Unsupported file format. Please upload a PDF or image (PNG, JPG, JPEG, WEBP, TIFF, BMP)."
        )

    try:
        company_profile = profile_service.require_complete_profile(db.supabase_client, org_info["org_id"])
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc

    result = await run_single_tender_pipeline(
        file, org_info["org_id"], org_info["user_id"], company_profile, db.supabase_client, db.groq_client
    )

    if result["status"] == "failed":
        raise HTTPException(status_code=500, detail=result["error"])

    return result


@router.post("/api/process-tender/batch")
@router.post("/api/tenders/batch")
async def process_tender_batch(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    org_info: dict = Depends(db.get_current_user_org),
):
    for f in files:
        if not is_supported_document(f.filename):
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file format for '{f.filename}'. Please upload PDF or image files (PNG, JPG, JPEG, WEBP, TIFF, BMP)."
            )

    org_id = org_info["org_id"]
    try:
        company_profile = profile_service.require_complete_profile(db.supabase_client, org_id)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc

    # 1. Create batch_jobs row
    batch_res = db.supabase_client.table("batch_jobs").insert({
        "org_id": org_id,
        "created_by": org_info["user_id"],
        "total_files": len(files),
        "completed_files": 0,
        "status": "processing",
        "results": []
    }).execute()

    batch_id = batch_res.data[0]["id"]

    # 2. Add to background tasks
    background_tasks.add_task(
        run_batch_processing_task,
        batch_id,
        files,
        org_id,
        org_info["user_id"],
        company_profile,
        db.supabase_client,
        db.groq_client,
    )

    return {"batch_id": batch_id, "status": "processing"}


@router.get("/api/my-analyses", response_model=List[schemas.TenderAnalysisSummary])
@router.get("/api/tenders", response_model=List[schemas.TenderAnalysisSummary])
def list_tenders(org_info: dict = Depends(db.get_current_user_org)):
    try:
        profile_service.require_complete_profile(db.supabase_client, org_info["org_id"])
    except ValueError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc

    rows = (
        db.supabase_client.table("tender_analyses")
        .select(
            "id, tender_id, filename, issuing_authority, decision, "
            "win_probability_score, confidence_score, created_at"
        )
        .eq("org_id", org_info["org_id"])
        .order("created_at", desc=True)
        .execute()
    )
    return [schemas.TenderAnalysisSummary(**row) for row in rows.data]


@router.get("/api/analysis/{analysis_id}", response_model=schemas.TenderAnalysisDetail)
@router.get("/api/tenders/{analysis_id}", response_model=schemas.TenderAnalysisDetail)
def get_tender(analysis_id: str, org_info: dict = Depends(db.get_current_user_org)):
    row = (
        db.supabase_client.table("tender_analyses")
        .select("*")
        .eq("id", analysis_id)
        .eq("org_id", org_info["org_id"])
        .execute()
    )
    if not row.data:
        raise HTTPException(status_code=404, detail="Tender analysis not found.")
    data = row.data[0]
    return schemas.TenderAnalysisDetail(
        id=data["id"],
        user_id=data["user_id"],
        filename=data["filename"],
        extracted_data=data["extracted_data"],
        evaluation_data=data["evaluation_data"],
        created_at=data["created_at"],
    )


@router.post("/api/tenders/compare")
def compare_two_tenders(
    request: schemas.CompareTendersRequest,
    org_info: dict = Depends(db.get_current_user_org),
):
    org_id = org_info["org_id"]
    company_profile = profile_service.require_complete_profile(db.supabase_client, org_id)

    tenders_data = []
    for tid in request.tender_ids:
        res = db.supabase_client.table("tender_analyses").select("*").eq("id", tid).eq("org_id", org_id).execute()
        if res.data:
            tenders_data.append(res.data[0])

    if len(tenders_data) < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 valid tenders to compare")

    result = comparison_engine.compare_tenders(tenders_data[0], tenders_data[1], company_profile, db.groq_client)
    return result
