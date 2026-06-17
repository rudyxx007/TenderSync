import os
import tempfile
import traceback
from datetime import datetime, timedelta
from typing import Optional

import dateparser
import uvicorn
import voyageai
from docling.document_converter import DocumentConverter
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from groq import Groq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from supabase.client import Client, create_client

from auth import get_token, resolve_user_id
from bid_engine import run_bid_evaluation
from profile_service import get_profile_status, require_complete_profile, upsert_profile
from schemas import (
    AuthUserResponse,
    CompanyProfileInput,
    CompanyProfileResponse,
    ProfileStatusResponse,
    TenderAnalysisDetail,
    TenderAnalysisSummary,
    TenderSchema,
)

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

load_dotenv()

app = FastAPI(
    title="TenderSync Backend",
    version="3.0.0",
    description="Multi-tenant RFP extraction, RAG pipeline, and Bid/No-Bid evaluation API.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_SECRET_KEY") or os.environ.get("SUPABASE_SERVICE_KEY")
voyage_key = os.environ.get("VOYAGE_API_KEY")
groq_key = os.environ.get("GROQ_API_KEY")

if not all([supabase_url, supabase_key, voyage_key, groq_key]):
    raise ValueError("Missing one or more credentials in your .env file.")

supabase_client: Client = create_client(supabase_url, supabase_key)
voyage_client = voyageai.Client(api_key=voyage_key)
groq_client = Groq(api_key=groq_key)


def get_current_user_id(token: Optional[str] = Depends(get_token)) -> str:
    return resolve_user_id(supabase_client, token)


EXTRACTION_SYSTEM_PROMPT = """You are a rigid legal-technical extraction agent for RFP/tender documents.
Extract ONLY what is explicitly stated or strongly implied in the provided text.
Return valid JSON with these exact keys:
- tender_id (string or null)
- issuing_authority (string)
- submission_deadline (string, preserve original wording)
- estimated_value_or_budget (string, include currency and ranges if present)
- key_deliverables (array of strings)
- mandatory_compliance_criteria (array of strings — certifications, insurance, registrations marked as required/mandatory/shall/must)
- confidence_score (float 0.0-1.0 — your confidence in extraction accuracy)

If a field is not found, use empty string, empty array, or null as appropriate.
Do not invent certifications, budgets, or deadlines not supported by the text."""


def build_extraction_prompt(context_text: str) -> str:
    return f"""Extract tender details from these RFP document chunks.

Focus on: budget/value, submission deadline, mandatory compliance (ISO, SOC, insurance, registrations),
and key deliverables/services required.

DOCUMENT CHUNKS:
{context_text}"""


def save_tender_analysis(
    user_id: str,
    filename: str,
    extracted_data: dict,
    evaluation_data: dict,
) -> str:
    row = {
        "user_id": user_id,
        "filename": filename,
        "tender_id": extracted_data.get("tender_id"),
        "issuing_authority": extracted_data.get("issuing_authority"),
        "submission_deadline": extracted_data.get("submission_deadline"),
        "estimated_value_or_budget": extracted_data.get("estimated_value_or_budget"),
        "confidence_score": extracted_data.get("confidence_score"),
        "decision": evaluation_data.get("decision"),
        "win_probability_score": evaluation_data.get("win_probability_score"),
        "extracted_data": extracted_data,
        "evaluation_data": evaluation_data,
    }
    result = supabase_client.table("tender_analyses").insert(row).execute()
    return result.data[0]["id"]


@app.get("/")
def health_check():
    return {
        "status": "TenderSync Enterprise Pipeline Operational",
        "version": "3.0.0",
        "auth": "Bearer JWT required for all /api/* routes except this health check",
    }


@app.get("/api/auth/me", response_model=AuthUserResponse)
def get_me(
    user_id: str = Depends(get_current_user_id),
    token: Optional[str] = Depends(get_token),
):
    email = None
    if token:
        try:
            user_resp = supabase_client.auth.get_user(token)
            if user_resp and user_resp.user:
                email = user_resp.user.email
        except Exception:
            pass
    return AuthUserResponse(user_id=user_id, email=email)


@app.get("/api/profile/status", response_model=ProfileStatusResponse)
def profile_status(user_id: str = Depends(get_current_user_id)):
    """If can_use_app=false, frontend must redirect to /onboarding."""
    return get_profile_status(supabase_client, user_id)


@app.get("/api/profile", response_model=CompanyProfileResponse)
def get_profile(user_id: str = Depends(get_current_user_id)):
    status_resp = get_profile_status(supabase_client, user_id)
    if not status_resp.exists or not status_resp.profile:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Company profile not found. Complete onboarding first.",
        )
    return status_resp.profile


@app.put("/api/profile", response_model=CompanyProfileResponse)
def create_or_update_profile(
    payload: CompanyProfileInput,
    user_id: str = Depends(get_current_user_id),
):
    return upsert_profile(supabase_client, user_id, payload)


@app.post("/api/process-tender")
async def process_tender(
    file: UploadFile = File(...),
    user_id: str = Depends(get_current_user_id),
):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    try:
        company_profile = require_complete_profile(supabase_client, user_id)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc

    temp_file_path = ""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name

        print(f"[{user_id}] 1. Parsing document...")
        converter = DocumentConverter()
        markdown_text = converter.convert(temp_file_path).document.export_to_markdown()

        print(f"[{user_id}] 2. Chunking and embedding...")
        chunks = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200).split_text(
            markdown_text
        )
        embeddings = voyage_client.embed(
            chunks, model="voyage-finance-2", input_type="document"
        ).embeddings

        print(f"[{user_id}] 3. Storing user-scoped context...")
        records = [
            {
                "content": text,
                "embedding": embedding,
                "metadata": {"filename": file.filename},
                "user_id": user_id,
            }
            for text, embedding in zip(chunks, embeddings)
        ]
        supabase_client.table("documents").insert(records).execute()

        print(f"[{user_id}] 4. Semantic retrieval...")
        query_vector = voyage_client.embed(
            ["budget, deliverables, compliance requirements, deadline"],
            model="voyage-finance-2",
            input_type="query",
        ).embeddings[0]
        matched_docs = supabase_client.rpc(
            "match_documents",
            {
                "query_embedding": query_vector,
                "match_count": 8,
                "filter_user_id": user_id,
            },
        ).execute().data
        context_text = "\n\n".join([doc["content"] for doc in matched_docs])

        print(f"[{user_id}] 5. Agentic extraction...")
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": build_extraction_prompt(context_text)},
            ],
            model="llama-3.3-70b-versatile",
            temperature=0,
            response_format={"type": "json_object"},
        )
        extracted_data = TenderSchema.model_validate_json(
            chat_completion.choices[0].message.content
        ).model_dump()

        print(f"[{user_id}] 6. Hybrid bid/no-bid evaluation...")
        evaluation_data = run_bid_evaluation(extracted_data, company_profile, groq_client)

        analysis_id = save_tender_analysis(user_id, file.filename, extracted_data, evaluation_data)

        return {
            "status": "success",
            "data": {**extracted_data, "evaluation": evaluation_data, "analysis_id": analysis_id},
        }

    except HTTPException:
        raise
    except Exception as exc:
        print("\n!!! PIPELINE EXCEPTION !!!")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)


@app.get("/api/tenders", response_model=list[TenderAnalysisSummary])
def list_tenders(user_id: str = Depends(get_current_user_id)):
    try:
        require_complete_profile(supabase_client, user_id)
    except ValueError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc

    rows = (
        supabase_client.table("tender_analyses")
        .select(
            "id, tender_id, filename, issuing_authority, decision, "
            "win_probability_score, confidence_score, created_at"
        )
        .eq("user_id", user_id)
        .order("created_at", desc=True)
        .execute()
    )
    return [TenderAnalysisSummary(**row) for row in rows.data]


@app.get("/api/tenders/{analysis_id}", response_model=TenderAnalysisDetail)
def get_tender(analysis_id: str, user_id: str = Depends(get_current_user_id)):
    row = (
        supabase_client.table("tender_analyses")
        .select("*")
        .eq("id", analysis_id)
        .eq("user_id", user_id)
        .execute()
    )
    if not row.data:
        raise HTTPException(status_code=404, detail="Tender analysis not found.")
    data = row.data[0]
    return TenderAnalysisDetail(
        id=data["id"],
        user_id=data["user_id"],
        filename=data["filename"],
        extracted_data=data["extracted_data"],
        evaluation_data=data["evaluation_data"],
        created_at=data["created_at"],
    )


def _parse_deadline_to_ics_date(deadline_string: str) -> str:
    parsed = dateparser.parse(deadline_string, settings={"PREFER_DATES_FROM": "future"})
    if parsed:
        return parsed.strftime("%Y%m%d")
    return (datetime.now() + timedelta(days=30)).strftime("%Y%m%d")


@app.post("/api/generate-calendar")
async def generate_calendar(
    deadline_string: str,
    user_id: str = Depends(get_current_user_id),
):
    try:
        require_complete_profile(supabase_client, user_id)
    except ValueError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc

    ics_date = _parse_deadline_to_ics_date(deadline_string)
    ics_content = f"""BEGIN:VCALENDAR
VERSION:2.0
PRODID:-//TenderSync//Enterprise Engine//EN
BEGIN:VEVENT
SUMMARY:Tender Submission Deadline
DESCRIPTION:Automatically generated by TenderSync. Original extracted deadline text: {deadline_string}
DTSTART;VALUE=DATE:{ics_date}
DTEND;VALUE=DATE:{ics_date}
STATUS:CONFIRMED
END:VEVENT
END:VCALENDAR"""
    return PlainTextResponse(
        content=ics_content,
        media_type="text/calendar",
        headers={"Content-Disposition": "attachment; filename=tender_deadline.ics"},
    )


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)
