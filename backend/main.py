import os
import uvicorn
from docling.document_converter import DocumentConverter
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Database and Clients
from db import supabase_client, groq_client, get_current_user_id, get_current_user_org
# Services & Engines
from document_extractor import extract_document_text, is_supported_document
from embedding_service import generate_embeddings
from pipeline_service import run_single_tender_pipeline, run_batch_processing_task
from profile_service import get_profile_status, require_complete_profile, upsert_profile
from bid_engine import run_bid_evaluation
from comparison_engine import compare_tenders
from proposal_engine import generate_proposal
from report_generator import render_evaluation_report
# Routers
from routers import (
    auth_router,
    orgs_router,
    profiles_router,
    tenders_router,
    proposals_router,
    market_router,
    batches_router,
    exports_router,
)

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

load_dotenv()

app = FastAPI(
    title="TenderSync Backend",
    version="4.0.0",
    description="Multi-tenant RFP extraction, RAG pipeline, and Bid/No-Bid evaluation API.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register Domain Routers
app.include_router(auth_router)
app.include_router(orgs_router)
app.include_router(profiles_router)
app.include_router(tenders_router)
app.include_router(proposals_router)
app.include_router(market_router)
app.include_router(batches_router)
app.include_router(exports_router)


@app.get("/")
def health_check():
    return {
        "status": "TenderSync Enterprise Pipeline Operational",
        "version": "4.0.0",
        "auth": "Bearer JWT required for all /api/* routes except this health check",
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)
