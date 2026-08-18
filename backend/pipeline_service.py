import os
import tempfile
import traceback
from datetime import datetime, timezone
from typing import List, Optional
from fastapi import UploadFile
from langchain_text_splitters import RecursiveCharacterTextSplitter
from groq import Groq
from supabase.client import Client

import schemas
from bid_engine import run_bid_evaluation
import document_extractor
import embedding_service

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
    supabase_client: Client,
    org_id: str,
    user_id: str,
    filename: str,
    extracted_data: dict,
    evaluation_data: dict,
) -> str:
    row = {
        "org_id": org_id,
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


async def run_single_tender_pipeline(
    file: UploadFile,
    org_id: str,
    user_id: str,
    company_profile: dict,
    supabase_client: Client,
    groq_client: Groq,
) -> dict:
    temp_file_path = ""
    try:
        ext = os.path.splitext(file.filename.lower())[1] if file.filename else ".pdf"
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name

        print(f"[{org_id}] 1. Parsing document {file.filename}...")
        markdown_text = document_extractor.extract_document_text(temp_file_path, file.filename or "document.pdf")
        if not markdown_text.strip():
            raise ValueError(f"Unable to extract text from uploaded document {file.filename}.")

        print(f"[{org_id}] 2. Chunking...")
        chunks = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200).split_text(markdown_text)

        matched_chunks = []
        embeddings = None

        print(f"[{org_id}] 3. Embedding and storing context...")
        try:
            embeddings = embedding_service.generate_embeddings(chunks, task_type="passage")
            if embeddings:
                records = [
                    {
                        "content": text,
                        "embedding": embedding,
                        "metadata": {"filename": file.filename},
                        "user_id": user_id,
                        "org_id": org_id,
                    }
                    for text, embedding in zip(chunks, embeddings)
                ]
                supabase_client.table("documents").insert(records).execute()
        except Exception as embed_exc:
            print(f"[{org_id}] Embedding note (using in-memory context): {embed_exc}")

        print(f"[{org_id}] 4. Semantic retrieval...")
        if embeddings:
            try:
                query_vectors = embedding_service.generate_embeddings(
                    ["budget, deliverables, compliance requirements, deadline"],
                    task_type="query",
                )
                if query_vectors:
                    query_vector = query_vectors[0]
                    matched_docs = supabase_client.rpc(
                        "match_documents_org",
                        {
                            "query_embedding": query_vector,
                            "match_count": 8,
                            "filter_org_id": org_id,
                        },
                    ).execute()
                    if matched_docs.data:
                        matched_chunks = [doc["content"] for doc in matched_docs.data if "content" in doc]
            except Exception as rpc_err:
                print(f"[{org_id}] Vector RPC note: {rpc_err}")

        if not matched_chunks:
            try:
                matched_docs = supabase_client.rpc(
                    "match_documents",
                    {
                        "query_embedding": query_vector,
                        "match_count": 8,
                        "filter_org_id": org_id,
                    },
                ).execute()
                if matched_docs.data:
                    matched_chunks = [doc["content"] for doc in matched_docs.data if "content" in doc]
            except Exception:
                pass

        # Resilient fallback: If RPC fails or returns no chunks, retrieve directly from documents table for this upload
        if not matched_chunks:
            print(f"[{org_id}] Fallback: retrieving document chunks directly from table...")
            try:
                doc_res = (
                    supabase_client.table("documents")
                    .select("content")
                    .eq("org_id", org_id)
                    .limit(8)
                    .execute()
                )
                if doc_res.data:
                    matched_chunks = [doc["content"] for doc in doc_res.data if "content" in doc]
            except Exception as select_err:
                print(f"[{org_id}] Direct table select note: {select_err}")

        # Final fallback: use in-memory parsed chunks
        if not matched_chunks:
            matched_chunks = chunks[:8]

        context_text = "\n\n".join(matched_chunks)

        print(f"[{org_id}] 5. Agentic extraction...")
        groq_model = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": build_extraction_prompt(context_text)},
            ],
            model=groq_model,
            temperature=0,
            response_format={"type": "json_object"},
        )
        extracted_data = schemas.TenderSchema.model_validate_json(
            chat_completion.choices[0].message.content
        ).model_dump()

        print(f"[{org_id}] 6. Hybrid bid/no-bid evaluation...")
        evaluation_data = run_bid_evaluation(extracted_data, company_profile, groq_client)

        analysis_id = save_tender_analysis(supabase_client, org_id, user_id, file.filename, extracted_data, evaluation_data)

        return {
            "status": "success",
            "filename": file.filename,
            "data": {**extracted_data, "evaluation": evaluation_data, "analysis_id": analysis_id},
        }

    except Exception as exc:
        print(f"\n!!! PIPELINE EXCEPTION FOR {file.filename} !!!")
        traceback.print_exc()
        return {
            "status": "failed",
            "filename": file.filename,
            "error": str(exc),
        }
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except Exception as cleanup_err:
                print(f"[Temp file cleanup] Warning: {cleanup_err}")


async def run_batch_processing_task(
    batch_id: str,
    files: List[UploadFile],
    org_id: str,
    user_id: str,
    company_profile: dict,
    supabase_client: Client,
    groq_client: Groq,
):
    results = []
    failed_count = 0
    completed_count = 0

    for file in files:
        res = await run_single_tender_pipeline(file, org_id, user_id, company_profile, supabase_client, groq_client)
        results.append(res)
        if res["status"] == "failed":
            failed_count += 1
        completed_count += 1

        # Update progress
        supabase_client.table("batch_jobs").update({
            "completed_files": completed_count,
            "results": results,
        }).eq("id", batch_id).execute()

    final_status = "failed" if failed_count == len(files) else "complete"
    supabase_client.table("batch_jobs").update({
        "status": final_status,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }).eq("id", batch_id).execute()
