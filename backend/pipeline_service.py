import os
import tempfile
import traceback
from datetime import datetime, timezone
from typing import List, Optional

from fastapi import UploadFile
from groq import Groq
from supabase.client import Client

import rag_service
import schemas
from bid_engine import run_bid_evaluation
import document_extractor

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

EXTRACTION_QUERY = (
    "mandatory compliance requirements certifications insurance registrations; "
    "contract budget value estimated value; submission deadline closing date; "
    "key deliverables services scope; tender reference number issuing authority"
)


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


def _build_retrieval_context(documents) -> str:
    """Use the parent section when available, otherwise the child chunk."""
    blocks = []
    for index, doc in enumerate(documents, start=1):
        path = " > ".join(doc.section_path) if doc.section_path else "Document"
        blocks.append(f"[{index}] ({path})\n{doc.context_text}")
    return "\n\n".join(blocks)


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
        markdown_text = document_extractor.extract_document_text(
            temp_file_path, file.filename or "document.pdf"
        )
        if not markdown_text.strip():
            raise ValueError(f"Unable to extract text from uploaded document {file.filename}.")

        print(f"[{org_id}] 2. Contextual parent/child chunking + embedding...")
        ingest_result = None
        documents = []
        rag_error: Optional[str] = None
        try:
            ingest_result = rag_service.ingest_document(
                supabase_client=supabase_client,
                org_id=org_id,
                user_id=user_id,
                filename=file.filename or "document.pdf",
                document_text=markdown_text,
            )
            documents = rag_service.retrieve(
                supabase_client=supabase_client,
                org_id=org_id,
                query=EXTRACTION_QUERY,
                document_id=ingest_result.document_id,
                top_k=12,
            )
        except Exception as rag_exc:
            # The upload must still produce an analysis if the vector schema has
            # not been migrated. Retrieval is scoped to this upload only, so the
            # fallback cannot leak another tender's text.
            rag_error = str(rag_exc)
            print(f"[{org_id}] RAG retrieval warning: {rag_error}")

        if documents:
            context_text = _build_retrieval_context(documents)
            retrieval_source = "hybrid_parent_child"
        else:
            # Single-document fallback: the current upload only.
            context_text = markdown_text[:24_000]
            retrieval_source = "document_text_fallback"

        print(f"[{org_id}] 3. Agentic extraction...")
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

        print(f"[{org_id}] 4. Hybrid bid/no-bid evaluation...")
        evaluation_data = run_bid_evaluation(extracted_data, company_profile, groq_client)

        analysis_id = save_tender_analysis(
            supabase_client, org_id, user_id, file.filename, extracted_data, evaluation_data
        )

        return {
            "status": "success",
            "filename": file.filename,
            "data": {
                **extracted_data,
                "evaluation": evaluation_data,
                "analysis_id": analysis_id,
                "citations": rag_service.format_citations(documents),
                "rag": {
                    "retrieval_source": retrieval_source,
                    "document_id": ingest_result.document_id if ingest_result else None,
                    "parents_indexed": ingest_result.parent_count if ingest_result else 0,
                    "chunks_indexed": ingest_result.child_count if ingest_result else 0,
                    "chunks_retrieved": len(documents),
                    "embedded": ingest_result.embedded if ingest_result else False,
                    "warning": rag_error,
                },
            },
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
        res = await run_single_tender_pipeline(
            file, org_id, user_id, company_profile, supabase_client, groq_client
        )
        results.append(res)
        if res["status"] == "failed":
            failed_count += 1
        completed_count += 1

        supabase_client.table("batch_jobs").update({
            "completed_files": completed_count,
            "results": results,
        }).eq("id", batch_id).execute()

    final_status = "failed" if failed_count == len(files) else "complete"
    supabase_client.table("batch_jobs").update({
        "status": final_status,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }).eq("id", batch_id).execute()
