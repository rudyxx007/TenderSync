import os

# --- UNIVERSAL WINDOWS DEADLOCK FIX ---
# Lock the thread count at the OS level BEFORE PyTorch or Docling initialize.
# This strictly prevents the multiprocessing module from infinitely spawning sub-processes.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# --------------------------------------

import tempfile
import uvicorn
import traceback
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
from dotenv import load_dotenv

# Native Core SDKs
from supabase.client import create_client, Client
import voyageai
from groq import Groq

# Document Parsing & Chunking
from docling.document_converter import DocumentConverter
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()

app = FastAPI(title="TenderSync Backend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Native Clients
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_SECRET_KEY") or os.environ.get("SUPABASE_SERVICE_KEY")
voyage_key = os.environ.get("VOYAGE_API_KEY")
groq_key = os.environ.get("GROQ_API_KEY")

if not all([supabase_url, supabase_key, voyage_key, groq_key]):
    raise ValueError("Missing one or more credentials in your .env file.")

supabase_client: Client = create_client(supabase_url, supabase_key)
voyage_client = voyageai.Client(api_key=voyage_key)
groq_client = Groq(api_key=groq_key)

class TenderSchema(BaseModel):
    tender_id: Optional[str] = Field(description="The unique reference code or ID number of the tender. Return 'Not specified' if missing.")
    issuing_authority: str = Field(description="The corporate or government body publishing the RFP.")
    submission_deadline: str = Field(description="The strict closing date/time for the bid proposal.")
    estimated_value_or_budget: str = Field(description="The financial value or budget limits defined. Return 'Not specified' if missing.")
    key_deliverables: List[str] = Field(description="Core products, services, or outcomes required by the client.")
    mandatory_compliance_criteria: List[str] = Field(description="Required legal, insurance, security clearances, or ISO certifications.")
    confidence_score: float = Field(description="Your overall confidence score in the accuracy of this extraction, from 0.0 to 1.0.")

@app.get("/")
def health_check():
    return {"status": "TenderSync Native Core Pipeline Operational"}

@app.post("/api/process-tender")
async def process_tender(file: UploadFile = File(...)):
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Invalid file type. Only PDFs are supported.")

    temp_file_path = ""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name

        print(f"Processing {file.filename} with Docling on local GPU context...")
        
        # Clean initialization without broken internal options
        converter = DocumentConverter()
        result = converter.convert(temp_file_path)
        markdown_text = result.document.export_to_markdown()

        print("Chunking text layout...")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000, 
            chunk_overlap=200
        )
        chunks = text_splitter.split_text(markdown_text)

        print(f"Generating vectors via Voyage AI for {len(chunks)} chunks...")

        voyage_response = voyage_client.embed(chunks, model="voyage-finance-2", input_type="document")
        embeddings = voyage_response.embeddings

        print("Writing vectors directly to Supabase pgvector tables...")

        records = []
        for text, embedding in zip(chunks, embeddings):
            records.append({
                "content": text,
                "embedding": embedding,
                "metadata": {"filename": file.filename}
            })
        
        supabase_client.table("documents").insert(records).execute()
        
        print("Generating query vector for targeted semantic search...")

        query_text = "tender ID, issuing authority, submission deadline, budget value, key deliverables, and mandatory compliance requirements"
        query_vector = voyage_client.embed([query_text], model="voyage-finance-2", input_type="query").embeddings[0]

        rpc_response = supabase_client.rpc(
            "match_documents",
            {
                "query_embedding": query_vector,
                "match_count": 8
            }
        ).execute()

        matched_docs = rpc_response.data
        context_text = "\n\n".join([doc["content"] for doc in matched_docs])

        print("Executing Native Groq Structured Extraction...")

        prompt = f"""
        You are an expert RFP bid analyst. Based ONLY on the following extracted document chunks, 
        extract the tender details. Return your response strictly as a JSON object matching this schema:
        
        {{
          "tender_id": "string or 'Not specified'",
          "issuing_authority": "string",
          "submission_deadline": "string",
          "estimated_value_or_budget": "string or 'Not specified'",
          "key_deliverables": ["string"],
          "mandatory_compliance_criteria": ["string"],
          "confidence_score": float between 0.0 and 1.0
        }}

        DOCUMENT CHUNKS:
        {context_text}
        """

        chat_completion = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a rigid legal-technical extraction agent. You only return valid, parseable JSON data."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model="llama-3.3-70b-versatile",
            temperature=0,
            response_format={"type": "json_object"}
        )

        raw_json_output = chat_completion.choices[0].message.content
        
        validated_data = TenderSchema.model_validate_json(raw_json_output)

        print("Pipeline Complete. Returning JSON payload.")
        return {"status": "success", "data": validated_data.model_dump()}
    
    except Exception as e:
        print("\n!!! DETAILED PIPELINE EXCEPTION !!!")
        traceback.print_exc()
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
        raise HTTPException(status_code=500, detail=f"Pipeline Error: {str(e)}")
    
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)