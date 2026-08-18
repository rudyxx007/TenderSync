import os
from typing import Optional, Dict
from dotenv import load_dotenv
from fastapi import Depends
from groq import Groq
from supabase.client import Client, create_client

from auth import get_token, resolve_user_id, resolve_user_org

load_dotenv()

supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_SECRET_KEY") or os.environ.get("SUPABASE_SERVICE_KEY")
hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_API_KEY")
groq_key = os.environ.get("GROQ_API_KEY")

if not supabase_url or not supabase_key or not groq_key or not hf_token:
    raise ValueError(
        "Missing one or more credentials in your .env file "
        "(requires SUPABASE_URL, SUPABASE_SECRET_KEY, GROQ_API_KEY, and HF_TOKEN)."
    )

supabase_client: Client = create_client(supabase_url, supabase_key)
groq_client = Groq(api_key=groq_key)


def get_current_user_id(token: Optional[str] = Depends(get_token)) -> str:
    return resolve_user_id(supabase_client, token)


def get_current_user_org(
    user_id: str = Depends(get_current_user_id)
) -> Dict[str, str]:
    org_info = resolve_user_org(supabase_client, user_id)
    org_info["user_id"] = user_id
    return org_info
