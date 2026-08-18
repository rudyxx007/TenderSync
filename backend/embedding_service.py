import os
from typing import List, Optional
from huggingface_hub import InferenceClient


def get_embedding_client(token: Optional[str] = None) -> InferenceClient:
    """
    Instantiate a Hugging Face InferenceClient with the provided or environment token.
    """
    resolved_token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_API_KEY")
    return InferenceClient(token=resolved_token)


def generate_embeddings(texts: List[str], task_type: str = "passage") -> List[List[float]]:
    """
    Generate 1024-dimensional embeddings using Hugging Face BAAI/bge-m3.

    Args:
        texts: A list of string texts or chunks to vectorize.
        task_type: The task type ('passage' or 'query').

    Returns:
        A list of 1024-dimensional float lists.
    """
    if not texts:
        return []

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_API_KEY")
    if not token:
        print("[HF BGE-M3 Embeddings] Error: HF_TOKEN is not set.")
        return []

    try:
        client = get_embedding_client(token)
        embeddings = []
        for text in texts:
            vec = client.feature_extraction(text, model="BAAI/bge-m3")
            if hasattr(vec, "tolist"):
                embeddings.append(vec.tolist())
            else:
                embeddings.append(list(vec))
        return embeddings
    except Exception as exc:
        print(f"[HF BGE-M3 Embeddings] Error: {exc}")
        return []
