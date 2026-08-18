import os
from groq import Groq
from typing import Dict, Any, List

COMPARISON_PROMPT = """You are an expert contract analyst.
Compare the following two tenders based on their details and requirements.
Identify:
1. Which one is larger in budget/value?
2. Which one has an earlier deadline?
3. Which one is a better fit based on the provided company profile (if applicable)?
4. Key differences in deliverables and compliance requirements.

Format your response as a JSON object with keys:
- budget_comparison (string)
- deadline_comparison (string)
- compliance_differences (string)
- deliverable_differences (string)
- recommendation (string)
"""

def compare_tenders(tender1: Dict[str, Any], tender2: Dict[str, Any], company_profile: Dict[str, Any], groq_client: Groq) -> Dict[str, Any]:
    context = f"""
    COMPANY PROFILE:
    {company_profile}

    TENDER 1:
    {tender1}

    TENDER 2:
    {tender2}
    """
    
    chat_completion = groq_client.chat.completions.create(
        messages=[
            {"role": "system", "content": COMPARISON_PROMPT},
            {"role": "user", "content": context},
        ],
        model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
        temperature=0,
        response_format={"type": "json_object"},
    )
    
    import json
    return json.loads(chat_completion.choices[0].message.content)
