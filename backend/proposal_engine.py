import os
from typing import Dict, Any, List, TypedDict, Annotated
import operator
from langgraph.graph import StateGraph, END
from groq import Groq

from pydantic import BaseModel, Field
import instructor

# Define the state for the LangGraph orchestrator
class ProposalState(TypedDict):
    tender_data: Dict[str, Any]
    company_profile: Dict[str, Any]
    analysis_notes: str
    research_notes: str
    draft_proposal: str
    final_proposal: str
    feedback: str
    revision_count: int

# Initialize Groq client (to be passed in)
# Default model: openai/gpt-oss-120b (configured via GROQ_MODEL env var)

class ProposalOutput(BaseModel):
    executive_summary: str = Field(description="A high-level overview of the proposal.")
    technical_approach: str = Field(description="The detailed technical solution.")
    past_performance: str = Field(description="Case studies and evidence.")
    compliance_matrix: str = Field(description="A statement of compliance with tender requirements.")


def call_llm(groq_client: Groq, system_prompt: str, user_content: str, json_mode: bool = False, response_model: Any = None) -> Any:
    kwargs = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "model": os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
        "temperature": 0.3,
    }
    
    if response_model:
        client = instructor.from_groq(groq_client, mode=instructor.Mode.JSON)
        kwargs["response_model"] = response_model
        return client.chat.completions.create(**kwargs)
        
    kwargs["response_format"] = {"type": "json_object"} if json_mode else None
    chat_completion = groq_client.chat.completions.create(**kwargs)
    return chat_completion.choices[0].message.content


def analyst_node(state: ProposalState, groq_client: Groq) -> Dict[str, Any]:
    print("[Agent] Analyst is working...")
    system = "You are a strategic proposal analyst. Analyze the tender requirements against the company profile to identify win themes, gaps, and key focus areas for the proposal."
    user = f"TENDER: {state['tender_data']}\n\nCOMPANY: {state['company_profile']}"
    notes = call_llm(groq_client, system, user)
    return {"analysis_notes": notes}

def researcher_node(state: ProposalState, groq_client: Groq) -> Dict[str, Any]:
    print("[Agent] Researcher is working...")
    system = "You are a proposal researcher. Based on the analyst's notes and company profile, draft specific case studies, past performance summaries, and technical approaches to be included in the proposal."
    user = f"ANALYST NOTES: {state['analysis_notes']}\n\nCOMPANY: {state['company_profile']}"
    notes = call_llm(groq_client, system, user)
    return {"research_notes": notes}

def writer_node(state: ProposalState, groq_client: Groq) -> Dict[str, Any]:
    print("[Agent] Writer is drafting...")
    system = """You are an expert proposal writer. Draft a professional, compelling, and compliant proposal based on the provided tender requirements, analysis, and research. If there is feedback, incorporate it to improve the draft.
You MUST output your draft as a structured JSON object containing exactly the following keys:
- "executive_summary": A high-level overview of the proposal.
- "technical_approach": The detailed technical solution.
- "past_performance": Case studies and evidence.
- "compliance_matrix": A statement of compliance with tender requirements.
"""
    user = f"TENDER: {state['tender_data']}\n\nANALYST: {state['analysis_notes']}\n\nRESEARCH: {state['research_notes']}\n\nPREVIOUS FEEDBACK: {state.get('feedback', 'None')}\n\nPREVIOUS DRAFT: {state.get('draft_proposal', 'None')}"
    draft = call_llm(groq_client, system, user, response_model=ProposalOutput)
    return {"draft_proposal": draft, "revision_count": state.get("revision_count", 0) + 1}

def reviewer_node(state: ProposalState, groq_client: Groq) -> Dict[str, Any]:
    print("[Agent] Reviewer is evaluating...")
    system = """You are a strict proposal reviewer. Evaluate the draft against the tender requirements.
If the draft is excellent and ready to submit, output exactly 'APPROVED'.
If it needs work, provide detailed feedback on what needs to be changed. Do NOT output 'APPROVED' if changes are needed."""
    user = f"TENDER: {state['tender_data']}\n\nDRAFT: {state['draft_proposal']}"
    feedback = call_llm(groq_client, system, user)
    
    if "APPROVED" in feedback.upper() or state.get("revision_count", 0) >= 2:
        return {"final_proposal": state["draft_proposal"], "feedback": "APPROVED"}
    else:
        return {"feedback": feedback}

def should_continue(state: ProposalState) -> str:
    if state.get("feedback") == "APPROVED":
        return "end"
    return "continue"

def build_proposal_graph(groq_client: Groq):
    workflow = StateGraph(ProposalState)
    
    # We use lambda or partial to pass the groq_client
    workflow.add_node("analyst", lambda state: analyst_node(state, groq_client))
    workflow.add_node("researcher", lambda state: researcher_node(state, groq_client))
    workflow.add_node("writer", lambda state: writer_node(state, groq_client))
    workflow.add_node("reviewer", lambda state: reviewer_node(state, groq_client))
    
    workflow.set_entry_point("analyst")
    workflow.add_edge("analyst", "researcher")
    workflow.add_edge("researcher", "writer")
    workflow.add_edge("writer", "reviewer")
    
    workflow.add_conditional_edges(
        "reviewer",
        should_continue,
        {
            "continue": "writer",
            "end": END
        }
    )
    
    return workflow.compile()

def generate_proposal(tender_data: Dict[str, Any], company_profile: Dict[str, Any], groq_client: Groq) -> Dict[str, Any]:
    graph = build_proposal_graph(groq_client)
    initial_state = ProposalState(
        tender_data=tender_data,
        company_profile=company_profile,
        analysis_notes="",
        research_notes="",
        draft_proposal="",
        final_proposal="",
        feedback="",
        revision_count=0
    )
    
    result = graph.invoke(initial_state)
    import json
    
    final_output = result.get("final_proposal", result.get("draft_proposal", ""))
    
    if isinstance(final_output, BaseModel):
        return final_output.model_dump()
        
    if isinstance(final_output, str):
        try:
            return json.loads(final_output)
        except json.JSONDecodeError:
            return {"error": "Failed to parse final proposal as JSON", "raw_content": final_output}
    
    return final_output
