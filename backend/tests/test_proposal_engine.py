import pytest
from unittest.mock import MagicMock
from proposal_engine import analyst_node, researcher_node, writer_node, reviewer_node, should_continue, ProposalOutput

def test_analyst_node():
    mock_groq = MagicMock()
    mock_groq.chat.completions.create.return_value.choices[0].message.content = "Analysis notes"
    state = {
        "tender_data": {"id": "1"},
        "company_profile": {"org_id": "1"}
    }
    result = analyst_node(state, mock_groq)
    assert result["analysis_notes"] == "Analysis notes"

def test_researcher_node():
    mock_groq = MagicMock()
    mock_groq.chat.completions.create.return_value.choices[0].message.content = "Research notes"
    state = {
        "analysis_notes": "Analysis notes",
        "company_profile": {"org_id": "1"}
    }
    result = researcher_node(state, mock_groq)
    assert result["research_notes"] == "Research notes"

def test_writer_node():
    mock_groq = MagicMock()
    expected_output = ProposalOutput(
        executive_summary="Summary",
        technical_approach="Approach",
        past_performance="Case studies",
        compliance_matrix="Compliant"
    )
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = expected_output
    
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("instructor.from_groq", lambda g, mode: mock_client)
        state = {
            "tender_data": {"id": "1"},
            "analysis_notes": "Notes",
            "research_notes": "Research",
            "company_profile": {"org_id": "1"}
        }
        result = writer_node(state, mock_groq)
        assert result["revision_count"] == 1
        assert "draft_proposal" in result

def test_reviewer_node_approved():
    mock_groq = MagicMock()
    mock_groq.chat.completions.create.return_value.choices[0].message.content = "APPROVED"
    state = {
        "tender_data": {"id": "1"},
        "draft_proposal": "Draft content",
        "revision_count": 1
    }
    result = reviewer_node(state, mock_groq)
    assert result["feedback"] == "APPROVED"
    assert result["final_proposal"] == "Draft content"

def test_reviewer_node_feedback_loop():
    mock_groq = MagicMock()
    mock_groq.chat.completions.create.return_value.choices[0].message.content = "Add more detail on ISO 27001."
    state = {
        "tender_data": {"id": "1"},
        "draft_proposal": "Draft content",
        "revision_count": 1
    }
    result = reviewer_node(state, mock_groq)
    assert "feedback" in result
    assert "APPROVED" not in result.get("final_proposal", "")

def test_should_continue_logic():
    assert should_continue({"feedback": "APPROVED"}) == "end"
    assert should_continue({"feedback": "Needs changes"}) == "continue"
