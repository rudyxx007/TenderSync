import pytest
from unittest.mock import MagicMock
from bid_engine import (
    run_bid_evaluation,
    evaluate_hard_gates,
    _parse_monetary_value,
    compute_pwin,
    make_decision,
    normalize_company_profile
)

def test_parse_monetary_value():
    assert _parse_monetary_value("$500,000") == 500000.0
    assert _parse_monetary_value("₹ 10,00,000") == 1000000.0
    assert _parse_monetary_value("₹ 50 Lakhs") == 5000000.0
    assert _parse_monetary_value("Rs. 2.5 Crore") == 25000000.0
    assert _parse_monetary_value("€ 250,000") == 250000.0
    assert _parse_monetary_value("£ 75,000") == 75000.0
    assert _parse_monetary_value(None) is None
    assert _parse_monetary_value("Negotiable") is None

def test_evaluate_hard_gates_pass(fake_tender_data, fake_company_profile):
    gates = evaluate_hard_gates(fake_tender_data, fake_company_profile)
    assert all(g.passed for g in gates)

def test_evaluate_hard_gates_missing_cert(fake_tender_data, fake_company_profile):
    fake_tender_data["mandatory_compliance_criteria"] = ["CMMI Level 5"]
    fake_company_profile["active_certifications"] = ["ISO 27001"]
    gates = evaluate_hard_gates(fake_tender_data, fake_company_profile)
    failed = [g for g in gates if not g.passed]
    assert any(g.gate_name == "mandatory_certification" for g in failed)

def test_evaluate_hard_gates_below_min_value(fake_tender_data, fake_company_profile):
    fake_company_profile["min_contract_value"] = 10000000  # $10M min, tender is $500k
    gates = evaluate_hard_gates(fake_tender_data, fake_company_profile)
    failed = [g for g in gates if not g.passed]
    assert any(g.gate_name == "minimum_contract_value" for g in failed)

def test_evaluate_hard_gates_above_max_value(fake_tender_data, fake_company_profile):
    fake_company_profile["max_contract_value"] = 100000  # $100k max, tender is $500k
    gates = evaluate_hard_gates(fake_tender_data, fake_company_profile)
    failed = [g for g in gates if not g.passed]
    assert any(g.gate_name == "maximum_contract_value" for g in failed)

def test_evaluate_hard_gates_low_confidence(fake_tender_data, fake_company_profile):
    fake_tender_data["confidence_score"] = 0.20  # Below 0.50 threshold
    gates = evaluate_hard_gates(fake_tender_data, fake_company_profile)
    failed = [g for g in gates if not g.passed]
    assert any(g.gate_name == "extraction_confidence" for g in failed)

def test_compute_pwin_and_decision():
    scores = {
        "capability_fit": 5,
        "compliance_readiness": 5,
        "past_performance": 5,
        "competitive_position": 5,
        "commercial_viability": 5,
        "resource_capacity": 5,
        "strategic_alignment": 5,
    }
    pwin = compute_pwin(scores)
    assert pwin == 100
    decision, reasons = make_decision(pwin, [])
    assert decision == "BID"

def test_run_bid_evaluation_full_bid(fake_tender_data, fake_company_profile):
    mock_groq = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = '{"strategic_alignment_score": 5, "strategic_alignment_rationale": "High fit", "competitor_density_score": 4, "competitor_density_rationale": "Moderate", "pricing_feasibility_score": 4, "pricing_feasibility_rationale": "Viable", "risk_rating_score": 4, "risk_rating_rationale": "Low risk"}'
    mock_groq.chat.completions.create.return_value.choices = [mock_choice]
    
    result = run_bid_evaluation(fake_tender_data, fake_company_profile, mock_groq)
    assert result["decision"] in ["BID", "CONDITIONAL", "NO-BID"]
    assert "factor_scores" in result
    assert "hard_gates" in result

def test_run_bid_evaluation_hard_gate_no_bid(fake_tender_data, fake_company_profile):
    fake_company_profile["min_contract_value"] = 10000000
    mock_groq = MagicMock()
    result = run_bid_evaluation(fake_tender_data, fake_company_profile, mock_groq)
    assert result["decision"] == "NO-BID"
    assert any(not g["passed"] for g in result["hard_gates"])
