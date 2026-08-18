import pytest
from report_generator import render_evaluation_report, generate_radar_chart

def test_generate_radar_chart():
    factor_scores = {"Technical": 4.5, "Financial": 4.0, "Risk": 3.5}
    png_bytes = generate_radar_chart(factor_scores)
    assert isinstance(png_bytes, bytes)
    assert len(png_bytes) > 0

def test_render_evaluation_report(fake_tender_data, fake_company_profile):
    analysis_data = {
        "filename": "test.pdf",
        "created_at": "2026-08-17",
        "extracted_data": fake_tender_data,
        "evaluation_data": {
            "decision": "BID",
            "win_probability_score": 90,
            "rationale": "High strategic alignment and capacity.",
            "factor_scores": {"Technical": 4, "Financial": 5, "Compliance": 5},
            "hard_gates": [
                {"gate_name": "minimum_contract_value", "passed": True},
                {"gate_name": "mandatory_certification", "passed": True}
            ]
        }
    }
    
    pdf_bytes = render_evaluation_report(analysis_data, fake_company_profile)
    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 1000
    assert pdf_bytes.startswith(b"%PDF")
