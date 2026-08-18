import pytest
from unittest.mock import MagicMock, patch

def test_compare_tenders_validation_error(client):
    # Needs at least 2 IDs
    response = client.post("/api/tenders/compare", json={"tender_ids": ["t1"]})
    assert response.status_code == 422

def test_compare_tenders_missing_tenders_returns_400(client, mock_external_services):
    mock_sb = mock_external_services
    mock_sb.table.return_value.select.return_value.eq.return_value.eq.return_value.execute.return_value.data = []
    response = client.post("/api/tenders/compare", json={"tender_ids": ["t1", "t2"]})
    assert response.status_code == 400

def test_compare_tenders_unauthenticated(unauthenticated_client):
    response = unauthenticated_client.post("/api/tenders/compare", json={"tender_ids": ["t1", "t2"]})
    assert response.status_code == 401

def test_compare_tenders_success(client, mock_external_services):
    mock_sb = mock_external_services
    mock_sb.table.return_value.select.return_value.eq.return_value.eq.return_value.execute.return_value.data = [
        {"id": "t1", "filename": "tender1.pdf", "extracted_data": {}, "evaluation_data": {}}
    ]
    
    with patch("comparison_engine.compare_tenders") as mock_cmp:
        mock_cmp.return_value = {
            "budget_comparison": "Tender 1 is larger",
            "deadline_comparison": "Tender 2 is sooner",
            "compliance_differences": "ISO 27001 required on Tender 1",
            "deliverable_differences": "Cloud migration vs Security audit",
            "recommendation": "Bid on Tender 1"
        }
        response = client.post("/api/tenders/compare", json={"tender_ids": ["t1", "t2"]})
        assert response.status_code == 200
        assert "recommendation" in response.json()
