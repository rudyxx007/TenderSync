import pytest
from unittest.mock import MagicMock, patch

def test_list_proposals(client, mock_external_services):
    mock_sb = mock_external_services
    mock_sb.table.return_value.select.return_value.eq.return_value.order.return_value.execute.return_value.data = [
        {"id": "prop-1", "title": "Test Proposal"}
    ]
    response = client.get("/api/proposals")
    assert response.status_code == 200
    assert len(response.json()) == 1

def test_list_proposals_unauthenticated(unauthenticated_client):
    response = unauthenticated_client.get("/api/proposals")
    assert response.status_code == 401

def test_get_proposal_success(client, mock_external_services):
    mock_sb = mock_external_services
    mock_sb.table.return_value.select.return_value.eq.return_value.eq.return_value.execute.return_value.data = [
        {"id": "prop-1", "title": "Test Proposal"}
    ]
    response = client.get("/api/proposals/prop-1")
    assert response.status_code == 200
    assert response.json()["id"] == "prop-1"

def test_get_proposal_not_found(client, mock_external_services):
    mock_sb = mock_external_services
    mock_sb.table.return_value.select.return_value.eq.return_value.eq.return_value.execute.return_value.data = []
    response = client.get("/api/proposals/non-existent-prop")
    assert response.status_code == 404

def test_generate_proposal_tender_not_found(client, mock_external_services):
    mock_sb = mock_external_services
    mock_sb.table.return_value.select.return_value.eq.return_value.eq.return_value.execute.return_value.data = []
    response = client.post("/api/proposals/generate", json={"analysis_id": "non-existent-tender"})
    assert response.status_code == 404
