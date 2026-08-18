import pytest

def test_list_market_tenders_success(client, mock_external_services):
    mock_sb = mock_external_services
    mock_sb.table.return_value.select.return_value.order.return_value.range.return_value.execute.return_value.data = [
        {"id": "market-1", "title": "Government Cloud Procurement"}
    ]
    response = client.get("/api/market-tenders")
    assert response.status_code == 200
    assert len(response.json()) == 1

def test_list_market_tenders_unauthenticated(unauthenticated_client):
    response = unauthenticated_client.get("/api/market-tenders")
    assert response.status_code == 401

def test_evaluate_market_tender_not_found(client, mock_external_services):
    mock_sb = mock_external_services
    mock_sb.table.return_value.select.return_value.eq.return_value.execute.return_value.data = []
    response = client.post("/api/market-tenders/non-existent-id/evaluate")
    assert response.status_code == 404

def test_evaluate_market_tender_501(client, mock_external_services):
    mock_sb = mock_external_services
    mock_sb.table.return_value.select.return_value.eq.return_value.execute.return_value.data = [{"id": "market-123"}]
    response = client.post("/api/market-tenders/market-123/evaluate")
    assert response.status_code == 501
