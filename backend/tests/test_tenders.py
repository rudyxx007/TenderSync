import pytest

def test_list_tenders(client, mock_external_services):
    mock_sb = mock_external_services
    mock_sb.table.return_value.select.return_value.eq.return_value.order.return_value.execute.return_value.data = [
        {
            "id": "analysis-123",
            "tender_id": "T-123",
            "filename": "test.pdf",
            "issuing_authority": "Test Auth",
            "decision": "BID",
            "win_probability_score": 85,
            "confidence_score": 0.9,
            "created_at": "2026-08-11"
        }
    ]
    response = client.get("/api/tenders")
    assert response.status_code == 200
    assert len(response.json()) == 1
    assert response.json()[0]["decision"] == "BID"

def test_list_tenders_unauthenticated(unauthenticated_client):
    response = unauthenticated_client.get("/api/tenders")
    assert response.status_code == 401

def test_get_tender_detail_success(client, mock_external_services):
    mock_sb = mock_external_services
    mock_sb.table.return_value.select.return_value.eq.return_value.eq.return_value.execute.return_value.data = [
        {
            "id": "analysis-123",
            "user_id": "test-user-id",
            "filename": "test.pdf",
            "extracted_data": {},
            "evaluation_data": {},
            "created_at": "2026-08-11"
        }
    ]
    response = client.get("/api/tenders/analysis-123")
    assert response.status_code == 200
    assert response.json()["id"] == "analysis-123"

def test_get_tender_detail_not_found(client, mock_external_services):
    mock_sb = mock_external_services
    mock_sb.table.return_value.select.return_value.eq.return_value.eq.return_value.execute.return_value.data = []
    response = client.get("/api/tenders/non-existent-id")
    assert response.status_code == 404

def test_generate_calendar_success(client):
    response = client.post("/api/generate-calendar?deadline_string=2026-12-31")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/calendar")
    assert "BEGIN:VCALENDAR" in response.text

def test_generate_calendar_unauthenticated(unauthenticated_client):
    response = unauthenticated_client.post("/api/generate-calendar?deadline_string=2026-12-31")
    assert response.status_code == 401
