import pytest

def test_get_batch_status_success(client, mock_external_services):
    mock_sb = mock_external_services
    mock_sb.table.return_value.select.return_value.eq.return_value.eq.return_value.execute.return_value.data = [
        {"id": "batch-1", "status": "complete", "total_files": 2, "processed_files": 2}
    ]
    response = client.get("/api/batches/batch-1")
    assert response.status_code == 200
    assert response.json()["status"] == "complete"

def test_get_batch_status_not_found(client, mock_external_services):
    mock_sb = mock_external_services
    mock_sb.table.return_value.select.return_value.eq.return_value.eq.return_value.execute.return_value.data = []
    response = client.get("/api/batches/non-existent-batch")
    assert response.status_code == 404

def test_get_batch_status_unauthenticated(unauthenticated_client):
    response = unauthenticated_client.get("/api/batches/batch-1")
    assert response.status_code == 401
