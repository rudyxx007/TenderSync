import pytest
import io

def test_process_tender_success(client, mock_external_services, fake_tender_data):
    mock_sb = mock_external_services
    mock_sb.table.return_value.insert.return_value.execute.return_value.data = [{"id": "analysis-123"}]
    mock_sb.rpc.return_value.execute.return_value.data = [{"content": "Test document chunk"}]
    
    file_content = b"%PDF-1.4\nTest PDF content"
    files = {"file": ("test.pdf", file_content, "application/pdf")}
    
    response = client.post("/api/process-tender", files=files)
    assert response.status_code == 200
    assert response.json()["status"] == "success"

def test_process_tender_image_success(client, mock_external_services, fake_tender_data, monkeypatch):
    monkeypatch.setattr("main.extract_document_text", lambda path, fname: "Sample tender extracted image text")
    monkeypatch.setattr("document_extractor.extract_document_text", lambda path, fname: "Sample tender extracted image text")
    mock_sb = mock_external_services
    mock_sb.table.return_value.insert.return_value.execute.return_value.data = [{"id": "analysis-123"}]
    mock_sb.rpc.return_value.execute.return_value.data = [{"content": "Test document chunk"}]
    
    file_content = b"fake image bytes"
    files = {"file": ("tender_scan.png", file_content, "image/png")}
    
    response = client.post("/api/process-tender", files=files)
    assert response.status_code == 200
    assert response.json()["status"] == "success"

def test_process_tender_rejects_unsupported_format(client):
    file_content = b"Plain text file"
    files = {"file": ("test.txt", file_content, "text/plain")}
    response = client.post("/api/process-tender", files=files)
    assert response.status_code == 400
    assert "Unsupported file format" in response.json()["detail"]

def test_process_tender_incomplete_profile_blocks(client, monkeypatch):
    monkeypatch.setattr(
        "main.require_complete_profile",
        lambda sb, org_id: (_ for _ in ()).throw(ValueError("Complete your company profile first."))
    )
    monkeypatch.setattr(
        "profile_service.require_complete_profile",
        lambda sb, org_id: (_ for _ in ()).throw(ValueError("Complete your company profile first."))
    )
    file_content = b"%PDF-1.4\nTest PDF content"
    files = {"file": ("test.pdf", file_content, "application/pdf")}
    response = client.post("/api/process-tender", files=files)
    assert response.status_code == 403
    assert "Complete your company profile" in response.json()["detail"]

def test_process_tender_unauthenticated(unauthenticated_client):
    file_content = b"%PDF-1.4\nTest PDF content"
    files = {"file": ("test.pdf", file_content, "application/pdf")}
    response = unauthenticated_client.post("/api/process-tender", files=files)
    assert response.status_code == 401

def test_process_tender_batch_success(client, mock_external_services):
    mock_sb = mock_external_services
    mock_sb.table.return_value.insert.return_value.execute.return_value.data = [{"id": "batch-123"}]
    
    file_content = b"%PDF-1.4\nTest PDF content"
    files = [
        ("files", ("test1.pdf", file_content, "application/pdf")),
        ("files", ("test2.pdf", file_content, "application/pdf"))
    ]
    
    response = client.post("/api/process-tender/batch", files=files)
    assert response.status_code == 200
    assert response.json()["status"] == "processing"
    assert response.json()["batch_id"] == "batch-123"

def test_process_tender_batch_empty_files_rejected(client):
    response = client.post("/api/process-tender/batch", files=[])
    assert response.status_code in [400, 422]
