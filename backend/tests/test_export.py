import pytest
from unittest.mock import patch, MagicMock

def test_export_pdf_404(client, mock_external_services):
    mock_sb = mock_external_services
    mock_sb.table.return_value.select.return_value.eq.return_value.eq.return_value.execute.return_value.data = []
    response = client.get("/api/tenders/analysis-123/export-pdf")
    assert response.status_code == 404

def test_export_pdf_unauthenticated(unauthenticated_client):
    response = unauthenticated_client.get("/api/tenders/analysis-123/export-pdf")
    assert response.status_code == 401

def test_export_pdf_success(client, mock_external_services):
    mock_sb = mock_external_services
    mock_sb.table.return_value.select.return_value.eq.return_value.eq.return_value.execute.return_value.data = [
        {
            "id": "analysis-123",
            "filename": "tender.pdf",
            "extracted_data": {},
            "evaluation_data": {}
        }
    ]
    with patch("report_generator.render_evaluation_report", return_value=b"%PDF-1.4 Mock PDF content"):
        response = client.get("/api/tenders/analysis-123/export-pdf")
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/pdf"
        assert response.content.startswith(b"%PDF")
