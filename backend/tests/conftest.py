import os
import pytest
import sys
from unittest.mock import MagicMock, AsyncMock, patch
from fastapi.testclient import TestClient

# Mock docling entirely to prevent DLL load errors on Windows
sys.modules['docling'] = MagicMock()
sys.modules['docling.document_converter'] = MagicMock()

# Set dummy env vars so main.py doesn't crash on import
os.environ["SUPABASE_URL"] = "http://fake.supabase.co"
os.environ["SUPABASE_SERVICE_KEY"] = "fake-service-key"
os.environ["HF_TOKEN"] = "fake-hf-token"
os.environ["GROQ_API_KEY"] = "fake-groq-key"
os.environ["ALLOW_DEV_BYPASS"] = "false"

@pytest.fixture(autouse=True)
def mock_external_services(monkeypatch):
    """Mock all external API clients so no real network calls are made."""
    mock_sb = MagicMock()
    monkeypatch.setattr("main.supabase_client", mock_sb)
    monkeypatch.setattr("db.supabase_client", mock_sb)
    
    monkeypatch.setattr("main.generate_embeddings", lambda texts, task_type="passage": [[0.1] * 1024 for _ in texts])
    monkeypatch.setattr("embedding_service.generate_embeddings", lambda texts, task_type="passage": [[0.1] * 1024 for _ in texts])
    
    mock_groq = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = '{"tender_id": "T-100", "issuing_authority": "Gov", "submission_deadline": "2026-12-31", "estimated_value_or_budget": "$100,000", "key_deliverables": ["Audit"], "mandatory_compliance_criteria": ["ISO 27001"], "confidence_score": 0.95}'
    mock_groq.chat.completions.create.return_value.choices = [mock_choice]
    monkeypatch.setattr("main.groq_client", mock_groq)
    monkeypatch.setattr("db.groq_client", mock_groq)
    
    mock_docling_conv = MagicMock()
    mock_conv_result = MagicMock()
    mock_conv_result.document.export_to_markdown.return_value = "# Tender Document\nBudget: $100,000\nDeadline: 2026-12-31\nRequirement: ISO 27001"
    mock_docling_conv.return_value.convert.return_value = mock_conv_result
    monkeypatch.setattr("main.DocumentConverter", mock_docling_conv)
    monkeypatch.setattr("document_extractor.DocumentConverter", mock_docling_conv)
    
    return mock_sb

@pytest.fixture
def fake_tender_data():
    return {
        "tender_id": "RFP-2026-001",
        "issuing_authority": "Ministry of IT",
        "submission_deadline": "2026-12-31",
        "estimated_value_or_budget": "$500,000",
        "key_deliverables": ["Cloud migration", "Security audit"],
        "mandatory_compliance_criteria": ["ISO 27001", "SOC 2"],
        "confidence_score": 0.92
    }

@pytest.fixture
def fake_company_profile():
    return {
        "org_id": "test-org-id",
        "organization_name": "Test Org",
        "min_contract_value": 10000,
        "max_contract_value": 5000000,
        "active_certifications": ["ISO 27001", "SOC 2"],
        "core_capabilities": ["Cloud migration", "Security audit"],
        "strategic_focus_areas": ["Public Sector"],
        "past_performance_sectors": ["Government"],
        "geographic_coverage": ["National"],
        "insurance_coverage": {},
        "min_bid_lead_time_days": 14,
        "team_capacity_score": 5,
        "relationship_strength_score": 5
    }

@pytest.fixture
def client(fake_company_profile, monkeypatch):
    """Create a FastAPI TestClient with authenticated owner dependencies."""
    from main import app, get_current_user_id, get_current_user_org
    import main
    
    # Mock require_complete_profile to return a valid profile by default
    monkeypatch.setattr("main.require_complete_profile", lambda sb, org_id: fake_company_profile)
    monkeypatch.setattr("profile_service.require_complete_profile", lambda sb, org_id: fake_company_profile)
    
    app.dependency_overrides[get_current_user_id] = lambda: "test-user-id"
    app.dependency_overrides[get_current_user_org] = lambda: {
        "org_id": "test-org-id", "role": "owner", "user_id": "test-user-id"
    }
    
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()

@pytest.fixture
def member_client(fake_company_profile, monkeypatch):
    """Create a FastAPI TestClient with 'member' role (for RBAC testing)."""
    from main import app, get_current_user_id, get_current_user_org
    
    monkeypatch.setattr("main.require_complete_profile", lambda sb, org_id: fake_company_profile)
    monkeypatch.setattr("profile_service.require_complete_profile", lambda sb, org_id: fake_company_profile)
    
    app.dependency_overrides[get_current_user_id] = lambda: "member-user-id"
    app.dependency_overrides[get_current_user_org] = lambda: {
        "org_id": "test-org-id", "role": "member", "user_id": "member-user-id"
    }
    
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()

@pytest.fixture
def unauthenticated_client():
    """Create a FastAPI TestClient without authentication overrides."""
    from main import app
    app.dependency_overrides.clear()
    with TestClient(app) as test_client:
        yield test_client
