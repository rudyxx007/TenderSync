import pytest

def test_profile_status_complete(client, mock_external_services, fake_company_profile):
    mock_sb = mock_external_services
    mock_sb.table.return_value.select.return_value.eq.return_value.execute.return_value.data = [fake_company_profile]
    response = client.get("/api/profile/status")
    assert response.status_code == 200
    assert response.json()["is_complete"] is True
    assert response.json()["completion_percentage"] > 50
    assert len(response.json()["missing_fields"]) == 0

def test_profile_status_incomplete(client, mock_external_services):
    mock_sb = mock_external_services
    mock_sb.table.return_value.select.return_value.eq.return_value.execute.return_value.data = []
    response = client.get("/api/profile/status")
    assert response.status_code == 200
    assert response.json()["is_complete"] is False
    assert response.json()["completion_percentage"] == 0
    assert len(response.json()["missing_fields"]) > 0

def test_get_profile_success(client, mock_external_services, fake_company_profile):
    mock_sb = mock_external_services
    mock_sb.table.return_value.select.return_value.eq.return_value.execute.return_value.data = [fake_company_profile]
    response = client.get("/api/profile")
    assert response.status_code == 200
    assert response.json()["organization_name"] == "Test Org"

def test_put_profile_owner_success(client, mock_external_services, fake_company_profile):
    mock_sb = mock_external_services
    mock_sb.table.return_value.select.return_value.eq.return_value.execute.return_value.data = [fake_company_profile]
    mock_sb.table.return_value.upsert.return_value.execute.return_value.data = [fake_company_profile]
    
    payload = {
        "organization_name": "New Name",
        "min_contract_value": 50000,
        "max_contract_value": 1000000,
        "active_certifications": ["ISO 9001"],
        "core_capabilities": ["DevOps"],
        "min_bid_lead_time_days": 10,
        "team_capacity_score": 4,
        "relationship_strength_score": 3,
        "strategic_focus_areas": [],
        "past_performance_sectors": [],
        "geographic_coverage": [],
        "insurance_coverage": {}
    }
    
    response = client.put("/api/profile", json=payload)
    assert response.status_code == 200

def test_put_profile_member_forbidden(member_client):
    payload = {
        "organization_name": "Test Org",
        "min_contract_value": 10000,
        "max_contract_value": 1000000,
        "active_certifications": ["ISO 9001"],
        "core_capabilities": ["DevOps"],
        "min_bid_lead_time_days": 10,
        "team_capacity_score": 4,
        "relationship_strength_score": 3,
        "strategic_focus_areas": [],
        "past_performance_sectors": [],
        "geographic_coverage": [],
        "insurance_coverage": {}
    }
    response = member_client.put("/api/profile", json=payload)
    assert response.status_code == 403

def test_put_profile_validation_error(client):
    response = client.put("/api/profile", json={"invalid": "payload"})
    assert response.status_code == 422
