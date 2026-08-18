import pytest

def test_create_org_returns_invite_code(client, mock_external_services):
    mock_sb = mock_external_services
    mock_sb.table.return_value.select.return_value.eq.return_value.execute.return_value.data = []
    mock_sb.table.return_value.insert.return_value.execute.return_value.data = [
        {"id": "org-123", "name": "Acme Corp", "invite_code": "a3f7c2e1"}
    ]
    response = client.post("/api/orgs", json={"name": "Acme Corp"})
    assert response.status_code == 200
    assert "invite_code" in response.json()
    assert response.json()["name"] == "Acme Corp"

def test_create_org_validation_error(client):
    response = client.post("/api/orgs", json={})
    assert response.status_code == 422

def test_join_org_success(client, mock_external_services):
    mock_sb = mock_external_services
    # 1st call for existing membership check: returns []
    # 2nd call for find org by invite_code: returns [org]
    mock_sb.table.return_value.select.return_value.eq.return_value.execute.side_effect = [
        type("Result", (), {"data": []})(),
        type("Result", (), {"data": [{"id": "org-123", "name": "Acme Corp", "invite_code": "a3f7c2e1"}]})()
    ]
    mock_sb.table.return_value.insert.return_value.execute.return_value.data = [
        {"id": "mem-123", "org_id": "org-123", "user_id": "test-user-id", "role": "member"}
    ]
    response = client.post("/api/orgs/join", json={"invite_code": "a3f7c2e1"})
    assert response.status_code == 200
    assert response.json()["id"] == "org-123"

def test_join_org_invalid_code_returns_404(client, mock_external_services):
    mock_sb = mock_external_services
    mock_sb.table.return_value.select.return_value.eq.return_value.execute.side_effect = [
        type("Result", (), {"data": []})(),
        type("Result", (), {"data": []})()
    ]
    response = client.post("/api/orgs/join", json={"invite_code": "BADCODE1"})
    assert response.status_code == 404

def test_list_org_members(client, mock_external_services):
    mock_sb = mock_external_services
    mock_sb.table.return_value.select.return_value.eq.return_value.execute.side_effect = None
    mock_sb.table.return_value.select.return_value.eq.return_value.execute.return_value.data = [
        {"id": "m-1", "user_id": "u-1", "role": "owner"}
    ]
    response = client.get("/api/orgs/members")
    assert response.status_code == 200
    assert len(response.json()) == 1

def test_regenerate_invite_code_owner(client, mock_external_services):
    mock_sb = mock_external_services
    mock_sb.table.return_value.select.return_value.eq.return_value.execute.side_effect = None
    mock_sb.table.return_value.update.return_value.eq.return_value.execute.return_value.data = [
        {"id": "test-org-id", "name": "Test Org", "invite_code": "newcode1"}
    ]
    response = client.post("/api/orgs/invite/regenerate")
    assert response.status_code == 200
    assert "invite_code" in response.json()

def test_regenerate_invite_code_member_forbidden(member_client):
    response = member_client.post("/api/orgs/invite/regenerate")
    assert response.status_code == 403
