import pytest
from unittest.mock import MagicMock
from fastapi import HTTPException
from auth import resolve_user_id, resolve_user_org

def test_get_me_success(client):
    response = client.get("/api/auth/me")
    assert response.status_code == 200
    assert response.json()["user_id"] == "test-user-id"

def test_get_me_unauthenticated(unauthenticated_client):
    response = unauthenticated_client.get("/api/auth/me")
    assert response.status_code == 401

def test_resolve_user_id_valid_jwt():
    mock_sb = MagicMock()
    mock_user = MagicMock()
    mock_user.id = "real-user-123"
    mock_sb.auth.get_user.return_value.user = mock_user
    
    uid = resolve_user_id(mock_sb, "valid.jwt.token")
    assert uid == "real-user-123"

def test_resolve_user_id_invalid_jwt():
    mock_sb = MagicMock()
    mock_sb.auth.get_user.side_effect = Exception("JWT signature invalid")
    
    with pytest.raises(HTTPException) as exc_info:
        resolve_user_id(mock_sb, "invalid.token")
    assert exc_info.value.status_code == 401

def test_resolve_user_org_no_org():
    mock_sb = MagicMock()
    mock_sb.table.return_value.select.return_value.eq.return_value.limit.return_value.execute.return_value.data = []
    
    with pytest.raises(HTTPException) as exc_info:
        resolve_user_org(mock_sb, "user-without-org")
    assert exc_info.value.status_code == 403
