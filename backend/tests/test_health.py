def test_health_check(client):
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "TenderSync Enterprise Pipeline Operational"
    assert "version" in data
    assert "auth" in data
