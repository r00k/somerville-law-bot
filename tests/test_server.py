from fastapi.testclient import TestClient

from app import server


def test_api_ask_reports_missing_anthropic_configuration(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    response = TestClient(server.app).post(
        "/api/ask", json={"question": "Can I keep hens?"}
    )

    assert response.status_code == 503
    assert response.json() == {
        "error": "anthropic_not_configured",
        "message": (
            "The Q&A service is not configured. For local development, "
            "add ANTHROPIC_API_KEY to the repository's .env file and restart the server."
        ),
    }
    assert not server._ip_requests
    assert not server._global_requests
