# Désactive Prometheus
import os
import time
import jwt
import numpy as np
import pytest
from starlette.testclient import TestClient
from service import rf_service, create_jwt_token, JWT_SECRET_KEY, JWT_ALGORITHM

@pytest.fixture(scope="module")
def client():
    app = rf_service.asgi_app
    with TestClient(app) as c:
        yield c


@pytest.fixture(autouse=True)
def mock_runner(monkeypatch):
    from src import service as service_module

    async def fake_async_run(arr):
        return np.array([0.4242])

    monkeypatch.setattr(
        service_module.students_rf_runner,
        "async_run",
        fake_async_run,
        raising=True,
    )
    yield


def valid_payload():
    return {
        "GRE_score": 311,
        "TOEFL_score": 105,
        "University_Rating": 2,
        "SOP": 3.0,
        "LOR": 2.0,
        "CGPA": 8.12,
        "Research": 1,
    }

def test_login_success_returns_token(client: TestClient):
    resp = client.post("/login", json={"username": "user123", "password": "password123"})
    assert resp.status_code == 200, resp.text
    token = resp.json().get("token")
    assert token, "Token manquant dans la réponse"
    payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
    assert payload.get("sub") == "user123"


def test_login_wrong_credentials_returns_401(client: TestClient):
    resp = client.post("/login", json={"username": "user123", "password": "wrong"})
    assert resp.status_code == 401
    assert resp.json().get("detail")


def test_auth_fails_when_token_missing(client: TestClient):
    resp = client.post("/v1/models/rf_regressor/predict", json=valid_payload())
    assert resp.status_code == 401
    assert resp.json().get("detail") == "Missing authentication token"


def test_auth_fails_when_token_invalid(client: TestClient):
    headers = {"Authorization": "Bearer invalid.token.value"}
    resp = client.post("/v1/models/rf_regressor/predict", json=valid_payload(), headers=headers)
    assert resp.status_code == 401
    assert resp.json().get("detail") == "Invalid token"


def test_auth_fails_when_token_expired(client: TestClient):
    # Crée un token expiré (exp dans le passé)
    expired_payload = {"sub": "user123", "exp": int(time.time()) - 60}
    expired_token = jwt.encode(expired_payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    headers = {"Authorization": f"Bearer {expired_token}"}
    resp = client.post("/v1/models/rf_regressor/predict", json=valid_payload(), headers=headers)
    assert resp.status_code == 401
    assert resp.json().get("detail") == "Token has expired"


def test_prediction_succeeds_with_valid_token_and_input(client: TestClient):
    token = create_jwt_token("user123")
    headers = {"Authorization": f"Bearer {token}"}
    resp = client.post("/v1/models/rf_regressor/predict", json=valid_payload(), headers=headers)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "prediction" in body
    assert isinstance(body["prediction"], list)
    assert len(body["prediction"]) == 1
    assert pytest.approx(body["prediction"][0], rel=1e-6) == 0.4242
    assert body.get("user") == "user123"


def test_prediction_fails_with_invalid_input(client: TestClient):
    token = create_jwt_token("user123")
    headers = {"Authorization": f"Bearer {token}"}
    # test sur donnees invalide
    invalid = {
        "GRE_score": "not-an-int", # text sur integer
        # "TOEFL_score" on enleve un champ
        "University_Rating": 2,
        "SOP": 3.0,
        "LOR": 2.0,
        "CGPA": 8.12,
        "Research": 1,
    }
    resp = client.post("/v1/models/rf_regressor/predict", json=invalid, headers=headers)
    # 400 ou 422
    assert resp.status_code in (400, 422)
