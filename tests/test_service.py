# Tests d'API: utilisent un serveur BentoML déjà démarré (comme src/test.py)
# Base URL configurable via la variable d'environnement API_BASE_URL
import os
import time
import jwt
import pytest
import requests
import socket
from urllib.parse import urlparse
from service import create_jwt_token, JWT_SECRET_KEY, JWT_ALGORITHM

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:3001")
LOGIN_URL = f"{API_BASE_URL}/login"
PREDICT_URL = f"{API_BASE_URL}/v1/models/rf_regressor/predict"


@pytest.fixture(scope="module", autouse=True)
def ensure_server():
    parsed = urlparse(API_BASE_URL)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or (443 if (parsed.scheme or "http") == "https" else 80)
    try:
        with socket.create_connection((host, port), timeout=3):
            pass
    except OSError:
        pytest.skip(f"API server not running at {API_BASE_URL}. Set API_BASE_URL or start the BentoML service.")


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


def test_login_success_returns_token():
    resp = requests.post(
        LOGIN_URL,
        headers={"Content-Type": "application/json"},
        json={"username": "user123", "password": "password123"},
        timeout=10,
    )
    assert resp.status_code == 200, resp.text
    token = resp.json().get("token")
    assert token, "Token manquant dans la réponse"
    payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
    assert payload.get("sub") == "user123"


def test_login_wrong_credentials_returns_401():
    resp = requests.post(
        LOGIN_URL,
        headers={"Content-Type": "application/json"},
        json={"username": "user123", "password": "wrong"},
        timeout=10,
    )
    assert resp.status_code == 401
    assert resp.json().get("detail")


def test_auth_fails_when_token_missing():
    resp = requests.post(
        PREDICT_URL,
        headers={"Content-Type": "application/json"},
        json=valid_payload(),
        timeout=10,
    )
    assert resp.status_code == 401
    assert resp.json().get("detail") == "Missing authentication token"


def test_auth_fails_when_token_invalid():
    headers = {"Content-Type": "application/json", "Authorization": "Bearer invalid.token.value"}
    resp = requests.post(PREDICT_URL, json=valid_payload(), headers=headers, timeout=10)
    assert resp.status_code == 401
    assert resp.json().get("detail") == "Invalid token"


def test_auth_fails_when_token_expired():
    # Crée un token expiré (exp dans le passé)
    expired_payload = {"sub": "user123", "exp": int(time.time()) - 60}
    expired_token = jwt.encode(expired_payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {expired_token}"}
    resp = requests.post(PREDICT_URL, json=valid_payload(), headers=headers, timeout=10)
    assert resp.status_code == 401
    assert resp.json().get("detail") == "Token has expired"


def test_prediction_succeeds_with_valid_token_and_input():
    token = create_jwt_token("user123")
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {token}"}
    resp = requests.post(PREDICT_URL, json=valid_payload(), headers=headers, timeout=10)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "prediction" in body
    assert isinstance(body["prediction"], list)
    assert len(body["prediction"]) == 1
    # La valeur exacte dépend du modèle en prod; on vérifie juste que c'est un nombre
    assert isinstance(body["prediction"][0], (int, float))
    assert body.get("user") == "user123"


def test_prediction_fails_with_invalid_input():
    token = create_jwt_token("user123")
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {token}"}
    # données invalides (type incorrect et champ manquant)
    invalid = {
        "GRE_score": "not-an-int",  # texte à la place d'un entier
        # "TOEFL_score" manquant
        "University_Rating": 2,
        "SOP": 3.0,
        "LOR": 2.0,
        "CGPA": 8.12,
        "Research": 1,
    }
    resp = requests.post(PREDICT_URL, json=invalid, headers=headers, timeout=10)
    # 400 ou 422 selon la version de pydantic/starlette utilisée par le serveur
    assert resp.status_code in (400, 422)
