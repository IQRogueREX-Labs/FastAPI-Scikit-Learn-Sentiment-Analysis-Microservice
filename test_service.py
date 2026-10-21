import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_positive_prediction():
    response = client.post("/predict", json={"text": "This platform is absolutely fantastic and super fast!"})
    assert response.status_code == 200
    data = response.json()
    assert data["sentiment"] == "POSITIVE"
    assert data["confidence"] > 0.5

def test_negative_prediction():
    response = client.post("/predict", json={"text": "Terrible quality, broke immediately and wasted my money."})
    assert response.status_code == 200
    data = response.json()
    assert data["sentiment"] == "NEGATIVE"
    assert data["confidence"] > 0.5
