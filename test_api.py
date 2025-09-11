#!/usr/bin/env python3

from fastapi.testclient import TestClient
from fast_api import app

def test_root_endpoint():
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    print("✅ Endpoint racine OK")

def test_predict_endpoint():
    client = TestClient(app)
    response = client.post("/predict?size=180&nb_rooms=3&garden=true")
    assert response.status_code == 200
    data = response.json()
    assert "y_pred" in data
    assert isinstance(data["y_pred"], (int, float))
    print("✅ Endpoint prédiction OK")

if __name__ == "__main__":
    test_root_endpoint()
    test_predict_endpoint()
    print("🎉 Tous les tests passés avec succès!") 