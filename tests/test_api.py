import sys
import os

# Add the root directory to the python path so 'api.main' can be resolved
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

from api.main import app
from fastapi.testclient import TestClient

client = TestClient(app)


def test_fraud_prediction_endpoint():
    """
    Day 21: Testing
    Validates that the /predict/fraud endpoint correctly handles valid requests 
    and returns the expected response schema.
    """
    payload = {
        "transaction_id": "test-tx-12345",
        "amount": 250.75,
        "features": [0.1] * 28  # Dummy PCA features
    }

    response = client.post("/predict/fraud", json=payload)

    assert response.status_code == 200, f"Expected 200 OK, got {response.status_code}"

    data = response.json()
    assert data["transaction_id"] == "test-tx-12345"
    assert "is_fraud" in data
    assert "fraud_probability" in data
    assert "latency_ms" in data
    assert type(data["latency_ms"]) == float


def test_energy_forecast_endpoint():
    """
    Validates the energy forecast endpoint.
    """
    response = client.get("/forecast/energy")
    assert response.status_code == 200

    data = response.json()
    assert "forecasts" in data
    assert "retrieved_at" in data
