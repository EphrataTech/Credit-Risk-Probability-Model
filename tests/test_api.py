"""Integration tests for the FastAPI credit risk scoring service."""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline

from src.api.main import app

SAMPLE_PAYLOAD = {
    "TransactionId": "T1000",
    "BatchId": "B100",
    "AccountId": "A100",
    "SubscriptionId": "S100",
    "CustomerId": "C100",
    "CurrencyCode": "UGX",
    "CountryCode": 256,
    "ProviderId": "P1",
    "ProductId": "PR1",
    "ProductCategory": "airtime",
    "ChannelId": "web",
    "Amount": 150.0,
    "Value": 150.0,
    "TransactionStartTime": "2024-01-15T08:30:00Z",
    "PricingStrategy": 2,
    "FraudResult": 0,
}


@pytest.fixture()
def mock_model():
    clf = DummyClassifier(strategy="constant", constant=0)
    clf.fit(np.zeros((2, 1)), [0, 1])
    pipe = Pipeline([("clf", clf)])

    def predict_proba(X):
        n = len(X) if hasattr(X, "__len__") else 1
        return np.column_stack([np.full(n, 0.7), np.full(n, 0.3)])

    pipe.predict_proba = predict_proba
    return pipe


@pytest.fixture()
def client(mock_model):
    background = pd.DataFrame([SAMPLE_PAYLOAD])
    with patch("src.api.main._get_model", return_value=mock_model), \
         patch("src.api.main._get_background", return_value=background), \
         patch("src.api.main.explain_prediction", return_value={
             "base_value": 0.2,
             "risk_probability": 0.3,
             "feature_contributions": [{"feature": "Amount", "shap_value": 0.05}],
         }):
        yield TestClient(app)


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_returns_valid_response(client):
    response = client.post("/predict", json=SAMPLE_PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    assert body["customer_id"] == "C100"
    assert 0.0 <= body["risk_probability"] <= 1.0
    assert isinstance(body["is_high_risk"], bool)


def test_predict_rejects_invalid_payload(client):
    bad_payload = {**SAMPLE_PAYLOAD, "Amount": "not-a-number"}
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422


def test_explain_returns_contributions(client):
    response = client.post("/explain", json=SAMPLE_PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    assert body["customer_id"] == "C100"
    assert "feature_contributions" in body
    assert len(body["feature_contributions"]) >= 1


def test_predict_service_unavailable():
    with patch("src.api.main._get_model", side_effect=RuntimeError("no model")):
        client = TestClient(app)
        response = client.post("/predict", json=SAMPLE_PAYLOAD)
        assert response.status_code == 503
