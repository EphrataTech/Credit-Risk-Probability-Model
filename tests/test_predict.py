"""Unit tests for inference utilities."""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.config import APIConfig
from src.predict import predict_proba, predict_risk


@pytest.fixture()
def sample_row():
    return pd.DataFrame([{
        "TransactionId": "T1",
        "BatchId": "B1",
        "AccountId": "A1",
        "SubscriptionId": "S1",
        "CustomerId": "C1",
        "CurrencyCode": "UGX",
        "CountryCode": 256,
        "ProviderId": "P1",
        "ProductId": "PR1",
        "ProductCategory": "airtime",
        "ChannelId": "web",
        "Amount": 100.0,
        "Value": 100.0,
        "TransactionStartTime": "2024-01-15T08:30:00Z",
        "PricingStrategy": 2,
        "FraudResult": 0,
    }])


@pytest.fixture()
def mock_pipeline():
    model = MagicMock()
    model.predict_proba.return_value = np.array([[0.6, 0.4]])
    return model


def test_predict_proba_returns_series(sample_row, mock_pipeline):
    result = predict_proba(mock_pipeline, sample_row)
    assert len(result) == 1
    assert result.iloc[0] == pytest.approx(0.4)


def test_predict_risk_structure(sample_row, mock_pipeline):
    cfg = APIConfig(risk_threshold=0.5)
    results = predict_risk(sample_row, config=cfg, model=mock_pipeline)
    assert results[0]["customer_id"] == "C1"
    assert results[0]["risk_probability"] == 0.4
    assert results[0]["is_high_risk"] is False


def test_predict_risk_high_risk_flag(sample_row, mock_pipeline):
    mock_pipeline.predict_proba.return_value = np.array([[0.1, 0.9]])
    cfg = APIConfig(risk_threshold=0.5)
    results = predict_risk(sample_row, config=cfg, model=mock_pipeline)
    assert results[0]["is_high_risk"] is True


@patch("src.predict.mlflow.sklearn.load_model")
def test_load_model_uses_registry(mock_load):
    from src.predict import load_model

    mock_load.return_value = MagicMock()
    model = load_model(APIConfig(registered_model_name="test-model", mlflow_tracking_uri="mlruns"))
    mock_load.assert_called_once()
    assert model is mock_load.return_value
