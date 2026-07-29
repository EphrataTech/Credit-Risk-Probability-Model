"""Model loading and batch/single inference utilities."""

from __future__ import annotations

import os
from typing import Any

import mlflow.sklearn
import pandas as pd
from sklearn.pipeline import Pipeline

from src.config import APIConfig, ModelConfig


def load_model(config: APIConfig | None = None) -> Pipeline:
    """
    Load the latest registered model from the MLflow Model Registry.

    Args:
        config: API configuration. Defaults to environment-based values.

    Returns:
        Fitted scikit-learn Pipeline ready for inference.
    """
    cfg = config or APIConfig()
    mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
    model_uri = f"models:/{cfg.registered_model_name}/latest"
    return mlflow.sklearn.load_model(model_uri)


def predict_proba(
    model: Pipeline,
    features: pd.DataFrame,
) -> pd.Series:
    """
    Return high-risk probabilities for each row in ``features``.

    Args:
        model: Fitted sklearn pipeline with ``predict_proba`` support.
        features: Raw transaction-level feature rows.

    Returns:
        Series of probabilities for the positive (high-risk) class.
    """
    probabilities = model.predict_proba(features)[:, 1]
    return pd.Series(probabilities, index=features.index, name="risk_probability")


def predict_risk(
    features: pd.DataFrame,
    config: APIConfig | None = None,
    model: Pipeline | None = None,
) -> list[dict[str, Any]]:
    """
    Score one or more transactions and return structured prediction results.

    Args:
        features: Raw transaction-level feature rows.
        config: API configuration for threshold and model loading.
        model: Optional pre-loaded model. Loaded from MLflow when omitted.

    Returns:
        List of dicts with customer_id, risk_probability, and is_high_risk.
    """
    cfg = config or APIConfig()
    estimator = model or load_model(cfg)
    probs = predict_proba(estimator, features)

    results: list[dict[str, Any]] = []
    for idx, prob in probs.items():
        customer_id = features.loc[idx, "CustomerId"] if "CustomerId" in features.columns else str(idx)
        probability = round(float(prob), 4)
        results.append({
            "customer_id": str(customer_id),
            "risk_probability": probability,
            "is_high_risk": probability >= cfg.risk_threshold,
        })
    return results


def save_model_locally(
    model: Pipeline,
    output_dir: str = "models",
    model_name: str | None = None,
) -> str:
    """
    Persist a trained pipeline to disk for dashboard/demo use without MLflow.

    Returns:
        Path to the saved model directory.
    """
    name = model_name or ModelConfig().registered_model_name
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, name)
    mlflow.sklearn.save_model(model, path)
    return path
