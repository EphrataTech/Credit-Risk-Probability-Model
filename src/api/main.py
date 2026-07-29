"""FastAPI application for real-time credit risk scoring."""

from __future__ import annotations

import os

import pandas as pd
from fastapi import FastAPI, HTTPException

from src.api.pydantic_models import ExplainResponse, FeatureContribution, PredictionResponse, TransactionFeatures
from src.config import APIConfig
from src.explain import explain_prediction
from src.predict import load_model, predict_risk

API_CFG = APIConfig()

app = FastAPI(
    title="Credit Risk Scoring API",
    description=(
        "Returns the probability that a customer is high-risk (proxy default) "
        "for Bati Bank's buy-now-pay-later product."
    ),
    version="1.1.0",
)

_model = None
_background_df: pd.DataFrame | None = None


def _get_model():
    global _model
    if _model is None:
        try:
            _model = load_model(API_CFG)
        except Exception as exc:
            raise RuntimeError(
                f"Could not load model '{API_CFG.registered_model_name}' "
                f"from '{API_CFG.mlflow_tracking_uri}': {exc}"
            ) from exc
    return _model


def _get_background() -> pd.DataFrame:
    """Load a small background sample for SHAP explanations."""
    global _background_df
    if _background_df is None:
        path = os.getenv("BACKGROUND_DATA_PATH", "data/processed/processed.csv")
        if os.path.exists(path):
            df = pd.read_csv(path)
            target_col = "is_high_risk"
            _background_df = df.drop(columns=[target_col]) if target_col in df.columns else df
        else:
            raise FileNotFoundError(
                f"Background data not found at '{path}'. "
                "Set BACKGROUND_DATA_PATH or run training first."
            )
    return _background_df


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(features: TransactionFeatures) -> PredictionResponse:
    try:
        model = _get_model()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    row = pd.DataFrame([features.model_dump()])
    try:
        result = predict_risk(row, config=API_CFG, model=model)[0]
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Prediction failed: {exc}") from exc

    return PredictionResponse(**result)


@app.post("/explain", response_model=ExplainResponse)
def explain(features: TransactionFeatures) -> ExplainResponse:
    try:
        model = _get_model()
        background = _get_background()
    except (RuntimeError, FileNotFoundError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    row = pd.DataFrame([features.model_dump()])
    try:
        explanation = explain_prediction(model, row, background)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Explanation failed: {exc}") from exc

    return ExplainResponse(
        customer_id=features.CustomerId,
        risk_probability=explanation["risk_probability"],
        base_value=explanation["base_value"],
        feature_contributions=[
            FeatureContribution(**item) for item in explanation["feature_contributions"]
        ],
    )
