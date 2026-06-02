import os

import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException

from src.api.pydantic_models import PredictionResponse, TransactionFeatures

REGISTERED_MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "credit-risk-best-model")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "mlruns")
RISK_THRESHOLD = float(os.getenv("RISK_THRESHOLD", "0.5"))

app = FastAPI(
    title="Credit Risk Scoring API",
    description="Returns the probability that a customer is high-risk (proxy default).",
    version="1.0.0",
)

_model = None


def _load_model():
    global _model
    if _model is None:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        model_uri = f"models:/{REGISTERED_MODEL_NAME}/latest"
        try:
            _model = mlflow.sklearn.load_model(model_uri)
        except Exception as exc:
            raise RuntimeError(
                f"Could not load model '{REGISTERED_MODEL_NAME}' "
                f"from '{MLFLOW_TRACKING_URI}': {exc}"
            ) from exc
    return _model


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(features: TransactionFeatures):
    try:
        model = _load_model()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    row = pd.DataFrame([features.model_dump()])

    try:
        prob = float(model.predict_proba(row)[0, 1])
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Prediction failed: {exc}")

    return PredictionResponse(
        customer_id=features.CustomerId,
        risk_probability=round(prob, 4),
        is_high_risk=prob >= RISK_THRESHOLD,
    )
