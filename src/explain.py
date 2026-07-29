"""SHAP-based model explainability for credit risk predictions."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline

from src.config import ExplainConfig


def _transform_features(model: Pipeline, features: pd.DataFrame) -> np.ndarray:
    """Run all pipeline steps except the final classifier."""
    transformed = model[:-1].transform(features)
    return np.asarray(transformed)


def _feature_names(model: Pipeline, n_features: int) -> list[str]:
    """Best-effort feature names after preprocessing."""
    try:
        col_step = model.named_steps.get("col")
        if col_step is not None and hasattr(col_step, "get_feature_names_out"):
            return list(col_step.get_feature_names_out())
    except Exception:
        pass
    return [f"feature_{i}" for i in range(n_features)]


def build_explainer(
    model: Pipeline,
    background: pd.DataFrame,
    config: ExplainConfig | None = None,
) -> shap.Explainer:
    """
    Create a SHAP explainer for the fitted pipeline's classifier.

    Uses TreeExplainer for tree-based models and LinearExplainer otherwise.
    """
    cfg = config or ExplainConfig()
    sample = background.sample(
        min(len(background), cfg.max_background_samples),
        random_state=42,
    )
    X_bg = _transform_features(model, sample)
    classifier = model.steps[-1][1]

    if hasattr(classifier, "feature_importances_"):
        return shap.TreeExplainer(classifier, data=X_bg)
    return shap.LinearExplainer(classifier, X_bg)


def explain_prediction(
    model: Pipeline,
    features: pd.DataFrame,
    background: pd.DataFrame,
    config: ExplainConfig | None = None,
) -> dict[str, Any]:
    """
    Explain a single prediction with per-feature SHAP contributions.

    Returns:
        Dict with base_value, risk_probability, and ranked feature impacts.
    """
    cfg = config or ExplainConfig()
    explainer = build_explainer(model, background, cfg)
    X = _transform_features(model, features.iloc[[0]])
    shap_values = explainer.shap_values(X)

    if isinstance(shap_values, list):
        values = shap_values[1][0]
        base = explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value
    else:
        values = shap_values[0]
        base = explainer.expected_value

    names = _feature_names(model, len(values))
    contributions = sorted(
        zip(names, values.tolist()),
        key=lambda item: abs(item[1]),
        reverse=True,
    )[: cfg.max_display_features]

    prob = float(model.predict_proba(features.iloc[[0]])[0, 1])
    return {
        "base_value": float(base),
        "risk_probability": round(prob, 4),
        "feature_contributions": [
            {"feature": name, "shap_value": round(val, 4)}
            for name, val in contributions
        ],
    }


def global_feature_importance(
    model: Pipeline,
    background: pd.DataFrame,
    config: ExplainConfig | None = None,
) -> pd.DataFrame:
    """
    Compute mean absolute SHAP values across a sample for global importance.

    Returns:
        DataFrame with columns [feature, mean_abs_shap], sorted descending.
    """
    cfg = config or ExplainConfig()
    sample = background.sample(
        min(len(background), cfg.max_background_samples),
        random_state=42,
    )
    explainer = build_explainer(model, sample, cfg)
    X = _transform_features(model, sample)
    shap_values = explainer.shap_values(X)

    if isinstance(shap_values, list):
        values = shap_values[1]
    else:
        values = shap_values

    names = _feature_names(model, values.shape[1])
    importance = pd.DataFrame({
        "feature": names,
        "mean_abs_shap": np.abs(values).mean(axis=0),
    })
    return importance.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
