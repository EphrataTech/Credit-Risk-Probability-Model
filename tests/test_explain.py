"""Unit tests for SHAP explainability helpers."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.explain import explain_prediction, global_feature_importance


@pytest.fixture()
def mini_dataset():
    rng = np.random.default_rng(42)
    n = 60
    X = rng.normal(size=(n, 5))
    return pd.DataFrame(X, columns=[f"feature_{i}" for i in range(5)])


@pytest.fixture()
def fitted_pipeline(mini_dataset):
    y = pd.Series((mini_dataset["feature_0"] + mini_dataset["feature_1"] > 0).astype(int))
    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000)),
    ])
    pipe.fit(mini_dataset, y)
    return pipe


def test_global_feature_importance_returns_dataframe(fitted_pipeline, mini_dataset):
    importance = global_feature_importance(fitted_pipeline, mini_dataset)
    assert "feature" in importance.columns
    assert "mean_abs_shap" in importance.columns
    assert len(importance) > 0


def test_explain_prediction_structure(fitted_pipeline, mini_dataset):
    row = mini_dataset.iloc[[0]]
    explanation = explain_prediction(fitted_pipeline, row, mini_dataset)
    assert "risk_probability" in explanation
    assert "base_value" in explanation
    assert len(explanation["feature_contributions"]) > 0
    assert "feature" in explanation["feature_contributions"][0]
