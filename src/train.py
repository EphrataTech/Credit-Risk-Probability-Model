import argparse
import os

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline

from src.data_processing import (
    build_column_transformer,
    build_preprocessing_pipeline,
    CATEGORICAL_COLS,
    NUMERICAL_COLS,
    build_column_transformer,
    build_preprocessing_pipeline,
    build_rfm_target,
)

EXPERIMENT_NAME = "credit-risk-model"
REGISTERED_MODEL_NAME = "credit-risk-best-model"
TEST_SIZE = 0.2
RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_data(processed_path: str, raw_path: str = None) -> tuple[pd.DataFrame, pd.Series]:
    """
    Loads the processed dataset if it exists, otherwise builds it from raw.
    Returns (X, y) where y is the is_high_risk column.
    """
    if os.path.exists(processed_path):
        df = pd.read_csv(processed_path)
    else:
        assert raw_path and os.path.exists(raw_path), (
            f"Neither {processed_path} nor {raw_path} found."
        )
        from src.data_processing import build_processed_dataset
        os.makedirs(os.path.dirname(processed_path), exist_ok=True)
        df = build_processed_dataset(raw_path, processed_path)

    y = df["is_high_risk"]
    X = df.drop(columns=["is_high_risk"])
    return X, y


# ---------------------------------------------------------------------------
# Metric helper
# ---------------------------------------------------------------------------
def evaluate(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred
    return {
        "accuracy":  accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall":    recall_score(y_test, y_pred, zero_division=0),
        "f1":        f1_score(y_test, y_pred, zero_division=0),
        "roc_auc":   roc_auc_score(y_test, y_prob),
    }


# ---------------------------------------------------------------------------
# Model definitions with hyperparameter search spaces
# ---------------------------------------------------------------------------
def model_configs(X_train, y_train) -> list[dict]:
    """Returns list of {name, estimator, param_dist} dicts."""

    lr_pipe = Pipeline([
        ("pre",  build_preprocessing_pipeline()),
        ("col",  build_column_transformer()),
        ("clf",  LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
    ])

    rf_pipe = Pipeline([
        ("pre",  build_preprocessing_pipeline()),
        ("col",  build_column_transformer()),
        ("clf",  RandomForestClassifier(random_state=RANDOM_STATE)),
    ])

    gb_pipe = Pipeline([
        ("pre",  build_preprocessing_pipeline()),
        ("col",  build_column_transformer()),
        ("clf",  GradientBoostingClassifier(random_state=RANDOM_STATE)),
    ])

    return [
        {
            "name": "LogisticRegression",
            "pipe": lr_pipe,
            "param_dist": {
                "clf__C": [0.01, 0.1, 1.0, 10.0],
                "clf__solver": ["lbfgs", "liblinear"],
            },
        },
        {
            "name": "RandomForest",
            "pipe": rf_pipe,
            "param_dist": {
                "clf__n_estimators": [100, 200],
                "clf__max_depth": [None, 5, 10],
                "clf__min_samples_split": [2, 5],
            },
        },
        {
            "name": "GradientBoosting",
            "pipe": gb_pipe,
            "param_dist": {
                "clf__n_estimators": [100, 200],
                "clf__learning_rate": [0.05, 0.1, 0.2],
                "clf__max_depth": [3, 5],
            },
        },
    ]


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train(
    processed_path: str = "data/processed/processed.csv",
    raw_path: str = "data/raw/data.csv",
    n_iter: int = 5,
    cv: int = 3,
) -> str:
    """
    Trains all model configs with RandomizedSearchCV, logs every run to MLflow,
    registers the best model by ROC-AUC in the Model Registry.

    Returns the name of the best model.
    """
    X, y = load_data(processed_path, raw_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    mlflow.set_experiment(EXPERIMENT_NAME)

    best_auc = -1.0
    best_run_id = None
    best_model_name = None

    for cfg in model_configs(X_train, y_train):
        with mlflow.start_run(run_name=cfg["name"]):
            search = RandomizedSearchCV(
                cfg["pipe"],
                param_distributions=cfg["param_dist"],
                n_iter=n_iter,
                cv=cv,
                scoring="roc_auc",
                random_state=RANDOM_STATE,
                n_jobs=-1,
                refit=True,
            )
            search.fit(X_train, y_train)
            best_estimator = search.best_estimator_

            metrics = evaluate(best_estimator, X_test, y_test)
            params = {k.replace("clf__", ""): v for k, v in search.best_params_.items()}

            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(best_estimator, artifact_path="model")

            print(f"[{cfg['name']}] best_params={params}  metrics={metrics}")

            if metrics["roc_auc"] > best_auc:
                best_auc = metrics["roc_auc"]
                best_run_id = mlflow.active_run().info.run_id
                best_model_name = cfg["name"]

    # Register the best model
    model_uri = f"runs:/{best_run_id}/model"
    mlflow.register_model(model_uri=model_uri, name=REGISTERED_MODEL_NAME)
    print(f"\nBest model: {best_model_name}  ROC-AUC={best_auc:.4f}")
    print(f"Registered as '{REGISTERED_MODEL_NAME}'  run_id={best_run_id}")
    return best_model_name


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed", default="data/processed/processed.csv")
    parser.add_argument("--raw",       default="data/raw/data.csv")
    parser.add_argument("--n-iter",    type=int, default=5)
    parser.add_argument("--cv",        type=int, default=3)
    args = parser.parse_args()

    train(
        processed_path=args.processed,
        raw_path=args.raw,
        n_iter=args.n_iter,
        cv=args.cv,
    )
