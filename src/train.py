import argparse

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

from src.config import ModelConfig, TrainingConfig
from src.constants import TARGET_COL
from src.data_processing import (
    build_column_transformer,
    build_preprocessing_pipeline,
    build_processed_dataset,
    build_rfm_target,
)

DEFAULT_MODEL_CONFIG = ModelConfig()


def load_data(
    processed_path: str,
    raw_path: str | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load the processed dataset if it exists, otherwise build it from raw data.

    Returns:
        Feature matrix X and binary target y (is_high_risk).
    """
    import os

    if os.path.exists(processed_path):
        df = pd.read_csv(processed_path)
    else:
        assert raw_path and os.path.exists(raw_path), (
            f"Neither {processed_path} nor {raw_path} found."
        )
        os.makedirs(os.path.dirname(processed_path), exist_ok=True)
        df = build_processed_dataset(raw_path, processed_path)

    y = df[TARGET_COL]
    X = df.drop(columns=[TARGET_COL])
    return X, y


def evaluate(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    """Compute classification metrics on a held-out test set."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob),
    }


def model_configs(random_state: int) -> list[dict]:
    """Return model pipelines and hyperparameter search spaces."""
    lr_pipe = Pipeline([
        ("pre", build_preprocessing_pipeline()),
        ("col", build_column_transformer()),
        ("clf", LogisticRegression(max_iter=1000, random_state=random_state)),
    ])
    rf_pipe = Pipeline([
        ("pre", build_preprocessing_pipeline()),
        ("col", build_column_transformer()),
        ("clf", RandomForestClassifier(random_state=random_state)),
    ])
    gb_pipe = Pipeline([
        ("pre", build_preprocessing_pipeline()),
        ("col", build_column_transformer()),
        ("clf", GradientBoostingClassifier(random_state=random_state)),
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


def train(config: TrainingConfig | None = None) -> str:
    """
    Train all model configs, log runs to MLflow, and register the best model.

    Returns:
        Name of the best-performing model family.
    """
    cfg = config or TrainingConfig()
    model_cfg = cfg.model

    X, y = load_data(cfg.processed_path, cfg.raw_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=model_cfg.test_size,
        random_state=model_cfg.random_state,
        stratify=y,
    )

    mlflow.set_experiment(model_cfg.experiment_name)

    best_auc = -1.0
    best_run_id = None
    best_model_name = None

    for model_cfg_item in model_configs(model_cfg.random_state):
        with mlflow.start_run(run_name=model_cfg_item["name"]):
            search = RandomizedSearchCV(
                model_cfg_item["pipe"],
                param_distributions=model_cfg_item["param_dist"],
                n_iter=cfg.n_iter,
                cv=cfg.cv,
                scoring="roc_auc",
                random_state=model_cfg.random_state,
                n_jobs=-1,
                refit=True,
            )
            search.fit(X_train, y_train)
            best_estimator = search.best_estimator_
            metrics = evaluate(best_estimator, X_test, y_test)
            params = {
                k.replace("clf__", ""): v for k, v in search.best_params_.items()
            }

            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(best_estimator, artifact_path="model")

            print(f"[{model_cfg_item['name']}] best_params={params}  metrics={metrics}")

            if metrics["roc_auc"] > best_auc:
                best_auc = metrics["roc_auc"]
                best_run_id = mlflow.active_run().info.run_id
                best_model_name = model_cfg_item["name"]

    model_uri = f"runs:/{best_run_id}/model"
    mlflow.register_model(model_uri=model_uri, name=model_cfg.registered_model_name)
    print(f"\nBest model: {best_model_name}  ROC-AUC={best_auc:.4f}")
    print(f"Registered as '{model_cfg.registered_model_name}'  run_id={best_run_id}")
    return best_model_name


# Backward-compatible module-level constants for tests
RANDOM_STATE = DEFAULT_MODEL_CONFIG.random_state
EXPERIMENT_NAME = DEFAULT_MODEL_CONFIG.experiment_name
REGISTERED_MODEL_NAME = DEFAULT_MODEL_CONFIG.registered_model_name
TEST_SIZE = DEFAULT_MODEL_CONFIG.test_size


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train credit risk models with MLflow tracking.")
    parser.add_argument("--processed", default="data/processed/processed.csv")
    parser.add_argument("--raw", default="data/raw/data.csv")
    parser.add_argument("--n-iter", type=int, default=5)
    parser.add_argument("--cv", type=int, default=3)
    args = parser.parse_args()

    train(TrainingConfig(
        processed_path=args.processed,
        raw_path=args.raw,
        n_iter=args.n_iter,
        cv=args.cv,
    ))
