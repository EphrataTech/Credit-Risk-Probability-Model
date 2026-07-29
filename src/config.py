"""Configuration dataclasses for training, inference, and API serving."""

from dataclasses import dataclass, field
import os


@dataclass(frozen=True)
class ModelConfig:
    """Core model and experiment settings."""

    experiment_name: str = "credit-risk-model"
    registered_model_name: str = "credit-risk-best-model"
    test_size: float = 0.2
    random_state: int = 42
    risk_threshold: float = 0.5


@dataclass(frozen=True)
class TrainingConfig:
    """Training pipeline configuration."""

    processed_path: str = "data/processed/processed.csv"
    raw_path: str = "data/raw/data.csv"
    n_iter: int = 5
    cv: int = 3
    model: ModelConfig = field(default_factory=ModelConfig)


@dataclass(frozen=True)
class APIConfig:
    """FastAPI serving configuration."""

    registered_model_name: str = field(
        default_factory=lambda: os.getenv("MLFLOW_MODEL_NAME", "credit-risk-best-model")
    )
    mlflow_tracking_uri: str = field(
        default_factory=lambda: os.getenv("MLFLOW_TRACKING_URI", "mlruns")
    )
    risk_threshold: float = field(
        default_factory=lambda: float(os.getenv("RISK_THRESHOLD", "0.5"))
    )


@dataclass(frozen=True)
class ExplainConfig:
    """SHAP explainability settings."""

    max_background_samples: int = 100
    max_display_features: int = 15
