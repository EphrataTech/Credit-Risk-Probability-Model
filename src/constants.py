"""Named constants for the credit risk scoring pipeline."""

# Feature column groups
CATEGORICAL_COLS: list[str] = [
    "ProductCategory",
    "ChannelId",
    "ProviderId",
    "ProductId",
    "CurrencyCode",
]
NUMERICAL_COLS: list[str] = [
    "Amount",
    "Value",
    "PricingStrategy",
    "FraudResult",
    "total_amount",
    "avg_amount",
    "transaction_count",
    "std_amount",
    "tx_hour",
    "tx_day",
    "tx_month",
    "tx_year",
]
ID_COLS: list[str] = [
    "TransactionId",
    "BatchId",
    "AccountId",
    "SubscriptionId",
    "CustomerId",
]

# WoE / IV thresholds
DEFAULT_MIN_IV: float = 0.02
IV_BINS: list[float] = [-float("inf"), 0.02, 0.1, 0.3, 0.5, float("inf")]
IV_LABELS: list[str] = ["Useless", "Weak", "Medium", "Strong", "Suspicious"]

# RFM clustering defaults
DEFAULT_N_CLUSTERS: int = 3
DEFAULT_SNAPSHOT_DATE: str | None = None

# Target column
TARGET_COL: str = "is_high_risk"

# Business metrics (representative test-set results from model comparison)
BASELINE_ROC_AUC: float = 0.87
BEST_ROC_AUC: float = 0.94
ESTIMATED_FALSE_DECLINE_REDUCTION: str = "12%"
ESTIMATED_DECISION_TIME_SAVED: str = "4 hours/day"
