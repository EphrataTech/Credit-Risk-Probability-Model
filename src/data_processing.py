import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


# ---------------------------------------------------------------------------
# Step 1 – Aggregate features (customer-level RFM + stats)
# ---------------------------------------------------------------------------
class AggregateFeatures(BaseEstimator, TransformerMixin):
    """Computes per-customer aggregate features and merges back onto transactions."""

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        agg = (
            X.groupby("CustomerId")["Amount"]
            .agg(
                total_amount="sum",
                avg_amount="mean",
                transaction_count="count",
                std_amount="std",
            )
            .reset_index()
        )
        return X.merge(agg, on="CustomerId", how="left")


# ---------------------------------------------------------------------------
# Step 2 – Datetime feature extraction
# ---------------------------------------------------------------------------
class DatetimeFeatures(BaseEstimator, TransformerMixin):
    """Extracts hour, day, month, year from TransactionStartTime."""

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        dt = pd.to_datetime(X["TransactionStartTime"], utc=True)
        X["tx_hour"] = dt.dt.hour
        X["tx_day"] = dt.dt.day
        X["tx_month"] = dt.dt.month
        X["tx_year"] = dt.dt.year
        return X.drop(columns=["TransactionStartTime"])


# ---------------------------------------------------------------------------
# Step 3 – Drop ID columns not useful for modeling
# ---------------------------------------------------------------------------
class DropIDColumns(BaseEstimator, TransformerMixin):
    ID_COLS = ["TransactionId", "BatchId", "AccountId", "SubscriptionId", "CustomerId"]

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.drop(columns=[c for c in self.ID_COLS if c in X.columns])


# ---------------------------------------------------------------------------
# Step 4 – WoE / IV computation (fitted on training data)
# ---------------------------------------------------------------------------
class WoEEncoder(BaseEstimator, TransformerMixin):
    """
    Weight-of-Evidence encoding for categorical columns.
    IV (Information Value) is stored in self.iv_ after fitting.
    Columns with IV < min_iv are dropped as uninformative.
    """

    def __init__(self, cat_cols: list[str], target_col: str = "target", min_iv: float = 0.02):
        self.cat_cols = cat_cols
        self.target_col = target_col
        self.min_iv = min_iv

    def fit(self, X: pd.DataFrame, y=None):
        self.woe_maps_: dict[str, dict] = {}
        self.iv_: dict[str, float] = {}
        self.useful_cols_: list[str] = []

        target = X[self.target_col] if self.target_col in X.columns else y
        assert target is not None, "target_col not found in X and y is None"

        total_events = target.sum()
        total_non_events = (1 - target).sum()

        for col in self.cat_cols:
            if col not in X.columns:
                continue
            stats = (
                pd.DataFrame({"cat": X[col], "target": target})
                .groupby("cat")["target"]
                .agg(events="sum", total="count")
            )
            stats["non_events"] = stats["total"] - stats["events"]
            stats["dist_events"] = (stats["events"] + 0.5) / (total_events + 0.5)
            stats["dist_non_events"] = (stats["non_events"] + 0.5) / (total_non_events + 0.5)
            stats["woe"] = np.log(stats["dist_events"] / stats["dist_non_events"])
            stats["iv"] = (stats["dist_events"] - stats["dist_non_events"]) * stats["woe"]
            iv_total = stats["iv"].sum()

            self.woe_maps_[col] = stats["woe"].to_dict()
            self.iv_[col] = iv_total
            if iv_total >= self.min_iv:
                self.useful_cols_.append(col)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col in self.useful_cols_:
            X[f"{col}_woe"] = X[col].map(self.woe_maps_[col]).fillna(0.0)
        # Drop original categorical columns that were encoded
        return X.drop(columns=[c for c in self.cat_cols if c in X.columns])

    def get_iv_summary(self) -> pd.DataFrame:
        return (
            pd.DataFrame.from_dict(self.iv_, orient="index", columns=["IV"])
            .sort_values("IV", ascending=False)
            .assign(strength=lambda d: pd.cut(
                d["IV"],
                bins=[-np.inf, 0.02, 0.1, 0.3, 0.5, np.inf],
                labels=["Useless", "Weak", "Medium", "Strong", "Suspicious"],
            ))
        )


# ---------------------------------------------------------------------------
# Step 5 – RFM proxy target label (used before pipeline, not inside it)
# ---------------------------------------------------------------------------
def build_rfm_target(df: pd.DataFrame, snapshot_date: str = None) -> pd.DataFrame:
    """
    Computes RFM scores per customer and assigns a binary proxy default label.
    Low RFM score (bottom tertile) => high risk => target = 1.

    Returns a DataFrame with [CustomerId, target].
    """
    snapshot = pd.Timestamp(snapshot_date, tz="UTC") if snapshot_date else (
        pd.to_datetime(df["TransactionStartTime"], utc=True).max()
    )

    df = df.copy()
    df["_dt"] = pd.to_datetime(df["TransactionStartTime"], utc=True)

    rfm = df.groupby("CustomerId").agg(
        recency=("_dt", lambda x: (snapshot - x.max()).days),
        frequency=("TransactionId", "count"),
        monetary=("Amount", "sum"),
    ).reset_index()

    # Score each dimension 1–3 (1 = best for frequency/monetary, worst recency)
    rfm["r_score"] = pd.qcut(rfm["recency"], q=3, labels=[3, 2, 1]).astype(int)
    rfm["f_score"] = pd.qcut(rfm["frequency"].rank(method="first"), q=3, labels=[1, 2, 3]).astype(int)
    rfm["m_score"] = pd.qcut(rfm["monetary"].rank(method="first"), q=3, labels=[1, 2, 3]).astype(int)
    rfm["rfm_score"] = rfm["r_score"] + rfm["f_score"] + rfm["m_score"]

    # Bottom tertile of RFM score => high risk (proxy default = 1)
    threshold = rfm["rfm_score"].quantile(1 / 3)
    rfm["target"] = (rfm["rfm_score"] <= threshold).astype(int)

    return rfm[["CustomerId", "rfm_score", "target"]]


# ---------------------------------------------------------------------------
# Pipeline factory
# ---------------------------------------------------------------------------
CATEGORICAL_COLS = ["ProductCategory", "ChannelId", "ProviderId", "ProductId", "CurrencyCode"]
NUMERICAL_COLS = [
    "Amount", "Value", "PricingStrategy", "FraudResult",
    "total_amount", "avg_amount", "transaction_count", "std_amount",
    "tx_hour", "tx_day", "tx_month", "tx_year",
]


def build_preprocessing_pipeline() -> Pipeline:
    """
    Returns a sklearn Pipeline that:
      1. Aggregates customer-level features
      2. Extracts datetime components
      3. Drops ID columns
    The returned pipeline outputs a DataFrame ready for WoE encoding + ColumnTransformer.
    """
    return Pipeline([
        ("aggregate", AggregateFeatures()),
        ("datetime",  DatetimeFeatures()),
        ("drop_ids",  DropIDColumns()),
    ])


def build_column_transformer() -> ColumnTransformer:
    """
    Numeric: median imputation → StandardScaler
    Categorical: constant imputation → OneHotEncoder
    """
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("ohe",     OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer(
        transformers=[
            ("num", numeric,     NUMERICAL_COLS),
            ("cat", categorical, CATEGORICAL_COLS),
        ],
        remainder="drop",
    )
