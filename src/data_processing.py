import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.constants import (
    CATEGORICAL_COLS,
    DEFAULT_MIN_IV,
    DEFAULT_N_CLUSTERS,
    ID_COLS,
    IV_BINS,
    IV_LABELS,
    NUMERICAL_COLS,
    TARGET_COL,
)


# ---------------------------------------------------------------------------
# Step 1 – Aggregate features (customer-level stats)
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
    """Removes identifier columns that should not be used as model features."""

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.drop(columns=[c for c in ID_COLS if c in X.columns])


# ---------------------------------------------------------------------------
# Step 4 – WoE / IV computation (fitted on training data)
# ---------------------------------------------------------------------------
class WoEEncoder(BaseEstimator, TransformerMixin):
    """
    Weight-of-Evidence encoding for categorical columns.
    IV (Information Value) is stored in self.iv_ after fitting.
    Columns with IV < min_iv are dropped as uninformative.
    """

    def __init__(
        self,
        cat_cols: list[str],
        target_col: str = TARGET_COL,
        min_iv: float = DEFAULT_MIN_IV,
    ):
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
        return X.drop(columns=[c for c in self.cat_cols if c in X.columns])

    def get_iv_summary(self) -> pd.DataFrame:
        return (
            pd.DataFrame.from_dict(self.iv_, orient="index", columns=["IV"])
            .sort_values("IV", ascending=False)
            .assign(strength=lambda d: pd.cut(
                d["IV"],
                bins=IV_BINS,
                labels=IV_LABELS,
            ))
        )


# ---------------------------------------------------------------------------
# Step 5 – RFM computation + K-Means clustering → is_high_risk label
# ---------------------------------------------------------------------------
def build_rfm_features(df: pd.DataFrame, snapshot_date: str = None) -> pd.DataFrame:
    """
    Computes Recency, Frequency, Monetary values per customer.

    Args:
        df: raw transaction DataFrame
        snapshot_date: ISO date string used as the 'today' reference for recency.
                       Defaults to the latest transaction date in the dataset.

    Returns:
        DataFrame with columns [CustomerId, recency, frequency, monetary]
    """
    snapshot = (
        pd.Timestamp(snapshot_date, tz="UTC")
        if snapshot_date
        else pd.to_datetime(df["TransactionStartTime"], utc=True).max()
    )
    df = df.copy()
    df["_dt"] = pd.to_datetime(df["TransactionStartTime"], utc=True)

    rfm = (
        df.groupby("CustomerId")
        .agg(
            recency=("_dt", lambda x: (snapshot - x.max()).days),
            frequency=("TransactionId", "count"),
            monetary=("Amount", "sum"),
        )
        .reset_index()
    )
    return rfm


def assign_rfm_clusters(
    rfm: pd.DataFrame,
    n_clusters: int = DEFAULT_N_CLUSTERS,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Scales RFM features and runs K-Means to segment customers.
    Identifies the high-risk cluster (lowest frequency + lowest monetary)
    and assigns is_high_risk = 1 to those customers.

    Args:
        rfm: DataFrame with [CustomerId, recency, frequency, monetary]
        n_clusters: number of K-Means clusters (default 3)
        random_state: for reproducibility

    Returns:
        rfm DataFrame with additional columns: cluster, is_high_risk
    """
    rfm = rfm.copy()
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm[["recency", "frequency", "monetary"]])

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    rfm["cluster"] = kmeans.fit_predict(rfm_scaled)

    # Identify high-risk cluster: highest recency (most days since last tx),
    # lowest frequency, lowest monetary — sum of scaled centroids determines rank
    centers = pd.DataFrame(
        kmeans.cluster_centers_,
        columns=["recency", "frequency", "monetary"],
    )
    # High risk = high recency score + low frequency + low monetary
    # Score: recency - frequency - monetary (higher = more disengaged)
    centers["risk_score"] = centers["recency"] - centers["frequency"] - centers["monetary"]
    high_risk_cluster = int(centers["risk_score"].idxmax())

    rfm["is_high_risk"] = (rfm["cluster"] == high_risk_cluster).astype(int)
    return rfm


def build_rfm_target(df: pd.DataFrame, snapshot_date: str = None) -> pd.DataFrame:
    """
    Convenience wrapper: computes RFM and returns [CustomerId, is_high_risk].
    """
    rfm = build_rfm_features(df, snapshot_date)
    rfm = assign_rfm_clusters(rfm)
    return rfm[["CustomerId", "is_high_risk"]]


# ---------------------------------------------------------------------------
# Step 6 – Full processed dataset builder (writes to data/processed/)
# ---------------------------------------------------------------------------
def build_processed_dataset(
    raw_path: str,
    output_path: str,
    snapshot_date: str = None,
) -> pd.DataFrame:
    """
    Reads raw transactions, engineers all features, attaches the is_high_risk
    target, and saves the result as a CSV.

    Args:
        raw_path:     path to raw transactions CSV
        output_path:  path to write the processed CSV
        snapshot_date: optional ISO date for recency calculation

    Returns:
        Processed DataFrame
    """
    df = pd.read_csv(raw_path)

    # Build and merge RFM target
    target = build_rfm_target(df, snapshot_date)
    df = df.merge(target, on="CustomerId", how="left")

    # Run preprocessing pipeline (aggregates, datetime, drop IDs)
    pipe = build_preprocessing_pipeline()
    processed = pipe.fit_transform(df)

    # Carry is_high_risk through (it was on df before drop_ids removed CustomerId)
    # Re-attach from the merged df using index alignment
    processed["is_high_risk"] = df["is_high_risk"].values

    processed.to_csv(output_path, index=False)
    print(f"Saved processed dataset to {output_path}  shape={processed.shape}")
    return processed


from src.constants import CATEGORICAL_COLS, NUMERICAL_COLS  # re-export for backward compatibility


# ---------------------------------------------------------------------------
# Pipeline factories
# ---------------------------------------------------------------------------
def build_preprocessing_pipeline() -> Pipeline:
    return Pipeline([
        ("aggregate", AggregateFeatures()),
        ("datetime",  DatetimeFeatures()),
        ("drop_ids",  DropIDColumns()),
    ])


def build_column_transformer() -> ColumnTransformer:
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


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    raw = sys.argv[1] if len(sys.argv) > 1 else "data/raw/data.csv"
    out = sys.argv[2] if len(sys.argv) > 2 else "data/processed/processed.csv"
    build_processed_dataset(raw, out)
