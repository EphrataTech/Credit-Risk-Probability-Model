import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier

from src.train import evaluate, RANDOM_STATE
from src.data_processing import (
    AggregateFeatures,
    DatetimeFeatures,
    DropIDColumns,
    WoEEncoder,
    build_rfm_features,
    assign_rfm_clusters,
    build_rfm_target,
    build_preprocessing_pipeline,
)


@pytest.fixture()
def sample_df():
    return pd.DataFrame({
        "TransactionId":        ["T1", "T2", "T3", "T4"],
        "BatchId":              ["B1", "B1", "B2", "B2"],
        "AccountId":            ["A1", "A1", "A2", "A2"],
        "SubscriptionId":       ["S1", "S1", "S2", "S2"],
        "CustomerId":           ["C1", "C1", "C2", "C2"],
        "CurrencyCode":         ["UGX", "UGX", "UGX", "UGX"],
        "CountryCode":          [256, 256, 256, 256],
        "ProviderId":           ["P1", "P2", "P1", "P3"],
        "ProductId":            ["PR1", "PR2", "PR1", "PR3"],
        "ProductCategory":      ["airtime", "airtime", "data_bundles", "utility_bill"],
        "ChannelId":            ["web", "android", "web", "ios"],
        "Amount":               [100.0, 200.0, 50.0, 300.0],
        "Value":                [100.0, 200.0, 50.0, 300.0],
        "TransactionStartTime": [
            "2023-01-15T08:30:00Z", "2023-03-20T14:00:00Z",
            "2023-06-05T22:15:00Z", "2023-11-01T06:45:00Z",
        ],
        "PricingStrategy":      [2, 2, 4, 4],
        "FraudResult":          [0, 0, 0, 1],
    })


@pytest.fixture()
def large_rfm_df():
    """50 customers with varied RFM so K-Means can form 3 non-trivial clusters."""
    np.random.seed(0)
    n = 50
    base = pd.Timestamp("2024-01-01", tz="UTC")
    records = []
    for i in range(n):
        cid = f"C{i}"
        n_tx = np.random.randint(1, 20)
        for j in range(n_tx):
            records.append({
                "TransactionId": f"T{i}_{j}",
                "CustomerId": cid,
                "Amount": np.random.uniform(10, 500),
                "TransactionStartTime": (
                    base - pd.Timedelta(days=np.random.randint(1, 365))
                ).isoformat(),
            })
    return pd.DataFrame(records)


# --- AggregateFeatures ---

def test_aggregate_adds_columns(sample_df):
    out = AggregateFeatures().fit_transform(sample_df)
    for col in ["total_amount", "avg_amount", "transaction_count", "std_amount"]:
        assert col in out.columns


def test_aggregate_values(sample_df):
    out = AggregateFeatures().fit_transform(sample_df)
    c1 = out[out["CustomerId"] == "C1"].iloc[0]
    assert c1["total_amount"] == 300.0
    assert c1["avg_amount"] == 150.0
    assert c1["transaction_count"] == 2


def test_aggregate_std_single_row(sample_df):
    single = sample_df[sample_df["CustomerId"] == "C1"].iloc[:1].copy()
    out = AggregateFeatures().fit_transform(single)
    assert np.isnan(out["std_amount"].iloc[0])


# --- DatetimeFeatures ---

def test_datetime_extracts_components(sample_df):
    out = DatetimeFeatures().fit_transform(sample_df)
    for col in ["tx_hour", "tx_day", "tx_month", "tx_year"]:
        assert col in out.columns
    assert "TransactionStartTime" not in out.columns


def test_datetime_values(sample_df):
    out = DatetimeFeatures().fit_transform(sample_df)
    assert out.iloc[0]["tx_hour"] == 8
    assert out.iloc[0]["tx_day"] == 15
    assert out.iloc[0]["tx_month"] == 1
    assert out.iloc[0]["tx_year"] == 2023


# --- DropIDColumns ---

def test_drop_id_columns(sample_df):
    out = DropIDColumns().fit_transform(sample_df)
    for col in ["TransactionId", "BatchId", "AccountId", "SubscriptionId", "CustomerId"]:
        assert col not in out.columns


def test_drop_id_preserves_other_columns(sample_df):
    out = DropIDColumns().fit_transform(sample_df)
    assert "Amount" in out.columns
    assert "ProductCategory" in out.columns


# --- WoEEncoder ---

@pytest.fixture()
def woe_df():
    np.random.seed(42)
    n = 200
    return pd.DataFrame({
        "ProductCategory": np.random.choice(["airtime", "data_bundles", "utility_bill"], n),
        "ChannelId":       np.random.choice(["web", "android", "ios"], n),
        "is_high_risk":    np.random.randint(0, 2, n),
    })


def test_woe_drops_original_cats(woe_df):
    enc = WoEEncoder(cat_cols=["ProductCategory", "ChannelId"]).fit(woe_df)
    out = enc.transform(woe_df)
    assert "ProductCategory" not in out.columns
    assert "ChannelId" not in out.columns


def test_woe_iv_summary_shape(woe_df):
    enc = WoEEncoder(cat_cols=["ProductCategory", "ChannelId"]).fit(woe_df)
    iv = enc.get_iv_summary()
    assert set(iv.columns) == {"IV", "strength"}
    assert len(iv) == 2


# --- build_rfm_features ---

def test_rfm_features_columns(large_rfm_df):
    rfm = build_rfm_features(large_rfm_df, snapshot_date="2024-01-01")
    for col in ["CustomerId", "recency", "frequency", "monetary"]:
        assert col in rfm.columns


def test_rfm_features_one_row_per_customer(large_rfm_df):
    rfm = build_rfm_features(large_rfm_df, snapshot_date="2024-01-01")
    assert rfm["CustomerId"].nunique() == rfm.shape[0]


def test_rfm_recency_non_negative(large_rfm_df):
    rfm = build_rfm_features(large_rfm_df, snapshot_date="2024-01-01")
    assert (rfm["recency"] >= 0).all()


def test_rfm_frequency_positive(large_rfm_df):
    rfm = build_rfm_features(large_rfm_df, snapshot_date="2024-01-01")
    assert (rfm["frequency"] > 0).all()


# --- assign_rfm_clusters ---

def test_cluster_assigns_three_clusters(large_rfm_df):
    rfm = build_rfm_features(large_rfm_df, snapshot_date="2024-01-01")
    rfm = assign_rfm_clusters(rfm, n_clusters=3, random_state=42)
    assert rfm["cluster"].nunique() == 3


def test_cluster_is_high_risk_binary(large_rfm_df):
    rfm = build_rfm_features(large_rfm_df, snapshot_date="2024-01-01")
    rfm = assign_rfm_clusters(rfm, n_clusters=3, random_state=42)
    assert set(rfm["is_high_risk"].unique()).issubset({0, 1})


def test_cluster_reproducible(large_rfm_df):
    rfm = build_rfm_features(large_rfm_df, snapshot_date="2024-01-01")
    r1 = assign_rfm_clusters(rfm, random_state=42)["is_high_risk"].values
    r2 = assign_rfm_clusters(rfm, random_state=42)["is_high_risk"].values
    np.testing.assert_array_equal(r1, r2)


def test_cluster_not_all_same_label(large_rfm_df):
    rfm = build_rfm_features(large_rfm_df, snapshot_date="2024-01-01")
    rfm = assign_rfm_clusters(rfm, n_clusters=3, random_state=42)
    assert rfm["is_high_risk"].sum() > 0
    assert rfm["is_high_risk"].sum() < len(rfm)


# --- build_rfm_target ---

def test_rfm_target_columns(large_rfm_df):
    target = build_rfm_target(large_rfm_df, snapshot_date="2024-01-01")
    assert list(target.columns) == ["CustomerId", "is_high_risk"]


def test_rfm_target_one_row_per_customer(large_rfm_df):
    target = build_rfm_target(large_rfm_df, snapshot_date="2024-01-01")
    assert target["CustomerId"].nunique() == target.shape[0]


# --- Full preprocessing pipeline smoke test ---

def test_pipeline_runs_without_error(sample_df):
    pipe = build_preprocessing_pipeline()
    out = pipe.fit_transform(sample_df)
    assert isinstance(out, pd.DataFrame)
    assert "tx_hour" in out.columns
    assert "total_amount" in out.columns
    assert "CustomerId" not in out.columns


# ---------------------------------------------------------------------------
# Tests for src/train.py helpers
# ---------------------------------------------------------------------------


@pytest.fixture()
def classification_data():
    """Small binary classification dataset for train.py helper tests."""
    np.random.seed(RANDOM_STATE)
    X = pd.DataFrame({"a": np.random.randn(100), "b": np.random.randn(100)})
    y = pd.Series(np.random.randint(0, 2, 100))
    return X, y


def test_evaluate_returns_all_metrics(classification_data):
    X, y = classification_data
    clf = DummyClassifier(strategy="most_frequent").fit(X, y)
    metrics = evaluate(clf, X, y)
    assert set(metrics.keys()) == {"accuracy", "precision", "recall", "f1", "roc_auc"}


def test_evaluate_metrics_in_range(classification_data):
    X, y = classification_data
    clf = DummyClassifier(strategy="stratified", random_state=RANDOM_STATE).fit(X, y)
    metrics = evaluate(clf, X, y)
    for name, val in metrics.items():
        assert 0.0 <= val <= 1.0, f"{name}={val} out of [0, 1]"


def test_evaluate_perfect_classifier(classification_data):
    X, y = classification_data
    clf = KNeighborsClassifier(n_neighbors=1).fit(X, y)
    metrics = evaluate(clf, X, y)
    assert metrics["accuracy"] == 1.0
    assert metrics["roc_auc"] == 1.0
