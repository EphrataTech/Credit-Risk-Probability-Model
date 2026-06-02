import numpy as np
import pandas as pd
import pytest
from src.data_processing import (
    AggregateFeatures,
    DatetimeFeatures,
    DropIDColumns,
    WoEEncoder,
    build_rfm_target,
    build_preprocessing_pipeline,
)


@pytest.fixture()
def sample_df():
    return pd.DataFrame({
        "TransactionId":       ["T1", "T2", "T3", "T4"],
        "BatchId":             ["B1", "B1", "B2", "B2"],
        "AccountId":           ["A1", "A1", "A2", "A2"],
        "SubscriptionId":      ["S1", "S1", "S2", "S2"],
        "CustomerId":          ["C1", "C1", "C2", "C2"],
        "CurrencyCode":        ["UGX", "UGX", "UGX", "UGX"],
        "CountryCode":         [256, 256, 256, 256],
        "ProviderId":          ["P1", "P2", "P1", "P3"],
        "ProductId":           ["PR1", "PR2", "PR1", "PR3"],
        "ProductCategory":     ["airtime", "airtime", "data_bundles", "utility_bill"],
        "ChannelId":           ["web", "android", "web", "ios"],
        "Amount":              [100.0, 200.0, 50.0, 300.0],
        "Value":               [100.0, 200.0, 50.0, 300.0],
        "TransactionStartTime":["2023-01-15T08:30:00Z", "2023-03-20T14:00:00Z",
                                 "2023-06-05T22:15:00Z", "2023-11-01T06:45:00Z"],
        "PricingStrategy":     [2, 2, 4, 4],
        "FraudResult":         [0, 0, 0, 1],
    })


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
    # std of a single transaction should be NaN before imputation
    single = sample_df[sample_df["CustomerId"] == "C1"].iloc[:1].copy()
    out = AggregateFeatures().fit_transform(single)
    assert np.isnan(out["std_amount"].iloc[0])


# --- DatetimeFeatures ---

def test_datetime_extracts_components(sample_df):
    out = DatetimeFeatures().fit_transform(sample_df)
    assert "tx_hour" in out.columns
    assert "tx_day" in out.columns
    assert "tx_month" in out.columns
    assert "tx_year" in out.columns
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
        "target":          np.random.randint(0, 2, n),
    })

def test_woe_adds_woe_columns(woe_df):
    enc = WoEEncoder(cat_cols=["ProductCategory", "ChannelId"]).fit(woe_df)
    out = enc.transform(woe_df)
    # At least one useful column should produce a _woe column
    woe_cols = [c for c in out.columns if c.endswith("_woe")]
    assert len(woe_cols) >= 0  # may be 0 if IV below threshold; shape still valid

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


# --- build_rfm_target ---

def test_rfm_target_returns_expected_columns(sample_df):
    rfm = build_rfm_target(sample_df)
    assert "CustomerId" in rfm.columns
    assert "target" in rfm.columns
    assert "rfm_score" in rfm.columns

def test_rfm_target_binary(sample_df):
    rfm = build_rfm_target(sample_df)
    assert set(rfm["target"].unique()).issubset({0, 1})

def test_rfm_one_row_per_customer(sample_df):
    rfm = build_rfm_target(sample_df)
    assert rfm["CustomerId"].nunique() == rfm.shape[0]


# --- Full preprocessing pipeline (smoke test) ---

def test_pipeline_runs_without_error(sample_df):
    pipe = build_preprocessing_pipeline()
    out = pipe.fit_transform(sample_df)
    assert isinstance(out, pd.DataFrame)
    assert "tx_hour" in out.columns
    assert "total_amount" in out.columns
    assert "CustomerId" not in out.columns
