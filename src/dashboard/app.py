"""Streamlit dashboard for credit risk model exploration and explainability."""

from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

# Allow running as `streamlit run src/dashboard/app.py`
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.constants import (
    BASELINE_ROC_AUC,
    BEST_ROC_AUC,
    ESTIMATED_DECISION_TIME_SAVED,
    ESTIMATED_FALSE_DECLINE_REDUCTION,
)
from src.explain import explain_prediction, global_feature_importance
from src.predict import load_model, predict_risk

st.set_page_config(
    page_title="Bati Bank Credit Risk Dashboard",
    page_icon="📊",
    layout="wide",
)

EXAMPLE_TRANSACTION = {
    "TransactionId": "T1000",
    "BatchId": "B100",
    "AccountId": "A100",
    "SubscriptionId": "S100",
    "CustomerId": "C947",
    "CurrencyCode": "UGX",
    "CountryCode": 256,
    "ProviderId": "P1",
    "ProductId": "PR1",
    "ProductCategory": "airtime",
    "ChannelId": "web",
    "Amount": 150.0,
    "Value": 150.0,
    "TransactionStartTime": "2024-01-15T08:30:00Z",
    "PricingStrategy": 2,
    "FraudResult": 0,
}


@st.cache_resource
def get_model():
    return load_model()


@st.cache_data
def get_background() -> pd.DataFrame:
    path = os.getenv("BACKGROUND_DATA_PATH", "data/processed/processed.csv")
    if not os.path.exists(path):
        return pd.DataFrame([EXAMPLE_TRANSACTION])
    df = pd.read_csv(path)
    return df.drop(columns=["is_high_risk"]) if "is_high_risk" in df.columns else df


def render_overview() -> None:
    st.title("Credit Risk Scoring Dashboard")
    st.caption("Bati Bank · Buy-Now-Pay-Later · Behavioral Credit Scoring")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Best ROC-AUC", f"{BEST_ROC_AUC:.0%}", delta=f"+{(BEST_ROC_AUC - BASELINE_ROC_AUC):.0%} vs LR")
    col2.metric("Baseline ROC-AUC", f"{BASELINE_ROC_AUC:.0%}")
    col3.metric("Manual Review Time Saved", ESTIMATED_DECISION_TIME_SAVED)
    col4.metric("False Decline Reduction", ESTIMATED_FALSE_DECLINE_REDUCTION)

    st.markdown("### Business Problem")
    st.write(
        "Bati Bank partners with an eCommerce platform to offer BNPL loans, but no historical "
        "default labels exist. This system scores applicants in real time using transaction "
        "behavior, with full audit trails for Basel II model risk requirements."
    )

    st.markdown("### Model Comparison")
    comparison = pd.DataFrame([
        {"Model": "Logistic Regression", "ROC-AUC": 0.87, "Interpretability": "High"},
        {"Model": "Random Forest", "ROC-AUC": 0.92, "Interpretability": "Medium"},
        {"Model": "Gradient Boosting", "ROC-AUC": 0.94, "Interpretability": "Medium + SHAP"},
    ])
    st.dataframe(comparison, use_container_width=True, hide_index=True)


def render_predict() -> None:
    st.header("Score an Applicant")
    st.write("Enter transaction details to receive a real-time credit risk probability.")

    with st.form("prediction_form"):
        c1, c2 = st.columns(2)
        customer_id = c1.text_input("Customer ID", value=EXAMPLE_TRANSACTION["CustomerId"])
        amount = c2.number_input("Amount (UGX)", min_value=0.0, value=150.0)
        category = c1.selectbox(
            "Product Category",
            ["airtime", "data_bundles", "utility_bill", "financial_services"],
        )
        channel = c2.selectbox("Channel", ["web", "android", "ios"])
        tx_time = c1.text_input("Transaction Time (ISO)", value=EXAMPLE_TRANSACTION["TransactionStartTime"])
        submitted = st.form_submit_button("Score Applicant", type="primary")

    if submitted:
        payload = {**EXAMPLE_TRANSACTION, "CustomerId": customer_id, "Amount": amount,
                   "Value": amount, "ProductCategory": category, "ChannelId": channel,
                   "TransactionStartTime": tx_time}
        row = pd.DataFrame([payload])
        try:
            model = get_model()
            result = predict_risk(row, model=model)[0]
        except Exception as exc:
            st.error(f"Could not score applicant: {exc}")
            st.info("Train a model first with `python src/train.py` and ensure MLflow registry is available.")
            return

        prob = result["risk_probability"]
        is_high = result["is_high_risk"]
        st.metric("Risk Probability", f"{prob:.1%}")
        if is_high:
            st.error("Recommendation: Decline or manual review — high-risk segment.")
        else:
            st.success("Recommendation: Approve — within acceptable risk threshold.")


def render_explainability() -> None:
    st.header("Model Explainability (SHAP)")
    st.write("Understand which features drive global and individual credit decisions.")

    try:
        model = get_model()
        background = get_background()
    except Exception as exc:
        st.warning(f"Model not available: {exc}")
        return

    with st.spinner("Computing global feature importance..."):
        importance = global_feature_importance(model, background)

    st.subheader("Global Feature Importance")
    top = importance.head(10)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(top["feature"][::-1], top["mean_abs_shap"][::-1], color="#1f4e79")
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Top 10 Risk Drivers")
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Individual Prediction Explanation")
    row = pd.DataFrame([EXAMPLE_TRANSACTION])
    explanation = explain_prediction(model, row, background)
    st.metric("Example Customer Risk", f"{explanation['risk_probability']:.1%}")

    contrib_df = pd.DataFrame(explanation["feature_contributions"])
    st.dataframe(contrib_df, use_container_width=True, hide_index=True)

    fig2, ax2 = plt.subplots(figsize=(8, 4))
    colors = ["#c0392b" if v > 0 else "#27ae60" for v in contrib_df["shap_value"]]
    ax2.barh(contrib_df["feature"][::-1], contrib_df["shap_value"][::-1], color=colors[::-1])
    ax2.axvline(0, color="black", linewidth=0.8)
    ax2.set_xlabel("SHAP value (impact on risk)")
    ax2.set_title(f"Why customer {EXAMPLE_TRANSACTION['CustomerId']} received this score")
    st.pyplot(fig2)
    plt.close(fig2)


def main() -> None:
    tab_overview, tab_predict, tab_explain = st.tabs(["Overview", "Predict", "Explainability"])
    with tab_overview:
        render_overview()
    with tab_predict:
        render_predict()
    with tab_explain:
        render_explainability()


if __name__ == "__main__":
    main()
