# Interim Report — Credit Risk Probability Model for Alternative Data

**Project:** Bati Bank Buy-Now-Pay-Later Credit Scoring
**Author:** Analytics Engineer, Bati Bank
**Scope:** Tasks 1 & 2 — Project Understanding and Exploratory Data Analysis

---

## 1. Project Overview

Bati Bank is partnering with an eCommerce platform to offer a buy-now-pay-later (BNPL) service. The objective is to build a production-grade credit scoring system that assigns a risk probability to each applicant, enabling the loan origination team to make real-time credit decisions.

The end-to-end deliverable covers five capabilities:

1. A proxy variable that categorizes customers as high-risk or low-risk
2. Observable features with high predictive correlation to the proxy default label
3. A model that assigns risk probability to new customers
4. A credit score derived from risk probability estimates
5. A model that predicts optimal loan amount and duration

---

## 2. Credit Scoring Business Understanding (Task 1)

### 2.1 Basel II and the Need for Interpretable, Documented Models

Basel II requires banks to hold capital reserves proportional to their credit risk exposure. Under the Internal Ratings-Based (IRB) approach, institutions must demonstrate to regulators that their models are valid, transparent, and auditable. This imposes three concrete obligations on the modeling team:

- **Interpretability:** Regulators and risk committees must understand *why* a borrower received a given score. Black-box models are difficult to defend; each risk driver must be identifiable and justifiable.
- **Documentation:** Every modeling choice — feature selection, target definition, validation methodology — must be recorded so auditors can reproduce results and assess model stability over time.
- **Ongoing monitoring:** Basel II mandates back-testing and periodic model validation. A well-documented pipeline makes it straightforward to detect model drift, recalibrate probabilities, and demonstrate continued compliance.

In practice, this favors Logistic Regression with Weight of Evidence (WoE) encoding as a baseline, where each coefficient maps directly to a risk driver, supplemented by explainability tools (e.g., SHAP) if a more complex model is ultimately deployed.

### 2.2 Proxy Variable Necessity and Associated Business Risks

The Xente eCommerce dataset contains no loan performance history — no records of missed payments or defaults. A supervised model requires a target label, so a proxy must be engineered from observable behavioral signals.

The chosen approach uses **RFM (Recency, Frequency, Monetary) segmentation**: customers who transact rarely, infrequently, and in low amounts are hypothesized to represent higher credit risk, on the basis that low engagement correlates with financial instability or low creditworthiness.

| Risk | Description |
|---|---|
| Label noise | The proxy may misclassify creditworthy customers as high-risk, leading to unfair loan denials. |
| Concept drift | The RFM–default relationship may shift over time or across segments, silently degrading model performance. |
| Regulatory scrutiny | Regulators may challenge whether the proxy is a legally and statistically defensible substitute for a true default label. |
| Feedback loops | Denying credit to proxy-labeled "high risk" customers prevents collecting ground-truth default data, making the proxy hard to validate or improve. |

Mitigation requires clear documentation of the proxy's construction rationale, conservative classification thresholds, and ongoing comparison against any emerging ground-truth default data.

### 2.3 Model Trade-offs: Logistic Regression + WoE vs. Gradient Boosting

| Dimension | Logistic Regression + WoE | Gradient Boosting (XGBoost / LightGBM) |
|---|---|---|
| Interpretability | High — coefficients map directly to risk drivers | Low by default — requires SHAP/LIME post-hoc |
| Regulatory acceptance | Well-established in Basel II IRB models | Requires additional justification without explainability layer |
| Predictive performance | Moderate — assumes linear log-odds | High — captures non-linear interactions |
| Feature engineering burden | High — manual binning, WoE, IV screening | Lower — handles raw features natively |
| Stability & monitoring | More stable under distribution shift | More sensitive to feature drift |
| Development speed | Faster to build and document | Slower due to tuning and explainability overhead |

**Recommendation:** Start with Logistic Regression + WoE as the interpretable baseline, then benchmark against Gradient Boosting. If the performance gain is material and a robust SHAP-based explainability layer is in place, the complex model can be deployed with documented justification submitted to the risk committee.

---

## 3. Dataset Description

The dataset is sourced from the Xente eCommerce platform. Each row represents a single customer transaction. Key fields are summarized below:

| Field | Type | Description |
|---|---|---|
| TransactionId | Categorical (ID) | Unique transaction identifier |
| AccountId / CustomerId | Categorical (ID) | Customer identifiers |
| Amount | Numerical | Transaction value; positive = debit, negative = credit/refund |
| Value | Numerical | Absolute value of Amount |
| TransactionStartTime | Datetime | Transaction timestamp |
| ProductCategory | Categorical | Broad product grouping |
| ChannelId | Categorical | Transaction channel (web, Android, iOS, pay-later, checkout) |
| PricingStrategy | Categorical | Merchant pricing category |
| FraudResult | Binary | Fraud flag (1 = fraud, 0 = legitimate) |

---

## 4. EDA Findings (Task 2)

### 4.1 Data Structure

The dataset contains transaction-level records with a mix of numerical, categorical, and datetime fields. ID columns (TransactionId, BatchId, AccountId, SubscriptionId, CustomerId) are high-cardinality and excluded from distributional analysis.

### 4.2 Summary Statistics

Numerical features (Amount, Value, CountryCode, PricingStrategy, FraudResult) show wide dispersion. Amount spans both positive and negative values, confirming the presence of credit/refund transactions alongside debits.

### 4.3 Distribution of Numerical Features

- **Amount and Value are heavily right-skewed** — a small number of very large transactions dominate the distribution. Log-transformation (`log1p`) substantially normalizes both features and will be applied during feature engineering.
- **Amount contains negative values** — these represent refunds or reversals and must be separated from debit transactions or handled via absolute-value transformation depending on the RFM aggregation strategy.

### 4.4 Distribution of Categorical Features

- **ProductCategory** is concentrated in a small number of categories with a long tail of low-frequency ones — a strong candidate for WoE encoding.
- **ChannelId** shows similar concentration; a few channels (web, Android) account for the majority of transactions, while pay-later and checkout are minority channels that may carry distinct risk profiles.
- **PricingStrategy** has limited unique values, making it straightforward to encode as a categorical risk factor.

### 4.5 Correlation Analysis

- **Amount and Value are near-perfectly correlated** (expected, since Value = |Amount|). Only one should be retained to avoid multicollinearity.
- **FraudResult shows weak linear correlation** with Amount and Value, suggesting fraud is not simply a function of transaction size and that non-linear models or interaction features may be needed.

### 4.6 Missing Values

No structural missing values were identified in the core fields. Imputation strategy will therefore focus on derived RFM features (e.g., customers with no recent transactions will have null recency if not handled explicitly) rather than raw columns.

### 4.7 Outlier Detection

IQR-based analysis confirms significant outliers in Amount and Value, consistent with the right-skewed distributions. These extreme values are likely legitimate high-value transactions rather than data errors, so **winsorization or log-transformation** is preferred over removal.

---

## 5. Top Insights and Implications for Feature Engineering

1. **Log-transform Amount and Value** — right skew is severe; raw values will distort distance-based and linear models. Apply `log1p(|Amount|)` as the monetary RFM input.

2. **Separate debits from credits** — negative Amount values (refunds) should be aggregated separately. Net monetary value and refund frequency are both potentially informative risk signals.

3. **FraudResult is highly imbalanced** — fraudulent transactions are a small minority. Any model using FraudResult as a feature or proxy component will require class-weight adjustment or resampling (SMOTE) to avoid a degenerate classifier.

4. **ProductCategory and ChannelId are high-signal categoricals** — their concentration and potential interaction with risk make them strong WoE encoding candidates. The pay-later channel in particular warrants close examination as a risk differentiator.

5. **No missing data simplifies the pipeline** — the absence of structural missingness means the feature engineering pipeline can focus on RFM aggregation and encoding rather than imputation, reducing a common source of model instability.

---

## 6. Next Steps (Tasks 3–5)

- **Task 3:** Engineer RFM features per customer, define the proxy default label via K-Means or percentile-based segmentation, and build the full feature engineering pipeline in `src/data_processing.py`.
- **Task 4:** Train and compare Logistic Regression + WoE, Random Forest, and XGBoost/LightGBM; track all experiments with MLflow; select the best model based on AUC-ROC and KS statistic.
- **Task 5:** Package the selected model as a FastAPI service, containerize with Docker, and automate testing via the GitHub Actions CI/CD pipeline.
