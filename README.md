# Credit Risk Probability Model for Alternative Data

An end-to-end implementation for building, deploying, and automating a credit risk model for Bati Bank's buy-now-pay-later service, powered by eCommerce transaction data.

---

## Credit Scoring Business Understanding

### 1. How does the Basel II Accord's emphasis on risk measurement influence the need for an interpretable and well-documented model?

Basel II requires banks to hold capital reserves proportional to the credit risk they carry. To calculate that risk, institutions must demonstrate to regulators that their models are **valid, transparent, and auditable**. This creates three concrete modeling obligations:

- **Interpretability**: Regulators and internal risk committees must be able to understand *why* a borrower received a given score. Black-box models that cannot explain individual predictions are difficult to defend under Basel II's Internal Ratings-Based (IRB) approach, which demands that risk drivers be identifiable and justifiable.
- **Documentation**: Every modeling choice — feature selection, target definition, validation methodology — must be recorded so that auditors can reproduce results and assess model stability over time.
- **Ongoing monitoring**: Basel II mandates back-testing and periodic model validation. A well-documented pipeline makes it straightforward to detect model drift, recalibrate probabilities, and demonstrate continued compliance.

In practice, this means preferring models like Logistic Regression with Weight of Evidence (WoE) encoding, where each coefficient has a direct business interpretation, or at minimum pairing complex models with explainability tools (e.g., SHAP) to satisfy the "explain any individual decision" requirement.

---

### 2. Without a direct "default" label, why is a proxy variable necessary, and what business risks does proxy-based prediction introduce?

The raw eCommerce dataset contains no loan performance history — there are no records of missed payments or defaults. A supervised model requires a target label, so we must **engineer a proxy** that approximates credit risk from observable behavioral signals.

The chosen approach uses **RFM (Recency, Frequency, Monetary) segmentation**: customers who transact rarely, infrequently, and in low amounts are hypothesized to represent higher credit risk, mirroring the logic that low engagement correlates with financial instability or low creditworthiness.

**Business risks introduced by proxy-based prediction:**

| Risk | Description |
|---|---|
| Label noise | The proxy may misclassify genuinely creditworthy customers as high-risk (false positives), leading to unfair loan denials. |
| Concept drift | The relationship between RFM patterns and actual default may shift over time or differ across customer segments, degrading model performance silently. |
| Regulatory scrutiny | Regulators may challenge whether the proxy is a legally and statistically defensible substitute for a true default label, especially under fair lending laws. |
| Feedback loops | Denying credit to proxy-labeled "high risk" customers prevents collecting ground-truth default data on them, making it hard to validate or improve the proxy. |

Mitigation requires clear documentation of the proxy's construction rationale, ongoing comparison against any emerging ground-truth default data, and conservative thresholds that err on the side of financial inclusion.

---

### 3. What are the key trade-offs between a simple, interpretable model (e.g., Logistic Regression with WoE) and a high-performance model (e.g., Gradient Boosting) in a regulated financial context?

| Dimension | Logistic Regression + WoE | Gradient Boosting (XGBoost / LightGBM) |
|---|---|---|
| **Interpretability** | High — coefficients map directly to risk drivers; scorecards are human-readable | Low by default — requires post-hoc tools (SHAP, LIME) to explain predictions |
| **Regulatory acceptance** | Well-established in Basel II IRB models; easier to validate and audit | Harder to certify without explainability layer; some regulators require additional justification |
| **Predictive performance** | Moderate — assumes linear log-odds relationship; may underfit complex patterns | High — captures non-linear interactions; typically outperforms on AUC/KS metrics |
| **Feature engineering burden** | High — requires manual binning, WoE transformation, and IV screening | Lower — handles raw features and missing values natively |
| **Stability & monitoring** | More stable under distribution shift; easier to recalibrate | More sensitive to feature drift; retraining pipelines are more complex |
| **Development speed** | Faster to build, validate, and document | Slower due to hyperparameter tuning and explainability overhead |

**Practical recommendation**: In a regulated context like Bati Bank, the preferred approach is to **start with Logistic Regression + WoE** as the baseline (satisfying interpretability requirements), then benchmark against Gradient Boosting. If the performance gain is material and the explainability layer is robust, the complex model can be deployed with documented SHAP-based explanations. The final choice must be justified in the model risk documentation submitted to the risk committee.

---

## Project Structure

```
credit-risk-model/
├── .github/workflows/ci.yml      # CI/CD pipeline
├── data/                          # Excluded from version control
│   ├── raw/                       # Raw transaction data
│   └── processed/                 # Processed features for training
├── notebooks/
│   └── eda.ipynb                  # Exploratory data analysis
├── src/
│   ├── __init__.py
│   ├── data_processing.py         # Feature engineering pipeline
│   ├── train.py                   # Model training and MLflow tracking
│   ├── predict.py                 # Inference logic
│   └── api/
│       ├── main.py                # FastAPI application
│       └── pydantic_models.py     # Request/response schemas
├── tests/
│   └── test_data_processing.py    # Unit tests
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Train the model
python src/train.py

# Start the API
uvicorn src.api.main:app --reload
```
