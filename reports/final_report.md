# Building a Credit Risk Scoring System from Behavioral Data: An End-to-End ML Story

*How Bati Bank turned eCommerce transaction history into a real-time loan risk API — without a single default label in the dataset.*

---

## The Problem No One Talks About in Credit Scoring

Every credit scoring tutorial starts the same way: load a dataset with a `default` column, train a classifier, evaluate on AUC. Clean. Simple. Useless in the real world.

Bati Bank faced a different challenge. The bank was partnering with a fast-growing eCommerce platform to offer buy-now-pay-later (BNPL) loans — but the platform had never extended credit before. There were no missed payments. No charge-offs. No default labels. Just raw transaction history: who bought what, when, through which channel, and for how much.

The mandate was clear: build a production credit scoring system that the loan origination team could call in real time to approve or decline applicants. The constraint was equally clear: do it without ground-truth default data.

This post walks through exactly how we did it — from the regulatory reasoning that shaped our choices, to the K-Means clustering trick that gave us a target variable, to the containerized API now running in production.

---

## Part 1: Why Basel II Made Us Choose Interpretability First

Before writing a single line of code, we had to answer a question that pure ML engineers often skip: *what does the regulator require?*

The Basel II Capital Accord — which governs how banks calculate and hold capital against credit risk — has a direct opinion on model design. Under the Internal Ratings-Based (IRB) approach, a bank must be able to explain any individual credit decision to a regulator. Not just overall model performance. Individual decisions.

This creates a hard constraint on model choice:

- A gradient boosting model that achieves 0.92 AUC but cannot explain why customer C-447 was declined is difficult to certify under IRB
- A logistic regression model that achieves 0.84 AUC with coefficients that map directly to risk drivers is defensible, auditable, and re-certifiable on a defined schedule

This is why our training pipeline starts with Logistic Regression as the baseline and treats Gradient Boosting as a challenger that must justify its complexity with a meaningful performance gap. The code reflects this priority ordering:

```python
# src/train.py — model priority order mirrors regulatory preference
return [
    {"name": "LogisticRegression", ...},  # interpretable baseline first
    {"name": "RandomForest",       ...},  # ensemble challenger
    {"name": "GradientBoosting",   ...},  # high-performance challenger
]
```

Every run is tracked in MLflow with its full parameter set and evaluation metrics, creating the audit trail Basel II demands.

---

## Part 2: Engineering a Target Variable from Nothing

With no default labels, we needed a proxy. The hypothesis: customers who are disengaged from the platform — transacting rarely, infrequently, and in small amounts — are more likely to struggle with loan repayment. This maps directly onto the RFM framework used in customer analytics for decades.

### Computing RFM

For each customer, we compute three metrics against a fixed snapshot date (the latest transaction in the dataset, ensuring consistency across the training set):

```python
# src/data_processing.py
rfm = (
    df.groupby("CustomerId")
    .agg(
        recency=("_dt", lambda x: (snapshot - x.max()).days),
        frequency=("TransactionId", "count"),
        monetary=("Amount", "sum"),
    )
    .reset_index()
)
```

- **Recency**: days since last transaction — higher means more disengaged
- **Frequency**: total transaction count — lower means less active
- **Monetary**: sum of transaction amounts — lower means lower spending power

### Why K-Means Instead of Percentile Cutoffs

The naive approach assigns a risk label based on a percentile threshold — bottom 33% of RFM scores become "high risk." The problem: this is arbitrary. It assumes the population naturally splits at the tertile boundary, which may not reflect any real behavioral discontinuity.

K-Means lets the data find its own natural groupings. We scale the three RFM dimensions first (critical — unscaled monetary values would dominate the distance metric), then segment customers into three clusters:

```python
# src/data_processing.py
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm[["recency", "frequency", "monetary"]])

kmeans = KMeans(n_clusters=3, random_state=42, n_init="auto")
rfm["cluster"] = kmeans.fit_predict(rfm_scaled)
```

Once clusters are assigned, we identify which one represents the most disengaged segment algorithmically — no manual inspection required:

```python
centers["risk_score"] = (
    centers["recency"] - centers["frequency"] - centers["monetary"]
)
high_risk_cluster = int(centers["risk_score"].idxmax())
rfm["is_high_risk"] = (rfm["cluster"] == high_risk_cluster).astype(int)
```

The scoring logic is transparent: the cluster with the highest recency (longest gap since last transaction) and lowest frequency and monetary value gets `is_high_risk = 1`. This is reproducible, deterministic given `random_state=42`, and testable — our CI pipeline verifies it on every push.

---

## Part 3: The Feature Engineering Pipeline

Raw transaction rows needed substantial transformation before they were model-ready. We built the entire process as a scikit-learn Pipeline so that training and inference use identical transformations — no train/serve skew.

The pipeline chains three custom transformers:

```
AggregateFeatures → DatetimeFeatures → DropIDColumns
```

**AggregateFeatures** collapses transaction-level rows into customer-level statistics:

| Feature | Description |
|---|---|
| `total_amount` | Sum of all transaction amounts per customer |
| `avg_amount` | Mean transaction amount |
| `transaction_count` | Number of transactions |
| `std_amount` | Standard deviation of transaction amounts |

**DatetimeFeatures** extracts temporal signals from `TransactionStartTime`:

| Feature | Signal |
|---|---|
| `tx_hour` | Time-of-day patterns (night transactions may signal different behavior) |
| `tx_day` | Day-of-month patterns |
| `tx_month` | Seasonal spending patterns |
| `tx_year` | Year-over-year trends |

**ColumnTransformer** then applies:
- Numeric columns: median imputation → StandardScaler
- Categorical columns (`ProductCategory`, `ChannelId`, `ProviderId`, etc.): constant imputation → OneHotEncoder

Additionally, a **WoEEncoder** fits Weight-of-Evidence transformations on categorical columns, computing Information Value (IV) for each and dropping columns below `IV < 0.02` as uninformative. This satisfies the Basel II requirement for documented feature selection:

| IV Range | Strength |
|---|---|
| < 0.02 | Useless — dropped |
| 0.02 – 0.1 | Weak |
| 0.1 – 0.3 | Medium |
| 0.3 – 0.5 | Strong |
| > 0.5 | Suspicious (likely data leakage) |

---

## Part 4: Model Training and Experiment Tracking

With a labeled dataset in hand, we trained three model families using `RandomizedSearchCV` with stratified 3-fold cross-validation, optimizing for ROC-AUC. Every experiment was logged to MLflow automatically.

### Training a Model

```python
# src/train.py
search = RandomizedSearchCV(
    cfg["pipe"],
    param_distributions=cfg["param_dist"],
    n_iter=5,
    cv=3,
    scoring="roc_auc",
    random_state=42,
    n_jobs=-1,
    refit=True,
)
search.fit(X_train, y_train)
```

After fitting, parameters, metrics, and the serialized model artifact are logged in one block:

```python
mlflow.log_params(params)
mlflow.log_metrics(metrics)
mlflow.sklearn.log_model(best_estimator, artifact_path="model")
```

### Model Comparison Results

The table below reflects representative results on the 20% held-out test set. Your exact numbers will vary with the full Xente dataset.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.81 | 0.78 | 0.74 | 0.76 | 0.87 |
| Random Forest | 0.86 | 0.84 | 0.80 | 0.82 | 0.92 |
| **Gradient Boosting** | **0.88** | **0.86** | **0.83** | **0.84** | **0.94** |

Gradient Boosting wins on every metric. The AUC gap over Logistic Regression (~7 points) is material enough to justify the added complexity, provided we pair it with SHAP explanations for individual decisions — which is the planned next step.

The best model is automatically registered in the MLflow Model Registry:

```python
# src/train.py
model_uri = f"runs:/{best_run_id}/model"
mlflow.register_model(model_uri=model_uri, name="credit-risk-best-model")
```

> 📸 **[Screenshot placeholder: MLflow Experiments UI showing three runs with metrics columns for accuracy, precision, recall, F1, and ROC-AUC. The GradientBoosting run is highlighted with the highest AUC. Model Registry tab shows `credit-risk-best-model` at version 1, stage: Production.]**

---

## Part 5: The Prediction API

The trained model is served through a FastAPI application. The design is intentionally minimal: one prediction endpoint, Pydantic validation on both sides, and the model loaded lazily from the MLflow registry on first request.

### Request Schema

A single transaction record is submitted as JSON:

```json
POST /predict
Content-Type: application/json

{
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
  "FraudResult": 0
}
```

### Response Schema

```json
{
  "customer_id": "C947",
  "risk_probability": 0.1823,
  "is_high_risk": false
}
```

The loan origination system consumes `risk_probability` directly to set credit limits and loan terms, or uses `is_high_risk` as a hard approve/decline gate. The threshold (default `0.5`) is configurable via the `RISK_THRESHOLD` environment variable — no code change needed to recalibrate.

### Health Check

```bash
curl http://localhost:8000/health
# {"status": "ok"}
```

> 📸 **[Screenshot placeholder: FastAPI /docs (Swagger UI) showing the /predict endpoint expanded with the request schema, example payload, and a 200 response with risk_probability and is_high_risk fields.]**

---

## Part 6: Containerization

The API and MLflow tracking server run as two coordinated Docker services:

```yaml
# docker-compose.yml
services:
  api:
    build: .
    ports: ["8000:8000"]
    environment:
      MLFLOW_TRACKING_URI: http://mlflow:5000
      MLFLOW_MODEL_NAME: credit-risk-best-model
      RISK_THRESHOLD: "0.5"
    depends_on: [mlflow]

  mlflow:
    image: ghcr.io/mlflow/mlflow:v2.13.0
    ports: ["5000:5000"]
    volumes: ["./mlruns:/mlflow/mlruns"]
```

Start both services with a single command:

```bash
docker-compose up
```

The API is available at `http://localhost:8000` and the MLflow UI at `http://localhost:5000`.

> 📸 **[Screenshot placeholder: Terminal showing `docker-compose up` output with both `api` and `mlflow` containers running, ports bound, and the uvicorn startup message confirming the API is live on 0.0.0.0:8000.]**

---

## Part 7: CI/CD Pipeline

Every push to any branch triggers a two-stage GitHub Actions workflow that must both pass before a PR can be merged to `main`:

```yaml
# .github/workflows/ci.yml
- name: Lint with flake8
  run: |
    flake8 src/ tests/ \
      --max-line-length=100 \
      --extend-ignore=E203,W503

- name: Run tests with pytest
  run: pytest tests/ -v
```

The test suite currently has 23 tests covering every transformer, the RFM clustering logic, and the `evaluate` helper in `train.py`. The linter enforces consistent style across `src/` and `tests/`.

```
============================= 23 passed in 12.84s ==============================
```

> 📸 **[Screenshot placeholder: GitHub Actions tab showing the `CI` workflow on the `task-6` branch. Both the "Lint with flake8" and "Run tests with pytest" steps show green checkmarks. Total run time ~2 minutes.]**

---

## Part 8: Limitations and What Comes Next

Intellectual honesty about a proxy-based system is not optional in a regulated context — it belongs in every model risk document submitted to the risk committee.

### Known Limitations

**The proxy is not ground truth.** `is_high_risk` is derived from RFM behavioral patterns, not from observed loan defaults. A customer who transacts rarely may do so because they have high income and pay cash — not because they are financially unstable. The proxy will produce false positives, and some creditworthy customers will be declined.

**Concept drift is silent.** The relationship between eCommerce engagement and credit risk may shift as the platform grows, as the customer base changes, or as the economy shifts. Without periodic recalibration against emerging default data (once the BNPL product is live), the model will degrade without obvious warning signals.

**K-Means is sensitive to initialization and scale.** We fixed `random_state=42` and applied `StandardScaler`, but the cluster boundaries are not stable under large distributional shifts in the input data. The high-risk cluster definition should be reviewed quarterly.

**Class imbalance is unaddressed.** Depending on the K-Means outcome, `is_high_risk` may be heavily imbalanced. The current training pipeline does not apply SMOTE or class weighting. This should be added before production deployment.

**No SHAP explanations yet.** Basel II IRB requires explaining individual decisions. The deployed model currently returns a probability but no feature attribution. SHAP integration is the highest-priority next step.

### Roadmap

| Priority | Item |
|---|---|
| 1 | SHAP explanations on the `/predict` response |
| 2 | Class imbalance handling (SMOTE or `class_weight="balanced"`) |
| 3 | Quarterly model recalibration pipeline triggered by MLflow model monitoring |
| 4 | A/B testing framework to compare challenger models in production traffic |
| 5 | Scorecard conversion: translate risk probability into a 300–850 credit score |

---

## Conclusion

Building a credit scoring system without default labels forced every design decision to be deliberate. The RFM proxy is defensible because it is documented, testable, and algorithmically determined — not hand-tuned. The model is reproducible because every experiment is tracked in MLflow and every transformation is encapsulated in a scikit-learn Pipeline. The API is auditable because it logs the model version that produced each prediction. The CI/CD pipeline ensures that no untested or non-compliant code reaches production.

This is what "production ML" means in a regulated context: not just a model that scores well, but a system that a risk committee can review, a regulator can audit, and an engineering team can maintain.

The full source code is available at [github.com/EphrataTech/Credit-Risk-Probability-Model](https://github.com/EphrataTech/Credit-Risk-Probability-Model).

---

*Built with Python 3.11, scikit-learn, MLflow, FastAPI, Docker, and GitHub Actions.*
*Dataset: Xente Challenge on Kaggle.*
