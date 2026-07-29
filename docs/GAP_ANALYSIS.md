# Capstone Gap Analysis & Improvement Plan

**Selected Project:** Credit Risk Probability Model for Alternative Data (Bati Bank BNPL)

**Why this project:** It is the strongest end-to-end ML system in the portfolio — it addresses a real finance-sector constraint (no default labels), includes production serving (FastAPI + Docker + MLflow), and maps directly to Basel II model risk requirements that finance recruiters recognize.

---

## Gap Analysis Checklist

| Category | Question | Status |
|---|---|---|
| **Code Quality** | Is the code modular and well-organized? | **Partial → Yes** (refactored with `config.py`, `constants.py`, separate `predict.py` / `explain.py`) |
| | Are there type hints on functions? | **Partial → Yes** (added across new modules; existing transformers retained hints) |
| | Is there a clear project structure? | **Yes** |
| **Testing** | Are there unit tests for core functions? | **Yes** (30+ tests covering processing, training, API, SHAP) |
| | Do tests run automatically on push? | **Yes** (GitHub Actions CI) |
| **Documentation** | Is the README comprehensive? | **Partial → Yes** (rewritten with business problem, results, quick start, demo) |
| | Are there docstrings on functions? | **Partial → Yes** (key public functions documented) |
| **Reproducibility** | Can someone else run this project? | **Yes** (requirements.txt, Docker, documented setup) |
| | Are dependencies in requirements.txt? | **Yes** |
| **Visualization** | Is there an interactive way to explore results? | **No → Yes** (Streamlit dashboard added) |
| **Business Impact** | Is the problem clearly articulated? | **Yes** |
| | Are success metrics defined? | **Partial → Yes** (ROC-AUC, time saved, false decline reduction in README) |

---

## Prioritized Improvement Plan

| Priority | Improvement | Estimate | Justification |
|---|---|---|---|
| 1 | **Code refactoring** — dataclasses, constants, modular inference/explain modules | 4 hours | Addresses critical maintainability gap; finance employers expect production-grade structure |
| 2 | **SHAP explainability** — global + local attributions, `/explain` API endpoint | 4 hours | Highest regulatory value (Basel II IRB); directly answers "why was this customer declined?" |
| 3 | **Streamlit dashboard** — metrics, interactive scoring, SHAP charts | 5 hours | Makes business impact tangible for non-technical stakeholders and recruiters |
| 4 | **Expanded test suite + CI badge** — API integration tests, predict/explain tests | 3 hours | Proves reliability; CI badge signals engineering maturity on GitHub |
| 5 | **Professional README + technical report** | 2 hours | First thing recruiters see; must tell the financial story clearly |

**Total estimated effort:** ~18 hours (achievable within one week alongside polish and demo prep)

---

## Implementation Status

- [x] Gap analysis completed
- [x] Code refactored (`src/config.py`, `src/constants.py`, `src/predict.py`, `src/explain.py`)
- [x] SHAP explainability module + `/explain` API endpoint
- [x] Streamlit dashboard (`src/dashboard/app.py`)
- [x] 30+ tests with API integration coverage
- [x] CI/CD pipeline (existing, verified)
- [x] CI badge added to README
- [x] README rewritten to capstone template
- [x] Technical report at `reports/final_report.md`
