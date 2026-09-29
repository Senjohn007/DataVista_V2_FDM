# 🚨 Metro CAD — Urban Incident Severity Prediction & Emergency Dispatch System

An intelligent decision-support system designed for **911 CAD Call-Takers** and **DOT Regional Traffic Management Centers** to classify vehicular incident impact in real-time and provide automated triage response directives.

---

## 📌 Problem Scenario & Objective
Emergency call centers often make triage decisions based on fragmented, high-stress caller observations, leading to either under-dispatching (delayed medical care) or over-allocation of municipal resources. This system ingests preliminary scene parameters—temporal factors, ambient weather readings, and road geometry hazards—to forecast traffic disruption on a multiclass impact scale from **Level 1 to Level 4** using machine learning.

- **Level 1 (Minor):** Localized shoulder incident; negligible traffic delay.
- **Level 2 (Moderate):** Routine single-lane disruption; short clearance duration (~70% of historical records).
- **Level 3 (Significant):** Multi-lane blockage; substantial tailbacks and prolonged clearance times.
- **Level 4 (Critical):** Complete corridor or highway closure; requires multi-agency EMS and upstream diversion routing.

---

## 🛠️ Tech Stack & Architecture
- **Language & Runtime:** Python 3.10+
- **Data Engineering & ML:** Pandas, NumPy, Scikit-Learn, Joblib
- **Modeling Algorithms:** HistGradientBoostingClassifier (Champion), Random Forest, Multinomial Logistic Regression, LinearSVC
- **Evaluation Strategy:** 5-Fold Stratified Cross-Validation evaluated via **Macro F1-Score** (cost-sensitive balanced weighting)
- **Web Interface:** Streamlit (Custom CAD CSS Tokens, Dynamic Status Bars, Simulation Presets, and Bidirectional Dark/Light Theme Support)

---

## 📂 Project Structure
```text
fdm-mini-project/
├── data/
│   ├── US_Accidents_2016_2023_1M.csv         # 1.0M clean master dataset (13% chunk-streamed, git-ignored)
│   └── train_working_slice.csv               # 100k stratified working slice for fast CV convergence
├── models/
│   ├── preprocessor.joblib                   # Serialized ColumnTransformer pipeline (imputer + scaler + OHE)
│   └── best_model.joblib                     # Tuned HistGradientBoosting champion model
├── notebooks/
│   ├── 01_problem_and_sampling.ipynb         # Member 1: Scenario framing, 25-col drop & 13% streaming logic
│   ├── 02_eda_and_data_insights.ipynb        # Member 2: Target skew, rush-hour peaks & junction risk analysis
│   ├── 03_cleaning_and_imputation.ipynb      # Member 3: Leakage-free stratified split, domain imputation & mean shift audit (<0.8%)
│   ├── 04_feature_eng_and_pipeline.ipynb     # Member 4: Temporal engineering, ColumnTransformer export & smoke test
│   └── 05_modeling.ipynb                     # 4-Algorithm benchmarking, 5-Fold CV, GridSearchCV & export
├── scripts/
│   └── extract_1m.py                         # Standalone chunk-streaming memory-safe extractor (<80 MB RAM)
├── src/
│   ├── assets/
│   │   └── style.css                         # Municipal CAD dark/light adaptive styling & radar animations
│   └── app.py                                # Live Streamlit dispatch decision-support application
├── .gitignore
├── requirements.txt
└── README.md