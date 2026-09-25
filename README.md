# 🚨 Urban Incident Severity Prediction & Emergency Dispatch System

An intelligent decision-support system designed for **911 CAD Call-Takers** and **DOT Regional Traffic Management Centers** to classify vehicular incident impact in real-time and provide automated triage response directives.

---

## 📌 Problem Scenario & Objective
Emergency call centers often make triage decisions based on fragmented caller observations, leading to under-dispatching or over-allocation of municipal resources. This system ingests preliminary scene parameters—temporal factors, ambient weather readings, and road geometry hazards—to forecast traffic disruption on a multiclass impact scale from **Level 1 to Level 4** using machine learning.

## 🛠️ Tech Stack & Architecture
- **Language & Runtime:** Python 3.10+
- **Data Engineering & ML:** Pandas, NumPy, Scikit-Learn, Joblib
- **Modeling Algorithms:** HistGradientBoostingClassifier, Random Forest, Multinomial Logistic Regression, LinearSVC
- **Web Interface:** Streamlit (Custom CAD CSS & Dispatch Scenario Presets)

---

## 📂 Project Structure
```text
fdm-mini-project/
├── data/                       # Contains raw/trimmed dataset (git-ignored)
├── models/
│   ├── preprocessor.joblib     # Serialized preprocessing ColumnTransformer
│   └── best_model.joblib       # Tuned HistGradientBoosting champion model
├── notebooks/
│   ├── 01_eda.ipynb            # Exploratory Data Analysis & visual plots
│   ├── 02_preprocessing.ipynb  # Cleaning, imputation, feature engineering, export
│   └── 03_modeling.ipynb       # 4-model evaluation, CV, and hyperparameter tuning
├── src/
│   ├── assets/
│   │   └── style.css           # Custom dispatch UI styling and animations
│   └── app.py                  # Live Streamlit decision-support application
├── .gitignore
├── requirements.txt
└── README.md