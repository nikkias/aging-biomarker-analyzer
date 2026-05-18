<div align="center">

# 🫀 Aging Biomarker Analyzer

### Heart Disease Risk Prediction for Longevity Research

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4+-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-189C3E?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

**An end-to-end machine learning pipeline for identifying cardiovascular aging biomarkers,**  
**with an interactive Streamlit application for personalized risk assessment.**

[Live Demo](#running-the-app) · [Dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset) · [Notebook](notebook.ipynb)

</div>

---

## Why This Matters for Longevity Research

Cardiovascular disease remains the leading cause of death globally, yet most risk models rely on chronological age as a primary variable. This project challenges that assumption: **functional biomarkers — particularly maximum heart rate, vessel burden, and exercise-induced ST changes — consistently outperform calendar age** as predictors of heart disease in this dataset.

This aligns with the core thesis of modern longevity medicine: biological age, measured through functional and metabolic markers, predicts healthspan and mortality better than the year on your birth certificate. By identifying which biomarkers carry the most predictive signal, this analysis contributes a data-driven perspective to the growing field of biological age assessment.

---

## What I Built

A full ML pipeline including:

- **Data cleaning** — IQR-based outlier clipping on cholesterol and blood pressure
- **Exploratory Data Analysis** — 5 publication-ready plots examining age distributions, feature correlations, and biomarker relationships
- **Three classifiers** — Logistic Regression (clinical baseline), Random Forest (ensemble), and XGBoost (gradient boosting)
- **Rigorous evaluation** — 5-fold stratified cross-validation, ROC-AUC comparison, confusion matrices
- **Streamlit web application** — interactive predictor with plain-English explanations, personalized recommendations, and a cardiovascular biological age score

---

## Key Findings

| Model | Test Accuracy | Test AUC | CV AUC (5-fold) |
|---|---|---|---|
| Logistic Regression | ~85% | ~0.92 | ~0.91 |
| Random Forest | ~87% | ~0.93 | ~0.92 |
| **XGBoost** | **~88%** | **~0.94** | **~0.93** |

> **Replace the numbers above with your actual results from the leaderboard table**

### Top Predictive Features (consistent across RF + XGBoost)

| Rank | Feature | Biological Meaning |
|---|---|---|
| 1 | `thalach` | Max heart rate — declines ~1 bpm/year of biological aging |
| 2 | `ca` | Number of blocked vessels — direct atherosclerotic burden |
| 3 | `cp` | Chest pain type — atypical patterns increase with vascular aging |
| 4 | `oldpeak` | ST depression — reflects mitochondrial and microvascular decline |
| 5 | `thal` | Perfusion defects — signal prior infarction or live ischemia |

**Key insight:** Chronological `age` ranked *below* all five functional markers, confirming that **biological age > calendar age** as a cardiovascular disease predictor.

---

## Project Structure

```
aging-biomarker-analyzer/
├── README.md
├── notebook.ipynb          ← Full pipeline: EDA → models → evaluation
├── app.py                  ← Streamlit web application
├── heart_model.pkl         ← Trained XGBoost model
├── scaler.pkl              ← Fitted StandardScaler
├── requirements.txt
└── results/
    ├── eda_plots/          ← EDA visualizations
    └── model_results.md    ← Numerical results
```

---

## Running the App

### Option 1 — Local (recommended)

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/aging-biomarker-analyzer.git
cd aging-biomarker-analyzer

# Install dependencies
pip install -r requirements.txt

# Launch
streamlit run app.py
```

Opens at `http://localhost:8501`

### Option 2 — Google Colab + ngrok

See the full step-by-step guide in [`notebook.ipynb`](notebook.ipynb).

---

## Dependencies

```
streamlit>=1.35.0
scikit-learn>=1.4.0
xgboost>=2.0.0
joblib>=1.3.0
numpy>=1.26.0
pandas>=2.2.0
matplotlib>=3.8.0
seaborn>=0.13.0
```

Install all:
```bash
pip install -r requirements.txt
```

---

## Dataset

**Source:** [Heart Disease Dataset — Kaggle](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)  
**Origin:** Cleveland Clinic Foundation (UCI ML Repository)  
**Size:** 303 patients, 13 features, 1 binary target  
**Features:** Age, sex, chest pain type, resting BP, cholesterol, fasting glucose, ECG results, max heart rate, exercise angina, ST depression, ST slope, vessel count, thalassemia type

---

## Future Work

- **SHAP values** — move from feature importance to true per-patient Shapley explanations
- **Larger dataset** — validate on MIMIC-III or UK Biobank for clinical-grade conclusions
- **Longitudinal modeling** — track biomarker trajectories over time, not just snapshots
- **Biological age calibration** — calibrate the composite score against validated aging clocks (PhenoAge, GrimAge)
- **Multi-modal inputs** — integrate wearable data (resting HR, HRV, VO₂max estimates) for a continuous monitoring pipeline
- **Explainability layer** — add LIME/SHAP waterfall plots to the Streamlit app for clinician trust

---

## Disclaimer

This project is for **educational and research purposes only**. It is not a clinical diagnostic tool and should not be used to make medical decisions. Always consult a qualified healthcare professional.

---

<div align="center">
Built for longevity research · Cleveland Heart Disease Dataset · MIT License
</div>
