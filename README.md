# 🎗️ Breast Cancer Survival Prediction — METABRIC Dashboard

> A machine learning web application that predicts breast cancer patient survival outcomes using clinical, genomic, and treatment data from the METABRIC dataset.

**Developer:** Malak Gamal Ahmed Sanad Eleza  
**Student ID:** 221000014   
**Programme:** CBIO313: Data Mining and Machine Learning-2026SPRG

---

## 🔗 Live Application

👉 **[Open the App on Streamlit Cloud](https://mlproject-6wmtqecszecbvomrtjqduh.streamlit.app)**  
📹 **[Video Walkthrough](https://YOUR-VIDEO-LINK)**  
💻 **[GitHub Repository](https://github.com/MalakGamalEleza/MLPROJECT)**

---

## 📋 Project Overview

Breast cancer survival depends on a complex combination of clinical factors, genomic markers, and treatment history. This project addresses the question:

> *"Can we reliably predict whether a breast cancer patient will be alive or deceased based on their clinical and genomic profile?"*

The solution is an end-to-end data science pipeline — from raw, uncleaned data to a deployed interactive web application.

---

## 📂 Repository Structure

```
├── app1.py                    # Streamlit web application
├── requirements.txt          # Python dependencies
├── notebook.ipynb            # Full analysis notebook (EDA → Modelling)
├── dataset.csv               # METABRIC dataset (raw)
├── best_model_rf.pkl         # Trained Random Forest model
├── scaler.pkl                # Fitted StandardScaler
├── selected_features.pkl     # Selected feature names list
└── README.md
```

---

## 🗃️ Dataset — METABRIC

| Property | Detail |
|---|---|
| Source | [Kaggle — METABRIC Breast Cancer Dataset](https://www.kaggle.com/datasets/gunesevitan/breast-cancer-metabric) |
| Rows | ~2,509 patient records |
| Columns | 34 clinical and genomic features |
| Target | Overall Survival Status (Living / Deceased) |

**Why this dataset is not pre-cleaned:**
- Missing values across multiple columns
- Mixed and inconsistent categorical encodings
- Class imbalance in the target variable
- Non-standardised numeric scales across features

---

## 🔬 Project Steps

### Step 1 — Data Cleaning
- Identified and handled missing values using median imputation (numeric) and mode imputation (categorical)
- Removed duplicate records
- Standardised inconsistent categorical labels
- Documented all changes made to the raw dataset

### Step 2 — Exploratory Data Analysis (EDA)
- Investigated 6+ variables using both univariate and bivariate analysis
- Explored distributions of age, tumour size, NPI, mutation count, and receptor statuses
- Analysed survival rate across histologic grade, subtype, and treatment groups
- Used 5+ plot types: histograms, box plots, bar charts, heatmaps, violin plots, and pair plots

### Step 3 — Feature Engineering
One new composite feature was created:

```python
High_Grade_Lymph_Risk = Histologic Grade × (Lymph Nodes Positive + 1)
```

This captures the combined effect of tumour aggressiveness and lymph node spread — a clinically meaningful risk indicator.

### Step 4 — Feature Selection
Used **Random Forest Feature Importance** (Embedded Method) to rank and select the most predictive features. The final feature set is saved as `selected_features.pkl`.

### Step 5 — Modelling

Three algorithms were trained and compared:

| Algorithm | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| Logistic Regression | ~74% | 0.80 | 0.82 | 0.83 |
| Decision Tree | ~76% | 0.75 | 0.73 | 0.74 |
| **Random Forest ✅** | **~82%** | **0.81** | **0.82** | **0.85** |

**Random Forest was selected** as the best model due to superior performance, ensemble robustness, and built-in feature importance.

### Step 6 — Hyperparameter Tuning
`GridSearchCV` with 5-fold cross-validation was used to tune:
- `n_estimators` (100, 200, 300)
- `max_depth` (None, 10, 20)
- `min_samples_split`, `min_samples_leaf`

**Why tuning matters:** Default hyperparameters rarely produce optimal models. GridSearchCV systematically finds the best configuration that generalises to unseen data, reducing overfitting.

### Step 7 — Validation & Evaluation
- **Validation:** 80/20 train/test split with stratified 5-fold cross-validation
- **Why validation matters:** Ensures the model generalises to new patients, not just memorises training data
- **Metrics used:** Accuracy, Precision, Recall, F1-Score
- Final model: **Precision = 0.81 | Recall = 0.79** (both above the 0.3 minimum threshold ✅)


---

## 🚀 Run Locally

```bash
# 1. Clone the repository
git clone (https://github.com/MalakGamalEleza/MLPROJECT)
cd MLPROJECT

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the app
streamlit run app1.py
```

---

## 📦 Dependencies

```
streamlit
pandas
numpy
scikit-learn
joblib
```

---

## 🖥️ Application Features

- **24 input parameters** grouped into 4 clinical sections (Clinical, Molecular, Treatment, Pathology)
- **Patient summary panel** with key metric chips (Age, Tumor Size, Grade, NPI, Lymph Risk)
- **Colour-coded prediction result** — green for Living, rose for Deceased
- **Confidence score** and dual class probability bars
- **Clinical disclaimer** — tool is for academic/research use only

---

## ⚠️ Disclaimer

This application is built for **academic and research purposes only**. Predictions generated by this model do not constitute medical advice and should not replace clinical judgment or consultation with a qualified healthcare professional.

---

*Final Project — Data Science Programme · 2024/2025*
