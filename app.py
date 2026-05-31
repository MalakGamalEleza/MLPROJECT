import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1. Page Configuration
st.set_page_config(
    page_title="Breast Cancer METABRIC Dashboard",
    page_icon="🎗️",
    layout="wide"
)

st.title("🎗️ Breast Cancer METABRIC — Survival Prediction")
st.markdown("### Developer Profile:\n* **Name:** Malak Gamal Ahmed Sanad Eleza\n* **ID:** 221000014\n---")

# 2. Load the Saved Artifacts
@st.cache_resource
def load_project_components():
    try:
        model = joblib.load('best_model_rf.pkl')
        scaler = joblib.load('scaler.pkl')
        features = joblib.load('selected_features.pkl')
        return model, scaler, features
    except Exception as e:
        st.error(f"❌ Error Loading Artifacts: {e}")
        return None, None, None

model, scaler, selected_features = load_project_components()

# 3. Sidebar UI Controls
st.sidebar.header("📋 Patient Diagnostic Parameters")
age = st.sidebar.slider("Age at Diagnosis (Years)", 20.0, 100.0, 60.0)
cohort = st.sidebar.selectbox("Patient Cohort Group ID", [1.0, 2.0, 3.0, 4.0, 5.0, 9.0])
hist_grade = st.sidebar.slider("Neoplasm Histologic Grade (1-3)", 1.0, 3.0, 2.0, step=1.0)
lymph_nodes = st.sidebar.number_input("Lymph Nodes Examined Positive Count", 0, 50, 2)
mutation_count = st.sidebar.number_input("Genetic Mutation Count", 1, 80, 5)
npi = st.sidebar.slider("Nottingham Prognostic Index (NPI)", 1.0, 8.0, 4.0, step=0.01)
tumor_size = st.sidebar.number_input("Tumor Pathological Size (mm)", 1.0, 200.0, 25.0)

# Categorical Inputs
surgery_type = st.sidebar.selectbox("Type of Breast Surgery", ["Mastectomy", "Breast Conserving"])
cellularity = st.sidebar.selectbox("Tumor Cellularity Density Level", ["High", "Moderate", "Low"])
chemotherapy = st.sidebar.selectbox("Chemotherapy Status", ["No", "Yes"])
pam50 = st.sidebar.selectbox("Pam50 + Claudin-low Subtype", ["LumA", "LumB", "Her2", "Basal", "Normal", "claudin-low", "NC"])
er_ihc = st.sidebar.selectbox("ER Status Measured by IHC", ["Positve", "Negative"])
er_status = st.sidebar.selectbox("Estrogen Receptor (ER) Status", ["Positive", "Negative"])
her2_snp6 = st.sidebar.selectbox("HER2 Status Measured by SNP6", ["Neutral", "Gain", "Loss", "Undef"])
her2_status = st.sidebar.selectbox("HER2 Expression Status", ["Positive", "Negative"])
hist_subtype = st.sidebar.selectbox("Tumor Histologic Subtype", ["Ductal/NST", "Lobular", "Mixed", "Medullary", "Mucinous", "Tubular", "Other"])
hormone_therapy = st.sidebar.selectbox("Hormone Therapy Status", ["No", "Yes"])
menopausal = st.sidebar.selectbox("Inferred Menopausal State", ["Post", "Pre"])
int_cluster = st.sidebar.selectbox("Integrative Genomic Cluster Assignment", ["1", "2", "3", "4ER-", "4ER+", "5", "6", "7", "8", "9", "10"])
laterality = st.sidebar.selectbox("Primary Tumor Laterality Location", ["Left", "Right"])
pr_status = st.sidebar.selectbox("Progesterone Receptor (PR) Status", ["Positive", "Negative"])
radio_therapy = st.sidebar.selectbox("Radio Therapy Status", ["No", "Yes"])
relapse_status = st.sidebar.selectbox("Relapse Free Survival Status", ["Not Recurred", "Recurred"])

# 4. Feature Engineering
high_grade_lymph_risk = round(hist_grade * (lymph_nodes + 1), 2)

# 5. Mapping Encodings
input_data = {
    'Age at Diagnosis': age, 'Cohort': cohort, 'Neoplasm Histologic Grade': hist_grade,
    'Lymph nodes examined positive': float(lymph_nodes), 'Mutation Count': float(mutation_count),
    'Nottingham prognostic index': npi, 'Tumor Size': tumor_size, 'High_Grade_Lymph_Risk': high_grade_lymph_risk,
    'Type of Breast Surgery': 1 if surgery_type == "Mastectomy" else 0,
    'Cellularity': ["High", "Low", "Moderate"].index(cellularity),
    'Chemotherapy': 1 if chemotherapy == "Yes" else 0,
    'Pam50 + Claudin-low subtype': ["Basal", "Her2", "LumA", "LumB", "NC", "Normal", "claudin-low"].index(pam50),
    'ER status measured by IHC': 1 if er_ihc == "Positve" else 0, 'ER Status': 1 if er_status == "Positive" else 0,
    'HER2 status measured by SNP6': ["Gain", "Loss", "Neutral", "Undef"].index(her2_snp6), 'HER2 Status': 1 if her2_status == "Positive" else 0,
    'Tumor Other Histologic Subtype': ["Ductal/NST", "Lobular", "Medullary", "Mixed", "Mucinous", "Other", "Tubular"].index(hist_subtype),
    'Hormone Therapy': 1 if hormone_therapy == "Yes" else 0, 'Inferred Menopausal State': 1 if menopausal == "Post" else 0,
    'Integrative Cluster': ["1", "10", "2", "3", "4ER-", "4ER+", "5", "6", "7", "8", "9"].index(int_cluster),
    'Primary Tumor Laterality': 0 if laterality == "Left" else 1, 'PR Status': 1 if pr_status == "Positive" else 0,
    'Radio Therapy': 1 if radio_therapy == "Yes" else 0, 'Relapse Free Status': 1 if relapse_status == "Recurred" else 0
}

if model is not None and scaler is not None and selected_features is not None:
    input_df = pd.DataFrame([input_data])[selected_features]
    
    st.subheader("📊 Patient Target Profile Preview")
    st.write(f"**Age:** {age} | **Tumor Size:** {tumor_size} mm | **Composite Lymph Risk:** {high_grade_lymph_risk}")
    
    # Scale and Predict
    processed_inputs = scaler.transform(input_df)
    prediction = model.predict(processed_inputs)[0]
    prediction_proba = model.predict_proba(processed_inputs)[0]
    
    st.markdown("---")
    st.subheader("🔮 ML Model Diagnostic Outcome")
    if prediction == 0:
        st.success(f"### Predicted Survival: **LIVING** ({prediction_proba[0]:.2%} confidence)")
    else:
        st.error(f"### Predicted Survival: **DECEASED** ({prediction_proba[1]:.2%} confidence)")