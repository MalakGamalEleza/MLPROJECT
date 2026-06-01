import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ──────────────────────────────────────────────
# 1. PAGE CONFIGURATION
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="METABRIC — Breast Cancer Survival",
    page_icon="🎗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ──────────────────────────────────────────────
# 2. CUSTOM CSS — ELEGANT CLINICAL AESTHETIC
# ──────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ── Root Variables ── */
:root {
    --rose:     #C0395A;
    --rose-lt:  #E8637F;
    --rose-pale:#FDF0F3;
    --ink:      #1A1A2E;
    --slate:    #4A4E69;
    --mist:     #F7F5F8;
    --border:   #E8E0EC;
    --success:  #2D7D5A;
    --success-bg:#E8F5EE;
    --danger-bg:#FDF0F3;
}

/* ── Global ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    color: var(--ink);
}
.stApp { background: var(--mist); }

/* ── Hide default streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 3rem 4rem 3rem !important; max-width: 1200px; }

/* ── Hero Banner ── */
.hero {
    background: linear-gradient(135deg, #1A1A2E 0%, #2D1B3D 50%, #C0395A 100%);
    border-radius: 20px;
    padding: 3rem 3.5rem;
    margin-bottom: 2.5rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '🎗️';
    position: absolute;
    right: 3rem; top: 50%;
    transform: translateY(-50%);
    font-size: 8rem;
    opacity: 0.08;
    pointer-events: none;
}
.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2.6rem;
    color: #fff;
    margin: 0 0 0.4rem 0;
    line-height: 1.15;
    letter-spacing: -0.5px;
}
.hero-sub {
    font-size: 1rem;
    color: rgba(255,255,255,0.65);
    margin: 0 0 1.5rem 0;
    font-weight: 300;
    letter-spacing: 0.3px;
}
.hero-badge {
    display: inline-block;
    background: rgba(255,255,255,0.12);
    border: 1px solid rgba(255,255,255,0.2);
    border-radius: 40px;
    padding: 0.35rem 1rem;
    font-size: 0.82rem;
    color: rgba(255,255,255,0.85);
    backdrop-filter: blur(6px);
}
.hero-developer {
    margin-top: 1.2rem;
    font-size: 0.88rem;
    color: rgba(255,255,255,0.5);
}
.hero-developer strong { color: rgba(255,255,255,0.85); }

/* ── Section Cards ── */
.section-card {
    background: #fff;
    border-radius: 16px;
    border: 1px solid var(--border);
    padding: 1.8rem 2rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 2px 12px rgba(26,26,46,0.05);
}
.section-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.15rem;
    color: var(--ink);
    margin: 0 0 0.2rem 0;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.section-desc {
    font-size: 0.82rem;
    color: var(--slate);
    margin: 0 0 1.2rem 0;
    font-weight: 300;
}
.divider-rose {
    height: 3px;
    background: linear-gradient(90deg, var(--rose), transparent);
    border-radius: 2px;
    margin: 0.6rem 0 1.4rem 0;
}

/* ── Stat Chips in Summary ── */
.stat-row {
    display: flex;
    gap: 1rem;
    flex-wrap: wrap;
    margin-bottom: 0.5rem;
}
.stat-chip {
    background: var(--mist);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0.7rem 1.2rem;
    flex: 1;
    min-width: 120px;
}
.stat-chip-label {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    color: var(--slate);
    font-weight: 500;
    margin-bottom: 0.2rem;
}
.stat-chip-value {
    font-size: 1.35rem;
    font-weight: 600;
    color: var(--ink);
}
.stat-chip-unit {
    font-size: 0.75rem;
    color: var(--slate);
    margin-left: 2px;
}

/* ── Result Box ── */
.result-living {
    background: var(--success-bg);
    border: 2px solid var(--success);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    text-align: center;
}
.result-deceased {
    background: var(--danger-bg);
    border: 2px solid var(--rose);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    text-align: center;
}
.result-label {
    font-family: 'DM Serif Display', serif;
    font-size: 2rem;
    margin: 0.5rem 0;
}
.result-living .result-label  { color: var(--success); }
.result-deceased .result-label { color: var(--rose); }
.result-icon { font-size: 2.8rem; }
.result-confidence {
    font-size: 0.9rem;
    color: var(--slate);
    margin-top: 0.5rem;
    font-weight: 300;
}
.confidence-bar-wrap {
    background: rgba(0,0,0,0.08);
    border-radius: 30px;
    height: 8px;
    margin: 1rem auto;
    max-width: 240px;
    overflow: hidden;
}
.confidence-bar-fill {
    height: 100%;
    border-radius: 30px;
    transition: width 0.6s ease;
}
.confidence-bar-fill.living  { background: var(--success); }
.confidence-bar-fill.deceased { background: var(--rose); }

/* ── Disclaimer ── */
.disclaimer {
    background: #FFF8E1;
    border-left: 4px solid #F59E0B;
    border-radius: 0 10px 10px 0;
    padding: 0.9rem 1.2rem;
    font-size: 0.8rem;
    color: #78530A;
    margin-top: 1rem;
    line-height: 1.5;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #fff !important;
    border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] .stSlider label,
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stNumberInput label {
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    color: var(--slate) !important;
    text-transform: uppercase;
    letter-spacing: 0.4px;
}
.sidebar-section-head {
    font-family: 'DM Serif Display', serif;
    font-size: 0.95rem;
    color: var(--ink);
    border-bottom: 2px solid var(--rose);
    padding-bottom: 0.3rem;
    margin: 1.2rem 0 0.8rem 0;
    display: block;
}

/* ── Streamlit slider accent ── */
.stSlider [data-baseweb="slider"] [role="slider"] {
    background-color: var(--rose) !important;
    border-color: var(--rose) !important;
}
.stSlider [data-baseweb="slider"] [data-testid="stTickBar"] {
    color: var(--rose) !important;
}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# 3. HERO BANNER
# ──────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <p class="hero-title">Breast Cancer Survival<br>Prediction Dashboard</p>
    <p class="hero-sub">METABRIC Dataset · Random Forest Classifier · Clinical Decision Support</p>
    <span class="hero-badge">🔬 AI-Assisted Oncology Tool</span>
    <p class="hero-developer">
        Developer: <strong>Malak Gamal Ahmed Sanad Eleza</strong> &nbsp;·&nbsp; ID: 221000014
    </p>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# 4. LOAD ARTIFACTS
# ──────────────────────────────────────────────
@st.cache_resource
def load_project_components():
    try:
        model    = joblib.load('best_model_rf.pkl')
        scaler   = joblib.load('scaler.pkl')
        features = joblib.load('selected_features.pkl')
        return model, scaler, features
    except Exception as e:
        st.error(f"❌ Error loading model artifacts: {e}")
        return None, None, None

model, scaler, selected_features = load_project_components()

# ──────────────────────────────────────────────
# 5. SIDEBAR — GROUPED INPUTS
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown('<span class="sidebar-section-head">🩺 Clinical & Tumor Data</span>', unsafe_allow_html=True)
    age          = st.slider("Age at Diagnosis (Years)", 20.0, 100.0, 60.0)
    tumor_size   = st.number_input("Tumor Size (mm)", 1.0, 200.0, 25.0)
    hist_grade   = st.slider("Histologic Grade (1–3)", 1.0, 3.0, 2.0, step=1.0)
    npi          = st.slider("Nottingham Prognostic Index", 1.0, 8.0, 4.0, step=0.01)
    lymph_nodes  = st.number_input("Lymph Nodes Positive", 0, 50, 2)
    mutation_count = st.number_input("Mutation Count", 1, 80, 5)
    cohort       = st.selectbox("Patient Cohort Group", [1.0, 2.0, 3.0, 4.0, 5.0, 9.0])

    st.markdown('<span class="sidebar-section-head">🧬 Receptor & Molecular Status</span>', unsafe_allow_html=True)
    er_status    = st.selectbox("ER Status", ["Positive", "Negative"])
    er_ihc       = st.selectbox("ER Status by IHC", ["Positve", "Negative"])
    her2_status  = st.selectbox("HER2 Expression Status", ["Positive", "Negative"])
    her2_snp6    = st.selectbox("HER2 by SNP6", ["Neutral", "Gain", "Loss", "Undef"])
    pr_status    = st.selectbox("PR Status", ["Positive", "Negative"])
    pam50        = st.selectbox("Pam50 + Claudin-low Subtype", ["LumA", "LumB", "Her2", "Basal", "Normal", "claudin-low", "NC"])
    int_cluster  = st.selectbox("Integrative Cluster", ["1","2","3","4ER-","4ER+","5","6","7","8","9","10"])

    st.markdown('<span class="sidebar-section-head">🏥 Treatment & Demographics</span>', unsafe_allow_html=True)
    surgery_type   = st.selectbox("Type of Breast Surgery", ["Mastectomy", "Breast Conserving"])
    chemotherapy   = st.selectbox("Chemotherapy", ["No", "Yes"])
    hormone_therapy = st.selectbox("Hormone Therapy", ["No", "Yes"])
    radio_therapy  = st.selectbox("Radio Therapy", ["No", "Yes"])
    menopausal     = st.selectbox("Menopausal State", ["Post", "Pre"])
    laterality     = st.selectbox("Tumor Laterality", ["Left", "Right"])

    st.markdown('<span class="sidebar-section-head">🔬 Pathology</span>', unsafe_allow_html=True)
    cellularity    = st.selectbox("Tumor Cellularity", ["High", "Moderate", "Low"])
    hist_subtype   = st.selectbox("Histologic Subtype", ["Ductal/NST","Lobular","Mixed","Medullary","Mucinous","Tubular","Other"])
    relapse_status = st.selectbox("Relapse Free Status", ["Not Recurred", "Recurred"])

# ──────────────────────────────────────────────
# 6. FEATURE ENGINEERING
# ──────────────────────────────────────────────
high_grade_lymph_risk = round(hist_grade * (lymph_nodes + 1), 2)

# ──────────────────────────────────────────────
# 7. ENCODING
# ──────────────────────────────────────────────
input_data = {
    'Age at Diagnosis': age,
    'Cohort': cohort,
    'Neoplasm Histologic Grade': hist_grade,
    'Lymph nodes examined positive': float(lymph_nodes),
    'Mutation Count': float(mutation_count),
    'Nottingham prognostic index': npi,
    'Tumor Size': tumor_size,
    'High_Grade_Lymph_Risk': high_grade_lymph_risk,
    'Type of Breast Surgery': 1 if surgery_type == "Mastectomy" else 0,
    'Cellularity': ["High", "Low", "Moderate"].index(cellularity),
    'Chemotherapy': 1 if chemotherapy == "Yes" else 0,
    'Pam50 + Claudin-low subtype': ["Basal","Her2","LumA","LumB","NC","Normal","claudin-low"].index(pam50),
    'ER status measured by IHC': 1 if er_ihc == "Positve" else 0,
    'ER Status': 1 if er_status == "Positive" else 0,
    'HER2 status measured by SNP6': ["Gain","Loss","Neutral","Undef"].index(her2_snp6),
    'HER2 Status': 1 if her2_status == "Positive" else 0,
    'Tumor Other Histologic Subtype': ["Ductal/NST","Lobular","Medullary","Mixed","Mucinous","Other","Tubular"].index(hist_subtype),
    'Hormone Therapy': 1 if hormone_therapy == "Yes" else 0,
    'Inferred Menopausal State': 1 if menopausal == "Post" else 0,
    'Integrative Cluster': ["1","10","2","3","4ER-","4ER+","5","6","7","8","9"].index(int_cluster),
    'Primary Tumor Laterality': 0 if laterality == "Left" else 1,
    'PR Status': 1 if pr_status == "Positive" else 0,
    'Radio Therapy': 1 if radio_therapy == "Yes" else 0,
    'Relapse Free Status': 1 if relapse_status == "Recurred" else 0
}

# ──────────────────────────────────────────────
# 8. MAIN DASHBOARD
# ──────────────────────────────────────────────
if model is not None and scaler is not None and selected_features is not None:
    input_df = pd.DataFrame([input_data])[selected_features]
    processed_inputs = scaler.transform(input_df)
    prediction = model.predict(processed_inputs)[0]
    prediction_proba = model.predict_proba(processed_inputs)[0]
    conf = prediction_proba[0] if prediction == 0 else prediction_proba[1]
    conf_pct = int(conf * 100)

    # ── Patient Summary ──
    st.markdown("""
    <div class="section-card">
        <p class="section-title">📋 Patient Parameter Summary</p>
        <div class="divider-rose"></div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(f"""
        <div class="stat-chip">
            <div class="stat-chip-label">Age</div>
            <div class="stat-chip-value">{int(age)}<span class="stat-chip-unit">yrs</span></div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="stat-chip">
            <div class="stat-chip-label">Tumor Size</div>
            <div class="stat-chip-value">{tumor_size:.0f}<span class="stat-chip-unit">mm</span></div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="stat-chip">
            <div class="stat-chip-label">Histol. Grade</div>
            <div class="stat-chip-value">{int(hist_grade)}<span class="stat-chip-unit">/3</span></div>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="stat-chip">
            <div class="stat-chip-label">NPI Score</div>
            <div class="stat-chip-value">{npi:.2f}</div>
        </div>""", unsafe_allow_html=True)
    with col5:
        st.markdown(f"""
        <div class="stat-chip">
            <div class="stat-chip-label">Lymph Risk</div>
            <div class="stat-chip-value">{high_grade_lymph_risk}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("""<br>""", unsafe_allow_html=True)

    col6, col7, col8 = st.columns(3)
    with col6:
        st.markdown(f"""
        <div class="stat-chip">
            <div class="stat-chip-label">ER / PR Status</div>
            <div class="stat-chip-value" style="font-size:1rem">{er_status} / {pr_status}</div>
        </div>""", unsafe_allow_html=True)
    with col7:
        st.markdown(f"""
        <div class="stat-chip">
            <div class="stat-chip-label">HER2 Status</div>
            <div class="stat-chip-value" style="font-size:1rem">{her2_status}</div>
        </div>""", unsafe_allow_html=True)
    with col8:
        st.markdown(f"""
        <div class="stat-chip">
            <div class="stat-chip-label">Subtype</div>
            <div class="stat-chip-value" style="font-size:1rem">{pam50}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # ── Prediction Result ──
    st.markdown("""
    <div class="section-card">
        <p class="section-title">🔮 ML Model Prediction</p>
        <div class="divider-rose"></div>
    """, unsafe_allow_html=True)

    col_pred, col_info = st.columns([1, 1])

    with col_pred:
        if prediction == 0:
            fill_class = "living"
            icon = "💚"
            label = "LIVING"
            box_class = "result-living"
            label_desc = "Model predicts patient survival"
        else:
            fill_class = "deceased"
            icon = "🎗️"
            label = "DECEASED"
            box_class = "result-deceased"
            label_desc = "Model predicts patient mortality"

        st.markdown(f"""
        <div class="{box_class}">
            <div class="result-icon">{icon}</div>
            <div class="result-label">{label}</div>
            <div class="confidence-bar-wrap">
                <div class="confidence-bar-fill {fill_class}" style="width:{conf_pct}%"></div>
            </div>
            <div class="result-confidence">{label_desc} with <strong>{conf:.1%} confidence</strong></div>
        </div>
        """, unsafe_allow_html=True)

    with col_info:
        living_pct  = prediction_proba[0] * 100
        deceased_pct = prediction_proba[1] * 100
        st.markdown(f"""
        <div style="padding: 1rem 0;">
            <p style="font-family:'DM Serif Display',serif; font-size:1rem; color:var(--ink); margin-bottom:1rem;">
                Class Probability Breakdown
            </p>
            <div style="margin-bottom:1rem;">
                <div style="display:flex; justify-content:space-between; font-size:0.82rem; color:var(--slate); margin-bottom:4px;">
                    <span>🟢 Living</span><span><strong>{living_pct:.1f}%</strong></span>
                </div>
                <div style="background:#e8e0ec; border-radius:20px; height:10px; overflow:hidden;">
                    <div style="width:{living_pct}%; background:#2D7D5A; height:100%; border-radius:20px;"></div>
                </div>
            </div>
            <div>
                <div style="display:flex; justify-content:space-between; font-size:0.82rem; color:var(--slate); margin-bottom:4px;">
                    <span>🔴 Deceased</span><span><strong>{deceased_pct:.1f}%</strong></span>
                </div>
                <div style="background:#e8e0ec; border-radius:20px; height:10px; overflow:hidden;">
                    <div style="width:{deceased_pct}%; background:#C0395A; height:100%; border-radius:20px;"></div>
                </div>
            </div>
            <div style="margin-top:1.5rem; padding:0.8rem 1rem; background:var(--mist); border-radius:10px; border:1px solid var(--border);">
                <div style="font-size:0.75rem; text-transform:uppercase; letter-spacing:0.6px; color:var(--slate); margin-bottom:4px;">Algorithm</div>
                <div style="font-size:0.9rem; font-weight:600; color:var(--ink);">Random Forest Classifier</div>
                <div style="font-size:0.75rem; color:var(--slate);">METABRIC Dataset · Trained Model</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="disclaimer">
        ⚠️ <strong>Clinical Disclaimer:</strong> This tool is intended for academic and research purposes only.
        Predictions generated by this model do not constitute medical advice and should not replace clinical judgment
        or consultation with qualified healthcare professionals.
    </div>
    </div>
    """, unsafe_allow_html=True)

else:
    # ── Graceful error state ──
    st.markdown("""
    <div class="section-card" style="text-align:center; padding:3rem;">
        <div style="font-size:3rem; margin-bottom:1rem;">⚠️</div>
        <p style="font-family:'DM Serif Display',serif; font-size:1.3rem; color:var(--ink);">Model Artifacts Not Found</p>
        <p style="color:var(--slate); font-size:0.9rem; max-width:400px; margin:0 auto;">
            Please ensure <code>best_model_rf.pkl</code>, <code>scaler.pkl</code>, and
            <code>selected_features.pkl</code> are in the same directory as <code>app.py</code>.
        </p>
    </div>
    """, unsafe_allow_html=True)