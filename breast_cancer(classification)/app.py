import streamlit as st
import pickle
import numpy as np
from sklearn.datasets import load_breast_cancer

# ── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Breast Cancer Detection",
    page_icon="🩺",
    layout="wide"
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .main { background-color: #f8f4f0; }

    h1, h2, h3 {
        font-family: 'DM Serif Display', serif;
    }

    .title-box {
        background: linear-gradient(135deg, #c0392b, #8e1a0e);
        color: white;
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
    }

    .title-box h1 { color: white; margin: 0; font-size: 2.2rem; }
    .title-box p  { color: #f5c6c6; margin: 0.4rem 0 0; font-size: 1rem; }

    .result-benign {
        background: linear-gradient(135deg, #1a7a4a, #27ae60);
        color: white;
        padding: 1.8rem;
        border-radius: 14px;
        text-align: center;
        font-size: 1.6rem;
        font-family: 'DM Serif Display', serif;
    }

    .result-malignant {
        background: linear-gradient(135deg, #c0392b, #e74c3c);
        color: white;
        padding: 1.8rem;
        border-radius: 14px;
        text-align: center;
        font-size: 1.6rem;
        font-family: 'DM Serif Display', serif;
    }

    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.07);
    }

    .metric-card h2 { font-size: 2rem; margin: 0; color: #c0392b; }
    .metric-card p  { margin: 0; color: #666; font-size: 0.85rem; }

    .info-box {
        background: #fff8f8;
        border-left: 4px solid #c0392b;
        padding: 1rem 1.2rem;
        border-radius: 0 10px 10px 0;
        margin-bottom: 1rem;
        font-size: 0.92rem;
        color: #444;
    }

    .stSlider > div > div { accent-color: #c0392b; }

    .stButton > button {
        background: linear-gradient(135deg, #c0392b, #8e1a0e);
        color: white;
        border: none;
        padding: 0.7rem 2.5rem;
        border-radius: 8px;
        font-size: 1rem;
        font-weight: 600;
        width: 100%;
        cursor: pointer;
    }

    .section-header {
        font-family: 'DM Serif Display', serif;
        font-size: 1.2rem;
        color: #8e1a0e;
        border-bottom: 2px solid #f0d0d0;
        padding-bottom: 0.4rem;
        margin: 1.5rem 0 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Load Model & Scaler ───────────────────────────────────────
@st.cache_resource
def load_model():
    with open('breast_cancer_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('breast_cancer_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

try:
    model, scaler = load_model()
    model_loaded  = True
except:
    model_loaded  = False

# ── Load Feature Names ────────────────────────────────────────
cancer   = load_breast_cancer()
features = cancer.feature_names

# ── Header ───────────────────────────────────────────────────
st.markdown("""
<div class="title-box">
    <h1>🩺 Breast Cancer Detection</h1>
    <p>Enter tumor measurements below to classify as Malignant or Benign using SVM model</p>
</div>
""", unsafe_allow_html=True)

if not model_loaded:
    st.error("❌ Model files not found! Make sure `breast_cancer_model.pkl` and `breast_cancer_scaler.pkl` are in the same folder as app.py")
    st.stop()

# ── Dataset Stats ─────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown('<div class="metric-card"><h2>569</h2><p>Total Samples</p></div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="metric-card"><h2>30</h2><p>Features</p></div>', unsafe_allow_html=True)
with col3:
    st.markdown('<div class="metric-card"><h2>357</h2><p>Benign Cases</p></div>', unsafe_allow_html=True)
with col4:
    st.markdown('<div class="metric-card"><h2>212</h2><p>Malignant Cases</p></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Input Mode ────────────────────────────────────────────────
mode = st.radio(
    "Choose Input Method:",
    ["🎚️ Use Sliders", "📋 Use Sample Data"],
    horizontal=True
)

# ── Feature Min/Max from dataset ──────────────────────────────
import pandas as pd
df = pd.DataFrame(cancer.data, columns=cancer.feature_names)

# ── Sidebar Info ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ℹ️ About This App")
    st.markdown("""
    <div class="info-box">
    This app uses a <b>Support Vector Machine (SVM)</b> model trained on the 
    built-in scikit-learn Breast Cancer Wisconsin dataset.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🎯 Model Performance")
    st.markdown("""
    <div class="info-box">
    <b>Accuracy :</b> ~98.2%<br>
    <b>Recall   :</b> ~99.8%<br>
    <b>F1 Score :</b> ~98.5%<br>
    <b>Algorithm:</b> SVM (RBF Kernel)
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### ⚠️ Disclaimer")
    st.markdown("""
    <div class="info-box">
    This tool is for <b>educational purposes only</b>. 
    Always consult a qualified medical professional for diagnosis.
    </div>
    """, unsafe_allow_html=True)

# ── Input Section ─────────────────────────────────────────────
input_values = []

if mode == "📋 Use Sample Data":
    sample_type = st.selectbox(
        "Select a sample patient:",
        ["Malignant Sample (Patient 1)", "Benign Sample (Patient 2)"]
    )
    if "Malignant" in sample_type:
        # First sample in dataset is malignant
        input_values = list(cancer.data[0])
        st.success("✅ Loaded Malignant sample — Actual label: Malignant")
    else:
        # Find first benign sample
        benign_idx   = list(cancer.target).index(1)
        input_values = list(cancer.data[benign_idx])
        st.success("✅ Loaded Benign sample — Actual label: Benign")

    # Show sample values
    sample_df = pd.DataFrame([input_values], columns=features)
    st.dataframe(sample_df.T.rename(columns={0: 'Value'}), height=300)

else:
    st.markdown('<div class="section-header">📐 Mean Features</div>', unsafe_allow_html=True)
    cols = st.columns(3)
    mean_features = [f for f in features if 'mean' in f]
    for i, feat in enumerate(mean_features):
        with cols[i % 3]:
            val = st.slider(
                feat,
                float(df[feat].min()),
                float(df[feat].max()),
                float(df[feat].mean()),
                key=f"slider_{feat}"
            )
            input_values.append(val)

    st.markdown('<div class="section-header">📏 Standard Error Features</div>', unsafe_allow_html=True)
    cols2 = st.columns(3)
    se_features = [f for f in features if 'error' in f]
    for i, feat in enumerate(se_features):
        with cols2[i % 3]:
            val = st.slider(
                feat,
                float(df[feat].min()),
                float(df[feat].max()),
                float(df[feat].mean()),
                key=f"slider_{feat}"
            )
            input_values.append(val)

    st.markdown('<div class="section-header">📊 Worst Features</div>', unsafe_allow_html=True)
    cols3 = st.columns(3)
    worst_features = [f for f in features if 'worst' in f]
    for i, feat in enumerate(worst_features):
        with cols3[i % 3]:
            val = st.slider(
                feat,
                float(df[feat].min()),
                float(df[feat].max()),
                float(df[feat].mean()),
                key=f"slider_{feat}"
            )
            input_values.append(val)

# ── Predict Button ────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
predict_btn = st.button("🔬 Predict Tumor Type")

if predict_btn:
    input_array  = np.array(input_values).reshape(1, -1)
    input_scaled = scaler.transform(input_array)

    prediction   = model.predict(input_scaled)[0]
    probability  = model.predict_proba(input_scaled)[0]

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("## 📊 Prediction Result")

    col_res1, col_res2 = st.columns([1, 1])

    with col_res1:
        if prediction == 1:
            st.markdown("""
            <div class="result-benign">
                ✅ BENIGN<br>
                <span style="font-size:1rem; font-family: DM Sans;">Non-Cancerous Tumor</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="result-malignant">
                ❌ MALIGNANT<br>
                <span style="font-size:1rem; font-family: DM Sans;">Cancerous Tumor Detected</span>
            </div>
            """, unsafe_allow_html=True)

    with col_res2:
        st.markdown("**Prediction Confidence:**")
        st.metric("Benign Probability",    f"{round(probability[1]*100, 2)}%")
        st.metric("Malignant Probability", f"{round(probability[0]*100, 2)}%")
        st.progress(float(probability[1] if prediction == 1 else probability[0]))

    # ── Confidence Bar ────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    col_b, col_m = st.columns(2)
    with col_b:
        st.markdown("🟢 **Benign Confidence**")
        st.progress(float(probability[1]))
        st.caption(f"{round(probability[1]*100, 2)}%")
    with col_m:
        st.markdown("🔴 **Malignant Confidence**")
        st.progress(float(probability[0]))
        st.caption(f"{round(probability[0]*100, 2)}%")

    if prediction == 0:
        st.warning("⚠️ Malignant tumor detected. Please consult a medical professional immediately.")
    else:
        st.success("✅ Tumor appears Benign. Regular checkups are still recommended.")