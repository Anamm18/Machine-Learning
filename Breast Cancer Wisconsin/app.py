import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_curve, auc
)

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Breast Cancer PCA Classifier",
    page_icon="🎗️",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    [data-testid="stAppViewContainer"] { background: #0f1117; }
    [data-testid="stSidebar"]          { background: #161b27; }
    h1, h2, h3                         { color: #f8a4c8; }
    .metric-card {
        background: #1c2333;
        border: 1px solid #2d3550;
        border-radius: 10px;
        padding: 1rem 1.4rem;
        text-align: center;
    }
    .metric-card .label { color: #8b9db5; font-size: 0.82rem; text-transform: uppercase; letter-spacing: .08em; }
    .metric-card .value { color: #f8a4c8; font-size: 2rem; font-weight: 700; margin-top: .2rem; }
</style>
""", unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    bc = load_breast_cancer()
    df = pd.DataFrame(bc.data, columns=bc.feature_names)
    df["target"] = bc.target
    df["diagnosis"] = df["target"].map({0: "Malignant", 1: "Benign"})
    return df, bc

def metric_card(label, value):
    st.markdown(
        f'<div class="metric-card"><div class="label">{label}</div>'
        f'<div class="value">{value}</div></div>',
        unsafe_allow_html=True,
    )

# ── Load ──────────────────────────────────────────────────────────────────────
df, bc = load_data()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    test_size   = st.slider("Test Set Size", 0.10, 0.40, 0.20, 0.05)
    n_components = st.slider("PCA Components", 2, 15, 2)
    random_state = st.number_input("Random Seed", 0, 999, 42, 1)
    st.markdown("---")
    st.markdown("**Dataset:** Breast Cancer Wisconsin")
    st.markdown(f"**Samples:** {len(df)}")
    st.markdown(f"**Features:** {len(bc.feature_names)}")

# ── Title ─────────────────────────────────────────────────────────────────────
st.markdown("# 🎗️ Breast Cancer Classification using PCA")
st.markdown(
    "Dimensionality reduction with **Principal Component Analysis** "
    "and classification via **Logistic Regression**."
)
st.markdown("---")

# ── Dataset Overview ──────────────────────────────────────────────────────────
st.markdown("## 📋 Dataset Overview")
col1, col2, col3, col4 = st.columns(4)
with col1: metric_card("Total Samples", len(df))
with col2: metric_card("Features", len(bc.feature_names))
with col3: metric_card("Benign", int((df.target == 1).sum()))
with col4: metric_card("Malignant", int((df.target == 0).sum()))

st.markdown("#### Sample Data")
st.dataframe(df.drop(columns=["target"]).head(5), use_container_width=True)

col_a, col_b = st.columns(2)

with col_a:
    st.markdown("#### Class Distribution")
    fig, ax = plt.subplots(figsize=(5, 3.5))
    fig.patch.set_facecolor("#1c2333")
    ax.set_facecolor("#1c2333")
    counts = df["diagnosis"].value_counts()
    colors = ["#f8a4c8", "#7eb8f7"]
    bars = ax.bar(counts.index, counts.values, color=colors, width=0.5, edgecolor="none")
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                str(val), ha="center", va="bottom", color="white", fontsize=11)
    ax.set_ylabel("Count", color="#8b9db5")
    ax.tick_params(colors="#8b9db5")
    for spine in ax.spines.values():
        spine.set_visible(False)
    st.pyplot(fig)
    plt.close()

with col_b:
    st.markdown("#### Feature Correlation (Top 10)")
    fig, ax = plt.subplots(figsize=(5, 3.5))
    fig.patch.set_facecolor("#1c2333")
    corr = df[bc.feature_names[:10]].corr()
    sns.heatmap(
        corr, ax=ax, cmap="RdPu", annot=False,
        linewidths=0.3, linecolor="#0f1117",
        cbar_kws={"shrink": 0.7}
    )
    ax.set_facecolor("#1c2333")
    ax.tick_params(colors="#8b9db5", labelsize=6)
    st.pyplot(fig)
    plt.close()

st.markdown("---")

# ── Pipeline ──────────────────────────────────────────────────────────────────
X = df[bc.feature_names].values
y = df["target"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_state, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

pca = PCA(n_components=n_components, random_state=random_state)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca  = pca.transform(X_test_scaled)

model = LogisticRegression(max_iter=1000, random_state=random_state)
model.fit(X_train_pca, y_train)
y_pred = model.predict(X_test_pca)
y_prob = model.predict_proba(X_test_pca)[:, 1]

# ── PCA Analysis ──────────────────────────────────────────────────────────────
st.markdown("## 🔻 PCA Analysis")
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Explained Variance (Scree Plot)")
    pca_full = PCA(random_state=random_state).fit(X_train_scaled)
    cumvar = np.cumsum(pca_full.explained_variance_ratio_) * 100

    fig, ax = plt.subplots(figsize=(5, 3.5))
    fig.patch.set_facecolor("#1c2333")
    ax.set_facecolor("#1c2333")
    ax.bar(range(1, 16), pca_full.explained_variance_ratio_[:15] * 100,
           color="#7eb8f7", alpha=0.8, edgecolor="none")
    ax2 = ax.twinx()
    ax2.plot(range(1, 16), cumvar[:15], "o-", color="#f8a4c8", lw=2, ms=5)
    ax2.axhline(95, color="#f8a4c8", linestyle="--", alpha=0.4, lw=1)
    ax2.set_ylabel("Cumulative Variance %", color="#f8a4c8", fontsize=9)
    ax2.tick_params(colors="#f8a4c8")
    ax.set_xlabel("Principal Component", color="#8b9db5")
    ax.set_ylabel("Explained Variance %", color="#7eb8f7")
    ax.tick_params(colors="#8b9db5")
    for spine in ax.spines.values(): spine.set_visible(False)
    for spine in ax2.spines.values(): spine.set_visible(False)
    st.pyplot(fig)
    plt.close()

with col2:
    st.markdown("#### PCA Scatter (PC1 vs PC2)")
    pca2 = PCA(n_components=2, random_state=random_state)
    X_2d = pca2.fit_transform(X_train_scaled)

    fig, ax = plt.subplots(figsize=(5, 3.5))
    fig.patch.set_facecolor("#1c2333")
    ax.set_facecolor("#1c2333")
    palette = {0: "#f8a4c8", 1: "#7eb8f7"}
    labels  = {0: "Malignant", 1: "Benign"}
    for cls in [0, 1]:
        mask = y_train == cls
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                   c=palette[cls], label=labels[cls], s=22, alpha=0.75, edgecolors="none")
    ax.set_xlabel("PC 1", color="#8b9db5")
    ax.set_ylabel("PC 2", color="#8b9db5")
    ax.tick_params(colors="#8b9db5")
    legend = ax.legend(facecolor="#1c2333", edgecolor="#2d3550", labelcolor="white", fontsize=9)
    for spine in ax.spines.values(): spine.set_visible(False)
    st.pyplot(fig)
    plt.close()

ev = pca.explained_variance_ratio_.sum() * 100
st.info(f"**{n_components} component(s)** retain **{ev:.1f}%** of the total variance.")

st.markdown("---")

# ── Model Performance ─────────────────────────────────────────────────────────
st.markdown("## 🤖 Model Performance")
acc  = accuracy_score(y_test, y_pred)
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
report = classification_report(y_test, y_pred, target_names=["Malignant", "Benign"], output_dict=True)

m1, m2, m3, m4 = st.columns(4)
with m1: metric_card("Accuracy",  f"{acc*100:.1f}%")
with m2: metric_card("ROC-AUC",   f"{roc_auc:.3f}")
with m3: metric_card("Precision", f"{report['weighted avg']['precision']:.3f}")
with m4: metric_card("Recall",    f"{report['weighted avg']['recall']:.3f}")

st.markdown("#### Classification Report")
report_df = pd.DataFrame(report).T.round(3)
st.dataframe(report_df.style.background_gradient(cmap="RdPu", axis=None), use_container_width=True)

st.markdown("---")

# ── Confusion Matrix & ROC ────────────────────────────────────────────────────
st.markdown("## 📊 Evaluation Plots")
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    fig.patch.set_facecolor("#1c2333")
    ax.set_facecolor("#1c2333")
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="RdPu",
        xticklabels=["Malignant", "Benign"],
        yticklabels=["Malignant", "Benign"],
        ax=ax, linewidths=1, linecolor="#0f1117",
        annot_kws={"size": 14, "weight": "bold", "color": "white"},
        cbar=False
    )
    ax.set_xlabel("Predicted", color="#8b9db5", fontsize=11)
    ax.set_ylabel("Actual",    color="#8b9db5", fontsize=11)
    ax.tick_params(colors="#8b9db5")
    st.pyplot(fig)
    plt.close()

with col2:
    st.markdown("#### ROC Curve")
    fig, ax = plt.subplots(figsize=(5, 4))
    fig.patch.set_facecolor("#1c2333")
    ax.set_facecolor("#1c2333")
    ax.plot(fpr, tpr, color="#f8a4c8", lw=2.5, label=f"AUC = {roc_auc:.3f}")
    ax.fill_between(fpr, tpr, alpha=0.12, color="#f8a4c8")
    ax.plot([0, 1], [0, 1], "--", color="#8b9db5", lw=1)
    ax.set_xlabel("False Positive Rate", color="#8b9db5")
    ax.set_ylabel("True Positive Rate",  color="#8b9db5")
    ax.tick_params(colors="#8b9db5")
    legend = ax.legend(facecolor="#1c2333", edgecolor="#2d3550", labelcolor="white")
    for spine in ax.spines.values(): spine.set_visible(False)
    st.pyplot(fig)
    plt.close()

st.markdown("---")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(
    "<p style='text-align:center;color:#8b9db5;font-size:0.85rem;'>"
    "Built by <strong style='color:#f8a4c8'>Anam Mulla</strong> · "
    "Breast Cancer PCA Classifier · Scikit-learn + Streamlit"
    "</p>",
    unsafe_allow_html=True,
)