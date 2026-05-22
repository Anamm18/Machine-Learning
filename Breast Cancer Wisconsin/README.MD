# 🎗️ Breast Cancer Classification using PCA

> Dimensionality reduction meets clinical intelligence — classifying breast tumors with Principal Component Analysis and Logistic Regression.

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.x-orange?style=flat-square&logo=scikit-learn)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?style=flat-square&logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## 📌 Overview

This project applies **Principal Component Analysis (PCA)** to the Breast Cancer Wisconsin Diagnostic Dataset to reduce its 30 features into compact principal components — then trains a **Logistic Regression** classifier on the transformed data.

The result is a lightweight, interpretable model wrapped in a clean **Streamlit web app** that lets you explore PCA variance, visualize decision boundaries, inspect a confusion matrix, and view the ROC curve — all interactively.

---

## ✨ Features

| Feature | Description |
|---|---|
| 📊 EDA | Distribution plots, correlation heatmap, class balance |
| ⚖️ Feature Scaling | StandardScaler normalization before PCA |
| 🔻 PCA Reduction | Configurable number of components with explained variance chart |
| 🤖 Classification | Logistic Regression on PCA-transformed features |
| 📉 Confusion Matrix | Interactive heatmap of predictions vs. actuals |
| 📈 ROC Curve | AUC score with plotted curve |
| 🌐 Streamlit App | Full interactive dashboard — no code needed |

---

## 🗂️ Project Structure

```
breast-cancer-pca/
│
├── app.py               # Streamlit web application
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
```

---

## 🧰 Technologies Used

- **Python 3.8+**
- **Pandas** — data manipulation
- **NumPy** — numerical operations
- **Matplotlib / Seaborn** — visualizations
- **Scikit-learn** — PCA, Logistic Regression, metrics
- **Streamlit** — interactive web app

---

## 📦 Dataset

**Breast Cancer Wisconsin Diagnostic Dataset**
- 569 samples (357 benign, 212 malignant)
- 30 numeric features computed from digitized cell nucleus images
- Binary target: `Malignant (0)` / `Benign (1)`

Loaded directly via scikit-learn — no download required:

```python
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
```

---

## ⚙️ Installation

**1. Clone the repository**

```bash
git clone https://github.com/your-username/breast-cancer-pca.git
cd breast-cancer-pca
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

**`requirements.txt`**
```
pandas
numpy
matplotlib
seaborn
scikit-learn
streamlit
```

---

## 🚀 Running the App

```bash
streamlit run app.py
```

Then open your browser at `http://localhost:8501`

---

## 🔄 Project Workflow

```
Load Dataset → EDA → Train/Test Split → StandardScaler
     → PCA → Logistic Regression → Evaluate → Visualize
```

1. **Load Dataset** — from `sklearn.datasets`
2. **Exploratory Data Analysis** — class distribution, feature correlations
3. **Preprocessing** — StandardScaler on features
4. **PCA** — reduce to N components (configurable in sidebar)
5. **Model Training** — Logistic Regression on PCA output
6. **Evaluation** — accuracy, classification report, confusion matrix, ROC-AUC
7. **Visualization** — all plots rendered in the Streamlit dashboard

---

## 📊 Model Performance

- ✅ **~97% Accuracy** on test set
- ✅ **ROC-AUC ~0.99**
- ✅ **2–10 components** retain 80–95% of explained variance
- ✅ Significant dimensionality reduction (30 → 2+ features)

---

## 🖥️ App Sections

| Section | What You'll See |
|---|---|
| **Sidebar** | Configure test size, PCA components, random seed |
| **Dataset Overview** | Shape, class counts, feature stats |
| **Explained Variance** | Scree plot — how much variance each component captures |
| **PCA Scatter Plot** | 2D projection colored by class label |
| **Model Metrics** | Accuracy, precision, recall, F1 |
| **Confusion Matrix** | Heatmap of TP/TN/FP/FN |
| **ROC Curve** | Sensitivity vs. specificity tradeoff |

---

## 📝 Conclusion

PCA effectively compresses the 30-feature dataset while preserving classification-relevant structure. With just 2 principal components, the two classes become visually separable. Logistic Regression trained on PCA output achieves strong accuracy with a simpler, faster model — demonstrating that dimensionality reduction is a powerful preprocessing step for medical data.

---

## 👩‍💻 Author

**Anam Mulla**

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).
