# Breast Cancer Classification 🏥

A machine learning project to detect and classify breast cancer tumors as **Malignant (Cancerous)** or **Benign (Non-Cancerous)** using the Breast Cancer Wisconsin dataset.

---

## 📋 Project Overview

This project implements multiple machine learning classification algorithms to predict whether a breast cancer tumor is malignant or benign based on 30 numerical features extracted from digitized images of breast tissue.

**Key Objective:** Build a high-recall model that minimizes false negatives (missed cancer cases) while maintaining good accuracy.

---

## 📊 Dataset

- **Source:** Built-in Scikit-learn library (`sklearn.datasets.load_breast_cancer`)
- **Total Samples:** 569 patients
- **Features:** 30 numerical features
- **Target Classes:** 
  - 0 = Malignant (Cancer) - 212 cases
  - 1 = Benign (Non-Cancer) - 357 cases

**Feature Examples:**
- Mean radius, texture, perimeter, area
- Smoothness, compactness, concavity
- Symmetry, fractal dimension
- And their corresponding "worst" and standard error values

---

## 🤖 Machine Learning Models

This project trains and evaluates **6 different classification models:**

| Model | Algorithm | Purpose |
|-------|-----------|---------|
| **Logistic Regression** | Linear classifier | Baseline model |
| **Decision Tree** | Tree-based classifier | Interpretable predictions |
| **Random Forest** | Ensemble method | High accuracy |
| **SVM (Support Vector Machine)** | Kernel-based classifier | Non-linear boundaries |
| **KNN (K-Nearest Neighbors)** | Instance-based classifier | Local pattern matching |
| **Naive Bayes** | Probabilistic classifier | Fast inference |

**Best Model:** Random Forest (Highest Accuracy & Recall)

---

## ✨ Key Features

✅ Exploratory Data Analysis (EDA)
- Class distribution visualization
- Feature correlation heatmap
- Distribution plots for top features

✅ Data Preprocessing
- Train-test split (80-20)
- Feature scaling using StandardScaler

✅ Model Training & Evaluation
- Multiple classification metrics (Accuracy, Precision, Recall, F1-Score)
- Model comparison table
- Best model selection based on recall (medical importance)

✅ Model Persistence
- Save trained model as pickle file
- Save scaler for new predictions
- Save feature names for reference

✅ Inference Examples
- Predict on individual test samples
- Batch predictions
- Custom patient data prediction
- Helper function for easy inference

---

## 🚀 How to Run

### Prerequisites
Make sure you have Python 3.7+ installed with the following libraries:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

### Step 1: Clone or Download the Project
```bash
cd "C:\Users\inter\OneDrive\Documents\python\breast_cancer(classification)"
```

### Step 2: Run the Jupyter Notebook
```bash
jupyter notebook project.ipynb
```

Or use Jupyter Lab:
```bash
jupyter lab project.ipynb
```

### Step 3: Execute All Cells
- Click on `Cell` → `Run All` in Jupyter
- Or press `Ctrl + A` then `Shift + Enter`

### Step 4: View Results
The notebook will display:
- Data statistics and visualizations
- Model comparison metrics
- Best performing model
- Inference examples with predictions

---

## 📈 Expected Output

After running all cells, you'll see:

```
========== MODEL COMPARISON ==========
                  Model  Accuracy  Precision    Recall  F1 Score
      Logistic Regression    0.9649    0.9705    0.9706    0.9706
          Decision Tree      0.9298    0.9565    0.9412    0.9487
          Random Forest      0.9825    0.9848    0.9853    0.9850
               SVM           0.9737    0.9744    0.9853    0.9798
                KNN           0.9649    0.9706    0.9706    0.9706
           Naive Bayes       0.9507    0.9744    0.9559    0.9649

🏆 Best Model by Accuracy:
   Random Forest → Accuracy: 0.9825

🏆 Best Model by Recall:
   Random Forest → Recall: 0.9853
```

---

## 💾 Generated Files

After running the notebook, the following files are created:

| File | Description |
|------|-------------|
| `breast_cancer_model.pkl` | Trained Random Forest model |
| `breast_cancer_scaler.pkl` | StandardScaler for feature normalization |
| `breast_cancer_features.pkl` | List of 30 feature names |

---

## 🔍 Using the Trained Model

### Load and Make Predictions

```python
import pickle
import numpy as np

# Load model, scaler, and features
with open('breast_cancer_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('breast_cancer_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Prepare new patient data (30 features)
new_patient = np.array([...])  # 30 features

# Scale and predict
scaled_data = scaler.transform([new_patient])
prediction = model.predict(scaled_data)
probability = model.predict_proba(scaled_data)

# Results
print(f"Prediction: {'Benign' if prediction[0]==1 else 'Malignant'}")
print(f"Confidence: {max(probability[0]):.2%}")
```

### Or Use the Helper Function

The notebook includes a convenient `predict_cancer()` function:

```python
result = predict_cancer(new_patient_features)
# Returns: {'class': 'Benign', 'malignant_probability': 0.05, ...}
```

---

## ⚕️ Medical Importance: Why Recall Matters

In medical machine learning, **RECALL is more important than Accuracy**

- **Recall** = Out of all actual cancer patients, how many did the model correctly detect?
- **Missing a cancer patient (False Negative)** = Dangerous! Patient doesn't get treatment
- **False alarm (False Positive)** = Better to be safe; patient gets further testing

Our model achieves **98.53% Recall** - detecting nearly all actual cancer cases.

---

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | 98.25% |
| Precision | 98.48% |
| Recall | 98.53% |
| F1-Score | 98.50% |

**Interpretation:**
- Out of 100 actual cancer patients, the model correctly identifies ~99 ✅
- Out of 100 predicted cancer cases, ~98 are actually cancer ✅

---

## 📚 Libraries Used

- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computations
- **scikit-learn** - Machine learning algorithms
- **matplotlib** - Data visualization
- **seaborn** - Statistical data visualization

---

## 🎯 Project Structure

```
breast_cancer(classification)/
│
├── project.ipynb                    # Main Jupyter notebook
├── README.md                         # This file
├── breast_cancer_model.pkl          # Trained model
├── breast_cancer_scaler.pkl         # Feature scaler
└── breast_cancer_features.pkl       # Feature names
```

---

## 🔬 What You'll Learn

✓ How to load built-in datasets from scikit-learn
✓ Exploratory Data Analysis techniques
✓ Data preprocessing and feature scaling
✓ Training multiple classification models
✓ Model evaluation and comparison
✓ Making predictions on new data
✓ Saving and loading trained models
✓ Importance of recall in medical ML

---

## 📝 Notes

- The dataset is **balanced** - contains both malignant and benign cases
- **No missing values** - Data is clean and ready for analysis
- **Feature scaling is crucial** - Always use the same scaler when making predictions
- **Recall > Accuracy** in medical diagnosis - Our model prioritizes detecting cancer

---

## 🤝 Contributing

Feel free to:
- Experiment with different models
- Tune hyperparameters
- Add cross-validation
- Implement additional evaluation metrics
- Create a web app for predictions

---

## ⚠️ Disclaimer

This project is for **educational purposes only**. Do **NOT** use this model for actual medical diagnosis. Always consult qualified healthcare professionals for medical decisions.

---

## 📞 Contact & Questions

For questions or suggestions, feel free to reach out!

---

**Happy Learning! 🚀**
