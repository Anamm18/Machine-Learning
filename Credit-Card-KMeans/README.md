
## Project Overview
This project applies K-Means clustering to group 8,950 credit card customers
into 3 distinct segments based on their spending and payment behavior.
No PCA or other models are used — pure K-Means only.

---

## Dataset
| Property | Detail |
|---|---|
| File | `CC_GENERAL.csv` |
| Source | [Kaggle — Credit Card Dataset](https://www.kaggle.com/datasets/arjunbhasin2013/ccdata) |
| Rows | 8,950 customers |
| Columns | 18 (17 features + CUST_ID) |
| Missing values | MINIMUM_PAYMENTS (313), CREDIT_LIMIT (1) — filled with median |

### Features used
| Feature | Description |
|---|---|
| BALANCE | Amount left on account |
| PURCHASES | Total purchases made |
| ONEOFF_PURCHASES | Single large purchases |
| INSTALLMENTS_PURCHASES | Purchases paid in instalments |
| CASH_ADVANCE | Cash taken in advance |
| PURCHASES_FREQUENCY | How often purchases are made (0–1) |
| CASH_ADVANCE_FREQUENCY | How often cash advance is taken (0–1) |
| CREDIT_LIMIT | Credit limit on card |
| PAYMENTS | Total payments made |
| MINIMUM_PAYMENTS | Minimum payments made |
| PRC_FULL_PAYMENT | Percentage of full payment made (0–1) |
| TENURE | Months as customer |

---

## Project Structure
```
project/
│
├── CC_GENERAL.csv                 # Raw dataset
├── kmeans_cc_general.py           # Main script (all 9 steps)
├── kmeans_model.pkl               # Saved model + scaler
├── CC_GENERAL_clustered.csv       # Dataset with Cluster labels added
│
├── graph1_scatter.png             # Balance vs Purchases scatter plot
├── graph2_bar.png                 # Avg metrics per cluster bar chart
├── graph3_pie.png                 # Cluster size distribution
├── graph4_heatmap.png             # Normalized feature heatmap
└── README.md                      # This file
```

---

## How to Run

### 1. Install dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### 2. Place dataset in the same folder
```
CC_GENERAL.csv  ← must be here
kmeans_cc_general.py
```

### 3. Run the script
```bash
python kmeans_cc_general.py
```

---

## Steps in the Code
| Step | What it does |
|---|---|
| 1 | Load CSV, explore shape and statistics |
| 2 | Fill missing values with column median |
| 3 | Drop CUST_ID (non-numeric, not useful) |
| 4 | Scale all features with StandardScaler |
| 5 | Elbow Method — test k=2 to 10, plot inertia + silhouette |
| 6 | Fit final KMeans with optimal k=3 |
| 7 | Profile clusters — compute mean values per group |
| 8 | Draw 4 visualizations |
| 9 | Save labelled dataset as CSV |

---

## Results
| Property | Value |
|---|---|
| Optimal k | 3 |
| Silhouette Score | 0.2400 |
| Algorithm | KMeans (k-means++ init) |
| Scaler | StandardScaler |

### Cluster sizes
| Cluster | Customers | Share |
|---|---|---|
| 0 | 1,275 | 14% |
| 1 | 6,114 | 68% |
| 2 | 1,561 | 17% |

### Cluster profiles (typical interpretation)
| Cluster | Label | Key Signal |
|---|---|---|
| 0 | Cash Advance Users | High CASH_ADVANCE, low PURCHASES |
| 1 | Revolvers | High BALANCE, low PRC_FULL_PAYMENT |
| 2 | Transactors | High PRC_FULL_PAYMENT, low BALANCE |

> Note: re-check labels against your own `graph4_heatmap.png` output
> as K-Means cluster numbering can vary between runs.

---

## Loading the Saved Model
```python
import pickle, pandas as pd
from sklearn.preprocessing import StandardScaler

with open("kmeans_model.pkl", "rb") as f:
    saved = pickle.load(f)

kmeans_loaded = saved["model"]
scaler_loaded = saved["scaler"]

# Predict cluster for a new customer
new_customer = pd.DataFrame([{ ... }])  # fill in feature values
scaled = scaler_loaded.transform(new_customer)
cluster = kmeans_loaded.predict(scaled)
print("Cluster:", cluster[0])
```

---

## Why Silhouette Score is Low (0.24)
A score of 0.24 is expected for this dataset. Credit card behavior data has
a lot of natural overlap — most customers are moderate in everything, so
clusters blend into each other rather than forming tight separate groups.
This is a known characteristic of real-world financial datasets and does not
mean the clustering is wrong — the segments are still meaningfully different
and useful for business decisions.

---

## Requirements
```
pandas
numpy
matplotlib
seaborn
scikit-learn
pickle (built-in)
```
