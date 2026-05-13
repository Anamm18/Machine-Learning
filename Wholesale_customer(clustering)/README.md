# Wholesale Customer Segmentation using K-Means Clustering

## Project Overview

This project uses the K-Means Clustering algorithm to segment wholesale customers based on their annual spending patterns across different product categories. The main goal of the project is to identify groups of customers with similar purchasing behavior.

This is an unsupervised machine learning project developed using Python and Scikit-learn.

---

# Features

* Data preprocessing and cleaning
* Feature scaling using StandardScaler
* K-Means clustering implementation
* Elbow Method for selecting optimal clusters
* Silhouette Score evaluation
* Customer segmentation analysis
* Data visualization using graphs
* Model saving using Pickle

---

# Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Jupyter Notebook

---

# Dataset

Dataset used:

`Wholesale customers data.csv`

The dataset contains annual spending data of wholesale customers in different product categories such as:

* Fresh
* Milk
* Grocery
* Frozen
* Detergents_Paper
* Delicassen

---

# Machine Learning Concepts Used

* Unsupervised Learning
* K-Means Clustering
* Elbow Method
* Silhouette Score
* Feature Scaling
* Customer Segmentation

---

# Project Workflow

1. Load Dataset
2. Data Cleaning
3. Feature Scaling
4. Find Optimal Number of Clusters
5. Train K-Means Model
6. Visualize Customer Segments
7. Save Trained Model and Scaler

---

# Visualizations

The project includes different visualizations for cluster analysis such as:

* Scatter Plot
* Cluster Distribution Graph
* Heatmap
* Pair Plot

---

# Files Included

```text
Wholesale-Customer-Segmentation/
│
├── project.ipynb
├── Wholesale customers data.csv
├── wholesale_model.pkl
├── scaler.pkl
└── README.md
```

---

# How to Run the Project

## Clone Repository

```bash
git clone <repository-link>
```

## Install Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Run Jupyter Notebook

```bash
jupyter notebook
```

Open:

`project.ipynb`

---

# Model Information

* Algorithm: K-Means Clustering
* Scaling Method: StandardScaler
* Model File: `wholesale_model.pkl`
* Scaler File: `scaler.pkl`

---

# Learning Outcomes

Through this project, I learned:

* Data preprocessing techniques
* Clustering algorithms
* Customer segmentation
* Model evaluation methods
* Data visualization
* Machine learning workflow

---

# Future Improvements

* Deploy project using Streamlit or Flask
* Add interactive dashboard
* Compare with DBSCAN and Hierarchical Clustering
* Improve cluster visualization and interpretation

---

# Author

Anam Mulla
