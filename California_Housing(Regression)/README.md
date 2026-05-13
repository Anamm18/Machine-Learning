# California Housing Price Prediction

## Overview

This project demonstrates a complete machine learning workflow for predicting median house values in California districts using regression techniques. The project uses the California Housing dataset, a built-in dataset from the scikit-learn library, which contains data derived from the 1990 U.S. Census.

## Dataset Description

The California Housing dataset consists of 20,640 samples and 8 features collected from California districts. The target variable is the median house value for each district.

### Features:
- **MedInc**: Median income in block group (in tens of thousands of dollars)
- **HouseAge**: Median house age in block group
- **AveRooms**: Average number of rooms per household
- **AveBedrms**: Average number of bedrooms per household
- **Population**: Block group population
- **AveOccup**: Average number of household members
- **Latitude**: Block group latitude
- **Longitude**: Block group longitude

### Target Variable:
- **MedHouseVal**: Median house value for California districts (in hundreds of thousands of dollars, capped at 5.0)

## Project Structure

```
california_housing(Regression)/
├── California_housing(Regression).ipynb  # Jupyter notebook with analysis
├── app.py                                # Streamlit web application
├── best_model.pkl                        # Serialized Random Forest model
├── scaler.pkl                            # Serialized StandardScaler
└── README.md                             # This file
```

## Requirements

- Python 3.7 or higher
- Required libraries:
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - scikit-learn
  - streamlit

## Installation

1. Clone or download this repository
2. Install the required dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn streamlit
   ```

## Usage

### Jupyter Notebook Analysis

Run the `California_housing(Regression).ipynb` notebook to:
- Load and explore the dataset
- Perform exploratory data analysis (EDA)
- Preprocess the data (scaling, train-test split)
- Train and evaluate multiple regression models
- Save the best performing model

### Streamlit Web Application

Launch the interactive dashboard:

```bash
streamlit run app.py
```

The app provides:
- **🔮 Predict**: Interactive sliders to input features and get house price predictions
- **📊 Model Performance**: View evaluation metrics and actual vs predicted plots
- **📈 Feature Importance**: Analyze which features most influence predictions
- **🗂️ Data Explorer**: Browse the dataset and visualize distributions

## Machine Learning Models

The project compares several regression algorithms:

1. **Linear Regression**: Baseline linear model
2. **Ridge Regression**: Linear regression with L2 regularization
3. **Lasso Regression**: Linear regression with L1 regularization
4. **Decision Tree Regressor**: Tree-based model with max depth of 5
5. **Random Forest Regressor**: Ensemble of 100 decision trees

### Model Performance

Based on evaluation metrics (MSE, MAE, R² Score) on the test set:

| Model              | MSE      | MAE      | R² Score |
|-------------------|----------|----------|----------|
| Linear Regression | ~0.56   | ~0.53   | ~0.58   |
| Ridge Regression  | ~0.56   | ~0.53   | ~0.58   |
| Lasso Regression  | ~0.89   | ~0.72   | ~0.28   |
| Decision Tree     | ~0.49   | ~0.46   | ~0.62   |
| Random Forest     | ~0.26   | ~0.33   | ~0.80   |

**Best Model**: Random Forest Regressor with R² = ~0.80

## Data Preprocessing

- **Feature Scaling**: StandardScaler applied to normalize features
- **Train-Test Split**: 80% training, 20% testing with random_state=42
- **No Missing Values**: Dataset is complete with no null values

## Key Insights

- Median income is the strongest predictor of house prices
- Location features (latitude/longitude) show geographic patterns
- Random Forest outperforms linear models due to non-linear relationships
- House age and average rooms also contribute significantly

## Deployment

The trained Random Forest model and scaler are saved as pickle files for easy deployment. The Streamlit app demonstrates how to load these artifacts and make real-time predictions.

## License

This project uses the California Housing dataset from scikit-learn, which is in the public domain.

## Contributing

Feel free to fork this repository and submit pull requests with improvements or additional features.
