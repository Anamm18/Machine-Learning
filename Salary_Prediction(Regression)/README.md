# Salary Prediction (Regression)

A Python notebook project for predicting salaries using regression models. The project loads a salary dataset, preprocesses the data, trains several regression models, evaluates their performance, and saves the best model and scaler for inference.

## Project Overview

- Dataset: `salary_prediction_data.csv`
- Goal: Predict employee salary based on features such as experience, education, and job title.
- Models trained:
  - Linear Regression
  - Ridge Regression
  - Lasso Regression
  - Decision Tree Regressor
  - Random Forest Regressor
  - Gradient Boosting Regressor
  - K-Nearest Neighbors Regressor

## Files

- `project.ipynb` - Jupyter notebook containing data loading, preprocessing, model training, evaluation, and model saving.
- `salary_prediction_data.csv` - Dataset used for training.
- `salary_model.pkl` - Saved trained model created by the notebook.
- `salary_scaler.pkl` - Saved scaler created by the notebook.

## Requirements

- Python 3.x
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

Install dependencies with:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## How to Run

Open `project.ipynb` in Jupyter Notebook or JupyterLab and run the cells in order. The notebook performs:

1. Data loading and exploration
2. Data preprocessing and encoding
3. Train/test split
4. Feature scaling
5. Model training and evaluation
6. Saving the model and scaler to `.pkl` files

## Inference Example

Use this Python code to load the saved model and scaler, prepare a new input row, and predict salary:

```python
import pandas as pd
import pickle

with open('salary_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

with open('salary_scaler.pkl', 'rb') as f:
    loaded_scaler = pickle.load(f)

new_data = pd.DataFrame([{
    'Experience': 5,
    'Education': 'MI',
    'Job_Title': 'Data Scientist'
    # add any additional columns required by your dataset
}])

new_data = pd.get_dummies(new_data)
new_data = new_data.reindex(columns=X.columns, fill_value=0)
X_new = loaded_scaler.transform(new_data)
predicted_salary = loaded_model.predict(X_new)
print('Predicted salary:', predicted_salary[0])
```

> Note: `X.columns` refers to the feature columns used during training in the notebook. If running this outside the notebook, save the feature list or manually recreate the same encoded columns.

## Notes

- Make sure `salary_model.pkl` and `salary_scaler.pkl` exist before running inference.
- If your dataset contains additional categorical columns, ensure they are encoded the same way as in the notebook with `pd.get_dummies()`.
- The notebook currently uses K-Nearest Neighbors as the final model, but you can choose the model with the best R² score from the evaluation results.
