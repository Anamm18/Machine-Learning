✍️ Handwritten Digit Recognition (MNIST)
A Machine Learning project to classify handwritten digits 0–9 using the built-in scikit-learn digits dataset. Trained and compared 6 classification models with SVM achieving the best accuracy of ~98.9%.

📦 Dataset
This project uses the built-in scikit-learn Digits dataset — no manual download required!
pythonfrom sklearn.datasets import load_digits
digits = load_digits()
PropertyDetailSourcesklearn.datasets.load_digits()Total Samples1,797 imagesImage Size8×8 pixels (64 features)Classes10 (digits 0 to 9)Samples per class~178–183 (balanced)Download needed?❌ No — built into scikit-learn

💡 Anyone who installs scikit-learn already has this dataset. No CSV or ZIP file needed!


🛠️ Tech Stack

Python 3.x
Scikit-learn — ML models + built-in dataset
Pandas — data manipulation
NumPy — numerical operations
Matplotlib & Seaborn — data visualization


📁 Project Structure
handwritten-digit-recognition/
│
├── project.ipynb           # Main Jupyter notebook
├── train_model.py          # Script to retrain & save model
├── digit_model.pkl         # Trained SVM model
├── digit_scaler.pkl        # StandardScaler
├── requirements.txt        # Dependencies
└── README.md               # Project info

📊 Models Compared
ModelAccuracySVM (RBF Kernel)~98.9% ✅ BestKNN~98.3%Logistic Regression~97.5%Random Forest~97.2%Decision Tree~85.8%Naive Bayes~81.9%

SVM performs best because it handles high-dimensional pixel data exceptionally well.


📈 Visualizations Included

✅ Sample digit images (0–9)
✅ Class distribution countplot
✅ Average digit image per class
✅ Model accuracy comparison bar chart
✅ Confusion matrix
✅ Misclassified digits visualization
✅ PCA 2D plot


🚀 How to Run
Step 1 — Clone the repository
bashgit clone https://github.com/your-username/handwritten-digit-recognition.git
cd handwritten-digit-recognition
Step 2 — Install dependencies
bashpip install -r requirements.txt
Step 3 — Retrain the model (generates pkl files)
bashpython train_model.py
Step 4 — Open the notebook
bashjupyter notebook project.ipynb

🔁 Retrain the Model
Run train_model.py to retrain and regenerate the pickle files:
pythonfrom sklearn.datasets import load_digits   # No download needed!
from sklearn.svm import SVC
import pickle

digits  = load_digits()                    # Built-in dataset loads instantly
X, y    = digits.data, digits.target
# ... train and save to digit_model.pkl

🔮 How to Use the Pickle File
pythonimport pickle
import numpy as np

# Load model and scaler
with open('digit_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('digit_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Predict on new image (8x8 = 64 pixel values flattened)
new_image  = np.array([[0,0,5,13,9,1,0,0,
                         0,0,13,15,10,15,5,0,
                         0,3,15,2,0,11,8,0,
                         0,4,12,0,0,8,8,0,
                         0,5,8,0,0,9,8,0,
                         0,4,11,0,1,12,7,0,
                         0,2,14,5,10,12,0,0,
                         0,0,6,13,10,0,0,0]])

scaled     = scaler.transform(new_image)
prediction = model.predict(scaled)
print(f"Predicted Digit: {prediction[0]}")

📋 Requirements
scikit-learn
pandas
numpy
matplotlib
seaborn
jupyter
Install all at once:
bashpip install -r requirements.txt

🖥️ Push to GitHub
bashgit init
git add .
git commit -m "✍️ Handwritten Digit Recognition ML Project"
git remote add origin https://github.com/your-username/handwritten-digit-recognition.git
git branch -M main
git push -u origin main

📌 Key Notes

The dataset is 100% built into scikit-learn — no external download needed
Always scale input data using the saved digit_scaler.pkl before predicting
Image pixels must be flattened to 64 values (8×8 → 1D array) before passing to model
SVM works best for this dataset due to high-dimensional pixel features
