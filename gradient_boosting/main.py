# Import NumPy for numerical operations
import numpy as np

# Import Pandas for handling datasets
import pandas as pd

# Import the Breast Cancer dataset
from sklearn.datasets import load_breast_cancer

# Import function to split dataset into training and testing sets
from sklearn.model_selection import train_test_split

# Import Gradient Boosting Classifier
from sklearn.ensemble import GradientBoostingClassifier

# Import evaluation metrics
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# --------------------------------------------------------
# STEP 1 : Load Dataset
# --------------------------------------------------------

# Load the built-in breast cancer dataset
data = load_breast_cancer()

# Features (Independent variables)
X = data.data

# Target (Dependent variable)
y = data.target

print("Shape of Dataset :", X.shape)

# --------------------------------------------------------
# STEP 2 : Split Dataset
# --------------------------------------------------------

# 80% Training
# 20% Testing

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# --------------------------------------------------------
# STEP 3 : Create Gradient Boosting Model
# --------------------------------------------------------

model = GradientBoostingClassifier(

    # Number of trees
    n_estimators=100,

    # Learning rate
    learning_rate=0.1,

    # Maximum depth of each tree
    max_depth=3,

    # Random state for reproducibility
    random_state=42
)

# --------------------------------------------------------
# STEP 4 : Train Model
# --------------------------------------------------------

model.fit(X_train, y_train)

# --------------------------------------------------------
# STEP 5 : Prediction
# --------------------------------------------------------

y_pred = model.predict(X_test)

# --------------------------------------------------------
# STEP 6 : Evaluate Model
# --------------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy : ", accuracy)

print("\nConfusion Matrix\n")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report\n")
print(classification_report(y_test, y_pred))

# --------------------------------------------------------
# STEP 7 : Feature Importance
# --------------------------------------------------------

importance = model.feature_importances_

print("\nFeature Importance\n")

for i, score in enumerate(importance):
    print(f"Feature {i+1} : {score:.4f}")