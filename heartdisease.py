
# Step 1: Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt


data = pd.read_csv("heart_disease.csv")
print("Dataset shape:", data.shape)
print(data.head())

print(data.isnull().sum())


X = data.drop(columns='target', axis=1)
y = data['target']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("✅ Model Accuracy:", accuracy)

print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Step 7: Predict on new data
sample = np.array([[63,1,3,145,233,1,0,150,0,2.3,0,0,1]]) 
prediction = model.predict(sample)
print("\nPredicted Output (1 = Disease, 0 = No Disease):", prediction)