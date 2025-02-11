import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Loading the TF-IDF features and labels
with open("data/tfidf_features.pkl", "rb") as f:
    X = pickle.load(f)

with open("data/labels.pkl", "rb") as f:
    y = pickle.load(f)

print(f"Feature matrix shape: {X.shape}")
print(f"Labels shape: {y.shape}")


# Split data into 80% train and 20% test
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set: {Xtrain.shape}, Testing set: {Xtest.shape}")

# Initialize the Logistic Regression model
model = LogisticRegression(max_iter=500)
model.fit(Xtrain, ytrain)

print("Model trained successfully!")

# Predict on the test set
yPred = model.predict(Xtest)

#finding accuracy
accuracy = accuracy_score(ytest, yPred)
print(f"Model accuracy: {accuracy:.4f}")

# Classification report
print("\nClassification Report:")
print(classification_report(ytest, yPred))

#saving

with open("data/model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved successfully!")
