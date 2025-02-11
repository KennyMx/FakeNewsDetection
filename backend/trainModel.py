import pickle
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

# Paths to saved files
tfidf_path = os.path.join("data", "tfidf_features.pkl")
labels_path = os.path.join("data", "labels.csv")

# Load TF-IDF Features
with open(tfidf_path, "rb") as f:
    X = pickle.load(f)

# Load Labels
df_labels = pd.read_csv(labels_path)
y = df_labels.values.ravel()  # Convert DataFrame column to 1D array

# Split data into 80% train and 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]  # Probabilities for ROC Curve

# Evaluation Metrics
accuracy = model.score(X_test, y_test)
print(f"\n✅ Model Accuracy: {accuracy:.4f}")
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))

#  Confusion Matrix
plt.figure(figsize=(6, 5))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Fake", "Real"], yticklabels=["Fake", "Real"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

#  ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color="blue", lw=2, label=f"AUC = {roc_auc:.4f}")
plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.show()

# ✅ Save Model
model_dir = "models"
os.makedirs(model_dir, exist_ok=True)  # Ensure directory exists
model_path = os.path.join(model_dir, "fake_news_model.pkl")

with open(model_path, "wb") as f:
    pickle.dump(model, f)

print(f"✅ Model saved successfully to {model_path}")

# 📌 Feature Importance (Which words matter most?)
importances = model.feature_importances_

# Load vectorizer to get feature names
vectorizer = TfidfVectorizer(max_features=5000)  # Ensure same vectorizer is used
with open("data/tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

features = np.array(vectorizer.get_feature_names_out())  # Extract feature names
sorted_indices = np.argsort(importances)[::-1]  # Sort by importance

top_n = 20  # Show top 20 words
print("\n🔍 Top Important Words in Classification:")
for i in range(top_n):
    print(f"{features[sorted_indices[i]]}: {importances[sorted_indices[i]]:.4f}")

# plot top words 
plt.figure(figsize=(10, 5))
plt.barh(features[sorted_indices[:top_n]], importances[sorted_indices[:top_n]], color="blue")
plt.xlabel("Importance Score")
plt.ylabel("Words")
plt.title("Top Words That Determine Fake/Real News")
plt.gca().invert_yaxis()
plt.show()
