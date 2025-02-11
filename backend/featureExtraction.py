import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import os

# Load the cleaned dataset
data_path = os.path.join("data", "cleanedNews.csv")
df = pd.read_csv(data_path)

# Ensure the column exists
if "cleaned_text" not in df.columns:
    raise ValueError("Column 'cleaned_text' not found in the dataset!")

print(f"Dataset loaded successfully. Initial rows: {len(df)}")

# 🔧 Fix: Remove NaN values
df = df.dropna(subset=["cleaned_text"])  # Drop rows where 'cleaned_text' is NaN
df["cleaned_text"] = df["cleaned_text"].fillna("")  # Replace remaining NaNs

print(f"After cleaning: {len(df)} rows remaining.")

# Initialize TF-IDF vectorizer
vectorizer = TfidfVectorizer(max_features=5000)

# Fit and transform the text
X_tfidf = vectorizer.fit_transform(df["cleaned_text"])

print(f"TF-IDF feature extraction complete. Shape: {X_tfidf.shape}")

# Save the transformed features
tfidf_output_path = os.path.join("data", "tfidf_features.pkl")
with open(tfidf_output_path, "wb") as file:
    pickle.dump(X_tfidf, file)

# Save the labels separately
labels_csv_path = os.path.join("data", "labels.csv")
labels_pkl_path = os.path.join("data", "labels.pkl")

df["label"].to_csv(labels_csv_path, index=False)  # Save as CSV
with open(labels_pkl_path, "wb") as file:
    pickle.dump(df["label"].values, file)  # Save as Pickle

print("Features and labels saved successfully.")