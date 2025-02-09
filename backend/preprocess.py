import nltk
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.data.path.append('/Users/kenny/nltk_data')
nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

# text cleaning function 
def cleanText(text):
    text = text.lower()  # Lowercase
    text = re.sub(r'@\w+', '', text)  # Remove @mentions
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'http\S+|www\S+', '', text)  # remove urls
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces

    words = word_tokenize(text)  # Tokenization
    words = [word for word in words if word not in stop_words]  # Remove stopwords
    return ' '.join(words)  


df = pd.read_csv('data/combinedNews.csv')

print("\n🔹 Original Text Sample:")
print(df["text"].iloc[0])  #  sample before cleaning for comparison

df["cleaned_text"] = df["text"].apply(cleanText)

print("\n✅ Cleaned Text Sample:")
print(df["cleaned_text"].iloc[0])  # Print cleaned sample

# Tokenization (apply word_tokenize to cleaned text)
df["tokens"] = df["cleaned_text"].apply(word_tokenize)

print("\n🔹 Tokenized Text Sample:")
print(df["tokens"].iloc[0])  # Print tokenized text sample

# Save cleaned dataset
df.to_csv('data/cleanedNews.csv', index=False, encoding='utf-8')

print("\n✅ Cleaned dataset saved as 'data/cleanedNews.csv'!")
