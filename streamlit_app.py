import streamlit as st
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load model and vectorizer
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

@st.cache_resource
def load_vectorizer():
    return joblib.load("vectorizer.pkl")

model = load_model()
vectorizer = load_vectorizer()

# Text Preprocessing Function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    tokens = word_tokenize(text)  # Tokenization
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(tokens)

# Streamlit UI
st.title("Fake News Detection App")
st.write("Enter a news article title and text below to classify it as **Real or Fake**.")

# User Input
title = st.text_input("Enter News Title:")
text = st.text_area("Enter News Content:")

if st.button("Predict"):
    if title and text:
        combined_text = clean_text(title + " " + text)
        transformed_text = vectorizer.transform([combined_text])
        prediction = model.predict(transformed_text)[0]
        result = "Real News ✅" if prediction == 1 else "Fake News ❌"
        st.subheader(f"Prediction: {result}")
    else:
        st.warning("Please enter both a title and content.")

# Upload CSV for Batch Predictions
st.subheader("Upload CSV for Bulk Predictions")
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df['cleaned_text'] = df['title'] + " " + df['text']
    df['cleaned_text'] = df['cleaned_text'].apply(clean_text)
    df_tfidf = vectorizer.transform(df['cleaned_text'])
    df['label'] = model.predict(df_tfidf)
    df['label'] = df['label'].apply(lambda x: 0 if x == 2 else x)
    st.write(df[['title', 'label']])
    st.download_button("Download Predictions", df.to_csv(index=False), "predictions.csv")
