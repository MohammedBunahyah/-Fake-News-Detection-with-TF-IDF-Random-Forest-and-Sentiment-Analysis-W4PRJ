import streamlit as st  # Import Streamlit for the web app
import pandas as pd  # Handle data
import re  # Regular expressions for text cleaning
import nltk  # Natural Language Processing toolkit
from nltk.corpus import stopwords  # Import stop words
from nltk.tokenize import word_tokenize  # Tokenization
from nltk.stem import WordNetLemmatizer  # Lemmatization
from sklearn.feature_extraction.text import TfidfVectorizer  # TF-IDF Vectorizer
from sklearn.ensemble import RandomForestClassifier  # Import the trained model
import pickle  # Save and load model
import os  # Check file paths

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load trained model
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as file:
        model = pickle.load(file)
    return model

# Load trained TF-IDF vectorizer
@st.cache_resource
def load_vectorizer():
    with open("vectorizer.pkl", "rb") as file:
        vectorizer = pickle.load(file)
    return vectorizer

# Initialize model and vectorizer
model = load_model()
vectorizer = load_vectorizer()

# Function to clean text
def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    tokens = word_tokenize(text)  # Tokenize text
    tokens = [word for word in tokens if word not in stopwords.words('english')]  # Remove stop words
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]  # Lemmatization
    return " ".join(tokens)  # Return cleaned text

# Streamlit UI
st.title("📰 Fake News Detector")
st.write("Enter a news article or headline below to check if it's **real or fake**.")

# User input box
user_input = st.text_area("Enter news text here:")

# Predict button
if st.button("Check"):
    cleaned_input = clean_text(user_input)  # Preprocess input text
    input_vector = vectorizer.transform([cleaned_input])  # Convert text to TF-IDF
    prediction = model.predict(input_vector)[0]  # Make prediction
    st.write("**Prediction:**", "✅ **Real News**" if prediction == 1 else "🚨 **Fake News**")

# Footer
st.markdown("---")
st.markdown("Developed by **Your Name** | Machine Learning | Streamlit Web App")
