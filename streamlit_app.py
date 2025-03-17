import streamlit as st
import pandas as pd
import joblib
import gdown
import os
import nltk
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# 🔧 Ensure required NLTK resources are downloaded
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")  # Fixes certain lemmatizer issues
nltk.download("averaged_perceptron_tagger")  # Helps with tokenization errors

# ✅ Load NLTK components
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# 🛠 Google Drive File IDs (Replace these with your actual Google Drive file IDs)
MODEL_FILE_ID = "1431m5bn3RJ0SAOpy3zuRRPJMW_LwpVBo"  # Your model file ID
VECTORIZER_FILE_ID = "1HliHGc-mq_q3CvAVzkKrubUv61S8I2Bp"  # Your vectorizer file ID
DATA_FILE_ID = "1AsdUWNsA981I0GXty9r345IBC4Ly_D1X"  # Your dataset file ID

# 📥 Function to download files from Google Drive
@st.cache_data
def download_from_gdrive(file_id, output_path):
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, output_path, quiet=False)
    return output_path

# ✅ Download and Load Model & Vectorizer
model_path = download_from_gdrive(MODEL_FILE_ID, "random_forest_model.pkl")
vectorizer_path = download_from_gdrive(VECTORIZER_FILE_ID, "tfidf_vectorizer.pkl")

# 🔍 Check if files exist before loading
if not os.path.exists(model_path):
    st.error("🚨 Model file not found! Check Google Drive file ID or upload manually.")
else:
    model = joblib.load(model_path)

if not os.path.exists(vectorizer_path):
    st.error("🚨 Vectorizer file not found! Check Google Drive file ID or upload manually.")
else:
    vectorizer = joblib.load(vectorizer_path)

# ✅ Load Dataset for Display (Optional)
data_path = download_from_gdrive(DATA_FILE_ID, "data.csv")

if os.path.exists(data_path):
    df = pd.read_csv(data_path)
else:
    st.error("🚨 Dataset file not found! Check Google Drive file ID or upload manually.")
    df = pd.DataFrame()  # Create empty DataFrame to prevent errors

# 🔎 Preprocessing Function
def clean_text(text):
    text = text.lower()
    text = re.sub(r"\d+", "", text)  # Remove numbers
    text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
    text = re.sub(r"\s+", " ", text).strip()  # Remove extra spaces
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(tokens)

# 🌐 Streamlit App UI
st.title("📰 Fake News Detection App")

st.write("### Dataset Overview:")
if not df.empty:
    st.write(df.head())  # Show first rows of dataset
else:
    st.write("No dataset available.")

# 📝 User Input
user_input = st.text_area("Enter a news headline or article:")

if st.button("Check News"):
    if not user_input.strip():
        st.warning("⚠️ Please enter a news headline or article.")
    else:
        cleaned_input = clean_text(user_input)  # Clean input text
        input_vector = vectorizer.transform([cleaned_input])  # Convert text to TF-IDF
        
        prediction = model.predict(input_vector)[0]  # Predict
        
        # 🎯 Show result
        st.write("### Prediction:")
        st.success("✅ Real News") if prediction == 1 else st.error("🚨 Fake News")
