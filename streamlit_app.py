import streamlit as st
import pandas as pd
import gdown
import os

# Google Drive File IDs (replace with yours)
DATA_FILE_ID = "1AsdUWNsA981I0GXty9r345IBC4Ly_D1X"
VALIDATION_FILE_ID = "1Wj8Z4FrhVvYZtfhLvaaiyoTYdtWFFRn9"

# Function to download CSV from Google Drive
@st.cache_data
def download_csv_from_gdrive(file_id, output_path):
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, output_path, fuzzy=True, quiet=False)
    return pd.read_csv(output_path)

# Download datasets
df = download_csv_from_gdrive(DATA_FILE_ID, "data.csv")
val_df = download_csv_from_gdrive(VALIDATION_FILE_ID, "validation_data.csv")

# Display dataset preview
st.title("Fake News Detection App")
st.write("Dataset Overview:")
st.write(df.head())

# User input field
user_input = st.text_area("Enter a news headline or article:")

if st.button("Check News"):
    # Load pre-trained model
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.ensemble import RandomForestClassifier

    # Load vectorizer and model
    vectorizer = TfidfVectorizer(max_features=5000)
    model = RandomForestClassifier(n_estimators=100, random_state=42)

    # Preprocess and predict
    def clean_text(text):
        import re, string, nltk
        from nltk.corpus import stopwords
        from nltk.tokenize import word_tokenize
        from nltk.stem import WordNetLemmatizer

        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')

        stop_words = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()

        text = text.lower()
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        tokens = word_tokenize(text)
        tokens = [word for word in tokens if word not in stop_words]
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
        return " ".join(tokens)

    cleaned_input = clean_text(user_input)
    input_vector = vectorizer.fit_transform(df["cleaned_text"]).transform([cleaned_input])
    prediction = model.fit(vectorizer.transform(df["cleaned_text"]), df["label"]).predict(input_vector)[0]

    st.write("Prediction:", "📰 **Real News**" if prediction == 1 else "🚨 **Fake News**")
