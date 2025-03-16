import streamlit as st  # Import Streamlit for the web app
import pandas as pd  # Import Pandas for data handling
import gdown  # Import gdown to download files from Google Drive
import re  # Regular expressions for text cleaning
import string  # String operations for punctuation removal
import nltk  # Natural Language Toolkit for text processing
from nltk.corpus import stopwords  # Stop words removal
from nltk.tokenize import word_tokenize  # Tokenization
from nltk.stem import WordNetLemmatizer  # Lemmatization
from sklearn.feature_extraction.text import TfidfVectorizer  # TF-IDF for text representation
from sklearn.ensemble import RandomForestClassifier  # Random Forest for classification
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report  # Evaluation metrics
import matplotlib.pyplot as plt  # Import Matplotlib for visualization
from sklearn.metrics import ConfusionMatrixDisplay  # Confusion Matrix visualization

# 📌 Google Drive File IDs (Replace these with your actual IDs)
DATA_FILE_ID = "your_data_file_id"  # File ID for data.csv
VALIDATION_FILE_ID = "your_validation_file_id"  # File ID for validation_data.csv
PREDICTIONS_FILE_ID = "your_predictions_file_id"  # File ID for validation_predictions.csv

# 🔽 Function to download CSV files from Google Drive
@st.cache_resource
def download_csv_from_gdrive(file_id, filename):
    """Downloads a CSV file from Google Drive using gdown."""
    url = f"https://drive.google.com/uc?id={file_id}"
    output_path = filename
    gdown.download(url, output_path, quiet=False)
    return pd.read_csv(output_path)

# ✅ Load datasets
df = download_csv_from_gdrive(DATA_FILE_ID, "data.csv")
val_df = download_csv_from_gdrive(VALIDATION_FILE_ID, "validation_data.csv")
val_predictions = download_csv_from_gdrive(PREDICTIONS_FILE_ID, "validation_predictions.csv")

# 🔹 Initialize NLP Tools
nltk.download('punkt')  
nltk.download('stopwords')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# 🔹 Text Cleaning Function
def clean_text(text):
    """Cleans input text: lowercase, remove numbers, punctuation, stopwords, and lemmatize."""
    text = text.lower()
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(tokens)

# 🔹 Apply Cleaning to Dataset
df['cleaned_text'] = df['title'] + " " + df['text']
df['cleaned_text'] = df['cleaned_text'].apply(clean_text)

# 📌 Split Data for Training & Testing
from sklearn.model_selection import train_test_split
X = df['cleaned_text']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🔹 Convert Text into TF-IDF Features
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# ✅ Train a Random Forest Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_tfidf, y_train)

# 📌 Streamlit Web App UI
st.title("📰 Fake News Detector")
st.write("### Enter a news headline or article to check if it's **Real or Fake**")

# 🔹 User Input
user_input = st.text_area("Enter text here:")

# 🔹 Check Button
if st.button("Check"):
    cleaned_input = clean_text(user_input)  # Preprocess input text
    input_vector = vectorizer.transform([cleaned_input])  # Convert to TF-IDF
    prediction = model.predict(input_vector)[0]  # Make prediction
    st.subheader("Prediction:")
    st.success("✅ **Real News**" if prediction == 1 else "🚨 **Fake News**")

# 📊 Display Data Preview
st.write("### 🔍 Preview of Dataset")
st.dataframe(df.head())

# 🎯 Model Evaluation
st.write("### 📊 Model Performance Evaluation")

# ✅ Accuracy
y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
st.write(f"**Model Accuracy:** {accuracy:.2f}")

# ✅ Confusion Matrix
st.write("### 🔍 Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(6,5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Fake News (0)", "Real News (1)"])
disp.plot(ax=ax, cmap="Blues", values_format="d")
st.pyplot(fig)

# ✅ Classification Report
st.write("### 🔍 Classification Report")
st.text(classification_report(y_test, y_pred))

# 📊 Show Common Words
st.write("### 📌 Most Common Words in Fake vs. Real News")

from collections import Counter
real_words = " ".join(df[df["label"] == 1]["cleaned_text"]).split()
fake_words = " ".join(df[df["label"] == 0]["cleaned_text"]).split()
real_counts = Counter(real_words).most_common(15)
fake_counts = Counter(fake_words).most_common(15)

fig, ax = plt.subplots(figsize=(12,5))
ax.bar(*zip(*real_counts), color='blue', label="Real News")
ax.bar(*zip(*fake_counts), color='red', alpha=0.7, label="Fake News")
plt.xticks(rotation=45)
plt.title("Most Common Words in Fake vs. Real News")
plt.legend()
st.pyplot(fig)

# 📌 Word Cloud Visualization
st.write("### ☁️ Word Cloud for Fake vs. Real News")

from wordcloud import WordCloud
real_wc = WordCloud(width=800, height=400, background_color="white").generate(" ".join(real_words))
fake_wc = WordCloud(width=800, height=400, background_color="black").generate(" ".join(fake_words))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,5))
ax1.imshow(real_wc, interpolation="bilinear")
ax1.axis("off")
ax1.set_title("Real News Word Cloud")

ax2.imshow(fake_wc, interpolation="bilinear")
ax2.axis("off")
ax2.set_title("Fake News Word Cloud")

st.pyplot(fig)

# ✅ Sentiment Analysis
st.write("### 📌 Sentiment Analysis in Fake vs. Real News")

from nltk.sentiment import SentimentIntensityAnalyzer
nltk.download("vader_lexicon")
sia = SentimentIntensityAnalyzer()
df["sentiment"] = df["cleaned_text"].apply(lambda text: sia.polarity_scores(text)["compound"])

fig, ax = plt.subplots(figsize=(10,5))
ax.hist(df[df["label"] == 1]["sentiment"], bins=20, alpha=0.7, label="Real News", color="blue")
ax.hist(df[df["label"] == 0]["sentiment"], bins=20, alpha=0.7, label="Fake News", color="red")
plt.title("Sentiment Distribution in Fake vs. Real News")
plt.xlabel("Sentiment Score")
plt.ylabel("Count")
plt.legend()
st.pyplot(fig)

st.write("### 🎯 Sentiment Score shows how emotional Fake vs. Real News is")

# ✅ Done! 🚀
