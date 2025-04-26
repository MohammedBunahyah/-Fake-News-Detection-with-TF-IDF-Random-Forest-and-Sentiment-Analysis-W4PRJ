# 🚀 Fake News Detection with TF-IDF, Random Forest, and Sentiment Analysis

This project builds a **Fake News Classifier** using **TF-IDF**, **Random Forest**, and explores additional **text analysis techniques** like **sentiment scoring**, **word clouds**, and **model comparisons**.

---

## 📚 Project Overview

- ✅ Load and clean a dataset of real and fake news articles
- ✂️ Preprocess text: lowercasing, removing stopwords, lemmatization
- 🔠 Transform articles into TF-IDF feature vectors
- 🌳 Train a Random Forest model to classify fake vs. real news
- 📈 Evaluate model performance with confusion matrix and classification report
- 📝 Clean and predict on a new set of validation articles
- 📊 Visualize most common words and word clouds
- 😎 Analyze sentiment distribution across real and fake news
- ⚡ Compare different models (Random Forest, Logistic Regression, Naive Bayes)
- 💾 Save final models and vectorizers for future use

---

## 🛠️ Tech Stack

| Component               | Tool/Library                    |
|--------------------------|---------------------------------|
| Text Processing          | `NLTK`, `re`, `WordNetLemmatizer` |
| Feature Extraction       | `TF-IDF` (`sklearn`)             |
| Classification Models    | `Random Forest`, `Logistic Regression`, `Naive Bayes` |
| Visualization            | `matplotlib`, `seaborn`, `wordcloud` |
| Sentiment Analysis       | `NLTK Vader`                    |
| Model Saving             | `joblib`                        |
| Environment              | Jupyter / Colab                 |

---

## 🧪 How It Works

1. 📂 Load the news dataset (`data.csv`) and validation set (`validation_data.csv`)
2. 🧹 Clean and preprocess text data for better model focus
3. 🔠 Convert text into TF-IDF vectors
4. 🌳 Train a Random Forest model on the training set
5. 📈 Evaluate with accuracy, precision, recall, and F1-score
6. 🔮 Predict labels for new unseen articles
7. 📊 Generate word clouds and sentiment histograms for deeper analysis
8. 🏆 Compare Random Forest, Logistic Regression, and Naive Bayes models

---

## 💻 Notebook Contents

- `Fake_News_Classification.ipynb`
  - [x] Data loading and cleaning
  - [x] TF-IDF feature engineering
  - [x] Random Forest model training
  - [x] Model evaluation and visualization
  - [x] Validation set prediction
  - [x] Word frequency analysis
  - [x] Sentiment distribution plotting
  - [x] Model benchmarking
  - [x] Saving models

---

## 🧠 Sample Results

| Model                  | Accuracy  |
|-------------------------|-----------|
| Random Forest           | ~96%      |
| Logistic Regression     | ~95%      |
| Naive Bayes             | ~94%      |

- 🎯 **TF-IDF + Random Forest** achieved the best accuracy.
- 📰 **Real News** had slightly more positive sentiment compared to **Fake News**.

---

## ⚠️ Known Issues / Limitations

- ❌ Classifier can struggle on very short or ambiguous news texts
- 🔁 TF-IDF doesn't capture deep semantic meaning (consider embeddings for future improvements)
- ⚡ Some overlap between dramatic real news and fake news can confuse the classifier

---

## 📈 Improvements & Next Steps

- 🔍 Try deep learning approaches like BERT fine-tuning
- 🧠 Incorporate more metadata (publication date, source)
- 🏷️ Perform Named Entity Recognition (NER) to identify suspicious claims
- 🧪 Add automated evaluation on external fake news datasets

---

## 🚀 Try It Yourself

Make sure to install required libraries first:
```bash
pip install nltk scikit-learn matplotlib seaborn wordcloud joblib
```

Download NLTK resources (run once):
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')
```

Then simply run the notebook!

---

✅ **DONE!**  
Would you also want me to give you a **bonus tip**:  
🔹 a suggested folder structure (`/models`, `/data`, `/notebooks`, etc.)?  
🔹 a `.gitignore` file (e.g., to ignore `.pkl` model files)?  

Tell me if you want — ready whenever you are! 🚀✨


# Data sets can be found here
https://drive.google.com/drive/folders/1oG1aPaNebt5dIrYEzA8rvsy2TJ2W4NLB?usp=sharing
