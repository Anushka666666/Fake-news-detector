import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import shap
import streamlit as st
import requests
from bs4 import BeautifulSoup

# Step 1: Setup

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()
    # keep letters and numbers, remove only special characters
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text

def get_article_text(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}  # avoid blocking
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.content, "html.parser")
        paragraphs = soup.find_all('p')
        article_text = ' '.join([p.get_text() for p in paragraphs])
        return article_text
    except:
        return ""


# Step 2: Load Dataset (larger sample)

df_fake = pd.read_csv('Fake.csv').sample(5000, random_state=42)
df_real = pd.read_csv('True.csv').sample(5000, random_state=42)

df_fake['label'] = 1  # FAKE
df_real['label'] = 0  # REAL
df = pd.concat([df_fake, df_real])
df = df[['text', 'label']]
df['clean_text'] = df['text'].apply(clean_text)


# Step 3: TF-IDF + Logistic Regression

vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1,2))
X = vectorizer.fit_transform(df['clean_text'])
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000, solver='saga')
model.fit(X_train, y_train)

# SHAP explainer
explainer = shap.LinearExplainer(model, X_train, feature_perturbation="interventional")


# Step 4: Streamlit GUI

st.title("🚨 Fake News Detector ")

input_type = st.radio("Select input type:", ("Paste text", "Enter URL"))

if input_type == "Paste text":
    input_news = st.text_area("Enter news headline or article:")
    if st.button("Predict Text"):
        if input_news.strip() != "":
            clean = clean_text(input_news)
            vectorized = vectorizer.transform([clean])
            pred = model.predict(vectorized)[0]
            label = "FAKE" if pred == 1 else "REAL"
            st.subheader(f"Prediction: {label}")

            # SHAP explanation
            st.subheader("Top words influencing prediction:")
            shap_values_input = explainer.shap_values(vectorized)
            feature_names = vectorizer.get_feature_names_out()
            word_contributions = dict(zip(feature_names, shap_values_input[0]))
            top_words = sorted(word_contributions.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
            for word, val in top_words:
                st.write(f"{word}: {round(val,3)}")

elif input_type == "Enter URL":
    url_input = st.text_input("Enter news URL:")
    if st.button("Predict URL"):
        if url_input.strip() != "":
            news_text = get_article_text(url_input)
            if news_text:
                clean = clean_text(news_text)
                vectorized = vectorizer.transform([clean])
                pred = model.predict(vectorized)[0]
                label = "FAKE" if pred == 1 else "REAL"
                st.subheader(f"Prediction: {label}")

                # SHAP explanation
                st.subheader("Top words influencing prediction:")
                shap_values_input = explainer.shap_values(vectorized)
                feature_names = vectorizer.get_feature_names_out()
                word_contributions = dict(zip(feature_names, shap_values_input[0]))
                top_words = sorted(word_contributions.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
                for word, val in top_words:
                    st.write(f"{word}: {round(val,3)}")
            else:
                st.error("Could not fetch article text from the URL.")
