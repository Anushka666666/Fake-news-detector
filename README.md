# Fake-news-detector
A Machine Learning + NLP project that classifies news articles as FAKE or REAL and explains predictions using SHAP. The app is built with Streamlit for a simple interactive demo, accepts raw text or a URL, and shows the top words that pushed the decision.

Features:
- Paste text or enter a URL of a news article.
- Predicts whether the article is REAL or FAKE.
- Shows important words influencing prediction using SHAP.
- Simple and interactive "Streamlit web app".

 🗂️ Dataset:
We used the Kaggle Fake News Dataset, which contains:
- True.csv: Verified real news articles
- Fake.csv: Fake/misleading news articles

Each file has:
- title: headline of the news
- text: full article body
- subject: category/topic of news
- date: publishing date

For this project, we mainly used the "text" field.

🛠️ Tech Stack:
- Python
- pandas, scikit-learn, nltk : Data cleaning & ML model
- TF-IDF + Logistic Regression: Fake news classification
- SHAP: Explainability 
- BeautifulSoup4 + requests: Extract news content from URLs
- Streamlit: Web app frontend

HOW TO RUM:
1. git clone: https://github.com/Anushka666666/Fake-news-detector

2. Install dependencies
   pip install -r requirements.txt

3.Run the app
   streamlit run fake.py

4.Open in browser at
    http://localhost:8501

