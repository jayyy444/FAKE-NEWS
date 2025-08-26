import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import yfinance as yf
import streamlit as st
from transformers import pipeline

@st.cache_resource
def get_sentiment_model():
    return pipeline("sentiment-analysis", model="prajjwal1/bert-tiny")

sentiment_model = get_sentiment_model()

# Load HuggingFace sentiment model once (cached for reuse)
 
@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    if "Top1" in df.columns:
        df.rename(columns={'Top1': 'Headline'}, inplace=True)
    df['Headline'] = df['Headline'].astype(str)
    df.dropna(subset=['Headline'], inplace=True)
    return df

def compute_sentiment(df):
    """
    Apply transformer sentiment analysis to the dataset headlines.
    Adds 'sentiment' column with polarity score.
    """
    try:
        results = sentiment_model(df['Headline'].tolist(), batch_size=32, truncation=True)
        sentiments = [(r['score'] if r['label'] == "POSITIVE" else -r['score']) for r in results]
        df['sentiment'] = sentiments
    except Exception:
        df['sentiment'] = 0.0
    return df
    

def compute_similarity(df, fake_headline):
    headlines = df['Headline'].tolist()
    all_text = headlines + [fake_headline]

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(all_text)

    sim_scores = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
    df['similarity'] = sim_scores[0]
    return df.sort_values(by='similarity', ascending=False)

def get_stock_data(ticker="TSLA", period="6mo"):
    """
    Fetch historical stock data using yfinance.
    Example tickers:
      TSLA -> Tesla
      AAPL -> Apple
      INFY.NS -> Infosys (NSE India)
      RELIANCE.NS -> Reliance Industries (NSE India)
    """
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)
        if df.empty:
            return None
        df.reset_index(inplace=True)
        return df
    except Exception as e:
        print(f"Error fetching stock data: {e}")
        return None
