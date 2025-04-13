import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import pysolr
import re
import time
import sys
import os
from wordcloud import WordCloud
from collections import Counter
import nltk
from nltk.corpus import stopwords
import json
import altair as alt
from spellchecker import SpellChecker

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Indexing.two_stage import answer_stock_question  # Import the function from two_stage.py

SYNONYMS = {
    "appl": "AAPL",
    "appin": "AAPL",
    "aplin": "AAPL",
    "appln": "AAPL",
    "aapl": "AAPL",

    "nvda": "NVDA",
    "nvida": "NVDA",
    "nvid": "NVDA",

    "amzn": "AMZN",
    "amazon": "AMZN",

    "goog": "GOOGL",
    "googl": "GOOGL",
    "google": "GOOGL",

    "meta": "META",
    "fb": "META",

    "msft": "MSFT",
    "microsoft": "MSFT",

    "tsla": "TSLA",
    "tesla": "TSLA",

    "nflx": "NFLX",
    "netflix": "NFLX",
}

def load_domain_dictionary(csv_file_path, text_column='ner_text_cleaned'):
    df = pd.read_csv(csv_file_path)
    texts = df[text_column].dropna()

    all_words = []
    for line in texts:
        line_clean = re.sub(r'[^\w\s]', '', line.lower())
        words = line_clean.split()
        all_words.extend(words)

    return all_words

@st.cache_resource
def build_spellchecker(csv_file_path):
    spell = SpellChecker(distance=3)
    domain_words = load_domain_dictionary(csv_file_path)
    spell.word_frequency.load_words(domain_words)

    stock_names = ["apple", "tesla", "nvidia", "amazon", "google", "meta", "microsoft"]
    for name in stock_names:
        spell.word_frequency.add(name, 1000000)

    return spell

def correct_query(spell, user_query):
    query_clean = re.sub(r'[^\w\s]', '', user_query.lower())
    query_words = query_clean.split()

    corrected_words = []
    for w in query_words:
        if w in SYNONYMS:
            corrected_words.append(SYNONYMS[w])
        elif w in spell:
            corrected_words.append(w)
        else:
            correction = spell.correction(w)
            corrected_words.append(correction if correction else w)

    return " ".join(corrected_words)

NER_CSV_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Classification", "NER_with_sentiment.csv"))
spell = build_spellchecker(NER_CSV_PATH)

# Ensure stopwords are downloaded
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Stopwords configuration
STOPWORDS = set(stopwords.words('english'))
CUSTOM_STOPWORDS = {
    'like', 'just', 'will', 'get', 'got', 'going', 'think', 'know', 'really',
    'stock', 'stocks', 'market', 'share', 'shares', 'price', 'buy', 'sell',
    'trading', 'im', 'ive', 'youre', 'dont', 'didnt', 'cant', 'wont', 'thats',
    'would', 'could', 'should', 'http', 'https', 'www', 'com', 'year', 'day',
    'one', 'two', 'three', 'first', 'second', 'last', 'next', 'even', 'much',
    'many', 'also', 'edit', 'deleted', 'removed'
}
ALL_STOPWORDS = STOPWORDS.union(CUSTOM_STOPWORDS)

# Solr Setup
SOLR_URL = "http://localhost:8983/solr/reddit_core"
solr = pysolr.Solr(SOLR_URL, always_commit=True, timeout=10)

st.title("📈 Stock Sentiment Analysis")

# === Tabs ===
tab1, tab2 = st.tabs(["Current Process", "Two-Stage Process"])

# === Tab 1: CSV-Based Sentiment Lookup ===
with tab1:
    st.header("Ask a Question (from NER Sentiment File)")
    custom_query = st.text_input("Example: What is the sentiment of Apple?", key="custom_query_csv")

    if custom_query:
        corrected_query = correct_query(spell, custom_query)
        if corrected_query.lower() != custom_query.lower():
            st.caption(f"🔧 Auto-corrected to: _{corrected_query}_")

        cleaned = re.sub(r'[^\w\s]', '', corrected_query.lower())
        tokens = cleaned.split()
        ignore_words = {"what", "is", "the", "of", "sentiment", "tell", "me", "about"}
        keywords = [w for w in tokens if w not in ignore_words]

        if keywords:
            stock_term = keywords[-1].upper()

            # Load and filter dataset
            csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Classification", "NER_with_sentiment.csv"))
            df = pd.read_csv(csv_path)

            # Find rows where stock_term exists in JSON key
            mask = df['ner_entity_sentiments'].astype(str).str.contains(stock_term)
            matched = df[mask].copy()

            if matched.empty:
                st.warning(f"No sentiment data found for entity: {stock_term}")
            else:
                st.write(f"🔍 Found {len(matched)} entries related to **{stock_term}**")

                # Parse and extract sentiment info
                parsed_rows = []
                for _, row in matched.iterrows():
                    try:
                        sentiment_data = json.loads(row['ner_entity_sentiments'])
                        if stock_term in sentiment_data:
                            finbert = sentiment_data[stock_term]['finbert']
                            vader = sentiment_data[stock_term]['vader']
                            parsed_rows.append({
                                'text': row['text'],
                                'finbert_label': finbert['label'],
                                'finbert_score': finbert['net_score'],
                                'vader_label': vader['label'],
                                'vader_score': vader['compound']
                            })
                    except Exception as e:
                        print(f"JSON parsing error: {e}")

                result_df = pd.DataFrame(parsed_rows)

                # Display table
                st.dataframe(result_df)

                # === Word Cloud ===
                st.subheader("🌥 Word Cloud from Text")
                combined_text = " ".join(result_df['text'].dropna().astype(str))
                combined_text = re.sub(r'https?://\S+|www\.\S+', '', combined_text)
                combined_text = re.sub(r'[^\w\s]', '', combined_text)
                combined_text = combined_text.lower()
                words = [w for w in combined_text.split() if w not in ALL_STOPWORDS and len(w) > 2]

                if words:
                    word_freq = Counter(words)
                    wordcloud = WordCloud(
                        width=800, height=400, background_color='white',
                        max_words=100, colormap='plasma'
                    ).generate_from_frequencies(word_freq)

                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.imshow(wordcloud, interpolation='bilinear')
                    ax.axis("off")
                    st.pyplot(fig)
                else:
                    st.info("Not enough content to generate a word cloud.")

                # === Aggregated Overview with Ordered Categories ===
                st.subheader("📊 Aggregated Sentiment Overview")

                sentiment_order = [
                    "Strongly Negative",
                    "Slightly Negative",
                    "Neutral",
                    "Slightly Positive",
                    "Strongly Positive"
                ]

                col1, col2 = st.columns(2)

            with col1:
                st.markdown("**FinBERT Sentiment Distribution**")
                finbert_counts = result_df['finbert_label'].value_counts().reindex(sentiment_order).fillna(0).reset_index()
                finbert_counts.columns = ['Sentiment', 'Count']

                finbert_chart = alt.Chart(finbert_counts).mark_bar(color='lightblue').encode(
                    x=alt.X('Sentiment', sort=sentiment_order),
                    y='Count'
                ).properties(height=300)

                st.altair_chart(finbert_chart, use_container_width=True)

            with col2:
                st.markdown("**VADER Sentiment Distribution**")
                vader_counts = result_df['vader_label'].value_counts().reindex(sentiment_order).fillna(0).reset_index()
                vader_counts.columns = ['Sentiment', 'Count']

                vader_chart = alt.Chart(vader_counts).mark_bar(color='lightblue').encode(
                    x=alt.X('Sentiment', sort=sentiment_order),
                    y='Count'
                ).properties(height=300)

                st.altair_chart(vader_chart, use_container_width=True)
            
            # === Smoothed FinBERT Sentiment Score Over Time (LOESS) ===
            st.subheader("📈 Smoothed FinBERT Sentiment Score Over Time")

            if 'created_utc' in df.columns and 'finbert_score' in result_df.columns:
                # ✅ Convert created_utc from UNIX timestamp to datetime
                df['created_utc'] = pd.to_datetime(df['created_utc'], unit='s', errors='coerce')

                # Combine datetime and scores
                time_df = pd.concat([
                    df['created_utc'].reset_index(drop=True),
                    result_df['finbert_score'].reset_index(drop=True)
                ], axis=1).dropna()

                # Floor to daily granularity
                time_df['date'] = time_df['created_utc'].dt.floor('D')

                # Group by date and compute average score
                agg_df = time_df.groupby('date')['finbert_score'].mean().reset_index()

                # Plot smoothed LOESS chart
                chart = alt.Chart(agg_df).transform_loess(
                    'date', 'finbert_score', bandwidth=0.4
                ).mark_line(color='deepskyblue').encode(
                    x=alt.X('date:T', title='Date'),
                    y=alt.Y('finbert_score:Q', title='Smoothed Sentiment Score'),
                    tooltip=['date:T', 'finbert_score:Q']
                ).properties(height=300).interactive()

                st.altair_chart(chart, use_container_width=True)

            else:
                st.info("Date or sentiment score data is missing.")
            # === Smoothed VADER Sentiment Score Over Time (LOESS) ===
            st.subheader("📉 Smoothed VADER Sentiment Score Over Time")

            if 'created_utc' in df.columns and 'vader_score' in result_df.columns:
                # Convert created_utc to datetime
                df['created_utc'] = pd.to_datetime(df['created_utc'], unit='s', errors='coerce')

                # Combine datetime and VADER scores
                time_df = pd.concat([
                    df['created_utc'].reset_index(drop=True),
                    result_df['vader_score'].reset_index(drop=True)
                ], axis=1).dropna()

                # Floor to daily granularity
                time_df['date'] = time_df['created_utc'].dt.floor('D')

                # Aggregate by date
                agg_df = time_df.groupby('date')['vader_score'].mean().reset_index()

                # Plot LOESS chart
                chart = alt.Chart(agg_df).transform_loess(
                    'date', 'vader_score', bandwidth=0.4
                ).mark_line(color='orange').encode(
                    x=alt.X('date:T', title='Date'),
                    y=alt.Y('vader_score:Q', title='Smoothed Sentiment Score'),
                    tooltip=['date:T', 'vader_score:Q']
                ).properties(height=300).interactive()

                st.altair_chart(chart, use_container_width=True)

            else:
                st.info("Date or VADER sentiment score data is missing.")

            
        else:
            st.warning("Could not extract any stock-related keyword from your question.")
            


# === Tab 2: Two-Stage Process ===
with tab2:
    st.header("Ask a Question (Two-Stage Process)")
    two_stage_query = st.text_input("Example: What is the sentiment of Tesla?", key="two_stage_query")

    if two_stage_query:
        st.write("Processing your query using the two-stage process...")
        answer,relevant_chunks = answer_stock_question(two_stage_query)
        st.subheader("Answer:")
        st.write(answer)
        st.subheader("Answer:")
        st.write(relevant_chunks)