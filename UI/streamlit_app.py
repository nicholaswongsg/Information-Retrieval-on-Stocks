import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import pysolr
import re
import time
from wordcloud import WordCloud
from collections import Counter
import nltk
from nltk.corpus import stopwords

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

# === Input Section ===
st.header("Ask a Question")
custom_query = st.text_input("Example: What is the sentiment of Apple?", key="custom_query")

# === Filters ===
st.sidebar.header("Filters")
available_subreddits = [
    "wallstreetbets", "investing", "stockmarket",
    "stocksandtrading", "stocks", "applestocks", "microsoft", "NVDA_Stock"
]
selected_subreddits = st.sidebar.multiselect("Subreddits:", available_subreddits)
min_score = st.sidebar.number_input("Minimum Score:", min_value=0, value=0)

# === Query Execution ===
if custom_query:
    cleaned = re.sub(r'[^\w\s]', '', custom_query.lower())
    tokens = cleaned.split()
    ignore_words = {"what", "is", "the", "of", "sentiment", "tell", "me", "about"}
    keywords = [w for w in tokens if w not in ignore_words]

    if keywords:
        stock_term = keywords[-1]
        st.write(f"Searching for sentiment related to: **{stock_term}**")

        fq = [f'text:"{stock_term}"']
        if selected_subreddits:
            subreddit_filter = " OR ".join([f'subreddit:"{s}"' for s in selected_subreddits])
            fq.append(f"({subreddit_filter})")
        if min_score > 0:
            fq.append(f"score:[{min_score} TO *]")

        start_time = time.time()
        results = solr.search("*:*", fq=fq, rows=1000)
        query_duration = round(time.time() - start_time, 3)

        st.write(f"⏱️ Query took {query_duration} seconds")

        flat_docs = [
            {k: (v[0] if isinstance(v, list) and len(v) == 1 else v) for k, v in doc.items()}
            for doc in results
        ]
        df = pd.DataFrame(flat_docs)

        if df.empty:
            st.warning("No results found.")
        else:
            if "created_utc" in df.columns:
                df["created_utc"] = pd.to_datetime(df["created_utc"], unit="s", errors="coerce")

            display_cols = ["author", "text", "score", "created_utc", "subreddit", "title"]
            display_cols = [col for col in display_cols if col in df.columns]

            if "score" in df.columns:
                df["score"] = pd.to_numeric(df["score"], errors="coerce")
                df = df.sort_values(by="score", ascending=False)

            st.subheader("Results for your query:")
            st.dataframe(df[display_cols].fillna(""))

            # === Word Cloud ===
            st.header("🌥 Word Cloud from Results")
            text_data = []
            for _, row in df.iterrows():
                if "text" in row and pd.notnull(row["text"]):
                    text_data.append(str(row["text"]))
                if "title" in row and pd.notnull(row["title"]):
                    text_data.append(str(row["title"]))

            combined_text = " ".join(text_data)
            combined_text = re.sub(r'https?://\S+|www\.\S+', '', combined_text)
            combined_text = re.sub(r'[^\w\s]', '', combined_text)
            combined_text = combined_text.lower()

            words = combined_text.split()
            words = [w for w in words if w not in ALL_STOPWORDS and len(w) > 2]

            if words:
                word_freq = Counter(words)
                wordcloud = WordCloud(
                    width=800,
                    height=400,
                    background_color='white',
                    max_words=100,
                    colormap='plasma'
                ).generate_from_frequencies(word_freq)

                fig, ax = plt.subplots(figsize=(10, 5))
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis("off")
                st.pyplot(fig)
            else:
                st.info("Not enough content to generate a word cloud.")

            # === Top Users ===
            st.header("🏆 Top Influential Reddit Users")
            user_ranking = df.groupby('author')['score'].sum().sort_values(ascending=False).head(10).reset_index()
            user_ranking.columns = ['User', 'Total Upvotes']
            st.dataframe(user_ranking)

            st.success("✅ Query completed.")
    else:
        st.warning("Could not extract any stock-related keyword from your question.")
