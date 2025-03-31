import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import pysolr
import re
import time
from wordcloud import WordCloud

# Solr Setup
SOLR_URL = "http://localhost:8983/solr/reddit_core"
solr = pysolr.Solr(SOLR_URL, always_commit=True, timeout=10)

st.title("📈 Stock Sentiment Analysis")

# === Sample Queries ===
with st.expander("💡 Example queries you can try"):
    st.markdown("""
    - What is the sentiment of Apple?
    - How does the sentiment change over a period of 1 year?
    - Is there a difference in sentiment between subreddits?
    - What are users saying about Nvidia earnings?
    - Which user has the most upvoted comments about Tesla?
    """)

# === Free Text Query Section ===
st.header("🧠 Ask a Question")
custom_query = st.text_input("Example: What is the sentiment of Apple?", key="custom_query")

if custom_query:
    cleaned = re.sub(r'[^\w\s]', '', custom_query.lower())
    tokens = cleaned.split()
    ignore_words = {"what", "is", "the", "of", "sentiment", "tell", "me", "about"}
    keywords = [w for w in tokens if w not in ignore_words]

    if keywords:
        stock_term = keywords[-1]
        st.write(f"🧠 Searching for sentiment related to: **{stock_term}**")

        # Start timing
        start_time = time.time()
        results = solr.search("*:*", fq=[f'text:"{stock_term}"'], rows=1000)
        query_duration = round(time.time() - start_time, 3)

        st.write(f"⏱️ Query took {query_duration} seconds")

        # Flatten Solr docs
        flat_docs = [
            {k: (v[0] if isinstance(v, list) and len(v) == 1 else v) for k, v in doc.items()}
            for doc in results
        ]
        df = pd.DataFrame(flat_docs)

        if df.empty:
            st.warning("No results found.")
        else:
            # Parse date
            if "created_utc" in df.columns:
                df["created_utc"] = pd.to_datetime(df["created_utc"], unit="s", errors="coerce")

            # Sort by score
            if "score" in df.columns:
                df["score"] = pd.to_numeric(df["score"], errors="coerce")
                df = df.sort_values(by="score", ascending=False)

            # Display selected fields
            display_cols = ["author", "text", "score", "created_utc", "subreddit", "title"]
            display_cols = [col for col in display_cols if col in df.columns]
            df_display = df[display_cols].copy()

            for col in df_display.columns:
                if df_display[col].dtype == object:
                    df_display[col] = df_display[col].fillna("N/A")

            st.subheader("🔍 Results for your query:")
            st.dataframe(df_display)

            # === Word Cloud ===
            st.header("🌥 Word Cloud from Results")
            text_fields = []
            if "text" in df.columns:
                text_fields += df["text"].dropna().astype(str).tolist()
            if "title" in df.columns:
                text_fields += df["title"].dropna().astype(str).tolist()

            text_data = " ".join(text_fields)
            if text_data.strip():
                wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text_data)
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis("off")
                st.pyplot(fig)
            else:
                st.info("⚠️ Not enough valid text to generate a word cloud.")

            # === Top Influential Users ===
            if "author" in df.columns and "score" in df.columns:
                st.header("🏆 Top Influential Reddit Users")
                user_ranking = df.groupby('author')['score'].sum().sort_values(ascending=False).head(10).reset_index()
                user_ranking.columns = ['User', 'Total Upvotes']
                st.dataframe(user_ranking)

            st.success("✅ Query completed.")
    else:
        st.warning("Could not extract any stock-related keyword from your question.")
