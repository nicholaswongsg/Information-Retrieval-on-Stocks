import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import re
import sys
import os
from wordcloud import WordCloud
from collections import Counter
import nltk
from nltk.corpus import stopwords
import altair as alt
from datetime import datetime, timedelta

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Indexing.inverted_index_edited import search_index

# Define the data path correctly
data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Classification", "NER_with_sentiment.csv"))

# Set up page configuration
st.set_page_config(
    page_title="Stock Sentiment Analysis",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Stock Sentiment Analysis")

# === Tabs ===
tab1, tab2 = st.tabs(["About", "Inverted Index"])


# === Tab 1: About ===
with tab1:
    st.header("About This Application")
    st.markdown("""
    ### Stock Sentiment Analysis Tool
    
    This application uses an inverted index to search through stock-related text data and analyze sentiment.
    
    #### Features:
    - Search for stock-related information using natural language queries
    - View sentiment analysis from multiple models (FinBERT and VADER)
    - Visualize word frequencies and sentiment distributions
    - Analyze sentiment trends over time
    
    #### How It Works:
    1. Enter a query about stocks or companies
    2. The system searches through the documents
    3. Matching documents are retrieved and analyzed
    4. Results are displayed with various visualizations
    
    #### Data Sources (Reddit):
    - r/applestocks: https://www.reddit.com/r/applestocks
    - r/microsoft: https://www.reddit.com/r/microsoft
    - r/NVDA_Stock: https://www.reddit.com/r/NVDA_Stock
    - r/wallstreetbets: https://www.reddit.com/r/wallstreetbets
    - r/stockmarket: https://www.reddit.com/r/stockmarket
    - r/stocks: https://www.reddit.com/r/stocks
    """)

# === Tab 2: Inverted Index Sentiment Lookup ===
with tab2:
    st.header("Enter a company name or stock topic")   
        
    custom_query = st.text_input("Example: Apple Stocks, Tesla earnings", key="custom_query_csv")

    # Add a search button
    search_button = st.button("Search",key="search1")

    if custom_query and search_button:
        with st.spinner("Searching documents..."):
            try:
                # Start timing
                start_time = datetime.now()

                # Call the search_index function from inverted_index_edited.py
                stopwords_list, searched_docs = search_index(data_path, custom_query)
                end_time = datetime.now()
                elapsed_time = end_time - start_time
                elapsed_ms = elapsed_time.total_seconds() * 1000
                result_documents = searched_docs
                
                # Display results
                st.info(f"⏱️ Query processed in {elapsed_ms:.2f} ms ({elapsed_time.total_seconds():.2f} seconds)")
                st.write(f"🔍 Found {len(result_documents)} matching documents")
                
                if not result_documents.empty:
                    # === Aggregated Overview with Ordered Categories ===
                    st.subheader("📊 Aggregated Sentiment Overview")

                    sentiment_order = [
                        "strongly negative",
                        "slightly negative",
                        "neutral",
                        "slightly positive",
                        "strongly positive"
                    ]

                    # Display numerical counts instead of histogram
                    if 'primary_sentiment' in result_documents.columns:
                        # Round scores to 1 decimal place and count occurrences
                        score_counts = result_documents['primary_sentiment'].value_counts().reset_index()
                        score_counts.columns = ['Sentiment', 'Count']
                        
                        # Also display as a table for precise numbers
                        st.markdown("**Sentiment Score Counts**")
                        st.dataframe(score_counts, use_container_width=True)
            except Exception as e:
                st.error(f"Error searching documents: {str(e)}")