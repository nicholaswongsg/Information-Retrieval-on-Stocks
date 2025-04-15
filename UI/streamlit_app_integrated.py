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

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Indexing.inverted_index_edited import search_index
from Indexing.two_stage import answer_stock_question  # Import the function from two_stage.py

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
tab1, tab2,tab3 = st.tabs(["About","Search & Analysis","Retrieval-Augmented Generation"])


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
    - r/investing: https://www.reddit.com/r/investing
    - r/stocksandtrading: https://www.reddit.com/r/stocksandtrading
    """)

# === Tab 2: Inverted Index Sentiment Lookup ===
with tab2:
    st.header("Ask a Question About Stock Sentiment")
    custom_query = st.text_input("Example: What is the sentiment of Apple?", key="custom_query_csv")

    # Add a search button
    search_button = st.button("Search")

    if custom_query and search_button:
        with st.spinner("Searching documents..."):
            try:
                # Call the search_index function from inverted_index_edited.py
                stopwords_list, result_documents = search_index(data_path, custom_query)
                
                # Display results
                st.write(f"🔍 Found {len(result_documents)} matching documents")
                
                if not result_documents.empty:
                    # Display table with limited columns for better readability
                    if 'title' in result_documents.columns and 'date' in result_documents.columns:
                        display_df = result_documents[['title', 'date']].copy()
                        st.dataframe(display_df)
                    else:
                        st.dataframe(result_documents)

                    # Word cloud from results
                    st.subheader("🌥 Word Cloud from Search Results")
                    if 'text' in result_documents.columns and len(result_documents) > 0:
                        combined_text = " ".join(result_documents['text'].dropna().astype(str))
                        combined_text = re.sub(r'https?://\S+|www\.\S+', '', combined_text)
                        combined_text = re.sub(r'[^\w\s]', '', combined_text)
                        combined_text = combined_text.lower()
                        
                        # Get stopwords from nltk if not already downloaded
                        try:
                            stop_words = set(stopwords.words('english'))
                        except:
                            nltk.download('stopwords')
                            stop_words = set(stopwords.words('english'))
                        
                        # Add the custom stopwords to the NLTK stopwords
                        stop_words.update(stopwords_list)
                        
                        words = [w for w in combined_text.split() if w not in stop_words and len(w) > 2]
                        
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
                    else:
                        st.warning("Text column not found in results or no results available.")

                    # === Aggregated Overview with Ordered Categories ===
                    st.subheader("📊 Aggregated Sentiment Overview")

                    sentiment_order = [
                        "Strongly Negative",
                        "Slightly Negative",
                        "Neutral",
                        "Slightly Positive",
                        "Strongly Positive"
                    ]

                    # Check if the required columns exist in the dataframe
                    if 'finbert_label' in result_documents.columns and 'vader_label' in result_documents.columns:
                        col1, col2 = st.columns(2)

                        with col1:
                            st.markdown("**FinBERT Sentiment Distribution**")
                            finbert_counts = result_documents['finbert_label'].value_counts().reindex(sentiment_order).fillna(0).reset_index()
                            finbert_counts.columns = ['Sentiment', 'Count']

                            finbert_chart = alt.Chart(finbert_counts).mark_bar(color='lightblue').encode(
                                x=alt.X('Sentiment', sort=sentiment_order),
                                y='Count'
                            ).properties(height=300)

                            st.altair_chart(finbert_chart, use_container_width=True)

                        with col2:
                            st.markdown("**VADER Sentiment Distribution**")
                            vader_counts = result_documents['vader_label'].value_counts().reindex(sentiment_order).fillna(0).reset_index()
                            vader_counts.columns = ['Sentiment', 'Count']

                            vader_chart = alt.Chart(vader_counts).mark_bar(color='lightblue').encode(
                                x=alt.X('Sentiment', sort=sentiment_order),
                                y='Count'
                            ).properties(height=300)

                            st.altair_chart(vader_chart, use_container_width=True)
                            
                        # Add a time-based analysis if date column exists
                        if 'date' in result_documents.columns:
                            st.subheader("📅 Sentiment Over Time")
                            try:
                                # Convert date to datetime if it's not already
                                if not pd.api.types.is_datetime64_any_dtype(result_documents['date']):
                                    result_documents['date'] = pd.to_datetime(result_documents['date'])
                                
                                # Group by date and count sentiments
                                time_data = result_documents.groupby([pd.Grouper(key='date', freq='D'), 'finbert_label']).size().reset_index(name='count')
                                
                                # Create time series chart
                                time_chart = alt.Chart(time_data).mark_line().encode(
                                    x='date:T',
                                    y='count:Q',
                                    color='finbert_label:N',
                                    tooltip=['date', 'finbert_label', 'count']
                                ).properties(
                                    width=700,
                                    height=400
                                )
                                
                                st.altair_chart(time_chart, use_container_width=True)
                            except Exception as e:
                                st.warning(f"Could not create time series chart: {str(e)}")
                    else:
                        st.warning("Sentiment label columns not found in the result data.")
                else:
                    st.info("No matching documents found for your query.")
            except Exception as e:
                st.error(f"Error searching documents: {str(e)}")

# === Tab 3: Two-Stage Process ===
with tab3:
    st.header("Ask a Question (Two-Stage Process)")
    two_stage_query = st.text_input("Example: What is the sentiment of Tesla?", key="two_stage_query")

    if two_stage_query:
        st.write("Processing your query using the two-stage process...")
        answer,relevant_chunks = answer_stock_question(two_stage_query)
        st.subheader("Answer:")
        st.write(answer)
        st.subheader("Relevant Chunks:")
        st.write(relevant_chunks)