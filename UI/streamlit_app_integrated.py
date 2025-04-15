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
import pysolr

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Indexing.inverted_index_edited import search_index
from Indexing.two_stage import answer_stock_question  # Import the function from two_stage.py

# Define the data path correctly
data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Classification", "NER_with_ensemble_sentiment.csv"))

# Set up Solr
solr = pysolr.Solr('http://localhost:8983/solr/reddit_core', always_commit=True, timeout=10)

# Set up page configuration
st.set_page_config(
    page_title="Stock Sentiment Analysis",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Stock Sentiment Analysis")

# === Tabs ===
tab1, tab2,tab3,tab4 = st.tabs(["About","Solr","Inverted Index","Retrieval-Augmented Generation"])


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

# === Tab 2: Solr Sentiment Lookup ===
with tab2:
    st.header("Enter a company name or stock topic")
    # Create filter section at the top
    with st.container():      
        # Create two columns for filters
        col1, col2 = st.columns(2)
        
        # Time filter in first column
        with col1:
            st.markdown("**Time Range**")
            
            # Define the min and max dates from your data
            min_date_utc = 1608636755
            max_date_utc = 1740806007 
            
            # Convert UTC timestamps to datetime for display
            min_date = datetime.fromtimestamp(min_date_utc)
            max_date = datetime.fromtimestamp(max_date_utc)
            
            # Create date input widgets in a row
            date_col1, date_col2 = st.columns(2)
            with date_col1:
                start_date = st.date_input("Start Date", min_date, min_value=min_date, max_value=max_date,key="solr_start_date")
            with date_col2:
                end_date = st.date_input("End Date", max_date, min_value=min_date, max_value=max_date,key="solr_end_date")
            
            # Convert selected dates to UTC timestamps for filtering
            start_datetime = datetime.combine(start_date, datetime.min.time())
            end_datetime = datetime.combine(end_date, datetime.max.time())
            
            start_utc = int(start_datetime.timestamp())
            end_utc = int(end_datetime.timestamp())
        
        # Subreddit filter in second column
        with col2:
            st.markdown("**Subreddit (Choose none for all subreddits)**")
            
            # List of available subreddits as specified
            available_subreddits = ["applestocks", "microsoft", "NVDA_Stock", 
                                    "wallstreetbets", "stockmarket", "stocks"]
            
            # Create multiselect widget for subreddits
            selected_subreddits = st.multiselect(
                "Select Subreddits",
                options=available_subreddits,
                default=[],  # Default selection
                key = "solr_subs"
            )
        
    solr_query = st.text_input("Example: Apple Stocks, Tesla earnings", key="solr_query_csv")

    # Add a search button
    search_button = st.button("Search",key="search1")

    if solr_query or search_button:
        with st.spinner("Searching documents..."):
            try:
                clauses = []
                clauses.append(f"(title:({solr_query}) OR selftext:({solr_query}) OR text:({solr_query}))")
                # Handle multiple subreddit selections
                if selected_subreddits:
                    # Create a subreddit clause that matches any of the selected subreddits
                    subreddit_terms = " OR ".join([f"subreddit:{s}" for s in selected_subreddits])
                    clauses.append(f"({subreddit_terms})")

                    clauses.append(f"(created_utc:[{start_utc} TO {end_utc}] OR comment_created_utc:[{start_utc} TO {end_utc}])")
                # If no fields were specified, default to all documents
                full_query = " AND ".join(clauses)
                result_solr = solr.search(full_query, q_op='OR', rows=2147483647)

                # Display results
                st.write(f"🔍 Found {len(result_solr)} matching documents")

                result_documents = pd.DataFrame(result_solr)
                for column in result_documents.columns:
                    # Check if the column contains lists
                    if result_documents[column].apply(lambda x: isinstance(x, list)).any():
                        # Extract the single value from each list
                        result_documents[column] = result_documents[column].apply(lambda x: x[0] if isinstance(x, list) and len(x) == 1 else x)
                if not result_documents.empty:
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
                        "strongly negative",
                        "slightly negative",
                        "neutral",
                        "slightly positive",
                        "strongly positive"
                    ]

                    # Check if the required columns exist in the dataframe
                    if 'primary_sentiment' in result_documents.columns:
                        # Create visualization for primary sentiment
                        st.markdown("**Primary Sentiment Distribution**")
                        primary_counts = result_documents['primary_sentiment'].value_counts().reindex(sentiment_order).fillna(0).reset_index()
                        primary_counts.columns = ['Sentiment', 'Count']
                        
                        primary_chart = alt.Chart(primary_counts).mark_bar(color='lightblue').encode(
                            x=alt.X('Sentiment', sort=sentiment_order),
                            y='Count'
                        ).properties(height=300)
                        
                        st.altair_chart(primary_chart, use_container_width=True)
                        
                        # Score distribution as well
                        if 'primary_score' in result_documents.columns:
                            st.markdown("**Sentiment Score Distribution**")
                            
                            # Create a histogram of sentiment scores
                            score_hist = alt.Chart(result_documents).mark_bar().encode(
                                x=alt.X('primary_score:Q', bin=alt.Bin(maxbins=20), title='Sentiment Score'),
                                y='count()'
                            ).properties(height=300)
                            
                            st.altair_chart(score_hist, use_container_width=True)
                        
                        # Add a time-based analysis
                        if 'created_utc' in result_documents.columns:
                            st.subheader("📅 Sentiment Over Time")
                            try:
                                # Convert timestamp to datetime if it's not already
                                if not pd.api.types.is_datetime64_any_dtype(result_documents['created_utc']):
                                    result_documents['date'] = pd.to_datetime(result_documents['created_utc'], unit='s')
                                
                                # Group by date and sentiment
                                time_data = result_documents.groupby([pd.Grouper(key='date', freq='D'), 'primary_sentiment']).size().reset_index(name='count')
                                
                                # Create time series chart
                                time_chart = alt.Chart(time_data).mark_line().encode(
                                    x='date:T',
                                    y='count:Q',
                                    color='primary_sentiment:N',
                                    tooltip=['date', 'primary_sentiment', 'count']
                                ).properties(
                                    width=700,
                                    height=400
                                )
                                
                                st.altair_chart(time_chart, use_container_width=True)
                            except Exception as e:
                                st.warning(f"Could not create time series chart: {str(e)}")
                            
                            # Parse the JSON in ner_entity_sentiments to extract FinBERT and VADER sentiments
                            if 'ner_entity_sentiments' in result_documents.columns:
                                st.subheader("**Model Specific Sentiment Analysis**")
                                
                                finbert_sentiments = []
                                vader_sentiments = []
                                finbert_scores = []
                                vader_scores = []
                                
                                for json_str in result_documents['ner_entity_sentiments']:
                                    try:
                                        import json
                                        data = json.loads(json_str.replace("'", '"'))
                                        
                                        # Extract first ticker's sentiment (simplified)
                                        for ticker, analyses in data.items():
                                            if 'finbert' in analyses:
                                                finbert_sentiments.append(analyses['finbert']['label'].lower())
                                                finbert_scores.append(analyses['finbert']['net_score'])
                                            if 'vader' in analyses:
                                                vader_sentiments.append(analyses['vader']['label'].lower())
                                                vader_scores.append(analyses['vader']['compound'])  # VADER uses 'compound' instead of 'net_score'
                                            break  # Just take the first ticker for simplicity
                                    except:
                                        continue
                                
                                # Create DataFrames for visualization
                                if finbert_sentiments:
                                    # Create two columns for sentiment distribution and score histogram
                                    col1, col2 = st.columns(2)
                                    
                                    with col1:
                                        finbert_df = pd.DataFrame({'Sentiment': finbert_sentiments})
                                        finbert_counts = finbert_df['Sentiment'].value_counts().reindex(sentiment_order).fillna(0).reset_index()
                                        finbert_counts.columns = ['Sentiment', 'Count']
                                        
                                        finbert_chart = alt.Chart(finbert_counts).mark_bar(color='lightblue').encode(
                                            x=alt.X('Sentiment', sort=sentiment_order),
                                            y='Count'
                                        ).properties(height=300)
                                        
                                        st.markdown("**FinBERT Sentiment Distribution**")
                                        st.altair_chart(finbert_chart, use_container_width=True)
                                    
                                    with col2:
                                        # Create histogram for FinBERT scores
                                        finbert_score_df = pd.DataFrame({'Score': finbert_scores})
                                        
                                        finbert_hist = alt.Chart(finbert_score_df).mark_bar(color='lightgreen').encode(
                                            x=alt.X('Score:Q', bin=alt.Bin(maxbins=20), title='FinBERT Net Score'),
                                            y='count()'
                                        ).properties(height=300)
                                        
                                        st.markdown("**FinBERT Score Distribution**")
                                        st.altair_chart(finbert_hist, use_container_width=True)
                                
                                if vader_sentiments:
                                    # Create two columns for sentiment distribution and score histogram
                                    col1, col2 = st.columns(2)
                                    
                                    with col1:
                                        vader_df = pd.DataFrame({'Sentiment': vader_sentiments})
                                        vader_counts = vader_df['Sentiment'].value_counts().reindex(sentiment_order).fillna(0).reset_index()
                                        vader_counts.columns = ['Sentiment', 'Count']
                                        
                                        vader_chart = alt.Chart(vader_counts).mark_bar(color='lightblue').encode(
                                            x=alt.X('Sentiment', sort=sentiment_order),
                                            y='Count'
                                        ).properties(height=300)
                                        
                                        st.markdown("**VADER Sentiment Distribution**")
                                        st.altair_chart(vader_chart, use_container_width=True)
                                    
                                    with col2:
                                        # Create histogram for VADER scores
                                        vader_score_df = pd.DataFrame({'Score': vader_scores})
                                        
                                        vader_hist = alt.Chart(vader_score_df).mark_bar(color='lightgreen').encode(
                                            x=alt.X('Score:Q', bin=alt.Bin(maxbins=20), title='VADER Compound Score'),
                                            y='count()'
                                        ).properties(height=300)
                                        
                                        st.markdown("**VADER Score Distribution**")
                                        st.altair_chart(vader_hist, use_container_width=True)
                else:
                    st.info("No matching documents found for your query.")
            except Exception as e:
                st.error(f"Error searching documents: {str(e)}")

# === Tab 3: Inverted Index Sentiment Lookup ===
with tab3:
    st.header("Enter a company name or stock topic")
    # Create filter section at the top
    with st.container():      
        # Create two columns for filters
        col1, col2 = st.columns(2)
        
        # Time filter in first column
        with col1:
            st.markdown("**Time Range**")
            
            # Define the min and max dates from your data
            min_date_utc = 1608636755
            max_date_utc = 1740806007 
            
            # Convert UTC timestamps to datetime for display
            min_date = datetime.fromtimestamp(min_date_utc)
            max_date = datetime.fromtimestamp(max_date_utc)
            
            # Create date input widgets in a row
            date_col1, date_col2 = st.columns(2)
            with date_col1:
                start_date = st.date_input("Start Date", min_date, min_value=min_date, max_value=max_date,key="inverted_start_date")
            with date_col2:
                end_date = st.date_input("End Date", max_date, min_value=min_date, max_value=max_date,key="inverted_end_date")
            
            # Convert selected dates to UTC timestamps for filtering
            start_datetime = datetime.combine(start_date, datetime.min.time())
            end_datetime = datetime.combine(end_date, datetime.max.time())
            
            start_utc = int(start_datetime.timestamp())
            end_utc = int(end_datetime.timestamp())
        
        # Subreddit filter in second column
        with col2:
            st.markdown("**Subreddit (Choose none for all subreddits)**")
            
            # List of available subreddits as specified
            available_subreddits = ["applestocks", "microsoft", "NVDA_Stock", 
                                    "wallstreetbets", "stockmarket", "stocks"]
            
            # Create multiselect widget for subreddits
            selected_subreddits = st.multiselect(
                "Select Subreddits",
                options=available_subreddits,
                default=[],  # Default selection
                key = "inverted_subs"
            )
        
    custom_query = st.text_input("Example: Apple Stocks, Tesla earnings", key="custom_query_csv")

    # Add a search button
    search_button = st.button("Search",key="search2")

    if custom_query or search_button:
        with st.spinner("Searching documents..."):
            try:
                # Call the search_index function from inverted_index_edited.py
                stopwords_list, searched_docs = search_index(data_path, custom_query)

                #1. Filter by time
                result_documents = searched_docs[(searched_docs['created_utc'] >= start_utc) & (searched_docs['created_utc'] <= end_utc)]

                # 2. Filter by selected subreddits if any are selected
                if selected_subreddits:
                    result_documents = result_documents[result_documents['subreddit'].isin(selected_subreddits)]
                else:
                    # If no subreddits are selected, keep all subreddits
                    pass

                # Display results
                st.write(f"🔍 Found {len(result_documents)} matching documents")
                
                if not result_documents.empty:
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
                        "strongly negative",
                        "slightly negative",
                        "neutral",
                        "slightly positive",
                        "strongly positive"
                    ]

                    # Check if the required columns exist in the dataframe
                    if 'primary_sentiment' in result_documents.columns:
                        # Create visualization for primary sentiment
                        st.markdown("**Primary Sentiment Distribution**")
                        primary_counts = result_documents['primary_sentiment'].value_counts().reindex(sentiment_order).fillna(0).reset_index()
                        primary_counts.columns = ['Sentiment', 'Count']
                        
                        primary_chart = alt.Chart(primary_counts).mark_bar(color='lightblue').encode(
                            x=alt.X('Sentiment', sort=sentiment_order),
                            y='Count'
                        ).properties(height=300)
                        
                        st.altair_chart(primary_chart, use_container_width=True)
                        
                        # Score distribution as well
                        if 'primary_score' in result_documents.columns:
                            st.markdown("**Sentiment Score Distribution**")
                            
                            # Create a histogram of sentiment scores
                            score_hist = alt.Chart(result_documents).mark_bar().encode(
                                x=alt.X('primary_score:Q', bin=alt.Bin(maxbins=20), title='Sentiment Score'),
                                y='count()'
                            ).properties(height=300)
                            
                            st.altair_chart(score_hist, use_container_width=True)
                        
                        # Add a time-based analysis
                        if 'created_utc' in result_documents.columns:
                            st.subheader("📅 Sentiment Over Time")
                            try:
                                # Convert timestamp to datetime if it's not already
                                if not pd.api.types.is_datetime64_any_dtype(result_documents['created_utc']):
                                    result_documents['date'] = pd.to_datetime(result_documents['created_utc'], unit='s')
                                
                                # Group by date and sentiment
                                time_data = result_documents.groupby([pd.Grouper(key='date', freq='D'), 'primary_sentiment']).size().reset_index(name='count')
                                
                                # Create time series chart
                                time_chart = alt.Chart(time_data).mark_line().encode(
                                    x='date:T',
                                    y='count:Q',
                                    color='primary_sentiment:N',
                                    tooltip=['date', 'primary_sentiment', 'count']
                                ).properties(
                                    width=700,
                                    height=400
                                )
                                
                                st.altair_chart(time_chart, use_container_width=True)
                            except Exception as e:
                                st.warning(f"Could not create time series chart: {str(e)}")
                            
                            # Parse the JSON in ner_entity_sentiments to extract FinBERT and VADER sentiments
                            if 'ner_entity_sentiments' in result_documents.columns:
                                st.subheader("**Model Specific Sentiment Analysis**")
                                
                                finbert_sentiments = []
                                vader_sentiments = []
                                finbert_scores = []
                                vader_scores = []
                                
                                for json_str in result_documents['ner_entity_sentiments']:
                                    try:
                                        import json
                                        data = json.loads(json_str.replace("'", '"'))
                                        
                                        # Extract first ticker's sentiment (simplified)
                                        for ticker, analyses in data.items():
                                            if 'finbert' in analyses:
                                                finbert_sentiments.append(analyses['finbert']['label'].lower())
                                                finbert_scores.append(analyses['finbert']['net_score'])
                                            if 'vader' in analyses:
                                                vader_sentiments.append(analyses['vader']['label'].lower())
                                                vader_scores.append(analyses['vader']['compound'])  # VADER uses 'compound' instead of 'net_score'
                                            break  # Just take the first ticker for simplicity
                                    except:
                                        continue
                                
                                # Create DataFrames for visualization
                                if finbert_sentiments:
                                    # Create two columns for sentiment distribution and score histogram
                                    col1, col2 = st.columns(2)
                                    
                                    with col1:
                                        finbert_df = pd.DataFrame({'Sentiment': finbert_sentiments})
                                        finbert_counts = finbert_df['Sentiment'].value_counts().reindex(sentiment_order).fillna(0).reset_index()
                                        finbert_counts.columns = ['Sentiment', 'Count']
                                        
                                        finbert_chart = alt.Chart(finbert_counts).mark_bar(color='lightblue').encode(
                                            x=alt.X('Sentiment', sort=sentiment_order),
                                            y='Count'
                                        ).properties(height=300)
                                        
                                        st.markdown("**FinBERT Sentiment Distribution**")
                                        st.altair_chart(finbert_chart, use_container_width=True)
                                    
                                    with col2:
                                        # Create histogram for FinBERT scores
                                        finbert_score_df = pd.DataFrame({'Score': finbert_scores})
                                        
                                        finbert_hist = alt.Chart(finbert_score_df).mark_bar(color='lightgreen').encode(
                                            x=alt.X('Score:Q', bin=alt.Bin(maxbins=20), title='FinBERT Net Score'),
                                            y='count()'
                                        ).properties(height=300)
                                        
                                        st.markdown("**FinBERT Score Distribution**")
                                        st.altair_chart(finbert_hist, use_container_width=True)
                                
                                if vader_sentiments:
                                    # Create two columns for sentiment distribution and score histogram
                                    col1, col2 = st.columns(2)
                                    
                                    with col1:
                                        vader_df = pd.DataFrame({'Sentiment': vader_sentiments})
                                        vader_counts = vader_df['Sentiment'].value_counts().reindex(sentiment_order).fillna(0).reset_index()
                                        vader_counts.columns = ['Sentiment', 'Count']
                                        
                                        vader_chart = alt.Chart(vader_counts).mark_bar(color='lightblue').encode(
                                            x=alt.X('Sentiment', sort=sentiment_order),
                                            y='Count'
                                        ).properties(height=300)
                                        
                                        st.markdown("**VADER Sentiment Distribution**")
                                        st.altair_chart(vader_chart, use_container_width=True)
                                    
                                    with col2:
                                        # Create histogram for VADER scores
                                        vader_score_df = pd.DataFrame({'Score': vader_scores})
                                        
                                        vader_hist = alt.Chart(vader_score_df).mark_bar(color='lightgreen').encode(
                                            x=alt.X('Score:Q', bin=alt.Bin(maxbins=20), title='VADER Compound Score'),
                                            y='count()'
                                        ).properties(height=300)
                                        
                                        st.markdown("**VADER Score Distribution**")
                                        st.altair_chart(vader_hist, use_container_width=True)
                else:
                    st.info("No matching documents found for your query.")
            except Exception as e:
                st.error(f"Error searching documents: {str(e)}")

# === Tab 4: RAG Sentiment Lookup ===
with tab4:
    st.header("Ask a Question")
    two_stage_query = st.text_input("Example: What is the sentiment of Tesla?", key="two_stage_query")

    # Add a search button
    search_button2 = st.button("Search",key="search3")

    if search_button2 or two_stage_query:
        st.write("Processing your query using the two-stage process...")
        answer,relevant_chunks = answer_stock_question(data_path,two_stage_query)
        st.subheader("Answer:")
        st.write(answer)
        st.subheader("Relevant Chunks:")
        st.write(relevant_chunks)