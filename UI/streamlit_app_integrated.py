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
import json

# Import Solr integration module
try:
    from solr_integration import render_solr_tab, perform_boolean_search
except ImportError:
    # Local import when running in the same directory
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))
    from solr_integration import render_solr_tab, perform_boolean_search

# Import spell checker module
from spell_checker import suggest_correction

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
tab1, tab2, tab3, tab4, tab5 = st.tabs(["About", "Inverted Index", "Retrieval-Augmented Generation", "Solr Search", "Boolean Search"])


# === Tab 1: About ===
with tab1:
    st.header("About This Application")
    st.markdown("""
    ### Stock Sentiment Analysis Tool
    
    This comprehensive application enables users to analyze stock market sentiment through various search methodologies and visualization techniques.

    #### Features:
    - Multiple search options: Inverted Index, RAG, Solr, and Boolean queries
    - Multi-model sentiment analysis (FinBERT, VADER, and ChatGPT)
    - Interactive visualizations including word clouds, sentiment distributions and time-based sentiment trend analysis
    - Filtering capabilities according to subreddit or date
    - Spelling correction for stock terminology and company names
                
    #### Search Options:
    - **Inverted Index**: Fast keyword-based document retrieval
    - **Retrieval-Augmented Generation**: Leverage natural language to ask questions about stocks
    - **Solr Search**: Powerful text search with advanced filtering
    - **Boolean Search**: More targeted queries using binary (AND, OR, NOT) operators

    #### How It Works:
    1. Select the appropriate search tab for your query needs
    2. Filter by date range and specific subreddits if desired
    3. Enter your search query about stocks or companies
    4. View comprehensive results with sentiment breakdowns and visualizations

    #### Data Sources:
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
                start_date = st.date_input("Start Date", min_date, min_value=min_date, max_value=max_date)
            with date_col2:
                end_date = st.date_input("End Date", max_date, min_value=min_date, max_value=max_date)
            
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
                default=[]  # Default selection
            )
        
    custom_query = st.text_input("Example: Apple Stocks, Tesla earnings", key="custom_query_csv")

    # Add a search button
    search_button = st.button("Search",key="search1")

    if custom_query:
        suggestion = suggest_correction(custom_query, data_path)
        if suggestion:
            st.info(f"Did you mean: **{suggestion}**?")

    if custom_query or search_button:
        with st.spinner("Searching documents..."):
            try:
                # Start timing
                start_time = datetime.now()

                # Call the search_index function from inverted_index_edited.py
                stopwords_list, result_documents = search_index(data_path, custom_query)

                #1. Filter by time
                if not result_documents.empty:
                    result_documents = result_documents[(result_documents['created_utc'] >= start_utc) & (result_documents['created_utc'] <= end_utc)]

                # 2. Filter by selected subreddits if any are selected
                if selected_subreddits:
                    result_documents = result_documents[result_documents['subreddit'].isin(selected_subreddits)]
                else:
                    # If no subreddits are selected, keep all subreddits
                    pass
                
                end_time = datetime.now()
                elapsed_time = end_time - start_time
                elapsed_ms = elapsed_time.total_seconds() * 1000
                
                # Display results
                st.info(f"⏱️ Query processed in {elapsed_ms:.2f} ms ({elapsed_time.total_seconds():.2f} seconds)")
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
                                        data = json.loads(json_str.replace("'", '"'))
                                        
                                        for ticker, analyses in data.items():
                                            if 'finbert' in analyses:
                                                finbert_sentiments.append(analyses['finbert']['label'].lower())
                                                finbert_scores.append(analyses['finbert']['net_score'])
                                            if 'vader' in analyses:
                                                vader_sentiments.append(analyses['vader']['label'].lower())
                                                vader_scores.append(analyses['vader']['compound'])  # VADER uses 'compound' instead of 'net_score'
                                    except:
                                        continue

                                chatgpt_sentiments=[]
                                for json_str in result_documents['human2_sentiment']:
                                    try:
                                        data = json.loads(json_str.replace("'", '"'))

                                        # Extract first ticker's sentiment (simplified)
                                        for ticker, analyses in data.items():
                                            if 'human2' in analyses:
                                                chatgpt_sentiments.append(analyses['human2']['label'].lower())
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
                                if chatgpt_sentiments:
                                    chatgpt_df = pd.DataFrame({'Sentiment': chatgpt_sentiments})
                                    chatgpt_counts = chatgpt_df['Sentiment'].value_counts().reindex(sentiment_order).fillna(0).reset_index()
                                    chatgpt_counts.columns = ['Sentiment', 'Count']
                                    
                                    chatgpt_chart = alt.Chart(chatgpt_counts).mark_bar(color='lightblue').encode(
                                        x=alt.X('Sentiment', sort=sentiment_order),
                                        y='Count'
                                    ).properties(height=300)
                                    
                                    st.markdown("**ChatGPT Sentiment Distribution**")
                                    st.altair_chart(chatgpt_chart, use_container_width=True)
                                    
                else:
                    st.info("No matching documents found for your query.")
            except Exception as e:
                st.error(f"Error searching documents: {str(e)}")

# === Tab 3: RAG Sentiment Lookup ===
with tab3:
    st.header("Ask a Question")
    two_stage_query = st.text_input("Example: What is the sentiment of Tesla?", key="two_stage_query")
    
    # Add a search button
    search_button2 = st.button("Search",key="search2")

    if custom_query:
        suggestion = suggest_correction(custom_query, data_path)
        if suggestion:
            st.info(f"Did you mean: **{suggestion}**?")
            
    if search_button2 and two_stage_query:
        start_time = datetime.now()
        st.write("Processing your query using the two-stage process...")
        answer,relevant_chunks = answer_stock_question(data_path,two_stage_query)
        
        end_time = datetime.now()
        elapsed_time = end_time - start_time
        elapsed_ms = elapsed_time.total_seconds() * 1000
            
        # Display timing information
        st.info(f"⏱️ Query processed in {elapsed_ms:.2f} ms ({elapsed_time.total_seconds():.2f} seconds)")
        st.subheader("Answer:")
        st.write(answer)
        st.subheader("Relevant Chunks:")
        st.write(relevant_chunks)

# === Tab 4: Solr Search ===
with tab4:
    # Call the Solr integration module to render this tab
    try:
        render_solr_tab()
    except Exception as e:
        st.error(f"Error loading Solr integration: {str(e)}")
        st.info("Please make sure Solr is running and accessible. You can start it using Docker with the provided setup files.")

# === Tab 5: Boolean Search ===
with tab5:
    st.header("Boolean Search")
    st.markdown("""
    Use Boolean operators (AND, OR, NOT) and parentheses to create complex queries.
    
    Examples:
    - `Apple OR Samsung` (documents containing either Apple or Samsung)
    - `Apple AND Samsung` (documents containing both Apple and Samsung)
    - `Apple NOT Samsung` (documents containing Apple but not Samsung)
    - `(Apple OR Samsung) AND earnings` (documents with either Apple or Samsung, and also containing earnings)
    """)
    
    boolean_query = st.text_input("Enter your Boolean query", key="boolean_query")
    
    # Create filter section
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
                start_date = st.date_input("Start Date", min_date, min_value=min_date, max_value=max_date, key="bool_start_date")
            with date_col2:
                end_date = st.date_input("End Date", max_date, min_value=min_date, max_value=max_date, key="bool_end_date")
            
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
                key="bool_subreddits"
            )
    
    # Add a search button
    boolean_search_button = st.button("Search", key="boolean_search")
    
    if boolean_search_button and boolean_query:
        try:
            with st.spinner("Executing Boolean search..."):
                start_time = datetime.now()
                
                # Call the perform_boolean_search function from solr_integration.py
                result_documents = perform_boolean_search(
                    query=boolean_query, 
                    start_time=start_utc, 
                    end_time=end_utc,
                    subreddits=selected_subreddits if selected_subreddits else None
                )
                
                end_time = datetime.now()
                elapsed_time = end_time - start_time
                elapsed_ms = elapsed_time.total_seconds() * 1000
                
                # Display timing information
                st.info(f"⏱️ Query processed in {elapsed_ms:.2f} ms ({elapsed_time.total_seconds():.2f} seconds)")
                
                if result_documents is not None and len(result_documents) > 0:
                    st.write(f"🔍 Found {len(result_documents)} matching documents")
                    st.dataframe(result_documents)
                    
                    # Word cloud from results, similar to Tab 2
                    st.subheader("🌥 Word Cloud from Search Results")
                    if 'text' in result_documents.columns:
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
                        
                        # Create custom stopwords list (similar to what's passed from inverted index)
                        custom_stopwords = []
                        stop_words.update(custom_stopwords)
                        
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
                    
                    # Display sentiment distribution if available
                    sentiment_order = [
                        "strongly negative",
                        "slightly negative",
                        "neutral",
                        "slightly positive",
                        "strongly positive"
                    ]
                    
                    if 'primary_sentiment' in result_documents.columns:
                        st.subheader("📊 Sentiment Distribution")
                        primary_counts = result_documents['primary_sentiment'].value_counts().reindex(sentiment_order).fillna(0).reset_index()
                        primary_counts.columns = ['Sentiment', 'Count']
                        
                        primary_chart = alt.Chart(primary_counts).mark_bar(color='lightblue').encode(
                            x=alt.X('Sentiment', sort=sentiment_order),
                            y='Count'
                        ).properties(height=300)
                        
                        st.altair_chart(primary_chart, use_container_width=True)
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
                                    # First, strip the outer array brackets if they exist
                                    if json_str.startswith('[') and json_str.endswith(']'):
                                        json_str = json_str[2:-2]
                                    
                                    # Parse the JSON string
                                    data = json.loads(json_str)
                                    
                                    # Process each ticker in the data
                                    for ticker, analyses in data.items():
                                        if 'finbert' in analyses:
                                            finbert_sentiments.append(analyses['finbert']['label'].lower())
                                            finbert_scores.append(analyses['finbert']['net_score'])
                                        if 'vader' in analyses:
                                            vader_sentiments.append(analyses['vader']['label'].lower())
                                            vader_scores.append(analyses['vader']['compound'])  # VADER uses 'compound' instead of 'net_score'
                                except Exception as e:
                                    print(f"Error processing JSON: {e}")
                                    continue

                            chatgpt_sentiments=[]
                            for json_str in result_documents['human2_sentiment']:
                                try:
                                    data = json.loads(json_str.replace("'", '"'))

                                    # Extract first ticker's sentiment (simplified)
                                    for ticker, analyses in data.items():
                                        if 'human2' in analyses:
                                            chatgpt_sentiments.append(analyses['human2']['label'].lower())
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
                            if chatgpt_sentiments:
                                chatgpt_df = pd.DataFrame({'Sentiment': chatgpt_sentiments})
                                chatgpt_counts = chatgpt_df['Sentiment'].value_counts().reindex(sentiment_order).fillna(0).reset_index()
                                chatgpt_counts.columns = ['Sentiment', 'Count']
                                
                                chatgpt_chart = alt.Chart(chatgpt_counts).mark_bar(color='lightblue').encode(
                                    x=alt.X('Sentiment', sort=sentiment_order),
                                    y='Count'
                                ).properties(height=300)
                                
                                st.markdown("**ChatGPT Sentiment Distribution**")
                                st.altair_chart(chatgpt_chart, use_container_width=True)
                else:
                    st.info("No matching documents found for your Boolean query.")
        except Exception as e:
            st.error(f"Error in Boolean search: {str(e)}")