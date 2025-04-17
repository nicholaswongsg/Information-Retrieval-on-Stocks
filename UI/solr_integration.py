import pysolr
import pandas as pd
import streamlit as st
import altair as alt
import re
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import nltk
import json
from datetime import datetime
import os

# Import spell checker module
from spell_checker import suggest_correction

# Define the data path correctly
data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Classification", "NER_with_sentiment.csv"))

def connect_to_solr(core_name="stock_sentiment"):
    """Connect to Solr core with fallbacks for different environments"""
    # Try different URLs in sequence
    solr_urls = [
        "http://localhost:8983/solr/",
        "http://127.0.0.1:8983/solr/",
        "http://solr:8983/solr/"
    ]
    
    # Try each URL until one works
    last_error = None
    for url in solr_urls:
        try:
            # Connect silently without showing messages
            solr = pysolr.Solr(f"{url}{core_name}", always_commit=True, timeout=10)
            
            # Test connection with a simple query instead of ping
            test_results = solr.search("*:*", rows=1)
            # If we get here without an exception, the connection works
            return solr
            
        except Exception as e:
            last_error = str(e)
            continue
    
    # If all connection attempts fail
    raise ConnectionError(f"Could not connect to Solr. Please ensure Solr is running and accessible. Last error: {last_error}")

def search_solr(query, subreddits=None, start_date=None, end_date=None):
    """Search Solr with filters"""
    try:
        # Try to connect to Solr
        solr = connect_to_solr()
        
        # Build filter queries
        fq = []
        if subreddits and len(subreddits) > 0:
            subreddit_fq = " OR ".join([f'subreddit:"{subreddit}"' for subreddit in subreddits])
            fq.append(f"({subreddit_fq})")
        
        if start_date and end_date:
            # Convert to UTC timestamp
            start_datetime = datetime.combine(start_date, datetime.min.time())
            end_datetime = datetime.combine(end_date, datetime.max.time())
            start_utc = int(start_datetime.timestamp())
            end_utc = int(end_datetime.timestamp())
            fq.append(f"created_utc:[{start_utc} TO {end_utc}]")
        
        # Ensure the query is safe for Solr
        safe_query = query.replace(':', ' ').replace('(', ' ').replace(')', ' ')
        
        # Check for ticker-like symbols (uppercase 1-5 characters)
        is_ticker = re.match(r'^[A-Z]{1,5}$', query)
        
        # Determine search strategy - direct search without testing fields first
        try:
            if is_ticker:
                # For ticker symbols, search in ner_recognized_tickers
                q = f"ner_recognized_tickers:{safe_query}"
                search_type = "ticker"
            else:
                # For non-ticker queries, search in ner_text_cleaned first, then fallback
                q = f"ner_text_cleaned:{safe_query}"
                search_type = "entity"
                
                # Execute search to see if we get results
                results = solr.search(q, fq=fq, rows=5)
                
                # If no results found with ner_text_cleaned, also try text and title
                if results.hits == 0:
                    q = f"(ner_text_cleaned:{safe_query}) OR (text:{safe_query} OR title:{safe_query})"
        except Exception as e:
            # Fall back to text and title fields on any error
            q = f"text:{safe_query} OR title:{safe_query}"
            search_type = "fulltext"
        
        # Execute search with error handling
        results = solr.search(q, fq=fq, rows=2147483647)

        # Convert results to DataFrame
        data = []
        for result in results:
            doc = {
                'title': result.get('title', ''),
                'text': result.get('text', ''),
                'primary_sentiment': result.get('primary_sentiment', ''),
                'primary_score': result.get('primary_score', 0),
                'created_utc': result.get('created_utc', 0),
                'subreddit': result.get('subreddit', ''),
                'human2_sentiment': result.get('human2_sentiment', ''),
            }
            
            # Add NER fields if available - handle potential list types
            if 'ner_text_cleaned' in result:
                ner_text = result.get('ner_text_cleaned', '')
                doc['ner_text_cleaned'] = ner_text if isinstance(ner_text, str) else str(ner_text)
                
            if 'ner_recognized_tickers' in result:
                tickers = result.get('ner_recognized_tickers', '')
                doc['ner_recognized_tickers'] = tickers if isinstance(tickers, str) else str(tickers)
                
            if 'ner_entity_sentiments' in result:
                sentiments = result.get('ner_entity_sentiments', '{}')
                doc['ner_entity_sentiments'] = sentiments if isinstance(sentiments, str) else json.dumps(sentiments)
            data.append(doc)
        
        return pd.DataFrame(data)
    
    except Exception as e:
        st.error(f"Error searching Solr: {str(e)}")
        # Re-raise the exception to be caught by the caller
        raise

def render_solr_tab():
    """Render the Solr tab in Streamlit"""
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
                start_date = st.date_input("Start Date", min_date, min_value=min_date, max_value=max_date, key="solr_start_date")
            with date_col2:
                end_date = st.date_input("End Date", max_date, min_value=min_date, max_value=max_date, key="solr_end_date")
        
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
                key="solr_subreddits"
            )
    
    # Query input
    query = st.text_input("Example: Apple Stocks, Tesla earnings", key="solr_query")
    
    # Search button
    search_button = st.button("Search", key="solr_search")
    
    if query:
        suggestion = suggest_correction(query, data_path)
        if suggestion:
            st.info(f"Did you mean: **{suggestion}**?")


    if query or search_button:
        with st.spinner("Searching..."):
            try:
                start_time = datetime.now()
                if not query or not query.strip():
                    query = '*:*'

                # Execute Solr search
                results = search_solr(
                    query, 
                    subreddits=selected_subreddits, 
                    start_date=start_date, 
                    end_date=end_date,
                )
                
                # Display results
                end_time = datetime.now()
                elapsed_time = end_time - start_time
                elapsed_ms = elapsed_time.total_seconds() * 1000
                
                # Display timing information
                st.info(f"⏱️ Query processed in {elapsed_ms:.2f} ms ({elapsed_time.total_seconds():.2f} seconds)")
                entity_type = "entity" if 'ner_text_cleaned' in results.columns else "term"
                st.write(f"🔍 Found {len(results)} matching documents with {entity_type} '{query}'")
                result_documents = results
                
                if not results.empty:
                    # Show recognized tickers if available
                    if 'ner_recognized_tickers' in results.columns and not results['ner_recognized_tickers'].isna().all():
                        tickers = results['ner_recognized_tickers'].dropna().unique().tolist()
                        if tickers:
                            st.subheader("📊 Recognized Stock Tickers")
                            ticker_html = " ".join([f"<span style='background-color: #e6f2ff; padding: 5px; margin: 5px; border-radius: 5px;'>{ticker}</span>" for ticker in tickers])
                            st.markdown(f"Tickers mentioned in results: {ticker_html}", unsafe_allow_html=True)
                    df = result_documents
                    for column in df.columns:
                        # Check if the column contains lists
                        if df[column].apply(lambda x: isinstance(x, list)).any():
                            # Extract the single value from each list
                            df[column] = df[column].apply(lambda x: x[0] if isinstance(x, list) and len(x) == 1 else x)
                    result_documents = df
                    # Display results dataframe
                    st.dataframe(results)
                    
                    # Word cloud from results - use ner_text_cleaned if available, otherwise use text
                    st.subheader("🌥 Word Cloud")
                    
                    # Determine which text field to use
                    if 'ner_text_cleaned' in results.columns and not results['ner_text_cleaned'].isna().all():
                        text_field = 'ner_text_cleaned'
                        cloud_title = "Entity Word Cloud"
                    else:
                        text_field = 'text'
                        cloud_title = "Text Word Cloud"
                    
                    st.markdown(f"**{cloud_title}**")
                    
                    if text_field in results.columns and len(results) > 0:
                        combined_text = " ".join(results[text_field].dropna().astype(str))
                        combined_text = re.sub(r'https?://\S+|www\.\S+', '', combined_text)
                        combined_text = re.sub(r'[^\w\s]', '', combined_text)
                        combined_text = combined_text.lower()
                        
                        # Get stopwords from nltk
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
                            st.info(f"Not enough content to generate a word cloud.")
                    else:
                        st.info(f"No text available for word cloud generation.")
                    
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

                                chatgpt_sentiments=[]
                                for json_str in result_documents['human2_sentiment']:
                                    try:
                                        import json
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
                    st.info(f"No matches found for '{query}'. Try a different term or company name.")
            except Exception as e:
                st.error(f"Error searching Solr: {str(e)}")
                st.info("""
                Please verify that Solr is running and properly configured.
                """)
                
if __name__ == "__main__":
    # For testing the module independently
    st.set_page_config(page_title="Solr Integration Test", layout="wide")
    render_solr_tab() 