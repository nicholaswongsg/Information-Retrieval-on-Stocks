from flask import Flask, request, render_template_string, Response
import pysolr
import time
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import io
import base64
import re
from collections import Counter
import numpy as np
import nltk
from nltk.corpus import stopwords

app = Flask(__name__)

# Initialize NLTK stopwords
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

STOPWORDS = set(stopwords.words('english'))
# Add some common Reddit and finance-specific words that aren't meaningful for word clouds
CUSTOM_STOPWORDS = {'like', 'just', 'will', 'get', 'got', 'going', 'think', 'know', 'really', 
                   'stock', 'stocks', 'market', 'share', 'shares', 'price', 'buy', 'sell', 
                   'trading', 'im', 'ive', 'youre', 'dont', 'didnt', 'cant', 'wont', 'thats',
                   'would', 'could', 'should', 'http', 'https', 'www', 'com', 'year', 'day',
                   'one', 'two', 'three', 'first', 'second', 'last', 'next', 'even', 'much',
                   'many', 'also', 'edit', 'deleted', 'removed'}
ALL_STOPWORDS = STOPWORDS.union(CUSTOM_STOPWORDS)

solr = pysolr.Solr('http://localhost:8983/solr/reddit_core', always_commit=True, timeout=10)

# Updated HTML template without using max() function
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Reddit Stock Search</title>
    <style>
        .checkbox-group {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 5px;
            margin-bottom: 15px;
        }
        .checkbox-item {
            display: flex;
            align-items: center;
        }
        .results-container {
            display: flex;
            flex-wrap: wrap;
        }
        .results-list {
            flex: 1;
            min-width: 300px;
        }
        .wordcloud-container {
            flex: 1;
            min-width: 300px;
            text-align: center;
            margin-top: 20px;
        }
        .wordcloud-img {
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 5px;
            box-shadow: 0 0 5px rgba(0,0,0,0.1);
        }
        .pagination {
            margin-top: 20px;
            text-align: center;
        }
        .pagination a, .pagination span {
            display: inline-block;
            padding: 8px 16px;
            text-decoration: none;
            border: 1px solid #ddd;
            margin: 0 4px;
        }
        .pagination a:hover {
            background-color: #f1f1f1;
        }
        .pagination .active {
            background-color: #4CAF50;
            color: white;
            border: 1px solid #4CAF50;
        }
        .pagination .disabled {
            color: #ddd;
            border: 1px solid #ddd;
        }
    </style>
</head>
<body>
    <h1>Search the Reddit Stock Index</h1>
    <form method="GET" action="/">
        <label for="title_query">Title:</label>
        <input type="text" name="title_query" id="title_query" value="{{ request.args.get('title_query', '') }}"><br><br>
        
        <label for="selftext_query">Selftext:</label>
        <input type="text" name="selftext_query" id="selftext_query" value="{{ request.args.get('selftext_query', '') }}"><br><br>
        
        <label for="comment_query">Comment Body:</label>
        <input type="text" name="comment_query" id="comment_query" value="{{ request.args.get('comment_query', '') }}"><br><br>
        
        <label>Subreddits:</label>
        <div class="checkbox-group">
            {% for subreddit in subreddits %}
                <div class="checkbox-item">
                    <input type="checkbox" name="subreddit_query" id="subreddit_{{ subreddit }}" value="{{ subreddit }}" 
                           {% if subreddit in selected_subreddits %}checked{% endif %}>
                    <label for="subreddit_{{ subreddit }}">{{ subreddit }}</label>
                </div>
            {% endfor %}
        </div>
        
        <label for="score_min">Score (Min):</label>
        <input type="number" name="score_min" id="score_min" value="{{ request.args.get('score_min', '') }}"><br><br>
        
        <label for="start_time">Start Time (UTC Timestamp):</label>
        <input type="number" name="start_time" id="start_time" value="{{ request.args.get('start_time', '') }}"><br><br>
        
        <label for="end_time">End Time (UTC Timestamp):</label>
        <input type="number" name="end_time" id="end_time" value="{{ request.args.get('end_time', '') }}"><br><br>
        
        <input type="submit" value="Search">
    </form>
    <hr>
    {% if results %}
        <h2>Results</h2>
        <p>Found {{ total_results }} items in {{ time }} ms</p>
        
        <div class="results-container">
            <div class="results-list">
                <ul>
                {% for doc in results.docs %}
                    <li>
                        <strong>{{ doc.get("title", "[no title]") }}</strong><br>
                        Subreddit: {{ doc.get("subreddit", "") }}<br>
                        Text: {{ doc.get("selftext", doc.get("comment_body", ""))|truncate(200)|safe }}<br>
                        Score: {{ doc.get("score", doc.get("comment_score", "")) }}<br>
                        Timestamp: {{ doc.get("created_utc", doc.get("comment_created_utc", "N/A")) }}
                    </li>
                {% endfor %}
                </ul>
                
                {% if total_pages > 1 %}
                <div class="pagination">
                    {% if page > 1 %}
                        <a href="{{ request.path }}?{{ request.query_string.decode('utf-8').replace('page='+page|string, 'page='+((page-1)|string)) if 'page=' in request.query_string.decode('utf-8') else request.query_string.decode('utf-8') + ('&' if request.query_string else '') + 'page='+((page-1)|string) }}">&laquo; Previous</a>
                    {% else %}
                        <span class="disabled">&laquo; Previous</span>
                    {% endif %}
                    
                    {% for p in page_range %}
                        {% if p == page %}
                            <span class="active">{{ p }}</span>
                        {% else %}
                            <a href="{{ request.path }}?{{ request.query_string.decode('utf-8').replace('page='+page|string, 'page='+p|string) if 'page=' in request.query_string.decode('utf-8') else request.query_string.decode('utf-8') + ('&' if request.query_string else '') + 'page='+p|string }}">{{ p }}</a>
                        {% endif %}
                    {% endfor %}
                    
                    {% if page < total_pages %}
                        <a href="{{ request.path }}?{{ request.query_string.decode('utf-8').replace('page='+page|string, 'page='+((page+1)|string)) if 'page=' in request.query_string.decode('utf-8') else request.query_string.decode('utf-8') + ('&' if request.query_string else '') + 'page='+((page+1)|string) }}">Next &raquo;</a>
                    {% else %}
                        <span class="disabled">Next &raquo;</span>
                    {% endif %}
                </div>
                {% endif %}
            </div>
            
            {% if wordcloud_img %}
            <div class="wordcloud-container">
                <h3>Word Cloud Analysis</h3>
                <p><small>Based on 50 most recent results, using up to 50 most frequent terms</small></p>
                <img src="data:image/png;base64,{{ wordcloud_img }}" alt="Word Cloud" class="wordcloud-img">
            </div>
            {% endif %}
        </div>
    {% endif %}
</body>
</html>
"""

def clean_text(text):
    """Clean up text for word cloud processing"""
    if not text:
        return ""
    
    # Convert list to string if needed
    if isinstance(text, list):
        text = ' '.join(str(item) for item in text)
    elif not isinstance(text, str):
        text = str(text)
        
    # Convert to lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    # Remove special characters and digits
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def generate_wordcloud(docs, max_words=100):
    """Generate a word cloud from the document text"""
    # Extract and combine text from documents
    text_data = []
    for doc in docs:
        # Get title text
        if "title" in doc:
            text_data.append(clean_text(doc["title"]))
        
        # Get either selftext or comment body
        if "selftext" in doc and doc["selftext"]:
            text_data.append(clean_text(doc["selftext"]))
        elif "comment_body" in doc and doc["comment_body"]:
            text_data.append(clean_text(doc["comment_body"]))
    
    if not text_data:
        return None
    
    combined_text = " ".join(text_data)
    
    # Filter out stopwords
    words = combined_text.split()
    filtered_words = [word for word in words if word not in ALL_STOPWORDS and len(word) > 2]
    filtered_text = " ".join(filtered_words)
    
    if not filtered_text:
        return None
    
    # Generate word frequencies
    word_counts = Counter(filtered_words)
    
    # Create word cloud
    wordcloud = WordCloud(
        width=800, 
        height=400,
        max_words=max_words,
        background_color='white',
        colormap='viridis',
        contour_width=1,
        contour_color='steelblue'
    ).generate_from_frequencies(word_counts)
    
    # Convert to image
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    
    # Save to base64 string
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png', bbox_inches='tight', pad_inches=0.1)
    img_buf.seek(0)
    plt.close()
    
    return base64.b64encode(img_buf.read()).decode('utf-8')

@app.route("/", methods=["GET"])
def search():
    # Define the specific subreddits for the checkboxes
    subreddits = ["wallstreetbets", "investing", "stockmarket", "stocksandtrading", "stocks", "applestocks", "microsoft", "NVDA_Stock"]
    
    # Grab field-specific inputs
    title_query = request.args.get("title_query", "").strip()
    selftext_query = request.args.get("selftext_query", "").strip()
    comment_query = request.args.get("comment_query", "").strip()
    
    # Get all selected subreddits (getlist returns all values when multiple checkboxes are selected)
    selected_subreddits = request.args.getlist("subreddit_query")
    
    score_min = request.args.get("score_min", "").strip()
    start_time = request.args.get("start_time", "").strip()
    end_time = request.args.get("end_time", "").strip()
    
    # Add pagination
    page = int(request.args.get("page", 1))
    results_per_page = 10
    start = (page - 1) * results_per_page
    
    clauses = []
    
    if title_query:
        clauses.append(f"title:({title_query})")
    
    if selftext_query:
        clauses.append(f"selftext:({selftext_query})")
    
    if comment_query:
        clauses.append(f"comment_body:({comment_query})")
    
    # Handle multiple subreddit selections
    if selected_subreddits:
        # Create a subreddit clause that matches any of the selected subreddits
        subreddit_terms = " OR ".join([f"subreddit:{s}" for s in selected_subreddits])
        clauses.append(f"({subreddit_terms})")
    
    if score_min:
        clauses.append(f"score:[{score_min} TO *]")
    
    # Time range filtering
    if start_time and end_time:
        clauses.append(f"(created_utc:[{start_time} TO {end_time}] OR comment_created_utc:[{start_time} TO {end_time}])")
    elif start_time:
        clauses.append(f"(created_utc:[{start_time} TO *] OR comment_created_utc:[{start_time} TO *])")
    elif end_time:
        clauses.append(f"(created_utc:[* TO {end_time}] OR comment_created_utc:[* TO {end_time}])")
    
    # If no fields were specified, default to all documents
    full_query = " AND ".join(clauses) if clauses else "*:*"

    # Measure local (client-side) time
    start_time_perf = time.perf_counter()
    
    # For display: paginated results
    paginated_results = solr.search(full_query, q_op='OR', rows=results_per_page, start=start)
    
    # For word cloud: get 50 results
    if any([title_query, selftext_query, comment_query, selected_subreddits]):
        wordcloud_results = solr.search(full_query, q_op='OR', rows=50,sort='created_utc desc')
        wordcloud_docs = wordcloud_results.docs
    else:
        wordcloud_docs = []
    
    end_time_perf = time.perf_counter()
    
    elapsed_time_ms = (end_time_perf - start_time_perf) * 1000
    
    # Generate word cloud if we have results
    wordcloud_img = None
    if wordcloud_docs:
        wordcloud_img = generate_wordcloud(wordcloud_docs, max_words=50)
    
    # Calculate pagination info
    total_results = paginated_results.hits
    total_pages = (total_results + results_per_page - 1) // results_per_page
    
    # Calculate pagination display range
    start_page = max(1, page - 2)
    end_page = min(total_pages + 1, page + 3)
    page_range = range(start_page, end_page)
    
    return render_template_string(
        HTML_TEMPLATE,
        request=request,
        results=paginated_results,
        time=int(elapsed_time_ms),
        subreddits=subreddits,
        selected_subreddits=selected_subreddits,
        wordcloud_img=wordcloud_img,
        page=page,
        total_pages=total_pages,
        total_results=total_results,
        page_range=page_range
    )

if __name__ == "__main__":
    app.run(debug=True)