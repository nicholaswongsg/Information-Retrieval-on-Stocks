from flask import Flask, request, render_template_string
import pysolr
import time

app = Flask(__name__)

solr = pysolr.Solr('http://localhost:8983/solr/reddit_core', always_commit=True, timeout=10)

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
        <p>Found {{ results.numFound }} items in {{ time }} ms</p>
        <ul>
        {% for doc in results.docs %}
            <li>
                <strong>{{ doc.get("title", "[no title]") }}</strong><br>
                Subreddit: {{ doc.get("subreddit", "") }}<br>
                Text: {{ doc.get("selftext", doc.get("comment_body", ""))|safe }}<br>
                Score: {{ doc.get("score", doc.get("comment_score", "")) }}<br>
                Timestamp: {{ doc.get("created_utc", doc.get("comment_created_utc", "N/A")) }}
            </li>
        {% endfor %}
        </ul>
    {% endif %}
</body>
</html>
"""

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
    results = solr.search(full_query, q_op='OR', rows=10)
    end_time_perf = time.perf_counter()
    
    elapsed_time_ms = (end_time_perf - start_time_perf) * 1000
    
    return render_template_string(
        HTML_TEMPLATE,
        request=request,
        results=results,
        time=int(elapsed_time_ms),
        subreddits=subreddits,
        selected_subreddits=selected_subreddits
    )

if __name__ == "__main__":
    app.run(debug=True)