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
        
        <label for="subreddit_query">Subreddit:</label>
        <input type="text" name="subreddit_query" id="subreddit_query" value="{{ request.args.get('subreddit_query', '') }}"><br><br>
        
        <label for="score_min">Score (Min):</label>
        <input type="number" name="score_min" id="score_min" value="{{ request.args.get('score_min', '') }}"><br><br>
        
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
                Score: {{ doc.get("score", doc.get("comment_score", "")) }}
            </li>
        {% endfor %}
        </ul>
    {% endif %}
</body>
</html>
"""

@app.route("/", methods=["GET"])
def search():
    # Grab field-specific inputs
    title_query = request.args.get("title_query", "").strip()
    selftext_query = request.args.get("selftext_query", "").strip()
    comment_query = request.args.get("comment_query", "").strip()
    subreddit_query = request.args.get("subreddit_query", "").strip()
    score_min = request.args.get("score_min", "").strip()
    
    clauses = []
    
    if title_query:
        clauses.append(f"title:({title_query})")
    
    if selftext_query:
        clauses.append(f"selftext:({selftext_query})")
    
    if comment_query:
        clauses.append(f"comment_body:({comment_query})")
    
    if subreddit_query:
        clauses.append(f"subreddit:({subreddit_query})")
    
    if score_min:
        clauses.append(f"score:[{score_min} TO *]")
        # If you have separate fields for post score vs. comment score, 
        # you could also handle them separately or unify them in your schema.
    
    # If no fields were specified, default to all documents
    full_query = " AND ".join(clauses) if clauses else "*:*"

    # Measure local (client-side) time
    start_time = time.perf_counter()
    results = solr.search(full_query, q_op='OR', rows=10)
    end_time = time.perf_counter()
    
    elapsed_time_ms = (end_time - start_time) * 1000
    
    return render_template_string(
        HTML_TEMPLATE,
        request=request,
        results=results,
        time=int(elapsed_time_ms)
    )

if __name__ == "__main__":
    app.run(debug=True)
