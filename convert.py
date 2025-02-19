import pysolr
import pandas as pd

# Connect to Solr
solr_url = "http://localhost:8983/solr/reddit_core"
solr = pysolr.Solr(solr_url, always_commit=True, timeout=10)

# Load CSV files
posts_df = pd.read_csv("reddit_stock_posts.csv")
comments_df = pd.read_csv("reddit_stock_comments.csv")

# Convert Posts to Solr format
posts_data = posts_df.to_dict(orient="records")
solr.add(posts_data)

# Convert Comments to Solr format
comments_data = comments_df.to_dict(orient="records")
solr.add(comments_data)

print("Converted into Solr format and added to Solr index.")
