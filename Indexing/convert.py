import pysolr
import pandas as pd
import re

def clean_text(text):
    """
    Perform lightweight cleaning on the provided text:
    - Replace URLs with <URL>
    - Replace Reddit user mentions (u/username) with <USER>
    - Replace subreddit references (r/subreddit) with <SUBREDDIT>
    - Replace stock tickers (e.g., $TSLA) with <TICKER>
    - Convert text to lowercase
    """
    if pd.isnull(text):
        return ""
    text = re.sub(r"http\S+", "<URL>", text)            # Replace URLs
    text = re.sub(r"u\/\w+", "<USER>", text)            # Replace user mentions
    text = re.sub(r"r\/\w+", "<SUBREDDIT>", text)       # Replace subreddit references
    text = re.sub(r"\$\w+", "<TICKER>", text)           # Replace stock tickers
    text = text.lower().strip()                         # Lowercase text and remove trailing spaces
    return text

# Connect to Solr
solr_url = "http://localhost:8983/solr/reddit_core"
solr = pysolr.Solr(solr_url, always_commit=True, timeout=10)

# Load CSV files
posts_df = pd.read_csv("../Crawling/reddit_stock_posts.csv")
comments_df = pd.read_csv("../Crawling/reddit_stock_comments.csv")

# Store initial counts before cleaning and deduplication
initial_post_count = len(posts_df)
initial_comment_count = len(comments_df)

# Apply cleaning to text fields in posts
if "title" in posts_df.columns:
    posts_df["title"] = posts_df["title"].apply(clean_text)
if "selftext" in posts_df.columns:
    posts_df["selftext"] = posts_df["selftext"].apply(clean_text)
if "text" in posts_df.columns:
    posts_df["text"] = posts_df["text"].apply(clean_text)

# Apply cleaning to text fields in comments
if "comment_body" in comments_df.columns:
    comments_df["comment_body"] = comments_df["comment_body"].apply(clean_text)
if "text" in comments_df.columns:
    comments_df["text"] = comments_df["text"].apply(clean_text)

# Remove duplicate posts (title + content must be the same)
if "title" in posts_df.columns and "selftext" in posts_df.columns:
    posts_df = posts_df.drop_duplicates(subset=["title", "selftext"], keep="first")

# Remove duplicate comments (same author & comment body)
if "comment_author" in comments_df.columns and "comment_body" in comments_df.columns:
    comments_df = comments_df.drop_duplicates(subset=["comment_author", "comment_body"], keep="first")

# Store final counts after cleaning and deduplication
final_post_count = len(posts_df)
final_comment_count = len(comments_df)

# Calculate removed duplicates
removed_posts = initial_post_count - final_post_count
removed_comments = initial_comment_count - final_comment_count

# Convert Posts to Solr format and add to Solr index
posts_data = posts_df.to_dict(orient="records")
solr.add(posts_data)

# Convert Comments to Solr format and add to Solr index
comments_data = comments_df.to_dict(orient="records")
solr.add(comments_data)

# Print summary
print("=== Data Cleaning & Deduplication Summary ===")
print(f"Initial Post Count: {initial_post_count}")
print(f"Final Post Count: {final_post_count} (Removed {removed_posts} duplicate posts with same title & content)")
print(f"Initial Comment Count: {initial_comment_count}")
print(f"Final Comment Count: {final_comment_count} (Removed {removed_comments} duplicate comments by same author & body)")
print("Converted into Solr format and added cleaned, deduplicated data to Solr index.")
