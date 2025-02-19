import praw
import pandas as pd
import os
from dotenv import load_dotenv
import csv

load_dotenv()

reddit = praw.Reddit(
    client_id=os.getenv("REDDIT_CLIENT_ID"),
    client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
    user_agent=os.getenv("REDDIT_USER_AGENT")
)

# Replace here with the subbreddit u doing
SUBREDDITS = ["stockmarket", "stocks", "investing", "stocksandtrading"]
POST_LIMIT = 1000

POSTS_CSV = "reddit_stock_posts.csv"
COMMENTS_CSV = "reddit_stock_comments.csv"

# Load existing post IDs and comment IDs to avoid duplicates
existing_post_ids = set()
existing_comment_ids = set()

if os.path.exists(POSTS_CSV):
    posts_df = pd.read_csv(POSTS_CSV, dtype=str)  # Load existing posts
    existing_post_ids = set(posts_df["post_id"].astype(str))
    print(f"Loaded {len(existing_post_ids)} existing posts.")

if os.path.exists(COMMENTS_CSV):
    comments_df = pd.read_csv(COMMENTS_CSV, dtype=str)  # Load existing comments
    existing_comment_ids = set(comments_df["comment_id"].astype(str))
    print(f"Loaded {len(existing_comment_ids)} existing comments.")

# Open CSVs in append mode
post_file = open(POSTS_CSV, mode="a", newline="", encoding="utf-8")
comment_file = open(COMMENTS_CSV, mode="a", newline="", encoding="utf-8")

post_writer = csv.DictWriter(post_file, fieldnames=[
    "post_id", "subreddit", "title", "selftext", "created_utc", "author",
    "score", "num_comments", "permalink", "text"
])

comment_writer = csv.DictWriter(comment_file, fieldnames=[
    "comment_id", "post_id", "subreddit", "comment_author",
    "comment_body", "comment_score", "comment_created_utc", "text"
])

# Write headers if files are empty
if os.stat(POSTS_CSV).st_size == 0:
    post_writer.writeheader()
if os.stat(COMMENTS_CSV).st_size == 0:
    comment_writer.writeheader()

# Scraping process
for sub in SUBREDDITS:
    print(f"Fetching posts from r/{sub}...")
    subreddit = reddit.subreddit(sub)

    for post in subreddit.new(limit=POST_LIMIT):
        if post.id in existing_post_ids:
            continue  # Skip if already processed

        # Save post immediately
        post_data = {
            "post_id": post.id,
            "subreddit": sub,
            "title": post.title,
            "selftext": post.selftext,
            "created_utc": post.created_utc,
            "author": post.author.name if post.author else "Unknown",
            "score": post.score,
            "num_comments": post.num_comments,
            "permalink": f"https://www.reddit.com{post.permalink}",
            "text": f"{post.title} {post.selftext}"
        }
        post_writer.writerow(post_data)
        post_file.flush()  # Immediately write to file
        existing_post_ids.add(post.id)  # Add to set to avoid duplicates

        # Fetch and save comments immediately
        try:
            post.comments.replace_more(limit=0)
            all_comments = post.comments.list()

            for comment in all_comments:
                if comment.id in existing_comment_ids:
                    continue  # Skip if already processed

                comment_data = {
                    "comment_id": comment.id,
                    "post_id": post.id,
                    "subreddit": sub,
                    "comment_author": comment.author.name if comment.author else "Unknown",
                    "comment_body": comment.body,
                    "comment_score": comment.score,
                    "comment_created_utc": comment.created_utc,
                    "text": comment.body
                }
                comment_writer.writerow(comment_data)
                comment_file.flush()  # Immediately write to file
                existing_comment_ids.add(comment.id)  # Avoid duplicates

        except Exception as e:
            print(f"Error fetching comments for post {post.id}: {e}")

# Close files properly
post_file.close()
comment_file.close()

print("Scraping complete!")
