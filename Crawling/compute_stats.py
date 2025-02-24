import pandas as pd

# Load the CSV files
posts = pd.read_csv("reddit_stock_posts.csv")
comments = pd.read_csv("reddit_stock_comments.csv")

# Combine text fields from posts (title and selftext) and comments (comment_body)
post_text = ' '.join(posts['title'].fillna('')) + ' ' + ' '.join(posts['selftext'].fillna(''))
comment_text = ' '.join(comments['comment_body'].fillna(''))
corpus = post_text + ' ' + comment_text

# Count total words and unique words (types)
words = corpus.split()
total_words = len(words)
unique_words = len(set(words))

print("Number of post records:", len(posts))
print("Number of comment records:", len(comments))
print("Total words in corpus:", total_words)
print("Total unique word types:", unique_words)
