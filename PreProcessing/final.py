import pandas as pd
import re

# ---------------------------------------------------------------------
# 1) Configure known stock symbols, synonyms, and subreddits
# ---------------------------------------------------------------------
STOCK_ALIASES = {
    "AAPL": ["apple", "aapl", "iphone", "ipad", "macbook", "ios", "macos", "apple watch", "airpods", "tim cook"],
    "MSFT": ["microsoft", "msft", "windows", "azure", "xbox", "office", "copilot", "bing", "linkedin", "m365", "satya nadella"],
    "NVDA": ["nvidia", "nvda", "geforce", "rtx", "cuda", "hbm", "gb200", "jensen huang"],
    "TSLA": ["tesla", "tsla", "model 3", "model s", "model x", "model y", "cybertruck", "elon musk"],
    "AMZN": ["amazon", "amzn", "aws", "alexa", "prime", "kindle"],
    "GOOGL": ["google", "googl", "alphabet", "android", "chrome", "youtube", "deepmind", "gemini", "bard", "waymo", "google cloud"],
    "META": ["facebook", "meta", "instagram", "whatsapp", "oculus", "threads", "reality labs", "mark zuckerberg"],
    "AMD": ["amd", "radeon", "ryzen", "epyc", "instinct", "hbm", "lisa su"],
    "INTC": ["intel", "intc", "core i7", "xeon", "arc gpu", "intel foundry", "pat gelsinger"],
    "PYPL": ["paypal", "pypl", "venmo"],
    "SQ": ["square", "sq", "cash app", "jack dorsey"],
    "SHOP": ["shopify", "shop", "ecommerce"],
    "NFLX": ["netflix", "nflx", "streaming", "original series"],
    "DIS": ["disney", "dis", "disney+", "disney plus", "marvel", "star wars", "espn", "hulu", "pixar"],
    "BA": ["boeing", "ba", "airplanes", "737 max", "787 dreamliner", "defense"],
    "GM": ["general motors", "gm", "chevy", "chevrolet", "cadillac", "gmc", "buick", "electric vehicles"],
    "F": ["ford", "f", "mustang", "f-150", "bronco", "ford lightning", "maverick"],
    "UBER": ["uber", "uber eats"],
    "LYFT": ["lyft"],
    "COIN": ["coinbase"],
    "HOOD": ["robinhood", "hood"],
    "C": ["citigroup", "citi", "citibank"],
    "ORCL": ["oracle", "cloud computing", "larry ellison"],
    "JPM": ["jpmorgan", "jpm", "chase", "jamie dimon"],
    "GS": ["goldman sachs"],
    "WFC": ["wells fargo", "wfc", "banking"],
    "BAC": ["bank of america", "bac", "boa"],
    "V": ["visa"],
    "MA": ["mastercard"],
}

# Regex for capturing $TICKER references, e.g., "$MSFT"
TICKER_PATTERN = re.compile(r"\$([A-Za-z]{1,5})")

def detect_stocks_in_text(text: str) -> set:
    """
    Return a set of all possible stock tickers found in the text.
    1) Look for $TICKER references (e.g., $MSFT).
    2) Also check for synonyms from STOCK_ALIASES (e.g., "apple" => "AAPL").
    """
    found = set()
    if not isinstance(text, str) or not text.strip():
        return found

    text_lower = text.lower()

    # 1) Regex-based $TICKER detection
    potential_tickers = TICKER_PATTERN.findall(text)
    for ticker_candidate in potential_tickers:
        ticker_candidate_up = ticker_candidate.upper()
        if ticker_candidate_up in STOCK_ALIASES:
            found.add(ticker_candidate_up)

    # 2) Synonym detection
    for symbol, synonyms in STOCK_ALIASES.items():
        for syn in synonyms:
            if syn in text_lower:
                found.add(symbol)

    return found

def assign_ticker_from_subreddit(subreddit: str) -> str:
    """
    Assign stock ticker based on subreddit name.
    Preserves general finance subreddits (stockmarket, stocks, investing, wallstreetbets)
    by returning None for them, so they stay in the dataset with 'UNKNOWN' if no ticker is matched.
    """
    if not isinstance(subreddit, str) or not subreddit.strip():
        return None

    subreddit_lower = subreddit.lower()

    # Match explicit keywords
    subreddit_fallback_map = {
        "apple": "AAPL",
        "microsoft": "MSFT",
        "nvidia": "NVDA",
    }
    for keyword, ticker in subreddit_fallback_map.items():
        if keyword in subreddit_lower:
            return ticker

    # Match ticker in subreddit name (e.g., NVDA_Stock -> NVDA)
    for ticker in STOCK_ALIASES.keys():
        if ticker in subreddit.upper():  # case-insensitive
            return ticker

    # Preserve well-known finance subreddits
    general_finance_subreddits = {"stockmarket", "stocks", "investing", "wallstreetbets"}
    if subreddit_lower in general_finance_subreddits:
        return None  # We'll handle these as UNKNOWN if no direct match is found

    return None  # Default to None if no match

def build_detection_text(row: pd.Series, is_post: bool = True) -> str:
    """
    Combine multiple text fields into one large string for detection coverage.
    For posts: merge title + selftext + text (if available).
    For comments: typically use 'text'.
    """
    if is_post:
        parts = []
        for col in ["title", "selftext", "text"]:
            val = row.get(col, "")
            if isinstance(val, str) and val.strip():
                parts.append(val.strip())
        return " \n ".join(parts)
    else:
        val = row.get("text", "")
        return val if isinstance(val, str) else ""

def transform_comments_df(df_comments: pd.DataFrame) -> pd.DataFrame:
    """
    Example transformation for comments.
    Adjust or remove as needed if your CSVs differ.
    """
    df_comments = df_comments.rename(columns={
        "comment_id": "id",
        "post_id": "post_id",
        "comment_author": "author",
        "comment_score": "score",
        "comment_created_utc": "created_utc"
    })
    # Mark as comment
    df_comments["type"] = "comment"

    # Ensure these columns exist
    desired_columns = [
        "id",
        "post_id",
        "subreddit",
        "author",
        "score",
        "created_utc",
        "type",
        "text",
        "special_char_ratio",
        "word_count",
        "unique_word_count",
        "sentiment",
    ]
    for col in desired_columns:
        if col not in df_comments.columns:
            df_comments[col] = None

    return df_comments[desired_columns]

def transform_posts_df(df_posts: pd.DataFrame) -> pd.DataFrame:
    """
    Example transformation for posts.
    Adjust or remove as needed if your CSVs differ.
    """
    df_posts = df_posts.rename(columns={
        "post_id": "id",
        "author": "author",
        "created_utc": "created_utc",
    })
    # Mark as post
    df_posts["type"] = "post"

    # Ensure these columns exist
    desired_columns = [
        "id",
        "subreddit",
        "author",
        "score",
        "created_utc",
        "type",
        "num_comments",
        "permalink",
        "title",
        "selftext",
        "text",
    ]
    for col in desired_columns:
        if col not in df_posts.columns:
            df_posts[col] = None

    return df_posts[desired_columns]

def build_combined_dataframe() -> pd.DataFrame:
    """
    Reads CSVs for Apple, Microsoft, NVIDIA, and general 'clean_reddit_stock_*.csv'.
    Merges them into one DataFrame, assigning 'resolved_stock' where applicable.
    Preserves rows/subreddits that don't match a known stock by setting them to 'UNKNOWN'.
    """

    # ---------------------------------------------------------------------
    # 1) Read CSV Data
    # ---------------------------------------------------------------------
    df_apple_posts = pd.read_csv("Apple_clean_reddit_stock_posts.csv")
    df_apple_comments = pd.read_csv("Apple_clean_reddit_stock_comments.csv")

    df_msft_posts = pd.read_csv("Microsoft_clean_reddit_stock_posts.csv")
    df_msft_comments = pd.read_csv("Microsoft_clean_reddit_stock_comments.csv")

    df_nvidia_posts = pd.read_csv("NVIDIA_clean_reddit_stock_posts.csv")
    df_nvidia_comments = pd.read_csv("NVIDIA_clean_reddit_stock_comments.csv")

    # These contain general subreddits like 'stockmarket', 'investing', etc.
    df_general_posts = pd.read_csv("clean_reddit_stock_posts.csv")
    df_general_comments = pd.read_csv("clean_reddit_stock_comments.csv")

    # ---------------------------------------------------------------------
    # 2) Transform DataFrames (optional, if columns differ)
    # ---------------------------------------------------------------------
    # Adjust if your CSVs already have standardized columns.
    df_apple_posts = transform_posts_df(df_apple_posts)
    df_apple_comments = transform_comments_df(df_apple_comments)

    df_msft_posts = transform_posts_df(df_msft_posts)
    df_msft_comments = transform_comments_df(df_msft_comments)

    df_nvidia_posts = transform_posts_df(df_nvidia_posts)
    df_nvidia_comments = transform_comments_df(df_nvidia_comments)

    df_general_posts = transform_posts_df(df_general_posts)
    df_general_comments = transform_comments_df(df_general_comments)

    # ---------------------------------------------------------------------
    # 3) Concatenate All Posts & Comments
    # ---------------------------------------------------------------------
    all_posts = pd.concat([
        df_apple_posts,
        df_msft_posts,
        df_nvidia_posts,
        df_general_posts
    ], ignore_index=True)

    all_comments = pd.concat([
        df_apple_comments,
        df_msft_comments,
        df_nvidia_comments,
        df_general_comments
    ], ignore_index=True)

    # ---------------------------------------------------------------------
    # 4) Assign 'resolved_stock' for Posts
    # ---------------------------------------------------------------------
    # a) First pass: subreddit-based match
    all_posts["resolved_stock"] = all_posts["subreddit"].apply(assign_ticker_from_subreddit)

    # b) Second pass: text-based match if still None
    missing_mask = all_posts["resolved_stock"].isna()
    detection_strings_posts = all_posts.apply(lambda row: build_detection_text(row, is_post=True), axis=1)
    all_posts.loc[missing_mask, "resolved_stock"] = detection_strings_posts[missing_mask].apply(
        lambda x: ",".join(detect_stocks_in_text(x)) if detect_stocks_in_text(x) else None
    )

    # ---------------------------------------------------------------------
    # 5) Assign 'resolved_stock' for Comments
    # ---------------------------------------------------------------------
    # a) First pass: subreddit-based match
    all_comments["resolved_stock"] = all_comments["subreddit"].apply(assign_ticker_from_subreddit)

    # b) Second pass: text-based match if still None
    missing_mask = all_comments["resolved_stock"].isna()
    detection_strings_comments = all_comments.apply(lambda row: build_detection_text(row, is_post=False), axis=1)
    all_comments.loc[missing_mask, "resolved_stock"] = detection_strings_comments[missing_mask].apply(
        lambda x: ",".join(detect_stocks_in_text(x)) if detect_stocks_in_text(x) else None
    )

    # ---------------------------------------------------------------------
    # 6) Combine into One Dataset
    # ---------------------------------------------------------------------
    combined_df = pd.concat([all_posts, all_comments], ignore_index=True)

    # ---------------------------------------------------------------------
    # 7) Final Fallback: Set NaN or Empty to "UNKNOWN"
    # ---------------------------------------------------------------------
    combined_df["resolved_stock"] = combined_df["resolved_stock"].fillna("UNKNOWN")
    combined_df.loc[combined_df["resolved_stock"] == "", "resolved_stock"] = "UNKNOWN"

    # ---------------------------------------------------------------------
    # 8) Write to CSV & Return
    # ---------------------------------------------------------------------
    combined_df.to_csv("combined_reddit_stock_data.csv", index=False)
    return combined_df

# ---------------------------------------------------------------------
# USAGE
# ---------------------------------------------------------------------
if __name__ == "__main__":
    df_final = build_combined_dataframe()
    print(f"Combined dataset has {len(df_final)} rows.")
    print(df_final["subreddit"].value_counts())
    print(df_final.sample(5))
