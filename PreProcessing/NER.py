import os
import re
import json
import pandas as pd
import numpy as np
import logging
from tqdm import tqdm
import spacy

from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    AutoModelForSequenceClassification,
    pipeline
)

# Configure logging
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

#########################################################
# 1. CONFIGURATION & CONSTANTS
#########################################################

FINAL_CSV_PATH = "NER_with_sentiment.csv"
SOURCE_CSV_PATH = "combined_reddit_stock_data.csv"

nlp = spacy.load("en_core_web_sm")

TICKER_SYNONYMS = {
    "Citibank": ["citigroup", "citi", "citibank", "citi group", "citibanks"],
    "Visa": ["visa"],
    "Ford": ["ford", "mustang", "f-150", "bronco", "ford lightning", "maverick"],
    "AAPL": ["apple", "aapl", "iphone", "ipad", "macbook", "ios", "macos", "apple watch",
             "airpods", "tim cook"],
    "MSFT": ["microsoft", "msft", "windows", "azure", "xbox", "office", "copilot", "bing",
             "linkedin", "m365", "satya nadella"],
    "NVDA": ["nvidia", "nvda", "geforce", "rtx", "cuda", "hbm", "gb200", "jensen huang"],
    "TSLA": ["tesla", "tsla", "model 3", "model s", "model x", "model y", "cybertruck",
             "elon musk"],
    "AMZN": ["amazon", "amzn", "aws", "alexa", "prime", "kindle"],
    "GOOGL": ["google", "googl", "alphabet", "android", "chrome", "youtube", "deepmind",
              "gemini", "bard", "waymo", "google cloud"],
    "META": ["facebook", "meta", "instagram", "whatsapp", "oculus", "threads", "reality labs",
             "mark zuckerberg"],
    "AMD": ["amd", "radeon", "ryzen", "epyc", "instinct", "hbm", "lisa su"],
    "INTC": ["intel", "intc", "core i7", "xeon", "arc gpu", "intel foundry", "pat gelsinger"],
    "PYPL": ["paypal", "pypl", "venmo"],
    "SQ": ["square", "sq", "cash app", "jack dorsey"],
    "SHOP": ["shopify", "shop", "ecommerce"],
    "NFLX": ["netflix", "nflx", "streaming", "original series"],
    "DIS": ["disney", "dis", "disney+", "disney plus", "marvel", "star wars", "espn", "hulu",
            "pixar"],
    "BA": ["boeing", "ba", "airplanes", "737 max", "787 dreamliner", "defense"],
    "GM": ["general motors", "gm", "chevy", "chevrolet", "cadillac", "gmc", "buick",
           "electric vehicles"],
    "UBER": ["uber", "uber eats"],
    "LYFT": ["lyft"],
    "COIN": ["coinbase"],
    "HOOD": ["robinhood", "hood"],
    "ORCL": ["oracle", "cloud computing", "larry ellison"],
    "JPM": ["jpmorgan", "jpm", "chase", "jamie dimon"],
    "GS": ["goldman sachs"],
    "WFC": ["wells fargo", "wfc", "banking"],
    "BAC": ["bank of america", "bac", "boa"],
    "MA": ["mastercard"]
}

# OPTIONAL: loading an NER pipeline if you want to expand entity detection
logging.info("Loading optional NER pipeline...")
ner_checkpoint = "dslim/bert-base-NER"
ner_pipeline = pipeline(
    "ner",
    model=AutoModelForTokenClassification.from_pretrained(ner_checkpoint),
    tokenizer=AutoTokenizer.from_pretrained(ner_checkpoint),
    aggregation_strategy="simple"
)

logging.info("Loading FinBERT for financial sentiment...")
sentiment_checkpoint = "ProsusAI/finbert"
finbert_tokenizer = AutoTokenizer.from_pretrained(sentiment_checkpoint)
finbert_model = AutoModelForSequenceClassification.from_pretrained(sentiment_checkpoint)
finbert_pipeline = pipeline(
    "sentiment-analysis",
    model=finbert_model,
    tokenizer=finbert_tokenizer,
    return_all_scores=True,
    truncation=True,
    max_length=512
)


def finbert_five_level(scores_dict):
    """Map net_score = pos - neg to 5-level categories."""
    pos = scores_dict.get("positive", 0.0)
    neg = scores_dict.get("negative", 0.0)
    net_score = pos - neg
    if net_score >= 0.6:
        return "Strongly Positive"
    elif net_score >= 0.3:
        return "Slightly Positive"
    elif net_score > -0.3:
        return "Neutral"
    elif net_score > -0.6:
        return "Slightly Negative"
    else:
        return "Strongly Negative"


#########################################################
# 2. HELPER FUNCTIONS
#########################################################

def clean_text(text: str) -> str:
    """Remove HTML, non-ASCII, punctuation, extra spaces."""
    text = re.sub(r"<.*?>", "", text)
    text = text.encode("ascii", "ignore").decode()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def unify_synonyms_with_tickers(text: str) -> str:
    """
    Standardize company names instead of converting them to tickers.
    E.g. 'citi', 'citibank' → 'Citibank', 'tesla', 'model 3' → 'TSLA', etc.
    """
    standardization_map = {
        "Citibank": ["citigroup", "citi", "citibank", "citi group", "citibanks"],
        "Visa": ["visa"],
        "Ford": ["ford", "mustang", "f-150", "bronco", "ford lightning", "maverick"],
        "AAPL": ["apple", "aapl", "iphone", "ipad", "macbook", "ios", "macos", "apple watch",
                 "airpods", "tim cook"],
        "MSFT": ["microsoft", "msft", "windows", "azure", "xbox", "office", "copilot", "bing",
                 "linkedin", "m365", "satya nadella"],
        "NVDA": ["nvidia", "nvda", "geforce", "rtx", "cuda", "hbm", "gb200", "jensen huang"],
        "TSLA": ["tesla", "tsla", "model 3", "model s", "model x", "model y", "cybertruck",
                 "elon musk"],
        "AMZN": ["amazon", "amzn", "aws", "alexa", "prime", "kindle"],
        "GOOGL": ["google", "googl", "alphabet", "android", "chrome", "youtube", "deepmind",
                  "gemini", "bard", "waymo", "google cloud"],
        "META": ["facebook", "meta", "instagram", "whatsapp", "oculus", "threads", "reality labs",
                 "mark zuckerberg"],
        "AMD": ["amd", "radeon", "ryzen", "epyc", "instinct", "hbm", "lisa su"],
        "INTC": ["intel", "intc", "core i7", "xeon", "arc gpu", "intel foundry", "pat gelsinger"],
        "PYPL": ["paypal", "pypl", "venmo"],
        "SQ": ["square", "sq", "cash app", "jack dorsey"],
        "SHOP": ["shopify", "shop", "ecommerce"],
        "NFLX": ["netflix", "nflx", "streaming", "original series"],
        "DIS": ["disney", "dis", "disney+", "disney plus", "marvel", "star wars", "espn", "hulu",
                "pixar"],
        "BA": ["boeing", "ba", "airplanes", "737 max", "787 dreamliner", "defense"],
        "GM": ["general motors", "gm", "chevy", "chevrolet", "cadillac", "gmc", "buick",
               "electric vehicles"],
        "UBER": ["uber", "uber eats"],
        "LYFT": ["lyft"],
        "COIN": ["coinbase"],
        "HOOD": ["robinhood", "hood"],
        "ORCL": ["oracle", "cloud computing", "larry ellison"],
        "JPM": ["jpmorgan", "jpm", "chase", "jamie dimon"],
        "GS": ["goldman sachs"],
        "WFC": ["wells fargo", "wfc", "banking"],
        "BAC": ["bank of america", "bac", "boa"],
        "MA": ["mastercard"]
    }

    for standard_name, synonyms in standardization_map.items():
        for syn in synonyms:
            # Use word-boundary matching, ignoring case
            text = re.sub(rf"\b{re.escape(syn)}\b", standard_name, text, flags=re.IGNORECASE)

    return text


def run_finbert_sentiment(text: str):
    """Run FinBERT on text and return dict: {label: <5-level sentiment>, score: <float>}."""
    results = finbert_pipeline(text)[0]
    scores_dict = {d["label"].lower(): d["score"] for d in results}
    net_score = round(scores_dict["positive"] - scores_dict["negative"], 4)
    label = finbert_five_level(scores_dict)
    return {"label": label, "score": net_score}


def detect_tickers_in_text(text: str):
    """
    Return a list of recognized ticker strings from the text.
    We'll check "C" plus everything in TICKER_SYNONYMS keys.
    """
    recognized_tickers = []
    all_possible_tickers = list(TICKER_SYNONYMS.keys()) + ["C"]

    for ticker in all_possible_tickers:
        pattern = rf"\b{ticker}\b"
        if re.search(pattern, text, flags=re.IGNORECASE):
            recognized_tickers.append(ticker)

    return list(set(recognized_tickers))


#########################################################
# 3. MAIN PROCESSING
#########################################################

def process_post_row(row):
    """
    Process a single post row: 
      - unify synonyms
      - detect tickers
      - run sentiment (once for entire text)
    Returns updated row and recognized_tickers.
    """
    original_text = str(row.get("text", ""))
    cleaned_text = clean_text(original_text)
    unified_text = unify_synonyms_with_tickers(cleaned_text)

    recognized_tickers = detect_tickers_in_text(unified_text)

    # If we found tickers, run sentiment
    if recognized_tickers:
        sentiment_result = run_finbert_sentiment(unified_text)
        # replicate same label/score for all recognized tickers
        ticker_sentiments = {
            t: sentiment_result for t in recognized_tickers
        }
    else:
        ticker_sentiments = {}

    # Store them as strings
    row["ner_text_cleaned"] = unified_text
    row["ner_recognized_tickers"] = ",".join(sorted(recognized_tickers))
    row["ner_entity_sentiments"] = json.dumps(ticker_sentiments)
    return row, recognized_tickers


def process_comment_row(row, post_ticker_dict):
    """
    Process a single comment row:
      - unify synonyms
      - detect tickers
      - If no tickers recognized, fallback to parent's recognized tickers (post_ticker_dict)
      - run sentiment on the comment text (once)
    Returns the updated row.
    """
    original_text = str(row.get("text", ""))
    cleaned_text = clean_text(original_text)
    unified_text = unify_synonyms_with_tickers(cleaned_text)

    recognized_tickers = detect_tickers_in_text(unified_text)

    # If no recognized tickers, check parent's recognized tickers
    parent_post_id = row.get("post_id", None)
    if not recognized_tickers and parent_post_id in post_ticker_dict:
        recognized_tickers = post_ticker_dict[parent_post_id]

    # If we have tickers, run sentiment on this comment text
    if recognized_tickers:
        sentiment_result = run_finbert_sentiment(unified_text)
        ticker_sentiments = {
            t: sentiment_result for t in recognized_tickers
        }
    else:
        ticker_sentiments = {}

    row["ner_text_cleaned"] = unified_text
    row["ner_recognized_tickers"] = ",".join(sorted(recognized_tickers))
    row["ner_entity_sentiments"] = json.dumps(ticker_sentiments)

    return row


def main():
    # 1) Load existing final CSV if it exists
    if os.path.exists(FINAL_CSV_PATH):
        existing_df = pd.read_csv(FINAL_CSV_PATH, dtype=str)
        processed_ids = set(existing_df["id"].unique())
        logging.info(f"Loaded {len(existing_df)} rows. Found {len(processed_ids)} unique processed IDs.")
    else:
        existing_df = pd.DataFrame()
        processed_ids = set()
        logging.info("No existing results found. Starting fresh.")

    # 2) Load the source CSV
    df = pd.read_csv(SOURCE_CSV_PATH, dtype=str)
    logging.info(f"Loaded {len(df)} rows from {SOURCE_CSV_PATH}")

    # Exclude rows already processed
    df = df[~df["id"].isin(processed_ids)]
    logging.info(f"Number of new rows to process: {len(df)}")
    if df.empty:
        logging.info("No new rows to process. Exiting.")
        return

    # 3) Separate posts vs comments
    posts_df = df[df["type"] == "post"].copy()
    comments_df = df[df["type"] == "comment"].copy()

    # 4) Process all posts first
    post_ticker_dict = {}  # dict: post_id -> list of recognized tickers
    post_results = []
    for _, row_data in tqdm(posts_df.iterrows(), total=len(posts_df), desc="Processing posts"):
        updated_row, recognized_tickers = process_post_row(row_data)
        post_results.append(updated_row)
        if recognized_tickers:
            post_ticker_dict[updated_row["id"]] = recognized_tickers
        else:
            # If no recognized tickers, store empty list
            post_ticker_dict[updated_row["id"]] = []

    # 5) Process comments
    comment_results = []
    for _, row_data in tqdm(comments_df.iterrows(), total=len(comments_df), desc="Processing comments"):
        updated_comment = process_comment_row(row_data, post_ticker_dict)
        comment_results.append(updated_comment)

    # 6) Combine posts + comments results
    results_df = pd.DataFrame(post_results + comment_results)

    # 7) If there was an existing_df, union them
    if not existing_df.empty:
        final_df = pd.concat([existing_df, results_df], ignore_index=True)
    else:
        final_df = results_df

    # 8) Save to CSV
    final_df.to_csv(FINAL_CSV_PATH, index=False)
    logging.info(f"Done processing. Appended results to {FINAL_CSV_PATH}")


if __name__ == "__main__":
    main()
