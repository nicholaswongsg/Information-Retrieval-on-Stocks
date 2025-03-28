import os
import csv
import json
import time
import re
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables (must include DASHSCOPE_API_KEY, etc.)
load_dotenv("../.env")

# Instantiate the DashScope-compatible OpenAI client
client = OpenAI(
    # If no environment variable, you can hardcode:
    # api_key="sk-xxx",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# For debugging: confirm we actually got the key
print("DASHSCOPE_API_KEY:", os.getenv("DASHSCOPE_API_KEY"))

file_path = "NER_with_sentiment.csv"
df = pd.read_csv(file_path, dtype=str)

if "qwen_sentiments" not in df.columns:
    df["qwen_sentiments"] = None

if "unique_word_count" in df.columns:
    df["unique_word_count"] = pd.to_numeric(df["unique_word_count"], errors="coerce")

filled_count_start = df["qwen_sentiments"].notna().sum()
print(f"At the start, we have {filled_count_start} rows with 'qwen_sentiments' filled in, out of {len(df)} total rows.")

df_missing_sentiment = df[df["qwen_sentiments"].isna()]

if df_missing_sentiment.empty:
    print("No missing sentiment data to process.")
    exit()

if "unique_word_count" in df_missing_sentiment.columns:
    df_missing_sentiment = df_missing_sentiment.sort_values(
        by="unique_word_count", ascending=False, na_position="last"
    )

df_missing_sentiment = df_missing_sentiment.reset_index(drop=True)

def call_qwen_api(prompt_text):
    """
    Call DashScope's Qwen model via 'openai' client, using the doc-style 'content' array.
    """
    try:
        # We put all instructions or text under a single "user" message with a "text" type.
        # If you had images, you'd add another dict with {"type": "image_url", "image_url": {"url": ...}}.
        
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt_text
                    }
                ]
            }
        ]

        # Model name from doc is "qwen-vl-plus", but you can switch if needed:
        response = client.chat.completions.create(
            model="qwen2.5-32b-instruct",
            messages=messages
        )

        # If you want to see the raw JSON of the entire response:
        # print(response.model_dump_json())

        # The assistant’s textual answer is in choices[0].message.content,
        # which should be a string. If your model is returning JSON in that string, you can parse it.
        raw_response = response.choices[0].message.content.strip()
        print("\n--- Raw response ---")
        print(raw_response)
        print("--- End Raw response ---\n")

        # Optionally remove code fences:
        raw_response = re.sub(r'^```(json|javascript)\n|\n```$', '', raw_response)

        # Attempt to parse if you expect well-formed JSON
        return json.loads(raw_response)

    except json.JSONDecodeError as e:
        print(f"JSON decoding error: {e}")
        return None
    except Exception as e:
        print(f"Error calling Qwen API: {e}")
        return None

def process_batch(batch_records, df):
    if not batch_records:
        return

    # If you still need “system”-style instructions, just prepend them into the text below.
    # E.g.: instructions + the row data.
    instructions = (
        "You are a helpful assistant that analyzes sentiment in Reddit stock comments.\n\n"
        "You MUST output strictly valid JSON—no additional text or disclaimers.\n"
        "If multiple rows are provided, respond with a JSON array. Each element is an object.\n"
        "If a row has no recognized ticker symbols, include an empty object for 'qwen_sentiments'.\n\n"
        "Output format example:\n"
        "[{\n"
        '  "id": "123",\n'
        '  "qwen_sentiments": {\n'
        '    "AAPL": {"qwen": {"label": "Neutral"}}\n'
        "  }\n"
        "}]\n\n"
        "Your task:\n"
        "1) For each row, read its 'id', 'ner_recognized_tickers', 'ner_text_cleaned'.\n"
        "2) For each recognized ticker, classify the text sentiment as:\n"
        "   - Strongly Positive\n"
        "   - Slightly Positive\n"
        "   - Neutral\n"
        "   - Slightly Negative\n"
        "   - Strongly Negative\n"
        "3) Return strictly valid JSON with one object per row.\n"
    )

    # Build the user “text” with your row data
    rows_text = []
    for row in batch_records:
        identifier = row["id"]
        tickers = row.get("ner_recognized_tickers", "")
        text = row.get("ner_text_cleaned", "")
        # Each row is appended as a textual block
        rows_text.append(f"Row:\n id={identifier}\n tickers={tickers}\n text={text}")

    # Combine instructions + row data
    full_prompt = instructions + "\n\nHere are the rows:\n" + "\n".join(rows_text)

    results = call_qwen_api(full_prompt)
    if results is None or not isinstance(results, list):
        print(f"Failed to retrieve valid JSON for batch of size {len(batch_records)}.")
        return

    for result in results:
        row_id = str(result.get("id"))
        sentiments_obj = result.get("qwen_sentiments", {})
        sentiments_str = json.dumps(sentiments_obj)
        df.loc[df["id"] == row_id, "qwen_sentiments"] = sentiments_str

    df.to_csv(file_path, index=False)
    print(f"Processed and saved {len(results)} rows.")

# Main loop
batch_size = 1
num_missing = len(df_missing_sentiment)
num_batches = (num_missing // batch_size) + (1 if num_missing % batch_size else 0)

for i in range(num_batches):
    start_idx = i * batch_size
    end_idx = start_idx + batch_size
    batch_df = df_missing_sentiment.iloc[start_idx:end_idx]
    batch_records = batch_df.to_dict(orient="records")

    print(f"\nProcessing batch {i+1}/{num_batches} (rows {start_idx} to {end_idx-1})")
    process_batch(batch_records, df)
    time.sleep(1)

print("Sentiment analysis completed using the DashScope Qwen API (doc-style).")
