import os
import csv
import json
import time
import re
import pandas as pd
from dotenv import load_dotenv
import subprocess

# Load environment variables
load_dotenv("../.env")

print("Testing if Qwen is running locally...")
def test_qwen():
    try:
        result = subprocess.run(
            ['ollama', 'run', 'qwen2.5:14b'],
            input='Hi',
            capture_output=True, text=True, check=True
        )
        print("Qwen test response:", result.stdout.strip())
    except Exception as e:
        print(f"Error during Qwen test: {e}")
        exit()

test_qwen()

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


def call_ollama_api(prompt):
    """
    Pseudo-stream approach: Use Popen and read lines from stdout
    without using --stream (since your Ollama version doesn't support it).
    """
    try:
        print("\nRunning Ollama locally using Qwen model (no --stream)...")
        proc = subprocess.Popen(
            ['ollama', 'run', 'qwen2.5:14b'],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        # Send prompt
        proc.stdin.write(prompt)
        proc.stdin.close()

        output_lines = []
        print("Reading partial lines from Ollama. May buffer until generation is done.\n")
        # Read line by line until process ends
        for line in proc.stdout:
            print("[ollama output]", line.rstrip())
            output_lines.append(line)

        proc.wait()

        # Check stderr
        error_output = proc.stderr.read().strip()
        if error_output:
            print("stderr from Ollama:", error_output)

        raw = "".join(output_lines).strip()
        print("\n--- Full raw response (first 1000 chars) ---")
        print(raw[:1000], "..." if len(raw) > 1000 else "")
        print("--- End raw response ---\n")

        # Remove backticks
        raw = re.sub(r'^```(json|javascript)\n|\n```$', '', raw)

        return json.loads(raw)

    except json.JSONDecodeError as e:
        print(f"JSON decoding error: {e}")
        return None
    except Exception as e:
        print(f"Error calling Ollama API: {e}")
        return None


def process_batch(batch_records, df):
    if not batch_records:
        return

    system_prompt = (
        "You are a helpful assistant that analyzes sentiment in Reddit stock comments.\n\n"
        "You MUST output strictly valid JSON—no additional text, disclaimers, or explanations.\n"
        "If multiple rows are provided, respond with a JSON array. Each element is an object.\n"
        "If a row has no recognized ticker symbols, include an empty object for 'qwen_sentiments'.\n\n"
        "Output format example:\n"
        "[{\n"
        '  \"id\": \"123\",\n'
        '  \"qwen_sentiments\": {\n'
        '    \"AAPL\": {\"qwen\": {\"label\": \"Neutral\"}}\n'
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

    user_prompt = []
    for row in batch_records:
        identifier = row["id"]
        tickers = row.get("ner_recognized_tickers", "")
        text = row.get("ner_text_cleaned", "")
        user_prompt.append(f"Row:\n id={identifier}\n tickers={tickers}\n text={text}")

    full_prompt = f"{system_prompt}\n\nHere are the rows:\n" + "\n".join(user_prompt)
    results = call_ollama_api(full_prompt)

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

print("Sentiment analysis completed using Qwen2.5:14B.")
