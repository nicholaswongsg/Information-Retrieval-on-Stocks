import os
import csv
import json
import time
import re
import pandas as pd
from openai import AzureOpenAI
from dotenv import load_dotenv

def main():
    """
    This script performs sentiment analysis on stock-related Reddit comments.
    It reads data from a CSV file, uses Azure OpenAI to classify sentiment,
    and saves the results to a new CSV with a checkpointing mechanism for resuming.
    """
    # Define the path to the .env file in the main directory
    env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".env"))

    # Load the environment variables from the specified .env file **BEFORE** creating AzureOpenAI client
    load_dotenv(dotenv_path=env_path)

    # Retrieve credentials from environment variables
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    api_version = os.getenv("OPENAI_API_VERSION")
    model_name = os.getenv("AZURE_OPENAI_MODEL_NAME")

    # Ensure credentials are loaded
    if not azure_endpoint or not api_key or not api_version or not model_name:
        raise ValueError("Missing required environment variables. Please check your .env file.")

    # Azure OpenAI configuration
    client = AzureOpenAI(
        azure_endpoint=azure_endpoint,
        api_key=api_key,
        api_version=api_version
    )

    # File paths
    input_file = "/Users/nicholaswong/Desktop/Github/Information-Retrieval-on-Stocks/Crawling/reddit_stock_comments.csv"
    output_file = "labelled_reddit_stock_comments.csv"
    checkpoint_file = "checkpoint.txt"

    # Read existing labelled file if it exists
    if os.path.exists(output_file):
        processed_df = pd.read_csv(output_file)
        processed_ids = set(processed_df["comment_id"].astype(str))
    else:
        processed_df = pd.DataFrame(columns=[
            "comment_id", "post_id", "subreddit", "comment_author", "comment_body", 
            "comment_score", "comment_created_utc", "text", "sentiment"
        ])
        processed_ids = set()

    # Read the main CSV
    df = pd.read_csv(input_file, dtype=str)

    # Filter rows that haven't been processed
    df = df[~df["comment_id"].astype(str).isin(processed_ids)]

    def analyze_sentiment(batch):
        """
        Sends a batch of comments to Azure OpenAI for sentiment analysis.
        Returns parsed JSON with comment IDs and sentiments in the format:
        [{"comment_id": <string>, "sentiment": "Positive"|"Negative"|"Neutral"}]
        """
        system_prompt = (
            "You are a helpful assistant that can analyze the sentiment of user comments. "
            "Classify each comment as Positive, Negative, or Neutral. "
            "Return the result as valid JSON WITHOUT any formatting like a markdown code block. "
            "Each element in the JSON array should look like: "
            '{"comment_id": <string>, "sentiment": <"Positive"|"Negative"|"Neutral">}.'
        )
        
        user_prompt = [
            f"{i+1}. Comment ID: {row['comment_id']} | Text: {row['comment_body']}"
            for i, row in enumerate(batch)
        ]
        joined_comments = "\n".join(user_prompt)
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Here are the comments:\n{joined_comments}\n"}
        ]
        
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=2000,
                temperature=0  # Deterministic output
            )
            

            if not response or not response.choices:
                print("Error: Empty response received from OpenAI API.")
                return []

            content = response.choices[0].message.content.strip()

            print("Raw API Response:", content)

            content = re.sub(r"^```json\n|\n```$", "", content.strip())

            print("Cleaned JSON Response:", content)

            # Attempt to parse JSON
            results = json.loads(content)
            return results
        except json.JSONDecodeError as parse_error:
            print(f"JSON Decode Error: {parse_error}. Response received: {content}")
            return []
        except Exception as e:
            print(f"Error calling Azure OpenAI API: {e}")
            return []
    
    # Process in batches
    batch_size = 50
    
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i+batch_size].to_dict(orient="records")
        try:
            results = analyze_sentiment(batch)
            if not results:
                print(f"Skipping batch {i//batch_size + 1} due to invalid or empty response.")
                continue
            
            for result in results:
                comment_id = str(result.get("comment_id"))
                sentiment = result.get("sentiment")
                df.loc[df["comment_id"] == comment_id, "sentiment"] = sentiment
            
            df.iloc[i:i+batch_size].to_csv(
                output_file, 
                mode='a', 
                index=False, 
                header=not os.path.exists(output_file)
            )
            
            with open(checkpoint_file, "w") as f:
                f.write(str(i + batch_size))
            
            print(f"Processed batch {i//batch_size + 1}")
            time.sleep(1)  # Avoid rate limiting
        except Exception as e:
            print(f"Error processing batch {i//batch_size + 1}: {str(e)}")
            break
    
    print("Sentiment analysis completed.")

if __name__ == "__main__":
    main()
