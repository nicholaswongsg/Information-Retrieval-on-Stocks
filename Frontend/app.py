import streamlit as st
import pandas as pd
import ast
import os

################################################################################
# HELPER FUNCTIONS
################################################################################

def load_data(csv_path):
    """Load the CSV containing NER and sentiment info."""
    return pd.read_csv(csv_path)

def parse_tickers(tickers_string):
    """
    Parse comma-separated tickers (e.g. 'AAPL,TSLA')
    into a clean list of ticker symbols.
    """
    if not isinstance(tickers_string, str) or not tickers_string.strip():
        return []
    return [ticker.strip() for ticker in tickers_string.split(",") if ticker.strip()]

def load_existing_human_sentiments(human_sentiments_value):
    """
    Convert the existing 'human_sentiments' column (string) 
    back into a Python dict or {} if empty/invalid.
    """
    if not isinstance(human_sentiments_value, str) or not human_sentiments_value.strip():
        return {}
    try:
        return ast.literal_eval(human_sentiments_value)
    except:
        return {}

def save_data(df, csv_path):
    """Save the modified DataFrame back to CSV."""
    df.to_csv(csv_path, index=False)

def remove_id_from_file(id_to_remove, txt_path):
    """Remove a single ID from the labeller's text file."""
    if not os.path.exists(txt_path):
        return
    with open(txt_path, "r") as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines if line.strip()]

    # Re-write all lines except the removed ID
    with open(txt_path, "w") as f:
        for line in lines:
            if line != str(id_to_remove):
                f.write(line + "\n")

################################################################################
# STREAMLIT APP
################################################################################
def main():
    st.title("NER Text Sentiment Labeling")

    # Path to your main CSV file
    csv_path = "../Classification/NER_with_sentiment.csv"
    if not os.path.exists(csv_path):
        st.error(f"CSV file not found at {csv_path}. Please check the path.")
        return

    # Ask who is labeling
    labellers = ["lewis", "nicholas", "zoey", "zhijian"]
    labeller_choice = st.selectbox("Who is labeling?", ["(Select your name)"] + labellers)
    if labeller_choice == "(Select your name)":
        st.warning("Please select your name to start labeling.")
        st.stop()

    # Define the path to the labeller's ID list
    labeller_txt_path = f"../Classification/{labeller_choice}.txt"
    if not os.path.exists(labeller_txt_path):
        st.error(f"Text file not found at {labeller_txt_path} for '{labeller_choice}'.")
        st.stop()

    # Load the main CSV
    df = load_data(csv_path)

    # Load IDs from the labeller's text file
    with open(labeller_txt_path, "r") as f:
        labeller_ids = [line.strip() for line in f if line.strip()]

    # Filter the DataFrame to only those rows whose 'id' is in the labeller_ids
    df_filtered = df[df["id"].astype(str).isin(labeller_ids)].copy()
    df_filtered.reset_index(drop=True, inplace=True)

    if len(df_filtered) == 0:
        st.info("No rows to label for this user (all done or no matching IDs).")
        return

    # Use a session state variable to track which row index we're on
    if "current_row_index" not in st.session_state:
        st.session_state["current_row_index"] = 0

    # If we've gone beyond the last row, show 'All done!'
    if st.session_state["current_row_index"] >= len(df_filtered):
        st.success("All done! No more records to label.")
        return

    # Retrieve the current row to label
    row = df_filtered.loc[st.session_state["current_row_index"]]

    # Display the row's text
    st.markdown("---")
    st.subheader(f"Row Index in Filtered Data: {st.session_state['current_row_index']}")
    st.write(f"**ID**: {row['id']}")
    ner_text_cleaned = row.get("ner_text_cleaned", "")
    st.write(f"**Text**: {ner_text_cleaned}")

    # Parse recognized tickers
    recognized_tickers = parse_tickers(row.get("ner_recognized_tickers", ""))

    if not recognized_tickers:
        st.write("No recognized tickers found in this text.")
    else:
        st.write("### Please select a sentiment for each recognized ticker:")
        # Load any existing 'human_sentiments' so that if we re-label, we keep old values
        existing_dict = load_existing_human_sentiments(row.get("human_sentiments", ""))
        # If there's no valid dictionary, create an empty one
        if not isinstance(existing_dict, dict):
            existing_dict = {}

        # We'll store the user's new selections in this dictionary
        new_sentiment_dict = {}

        # Define the options for sentiment
        sentiment_options = [
            "Strongly Positive",
            "Slightly Positive",
            "Neutral",
            "Slightly Negative",
            "Strongly Negative"
        ]

        # For each recognized ticker, let the user pick a sentiment
        for ticker in recognized_tickers:
            default_label = "Neutral"
            if ticker in existing_dict and "human" in existing_dict[ticker]:
                default_label = existing_dict[ticker]["human"].get("label", "Neutral")

            selected_label = st.selectbox(
                f"Sentiment for {ticker}:", 
                sentiment_options,
                index=sentiment_options.index(default_label) if default_label in sentiment_options else 2
            )
            new_sentiment_dict[ticker] = {"human": {"label": selected_label}}

    # Button to submit the label and move to the next record
    if st.button("Submit Label"):
        # Update the main DataFrame (df) for this row's 'human_sentiments'
        # - We need to find the original index in df
        matching_idx_list = df.index[df["id"] == row["id"]].tolist()
        if matching_idx_list:
            main_idx = matching_idx_list[0]
            # If there are tickers, use the new sentiment dictionary. If none, just store '{}'
            df.at[main_idx, "human_sentiments"] = str(new_sentiment_dict) if recognized_tickers else "{}"

            # Save changes to CSV
            save_data(df, csv_path)

            # Remove this ID from the labeller's .txt file so it won't reappear
            remove_id_from_file(row["id"], labeller_txt_path)

        # Move to the next row
        st.session_state["current_row_index"] += 1
        st.st.rerun()

    st.markdown("---")
    st.write("Use the **Submit Label** button above to save this record and move on to the next.")

if __name__ == "__main__":
    main()
