import os
import pickle
# import nltk
# nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import pandas as pd
from collections import Counter
from tqdm import tqdm
import regex as re

lemmatizer = WordNetLemmatizer()
# Data path can be changed
data_path = "Classification/NER_with_sentiment.csv"
index_path = os.path.join(os.path.dirname(__file__), 'positional_index.pkl')
stopwords_path = os.path.join(os.path.dirname(__file__), 'stopwords.pkl')
df_path = os.path.join(os.path.dirname(__file__), 'dataframe.pkl')

def lemmatize_text(text):
    if pd.isna(text):
        return ""
    text = str(text)
    
    # Remove punctuations but preserve decimal numbers
    text = re.sub(r'[^\w\s.]', '', text)
    # Fix standalone periods (not part of numbers)
    text = re.sub(r'(?<!\d)\.(?!\d)', '', text)
    
    tokens = word_tokenize(text)
    # Only include alphabetic tokens or numeric tokens with at most one decimal point
    lemmatized_tokens = [lemmatizer.lemmatize(token.lower()) for token in tokens 
                         if token.isalpha() or (token.replace('.', '', 1).isdigit() and token.count('.') <= 1)]
    
    return lemmatized_tokens  # List of tokens

def collate_stopwords(documents):
    word_counter = Counter()
    print("Collating stopwords...")
    for text in tqdm(documents):
        # Only count alphabetic tokens for stopwords (no numbers or punctuation)
        alphabetic_tokens = [token for token in lemmatize_text(text) if token.isalpha()]
        word_counter.update(alphabetic_tokens)
    
    # Return most common 50 to list as stopwords (according to Zipf's Law)
    return [token for token, _ in word_counter.most_common(50)]

def build_index(documents, stopword_list):
    # Format: {term: {doc_id: [positions]}}
    print("Building positional index...")
    # Use regular dict to avoid pickling issues
    positional_index = {}

    for doc_id, text in enumerate(documents):
        tokens = lemmatize_text(text)
        for pos, term in enumerate(tokens):
            # Additional check to ensure no punctuation in the index
            if term not in stopword_list and (term.isalpha() or (term.replace('.', '', 1).isdigit() and term.count('.') <= 1)):
                if term not in positional_index:
                    positional_index[term] = {}
                if doc_id not in positional_index[term]:
                    positional_index[term][doc_id] = []
                positional_index[term][doc_id].append(pos)

    return positional_index

def query_index(index, query, stopword_list, mode='and'):
    mode = mode.lower()
    if mode not in ['and', 'or']:
        return "Mode not found. Supported query modes: ['and', 'or']"

    # Preprocess query
    query_terms = [t for t in lemmatize_text(query) if t not in stopword_list]

    # Boolean retrieval: Combine based on mode
    postings = [set(index.get(term, {})) for term in query_terms]
    
    if not postings: # No matches found
        return set()
    
    if mode == 'and':
        return set.intersection(*postings)
    elif mode == 'or':
        return set.union(*postings)

# Save index, stopwords, and the entire dataframe
def persist_data(index, stopword_list, df):
    with open(index_path, "wb") as f:
        pickle.dump(index, f)
    
    with open(stopwords_path, "wb") as f:
        pickle.dump(stopword_list, f)
    
    with open(df_path, "wb") as f:
        pickle.dump(df, f)

# Load index, stopwords, and the entire dataframe
def load_data():
    with open(index_path, "rb") as f:
        index = pickle.load(f)
    
    with open(stopwords_path, "rb") as f:
        stopword_list = pickle.load(f)
    
    with open(df_path, "rb") as f:
        df = pickle.load(f)
    
    return index, stopword_list, df

def search_index(data_path,query, mode='and'):
    # Check if all required files exist
    files_exist = (
        os.path.exists(index_path) and 
        os.path.exists(stopwords_path) and 
        os.path.exists(df_path)
    )
    
    if files_exist:
        # Load existing data
        index, stopword_list, df = load_data()
    else:
        # Build new index and save data
        df = pd.read_csv(data_path)
        documents = list(df['text'])
        stopword_list = collate_stopwords(documents)
        index = build_index(documents, stopword_list)
        persist_data(index, stopword_list, df)

    # Query the index
    result_doc_ids = query_index(index, query, stopword_list, mode)
    
    # Return both IDs and the complete rows from the original dataframe
    result_documents = df.iloc[list(result_doc_ids)].copy() if result_doc_ids else pd.DataFrame()
    return stopword_list, result_documents

def main():
    # Query index
    sample_query = "Apple stocks"
    stopword_list, result_documents = search_index(data_path,sample_query)
    print(f"Found {len(result_documents)} matching documents")
    print("Stopwords:")
    print(stopword_list)
    
    # Display a sample of the matching documents with all fields
    if not result_documents.empty:
        print("\nSample of matching documents (first 3):")
        print(result_documents[['title']].head(3))
        
        # Show the column names available in the original dataset
        print("\nAvailable columns in the dataset:")
        print(result_documents.columns.tolist())

if __name__ == "__main__":
    main()
