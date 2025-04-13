import os
import pickle
import nltk
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import pandas as pd
from collections import Counter, defaultdict
from tqdm import tqdm

lemmatizer = WordNetLemmatizer()
# Data path can be changed
data_path = "Classification/NER_with_sentiment.csv"
index_path = os.path.join(os.path.dirname(__file__), 'positional_index.pkl')

def lemmatize_text(text):
    if pd.isna(text):
        return ""
    text = str(text)
    tokens = word_tokenize(text)
    lemmatized_tokens = [lemmatizer.lemmatize(token.lower()) for token in tokens]
    return lemmatized_tokens # List of tokens

def collate_stopwords(documents):
    word_counter = Counter()
    print("Collating stopwords...")
    for text in tqdm(documents):
        word_counter.update(lemmatize_text(text))
    # Return most common 50 to list as stopwords (according to Zipf's Law)
    return [token for token, _ in word_counter.most_common(50)]

def build_index(documents, stopword_list):
    # Format: {term: {doc_id: [positions]}}
    print("Building positional index...")
    positional_index = defaultdict(lambda: defaultdict(list))

    for doc_id, text in enumerate(documents):
        tokens = lemmatize_text(text)
        for pos, term in enumerate(tokens):
            if term not in stopword_list:
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

# Save index
def persist_index(index, index_path):
    with open(index_path, "wb") as f:
        pickle.dump(index, f)

# Load index
def load_index(index_path):
    with open(index_path, "rb") as f:
        index = pickle.load(f)
        return index

def main():
    index_path = os.path.join(os.path.dirname(__file__), 'positional_index.pkl')
    # Load csv into memory
    df = pd.read_csv(data_path)
    documents = list(df['text']) # The index will be referring to this list

    # Process text to get stopword list from Zipf's Law
    stopword_list = collate_stopwords(documents)

    # Build index from documents
    # Load from path if index is already persistent
    index = load_index(index_path) if os.path.exists(index_path) else build_index(documents, stopword_list)
    print(type(index))
    # Query index (Can integrate interactivity downstream to configure IO for query)
    sample_query = "Apple stocks"
    result = query_index(index, sample_query, stopword_list)
    print("Matching documents:", result)
    
    # One-time: Persist index
    if not os.path.exists(index_path):
        persist_index(index, index_path)

if __name__ == "__main__":
    main()