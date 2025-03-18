import pandas as pd
import os
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from tqdm import tqdm
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain.schema import Document
from langchain_core.messages import SystemMessage, HumanMessage
import json
import nltk

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Define lemmatization function
def lemmatize_text(text):
    if pd.isna(text):
        return ""
    text = str(text)
    tokens = word_tokenize(text)
    lemmatized_tokens = [lemmatizer.lemmatize(token.lower()) for token in tokens]
    return ' '.join(lemmatized_tokens)

"""Process the dataset and create a vector store"""
# Load and clean the dataset
print("Loading and cleaning dataset...")
data_path = os.path.join(os.path.dirname(__file__), '..', 'PreProcessing', 'combined_reddit_stock_data.csv')
df = pd.read_csv(data_path)
df = df[['title', 'selftext', 'text', 'post_id','id']]

# Apply lemmatization
print("Lemmatizing text...")
df['title'] = df['title'].apply(lemmatize_text)
df['selftext'] = df['selftext'].apply(lemmatize_text)
df['text'] = df['text'].apply(lemmatize_text)

# Convert to documents
print("Converting to documents...")
documents = []
for i, row in df.iterrows():
    content = f"Title: {row['title']}\n"
    content += f"Selftext: {row['selftext']}\n" if not pd.isna(row['selftext']) else ""
    content += f"Text: {row['text']}\n" if not pd.isna(row['text']) else ""

    if pd.isna(row['post_id']):
        source = row['id']
    else:
        source = row['post_id']
        
    doc = Document(page_content=content, metadata={"source": f"{source}"})
    documents.append(doc)

# Split into chunks with appropriate size
print("Splitting documents into chunks...")
chunk_size = 1000
chunk_overlap = 400
text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
chunks = text_splitter.split_documents(documents)
print(f"Created {len(chunks)} chunks")


print("Chunk 1 -----------------------")
print(chunks[0])
print("Chunk 2 -----------------------")
print(chunks[1])
