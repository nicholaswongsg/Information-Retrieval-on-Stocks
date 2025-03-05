import os
import json
import logging
from datetime import datetime

import pandas as pd
import tqdm

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.retrievers import ContextualCompressionRetriever

# Updated import for Cohere and HuggingFace Embeddings
from langchain_cohere.llms import Cohere
from langchain_cohere import CohereRerank
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('vectorstore_creation.log'),
        logging.StreamHandler()
    ]
)

# File constants
CHECKPOINT_FILE = 'vectorstore_checkpoint.json'
PERSIST_DIRECTORY = os.path.join(os.path.dirname(__file__), 'vectorstore')
DOCS_FILE = "processed_documents.json"  # File to store processed documents

def save_checkpoint(stage: str, details: dict = None):
    """Save checkpoint information"""
    checkpoint_data = {
        'timestamp': datetime.now().isoformat(),
        'stage': stage,
        'details': details or {}
    }
    try:
        with open(CHECKPOINT_FILE, 'w') as f:
            json.dump(checkpoint_data, f, indent=4)
        logging.info(f"Checkpoint saved: {stage}")
    except Exception as e:
        logging.error(f"Failed to save checkpoint: {e}")

def load_checkpoint():
    """Load the last checkpoint"""
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"Failed to load checkpoint: {e}")
    return None

def document_to_dict(doc: Document) -> dict:
    """Convert a Document object to a dictionary for JSON serialization."""
    return {
        "page_content": doc.page_content,
        "metadata": doc.metadata
    }

def document_from_dict(d: dict) -> Document:
    """Convert a dictionary back to a Document object."""
    return Document(page_content=d["page_content"], metadata=d["metadata"])

def load_processed_docs():
    """Load previously processed documents to avoid re-processing."""
    if os.path.exists(DOCS_FILE):
        try:
            with open(DOCS_FILE, "r") as f:
                docs_data = json.load(f)
            # Convert list of dicts back to Document objects
            return [document_from_dict(d) for d in docs_data]
        except Exception as e:
            logging.error(f"Failed to load processed documents: {e}")
    return None

def save_processed_docs(docs):
    """Save processed documents (as dicts) to avoid re-processing."""
    try:
        with open(DOCS_FILE, "w") as f:
            # Convert each Document to a dict before saving
            json.dump([document_to_dict(doc) for doc in docs], f, indent=4)
        logging.info("Processed documents saved.")
    except Exception as e:
        logging.error(f"Failed to save processed documents: {e}")

# Subclass Cohere to remove duplicate "model" parameter.
class FixedCohere(Cohere):
    def __init__(self, **kwargs):
        # Remove "model" from the kwargs so it isn’t passed twice
        model = kwargs.pop("model", None)
        super().__init__(**kwargs)
        self.model = model

    def _call(self, prompt, stop=None, run_manager=None, **kwargs):
        # Ensure no 'model' in kwargs to avoid duplicate parameter
        kwargs.pop("model", None)
        return super()._call(prompt, stop=stop, run_manager=run_manager, **kwargs)

def main():
    checkpoint = load_checkpoint()
    if checkpoint:
        logging.info(f"Loaded checkpoint: {checkpoint['stage']} at {checkpoint['timestamp']}")

    # Specify CSV file path
    data_path = os.path.join(os.path.dirname(__file__), '..', 'PreProcessing', 'combined_reddit_stock_data.csv')

    # Fix for DtypeWarning when reading CSV by loading all columns as string
    try:
        data = pd.read_csv(data_path, dtype=str, low_memory=False)
        logging.info(f"Data loaded successfully. Shape: {data.shape}")
        logging.info(f"Columns: {list(data.columns)}")
        save_checkpoint('data_loaded', {'rows': len(data), 'columns': list(data.columns)})
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise

    # Initialize HuggingFace Embeddings using the SentenceTransformer model recommended by HuggingFace.
    try:
        embedding = HuggingFaceEmbeddings(model_name="deepset/all-mpnet-base-v2-table")
        save_checkpoint('embeddings_initialized')
    except Exception as e:
        logging.error(f"Failed to initialize embeddings: {e}")
        raise

    # Load processed documents if they exist, else create them
    docs = load_processed_docs()
    if not docs:
        try:
            docs = []
            splitter = RecursiveCharacterTextSplitter(chunk_size=2048, chunk_overlap=100)
            for i in tqdm.tqdm(range(len(data))):
                if pd.isna(data['text'][i]) or not data['text'][i].strip():
                    continue  # Skip empty text

                metadata = {
                    col: str(data[col][i]) if not pd.isna(data[col][i]) else '' 
                    for col in data.columns if col != 'text'
                }
                for chunk in splitter.split_text(str(data['text'][i])):
                    docs.append(Document(
                        page_content=chunk, 
                        metadata=metadata
                    ))
            logging.info(f"Number of documents created: {len(docs)}")
            save_checkpoint('documents_created', {'total_documents': len(docs)})
            save_processed_docs(docs)
        except Exception as e:
            logging.error(f"Failed during document creation: {e}")
            raise
    else:
        logging.info("Loaded previously processed documents. Skipping document chunking.")

    # Create persist directory if it doesn't exist
    os.makedirs(PERSIST_DIRECTORY, exist_ok=True)

    # Load or create Chroma Vector Store
    if os.path.exists(os.path.join(PERSIST_DIRECTORY, "chroma.sqlite")):
        try:
            vectordb = Chroma(
                persist_directory=PERSIST_DIRECTORY, 
                embedding_function=embedding
            )
            logging.info("Loaded existing vector store from disk.")
        except Exception as e:
            logging.error(f"Failed to load existing vector store: {e}")
            raise
    else:
        try:
            sample_docs = docs[:100]  # Use a subset for the vector store
            vectordb = Chroma.from_documents(
                documents=sample_docs, 
                persist_directory=PERSIST_DIRECTORY, 
                embedding=embedding
            )
            save_checkpoint('vector_store_created', {'persist_directory': PERSIST_DIRECTORY})
        except Exception as e:
            logging.error(f"Failed to create vector store: {e}")
            raise

    save_checkpoint('vector_db_loaded')

    # Use ConversationBufferMemory (Deprecation warnings may appear; see LangChain migration guide)
    memory = ConversationBufferMemory(return_messages=True)

    # Configure Retriever
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})

    # Load Cohere API key from environment
    cohere_api_key = os.getenv('COHERE_API_KEY')
    if not cohere_api_key:
        logging.error("Cohere API key not set")
        raise ValueError("Please set the COHERE_API_KEY environment variable")

    # Initialize Cohere services.
    try:
        llm = FixedCohere(model="command-r", cohere_api_key=cohere_api_key)
        compressor = CohereRerank(model="rerank-english-v2.0", cohere_api_key=cohere_api_key)
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, 
            base_retriever=retriever
        )
        save_checkpoint('cohere_initialized')
    except Exception as e:
        logging.error(f"Failed to initialize Cohere services: {e}")
        raise

    # Create Conversational Retrieval Chain
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm, 
        memory=memory,
        retriever=compression_retriever
    )

    # Function to query and get reranked results
    def query_with_reranking(query: str, top_k: int = 3):
        compressed_docs = compression_retriever.get_relevant_documents(query)
        source_df = pd.DataFrame([
            {
                'Text': doc.page_content, 
                'id': doc.metadata.get('id', ''), 
                'title': doc.metadata.get('title', ''),
                'relevance_score': getattr(doc, 'relevance_score', None)
            }
            for doc in compressed_docs[:top_k]
        ])
        return source_df

    # Example usage
    query = "What is APPL?"
    results_df = query_with_reranking(query)
    logging.info("Query Results:")
    logging.info(results_df.to_string())

    # Run QA Chain
    response = qa_chain.invoke({'question': query, 'chat_history': []})
    logging.info("\nQA Chain Response:")
    logging.info(response['answer'])

    save_checkpoint('process_completed')

if __name__ == "__main__":
    main()
