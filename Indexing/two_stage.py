import os
import tqdm
import pandas as pd
import logging
import json
from datetime import datetime

import os
import tqdm
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank
from langchain_huggingface import HuggingFaceEmbeddings

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('vectorstore_creation.log'),
        logging.StreamHandler()
    ]
)

# Checkpoint file to track progress
CHECKPOINT_FILE = 'vectorstore_checkpoint.json'

def save_checkpoint(stage, details=None):
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

def main():
    # Check for existing checkpoint
    checkpoint = load_checkpoint()
    if checkpoint:
        logging.info(f"Loaded checkpoint: {checkpoint['stage']} at {checkpoint['timestamp']}")

    # Specify the full path to your CSV file
    data_path = os.path.join(os.path.dirname(__file__), '..', 'PreProcessing', 'combined_reddit_stock_data.csv')

    # Read the CSV file
    try:
        data = pd.read_csv(data_path)
        logging.info(f"Data loaded successfully. Shape: {data.shape}")
        logging.info(f"Columns: {list(data.columns)}")
        save_checkpoint('data_loaded', {'rows': len(data), 'columns': list(data.columns)})
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise

    # Initialize embeddings
    try:
        embedding = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
        save_checkpoint('embeddings_initialized')
    except Exception as e:
        logging.error(f"Failed to initialize embeddings: {e}")
        raise

    # Initialize an empty list to store documents
    docs = []

    # Text Splitting and Document Creation
    try:
        for i in tqdm.tqdm(range(len(data))):
            # Skip rows with empty or None text
            if not data['text'][i] or pd.isna(data['text'][i]):
                continue
            
            # Use RecursiveCharacterTextSplitter to break text into chunks
            splitter = RecursiveCharacterTextSplitter(chunk_size=2048)
            
            # Convert all relevant fields to string to ensure compatibility
            metadata = {
                col: str(data[col][i]) if not pd.isna(data[col][i]) else '' 
                for col in data.columns if col != 'text'
            }
            
            # Split text and create Document objects
            for chunk in splitter.split_text(str(data['text'][i])):
                docs.append(Document(
                    page_content=chunk, 
                    metadata=metadata
                ))
        
        save_checkpoint('documents_created', {'total_documents': len(docs)})
    except Exception as e:
        logging.error(f"Failed during document creation: {e}")
        raise

    # Create persist directory if it doesn't exist
    persist_directory = os.path.join(os.path.dirname(__file__), 'vectorstore')
    os.makedirs(persist_directory, exist_ok=True)

    # Create Chroma Vector Store
    try:
        vector_collection = Chroma.from_documents(
            documents=docs, 
            persist_directory=persist_directory, 
            embedding=embedding
        )
        save_checkpoint('vector_store_created', {'persist_directory': persist_directory})
    except Exception as e:
        logging.error(f"Failed to create vector store: {e}")
        raise

    # Load Vector Database
    try:
        vectordb = Chroma(
            persist_directory=persist_directory, 
            embedding_function=embedding
        )
        save_checkpoint('vector_db_loaded')
    except Exception as e:
        logging.error(f"Failed to load vector database: {e}")
        raise

    # Setup Conversation Memory
    memory = ConversationBufferMemory(
        return_messages=True, 
        memory_key='chat_history'
    )

    # Configure Retriever (top 3 similar documents)
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})

    # For local execution, use environment variable for API key
    cohere_api_key = os.getenv('COHERE_API_KEY')
    if not cohere_api_key:
        logging.error("Cohere API key not set")
        raise ValueError("Please set the COHERE_API_KEY environment variable")

    # Initialize Cohere Language Model and Reranker
    try:
        from langchain_community.llms import Cohere
        llm = Cohere(model='command', cohere_api_key=cohere_api_key)

        # Create Cohere Reranker
        compressor = CohereRerank(cohere_api_key=cohere_api_key)
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
    def query_with_reranking(query, top_k=3):
        # Perform query with compression retriever
        compressed_docs = compression_retriever.get_relevant_documents(query)
        
        # Create DataFrame with results
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

    # Optionally, you can also use the QA chain
    response = qa_chain({'question': query, 'chat_history': []})
    logging.info("\nQA Chain Response:")
    logging.info(response['answer'])

    # Final checkpoint
    save_checkpoint('process_completed')

if __name__ == "__main__":
    main()