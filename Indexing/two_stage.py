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
from dotenv import load_dotenv
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import time

# Import .env variables
load_dotenv(override=True)

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Define
VECTOR_STORE_PATH = "reddit_stock_faiss_index"
data_path = os.path.join(os.path.dirname(__file__), '..', 'Classification', 'NER_with_ensemble_sentiment_SMALL.csv')

# Define lemmatization function
def lemmatize_text(text):
    if pd.isna(text):
        return ""
    text = str(text)
    tokens = word_tokenize(text)
    lemmatized_tokens = [lemmatizer.lemmatize(token.lower()) for token in tokens]
    return ' '.join(lemmatized_tokens)

def initialize_vector_store(data_path):
    """Process the dataset and create a vector store"""
    # Load and clean the dataset
    print("Loading and cleaning dataset...")
    df = pd.read_csv(data_path)

    # Keep only the columns we need: 'ner_text_cleaned' plus ID fields
    # Adjust these columns if you need additional metadata
    df = df[['ner_text_cleaned','post_id','id']]

    # Apply lemmatization to the ner_text_cleaned field
    print("Lemmatizing text...")
    df['ner_text_cleaned'] = df['ner_text_cleaned'].apply(lemmatize_text)

    # Convert to documents (page_content + metadata)
    # Using ner_text_cleaned as the main text source
    print("Converting to documents...")
    documents = []
    for i, row in df.iterrows():
        # The main text chunk is now ner_text_cleaned
        content = f"{row['ner_text_cleaned']}\n"

        # If there's a post_id, use it as 'source'; otherwise fallback to 'id'
        source = row['post_id'] if not pd.isna(row['post_id']) else row['id']

        # Store the entire document content as a single document
        # But also include the full content in metadata
        doc = Document(
            page_content=content,
            metadata={"source": f"{source}", "full_content": content, "original_id": i}
        )
        documents.append(doc)

    # Split into chunks, but with larger chunk size and overlap to preserve context
    print("Splitting documents into chunks...")
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    chunk_size = 4000  # Much larger chunk size
    chunk_overlap = 500  # More overlap to maintain context
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = []
    
    # Process documents in smaller batches to avoid memory issues
    batch_size_documents = 10000  # Process 10k documents at a time for chunking
    for batch_start in range(0, len(documents), batch_size_documents):
        batch_end = min(batch_start + batch_size_documents, len(documents))
        print(f"Processing documents {batch_start} to {batch_end-1} for chunking...")
        
        batch_docs = documents[batch_start:batch_end]
        for doc in tqdm(batch_docs, desc="Splitting into chunks"):
            # For each chunk, preserve the original document's metadata
            doc_chunks = text_splitter.split_documents([doc])
            for chunk in doc_chunks:
                # Ensure each chunk has a reference to its original document
                chunk.metadata["full_content"] = doc.metadata["full_content"]
                chunk.metadata["original_id"] = doc.metadata["original_id"]
            chunks.extend(doc_chunks)
    
    print(f"Created {len(chunks)} chunks from {len(documents)} documents")

    # Initialize embeddings
    embeddings = AzureOpenAIEmbeddings(
        azure_endpoint=os.environ['AZURE_OPENAI_ENDPOINT'],
        api_key=os.environ['AZURE_OPENAI_API_KEY'],
        azure_deployment="text-embedding-ada-002",
        model='text-embedding-ada-002',
    )

    # Process in smaller batches with delays to avoid rate limits
    print("Creating vector database in batches with rate limiting...")
    batch_size = 100  # Much smaller batch size to avoid rate limits
    vector_store = None

    for i in tqdm(range(0, len(chunks), batch_size)):
        batch = chunks[i:i+batch_size]
        
        # Add retry logic for rate limit errors
        max_retries = 5
        retry_delay = 5  # Start with 5 seconds delay
        
        for retry in range(max_retries):
            try:
                print(f"Processing batch {i//batch_size + 1}/{len(chunks)//batch_size + 1}")
                
                if vector_store is None:
                    vector_store = FAISS.from_documents(batch, embeddings)
                else:
                    batch_vector_store = FAISS.from_documents(batch, embeddings)
                    vector_store.merge_from(batch_vector_store)
                
                # Save progress every 10 batches
                if (i // batch_size) % 10 == 0 and vector_store is not None:
                    print(f"Saving progress at batch {i//batch_size + 1}...")
                    temp_path = f"{VECTOR_STORE_PATH}_partial"
                    vector_store.save_local(temp_path)
                
                # Add a delay between batches to avoid rate limits
                time.sleep(3)  # 3 second delay between batches
                break  # Success, exit retry loop
                
            except Exception as e:
                if "429" in str(e) and retry < max_retries - 1:
                    # Rate limit error, retry with exponential backoff
                    wait_time = retry_delay * (2 ** retry)
                    print(f"Rate limit exceeded. Waiting {wait_time} seconds before retrying...")
                    time.sleep(wait_time)
                else:
                    # Other error or max retries exceeded
                    raise

    print(f"Vector database created with {vector_store.index.ntotal} chunks")
    vector_store.save_local(VECTOR_STORE_PATH)
    return vector_store, embeddings

def load_vector_store(data_path):
    """Load an existing vector store"""
    print("DEBUG CHECKING IF CORRECT: ",os.environ['AZURE_OPENAI_API_KEY'], os.environ['AZURE_OPENAI_ENDPOINT'])
    embeddings = AzureOpenAIEmbeddings(
        azure_endpoint=os.environ['AZURE_OPENAI_ENDPOINT'],
        api_key=os.environ['AZURE_OPENAI_API_KEY'],
        azure_deployment="text-embedding-ada-002",
        model='text-embedding-ada-002'
    )
    
    # Check for a partial index first
    partial_path = f"{VECTOR_STORE_PATH}_partial"
    if os.path.exists(VECTOR_STORE_PATH):
        print("Loading existing vector store...")
        return FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True), embeddings
    elif os.path.exists(partial_path):
        print("Loading partial vector store... (This is not complete but can be used for testing)")
        return FAISS.load_local(partial_path, embeddings, allow_dangerous_deserialization=True), embeddings
    else:
        print("No existing vector store found. Creating new one...")
        return initialize_vector_store(data_path)

def process_query(query, vector_store):
    """Clean query and perform similarity search"""
    # Clean and lemmatize the query
    cleaned_query = lemmatize_text(query)
    
    # Use k instead of similarity threshold to ensure we get sufficient results
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={'k': 100}  # Get top 100 results regardless of score
    )
    search_results = retriever.invoke(cleaned_query)
    
    return search_results

def rerank_results(query, search_results, top_n=100):
    """Rerank results"""
    print(f"Reranking {len(search_results)} results...")
    
    try:
        # Load cross-encoder reranker model
        model_name = "BAAI/bge-reranker-v2-m3"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
        # Prepare pairs of (query, document) for reranking
        pairs = []
        for doc in search_results:
            pairs.append((query, doc.page_content))
        
        # Get reranker scores
        with torch.no_grad():
            inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
            scores = model(**inputs, return_dict=True).logits.view(-1, ).float()
        
        # Add new reranker scores to document metadata
        for i, doc in enumerate(search_results):
            doc.metadata['rerank_score'] = float(scores[i])
        
        # Sort by reranker score (higher is better)
        reranked_results = sorted(search_results, key=lambda x: x.metadata['rerank_score'], reverse=True)
        
        # Apply cutoff to only keep top_n results
        reranked_results = reranked_results[:top_n]
        
        # Print reranking information for debugging
        for i, doc in enumerate(reranked_results):
            initial_score = doc.metadata.get('score', 0.0)
            print(f"Rank {i+1}: Rerank score {doc.metadata['rerank_score']:.4f}")
        return reranked_results
        
    except Exception as e:
        print(f"Reranking error: {e}. Falling back to initial ranking.")
        return search_results[:top_n]

def grade_results(query, search_results):
    """Filter out irrelevant chunks using GPT-4o-mini"""
    # Initialize the grader LLM
    grader_llm = AzureChatOpenAI(
        azure_endpoint=os.environ['AZURE_OPENAI_ENDPOINT'],
        api_key=os.environ['AZURE_OPENAI_API_KEY'],
        azure_deployment="gpt-4o-mini",
        model="gpt-4o-mini",
        api_version="2024-02-01",
        temperature=0.7
    )
  
    # Prepare content for grading
    chunks_with_index = []
    for i, result in enumerate(search_results):
        chunks_with_index.append(f"[{i}] {result.page_content}")
    
    chunks_text = "\n\n---\n\n".join(chunks_with_index)
    
    # Create grading prompt
    system_message = """You are evaluating search results for relevance to a user query about stocks, companies, or products.
                        Your task is to identify chunks that contain ANY information remotely relevant to the query.
                        Be EXTREMELY inclusive. Even if a chunk has only a passing mention or minimal information,
                        include it in your selection.
                        Return only the indices of relevant chunks as comma-separated numbers."""

    human_message = f"""USER QUERY: {query}

                        SEARCH RESULTS:
                        {chunks_text}

                        For each search result, determine if it contains ANY information even remotely related to the query.
                        Be MAXIMALLY inclusive - include ALL chunks that might have ANY relevance, 
                        even if they only tangentially mention the topic.
                        
                        For products like iPhone, include ALL mentions regardless of how brief.
                        For companies, include ALL mentions of their products, services, or market position.
                        For stocks, include even minor references to the company or ticker.
                        
                        Return the indices of ALL potentially relevant chunks as comma-separated numbers.
                        Example response: 0, 2, 5, 8, 10, 12, 15
                        
                        IMPORTANT: Your goal is to INCLUDE as many potentially relevant chunks as possible,
                        not to filter them. When in doubt, INCLUDE the chunk."""

    messages = [
        SystemMessage(content=system_message),
        HumanMessage(content=human_message)
    ]
    
    # Get graded results
    response = grader_llm.invoke(messages)
    
    try:
        # Parse comma-separated numbers
        indices_text = response.content.strip()
        relevant_indices = [int(idx.strip()) for idx in indices_text.split(',') if idx.strip().isdigit()]
        relevant_chunks = [search_results[i].page_content for i in relevant_indices if i < len(search_results)]

        return relevant_chunks
    except:
        # Fallback if parsing fails
        print("Warning: Failed to parse grader response. Using all chunks.")
        return [result.page_content for result in search_results]

def generate_answer(query, relevant_chunks):
    """Generate final answer using GPT-4o"""
    answer_llm = AzureChatOpenAI(
        azure_endpoint=os.environ['AZURE_OPENAI_ENDPOINT'],
        api_key=os.environ['AZURE_OPENAI_API_KEY'],
        azure_deployment="gpt-4o",
        model="gpt-4o",
        api_version="2024-02-01",
        temperature=0.7
    )
    
    # Create system message
    system_message = """You will act as a helpful assistant that provides insights about stocks based on Reddit opinions. Your task is to analyze and summarize the provided context, which consists of Reddit discussions about stocks, and respond to the user's query.
                        Your response should be:
                        - Informative: Extract key points from the Reddit discussions and present them clearly.
                        - Balanced: Highlight different perspectives if they exist, including both bullish and bearish sentiments.
                        - Transparent: Clearly state that these insights are opinions from Reddit users and do not constitute financial advice.
                        - Context-Aware: If the provided context lacks relevant information to answer the query, acknowledge this limitation rather than speculating.

                        Response Structure:
                        Sentiment Analysis: Assign a sentiment label to each perspective, categorizing it as very negative, slightly negative, neutral, slightly positive, or very positive based on the tone and content of the discussions. Ensure that the sentiment is derived only from the provided context and does not include external sources.
                        Overview: A brief summary of the Reddit discussion, outlining the main topics or recurring themes.
                        Bullish Perspectives (Positive Sentiment):
                        - Summarize the arguments supporting a positive outlook on the stock.
                        Bearish Perspectives (Negative Sentiment):
                        - Summarize the arguments supporting a negative outlook on the stock.
                        Neutral or Mixed Opinions (if applicable):
                        - Highlight any discussions that present a neutral stance or acknowledge both bullish and bearish points.
                        Conclusion & Disclaimer:
                        - Provide a concise closing statement summarizing the overall sentiment.
                        - Clearly state that these insights are from Reddit discussions and do not constitute financial advice.
                        
                        The assistant should treat all opinions equally, rather than prioritizing the most upvoted or high-engagement comments. Specific Reddit users may be referenced if mentioned in the context, but anonymity is also acceptable."""

    # Format context
    formatted_chunks = "\n\n---\n\n".join(relevant_chunks) if relevant_chunks else "No relevant information found."
    
    # Create messages
    messages = [
        SystemMessage(content=system_message),
        HumanMessage(content=f"""CONTEXT:
                                {formatted_chunks}

                                USER QUERY:
                                {query}
                    """)
    ]
    
    # Generate answer
    response = answer_llm.invoke(messages)
    return response.content

def process_query_without_index(query, data_path):
    """Process a query without using the vector store - direct keyword matching"""
    print("Using direct keyword matching (fallback method without vector store)")
    
    # Load the dataset
    df = pd.read_csv(data_path)
    
    # Simple keyword matching
    keywords = query.lower().split()
    matches = []
    
    for _, row in df.iterrows():
        text = str(row['ner_text_cleaned']).lower()
        if any(keyword in text for keyword in keywords):
            matches.append({
                'text': row['ner_text_cleaned'],
                'score': sum(1 for keyword in keywords if keyword in text),
                'id': row.get('id', 'unknown')
            })
    
    # Sort by relevance score (number of keywords matched)
    matches.sort(key=lambda x: x['score'], reverse=True)
    
    # Return top matches
    top_matches = [match['text'] for match in matches[:20]]
    return top_matches

def answer_stock_question(data_path, query):
    """Answer a stock question using either vector search or fallback method"""
    try:
        # Try to load vector store
        vector_store, embeddings = load_vector_store(data_path)
        
        # Process query and get search results
        search_results = process_query(query, vector_store)
        
        if not search_results:
            print("No vector search results, trying fallback method...")
            relevant_chunks = process_query_without_index(query, data_path)
            if not relevant_chunks:
                return "I couldn't find any relevant information about that stock query in the Reddit data.", []
            
            # Generate answer using fallback results
            answer = generate_answer(query, relevant_chunks)
            return answer, relevant_chunks
        
        # Rerank results using cross-encoder model
        reranked_results = rerank_results(query, search_results)
        
        if not reranked_results:
            return "Although I found some potentially related information, none of it seems directly relevant to your query about stocks.", []
        
        # ALWAYS include all reranked results as part of relevant_chunks for display
        # Extract full content from metadata if available, otherwise use page_content
        all_chunks = []
        for doc in reranked_results[:20]:
            # Try to get full content if available
            if 'full_content' in doc.metadata:
                all_chunks.append(doc.metadata['full_content'])
            else:
                all_chunks.append(doc.page_content)
        
        # Grade results for answer generation (for more focused answer)
        graded_chunks = grade_results(query, reranked_results)
        
        # If we have very few graded chunks, use all reranked results
        if len(graded_chunks) < 5:
            print("Warning: Too few relevant chunks found. Using all reranked results.")
            graded_chunks = all_chunks
        
        if not graded_chunks:
            return "Although I found some potentially related information, none of it seems directly relevant to your query about stocks.", all_chunks
        
        # Generate final answer
        answer = generate_answer(query, graded_chunks)
        return answer, all_chunks  # Return ALL chunks for display
    
    except Exception as e:
        print(f"Error with vector store: {str(e)}. Using fallback method.")
        # Fallback to direct text search if vector store fails
        relevant_chunks = process_query_without_index(query, data_path)
        if not relevant_chunks:
            return f"An error occurred ({str(e)}) and I couldn't find information about your query.", []
        
        # Generate answer using fallback results
        answer = generate_answer(query, relevant_chunks)
        return answer, relevant_chunks

def main(query):
    print("\nSearching for relevant information...")
    answer, relevant_chunks = answer_stock_question(data_path, query)
    return answer, relevant_chunks

if __name__ == "__main__":
    # Example usage:
    # python script.py
    example_query = "What is the sentiment on TSLA this week?"
    final_answer, chunks = main(example_query)
    print("ANSWER:\n", final_answer)
