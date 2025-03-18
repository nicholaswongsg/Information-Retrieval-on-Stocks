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
from dotenv import load_dotenv

# Import .env variables
load_dotenv(override=True)

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

def initialize_vector_store():
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

    # Initialize embeddings
    embeddings = AzureOpenAIEmbeddings(
        azure_endpoint=os.environ['AZURE_OPENAI_ENDPOINT'],
        api_key=os.environ['AZURE_OPENAI_APIKEY'],
        azure_deployment="text-embedding-ada-002",
        model='text-embedding-ada-002'
    )

    # Process in batches to avoid memory issues
    print("Creating vector database in batches...")
    batch_size = 100
    vector_store = None

    for i in tqdm(range(0, len(chunks), batch_size)):
        batch = chunks[i:i+batch_size]
        
        if vector_store is None:
            vector_store = FAISS.from_documents(batch, embeddings)
        else:
            batch_vector_store = FAISS.from_documents(batch, embeddings)
            vector_store.merge_from(batch_vector_store)

    print(f"Vector database created with {vector_store.index.ntotal} chunks")
    vector_store.save_local("reddit_stock_faiss_index")
    return vector_store, embeddings

def load_vector_store():
    """Load an existing vector store"""
    embeddings = AzureOpenAIEmbeddings(
        azure_endpoint=os.environ['AZURE_OPENAI_ENDPOINT'],
        api_key=os.environ['AZURE_OPENAI_APIKEY'],
        azure_deployment="text-embedding-ada-002",
        model='text-embedding-ada-002'
    )
    
    if os.path.exists("reddit_stock_faiss_index"):
        print("Loading existing vector store...")
        return FAISS.load_local("reddit_stock_faiss_index", embeddings), embeddings
    else:
        print("No existing vector store found. Creating new one...")
        return initialize_vector_store()

def process_query(query, vector_store):
    """Clean query and perform similarity search"""
    # Clean and lemmatize the query
    cleaned_query = lemmatize_text(query)
    
    # Perform similarity search with scores
    docs_and_scores = vector_store.similarity_search_with_score(
        query = cleaned_query,
        score_threshold=0.5  # Minimum similarity score
    )
    
    # Add scores to metadata and create result list
    search_results = []
    for doc, score in docs_and_scores:
        # Add the score to the document's metadata
        doc.metadata['score'] = score
        search_results.append(doc)
    
    # Sort results by score (highest first)
    search_results.sort(key=lambda x: x.metadata['score'], reverse=True)
    
    return search_results

def grade_results(query, search_results):
    """Filter out irrelevant chunks using GPT-4o-mini"""
    # Initialize the grader LLM
    grader_llm = AzureChatOpenAI(
        azure_endpoint=os.environ['AZURE_OPENAI_ENDPOINT'],
        api_key=os.environ['AZURE_OPENAI_APIKEY'],
        azure_deployment="gpt-4o-mini",
        temperature=0.7
    )
    
    # Prepare content for grading
    chunks_with_index = []
    for i, result in enumerate(search_results):
        chunks_with_index.append(f"[{i}] {result.page_content}")
    
    chunks_text = "\n\n---\n\n".join(chunks_with_index)
    
    # Create grading prompt
    system_message = """You are evaluating search results for relevance to a user query about stocks.
                        Your task is to identify which chunks are truly relevant to answering the query.
                        Return only the indices of relevant chunks as a JSON array of numbers."""

    human_message = f"""USER QUERY: {query}

                        SEARCH RESULTS:
                        {chunks_text}

                        For each search result, determine if it is truly relevant to the query.
                        Return only a JSON array containing the indices of relevant chunks.
                        Example response: [0, 2, 5]
                        If a chunk has a high similarity score but is not relevant, exclude it."""

    messages = [
        SystemMessage(content=system_message),
        HumanMessage(content=human_message)
    ]
    
    # Get graded results
    response = grader_llm.invoke(messages)
    
    try:
        # Parse JSON response to get indices of relevant chunks
        relevant_indices = json.loads(response.content)
        relevant_chunks = [search_results[i].page_content for i in relevant_indices if i < len(search_results)]
        return relevant_chunks
    except:
        # Fallback if JSON parsing fails
        print("Warning: Failed to parse grader response as JSON. Using all chunks.")
        return [result.page_content for result in search_results]

def generate_answer(query, relevant_chunks):
    """Generate final answer using GPT-4o"""
    answer_llm = AzureChatOpenAI(
        azure_endpoint=os.environ['AZURE_OPENAI_ENDPOINT'],
        api_key=os.environ['AZURE_OPENAI_APIKEY'],
        azure_deployment="gpt-4o",
        temperature=0.7
    )
    
    # Create system message
    system_message = """You are a helpful assistant that provides insights about stocks based on Reddit opinions.
                        Your task is to answer the user's query about stocks using the provided context from Reddit discussions.
                        Be informative, balanced, and highlight different perspectives when they exist.
                        Make it clear that these are opinions from Reddit users, not financial advice.
                        If the context doesn't contain relevant information to answer the query, acknowledge this limitation."""
    
    # Format context
    formatted_chunks = "\n\n---\n\n".join(relevant_chunks) if relevant_chunks else "No relevant information found."
    
    # Create messages
    messages = [
        SystemMessage(content=system_message),
        HumanMessage(content=f"""CONTEXT:
                                {formatted_chunks}

                                USER QUERY:
                                {query}

                                Please provide a comprehensive answer based on the Reddit opinions in the context.""")
    ]
    
    # Generate answer
    response = answer_llm.invoke(messages)
    
    return response.content

def answer_stock_question(query):
    # Load vector store
    vector_store, embeddings = load_vector_store()
    
    # Steps 2-4: Process query and get search results
    search_results = process_query(query, vector_store)
    
    if not search_results:
        return "I couldn't find any relevant information about that stock query in the Reddit data."
    
    # Step 5: Grade results
    relevant_chunks = grade_results(query, search_results)
    
    if not relevant_chunks:
        return "Although I found some potentially related information, none of it seems directly relevant to your query about stocks."
    
    # Step 6: Generate final answer
    answer = generate_answer(query, relevant_chunks)
    
    return answer

def main():
    # Check if vector store exists, if not create it
    if not os.path.exists("reddit_stock_faiss_index"):
        initialize_vector_store()
    
    print("Type 'exit' to quit.")
    
    while True:
        query = input("\nAsk a question about stocks (based on Reddit opinions): ")
        
        if query.lower() in ['exit']:
            break
            
        print("\nSearching for relevant information...")
        answer = answer_stock_question(query)
        print("\nAnswer:")
        print(answer)

if __name__ == "__main__":
    main()
