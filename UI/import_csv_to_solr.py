# pip install pysolr pandas tqdm requests

import pysolr
import pandas as pd
import json
import time
import os
import sys
from tqdm import tqdm
import requests

def main():
    # Get absolute directory path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
    # Configuration with absolute path
    csv_file = os.path.join(project_root, "Classification", "NER_with_ensemble_sentiment.csv")
    solr_url = "http://localhost:8983/solr/stock_sentiment"
    batch_size = 50  # Smaller batches for better reliability
    
    print(f"Script running from: {current_dir}")
    print(f"Project root: {project_root}")
    
    # Check if the CSV file exists
    if not os.path.exists(csv_file):
        print(f"ERROR: CSV file not found at {csv_file}")
        return
    
    print(f"Found CSV file: {csv_file}")
    print(f"File size: {os.path.getsize(csv_file) / (1024*1024):.2f} MB")
    
    # Connect to Solr
    print(f"Connecting to Solr at {solr_url}...")
    try:
        # First check if Solr is responding
        response = requests.get(f"{solr_url}/select?q=*:*&rows=0")
        if response.status_code != 200:
            print(f"ERROR: Could not connect to Solr. Status code: {response.status_code}")
            print(f"Response: {response.text}")
            return
        print("Solr connection successful.")
            
        # Connect with pysolr
        solr = pysolr.Solr(solr_url, always_commit=True, timeout=60)
    except Exception as e:
        print(f"ERROR: Could not connect to Solr: {e}")
        print("Please ensure Solr is running and the core 'stock_sentiment' exists.")
        return
    
    # Load CSV data
    print(f"Loading data from {csv_file}...")
    try:
        # Read with explicit encoding and handling potential errors
        df = pd.read_csv(csv_file, encoding='utf-8', on_bad_lines='warn')
        print(f"Loaded {len(df)} records from CSV.")
        print(f"Columns found: {', '.join(df.columns)}")
    except Exception as e:
        print(f"ERROR: Could not read CSV file: {e}")
        return
    
    # Process JSON columns - handle potential errors
    json_columns = ['ner_entity_sentiments', 'qwen_sentiments', 
                    'human1_sentiment', 'human2_sentiment', 'ensemble_results']
    
    for col in json_columns:
        if col in df.columns:
            print(f"Converting column {col} from string to JSON object...")
            try:
                df[col] = df[col].apply(lambda x: 
                    json.loads(x) if isinstance(x, str) and x.strip() and x.strip()[0] in ('{', '[') 
                    else {})
            except Exception as e:
                print(f"WARNING: Error processing JSON column {col}: {e}")
                print("Continuing with original string format.")
    
    # Delete existing documents 
    print("Deleting any existing documents in Solr...")
    try:
        solr.delete(q='*:*')
        solr.commit()
        print("All existing documents deleted.")
    except Exception as e:
        print(f"WARNING: Could not delete existing documents: {e}")
    
    # Process in batches with error handling for each batch
    total_records = len(df)
    total_batches = (total_records + batch_size - 1) // batch_size
    successful_records = 0
    failed_batches = 0
    
    print(f"Processing {total_records} records in {total_batches} batches...")
    
    for i in tqdm(range(0, total_records, batch_size)):
        batch_df = df.iloc[i:i+batch_size].copy()
        end_idx = min(i+batch_size, total_records)
        
        print(f"\nProcessing batch {i//batch_size + 1}/{total_batches} (records {i+1}-{end_idx})...")
        
        # Fill missing values and convert to dict
        for col in batch_df.columns:
            if batch_df[col].dtype == 'object':
                batch_df[col] = batch_df[col].fillna('')
            else:
                batch_df[col] = batch_df[col].fillna(0)
        
        try:
            # Convert to list of dictionaries
            batch_docs = batch_df.to_dict('records')
            
            # Convert any non-string values to strings for safer handling
            cleaned_docs = []
            for doc in batch_docs:
                cleaned_doc = {}
                for key, value in doc.items():
                    if isinstance(value, (dict, list)):
                        cleaned_doc[key] = json.dumps(value)
                    else:
                        cleaned_doc[key] = value
                cleaned_docs.append(cleaned_doc)
                
            # Add to Solr
            solr.add(cleaned_docs)
            successful_records += len(cleaned_docs)
            print(f"Successfully added {len(cleaned_docs)} records.")
            
        except Exception as e:
            print(f"ERROR in batch {i//batch_size + 1}: {e}")
            failed_batches += 1
            # Continue with next batch
            continue
    
    # Final commit
    print("Final commit to Solr...")
    solr.commit()
    
    # Verify import
    print("\nVerifying import...")
    try:
        results = solr.search('*:*', rows=0)
        doc_count = results.hits
        print(f"Documents in Solr: {doc_count}")
        
        if doc_count > 0:
            print(f"Import summary:")
            print(f"- Total records processed: {total_records}")
            print(f"- Successfully imported: {successful_records}")
            print(f"- Failed batches: {failed_batches}")
            print(f"- Verification count in Solr: {doc_count}")
        else:
            print("Import may have failed. No documents found in Solr.")
    except Exception as e:
        print(f"ERROR during verification: {e}")

if __name__ == "__main__":
    main() 