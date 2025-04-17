#!/usr/bin/env python3

import pandas as pd
import os
import sys

def verify_csv(file_path):
    """
    Verify that the CSV file contains the expected fields,
    particularly checking for ner_text_cleaned
    """
    try:
        print(f"Reading CSV file: {file_path}")
        
        if not os.path.exists(file_path):
            print(f"\n❌ ERROR: CSV file not found at: {file_path}")
            print("Please make sure the file exists and the path is correct.")
            return False
            
        # Try to read file size first to check if it's accessible
        file_size = os.path.getsize(file_path)
        print(f"CSV file size: {file_size / (1024*1024):.2f} MB")
        
        # Read just the header to check columns
        try:
            df_header = pd.read_csv(file_path, nrows=0)
        except Exception as e:
            print(f"\n❌ ERROR reading CSV header: {str(e)}")
            print("This might be due to file format issues or permissions.")
            return False
            
        print("\nColumns in the CSV file:")
        for col in df_header.columns:
            print(f"  - {col}")
        
        # Check if ner_text_cleaned exists
        if 'ner_text_cleaned' in df_header.columns:
            print("\n✅ 'ner_text_cleaned' field found in the CSV!")
        else:
            print("\n⚠️ WARNING: 'ner_text_cleaned' field NOT found in the CSV!")
            print("The Solr search functionality will not work properly without this field.")
            print("Available text fields that could be used instead:")
            text_fields = [col for col in df_header.columns if 'text' in col.lower()]
            for field in text_fields:
                print(f"  - {field}")
        
        # Check if ner_recognized_tickers exists
        if 'ner_recognized_tickers' in df_header.columns:
            print("\n✅ 'ner_recognized_tickers' field found in the CSV!")
        else:
            print("\n⚠️ WARNING: 'ner_recognized_tickers' field NOT found in the CSV!")
            print("Stock ticker search may not work properly without this field.")
        
        # Sample data
        try:
            print("\nReading sample data (first 2 rows)...")
            df_sample = pd.read_csv(file_path, nrows=2)
            
            for i, row in df_sample.iterrows():
                print(f"\nSample row {i+1}:")
                for col in ['title', 'text', 'ner_text_cleaned', 'ner_recognized_tickers', 'primary_sentiment']:
                    if col in row:
                        value = row.get(col, "N/A")
                        # Truncate long text fields for display
                        if isinstance(value, str) and len(value) > 100:
                            value = value[:100] + "..."
                        print(f"  {col}: {value}")
            
            print("\nCSV verification complete.")
            return True
            
        except Exception as e:
            print(f"\n⚠️ WARNING: Could not read sample data: {str(e)}")
            print("The CSV file structure looks valid but we couldn't read sample rows.")
            # Still return True as the headers were verified
            return True
        
    except Exception as e:
        print(f"\n❌ ERROR verifying CSV: {str(e)}")
        return False

if __name__ == "__main__":
    try:
        # Path to the CSV file
        csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Classification", "NER_with_sentiment.csv"))
        
        if verify_csv(csv_path):
            sys.exit(0)  # Success
        else:
            sys.exit(1)  # Failure
    except Exception as e:
        print(f"Unhandled exception: {str(e)}")
        sys.exit(1) 