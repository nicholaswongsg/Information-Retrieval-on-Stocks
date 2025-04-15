#!/bin/bash

# Set variables
CSV_FILE="/Users/nicholaswong/Desktop/Github/Information-Retrieval-on-Stocks/Classification/NER_with_ensemble_sentiment.csv"
SOLR_URL="http://localhost:8983/solr"
CORE_NAME="stock_sentiment"

# Check if Solr is running by directly checking the core
echo "Checking if Solr is running..."
if ! curl --silent "${SOLR_URL}/${CORE_NAME}/select?q=*:*&rows=0" | grep -q "responseHeader"; then
    echo "ERROR: Solr doesn't appear to be running at ${SOLR_URL} or core doesn't exist"
    echo "Please start Solr before running this script."
    exit 1
else
    echo "Solr is running and core ${CORE_NAME} is accessible."
fi

# Check if CSV file exists
if [ ! -f "${CSV_FILE}" ]; then
    echo "ERROR: CSV file not found at ${CSV_FILE}"
    exit 1
fi

# Show info about file
echo "Found CSV file: ${CSV_FILE}"
echo "File size: $(du -h "${CSV_FILE}" | cut -f1)"
echo "First few rows:"
head -n 3 "${CSV_FILE}"

# Use curl directly for the import as a more reliable method
echo "Importing CSV data to Solr using curl..."
echo "This might take a while depending on the file size..."

# First, create a temporary cleaned CSV file with proper headers
TMP_CSV="/tmp/clean_data_for_solr.csv"
echo "Creating cleaned CSV file at ${TMP_CSV}..."
head -n 1 "${CSV_FILE}" > "${TMP_CSV}"
tail -n +2 "${CSV_FILE}" >> "${TMP_CSV}"

# Post the file using curl
echo "Posting data to Solr..."
curl -X POST -H "Content-Type: application/csv" --data-binary @"${TMP_CSV}" "${SOLR_URL}/${CORE_NAME}/update?commit=true&csvFieldMapping=id:/id,title:/title,text:/text,subreddit:/subreddit,created_utc:/created_utc,ner_text_cleaned:/ner_text_cleaned,ner_recognized_tickers:/ner_recognized_tickers,ner_entity_sentiments:/ner_entity_sentiments,primary_sentiment:/primary_sentiment,primary_score:/primary_score"

# Verify import
echo "Verifying import..."
DOC_COUNT=$(curl -s "${SOLR_URL}/${CORE_NAME}/select?q=*:*&rows=0" | grep -oP '"numFound":\K\d+')
echo "Documents in Solr: ${DOC_COUNT}"

if [ "$DOC_COUNT" -gt 0 ]; then
    echo "Import successful! Found ${DOC_COUNT} documents in Solr."
else
    echo "Import may have failed. No documents found in Solr."
    echo "Check logs for errors."
    echo "Try the Python script instead: python import_csv_to_solr.py"
fi 