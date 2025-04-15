#!/bin/bash

# Wait for Solr to start
echo "Waiting for Solr to start..."
sleep 5

# Define Solr URL
SOLR_URL="http://localhost:8983/solr"
CORE_NAME="stock_sentiment"

# Define field types and field setup
echo "Setting up schema for ${CORE_NAME}..."

# Add text field type with analyzers for better text search
curl -X POST -H 'Content-type:application/json' --data-binary '{
  "add-field-type": {
    "name": "text_general",
    "class": "solr.TextField",
    "positionIncrementGap": "100",
    "analyzer": {
      "tokenizer": {
        "class": "solr.StandardTokenizerFactory"
      },
      "filters": [
        {"class": "solr.StopFilterFactory", "words": "stopwords.txt", "ignoreCase": "true"},
        {"class": "solr.LowerCaseFilterFactory"}
      ]
    }
  }
}' ${SOLR_URL}/${CORE_NAME}/schema

# Add date field type
curl -X POST -H 'Content-type:application/json' --data-binary '{
  "add-field-type": {
    "name": "pdate",
    "class": "solr.DatePointField",
    "docValues": true
  }
}' ${SOLR_URL}/${CORE_NAME}/schema

# Add fields for the CSV data
curl -X POST -H 'Content-type:application/json' --data-binary '{
  "add-field": {
    "name": "title",
    "type": "text_general",
    "indexed": true,
    "stored": true
  }
}' ${SOLR_URL}/${CORE_NAME}/schema

curl -X POST -H 'Content-type:application/json' --data-binary '{
  "add-field": {
    "name": "text",
    "type": "text_general",
    "indexed": true,
    "stored": true
  }
}' ${SOLR_URL}/${CORE_NAME}/schema

curl -X POST -H 'Content-type:application/json' --data-binary '{
  "add-field": {
    "name": "ner_text_cleaned",
    "type": "text_general",
    "indexed": true,
    "stored": true
  }
}' ${SOLR_URL}/${CORE_NAME}/schema

curl -X POST -H 'Content-type:application/json' --data-binary '{
  "add-field": {
    "name": "ner_recognized_tickers",
    "type": "string",
    "indexed": true,
    "stored": true
  }
}' ${SOLR_URL}/${CORE_NAME}/schema

curl -X POST -H 'Content-type:application/json' --data-binary '{
  "add-field": {
    "name": "primary_sentiment",
    "type": "string",
    "indexed": true,
    "stored": true
  }
}' ${SOLR_URL}/${CORE_NAME}/schema

curl -X POST -H 'Content-type:application/json' --data-binary '{
  "add-field": {
    "name": "primary_score",
    "type": "pfloat",
    "indexed": true,
    "stored": true
  }
}' ${SOLR_URL}/${CORE_NAME}/schema

curl -X POST -H 'Content-type:application/json' --data-binary '{
  "add-field": {
    "name": "ner_entity_sentiments",
    "type": "text_general",
    "indexed": true,
    "stored": true
  }
}' ${SOLR_URL}/${CORE_NAME}/schema

curl -X POST -H 'Content-type:application/json' --data-binary '{
  "add-field": {
    "name": "created_utc",
    "type": "plong",
    "indexed": true,
    "stored": true
  }
}' ${SOLR_URL}/${CORE_NAME}/schema

curl -X POST -H 'Content-type:application/json' --data-binary '{
  "add-field": {
    "name": "subreddit",
    "type": "string",
    "indexed": true,
    "stored": true
  }
}' ${SOLR_URL}/${CORE_NAME}/schema

# Add new fields for ensemble sentiment data
curl -X POST -H 'Content-type:application/json' --data-binary '{
  "add-field": {
    "name": "qwen_sentiments",
    "type": "text_general",
    "indexed": true,
    "stored": true
  }
}' ${SOLR_URL}/${CORE_NAME}/schema

curl -X POST -H 'Content-type:application/json' --data-binary '{
  "add-field": {
    "name": "human1_sentiment",
    "type": "text_general",
    "indexed": true,
    "stored": true
  }
}' ${SOLR_URL}/${CORE_NAME}/schema

curl -X POST -H 'Content-type:application/json' --data-binary '{
  "add-field": {
    "name": "human2_sentiment",
    "type": "text_general",
    "indexed": true,
    "stored": true
  }
}' ${SOLR_URL}/${CORE_NAME}/schema

curl -X POST -H 'Content-type:application/json' --data-binary '{
  "add-field": {
    "name": "ensemble_results",
    "type": "text_general",
    "indexed": true,
    "stored": true
  }
}' ${SOLR_URL}/${CORE_NAME}/schema

echo "Schema setup complete. Ready to import data." 