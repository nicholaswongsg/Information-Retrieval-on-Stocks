#!/bin/bash

# Make scripts executable
chmod +x setup_solr_schema.sh
chmod +x verify_csv.py

echo "Verifying CSV file structure..."
# Activate virtual environment if it exists
if [ -d "../venv/bin" ]; then
  source ../venv/bin/activate
  python3 verify_csv.py
else
  # Otherwise try python3 command directly
  python3 verify_csv.py
fi

if [ $? -ne 0 ]; then
  echo "CSV verification failed. Please check the output for more details."
  echo "The application may not work correctly if the 'ner_text_cleaned' field is missing."
  read -p "Continue anyway? (y/n): " confirm
  if [[ $confirm != "y" && $confirm != "Y" ]]; then
    echo "Aborted."
    exit 1
  fi
fi

echo "Starting Streamlit application with Solr integration..."
echo "This may take a few minutes for initial setup."

# Build and start the containers
docker-compose up --build

# Note that this may take some time on first run as it needs to:
# 1. Build the Docker image
# 2. Download Solr and set it up
# 3. Create the Solr core
# 4. Import the CSV data
# 5. Start the Streamlit application

echo "After startup completes, you can access:"
echo "- Streamlit app: http://localhost:8501"
echo "- Solr admin: http://localhost:8983/solr/" 