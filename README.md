## Setup Instructions

### 1. Create a Virtual Environment
Run the following command to create and activate a virtual environment:

```sh
# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies
Install the required Python packages using:
```sh
pip install -r requirements.txt
```

### 3. Configure .env file
- Follow format in .env.example file

### 4. Start Solr using Docker Compose
1. Start docker
2. cd UI
3. run  
```sh 
./run.sh
```
4. Run the following command to start Solr in a Docker container:
```sh
docker-compose up -d
```

### 4. Ensure Dataset Files are Present
Make sure the following dataset files are inside the project directory before proceeding:
```sh
ls -l Classification/NER_with_ensemble_sentiment.csv
ls -l Classification/NER_with_ensemble_sentiment_SMALL.csv
```

### 5. Convert Data and Load into Solr
Run the conversion script to process the data and load it into Solr:
```sh 
python import_csv_to_solr.py
```

### 6. Build index
Run 
```sh
python Indexing/inverted_index_edited.py
```

### 7. Run Application
```sh
streamlit run UI/streamlit_app_integrated.py
```
