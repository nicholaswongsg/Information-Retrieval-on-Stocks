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

### 3. Start Solr using Docker Compose
Run the following command to start Solr in a Docker container:
```sh
docker-compose up -d
```
This will pull and start the Solr image defined in `docker-compose.yml`.

### 4. Ensure Dataset Files are Present
Make sure the following dataset files are inside the project directory before proceeding:
```sh
ls -l reddit_stock_posts.csv reddit_stock_comments.csv
```

### 5. Convert Data and Load into Solr
Run the conversion script to process the data and load it into Solr:
```sh
python convert.py
```
