FROM python:3.9-slim

# Install Java for Solr and build dependencies for Python packages
RUN apt-get update && \
    apt-get install -y default-jre wget unzip curl gcc build-essential && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
WORKDIR /app
COPY ../requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install streamlit nltk wordcloud matplotlib faiss-cpu langchain langchain_text_splitters langchain_community langchain_openai pysolr

# Download and setup Solr
ENV SOLR_VERSION=8.11.2
RUN wget https://archive.apache.org/dist/lucene/solr/${SOLR_VERSION}/solr-${SOLR_VERSION}.tgz && \
    tar xzf solr-${SOLR_VERSION}.tgz && \
    mv solr-${SOLR_VERSION} /opt/solr && \
    rm solr-${SOLR_VERSION}.tgz

# Add Solr to PATH
ENV PATH="/opt/solr/bin:${PATH}"

# Download NLTK data
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet'); nltk.download('stopwords')"

# Copy application code
COPY . /app/UI
COPY ../Classification/NER_with_ensemble_sentiment.csv /app/Classification/NER_with_ensemble_sentiment.csv
COPY ../.env /app/.env

# Create Solr core for stock sentiment data
RUN /opt/solr/bin/solr start -force && \
    /opt/solr/bin/solr create -c stock_sentiment -force && \
    /opt/solr/bin/solr stop

# Create a script to import CSV to Solr
RUN echo '#!/bin/bash\n\
solr start -force\n\
# Wait for Solr to start\n\
sleep 10\n\
# Set up schema\n\
if [ -f /app/setup_solr_schema.sh ]; then\n\
  chmod +x /app/setup_solr_schema.sh\n\
  /app/setup_solr_schema.sh\n\
  echo "Schema setup completed"\n\
fi\n\
# Post the CSV data to Solr\n\
echo "Importing CSV data..."\n\
/opt/solr/bin/post -c stock_sentiment -f csv -commit yes /app/Classification/NER_with_ensemble_sentiment.csv\n\
echo "Data import completed"\n\
# Start Streamlit app\n\
cd /app/UI\n\
streamlit run streamlit_app_integrated.py --server.port=8501\n\
' > /app/start.sh && \
    chmod +x /app/start.sh

EXPOSE 8501 8983

CMD ["/app/start.sh"] 