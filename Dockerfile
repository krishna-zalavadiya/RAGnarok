FROM python:3.10-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy model
RUN python -m spacy download en_core_web_sm

# Pre-download sentence-transformer models into image
RUN python -c "
from sentence_transformers import SentenceTransformer, CrossEncoder
SentenceTransformer('all-MiniLM-L6-v2')
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
"

# Copy source code
COPY config.py .
COPY rank.py .
COPY precompute.py .
COPY precompute_llm.py .
COPY build_indexes.py .
COPY job_description.md .
COPY parsed_job_description.json .
COPY submission_metadata.yaml .
COPY pipeline/ ./pipeline/
COPY indexing/ ./indexing/
COPY ontology/ ./ontology/
COPY retrieval/ ./retrieval/
COPY scoring/ ./scoring/
COPY trust/ ./trust/
COPY scripts/ ./scripts/
COPY api/ ./api/

# Copy pre-built indexes (the key part)
COPY data/indexes/ ./data/indexes/

# Download and bake in the LLM model
RUN python precompute_llm.py

# Create output directory
RUN mkdir -p output

# Default command
CMD ["python", "rank.py", \
     "--input", "data/candidates.jsonl", \
     "--output", "output/submission.csv", \
     "--top-k", "100"]

#      # 1. Pull the image
# docker pull yourusername/ragnarok:v1

# # 2. Place their candidates.jsonl in current directory
# # then run:
# docker run \
#   -v $(pwd)/candidates.jsonl:/app/data/candidates.jsonl \
#   -v $(pwd)/output:/app/output \
#   yourusername/ragnarok:v1

# # 3. Output appears in ./output/submission.csv