# Multi-stage build for Medical NLP Azure Pipeline
FROM python:3.9-slim as builder

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Download required NLTK and spaCy models
RUN python -m nltk.downloader punkt stopwords wordnet averaged_perceptron_tagger maxent_ne_chunker words
RUN python -m spacy download en_core_web_sm

# Production stage
FROM python:3.9-slim

# Create non-root user
RUN groupadd -r nlpuser && useradd -r -g nlpuser nlpuser

# Set working directory
WORKDIR /app

# Copy Python packages from builder stage
COPY --from=builder /root/.local /home/nlpuser/.local
COPY --from=builder /root/nltk_data /home/nlpuser/nltk_data

# Make sure scripts in .local are usable:
ENV PATH=/home/nlpuser/.local/bin:$PATH
ENV NLTK_DATA=/home/nlpuser/nltk_data

# Copy application code
COPY src/ ./src/
COPY config/ ./config/
COPY cli.py ./
COPY README.md ./

# Create necessary directories
RUN mkdir -p models outputs data && \
    chown -R nlpuser:nlpuser /app

# Switch to non-root user
USER nlpuser

# Expose port for API service
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Default command (can be overridden)
CMD ["python", "src/api_service.py"]
