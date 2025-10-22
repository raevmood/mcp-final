FROM python:3.9-slim-bullseye as builder
WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
  build-essential \
  gcc \
  g++ \
  cmake \
  && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

FROM python:3.9-slim-bullseye
WORKDIR /app

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application files
COPY mcp_server.py ingest_data.py knowledge_base.txt ./

# Validate knowledge base exists and is not empty
RUN if [ ! -s knowledge_base.txt ]; then \
  echo "ERROR: knowledge_base.txt is missing or empty!"; \
  exit 1; \
  fi

# Create data directory and ingest during build
RUN mkdir -p /app/chroma_rag_db && \
  echo "Starting data ingestion..." && \
  python ingest_data.py /app/knowledge_base.txt && \
  echo "Verifying ingestion..." && \
  python ingest_data.py --stats

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python -c "import requests; requests.get('http://localhost:8001/').raise_for_status()" || exit 1

EXPOSE 8001

# Start server
CMD ["uvicorn", "mcp_server:app", "--host", "0.0.0.0", "--port", "8001"]