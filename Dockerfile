# ----------------------------
# Stage 1: Builder
# ----------------------------
FROM python:3.11-slim-bookworm AS builder
WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
  build-essential gcc g++ cmake \
  && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ----------------------------
# Stage 2: Runtime
# ----------------------------
FROM python:3.11-slim-bookworm
WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11 /usr/local/lib/python3.11
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy your application files
COPY mcp_server.py ingest_data.py knowledge_base.txt ./

# Validate that the knowledge base exists and is not empty
RUN if [ ! -s knowledge_base.txt ]; then \
  echo "ERROR: knowledge_base.txt is missing or empty!"; \
  exit 1; \
  fi

# Create data directory
RUN mkdir -p /app/chroma_rag_db

# Optional: Pre-ingest data during build (can comment out if you prefer runtime ingestion)
RUN echo "Starting data ingestion..." && \
  python ingest_data.py /app/knowledge_base.txt && \
  echo "Verifying ingestion..." && \
  python ingest_data.py --stats

# Health check for container monitoring
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python -c "import requests; requests.get('http://localhost:8001/').raise_for_status()" || exit 1

EXPOSE 8001

# Start the server
CMD ["uvicorn", "mcp_server:app", "--host", "0.0.0.0", "--port", "8001"]
