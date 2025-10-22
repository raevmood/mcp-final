FROM python:3.9-slim-bullseye as builder
WORKDIR /app

# Install system dependencies and build SQLite from source
RUN apt-get update && apt-get install -y --no-install-recommends \
  build-essential \
  gcc \
  g++ \
  cmake \
  wget \
  tar \
  && wget https://www.sqlite.org/2024/sqlite-autoconf-3460000.tar.gz \
  && tar xzf sqlite-autoconf-3460000.tar.gz \
  && cd sqlite-autoconf-3460000 \
  && ./configure --prefix=/usr/local \
  && make -j$(nproc) && make install \
  && rm -rf /var/lib/apt/lists/* sqlite-autoconf-3460000*

# Verify SQLite version
RUN sqlite3 --version

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---- runtime stage ----
FROM python:3.9-slim-bullseye
WORKDIR /app

# Copy built SQLite and Python deps from builder
COPY --from=builder /usr/local/lib /usr/local/lib
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy app files
COPY mcp_server.py ingest_data.py knowledge_base.txt ./

# Validate knowledge base exists
RUN if [ ! -s knowledge_base.txt ]; then \
  echo "ERROR: knowledge_base.txt is missing or empty!"; \
  exit 1; \
  fi

# Ingest data
RUN mkdir -p /app/chroma_rag_db && \
  echo "Starting data ingestion..." && \
  python ingest_data.py /app/knowledge_base.txt && \
  echo "Verifying ingestion..." && \
  python ingest_data.py --stats

EXPOSE 8001
CMD ["uvicorn", "mcp_server:app", "--host", "0.0.0.0", "--port", "8001"]
