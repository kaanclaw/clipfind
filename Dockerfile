FROM python:3.11-slim

# Install ffmpeg
RUN apt-get update && apt-get install -y ffmpeg gcc && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Create required dirs
RUN mkdir -p uploads clips indexes

EXPOSE 8080

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]

# Pre-download ChromaDB ONNX embedding model to avoid cold-start delay
RUN python3 -c "import chromadb; c=chromadb.EphemeralClient(); c.create_collection('warmup'); c.delete_collection('warmup'); print('ChromaDB model cached')" 2>/dev/null || true
