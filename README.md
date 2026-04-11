# Project Overview

A minimal Retrieval-Augmented Generation (RAG) system with:
- Local document ingestion (`.txt`, `.md`)
- chunking with overlap
- Vector embeddings using `SentenceTransformers`
- MMR (Maximal Marginal Relevance) for better retrieval
- Local LLM via `Ollama`

## Features
- Fully local (no API cost)
- Lightweight (CPU-friendly)
- Diversity-aware retrieval (MMR)
- Simple, hackable codebase

## Architecture
```
Documents → Chunking → Embeddings → Vector Index
                                      ↓
Query → Embedding → MMR Retrieval → Context → LLM → Answer
```

## Components

1. Document Loader
Reads `.txt` and `.md` files from `data/`

2. Chunking
Fixed-size word chunks
Overlap to preserve context

3. 🔢 Embeddings
Model: `all-MiniLM-L6-v2`
Normalized vectors for cosine similarity

4. 🎯 Retrieval (MMR)

Balances:
```
Relevance
Diversity
```

5. Generation

Uses `ollama run phi`
Context-grounded answers only

## Setup

### 1. Install dependencies
```pip install sentence-transformers numpy```

### 2. Install Ollama
Install Ollama
Pull model:
```ollama pull phi```

### 3. Prepare data
```
project/
 ├── data/
 │    ├── doc1.txt
 │    ├── notes.md
```

### 4. Run
```
python rag.py
```

## Usage
```
>> What is X?
>> Explain Y
>> exit
```

## 🧠 MMR Explained

MMR improves retrieval by avoiding redundant chunks.

Instead of:

Top-K → similar but repetitive

It does:

Top-N → MMR → diverse + relevant

### Example Flow
Query: ```What is overfitting?```

- Embedding
- Retrieve top 10 chunks
- Apply MMR
- Select 2 diverse chunks
- Send to LLM
- Generate grounded answer

### Limitations
- No persistent vector DB (in-memory only)
- Basic chunking (word-based)
- No metadata filtering
- No reranking beyond MMR

📜 License

MIT
