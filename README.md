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
- persistent vector storage
- Embedding and index saved to disk
- no recomputation on restart
- faster startup
- load prebuilt index in miliseconds
- scalable retrieval
- handles large data effenciently
- pdf support

## whats new 
make RAG adaptive by updating the FAISS database when data updates- 
- new file is added to `data` dir
- a file is removed from `data` dir


## Architecture
```
data/ → hash check
       ↓
(no change) → load index
(change)    → rebuild index

Query → FAISS → MMR → LLM
```

## Components

### 1. Document Loader
Reads `.txt`, `.md` and `.pdf` files from `data/`

### 2. Chunking
Fixed-size word chunks
Overlap to preserve context

### 3. Embeddings
Model: `all-MiniLM-L6-v2`
Normalized vectors for cosine similarity

### 4. Retrieval (MMR)
Balances: Relevance and Diversity

### 5. Generation
Uses `ollama run phi`
Context-grounded answers only

## Setup
### 1. Install dependencies
```pip install sentence-transformers faiss-cpu pypdf numpy```

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
 │    ├── sample.pdf
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

## MMR Explained

MMR improves retrieval by avoiding redundant chunks.
Instead of:
Top-K → similar but repetitive
It does:
Top-N → MMR → diverse + relevant

### Example Flow
- Query
- Embed
- FAISS (top-N)
- MMR rerank
- Top-K chunks
- LLM

## 📜 License
MIT
