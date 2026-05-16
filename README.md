# mini-RAG

A minimal, fully-local Retrieval-Augmented Generation (RAG) system.

## Features
- **Fully local** — no API costs, runs on CPU
- **Lightweight** — `all-MiniLM-L6-v2` embeddings + FAISS
- **Sentence-aware chunking** — preserves context boundaries with overlap
- **Persistent storage** — FAISS index and chunk metadata saved to disk
- **MMR retrieval** — balances relevance and diversity
- **Source attribution** — answers include document name and page number
- **Incremental detection** — rebuilds index automatically when `data/` changes

## Architecture

```
data/ → hash check
       ↓
(no change) → load index
(change)    → rebuild index

Query → FAISS → MMR → Ollama LLM
```

## Components

### 1. Document Loader
Reads `.txt`, `.md`, and `.pdf` files from `data/`.

### 2. Chunking
Splits text into sentence-based chunks with configurable overlap to preserve context across boundaries.

### 3. Embeddings
- Model: `sentence-transformers/all-MiniLM-L6-v2`
- Vectors are normalized for cosine similarity via FAISS `IndexFlatIP`

### 4. Retrieval (MMR)
Fetches top-N candidates from FAISS, then reranks with Maximal Marginal Relevance to select a diverse, relevant top-K.

### 5. Generation
Calls a local Ollama model (default: `phi`) over HTTP. Context is injected into the prompt; answers are grounded in retrieved chunks.

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

The sentence-transformer model downloads on first run (~80 MB).

### 2. Install and run Ollama
```bash
# Pull the model
ollama pull phi

# Start the server (must be running before you query)
ollama serve
```

### 3. Prepare data
Create a `data/` directory and add your documents:
```
project/
├── data/
│   ├── doc1.txt
│   ├── notes.md
│   └── sample.pdf
```

### 4. Run
```bash
python rag.py
```

## Usage
```
>> What is X?
>> Explain Y
>> exit
```

## MMR Explained

Standard top-K retrieval can return repetitive chunks. MMR avoids this by scoring each candidate with:

```
score = λ · relevance − (1 − λ) · diversity
```

where `diversity` is the maximum similarity to already-selected chunks. This produces a set that is both relevant and diverse.

## License
MIT
