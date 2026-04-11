import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

import os 
import subprocess


DATA_DIR = "data"
TOP_K = 2
TOP_N = 10

OLLAMA_MODEL = "phi"
ST_MODEL = "all-MiniLM-L6-v2"

CHUNK_SIZE = 125
OVERLAP = 25

INDEX_FILE = "faiss.index"
CHUNKS_FILE = "chunks.npy"


embedder = SentenceTransformer(ST_MODEL)

def load_docs(data_dir):
    texts = []
    for file in os.listdir(data_dir):
        if file.endswith(".txt") or file.endswith(".md"):
            with open(os.path.join(data_dir, file), "r", encoding="utf-8") as f:
                texts.append(f.read())
    return texts

def chunk_text(text):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i+CHUNK_SIZE]
        chunks.append(" ".join(chunk))
        i += CHUNK_SIZE - OVERLAP
    return chunks

def build_index(chunks):
    X = embedder.encode(chunks, convert_to_numpy=True, show_progress_bar=True)
    X = X / np.linalg.norm(X, axis=1, keepdims=True)

    dim = X.shape[1]
    index = faiss.IndexFlatIP(dim) # cosine via normal vector
    index.add(X)

    faiss.write_index(index,INDEX_FILE)
    np.save(CHUNKS_FILE, np.array(chunks))

    return index, chunks

def load_index():
    if os.path.exists(INDEX_FILE) and os.path.exists(CHUNKS_FILE):
        index = faiss.read_index(INDEX_FILE)
        chunks = np.load(CHUNKS_FILE, allow_pickle=True).tolist()
        return index, chunks
    return None, None

def retrieve_mmr(query, index, chunks, top_k=TOP_K, top_n=10, lambda_param=0.6):
    q = embedder.encode([query], convert_to_numpy=True)
    q = q / np.linalg.norm(q, axis=1, keepdims=True)

    # Step 1: FAISS search
    sims, idxs = index.search(q, top_n)
    candidate_idx = idxs[0]
    sims = sims[0]

    selected_idx = []

    for i in range(top_k):
        if i == 0:
            selected_idx.append(candidate_idx[0])
            continue

        mmr_scores = []
        for idx in candidate_idx:
            if idx in selected_idx:
                continue

            relevance = np.dot(q[0], index.reconstruct(int(idx)))

            diversity = max(
                np.dot(index.reconstruct(int(idx)), index.reconstruct(int(j)))
                for j in selected_idx
            )

            score = lambda_param * relevance - (1 - lambda_param) * diversity
            mmr_scores.append((score, idx))

        if not mmr_scores:
            break

        _, best_idx = max(mmr_scores)
        selected_idx.append(best_idx)

    return [chunks[i] for i in selected_idx]

def ask_ollama(prompt):
    result = subprocess.run(
        ["ollama", "run", OLLAMA_MODEL],
        input = prompt.encode(),
        stdout = subprocess.PIPE
    )
    return result.stdout.decode()

def rag(query, X, chunks):
    top_chunks = retrieve_mmr(query, index, chunks)

    context = "\n\n".join(top_chunks)

    prompt = f"""
Answer ONLY from the context. If not found, say "Not in context".

context:{context}

Question:{query}
    """
    return ask_ollama(prompt)

if __name__ == "__main__":
    print("Loading index...")

    index, chunks = load_index()

    if index is None:
        print("Building index...")
        texts = load_docs(DATA_DIR)

        chunks = []
        for t in texts:
            chunks.extend(chunk_text(t))

        index, chunks = build_index(chunks)

    print("Ready.\n")

    while True:
        query = input(">> ")
        if query.lower() == "exit":
            break

        answer = rag(query, index, chunks)
        print(answer)
