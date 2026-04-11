import numpy as np
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
    return X

def retrieve_mmr(query, X, chunks, top_k=TOP_K, lambda_param=0.7, top_n=TOP_N):
    q = embedder.encode([query], convert_to_numpy=True)
    q = q / np.linalg.norm(q, axis=1, keepdims=True)

    sims = np.dot(q, X.T).flatten()

    candidate_idx = np.argsort(sims)[::-1][:top_n]

    selected = []
    selected_idx = []

    for _ in range(top_k):
        if len(selected_idx) == 0:
            idx = candidate_idx[0]
            selected_idx.append(idx)
            continue

        mmr_scores = []

        for idx in candidate_idx:
            if idx in selected_idx:
                continue

            relevance = sims[idx]

            diversity = max(
                np.dot(X[idx], X[j]) for j in selected_idx
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
    top_chunks = retrieve_mmr(query, X, chunks)

    context = "\n\n".join(top_chunks)

    prompt = f"""
Answer ONLY from the context. If not found, say "Not in context".

context:{context}

Question:{query}
    """
    return ask_ollama(prompt)

if __name__ == "__main__":
    print("loading documents...")
    texts = load_docs(DATA_DIR)

    chunks = []
    for t in texts:
        chunks.extend(chunk_text(t))

    print("Building Index...")
    X = build_index(chunks)

    print("Ready. Ask Questions (type 'exit' to quit)\n")

    while True:
        query = input(">> ")
        if query.lower() == "exit":
            break

        answer = rag(query, X, chunks)
        print(answer)
