import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader

import os 
import subprocess
import json
import hashlib


DATA_DIR = "data"
TOP_K = 2
TOP_N = 10

OLLAMA_MODEL = "phi"
ST_MODEL = "all-MiniLM-L6-v2"

CHUNK_SIZE = 125
OVERLAP = 25

INDEX_FILE = "faiss.index"
CHUNKS_FILE = "chunks.npy"

META_FILE = "storage/meta.json"
EMBED_FILE = "embeddings.npy"


embedder = SentenceTransformer(ST_MODEL)

def load_docs(data_dir):
    texts = []

    for file in os.listdir(data_dir):
        path = os.path.join(data_dir, file)

        # TXT / MD
        if file.endswith(".txt") or file.endswith(".md"):
            with open(path, "r", encoding="utf-8") as f:
                texts.append(f.read())

        # PDF
        elif file.endswith(".pdf"):
            try:
                reader = PdfReader(path)
                pdf_text = []

                for page in reader.pages:
                    text = page.extract_text()
                    if text:
                        pdf_text.append(text)

                texts.append("\n".join(pdf_text))

            except Exception as e:
                print(f"Error reading {file}: {e}")

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

def compute_data_hash(data_dir):
    hash_md5 = hashlib.md5()

    for file in sorted(os.listdir(data_dir)):
        path = os.path.join(data_dir, file)

        if not (file.endswith(".txt") or file.endswith(".md") or file.endswith(".pdf")):
            continue

        hash_md5.update(file.encode())

        with open(path, "rb") as f:
            while chunk := f.read(4096):
                hash_md5.update(chunk)

    return hash_md5.hexdigest()

def save_meta(data_hash):
    temp_file = META_FILE + ".tmp"

    with open(temp_file, "w") as f:
        json.dump({"data_hash": data_hash}, f)

    os.replace(temp_file, META_FILE)  # atomic write

def load_meta():
    if not os.path.exists(META_FILE):
        return None

    try:
        with open(META_FILE, "r") as f:
            return json.load(f).get("data_hash")
    except Exception:
        return None  # treat as no meta → rebuild

def load_or_build(data_dir):
    current_hash = compute_data_hash(data_dir)
    saved_hash = load_meta()

    if (
        os.path.exists(INDEX_FILE)
        and os.path.exists(CHUNKS_FILE)
        and os.path.exists(EMBED_FILE)
        and current_hash == saved_hash
    ):
        print("Loading existing index...")

        index = faiss.read_index(INDEX_FILE)
        chunks = np.load(CHUNKS_FILE, allow_pickle=True).tolist()
        embeddings = np.load(EMBED_FILE)

        return index, chunks, embeddings

    print("Data changed → rebuilding index...")

    texts = load_docs(data_dir)

    chunks = []
    for t in texts:
        chunks.extend(chunk_text(t))

    X = embedder.encode(chunks, convert_to_numpy=True)
    X = X / np.linalg.norm(X, axis=1, keepdims=True)

    index = faiss.IndexFlatIP(X.shape[1])
    index.add(X)

    # Save
    faiss.write_index(index, INDEX_FILE)
    np.save(CHUNKS_FILE, np.array(chunks))
    np.save(EMBED_FILE, X)
    save_meta(current_hash)

    return index, chunks, X

def retrieve_mmr(query, index, chunks, embeddings, top_k=TOP_K, top_n=TOP_N, lambda_param=0.6):
    q = embedder.encode([query], convert_to_numpy=True)
    q = q / np.linalg.norm(q, axis=1, keepdims=True)

    sims, idxs = index.search(q, top_n)
    candidate_idx = idxs[0]

    selected_idx = []

    for i in range(top_k):
        if i == 0:
            selected_idx.append(candidate_idx[0])
            continue

        mmr_scores = []

        for idx in candidate_idx:
            if idx in selected_idx:
                continue

            doc_vec = embeddings[idx]

            relevance = np.dot(q[0], doc_vec)

            diversity = max(
                np.dot(doc_vec, embeddings[j])
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

def rag(query, index, chunks, embeddings):
    top_chunks = retrieve_mmr(query, index, chunks, embeddings)

    context = "\n\n".join(top_chunks)

    prompt = f"""
Answer ONLY from the context. If not found, say "Not in context".

context:{context}

Question:{query}
    """
    return ask_ollama(prompt)


if __name__ == "__main__":
    print("Loading system...")

    index, chunks, embeddings = load_or_build(DATA_DIR)

    print("Ready.\n")

    while True:
        query = input(">> ")
        if query.lower() == "exit":
            break

        answer = rag(query, index, chunks, embeddings)
        print(answer)
