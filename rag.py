import os
import re
import json
import hashlib
import urllib.request
import urllib.error

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader


DATA_DIR = "data"
TOP_K = 2
TOP_N = 10

OLLAMA_MODEL = "phi"
ST_MODEL = "all-MiniLM-L6-v2"

CHUNK_SIZE = 125
OVERLAP = 25

INDEX_FILE = "faiss.index"
CHUNKS_FILE = "chunks.json"
META_FILE = "storage/meta.json"


def load_docs(data_dir):
    """Load supported documents from data_dir with source metadata."""
    docs = []
    for file in sorted(os.listdir(data_dir)):
        path = os.path.join(data_dir, file)
        if not os.path.isfile(path):
            continue

        if file.endswith((".txt", ".md")):
            with open(path, "r", encoding="utf-8") as f:
                docs.append({"text": f.read(), "source": file, "page": None})

        elif file.endswith(".pdf"):
            try:
                reader = PdfReader(path)
                for i, page in enumerate(reader.pages):
                    text = page.extract_text()
                    if text:
                        docs.append({"text": text, "source": file, "page": i + 1})
            except Exception as e:
                print(f"Error reading {file}: {e}")

    return docs


def chunk_docs(docs, chunk_size=CHUNK_SIZE, overlap=OVERLAP):
    """Sentence-aware chunking with word-budget overlap."""
    chunks = []
    for doc in docs:
        sentences = re.split(r"(?<=[.!?])\s+", doc["text"])
        sentences = [s.strip() for s in sentences if s.strip()]

        i = 0
        while i < len(sentences):
            chunk_sentences = []
            words = 0
            j = i
            while j < len(sentences) and words < chunk_size:
                chunk_sentences.append(sentences[j])
                words += len(sentences[j].split())
                j += 1

            if not chunk_sentences:
                break

            chunks.append({
                "text": " ".join(chunk_sentences),
                "source": doc["source"],
                "page": doc["page"],
            })

            # step back for overlap (guaranteed progress)
            overlap_words = 0
            overlap_count = 0
            for s in reversed(chunk_sentences):
                if overlap_words >= overlap:
                    break
                overlap_words += len(s.split())
                overlap_count += 1

            i = max(j - overlap_count, i + 1)

    return chunks


def compute_data_hash(data_dir):
    """Content-addressable hash of supported files in data_dir."""
    hash_md5 = hashlib.md5()
    for file in sorted(os.listdir(data_dir)):
        path = os.path.join(data_dir, file)
        if not os.path.isfile(path):
            continue
        if not file.endswith((".txt", ".md", ".pdf")):
            continue
        hash_md5.update(file.encode())
        with open(path, "rb") as f:
            while chunk := f.read(4096):
                hash_md5.update(chunk)
    return hash_md5.hexdigest()


def save_meta(data_hash):
    os.makedirs(os.path.dirname(META_FILE), exist_ok=True)
    temp_file = META_FILE + ".tmp"
    with open(temp_file, "w", encoding="utf-8") as f:
        json.dump({"data_hash": data_hash}, f)
    os.replace(temp_file, META_FILE)


def load_meta():
    if not os.path.exists(META_FILE):
        return None
    try:
        with open(META_FILE, "r", encoding="utf-8") as f:
            return json.load(f).get("data_hash")
    except Exception:
        return None


def load_or_build(data_dir, embedder):
    """Load cached index or rebuild if data changed."""
    current_hash = compute_data_hash(data_dir)
    saved_hash = load_meta()

    if (
        os.path.exists(INDEX_FILE)
        and os.path.exists(CHUNKS_FILE)
        and current_hash == saved_hash
    ):
        print("Loading existing index...")
        index = faiss.read_index(INDEX_FILE)
        with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        return index, chunks

    print("Data changed → rebuilding index...")
    docs = load_docs(data_dir)

    if not docs:
        print(f"Warning: no supported documents found in '{data_dir}'.")
        dim = embedder.get_sentence_embedding_dimension()
        index = faiss.IndexFlatIP(dim)
        return index, []

    chunks = chunk_docs(docs)
    texts = [c["text"] for c in chunks]

    X = embedder.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    X = X / np.linalg.norm(X, axis=1, keepdims=True)

    dim = X.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(X)

    faiss.write_index(index, INDEX_FILE)
    with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    save_meta(current_hash)

    return index, chunks


def retrieve_mmr(query, index, chunks, embedder, top_k=TOP_K, top_n=TOP_N, lambda_param=0.6):
    """Retrieve top_k chunks using Maximal Marginal Relevance."""
    if not chunks:
        return []

    q = embedder.encode([query], convert_to_numpy=True)
    q = q / np.linalg.norm(q, axis=1, keepdims=True)

    sims, idxs = index.search(q, top_n)
    candidate_idx = [int(i) for i in idxs[0] if int(i) >= 0]

    if not candidate_idx:
        return []

    # Pre-fetch embeddings for candidate docs from FAISS
    candidate_embeds = {idx: index.reconstruct(idx) for idx in candidate_idx}

    selected_idx = []
    for i in range(top_k):
        if i == 0:
            selected_idx.append(candidate_idx[0])
            continue

        mmr_scores = []
        for idx in candidate_idx:
            if idx in selected_idx:
                continue
            doc_vec = candidate_embeds[idx]
            relevance = np.dot(q[0], doc_vec)
            diversity = max(
                np.dot(doc_vec, candidate_embeds[j])
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
    """Call Ollama via local HTTP API (keeps model warm, no per-query subprocess)."""
    payload = json.dumps({
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
    }).encode("utf-8")

    req = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("response", "").strip()
    except urllib.error.URLError as e:
        return f"[Error: Cannot reach Ollama at localhost:11434. Is it running? ({e})]"
    except Exception as e:
        return f"[Error: {e}]"


def rag(query, index, chunks, embedder):
    top_chunks = retrieve_mmr(query, index, chunks, embedder)

    if not top_chunks:
        return "No relevant context found."

    context_parts = []
    for c in top_chunks:
        source = c["source"]
        page_info = f" (page {c['page']})" if c["page"] else ""
        context_parts.append(f"[Source: {source}{page_info}]\n{c['text']}")

    context = "\n\n".join(context_parts)

    prompt = f"""Answer ONLY from the context below. If the answer is not found, say "Not in context".

{context}

Question: {query}
"""
    return ask_ollama(prompt)


if __name__ == "__main__":
    print("Loading system...")

    embedder = SentenceTransformer(ST_MODEL)

    if not os.path.isdir(DATA_DIR):
        print(f"Error: data directory '{DATA_DIR}' not found.")
        exit(1)

    index, chunks = load_or_build(DATA_DIR, embedder)

    if not chunks:
        print("Warning: index is empty. Add documents to 'data/' and restart.")

    print(f"Indexed {len(chunks)} chunks. Ready.\n")

    while True:
        try:
            query = input(">>")
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if query.lower().strip() == "exit":
            break
        if not query.strip():
            continue

        answer = rag(query, index, chunks, embedder)
        print(answer)
