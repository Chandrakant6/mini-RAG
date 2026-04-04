import numpy as np 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import os 
import subprocess


DATA_DIR = "data"
TOP_K = 3
MODEL = "phi"


def load_docs(data_dir):
    texts = []
    for file in os.listdir(data_dir):
        if file.endswith(".txt") or file.endswith(".md"):
            with open(os.path.join(data_dir, file), "r", encoding="utf-8") as f:
                texts.append(f.read())
    return texts

def chunk_text(text, chunk_size=250, overlap=50):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i+chunk_size]
        chunks.append(" ".join(chunk))
        i += chunk_size - overlap
    return chunks

def build_index(texts):
    chunks = []
    for t in texts:
        chunks.extend(chunk_text(t))

    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(chunks)

    return vectorizer, X, chunks


def retrive(query, vectorizer, X, chunks):
    q_vec = vectorizer.transform([query])
    sims = cosine_similarity(q_vec, X).flatten()

    top_idx = np.argsort(sims)[::-1][:TOP_K]
    return [chunks[i] for i in top_idx]


def ask_ollama(prompt):
    result = subprocess.run(
        ["ollama", "run", MODEL],
        input = prompt.encode(),
        stdout = subprocess.PIPE
    )
    return result.stdout.decode()

def rag(query, vectorizer, X, chunks):
    top_chunks = retrive(query, vectorizer, X, chunks)

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

    print("Building Index...")
    vectorizer, X, chunks = build_index(texts)

    print("Ready. Ask Questions (type 'exit' to quit)\n")

    while True:
        query = input(">> ")
        if query.lower() == "exit":
            break

        answer = rag(query, vectorizer, X, chunks)
        print(answer)
