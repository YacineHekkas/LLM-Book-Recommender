# query_index.py
import json, numpy as np, faiss, argparse
from sentence_transformers import SentenceTransformer

MODEL = "all-MiniLM-L6-v2"
EMB_FILE = "book_embs.npy"
META_FILE = "books.json"
INDEX_FILE = "books_hnsw.index"

def load_all():
    model = SentenceTransformer(MODEL)
    index = faiss.read_index(INDEX_FILE)
    with open(META_FILE, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    return model, index, meta

def query(q, k=5):
    model, index, meta = load_all()
    q_emb = model.encode([q], convert_to_numpy=True).astype('float32')
    faiss.normalize_L2(q_emb)
    D, I = index.search(q_emb, k)
    out = []
    for score, idx in zip(D[0].tolist(), I[0].tolist()):
        out.append({"score": float(score), "title": meta[idx]["title"]})
    return out

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--q", required=True)
    p.add_argument("--k", type=int, default=5)
    a = p.parse_args()
    for r in query(a.q, a.k):
        print(f"{r['score']:.4f}\t{r['title']} ")
