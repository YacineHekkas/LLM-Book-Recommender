# reranker/rerank_inference.py
import json
import numpy as np
import faiss
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from joblib import load
import os

# === Config - adjust paths if needed ===
META_FILE = "./books.json"           # path to your metadata
INDEX_FILE = "./books_hnsw.index"    # path to faiss index
BI_MODEL = "all-MiniLM-L6-v2"         # model used for embeddings
RERANKER_PT = "reranker.pt"           # your PyTorch reranker saved earlier
TFIDF_PICKLE = "tfidf_vect.joblib"    # you created this
TOP_N = 50
TOP_K = 10
HIDDEN_DIM = 64
DEVICE = "cpu"

# === Reranker class - must match training architecture ===
class Reranker(nn.Module):
    def __init__(self, q_dim, d_dim, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.q_proj = nn.Linear(q_dim, hidden_dim)
        self.d_proj = nn.Linear(d_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, q, d):
        q_p = self.q_proj(q)
        d_p = self.d_proj(d)
        x = torch.cat([q_p, d_p], dim=1)
        return torch.sigmoid(self.fc(x)).squeeze(-1)

# === helpers ===
def load_assets():
    print("Loading metadata...")
    with open(META_FILE, "r", encoding="utf-8") as f:
        meta = json.load(f)
    print("Loading FAISS index:", INDEX_FILE)
    index = faiss.read_index(INDEX_FILE)
    bi = SentenceTransformer(BI_MODEL)
    tfidf = None
    if os.path.exists(TFIDF_PICKLE):
        try:
            tfidf = load(TFIDF_PICKLE)
            print("Loaded TF-IDF vectorizer:", TFIDF_PICKLE)
        except Exception as e:
            print("Failed to load TF-IDF vectorizer:", e)
            tfidf = None
    else:
        print("TF-IDF vectorizer not found; continuing without it.")
    return meta, index, bi, tfidf

def build_model_and_load(q_dim, d_dim, reranker_path, device="cpu"):
    model = Reranker(q_dim, d_dim).to(device)
    if not os.path.exists(reranker_path):
        raise FileNotFoundError(f"{reranker_path} not found. Train the reranker and place the file here.")
    state = torch.load(reranker_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model

def retrieve_candidates(bi_model, index, query_text, top_n=TOP_N):
    q_emb = bi_model.encode([query_text], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(q_emb)
    D, I = index.search(q_emb, top_n)
    return D[0], I[0].tolist()

def make_doc_text(meta_item):
    return meta_item.get("text_for_embedding") or " — ".join([meta_item.get("title",""), meta_item.get("description","")])

def rerank_query(query, meta, index, bi_model, reranker_model, top_n=TOP_N, top_k=TOP_K, device="cpu"):
    D, ids = retrieve_candidates(bi_model, index, query, top_n)
    cand_texts = [make_doc_text(meta[i]) for i in ids]

    # embed query and candidates with same bi-model used during pair generation
    q_emb = bi_model.encode([query], convert_to_numpy=True).astype("float32")      # (1, d)
    d_embs = bi_model.encode(cand_texts, convert_to_numpy=True).astype("float32")  # (top_n, d)
    q_embs = np.repeat(q_emb, repeats=d_embs.shape[0], axis=0)                     # (top_n, d)

    q_t = torch.from_numpy(q_embs).float().to(device)
    d_t = torch.from_numpy(d_embs).float().to(device)

    with torch.no_grad():
        scores = reranker_model(q_t, d_t).cpu().numpy()

    combined = list(zip(ids, D.tolist(), scores, cand_texts))
    ranked = sorted(combined, key=lambda x: x[2], reverse=True)[:top_k]

    results = []
    for idx, faiss_score, rerank_score, snippet in ranked:
        b = meta[idx]
        results.append({
            "id": b.get("isbn13") or b.get("title"),
            "title": b.get("title"),
            "authors": b.get("authors", ""),
            "faiss_score": float(faiss_score),
            "rerank_score": float(rerank_score),
            "snippet": snippet[:300]
        })
    return results

# === CLI ===
if __name__ == "__main__":
    meta, index, bi, tfidf = load_assets()
    sample_emb = bi.encode(["hello"], convert_to_numpy=True).astype("float32")
    q_dim = sample_emb.shape[1]
    d_dim = sample_emb.shape[1]
    print("Embedding dimension:", q_dim)
    reranker = build_model_and_load(q_dim, d_dim, RERANKER_PT, device=DEVICE)
    print("Reranker loaded:", RERANKER_PT)

    while True:
        q = input("\nQuery (or 'exit')> ").strip()
        if not q or q.lower() in ("exit", "quit"):
            break
        out = rerank_query(q, meta, index, bi, reranker)
        import json as _j
        print(_j.dumps(out, indent=2, ensure_ascii=False))
