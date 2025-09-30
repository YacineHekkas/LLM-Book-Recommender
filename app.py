# app.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import json, numpy as np, faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
from pathlib import Path

# Config
MODEL_BI = "all-MiniLM-L6-v2"
MODEL_CROSS = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # optional
EMB_FILE = "book_embs.npy"
META_FILE = "books.json"
INDEX_FILE = "books_hnsw.index"

# Load once
bi = SentenceTransformer(MODEL_BI)
# lazy load cross when needed to save memory
cross = None

embs = np.load(EMB_FILE).astype('float32')
with open(META_FILE, 'r', encoding='utf-8') as f:
    books = json.load(f)
index = faiss.read_index(INDEX_FILE)

app = FastAPI(title="Book Recommender")

class Req(BaseModel):
    prompt: str
    feeling: Optional[str] = ""
    k: Optional[int] = 6
    rerank: Optional[bool] = True   # set False to skip cross-encoder

def make_reason(prompt, feeling, book):
    # deterministic, safe reasoner: combines keyword + metadata matching
    reasons = []
    p = (prompt or "").lower()
    # genres/categories match
    cats = book.get("categories") or ""
    if isinstance(cats, list):
        cats_list = cats
    else:
        cats_list = [s.strip() for s in str(cats).split(";") if s.strip()]
    for c in cats_list:
        if c.lower() in p:
            reasons.append(f"Matches category '{c}'")
    # keyword match (small heuristics)
    keywords = ["wizard","magical","fox","romance","quantum","machine learning","murder","detective","thriller","inspiring","funny"]
    desc = (book.get("description") or "").lower()
    for kw in keywords:
        if kw in p and kw in desc:
            reasons.append(f"Contains keyword '{kw}' in description")
    if feeling:
        reasons.append(f"Suggested for mood: {feeling}")
    if not reasons:
        reasons.append("Top semantic match to your query.")
    return "; ".join(reasons)

@app.post("/recommend")
def recommend(r: Req):
    global cross
    qtext = (r.prompt or "") + (" | mood: " + r.feeling if r.feeling else "")
    # embed query
    q_emb = bi.encode([qtext], convert_to_numpy=True).astype('float32')
    faiss.normalize_L2(q_emb)
    TOP_N = max(50, r.k * 5)   # retrieve more to rerank
    D, I = index.search(q_emb, TOP_N)
    cand_ids = I[0].tolist()
    candidate_texts = [(cid, (books[cid].get("title","") + " — " + (books[cid].get("description") or "") ) ) for cid in cand_ids]

    # rerank with cross-encoder if requested
    if r.rerank:
        if cross is None:
            cross = CrossEncoder(MODEL_CROSS)  # lazy load
        pairs = [ (qtext, text) for _, text in candidate_texts ]
        scores = cross.predict(pairs)  # list of floats
        ranked = sorted(zip(cand_ids, scores), key=lambda x: x[1], reverse=True)[:r.k]
    else:
        # use FAISS scores (D) as base
        ranked = list(zip(cand_ids[:r.k], D[0].tolist()[:r.k]))

    results = []
    for cid, score in ranked:
        book = books[cid]
        results.append({
            "id": book.get("isbn13") or book.get("isbn10") or str(cid),
            "title": book.get("title"),
            "authors": book.get("authors"),
            "categories": book.get("categories"),
            "published_year": book.get("published_year"),
            "avg_rating": book.get("average_rating"),
            "score": float(score),
            "reason": make_reason(r.prompt, r.feeling, book)
        })
    return {"results": results}
