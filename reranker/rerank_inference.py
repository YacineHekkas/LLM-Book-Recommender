# rerank_inference.py
import json, numpy as np, faiss
from joblib import load
from features import make_features_for_pair, build_tfidf_vectorizer, BM25
from sentence_transformers import SentenceTransformer

META = "books.json"
INDEX = "books_hnsw.index"
EMB = "book_embs.npy"
VECT_PICKLE = "tfidf_vect.joblib"
RERANKER_MODEL = "reranker_lr.joblib"

# load meta and index
meta = json.load(open(META, 'r', encoding='utf-8'))
index = faiss.read_index(INDEX)
# bi-encoder for embedding queries (optional)
bi = SentenceTransformer("all-MiniLM-L6-v2")
# load tfidf vectorizer and reranker
tfidf = load(VECT_PICKLE)
reranker = load(RERANKER_MODEL)
# build BM25 over docs (for bm25 scoring)
docs = [b['text_for_embedding'] for b in meta]
bm25 = BM25(docs)

def retrieve_faiss(query, top_n=50):
    q_emb = bi.encode([query], convert_to_numpy=True).astype('float32')
    faiss.normalize_L2(q_emb)
    D, I = index.search(q_emb, top_n)
    return D[0].tolist(), I[0].tolist()

def rerank(query, top_n=50, k=10):
    D, ids = retrieve_faiss(query, top_n)
    candidate_texts = [meta[i]['text_for_embedding'] for i in ids]
    feats = []
    extras = []
    for pos, idx in enumerate(ids):
        extra = {'doc_index': idx, 'ratings_count': meta[idx].get('ratings_count',0), 'average_rating': meta[idx].get('average_rating',0), 'categories':meta[idx].get('categories','')}
        feat_vec, keys = make_features_for_pair(query, candidate_texts[pos], tfidf, bm25, extra_meta=extra, keywords=[])
        feats.append(feat_vec)
        extras.append((idx, candidate_texts[pos]))
    X = np.vstack(feats)
    scores = reranker.predict_proba(X)[:,1]  # probability of relevance
    ranked_idx = np.argsort(-scores)[:k]
    results = []
    for r in ranked_idx:
        idx, text = extras[r]
        b = meta[idx]
        results.append({
            'title': b.get('title'),
            'authors': b.get('authors',''),
            'score': float(scores[r]),
            'faiss_score': float(D[r]),  # note: D[r] corresponds to original order; ok for rough info
            'snippet': text[:250]
        })
    return results

if __name__ == "__main__":
    while True:
        q = input("Query > ").strip()
        if not q:
            break
        res = rerank(q, top_n=50, k=10)
        import json as _j
        print(_j.dumps(res, indent=2, ensure_ascii=False))
