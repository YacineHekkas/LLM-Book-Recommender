# eval.py
import json, numpy as np, faiss
from sentence_transformers import SentenceTransformer

META = "books.json"
INDEX = "books_hnsw.index"
EMB = "book_embs.npy"
MODEL = "all-MiniLM-L6-v2"

def load():
    with open(META, "r", encoding="utf-8") as f:
        meta = json.load(f)
    idx = faiss.read_index(INDEX)
    model = SentenceTransformer(MODEL)
    return meta, idx, model

def evaluate(test_queries_path, k=10):
    meta, idx, model = load()
    with open(test_queries_path, "r", encoding="utf-8") as f:
        tests = json.load(f)

    precisions = []
    rr_sum = 0.0

    for t in tests:
        q = t["query"]
        pos = t.get("positives", [])
        # normalize positives to a set
        if isinstance(pos, str):
            positives = {pos}
        else:
            positives = set(pos)

        # encode query
        q_emb = model.encode([q], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(q_emb)
        D, I = idx.search(q_emb, k)
        retrieved = [meta[i]["title"] for i in I[0]]

        # ---- Debug prints ----
        print("=" * 50)
        print("Query:", q)
        print("Positives:", positives)
        print("Retrieved:", retrieved)

        # precision@k
        p = sum(1 for r in retrieved if r in positives) / k
        precisions.append(p)

        # reciprocal rank
        rr = 0.0
        for rank, r in enumerate(retrieved, start=1):
            if r in positives:
                rr = 1.0 / rank
                break
        rr_sum += rr

    print("=" * 50)
    print("Precision@k:", sum(precisions) / len(precisions))
    print("MRR:", rr_sum / len(tests))

if __name__ == "__main__":
    import sys
    evaluate(sys.argv[1])
