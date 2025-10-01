# build_pair_dataset.py
import json
import random
import numpy as np
from features import build_tfidf_vectorizer, BM25, make_features_for_pair
from joblib import dump

META_FILE = "books.json"
PAIRS_OUT = "train_pairs.npz"  # will save features and labels
VECT_PICKLE = "tfidf_vect.joblib"

def load_meta():
    with open(META_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)

def build_dataset(test_queries, meta, negative_samples=5):
    """
    test_queries: list of dicts, each with {'query':..., 'positives':[title1, title2...]}
    meta: list of book dicts (must contain 'title' etc)
    """
    titles_to_idx = {b['title']: i for i,b in enumerate(meta)}
    docs = [b['text_for_embedding'] for b in meta]  # use that field as doc text
    # build TF-IDF and BM25 on docs
    tfidf = build_tfidf_vectorizer(docs, max_features=10000)
    bm25 = BM25(docs)

    X = []
    y = []

    for q in test_queries:
        query_text = q['query']
        positives = q.get('positives', [])
        if isinstance(positives, str):
            positives = [positives]
        # add positive pairs
        for p in positives:
            if p not in titles_to_idx:
                print("Warning: positive title not found:", p)
                continue
            idx = titles_to_idx[p]
            candidate_text = docs[idx]
            feat_vec, _ = make_features_for_pair(query_text, candidate_text, tfidf, bm25, extra_meta={'doc_index': idx, 'ratings_count': meta[idx].get('ratings_count',0), 'average_rating':meta[idx].get('average_rating',0), 'categories':meta[idx].get('categories','')})
            X.append(feat_vec)
            y.append(1)
        # sample negatives (random or via FAISS later)
        all_idx = list(range(len(meta)))
        neg_candidates = random.sample([i for i in all_idx if meta[i]['title'] not in positives], min(negative_samples, len(meta)-1))
        for nidx in neg_candidates:
            candidate_text = docs[nidx]
            feat_vec, _ = make_features_for_pair(query_text, candidate_text, tfidf, bm25, extra_meta={'doc_index': nidx, 'ratings_count': meta[nidx].get('ratings_count',0), 'average_rating':meta[nidx].get('average_rating',0), 'categories':meta[nidx].get('categories','')})
            X.append(feat_vec)
            y.append(0)

    X = np.vstack(X)
    y = np.array(y, dtype=np.int32)
    # save tfidf vectorizer for inference
    dump(tfidf, VECT_PICKLE)
    np.savez(PAIRS_OUT, X=X, y=y)
    print("Saved dataset:", PAIRS_OUT, "shape:", X.shape)
    return PAIRS_OUT, VECT_PICKLE
