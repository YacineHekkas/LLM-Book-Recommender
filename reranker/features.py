# features.py
import math
import re
from collections import Counter, defaultdict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# -------------------------
# Basic tokenizer (simple)
# -------------------------
_token_re = re.compile(r"[a-z0-9]+")

def tokenize(text):
    if text is None:
        return []
    return _token_re.findall(text.lower())

# -------------------------
# TF-IDF vectorizer wrapper (scikit-learn)
# -------------------------
def build_tfidf_vectorizer(docs, max_features=10000, ngram_range=(1,2)):
    vect = TfidfVectorizer(max_features=max_features, stop_words='english', ngram_range=ngram_range, tokenizer=tokenize)
    vect.fit(docs)
    return vect

def cosine_similarity_vec(vec_a, vec_b):
    # vec_a, vec_b are 1-D numpy arrays (dense)
    a_norm = np.linalg.norm(vec_a)
    b_norm = np.linalg.norm(vec_b)
    if a_norm == 0 or b_norm == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (a_norm * b_norm))

# -------------------------
# BM25 implementation (lightweight)
# -------------------------
class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        """
        docs: list[str] (pre-tokenized strings or raw strings)
        We'll tokenize here using tokenize().
        """
        self.k1 = k1
        self.b = b
        self.docs_tokens = [tokenize(d) for d in docs]
        self.N = len(self.docs_tokens)
        self.avgdl = sum(len(d) for d in self.docs_tokens) / max(1, self.N)
        self.doc_freq = defaultdict(int)
        for tokens in self.docs_tokens:
            for t in set(tokens):
                self.doc_freq[t] += 1
        # precompute term frequencies per doc
        self.tf = [Counter(tokens) for tokens in self.docs_tokens]

    def idf(self, term):
        # idf with smoothing
        df = self.doc_freq.get(term, 0)
        # avoid negative idf
        return math.log((self.N - df + 0.5)/(df + 0.5) + 1e-9)

    def score(self, query, doc_index):
        # query: raw text string; doc_index: integer index
        q_tokens = tokenize(query)
        score = 0.0
        doc_len = len(self.docs_tokens[doc_index])
        for term in q_tokens:
            tf = self.tf[doc_index].get(term, 0)
            if tf == 0:
                continue
            idf = self.idf(term)
            denom = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
            score += idf * (tf * (self.k1 + 1)) / denom
        return float(score)

# -------------------------
# Jaccard overlap (set-based)
# -------------------------
def jaccard_similarity(a, b):
    s1 = set(tokenize(a))
    s2 = set(tokenize(b))
    if not s1 and not s2:
        return 0.0
    inter = s1.intersection(s2)
    uni = s1.union(s2)
    return float(len(inter) / len(uni))

# -------------------------
# Keyword match count
# -------------------------
def keyword_count(query, text, keywords):
    # count how many keywords present in both query and text
    q_tokens = set(tokenize(query))
    t_tokens = set(tokenize(text))
    count = sum(1 for kw in keywords if kw.lower() in t_tokens and kw.lower() in q_tokens)
    return count

# -------------------------
# Combine all features into a vector for a pair
# -------------------------
def make_features_for_pair(query, candidate_text, tfidf_vect=None, bm25_obj=None, extra_meta=None, keywords=None):
    """
    tfidf_vect: sklearn vectorizer (fitted)
    bm25_obj: BM25 instance (built over candidate corpus)
    extra_meta: dict with candidate metadata (num_pages, ratings_count, categories)
    keywords: list of keywords to check
    """
    features = {}
    # 1) cosine TF-IDF similarity
    if tfidf_vect is not None:
        q_vec = tfidf_vect.transform([query]).toarray()[0]
        c_vec = tfidf_vect.transform([candidate_text]).toarray()[0]
        features['cosine_tfidf'] = cosine_similarity_vec(q_vec, c_vec)
    else:
        features['cosine_tfidf'] = 0.0

    # 2) bm25
    if bm25_obj is not None:
        # bm25 gives raw score; we keep it
        # NOTE: bm25 expects candidate index; but here we don't have index, so bm25.score with doc_index not available.
        # If you call BM25 built over candidates, you'll use doc index. For single pair, you can call bm25.score(query, idx)
        features['bm25'] = float(bm25_obj.score(query, extra_meta.get('doc_index')) if bm25_obj and extra_meta else 0.0)
    else:
        features['bm25'] = 0.0

    # 3) jaccard
    features['jaccard'] = jaccard_similarity(query, candidate_text)

    # 4) shared token fraction
    q_tokens = set(tokenize(query))
    c_tokens = set(tokenize(candidate_text))
    features['shared_frac'] = float(len(q_tokens.intersection(c_tokens)) / (len(q_tokens) + 1e-9))

    # 5) keywords count
    features['keyword_count'] = keyword_count(query, candidate_text, keywords or [])

    # 6) genre/category match boolean (if categories present)
    if extra_meta and extra_meta.get('categories'):
        cats = [c.strip().lower() for c in str(extra_meta.get('categories')).split(";")]
        features['genre_match'] = 1.0 if any(cat in query.lower() for cat in cats) else 0.0
    else:
        features['genre_match'] = 0.0

    # 7) length diff (abs difference in words)
    features['len_diff'] = abs(len(tokenize(query)) - len(tokenize(candidate_text)))

    # 8) popularity signals
    features['ratings_count'] = float(extra_meta.get('ratings_count', 0.0) if extra_meta else 0.0)
    features['avg_rating'] = float(extra_meta.get('average_rating', 0.0) if extra_meta else 0.0)

    # convert to array (ordered)
    feat_keys = ['cosine_tfidf','bm25','jaccard','shared_frac','keyword_count','genre_match','len_diff','ratings_count','avg_rating']
    feat_vec = np.array([features[k] for k in feat_keys], dtype=float)
    return feat_vec, feat_keys
