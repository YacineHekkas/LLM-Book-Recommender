# build_faiss_index.py
import numpy as np
import faiss

EMB_FILE = "book_embs.npy"       # or book_embs_tfidf.npy if you used TF-IDF
INDEX_OUT = "books_hnsw.index"

embs = np.load(EMB_FILE).astype('float32')
print("Loaded embeddings", embs.shape)

# normalize to use inner product as cosine similarity
faiss.normalize_L2(embs)

d = embs.shape[1]
index = faiss.IndexHNSWFlat(d, 32)   # M=32
index.hnsw.efConstruction = 40
index.add(embs)
faiss.write_index(index, INDEX_OUT)
print("Wrote index to", INDEX_OUT)
