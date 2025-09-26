# build_embeddings.py
import pandas as pd
import numpy as np
import json
from sentence_transformers import SentenceTransformer

CSV_FILE = "books.csv"
JSON_FILE = "books.json"
EMB_FILE = "book_embs.npy"
MODEL_NAME = "all-MiniLM-L6-v2"


def prepare_metadata():
    """Load CSV and build metadata list with text_for_embedding."""
    df = pd.read_csv(CSV_FILE)

    # Fill NaN with empty string
    df = df.fillna("")

    books = []
    for _, row in df.iterrows():
        text_parts = [
            str(row.get("title", "")),
            str(row.get("subtitle", "")),
            str(row.get("description", "")),
            str(row.get("categories", "")),
        ]
        text_for_embedding = " | ".join([t for t in text_parts if t.strip()])

        book = {
            "title": row.get("title", ""),
            "subtitle": row.get("subtitle", ""),
            "description": row.get("description", ""),
            "categories": row.get("categories", ""),
            "text_for_embedding": text_for_embedding,
        }
        books.append(book)

    with open(JSON_FILE, "w", encoding="utf-8") as f:
        json.dump(books, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(books)} books to {JSON_FILE}")
    return books


def build_embeddings(books):
    """Encode book texts into dense embeddings."""
    model = SentenceTransformer(MODEL_NAME)
    texts = [b["text_for_embedding"] for b in books]
    print(f"Encoding {len(texts)} books with {MODEL_NAME}...")

    # encode in batches
    embs = model.encode(texts, batch_size=32, convert_to_numpy=True, show_progress_bar=True)
    embs = embs.astype("float32")

    # optional: normalize for cosine similarity
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    embs = embs / norms

    np.save(EMB_FILE, embs)
    print(f"Saved embeddings to {EMB_FILE}, shape={embs.shape}")


if __name__ == "__main__":
    books = prepare_metadata()
    build_embeddings(books)
