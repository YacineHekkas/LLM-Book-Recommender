# prepare_data.py
import pandas as pd
import json
import html
import re

IN = "books.csv"
OUT = "books.json"

def clean_text(s):
    if pd.isna(s):
        return ""
    s = str(s)
    s = html.unescape(s)            # unescape html entities
    s = re.sub(r"\s+", " ", s).strip()
    return s

def text_for_embedding(row):
    parts = []
    for col in ["title", "title_and_subtitle", "description", "categories", "tagged_description"]:
        if col in row and not pd.isna(row[col]):
            parts.append(str(row[col]))
    # join with a long separator to avoid accidental merges
    return " — ".join(clean_text(p) for p in parts if p)

def main():
    df = pd.read_csv(IN)
    df.fillna("", inplace=True)
    out = []
    for _, r in df.iterrows():
        item = {
            "isbn13": clean_text(r.get("isbn13","")),
            "isbn10": clean_text(r.get("isbn10","")),
            "title": clean_text(r.get("title","")),
            "authors": clean_text(r.get("authors","")),
            "categories": clean_text(r.get("categories","")),
            "thumbnail": clean_text(r.get("thumbnail","")),
            "description": clean_text(r.get("description","")),
            "published_year": r.get("published_year") if not pd.isna(r.get("published_year")) else None,
            "average_rating": r.get("average_rating") if not pd.isna(r.get("average_rating")) else None,
            "num_pages": r.get("num_pages") if not pd.isna(r.get("num_pages")) else None,
            "ratings_count": r.get("ratings_count") if not pd.isna(r.get("ratings_count")) else None,
        }
        item["text_for_embedding"] = text_for_embedding(r)
        out.append(item)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(out)} books to {OUT}")

if __name__ == "__main__":
    main()
