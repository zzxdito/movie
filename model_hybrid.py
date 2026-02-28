import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# TF-IDF MATRIX (HYBRID)
def build_tfidf_matrix(df: pd.DataFrame):
    # Inisialisasi TF-IDF Vectorizer
    vectorizer = TfidfVectorizer()
    # Fitur 'tags' (overview + genre + keyword)
    tfidf_matrix = vectorizer.fit_transform(df["tags"])
    return tfidf_matrix

# REKOMENDASI FILM (HYBRID)
def recommend_hybrid(title: str, df: pd.DataFrame, tfidf_matrix, top_n: int = 10):
    """
    Menghasilkan rekomendasi film hybrid.
    Data (df) dan Matrix (tfidf_matrix) dikirim dari app.py.
    """
    if title not in df["title"].values:
        raise ValueError("Judul film tidak ditemukan dalam dataset.")

    # Ambil index film input
    idx = df[df["title"] == title].index[0]

    # Hitung cosine similarity
    cosine_scores = cosine_similarity(
        tfidf_matrix[idx],
        tfidf_matrix
    ).flatten()

    # Urutkan berdasarkan similarity
    similarity_indices = cosine_scores.argsort()[::-1]

    results = []
    for i in similarity_indices:
        if i == idx:
            continue

        results.append({
            "title": df.iloc[i]["title"],
            "score": round(float(cosine_scores[i]), 4)
        })

        if len(results) == top_n:
            break

    return results