import sys, os, pandas as pd

# Setup Path & Import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from preprocessing import preprocess_dataset
from model_baseline import recommend_baseline, build_tfidf_matrix as build_baseline_matrix
from model_hybrid import recommend_hybrid, build_tfidf_matrix as build_hybrid_matrix

def evaluate_model(model_type: str, test_titles: list, top_n: int = 2000):
    # 1. Load Data
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'tmdb_5000_movies.csv')
    df = preprocess_dataset(csv_path)

    # 2. Build Matrix (Langsung pilih fungsi tanpa print)
    if model_type == "baseline":
        tfidf_matrix = build_baseline_matrix(df)
        rec_func = recommend_baseline
    elif model_type == "hybrid":
        tfidf_matrix = build_hybrid_matrix(df)
        rec_func = recommend_hybrid
    else:
        raise ValueError("Model tidak valid.")

    results = []

    # 3. Iterasi Pengujian
    for title in test_titles:
        if title not in df["title"].values: continue

        # Ambil Genre Ground Truth
        input_genres = set(df[df["title"] == title].iloc[0]["genres_parsed"])

        # Ambil Rekomendasi
        recommendations = rec_func(title, df, tfidf_matrix, top_n)

        # Hitung Jumlah Relevan (Intersection > 0)
        relevant_count = 0
        for rec in recommendations:
            match = df[df["title"] == rec["title"]]
            if not match.empty:
                rec_genres = set(match.iloc[0]["genres_parsed"])
                if len(input_genres.intersection(rec_genres)) > 0:
                    relevant_count += 1

        # Hitung Total Relevan di Database
        total_relevant = sum(1 for g in df["genres_parsed"] if len(input_genres.intersection(set(g))) > 0)
        # Hitung Metrik
        precision = relevant_count / top_n if top_n > 0 else 0
        recall = relevant_count / total_relevant if total_relevant > 0 else 0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0

        results.append({
            "Title": title, "Precision": precision, "Recall": recall, "F1-Score": f1
        })

    # 4. Format Output
    pd.options.display.float_format = '{:.4f}'.format
    return pd.DataFrame(results)