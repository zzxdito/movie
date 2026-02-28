import sys, os, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Setup Path & Import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from preprocessing import preprocess_dataset

# 1. Load Data
csv_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'tmdb_5000_movies.csv')
print("Memuat dataset dari:", csv_path)
df = preprocess_dataset(csv_path).reset_index(drop=True)

# 2. Build TF-IDF (Nama variabel tetap)
tfidf_baseline = TfidfVectorizer()
tfidf_matrix_baseline = tfidf_baseline.fit_transform(df["overview_clean"])

tfidf_hybrid = TfidfVectorizer()
tfidf_matrix_hybrid = tfidf_hybrid.fit_transform(df["tags"])

# 3. Pilih Film Target
movie_title = "Batman"
if movie_title not in df["title"].values: raise ValueError("Judul film tidak ditemukan.")
movie_idx = df[df["title"] == movie_title].index[0]

# 4. Fungsi Helper (Dibersihkan)
def get_top_tfidf(vectorizer, matrix, idx, top_n=10):
    tfidf_df = pd.DataFrame({
        "term": vectorizer.get_feature_names_out(),
        "tfidf": matrix[idx].toarray()[0]
    })
    # Filter > 0, Sort, dan ambil Top N langsung
    return tfidf_df[tfidf_df["tfidf"] > 0].sort_values("tfidf", ascending=False).head(top_n)

def get_top_cosine(cosine_matrix, idx, top_n=10):
    scores = list(enumerate(cosine_matrix[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    
    results = []
    for i, score in scores:
        if i == idx: continue
        results.append((df.iloc[i]["title"], round(float(score), 4)))
        if len(results) == top_n: break
    return results

# 5. Print Hasil TF-IDF (Sesuai aslinya)
print("\n=== TOP TF-IDF BASELINE (overview_clean) ===")
print(get_top_tfidf(tfidf_baseline, tfidf_matrix_baseline, movie_idx))

print("\n=== TOP TF-IDF HYBRID (tags) ===")
print(get_top_tfidf(tfidf_hybrid, tfidf_matrix_hybrid, movie_idx))

# 6. Hitung & Print Cosine Similarity
cosine_baseline = cosine_similarity(tfidf_matrix_baseline)
cosine_hybrid = cosine_similarity(tfidf_matrix_hybrid)

print("\n=== TOP COSINE SIMILARITY BASELINE ===")
for title, score in get_top_cosine(cosine_baseline, movie_idx):
    print(title, score)

print("\n=== TOP COSINE SIMILARITY HYBRID ===")
for title, score in get_top_cosine(cosine_hybrid, movie_idx):
    print(title, score)

print("\n=== SELESAI ===")