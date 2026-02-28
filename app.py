import streamlit as st
import pandas as pd
from preprocessing import preprocess_dataset
from model_baseline import recommend_baseline, build_tfidf_matrix as build_baseline_matrix
from model_hybrid import recommend_hybrid, build_tfidf_matrix as build_hybrid_matrix
from tmdb_api import get_movie_poster

st.set_page_config(page_title="Rekomendasi Film", layout="wide")
TOP_N = 10 

st.markdown("""
<style>
.section-title{font-size:28px;font-weight:700;margin-bottom:6px}
.section-subtitle{font-size:14px;color:#b0b0b0;margin-bottom:20px}
.movie-card{background:linear-gradient(180deg,rgba(24,26,35,.95),rgba(16,18,25,.95));border-radius:12px;padding:12px;text-align:center;box-shadow:0 4px 10px rgba(0,0,0,.5);margin-bottom:20px;color:white;height:380px;display:flex;flex-direction:column}
.movie-card img{width:100%;height:180px;object-fit:cover;border-radius:8px;margin-bottom:8px;background:#0e1117}
.movie-title{font-size:13px;font-weight:700;color:#f1f1f1;margin-bottom:2px;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden;height:34px;line-height:1.2;align-items:center;justify-content:center}
.movie-meta{font-size:10px;color:#4CAF50;font-weight:600;margin-bottom:6px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.movie-overview{font-size:10px;color:#b0b0b0;line-height:1.3;display:-webkit-box;-webkit-line-clamp:3;-webkit-box-orient:vertical;overflow:hidden;text-align:justify;padding:0 4px}
.movie-score{font-size:11px;color:#FFD700;font-weight:bold;margin-top:auto;padding-top:6px;border-top:1px solid rgba(255,255,255,.1)}
</style>
""", unsafe_allow_html=True)

# LOAD DATA & BUILD MATRIX
@st.cache_resource
def init_system():
    df = preprocess_dataset("data/tmdb_5000_movies.csv")
    matrix_b = build_baseline_matrix(df)
    matrix_h = build_hybrid_matrix(df)
    return df, matrix_b, matrix_h

try:
    with st.spinner('Sedang memuat model & dataset... Harap tunggu sebentar...'):
        df, tfidf_matrix_baseline, tfidf_matrix_hybrid = init_system()
except Exception as e:
    st.error(f"Terjadi kesalahan saat memuat data: {e}")
    st.stop()

# SIDEBAR NAVIGASI
menu = st.sidebar.radio("Navigasi", ["Home", "Rekomendasi", "Data Film"])

# HOME
if menu == "Home":
    st.markdown("<div class='section-title'>🎬 Rekomendasi Film</div>", unsafe_allow_html=True)
    st.write("""
        Sistem rekomendasi film berbasis **Content-Based Filtering**
        yang menggunakan metode **TF-IDF** dan **Cosine Similarity**.
        Sistem menyediakan dua pendekatan:
        - **Model Baseline** (sinopsis film)
        - **Model Hybrid** (sinopsis, genre, dan keywords)
        """)

# REKOMENDASI
elif menu == "Rekomendasi":
    st.markdown("<div class='section-title'>📌 Hasil Rekomendasi</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-subtitle'>Menampilkan 10 film rekomendasi teratas</div>", unsafe_allow_html=True)

    col_filter, col_space = st.columns([1.2, 3.8])
    with col_filter:
        model_type = st.selectbox("Pilih Model", ["— Pilih Model —", "Baseline", "Hybrid"])
        title_input = st.selectbox("Pilih Judul Film", ["— Pilih Judul Film —"] + sorted(df["title"].unique()))
        search_btn = st.button("Cari Rekomendasi", disabled=(model_type.startswith("—") or title_input.startswith("—")))

    if search_btn:
        movie_data = df[df["title"] == title_input].iloc[0]
        overview_text = movie_data["overview"]
        genres_text = ", ".join([g.title() for g in movie_data["genres_parsed"]])
        keywords_text = ", ".join(movie_data["keywords_parsed"][:10])

        st.markdown("---")
        col_header_poster, col_header_info = st.columns([1, 4]) 

        with col_header_poster:
            input_poster_url = get_movie_poster(title_input)
            if input_poster_url:
                st.image(input_poster_url, width=150)
            else:
                st.write("No Image")

        with col_header_info:
            st.header(title_input) 
            st.markdown(f":green[**{genres_text}**]")
            st.write(overview_text)
            if keywords_text:
                st.caption(f"Keywords: {keywords_text}")

        st.markdown("---")

        if model_type == "Baseline":
            results = recommend_baseline(title_input, df, tfidf_matrix_baseline, TOP_N)
        else:
            results = recommend_hybrid(title_input, df, tfidf_matrix_hybrid, TOP_N)

        cols_per_row = 5
        rows = [results[i:i + cols_per_row] for i in range(0, len(results), cols_per_row)]

        for row in rows:
            cols = st.columns(cols_per_row, gap="large")
            for col, rec in zip(cols, row):
                with col:
                    # Ambil Detail Film
                    rec_row = df[df["title"] == rec["title"]].iloc[0]
                    rec_genres = ", ".join(rec_row["genres_parsed"][:2]) 
                    rec_overview = rec_row["overview"]
                    
                    poster_url = get_movie_poster(rec["title"]) or ""

                    # FIX: Rapat kiri
                    html_card = f"""
<div class="movie-card">
    <img src="{poster_url}">
    <div class="movie-title">{rec['title']}</div>
    <div class="movie-meta">Genre: {rec_genres}</div>
    <div class="movie-overview">{rec_overview}</div>
    <div class="movie-score">Similarity: {rec['score']}</div>
</div>
"""
                    st.markdown(html_card, unsafe_allow_html=True)

# DATA FILM
elif menu == "Data Film":
    st.markdown("<div class='section-title'>📂 Data Film</div>", unsafe_allow_html=True)
    df_sorted = df.sort_values(by="title").reset_index(drop=True)
    st.dataframe(df_sorted[["title", "genres", "overview"]].head(1000))