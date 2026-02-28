# tmdb_api.py
import requests
import streamlit as st

# Ambil dari secrets, bukan hardcode
TMDB_API_KEY = st.secrets["TMDB_API_KEY"]
BASE_URL = "https://api.themoviedb.org/3"
IMAGE_BASE = "https://image.tmdb.org/t/p/w500"

def get_movie_poster(title):
    url = f"{BASE_URL}/search/movie"
    params = {
        "api_key": TMDB_API_KEY,
        "query": title
    }

    response = requests.get(url, params=params).json()

    if response.get("results"):
        poster_path = response["results"][0].get("poster_path")
        if poster_path:
            return IMAGE_BASE + poster_path

    return None
