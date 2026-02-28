import pandas as pd
import json
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# DOWNLOAD RESOURCE NLTK
nltk.download("stopwords", quiet=True)

# INISIALISASI
stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))

# FUNGSI: PREPROCESS TEKS OVERVIEW
def clean_text(text: str) -> str:
    """
    Membersihkan teks sinopsis (overview):
    - case folding
    - hapus simbol & angka
    - tokenisasi
    - stopword removal
    - stemming
    """
    if not isinstance(text, str):
        return ""

    # Case folding
    text = text.lower()

    # Punctuation Removal
    text = re.sub(r"[^a-z\s]", " ", text)

    # Tokenisasi
    tokens = text.split()

    # Stopword removal + stemming
    tokens = [
        stemmer.stem(word)
        for word in tokens
        if word not in stop_words
    ]

    return " ".join(tokens)


# FUNGSI: PARSING METADATA (GENRES / KEYWORDS)
def parse_metadata(text: str) -> list:
    """
    Mengubah metadata JSON (genres / keywords)
    menjadi list token tanpa spasi
    """
    if not isinstance(text, str):
        return []

    try:
        data = json.loads(text)
        return [
            item["name"].replace(" ", "").lower()
            for item in data
            if "name" in item
        ]
    except Exception:
        return []

# FUNGSI UTAMA PREPROCESSING
def preprocess_dataset(csv_path: str) -> pd.DataFrame:
    """
    Melakukan pra-pemrosesan dataset film:
    - overview_clean (baseline)
    - tags (hybrid)
    """
    df = pd.read_csv(csv_path)

    # Pastikan kolom tidak NaN
    df["overview"] = df["overview"].fillna("")
    df["genres"] = df["genres"].fillna("")
    df["keywords"] = df["keywords"].fillna("")

    # Preprocess overview
    df["overview_clean"] = df["overview"].apply(clean_text)

    # Parse metadata
    df["genres_parsed"] = df["genres"].apply(parse_metadata)
    df["keywords_parsed"] = df["keywords"].apply(parse_metadata)

    # Penggabungan Fitur hybrid
    df["tags"] = (
        df["overview_clean"] + " " +
        df["genres_parsed"].apply(lambda x: " ".join(x)) + " " +
        df["keywords_parsed"].apply(lambda x: " ".join(x))
    )

    return df
