"""
Data loading utilities — fetches data from Hugging Face Hub if not cached locally.

Dataset: marlesson/anime-recommendation-database-2020
  - anime.csv   ~12 000 titles, ~900 KB
  - ratings.csv ~57 M rows,     ~1.2 GB (we sample a subset for performance)
"""

import os
import io
import hashlib
import pandas as pd
import numpy as np
import streamlit as st

# ── paths ──────────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(_HERE, ".data_cache")

# Hugging Face raw URLs (CSV files in the dataset repository)
_HF_BASE = "https://huggingface.co/datasets/marlesson/anime-recommendation-database-2020/resolve/main/data"
ANIME_URL   = f"{_HF_BASE}/anime.csv"
RATINGS_URL = f"{_HF_BASE}/ratings.csv"

# How many rating rows to keep (enough for good CF, small enough to be fast)
RATINGS_SAMPLE = 200_000


# ── helpers ────────────────────────────────────────────────────────────────────
def _cache_path(name: str) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, name)


def _download(url: str, dest: str, label: str) -> None:
    """Download url → dest with a Streamlit progress bar."""
    import urllib.request

    progress = st.progress(0, text=f"Загрузка {label}…")
    try:
        with urllib.request.urlopen(url) as r:
            total = int(r.headers.get("Content-Length", 0))
            chunk = 1 << 16  # 64 KB
            downloaded = 0
            buf = io.BytesIO()
            while True:
                data = r.read(chunk)
                if not data:
                    break
                buf.write(data)
                downloaded += len(data)
                if total:
                    progress.progress(
                        min(downloaded / total, 1.0),
                        text=f"Загрузка {label}… {downloaded // 1024:,} KB",
                    )
        with open(dest, "wb") as f:
            f.write(buf.getvalue())
    finally:
        progress.empty()


# ── public API ─────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return (anime_df, ratings_df).

    On first run the CSVs are downloaded from Hugging Face and cached in
    .data_cache/; subsequent runs load from disk instantly.
    """
    anime_cache   = _cache_path("anime.csv")
    ratings_cache = _cache_path(f"ratings_{RATINGS_SAMPLE}.csv")

    # ── anime ──────────────────────────────────────────────────────────────────
    if not os.path.exists(anime_cache):
        _download(ANIME_URL, anime_cache, "anime.csv")

    anime_df = pd.read_csv(anime_cache)
    anime_df = anime_df.dropna(subset=["name", "genre"])
    anime_df["rating"] = pd.to_numeric(anime_df["rating"], errors="coerce").fillna(0)
    anime_df["genre"]  = anime_df["genre"].str.strip()
    anime_df["type"]   = anime_df["type"].fillna("Unknown")

    # ── ratings ────────────────────────────────────────────────────────────────
    if not os.path.exists(ratings_cache):
        raw_cache = _cache_path("ratings_raw.csv")
        if not os.path.exists(raw_cache):
            _download(RATINGS_URL, raw_cache, "ratings.csv")

        # Read in chunks and keep only valid ratings, then sample
        st.info("Обработка ratings.csv (один раз)…")
        chunks = []
        for chunk in pd.read_csv(raw_cache, chunksize=500_000):
            chunk = chunk[chunk["rating"].between(1, 10)]
            chunks.append(chunk)
        full = pd.concat(chunks, ignore_index=True)

        # Stratified sample: keep users with ≥5 ratings for better CF quality
        user_counts = full["user_id"].value_counts()
        active_users = user_counts[user_counts >= 5].index
        full = full[full["user_id"].isin(active_users)]

        sampled = full.sample(n=min(RATINGS_SAMPLE, len(full)), random_state=42)
        sampled.to_csv(ratings_cache, index=False)

        # Clean up raw download to save space
        try:
            os.remove(raw_cache)
        except OSError:
            pass

    ratings_df = pd.read_csv(ratings_cache)

    return anime_df, ratings_df


def get_all_genres(anime_df: pd.DataFrame) -> list[str]:
    genres: set[str] = set()
    for s in anime_df["genre"].dropna():
        for g in s.split(","):
            genres.add(g.strip())
    return sorted(genres)


def get_all_types(anime_df: pd.DataFrame) -> list[str]:
    return sorted(anime_df["type"].dropna().unique().tolist())


def get_anime_names(anime_df: pd.DataFrame) -> list[str]:
    return sorted(anime_df["name"].dropna().unique().tolist())
