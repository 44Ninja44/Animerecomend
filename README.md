# 🎌 Anime Recommender

A Streamlit app that recommends anime using two approaches:
- **Content-Based Filtering** — cosine similarity on genre/type features
- **Collaborative Filtering** — item-based and user-based (cosine on user-item matrix)

**Live demo:** *(paste your Streamlit Cloud URL here)*

---

## ✨ Features

- Rate anime you've watched (1–10) and get personalised recommendations
- Filter results by genre and type
- Side-by-side comparison of CB vs CF with overlap metrics and genre charts
- Dark anime-themed UI

---

## 🚀 Deploy to Streamlit Cloud (no data files needed)

The data files are **not committed to this repo** — they are downloaded automatically from [Hugging Face](https://huggingface.co/datasets/marlesson/anime-recommendation-database-2020) on first run and cached locally in `.data_cache/`.

1. Fork / push this repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**.
3. Select your repo, branch `main`, and main file `app.py`.
4. Click **Deploy** — done! The first cold start takes ~1–2 min to download and preprocess the data.

---

## 💻 Run locally

```bash
git clone https://github.com/YOUR_USERNAME/anime-recommender
cd anime-recommender
pip install -r requirements.txt
streamlit run app.py
```

The first run downloads `anime.csv` (~900 KB) from Hugging Face and samples 200 000 rows from `ratings.csv` (~1.2 GB download, processed once and cached as a small file). Subsequent runs start instantly from cache.

---

## 📂 Project structure

```
anime_recommender/
├── app.py                  # Streamlit UI
├── requirements.txt
├── .streamlit/
│   └── config.toml         # Dark theme
├── models/
│   ├── content_based.py    # CB recommender
│   └── collaborative.py    # CF recommender (item- & user-based)
└── utils/
    ├── data_loader.py      # HuggingFace download + caching
    └── evaluation.py       # Overlap / genre distribution metrics
```

---

## 📊 Dataset

[MyAnimeList Recommendation Database 2020](https://huggingface.co/datasets/marlesson/anime-recommendation-database-2020) — ~12 000 anime titles and 57 M ratings.

We sample 200 000 ratings (from users with ≥ 5 ratings each) so the app stays responsive on free-tier cloud instances.
