"""
Anime Recommender System — Streamlit App
Two approaches: Content-Based Filtering & Collaborative Filtering
"""

import streamlit as st
import pandas as pd
import numpy as np

from data_loader import load_data, get_all_genres, get_all_types, get_anime_names
from content_based import ContentBasedRecommender
from collaborative import CollaborativeFilteringRecommender
from evaluation import compare_recommendations, genre_distribution

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Anime Recommender",
    page_icon="🎌",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif;
}

.main-header {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    padding: 2rem 2.5rem;
    border-radius: 16px;
    margin-bottom: 2rem;
    border: 1px solid rgba(229, 57, 53, 0.3);
}

.main-header h1 {
    color: #ffffff;
    font-size: 2.4rem;
    font-weight: 700;
    margin: 0;
    letter-spacing: -0.5px;
}

.main-header p {
    color: rgba(255,255,255,0.65);
    margin: 0.4rem 0 0 0;
    font-size: 1rem;
}

.accent { color: #e53935; }

.method-badge {
    display: inline-block;
    padding: 0.2rem 0.75rem;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
    margin-bottom: 0.75rem;
}

.badge-cb { background: rgba(229, 57, 53, 0.15); color: #e53935; border: 1px solid rgba(229,57,53,0.4); }
.badge-cf { background: rgba(33, 150, 243, 0.15); color: #2196F3; border: 1px solid rgba(33,150,243,0.4); }

.rec-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 0.9rem 1rem;
    margin-bottom: 0.6rem;
    transition: border-color 0.2s;
}

.rec-card:hover {
    border-color: rgba(229, 57, 53, 0.4);
}

.rec-title {
    font-weight: 600;
    font-size: 0.95rem;
    color: #ffffff;
}

.rec-meta {
    font-size: 0.78rem;
    color: rgba(255,255,255,0.5);
    margin-top: 0.2rem;
}

.stat-box {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
}

.stat-num {
    font-size: 1.8rem;
    font-weight: 700;
    color: #e53935;
}

.stat-label {
    font-size: 0.75rem;
    color: rgba(255,255,255,0.5);
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.section-title {
    font-size: 1.1rem;
    font-weight: 600;
    color: rgba(255,255,255,0.9);
    margin-bottom: 0.75rem;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid rgba(255,255,255,0.08);
}

.overlap-pill {
    background: rgba(76, 175, 80, 0.15);
    color: #4CAF50;
    border: 1px solid rgba(76,175,80,0.3);
    border-radius: 20px;
    padding: 0.2rem 0.75rem;
    font-size: 0.82rem;
    font-weight: 600;
    display: inline-block;
}

.stButton > button {
    background: linear-gradient(135deg, #e53935, #c62828) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Outfit', sans-serif !important;
    font-weight: 600 !important;
    padding: 0.5rem 1.5rem !important;
    transition: opacity 0.2s !important;
}

.stButton > button:hover {
    opacity: 0.85 !important;
}

/* Dark background for entire app */
.stApp {
    background-color: #0d0d1a;
}
</style>
""", unsafe_allow_html=True)


# ─── Load data & models (cached) ──────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_models():
    with st.status("Подготовка данных…", expanded=True) as status:
        st.write("📥 Загрузка датасета (при первом запуске скачивается с Hugging Face)…")
        anime_df, ratings_df = load_data()
        st.write("🧠 Построение Content-Based модели…")
        cb_model = ContentBasedRecommender(anime_df)
        st.write("🤝 Построение Collaborative Filtering модели…")
        cf_model = CollaborativeFilteringRecommender(anime_df, ratings_df)
        status.update(label="✅ Модели готовы!", state="complete", expanded=False)
    return anime_df, ratings_df, cb_model, cf_model


anime_df, ratings_df, cb_model, cf_model = load_models()
all_genres = ["All"] + get_all_genres(anime_df)
all_types = ["All"] + get_all_types(anime_df)
all_names = get_anime_names(anime_df)


# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🎌 Anime <span class="accent">Recommender</span></h1>
    <p>Content-Based & Collaborative Filtering · MyAnimeList Dataset</p>
</div>
""", unsafe_allow_html=True)


# ─── Sidebar: User Input ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📋 Ваши оценки")
    st.markdown(
        "<small style='color:rgba(255,255,255,0.5)'>Добавьте просмотренные аниме и ваши оценки</small>",
        unsafe_allow_html=True,
    )

    # Dynamic list of watched anime
    if "user_ratings" not in st.session_state:
        st.session_state.user_ratings = {}

    # Add anime
    selected_anime = st.selectbox(
        "Выберите аниме",
        ["— выберите —"] + all_names,
        key="anime_select",
    )
    user_score = st.slider("Ваша оценка", 1, 10, 8, key="score_slider")

    col_add, col_clear = st.columns(2)
    with col_add:
        if st.button("➕ Добавить", use_container_width=True):
            if selected_anime != "— выберите —":
                st.session_state.user_ratings[selected_anime] = user_score
                st.rerun()
    with col_clear:
        if st.button("🗑 Очистить", use_container_width=True):
            st.session_state.user_ratings = {}
            st.rerun()

    # Show current ratings
    if st.session_state.user_ratings:
        st.markdown("---")
        st.markdown("**Добавлено:**")
        for name, score in list(st.session_state.user_ratings.items()):
            c1, c2, c3 = st.columns([5, 2, 1])
            c1.markdown(f"<small>{name}</small>", unsafe_allow_html=True)
            c2.markdown(f"<b style='color:#e53935'>{score}/10</b>", unsafe_allow_html=True)
            if c3.button("×", key=f"del_{name}"):
                del st.session_state.user_ratings[name]
                st.rerun()

    st.markdown("---")

    # Filters
    st.markdown("### 🔍 Фильтры")
    genre_filter = st.selectbox("Жанр", all_genres)
    type_filter = st.selectbox("Тип", all_types)
    top_n = st.slider("Топ-N рекомендаций", 5, 20, 10)
    cf_method = st.radio(
        "CF метод",
        ["Item-Based", "User-Based"],
        help="Item-based: похожесть между аниме. User-based: похожесть между пользователями.",
    )

    st.markdown("---")
    run_btn = st.button("🚀 Получить рекомендации", use_container_width=True)


# ─── Dataset Stats ─────────────────────────────────────────────────────────────
stats = cf_model.get_stats()
col1, col2, col3, col4 = st.columns(4)
for col, (num, label) in zip(
    [col1, col2, col3, col4],
    [
        (len(anime_df), "Аниме"),
        (stats["n_users"], "Пользователей"),
        (stats["n_ratings"], "Оценок"),
        (f"{stats['sparsity']*100:.1f}%", "Разреженность"),
    ],
):
    col.markdown(
        f"""<div class="stat-box">
            <div class="stat-num">{num}</div>
            <div class="stat-label">{label}</div>
        </div>""",
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)


# ─── Recommendations ──────────────────────────────────────────────────────────
if run_btn or st.session_state.get("ran_once"):
    st.session_state["ran_once"] = True
    user_ratings = st.session_state.user_ratings

    if not user_ratings:
        st.warning("⚠️ Добавьте хотя бы одно аниме с оценкой в боковой панели.")
    else:
        # Run models
        with st.spinner("Вычисление рекомендаций..."):
            cb_recs = cb_model.recommend(
                user_ratings, top_n=top_n,
                genre_filter=genre_filter, type_filter=type_filter
            )
            if cf_method == "Item-Based":
                cf_recs = cf_model.recommend_item_based(
                    user_ratings, top_n=top_n,
                    genre_filter=genre_filter, type_filter=type_filter
                )
            else:
                cf_recs = cf_model.recommend_user_based(
                    user_ratings, top_n=top_n,
                    genre_filter=genre_filter, type_filter=type_filter
                )

        # ── Two columns: CB left, CF right ──
        left, right = st.columns(2)

        def render_recs(container, recs: pd.DataFrame, label: str, badge_class: str, color: str):
            with container:
                st.markdown(
                    f'<span class="method-badge {badge_class}">{label}</span>',
                    unsafe_allow_html=True,
                )
                if recs.empty:
                    st.info("Нет результатов — попробуйте другие фильтры.")
                    return
                for i, row in recs.iterrows():
                    score_pct = min(row["score"] * 100, 100)
                    st.markdown(
                        f"""<div class="rec-card">
                            <div style="display:flex;justify-content:space-between;align-items:start">
                                <div class="rec-title">{row['name']}</div>
                                <div style="font-size:0.8rem;font-weight:700;color:{color}">{row['score']:.3f}</div>
                            </div>
                            <div class="rec-meta">
                                🏷 {row['genre'][:55]}{'…' if len(row['genre'])>55 else ''} &nbsp;|&nbsp;
                                📺 {row['type']} &nbsp;|&nbsp;
                                ⭐ MAL {row['mal_rating']}
                            </div>
                            <div style="height:3px;background:rgba(255,255,255,0.05);border-radius:2px;margin-top:0.5rem">
                                <div style="height:3px;width:{score_pct:.1f}%;background:{color};border-radius:2px"></div>
                            </div>
                        </div>""",
                        unsafe_allow_html=True,
                    )

        render_recs(left, cb_recs, "Content-Based Filtering", "badge-cb", "#e53935")
        render_recs(right, cf_recs, f"Collaborative ({cf_method})", "badge-cf", "#2196F3")

        # ── Comparison section ──
        st.markdown("---")
        st.markdown('<div class="section-title">📊 Сравнение подходов</div>', unsafe_allow_html=True)

        if not cb_recs.empty and not cf_recs.empty:
            comparison = compare_recommendations(cb_recs, cf_recs)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Пересечение", f"{comparison['overlap']} тайтлов")
            m2.metric("Overlap ratio", f"{comparison['overlap_ratio']:.0%}")
            m3.metric("Avg MAL (CB)", comparison["avg_mal_rating_cb"])
            m4.metric("Avg MAL (CF)", comparison["avg_mal_rating_cf"])

            if comparison["shared"]:
                st.markdown("**Совпадают в обоих:** " + ", ".join(
                    [f"`{t}`" for t in comparison["shared"]]
                ))

            # Genre distribution comparison
            cb_genres = genre_distribution(cb_recs).head(8)
            cf_genres = genre_distribution(cf_recs).head(8)

            if not cb_genres.empty or not cf_genres.empty:
                gc1, gc2 = st.columns(2)
                with gc1:
                    st.markdown("**Топ жанры (CB):**")
                    st.bar_chart(cb_genres)
                with gc2:
                    st.markdown(f"**Топ жанры (CF — {cf_method}):**")
                    st.bar_chart(cf_genres)

        # ── Model explanation ──
        with st.expander("ℹ️ Как работают модели?"):
            st.markdown("""
**Content-Based Filtering**
- Представляет каждое аниме вектором признаков (жанры + тип через one-hot encoding)
- Вычисляет косинусное сходство между всеми аниме
- Для ваших оценок строит взвешенный профиль (высокая оценка → больший вес)
- Рекомендует аниме, максимально похожие по характеристикам

**Collaborative Filtering (Item-Based)**
- Строит матрицу пользователь × аниме из реальных оценок
- Находит аниме, которые пользователи оценивают похоже (item-item similarity)
- Рекомендует на основе коллективного поведения аудитории

**Collaborative Filtering (User-Based)**
- Создаёт виртуальный профиль на основе ваших оценок
- Находит похожих пользователей по косинусному сходству
- Агрегирует оценки топ-N соседей для несмотренных тайтлов

**Ключевое различие:**
| | Content-Based | Collaborative |
|---|---|---|
| Основа | Характеристики аниме | Паттерны пользователей |
| Cold start | ✅ Работает с новыми тайтлами | ❌ Нужна история рейтингов |
| Неожиданные находки | ❌ Похожее → ограниченное | ✅ Может открыть новые жанры |
            """)

else:
    # Welcome state
    st.markdown("""
    <div style="text-align:center; padding: 3rem; color: rgba(255,255,255,0.4);">
        <div style="font-size:3rem">🎌</div>
        <div style="font-size:1.1rem; margin-top:1rem">
            Добавьте аниме с оценками в боковой панели<br>
            и нажмите <b style="color:#e53935">Получить рекомендации</b>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Quick start suggestions
    st.markdown('<div class="section-title">💡 Быстрый старт — популярные тайтлы</div>', unsafe_allow_html=True)
    suggestions = [
        ("Frieren: Beyond Journey's End", 10),
        ("Fullmetal Alchemist: Brotherhood", 9),
        ("Hunter x Hunter (2011)", 9),
        ("Steins;Gate", 10),
        ("Monster", 9),
    ]
    cols = st.columns(len(suggestions))
    for col, (name, score) in zip(cols, suggestions):
        with col:
            if st.button(f"➕ {name.split(':')[0]}", use_container_width=True, key=f"quick_{name}"):
                st.session_state.user_ratings[name] = score
                st.rerun()
