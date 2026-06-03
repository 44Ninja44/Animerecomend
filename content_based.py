"""
Content-Based Filtering for Anime Recommendations.

Approach:
- Represent each anime as a feature vector (genres + type)
- Compute cosine similarity between all anime
- For a user's rated anime, find most similar unseen titles
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity


class ContentBasedRecommender:
    def __init__(self, anime_df: pd.DataFrame):
        self.anime_df = anime_df.copy().reset_index(drop=True)
        self.similarity_matrix = None
        self.feature_matrix = None
        self._build()

    def _build(self):
        """Build the feature matrix and similarity matrix."""
        # --- Genre features ---
        mlb = MultiLabelBinarizer()
        genres = self.anime_df["genre"].fillna("").str.split(",").apply(
            lambda x: [g.strip() for g in x]
        )
        genre_matrix = mlb.fit_transform(genres)
        self.genre_labels = mlb.classes_

        # --- Type one-hot encoding ---
        type_dummies = pd.get_dummies(
            self.anime_df["type"].fillna("Unknown"), prefix="type"
        ).values

        # --- Combine features ---
        self.feature_matrix = np.hstack([genre_matrix, type_dummies])

        # --- Cosine similarity ---
        self.similarity_matrix = cosine_similarity(self.feature_matrix)

    def recommend(
        self,
        user_ratings: dict,        # {anime_name: rating (1-10)}
        top_n: int = 10,
        genre_filter: str = None,  # e.g. "Action"
        type_filter: str = None,   # e.g. "Movie"
    ) -> pd.DataFrame:
        """
        Generate content-based recommendations.

        user_ratings: dict of {anime_name: score}
        Returns a DataFrame with recommended anime sorted by score.
        """
        name_to_idx = {name: i for i, name in enumerate(self.anime_df["name"])}

        # Find indices of user-rated anime
        rated_indices = []
        for name, rating in user_ratings.items():
            if name in name_to_idx:
                rated_indices.append((name_to_idx[name], rating))

        if not rated_indices:
            return pd.DataFrame()

        # Weighted average similarity across rated titles
        # Higher-rated anime contribute more
        sim_scores = np.zeros(len(self.anime_df))
        weight_total = 0.0

        for idx, rating in rated_indices:
            weight = rating / 10.0  # normalize to [0, 1]
            sim_scores += weight * self.similarity_matrix[idx]
            weight_total += weight

        if weight_total > 0:
            sim_scores /= weight_total

        # Remove already-rated anime
        rated_set = {name_to_idx[n] for n, _ in rated_indices if n in name_to_idx}
        for i in rated_set:
            sim_scores[i] = -1

        # Build result
        result = self.anime_df.copy()
        result["cb_score"] = sim_scores

        # Filters
        if genre_filter and genre_filter != "All":
            result = result[
                result["genre"].str.contains(genre_filter, na=False, case=False)
            ]
        if type_filter and type_filter != "All":
            result = result[result["type"] == type_filter]

        result = result[result["cb_score"] >= 0]
        result = result.sort_values("cb_score", ascending=False).head(top_n)

        return result[["name", "genre", "type", "rating", "cb_score"]].rename(
            columns={"cb_score": "score", "rating": "mal_rating"}
        )

    def get_similar(self, anime_name: str, top_n: int = 5) -> pd.DataFrame:
        """Return top_n most similar anime to a given title."""
        name_to_idx = {name: i for i, name in enumerate(self.anime_df["name"])}
        if anime_name not in name_to_idx:
            return pd.DataFrame()

        idx = name_to_idx[anime_name]
        scores = self.similarity_matrix[idx].copy()
        scores[idx] = -1  # exclude self

        result = self.anime_df.copy()
        result["similarity"] = scores
        result = result.sort_values("similarity", ascending=False).head(top_n)
        return result[["name", "genre", "type", "rating", "similarity"]]
