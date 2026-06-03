"""
Collaborative Filtering for Anime Recommendations.

Implements two approaches:
1. Item-Based CF: find anime similar to what the user liked based on other users' patterns
2. User-Based CF: find similar users, use their ratings to predict scores

Both use cosine similarity on the user-item matrix.
"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class CollaborativeFilteringRecommender:
    def __init__(self, anime_df: pd.DataFrame, ratings_df: pd.DataFrame):
        self.anime_df = anime_df.copy()
        self.ratings_df = ratings_df.copy()
        self.user_item_matrix = None
        self.item_similarity = None
        self.user_similarity = None
        self._build()

    def _build(self):
        """Build user-item matrix and precompute similarity matrices."""
        # Pivot to user-item matrix (rows=users, cols=anime)
        self.user_item_matrix = self.ratings_df.pivot_table(
            index="user_id", columns="anime_id", values="rating"
        ).fillna(0)

        # Item-item cosine similarity (transpose so items are rows)
        item_matrix = self.user_item_matrix.T.values
        self.item_similarity = cosine_similarity(item_matrix)
        self.item_sim_df = pd.DataFrame(
            self.item_similarity,
            index=self.user_item_matrix.columns,
            columns=self.user_item_matrix.columns,
        )

        # User-user cosine similarity
        user_matrix = self.user_item_matrix.values
        self.user_similarity = cosine_similarity(user_matrix)
        self.user_sim_df = pd.DataFrame(
            self.user_similarity,
            index=self.user_item_matrix.index,
            columns=self.user_item_matrix.index,
        )

    def recommend_item_based(
        self,
        user_ratings: dict,        # {anime_name: rating}
        top_n: int = 10,
        genre_filter: str = None,
        type_filter: str = None,
    ) -> pd.DataFrame:
        """
        Item-based collaborative filtering.
        For each rated anime, find similar items based on user-rating patterns.
        """
        name_to_id = dict(zip(self.anime_df["name"], self.anime_df["anime_id"]))
        id_to_name = dict(zip(self.anime_df["anime_id"], self.anime_df["name"]))

        rated_ids = {}
        for name, rating in user_ratings.items():
            if name in name_to_id:
                aid = name_to_id[name]
                if aid in self.item_sim_df.index:
                    rated_ids[aid] = rating

        if not rated_ids:
            return pd.DataFrame()

        # Weighted sum of similarities from rated items
        all_items = self.item_sim_df.columns.tolist()
        score_dict = {}
        weight_total = 0.0

        for aid, rating in rated_ids.items():
            weight = rating / 10.0
            sim_row = self.item_sim_df.loc[aid]
            for item_id, sim in sim_row.items():
                if item_id not in rated_ids:
                    score_dict[item_id] = score_dict.get(item_id, 0) + weight * sim
            weight_total += weight

        if weight_total > 0:
            score_dict = {k: v / weight_total for k, v in score_dict.items()}

        if not score_dict:
            return pd.DataFrame()

        scores_df = pd.DataFrame(
            list(score_dict.items()), columns=["anime_id", "cf_score"]
        )
        result = scores_df.merge(self.anime_df, on="anime_id")

        # Filters
        if genre_filter and genre_filter != "All":
            result = result[
                result["genre"].str.contains(genre_filter, na=False, case=False)
            ]
        if type_filter and type_filter != "All":
            result = result[result["type"] == type_filter]

        result = result.sort_values("cf_score", ascending=False).head(top_n)
        return result[["name", "genre", "type", "rating", "cf_score"]].rename(
            columns={"cf_score": "score", "rating": "mal_rating"}
        )

    def recommend_user_based(
        self,
        user_ratings: dict,
        top_n: int = 10,
        genre_filter: str = None,
        type_filter: str = None,
        n_neighbors: int = 20,
    ) -> pd.DataFrame:
        """
        User-based collaborative filtering.
        Build a virtual user profile, find similar real users, aggregate their ratings.
        """
        name_to_id = dict(zip(self.anime_df["name"], self.anime_df["anime_id"]))

        # Build virtual user rating vector aligned with matrix columns
        virtual_user = pd.Series(0.0, index=self.user_item_matrix.columns)
        rated_ids = set()
        for name, rating in user_ratings.items():
            if name in name_to_id:
                aid = name_to_id[name]
                if aid in virtual_user.index:
                    virtual_user[aid] = float(rating)
                    rated_ids.add(aid)

        if not rated_ids:
            return pd.DataFrame()

        # Cosine similarity between virtual user and all real users
        virtual_vec = virtual_user.values.reshape(1, -1)
        real_matrix = self.user_item_matrix.values
        sims = cosine_similarity(virtual_vec, real_matrix)[0]

        sim_df = pd.Series(sims, index=self.user_item_matrix.index).sort_values(
            ascending=False
        ).head(n_neighbors)

        # Weighted average of neighbor ratings for unseen items
        score_dict = {}
        weight_total = sim_df.sum()

        for user_id, sim in sim_df.items():
            if sim <= 0:
                continue
            user_row = self.user_item_matrix.loc[user_id]
            for item_id, r in user_row.items():
                if r > 0 and item_id not in rated_ids:
                    score_dict[item_id] = score_dict.get(item_id, 0) + sim * r

        if weight_total > 0:
            score_dict = {k: v / weight_total for k, v in score_dict.items()}

        if not score_dict:
            return pd.DataFrame()

        scores_df = pd.DataFrame(
            list(score_dict.items()), columns=["anime_id", "cf_score"]
        )
        result = scores_df.merge(self.anime_df, on="anime_id")

        if genre_filter and genre_filter != "All":
            result = result[
                result["genre"].str.contains(genre_filter, na=False, case=False)
            ]
        if type_filter and type_filter != "All":
            result = result[result["type"] == type_filter]

        result = result.sort_values("cf_score", ascending=False).head(top_n)
        return result[["name", "genre", "type", "rating", "cf_score"]].rename(
            columns={"cf_score": "score", "rating": "mal_rating"}
        )

    def get_stats(self) -> dict:
        """Return dataset statistics."""
        return {
            "n_users": len(self.user_item_matrix.index),
            "n_anime": len(self.user_item_matrix.columns),
            "n_ratings": int((self.user_item_matrix > 0).sum().sum()),
            "sparsity": round(
                1 - (self.user_item_matrix > 0).sum().sum()
                / (self.user_item_matrix.shape[0] * self.user_item_matrix.shape[1]),
                4,
            ),
        }
