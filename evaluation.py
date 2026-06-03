"""
Utilities for comparing content-based and collaborative filtering models.
"""

import pandas as pd
import numpy as np


def compare_recommendations(cb_results: pd.DataFrame, cf_results: pd.DataFrame) -> dict:
    """
    Compare two recommendation lists.
    
    Returns:
        - overlap: number of shared titles
        - overlap_ratio: overlap / min(len, len)
        - avg_mal_rating_cb: average MAL rating of CB recommendations
        - avg_mal_rating_cf: average MAL rating of CF recommendations
        - unique_to_cb: titles only in CB
        - unique_to_cf: titles only in CF
    """
    if cb_results.empty or cf_results.empty:
        return {}

    cb_names = set(cb_results["name"])
    cf_names = set(cf_results["name"])
    overlap = cb_names & cf_names

    return {
        "overlap": len(overlap),
        "overlap_ratio": round(len(overlap) / min(len(cb_names), len(cf_names)), 2),
        "avg_mal_rating_cb": round(cb_results["mal_rating"].mean(), 2),
        "avg_mal_rating_cf": round(cf_results["mal_rating"].mean(), 2),
        "unique_to_cb": sorted(cb_names - cf_names),
        "unique_to_cf": sorted(cf_names - cb_names),
        "shared": sorted(overlap),
    }


def genre_distribution(results: pd.DataFrame) -> pd.Series:
    """Count genre frequency in recommendation results."""
    if results.empty:
        return pd.Series(dtype=int)
    genres = []
    for g_str in results["genre"].dropna():
        genres.extend([g.strip() for g in g_str.split(",")])
    return pd.Series(genres).value_counts()
