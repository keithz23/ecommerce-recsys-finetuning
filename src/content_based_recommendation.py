"""
content_based_recommendation.py

Content-based recommendation utilities.

This module builds recommendations from product metadata and a user's own
interaction history. It supports simple categorical matching and, when title or
description metadata is available, TF-IDF text similarity.
"""

# ---------------------------------------------------------
# IMPORT LIBRARIES
# ---------------------------------------------------------

from __future__ import annotations

import logging
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------


def _find_column(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    """Return the first matching column name from a list of candidates."""

    for column in candidates:
        if column in df.columns:
            return column
    return None


def _empty_recommendations(products: pd.DataFrame) -> pd.DataFrame:
    """Create an empty recommendation frame with a stable output schema."""

    columns = list(products.columns)
    for column in ["score", "source"]:
        if column not in columns:
            columns.append(column)
    return pd.DataFrame(columns=columns)


def _history_items(
    interactions: Optional[pd.DataFrame],
    user_id: str,
    user_col: str,
    item_col: str,
    default_weight: float,
) -> pd.DataFrame:
    """Collect item ids and weights for a user from an interaction table."""

    if interactions is None or interactions.empty:
        return pd.DataFrame(columns=[item_col, "weight"])

    if user_col not in interactions.columns or item_col not in interactions.columns:
        return pd.DataFrame(columns=[item_col, "weight"])

    user_interactions = interactions.loc[
        interactions[user_col] == user_id, [item_col] + [
            col for col in ["rating", "score", "weight"] if col in interactions.columns
        ]
    ].copy()

    if user_interactions.empty:
        return pd.DataFrame(columns=[item_col, "weight"])

    if "weight" in user_interactions.columns:
        user_interactions["weight"] = pd.to_numeric(
            user_interactions["weight"], errors="coerce"
        ).fillna(default_weight)
    elif "rating" in user_interactions.columns:
        user_interactions["weight"] = (
            pd.to_numeric(user_interactions["rating"], errors="coerce")
            .fillna(3.0)
            .clip(lower=1.0, upper=5.0)
            / 5.0
            * default_weight
        )
    elif "score" in user_interactions.columns:
        user_interactions["weight"] = pd.to_numeric(
            user_interactions["score"], errors="coerce"
        ).fillna(default_weight)
    else:
        user_interactions["weight"] = default_weight

    return user_interactions[[item_col, "weight"]].dropna(subset=[item_col])


def _normalise(series: pd.Series) -> pd.Series:
    """Scale a numeric series to 0..1 while handling constant values."""

    values = pd.to_numeric(series, errors="coerce").fillna(0.0).astype(float)
    minimum = values.min()
    maximum = values.max()

    if maximum <= minimum:
        return pd.Series(np.where(values > 0, 1.0, 0.0), index=series.index)

    return (values - minimum) / (maximum - minimum)


def _weighted_preference_score(
    candidates: pd.DataFrame,
    user_products: pd.DataFrame,
    history_weights: pd.Series,
    column: str,
) -> pd.Series:
    """Score candidates by weighted overlap with a user metadata preference."""

    if column not in candidates.columns or column not in user_products.columns:
        return pd.Series(0.0, index=candidates.index)

    preferences = (
        user_products.assign(_weight=history_weights.reindex(user_products.index))
        .dropna(subset=[column])
        .groupby(column)["_weight"]
        .sum()
    )

    if preferences.empty or preferences.max() == 0:
        return pd.Series(0.0, index=candidates.index)

    preferences = preferences / preferences.max()
    return candidates[column].map(preferences).fillna(0.0).astype(float)


def _text_similarity_score(
    products: pd.DataFrame,
    candidates: pd.DataFrame,
    user_products: pd.DataFrame,
    history_weights: pd.Series,
    text_columns: list[str],
) -> pd.Series:
    """Compute TF-IDF similarity between candidate products and user profile."""

    available_columns = [col for col in text_columns if col in products.columns]
    if not available_columns:
        return pd.Series(0.0, index=candidates.index)

    text = (
        products[available_columns]
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
        .str.strip()
    )

    if text.eq("").all():
        return pd.Series(0.0, index=candidates.index)

    try:
        tfidf_matrix = TfidfVectorizer(stop_words="english", min_df=1).fit_transform(
            text
        )
    except ValueError:
        return pd.Series(0.0, index=candidates.index)

    user_positions = products.index.get_indexer(user_products.index)
    candidate_positions = products.index.get_indexer(candidates.index)
    valid_user_mask = user_positions >= 0
    valid_candidate_mask = candidate_positions >= 0

    if not valid_user_mask.any() or not valid_candidate_mask.any():
        return pd.Series(0.0, index=candidates.index)

    user_positions = user_positions[valid_user_mask]
    weights = history_weights.reindex(user_products.index).to_numpy(dtype=float)[
        valid_user_mask
    ]
    if weights.sum() <= 0:
        weights = np.ones_like(weights)

    user_matrix = tfidf_matrix[user_positions]
    user_profile = np.asarray(
        user_matrix.multiply(weights[:, None]).sum(axis=0)
    ) / weights.sum()

    scores = pd.Series(0.0, index=candidates.index)
    valid_candidate_positions = candidate_positions[valid_candidate_mask]
    valid_candidate_index = candidates.index[valid_candidate_mask]
    scores.loc[valid_candidate_index] = linear_kernel(
        tfidf_matrix[valid_candidate_positions], user_profile
    ).ravel()

    return scores.astype(float)


# ---------------------------------------------------------
# CONTENT-BASED RECOMMENDATION
# ---------------------------------------------------------


def content_based_filtering(
    user_id: str,
    purchases: Optional[pd.DataFrame],
    browsing_history: Optional[pd.DataFrame],
    products: pd.DataFrame,
    top_n: Optional[int] = 10,
    user_col: str = "user_id",
    item_col: Optional[str] = None,
    text_columns: Optional[list[str]] = None,
    weights: Optional[dict[str, float]] = None,
) -> pd.DataFrame:
    """
    Generate content-based recommendations for a user.

    The function builds a user profile from browsing and purchase history, then
    scores unseen products using available metadata. It supports both
    ``product_id`` and ``prod_id`` naming conventions.

    Parameters:
    - user_id: User identifier to recommend for.
    - purchases: Optional purchase interactions with user and product columns.
    - browsing_history: Optional browsing interactions with user and product columns.
    - products: Product metadata table.
    - top_n: Number of recommendations to return. If None, return all.
    - user_col: Name of the user id column in interaction tables.
    - item_col: Product id column. If None, infer from product_id/prod_id.
    - text_columns: Metadata text columns for TF-IDF similarity.
    - weights: Optional signal weights for category, brand, text, rating, popularity.

    Returns:
    - pd.DataFrame: Recommended products sorted by descending score.
    """

    logger.debug("Content-Based Filtering for user_id: %s", user_id)

    if products is None or products.empty:
        logger.debug("No product metadata was provided.")
        return pd.DataFrame(columns=["score", "source"])

    item_col = item_col or _find_column(products, ["product_id", "prod_id"])
    if item_col is None:
        raise ValueError("products must contain either 'product_id' or 'prod_id'.")

    if text_columns is None:
        text_columns = [
            "product_name",
            "title",
            "description",
            "category",
            "brand",
        ]

    signal_weights = {
        "category": 0.30,
        "brand": 0.15,
        "text": 0.35,
        "rating": 0.15,
        "popularity": 0.05,
    }
    if weights is not None:
        signal_weights.update(weights)

    browse_items = _history_items(
        browsing_history, user_id, user_col, item_col, default_weight=1.0
    )
    purchase_items = _history_items(
        purchases, user_id, user_col, item_col, default_weight=2.0
    )

    history = pd.concat([browse_items, purchase_items], ignore_index=True)
    if history.empty:
        logger.debug("No browsing or purchase history for user_id: %s", user_id)
        return _empty_recommendations(products)

    history = history.groupby(item_col, as_index=False)["weight"].sum()
    history_items_set = set(history[item_col])
    logger.debug("User history contains %d unique products.", len(history_items_set))

    user_products = products[products[item_col].isin(history_items_set)].copy()
    if user_products.empty:
        logger.debug("No matching product metadata for user history.")
        return _empty_recommendations(products)

    history_weights = (
        user_products[item_col]
        .map(history.set_index(item_col)["weight"])
        .fillna(1.0)
        .astype(float)
    )
    history_weights.index = user_products.index

    candidates = products[~products[item_col].isin(history_items_set)].copy()
    if candidates.empty:
        logger.debug("No unseen products available for recommendation.")
        return _empty_recommendations(products)

    signals = pd.DataFrame(index=candidates.index)
    signals["category"] = _weighted_preference_score(
        candidates, user_products, history_weights, "category"
    )
    signals["brand"] = _weighted_preference_score(
        candidates, user_products, history_weights, "brand"
    )
    signals["text"] = _text_similarity_score(
        products, candidates, user_products, history_weights, text_columns
    )

    rating_col = _find_column(products, ["rating", "avg_rating", "average_rating"])
    if rating_col:
        signals["rating"] = _normalise(candidates[rating_col])
    else:
        signals["rating"] = 0.0

    popularity_col = _find_column(
        products,
        ["rating_count", "num_ratings", "review_count", "reviews_count", "popularity"],
    )
    if popularity_col:
        signals["popularity"] = _normalise(np.log1p(candidates[popularity_col]))
    else:
        signals["popularity"] = 0.0

    active_weights = {
        name: weight
        for name, weight in signal_weights.items()
        if name in signals.columns and signals[name].sum() > 0 and weight > 0
    }
    if not active_weights:
        logger.debug("No usable content signals for recommendations.")
        return _empty_recommendations(products)

    weight_sum = sum(active_weights.values())
    candidates["score"] = sum(
        signals[name] * (weight / weight_sum)
        for name, weight in active_weights.items()
    )
    candidates["source"] = "Content-Based Filtering"

    recommendations = candidates.sort_values(
        ["score", item_col], ascending=[False, True]
    )
    if top_n is not None:
        recommendations = recommendations.head(top_n)

    logger.debug(
        "Generated %d content-based recommendations for user_id: %s",
        len(recommendations),
        user_id,
    )
    return recommendations.reset_index(drop=True)
