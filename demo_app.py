"""
demo_app.py

Streamlit UI for demonstrating product recommendations.

Run:
    streamlit run demo_app.py
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

from src.content_based_recommendation import content_based_filtering


APP_VERSION = "streamlit-demo-v1"
PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_INPUT_DIR = PROJECT_ROOT / "data" / "model_input"
INTERACTION_TRAIN_PATHS = [
    MODEL_INPUT_DIR / "svd" / "interaction_train.csv",
    MODEL_INPUT_DIR / "popularity" / "interaction_train.csv",
    MODEL_INPUT_DIR / "cf" / "interaction_train.csv",
]
PRODUCT_METADATA_PATHS = [
    MODEL_INPUT_DIR / "content_based" / "products.csv",
]
FINAL_SVD_MODEL_PATH = PROJECT_ROOT / "models" / "final_model_svd.pkl"
MODEL_METRICS_PATH = PROJECT_ROOT / "reports" / "model_metrics" / "all_model_metrics.csv"
RELEVANCE_THRESHOLD = 4.5
DEMO_CANDIDATE_LIMIT = 500

logger = logging.getLogger(__name__)


class RecommendationData:
    def __init__(self) -> None:
        self.df, self.data_source = self._load_ratings_data()
        self.product_metadata, self.metadata_source = self._load_product_metadata()
        self.rank_scores = self._compute_rank_scores()
        self.svd_model, self.svd_status = self._load_svd_model()
        self.browsing_history = self.df[["user_id", "prod_id", "rating"]].copy()
        self.purchases = self.df.loc[
            self.df["rating"] >= RELEVANCE_THRESHOLD,
            ["user_id", "prod_id", "rating"],
        ].copy()

    def _load_ratings_data(self) -> tuple[pd.DataFrame, str]:
        source_path = next((path for path in INTERACTION_TRAIN_PATHS if path.exists()), None)
        if source_path is None:
            raise FileNotFoundError(
                "Could not load ratings data. Expected interaction_train.csv under data/model_input/."
            )

        df = pd.read_csv(source_path)
        rename_map = {
            "UserId": "user_id",
            "ProductId": "prod_id",
            "Rating": "rating",
            "userId": "user_id",
            "productId": "prod_id",
        }
        df = df.rename(columns=rename_map)
        required_cols = {"user_id", "prod_id", "rating"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"{source_path} is missing required columns: {sorted(missing)}")

        df = df[["user_id", "prod_id", "rating"]].copy()
        df = df.dropna(subset=["user_id", "prod_id", "rating"])
        df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
        df = df.dropna(subset=["rating"])
        df["user_id"] = df["user_id"].astype(str)
        df["prod_id"] = df["prod_id"].astype(str)

        active_users = df["user_id"].value_counts()
        active_products = df["prod_id"].value_counts()
        df = df[
            df["user_id"].isin(active_users[active_users >= 5].index)
            & df["prod_id"].isin(active_products[active_products >= 3].index)
        ].copy()

        return df[["user_id", "prod_id", "rating"]], f"dataset from {source_path.relative_to(PROJECT_ROOT)}"

    def _load_svd_model(self):
        if not FINAL_SVD_MODEL_PATH.exists():
            return None, "SVD model file not found"

        try:
            with open(FINAL_SVD_MODEL_PATH, "rb") as model_file:
                return pickle.load(model_file), "SVD model loaded"
        except Exception as exc:
            return None, f"SVD model unavailable: {exc}"

    def _load_product_metadata(self) -> tuple[pd.DataFrame, str]:
        source_path = next((path for path in PRODUCT_METADATA_PATHS if path.exists()), None)
        if source_path is None:
            return self._fallback_metadata(), "Using fallback metadata"

        raw = pd.read_csv(source_path)

        def first_existing(*names: str) -> pd.Series | None:
            for name in names:
                if name in raw.columns:
                    return raw[name]
            return None

        prod_series = first_existing("prod_id", "ProductId", "product_id", "prodId")
        if prod_series is None:
            raise ValueError(
                f"{source_path} must contain a product id column such as ProductId or prod_id."
            )

        product_name_series = first_existing("product_name", "Title", "title")
        category_series = first_existing("category", "MainCategory", "main_category", "Categories")
        brand_series = first_existing("brand", "Brand")

        metadata = pd.DataFrame(
            {
                "prod_id": prod_series.astype(str),
                "product_name": (
                    product_name_series.astype(str)
                    if product_name_series is not None
                    else pd.Series([None] * len(raw))
                ),
                "category": (
                    category_series.astype(str)
                    if category_series is not None
                    else pd.Series([None] * len(raw))
                ),
                "brand": (
                    brand_series.astype(str)
                    if brand_series is not None
                    else pd.Series([None] * len(raw))
                ),
            }
        )
        metadata["product_name"] = metadata["product_name"].replace("None", None)
        metadata["category"] = metadata["category"].replace("None", None)
        metadata["brand"] = metadata["brand"].replace("None", None)
        metadata = metadata.drop_duplicates("prod_id")

        demo_product_ids = set(self.df["prod_id"])
        metadata = metadata[metadata["prod_id"].isin(demo_product_ids)].copy()
        if metadata.empty:
            return self._fallback_metadata(), "Using fallback metadata"

        return metadata, f"Using product metadata from {source_path.relative_to(PROJECT_ROOT)}"

    def _fallback_metadata(self) -> pd.DataFrame:
        metadata = (
            self.df.groupby("prod_id")
            .agg(avg_rating=("rating", "mean"), rating_count=("rating", "count"))
            .reset_index()
        )
        metadata["category"] = "Electronics"
        metadata["product_name"] = "Product " + metadata["prod_id"].astype(str)
        return metadata

    def _compute_rank_scores(self) -> pd.DataFrame:
        scores = (
            self.df.groupby("prod_id")
            .agg(avg_rating=("rating", "mean"), rating_count=("rating", "count"))
            .reset_index()
        )
        global_mean = self.df["rating"].mean()
        prior_weight = max(scores["rating_count"].mean(), 1)
        scores["score"] = (
            scores["rating_count"] * scores["avg_rating"] + prior_weight * global_mean
        ) / (scores["rating_count"] + prior_weight)
        return scores

    def summary(self) -> dict:
        return {
            "rows": int(len(self.df)),
            "users": int(self.df["user_id"].nunique()),
            "products": int(self.df["prod_id"].nunique()),
            "avg_rating": round(float(self.df["rating"].mean()), 3),
            "metadata_source": f"{self.metadata_source}, {self.data_source}",
            "warning": (
                "Fallback metadata is enough for a runnable demo. Add "
                "data/model_input/content_based/products.csv with category, brand, title, "
                "and description for a stronger content-based demo."
                if not any(path.exists() for path in PRODUCT_METADATA_PATHS)
                else "Product metadata file detected."
            ),
            "cf_status": self.svd_status,
        }

    def metrics(self) -> pd.DataFrame:
        if not MODEL_METRICS_PATH.exists():
            return pd.DataFrame()

        try:
            return pd.read_csv(MODEL_METRICS_PATH)
        except Exception as exc:
            logger.warning("Could not read model metrics: %s", exc)
            return pd.DataFrame()

    def users(self, limit: int = 100) -> list[dict]:
        counts = self.df["user_id"].value_counts().head(limit)
        return [
            {"user_id": str(user_id), "interactions": int(count)}
            for user_id, count in counts.items()
        ]

    def _merge_metadata(self, recommendations: pd.DataFrame) -> pd.DataFrame:
        metadata_cols = [
            col
            for col in ["prod_id", "product_name", "category", "brand", "avg_rating", "rating_count"]
            if col in self.product_metadata.columns
        ]
        metadata_cols = [
            col
            for col in metadata_cols
            if col == "prod_id" or col not in recommendations.columns
        ]
        metadata = self.product_metadata[metadata_cols].drop_duplicates("prod_id")
        return recommendations.merge(metadata, on="prod_id", how="left")

    def rank_recommendations(self, user_id: str, top_n: int) -> pd.DataFrame:
        seen_products = set(self.df.loc[self.df["user_id"] == user_id, "prod_id"])
        recommendations = self.rank_scores[
            ~self.rank_scores["prod_id"].isin(seen_products)
        ].copy()
        recommendations = recommendations.sort_values(
            ["score", "rating_count"], ascending=[False, False]
        ).head(top_n)
        recommendations = self._merge_metadata(recommendations)
        recommendations["source"] = "Rank-Based"
        return recommendations

    def _candidate_products(self, user_id: str, limit: int = DEMO_CANDIDATE_LIMIT) -> list[str]:
        seen_products = set(self.df.loc[self.df["user_id"] == user_id, "prod_id"])
        ranked_candidates = self.rank_scores[
            ~self.rank_scores["prod_id"].isin(seen_products)
        ].sort_values(["score", "rating_count"], ascending=[False, False])

        if limit:
            ranked_candidates = ranked_candidates.head(limit)

        candidates = ranked_candidates["prod_id"].astype(str).tolist()
        if candidates:
            return candidates

        return [
            prod_id
            for prod_id in self.product_metadata["prod_id"].drop_duplicates().astype(str)
            if prod_id not in seen_products
        ][:limit]

    def _svd_candidate_scores(self, user_id: str) -> pd.DataFrame:
        if self.svd_model is None:
            raise RuntimeError(
                "SVD Collaborative Filtering is unavailable. "
                f"{self.svd_status}. Run this demo in the project conda environment "
                "with scikit-surprise installed, or retrain/export the SVD model first."
            )

        predictions = [
            {
                "prod_id": prod_id,
                "score": float(self.svd_model.predict(user_id, prod_id).est),
            }
            for prod_id in self._candidate_products(user_id)
        ]
        return pd.DataFrame(predictions)

    def svd_recommendations(self, user_id: str, top_n: int) -> pd.DataFrame:
        recommendations = self._svd_candidate_scores(user_id)
        recommendations = recommendations.sort_values("score", ascending=False).head(top_n)
        recommendations = self._merge_metadata(recommendations)
        recommendations["source"] = "SVD Collaborative Filtering"
        return recommendations

    def hybrid_recommendations(self, user_id: str, top_n: int) -> pd.DataFrame:
        recommendations = self._svd_candidate_scores(user_id)
        rank_lookup = self.rank_scores.set_index("prod_id")["score"].to_dict()
        recommendations["rank_score"] = (
            recommendations["prod_id"].map(rank_lookup).fillna(self.df["rating"].mean())
        )
        recommendations["score"] = (
            0.8 * recommendations["score"] + 0.2 * recommendations["rank_score"]
        )
        recommendations = recommendations.sort_values("score", ascending=False).head(top_n)
        recommendations = self._merge_metadata(recommendations)
        recommendations["source"] = "Hybrid SVD + Rank"
        return recommendations

    def content_recommendations(self, user_id: str, top_n: int) -> pd.DataFrame:
        candidate_ids = set(self._candidate_products(user_id, limit=DEMO_CANDIDATE_LIMIT))
        history_ids = set(self.df.loc[self.df["user_id"] == user_id, "prod_id"].astype(str))
        product_subset = self.product_metadata[
            self.product_metadata["prod_id"].astype(str).isin(candidate_ids | history_ids)
        ].copy()
        if product_subset.empty:
            product_subset = self.product_metadata.head(DEMO_CANDIDATE_LIMIT).copy()

        return content_based_filtering(
            user_id=user_id,
            purchases=self.purchases,
            browsing_history=self.browsing_history,
            products=product_subset,
            top_n=top_n,
            item_col="prod_id",
        )

    def recommend(self, user_id: str, method: str, top_n: int) -> pd.DataFrame:
        if method == "rank":
            recommendations = self.rank_recommendations(user_id, top_n)
        elif method == "svd":
            recommendations = self.svd_recommendations(user_id, top_n)
        elif method == "hybrid":
            recommendations = self.hybrid_recommendations(user_id, top_n)
        elif method == "content":
            recommendations = self.content_recommendations(user_id, top_n)
        else:
            raise ValueError(f"Unknown recommendation method: {method}")

        recommendations = recommendations.copy()
        recommendations.insert(0, "rank", range(1, len(recommendations) + 1))
        return recommendations


@st.cache_resource(show_spinner="Loading recommendation data...")
def load_data() -> RecommendationData:
    return RecommendationData()


def prepare_display_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    display = df.copy()
    for column in ["score", "avg_rating", "rank_score"]:
        if column in display.columns:
            display[column] = pd.to_numeric(display[column], errors="coerce").round(4)

    preferred_columns = [
        "rank",
        "prod_id",
        "product_name",
        "category",
        "brand",
        "score",
        "avg_rating",
        "rating_count",
        "rank_score",
        "source",
    ]
    columns = [col for col in preferred_columns if col in display.columns]
    columns.extend(col for col in display.columns if col not in columns)
    return display[columns]


def render_summary(summary: dict) -> None:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Interactions", f"{summary['rows']:,}")
    col2.metric("Users", f"{summary['users']:,}")
    col3.metric("Products", f"{summary['products']:,}")
    col4.metric("Avg rating", summary["avg_rating"])

    st.caption(summary["metadata_source"])
    st.info(summary["cf_status"])
    if summary["warning"]:
        st.warning(summary["warning"])


def main() -> None:
    st.set_page_config(
        page_title="Product Recommendation Demo",
        layout="wide",
    )

    st.title("Product Recommendation Demo")
    st.caption(f"Demo version: {APP_VERSION}")

    try:
        data = load_data()
    except Exception as exc:
        st.error(f"Cannot load demo data: {exc}")
        st.stop()

    summary = data.summary()
    render_summary(summary)

    method_options = {
        "rank": "Popularity / Rank-Based",
        "svd": "SVD Collaborative Filtering",
        "hybrid": "Hybrid SVD + Rank",
        "content": "Content-Based",
    }
    top_users = data.users(limit=200)
    user_labels = [
        f"{item['user_id']} ({item['interactions']} interactions)"
        for item in top_users
    ]
    user_lookup = dict(zip(user_labels, [item["user_id"] for item in top_users]))

    with st.sidebar:
        st.header("Recommendation Settings")
        selected_label = st.selectbox("User ID", user_labels)
        typed_user_id = st.text_input(
            "Or enter a user ID",
            value=user_lookup[selected_label],
            help="Use a real user_id from the training data for personalized recommendations.",
        )
        method = st.selectbox(
            "Model",
            options=list(method_options.keys()),
            format_func=lambda key: method_options[key],
        )
        top_n = st.slider("Top N", min_value=1, max_value=50, value=10)
        generate = st.button("Generate Recommendations", type="primary", width="stretch")

    st.subheader("Top-N Recommendations")
    if not generate:
        st.write("Choose a user, model, and Top N value, then generate recommendations.")
        st.stop()

    user_id = typed_user_id.strip()
    if not user_id:
        st.error("Please enter a user ID.")
        st.stop()

    try:
        recommendations = data.recommend(user_id=user_id, method=method, top_n=top_n)
    except RuntimeError as exc:
        st.error(str(exc))
        st.stop()
    except Exception as exc:
        st.error(f"Could not generate recommendations: {exc}")
        st.stop()

    if recommendations.empty:
        st.warning("No recommendations found for this user.")
    else:
        st.dataframe(
            prepare_display_frame(recommendations),
            width="stretch",
            hide_index=True,
        )

    with st.expander("Model Metrics", expanded=False):
        metrics = data.metrics()
        if metrics.empty:
            st.write("Model metrics file not found.")
        else:
            st.dataframe(metrics, width="stretch", hide_index=True)


if __name__ == "__main__":
    main()
