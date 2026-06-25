"""
demo_app.py

Lightweight local UI for demonstrating recommendations.

Run:
    python demo_app.py

Then open:
    http://localhost:8000
"""

from __future__ import annotations

import json
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pandas as pd

from src.content_based_recommendation import content_based_filtering


HOST = "127.0.0.1"
PORT = 8000
FALLBACK_PORTS = [8001, 8002, 8003]
APP_VERSION = "fast-demo-v2"
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


HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Recommendation Demo</title>
  <style>
    :root {
      --bg: #f6f7f9;
      --panel: #ffffff;
      --ink: #151922;
      --muted: #606979;
      --line: #dfe3ea;
      --accent: #2364aa;
      --accent-dark: #174a7f;
      --good: #0f766e;
      --warn: #b45309;
      --shadow: 0 8px 24px rgba(21, 25, 34, 0.08);
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      min-height: 100vh;
      color: var(--ink);
      background: var(--bg);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      letter-spacing: 0;
    }

    header {
      border-bottom: 1px solid var(--line);
      background: var(--panel);
    }

    .wrap {
      width: min(1180px, calc(100vw - 32px));
      margin: 0 auto;
    }

    .topbar {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      min-height: 68px;
    }

    h1 {
      margin: 0;
      font-size: 22px;
      line-height: 1.2;
      font-weight: 700;
    }

    .status {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      min-height: 32px;
      padding: 0 10px;
      border: 1px solid var(--line);
      border-radius: 6px;
      color: var(--muted);
      background: #fbfcfe;
      font-size: 13px;
      white-space: nowrap;
    }

    .dot {
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: var(--good);
    }

    main {
      padding: 24px 0 40px;
    }

    .layout {
      display: grid;
      grid-template-columns: 310px 1fr;
      gap: 20px;
      align-items: start;
    }

    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
    }

    .controls {
      padding: 16px;
      position: sticky;
      top: 16px;
    }

    .field {
      display: grid;
      gap: 6px;
      margin-bottom: 14px;
    }

    label {
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
    }

    select,
    input {
      width: 100%;
      min-height: 38px;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 0 10px;
      color: var(--ink);
      background: #fff;
      font: inherit;
    }

    button {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      width: 100%;
      min-height: 40px;
      border: 0;
      border-radius: 6px;
      color: #fff;
      background: var(--accent);
      font: inherit;
      font-weight: 700;
      cursor: pointer;
    }

    button:hover { background: var(--accent-dark); }
    button:disabled { opacity: 0.6; cursor: wait; }

    .summary,
    .metrics-summary {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
      margin-bottom: 20px;
    }

    .metric {
      padding: 14px;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
    }

    .metric strong {
      display: block;
      font-size: 22px;
      line-height: 1.2;
    }

    .metric span {
      display: block;
      margin-top: 4px;
      color: var(--muted);
      font-size: 12px;
    }

    .results {
      overflow: hidden;
    }

    .metrics {
      margin-bottom: 20px;
    }

    .results-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      padding: 16px;
      border-bottom: 1px solid var(--line);
    }

    h2 {
      margin: 0;
      font-size: 17px;
      line-height: 1.25;
    }

    .method-note {
      color: var(--muted);
      font-size: 13px;
      text-align: right;
    }

    table {
      width: 100%;
      border-collapse: collapse;
    }

    th,
    td {
      padding: 12px 14px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      vertical-align: top;
      font-size: 14px;
    }

    th {
      color: var(--muted);
      background: #fafbfc;
      font-size: 12px;
      text-transform: uppercase;
    }

    tr:last-child td { border-bottom: 0; }

    th.important-col,
    td.important-col {
      background: #eef6ff;
    }

    th.important-col {
      color: var(--accent-dark);
      box-shadow: inset 0 -2px 0 var(--accent);
    }

    td.important-col {
      box-shadow: inset 3px 0 0 rgba(35, 100, 170, 0.28);
    }

    .score {
      color: var(--good);
      font-weight: 700;
      white-space: nowrap;
    }

    .muted {
      color: var(--muted);
      font-size: 13px;
    }

    .empty {
      padding: 28px 16px;
      color: var(--muted);
      text-align: center;
    }

    .warning {
      margin-top: 14px;
      padding: 10px;
      border: 1px solid #f0c36a;
      border-radius: 6px;
      color: var(--warn);
      background: #fff8e8;
      font-size: 13px;
      line-height: 1.4;
    }

    @media (max-width: 820px) {
      .layout { grid-template-columns: 1fr; }
      .controls { position: static; }
      .summary,
      .metrics-summary { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .topbar { align-items: flex-start; flex-direction: column; padding: 14px 0; }
      .status { white-space: normal; }
      table { min-width: 720px; }
      .table-scroll { overflow-x: auto; }
    }
  </style>
</head>
<body>
  <header>
    <div class="wrap topbar">
      <h1>Amazon Product Recommendation Demo</h1>
      <div class="status"><span class="dot"></span><span id="metadataStatus">Loading dataset</span> · fast-demo-v2</div>
    </div>
  </header>

  <main class="wrap">
    <section class="summary" id="summary"></section>
    <section class="metrics">
      <div class="results-head" style="padding-left: 0; padding-right: 0;">
        <h2>Model Metrics</h2>
        <div class="method-note" id="metricsNote"></div>
      </div>
      <section class="panel results">
        <div class="table-scroll">
          <table>
            <thead>
              <tr>
                <th>Model</th>
                <th>K</th>
                <th>Evaluated Users</th>
                <th>Precision@K</th>
                <th>Recall@K</th>
                <th>F1@K</th>
                <th>HitRate@K</th>
                <th>MAP@K</th>
                <th>MRR@K</th>
                <th>NDCG@K</th>
                <th>Catalog Coverage</th>
                <th>RMSE</th>
                <th>MAE</th>
                <th>Rating Eval Rows</th>
              </tr>
            </thead>
            <tbody id="metricsBody">
              <tr><td colspan="14" class="empty">Loading metrics...</td></tr>
            </tbody>
          </table>
        </div>
      </section>
    </section>

    <section class="layout">
      <aside class="panel controls">
        <div class="field">
          <label for="userSelect">User</label>
          <select id="userSelect"></select>
        </div>
        <div class="field">
          <label for="methodSelect">Method</label>
          <select id="methodSelect">
            <option value="rank" selected>Rank-Based</option>
            <option value="svd">SVD Collaborative Filtering</option>
            <option value="hybrid">Hybrid SVD + Rank</option>
            <option value="content">Content-Based</option>
          </select>
        </div>
        <div class="field">
          <label for="topN">Top N</label>
          <input id="topN" type="number" min="1" max="50" value="10" />
        </div>
        <button id="recommendButton">Generate Recommendations</button>
        <div class="warning" id="warning"></div>
      </aside>

      <section class="panel results">
        <div class="results-head">
          <h2 id="resultTitle">Recommendations</h2>
          <div class="method-note" id="methodNote"></div>
        </div>
        <div class="table-scroll">
          <table>
            <thead>
              <tr>
                <th data-col="user">User ID</th>
                <th data-col="rank">Rank</th>
                <th data-col="product">Product</th>
                <th data-col="category">Category</th>
                <th data-col="rating">Rating</th>
                <th data-col="count">Count</th>
                <th data-col="score">Predicted Score</th>
              </tr>
            </thead>
            <tbody id="resultsBody">
              <tr><td colspan="7" class="empty">Choose a user and generate recommendations.</td></tr>
            </tbody>
          </table>
        </div>
      </section>
    </section>
  </main>

  <script>
    const state = { users: [], summary: null, metrics: [] };

    const fmt = new Intl.NumberFormat("en-US");

    function metric(value, label) {
      return `<div class="metric"><strong>${value}</strong><span>${label}</span></div>`;
    }

    async function api(path, options = {}) {
      const response = await fetch(path, {
        headers: { "Content-Type": "application/json" },
        ...options,
      });
      if (!response.ok) {
        const message = await response.text();
        let errorMessage = message;
        try {
          const payload = JSON.parse(message);
          errorMessage = payload.error || message;
        } catch {
          errorMessage = message;
        }
        throw new Error(errorMessage || `Request failed: ${response.status}`);
      }
      return response.json();
    }

    function renderSummary(summary) {
      document.getElementById("summary").innerHTML = [
        metric(fmt.format(summary.rows), "Ratings"),
        metric(fmt.format(summary.users), "Users"),
        metric(fmt.format(summary.products), "Products"),
        metric(Number(summary.avg_rating).toFixed(2), "Average Rating"),
      ].join("");
      document.getElementById("metadataStatus").textContent = summary.metadata_source;
      document.getElementById("warning").textContent = summary.warning;
    }

    function formatMetricValue(value) {
      if (value === null || value === undefined || value === "") {
        return "N/A";
      }
      const numeric = Number(value);
      if (!Number.isNaN(numeric)) {
        if (Math.abs(numeric) < 1 && numeric !== 0) {
          return numeric.toFixed(3);
        }
        return numeric.toFixed(Number.isInteger(numeric) ? 0 : 3);
      }
      return String(value);
    }

    function renderMetrics(metrics) {
      const body = document.getElementById("metricsBody");
      if (!metrics.length) {
        body.innerHTML = `<tr><td colspan="14" class="empty">No model metrics found.</td></tr>`;
        document.getElementById("metricsNote").textContent = "Missing reports/model_metrics/all_model_metrics.csv";
        return;
      }

      document.getElementById("metricsNote").textContent = `${metrics.length} rows from all_model_metrics.csv`;
      body.innerHTML = metrics.map((row) => `
        <tr>
          <td><strong>${row.Model ?? row.model ?? "N/A"}</strong></td>
          <td>${formatMetricValue(row.K ?? row.k)}</td>
          <td>${formatMetricValue(row.EvaluatedUsers ?? row.evaluated_users)}</td>
          <td>${formatMetricValue(row["Precision@K"] ?? row.precision_at_k)}</td>
          <td>${formatMetricValue(row["Recall@K"] ?? row.recall_at_k)}</td>
          <td>${formatMetricValue(row["F1@K"] ?? row.f1_at_k)}</td>
          <td>${formatMetricValue(row["HitRate@K"] ?? row.hit_rate_at_k)}</td>
          <td>${formatMetricValue(row["MAP@K"] ?? row.map_at_k)}</td>
          <td>${formatMetricValue(row["MRR@K"] ?? row.mrr_at_k)}</td>
          <td>${formatMetricValue(row["NDCG@K"] ?? row.ndcg_at_k)}</td>
          <td>${formatMetricValue(row["CatalogCoverage@K"] ?? row["CatalogCoverage"] ?? row.catalog_coverage_at_k)}</td>
          <td>${formatMetricValue(row.RMSE ?? row.rmse)}</td>
          <td>${formatMetricValue(row.MAE ?? row.mae)}</td>
          <td>${formatMetricValue(row.RatingEvalRows ?? row.rating_eval_rows)}</td>
        </tr>
      `).join("");
    }

    function renderUsers(users) {
      const select = document.getElementById("userSelect");
      select.innerHTML = users
        .map((user) => `<option value="${user.user_id}">${user.user_id} (${user.interactions} ratings)</option>`)
        .join("");
    }

    function renderRecommendations(payload) {
      document.getElementById("resultTitle").textContent = `${payload.method_label} Recommendations`;
      document.getElementById("methodNote").textContent = `User ${payload.user_id}`;
      highlightImportantColumns(payload.method);

      const body = document.getElementById("resultsBody");
      if (!payload.recommendations.length) {
        body.innerHTML = `<tr><td colspan="7" class="empty">No recommendations found for this user.</td></tr>`;
        return;
      }

      body.innerHTML = payload.recommendations.map((item, index) => `
        <tr>
          <td data-col="user">${payload.user_id}</td>
          <td data-col="rank">${index + 1}</td>
          <td data-col="product">
            <strong>${item.product_name || item.prod_id}</strong>
            <div class="muted">${item.prod_id}</div>
          </td>
          <td data-col="category">${item.category || "N/A"}</td>
          <td data-col="rating">${item.avg_rating ?? item.rating ?? "N/A"}</td>
          <td data-col="count">${item.rating_count ?? "N/A"}</td>
          <td data-col="score" class="score">${Number(item.score).toFixed(4)}</td>
        </tr>
      `).join("");
      highlightImportantColumns(payload.method);
    }

    function importantColumnsFor(method) {
      const columnsByMethod = {
        content: ["category", "rating", "score"],
        rank: ["rating", "count", "score"],
        svd: ["product", "score"],
        hybrid: ["rating", "count", "score"],
      };
      return columnsByMethod[method] || ["score"];
    }

    function highlightImportantColumns(method) {
      const importantColumns = new Set(importantColumnsFor(method));
      document.querySelectorAll("[data-col]").forEach((cell) => {
        cell.classList.toggle("important-col", importantColumns.has(cell.dataset.col));
      });
    }

    async function loadInitialData() {
      const [summary, users, metrics] = await Promise.all([
        api("/api/summary"),
        api("/api/users?limit=100"),
        api("/api/metrics"),
      ]);
      state.summary = summary;
      state.users = users.users;
        state.metrics = metrics.metrics;
        renderSummary(summary);
        renderMetrics(state.metrics);
        renderUsers(state.users);
        document.getElementById("resultTitle").textContent = "Recommendations";
        document.getElementById("methodNote").textContent = "Choose a user, then click Generate.";
      }

    async function generateRecommendations() {
      const button = document.getElementById("recommendButton");
      button.disabled = true;
      button.textContent = "Generating...";
      try {
        const payload = await api("/api/recommend", {
          method: "POST",
          body: JSON.stringify({
            user_id: document.getElementById("userSelect").value,
            method: document.getElementById("methodSelect").value,
            top_n: Number(document.getElementById("topN").value || 10),
          }),
        });
        renderRecommendations(payload);
      } catch (error) {
        document.getElementById("resultsBody").innerHTML =
          `<tr><td colspan="7" class="empty">${error.message}</td></tr>`;
      } finally {
        button.disabled = false;
        button.textContent = "Generate Recommendations";
      }
    }

    document.getElementById("recommendButton").addEventListener("click", generateRecommendations);
    document.getElementById("methodSelect").addEventListener("change", (event) => {
      highlightImportantColumns(event.target.value);
    });

    loadInitialData()
      .then(() => {
        document.getElementById("methodSelect").value = "rank";
        highlightImportantColumns(document.getElementById("methodSelect").value);
      })
      .catch((error) => {
        document.getElementById("metadataStatus").textContent = "Dataset failed to load";
        document.getElementById("warning").textContent = error.message;
        document.getElementById("metricsBody").innerHTML =
          `<tr><td colspan="14" class="empty">Metrics unavailable.</td></tr>`;
      });
  </script>
</body>
</html>
"""


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
                "Could not load ratings data. Expected a filtered interaction_train.csv "
                "under data/model_input/."
            )

        df = pd.read_csv(source_path)
        rename_map = {
            "UserId": "user_id",
            "ProductId": "prod_id",
            "Rating": "rating",
            "userId": "user_id",
            "productId": "prod_id",
            "rating": "rating",
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
            import pickle

            with open(FINAL_SVD_MODEL_PATH, "rb") as model_file:
                return pickle.load(model_file), "SVD model loaded"
        except Exception as exc:
            return None, f"SVD model unavailable: {exc}"

    def _load_product_metadata(self) -> tuple[pd.DataFrame, str]:
        source_path = next((path for path in PRODUCT_METADATA_PATHS if path.exists()), None)
        if source_path is not None:
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

        return self._fallback_metadata(), "Using fallback metadata"

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
                "data/model_input/content_based/products.csv "
                "with category, brand, title, and description for a stronger content-based demo."
                if not any(path.exists() for path in PRODUCT_METADATA_PATHS)
                else "Product metadata file detected."
            ),
            "cf_status": self.svd_status,
        }

    def metrics(self) -> list[dict]:
        if not MODEL_METRICS_PATH.exists():
            return []

        try:
            metrics_df = pd.read_csv(MODEL_METRICS_PATH)
        except Exception as exc:
            logger.warning("Could not read model metrics: %s", exc)
            return []

        return dataframe_to_records(metrics_df)

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
        merged = recommendations.merge(metadata, on="prod_id", how="left")
        return merged

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

        candidate_products = self._candidate_products(user_id)

        predictions = [
            {
                "prod_id": prod_id,
                "score": float(self.svd_model.predict(user_id, prod_id).est),
            }
            for prod_id in candidate_products
        ]
        return pd.DataFrame(predictions)

    def svd_recommendations(self, user_id: str, top_n: int) -> pd.DataFrame:
        recommendations = self._svd_candidate_scores(user_id)
        recommendations = recommendations.sort_values("score", ascending=False).head(top_n)
        recommendations = self._merge_metadata(recommendations)
        recommendations["source"] = "SVD Collaborative Filtering"
        return recommendations

    def hybrid_recommendations(self, user_id: str, top_n: int) -> pd.DataFrame:
        svd_recommendations = self._svd_candidate_scores(user_id)
        rank_lookup = self.rank_scores.set_index("prod_id")["score"].to_dict()
        svd_recommendations["rank_score"] = (
            svd_recommendations["prod_id"].map(rank_lookup).fillna(self.df["rating"].mean())
        )
        svd_recommendations["score"] = (
            0.8 * svd_recommendations["score"] + 0.2 * svd_recommendations["rank_score"]
        )
        svd_recommendations = svd_recommendations.sort_values(
            "score", ascending=False
        ).head(top_n)
        svd_recommendations["source"] = "Hybrid SVD + Rank"
        return svd_recommendations

    def content_recommendations(self, user_id: str, top_n: int) -> pd.DataFrame:
        candidate_ids = set(self._candidate_products(user_id, limit=DEMO_CANDIDATE_LIMIT))
        history_ids = set(self.df.loc[self.df["user_id"] == user_id, "prod_id"].astype(str))
        product_subset = self.product_metadata[
            self.product_metadata["prod_id"].astype(str).isin(candidate_ids | history_ids)
        ].copy()
        if product_subset.empty:
            product_subset = self.product_metadata.head(DEMO_CANDIDATE_LIMIT).copy()

        recommendations = content_based_filtering(
            user_id=user_id,
            purchases=self.purchases,
            browsing_history=self.browsing_history,
            products=product_subset,
            top_n=top_n,
            item_col="prod_id",
        )
        return recommendations


DATA = RecommendationData()


def dataframe_to_records(df: pd.DataFrame) -> list[dict]:
    safe = df.replace({pd.NA: None}).where(pd.notna(df), None)
    records = safe.to_dict(orient="records")

    def normalize(value):
        if value is None:
            return None
        if pd.isna(value):
            return None
        if isinstance(value, float):
            return round(value, 4)
        if hasattr(value, "item"):
            item = value.item()
            return normalize(item)
        return value

    return [{key: normalize(value) for key, value in record.items()} for record in records]


class DemoHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send_html(HTML)
            return

        if parsed.path == "/api/summary":
            self._send_json(DATA.summary())
            return

        if parsed.path == "/api/users":
            query = parse_qs(parsed.query)
            limit = int(query.get("limit", ["100"])[0])
            self._send_json({"users": DATA.users(limit=limit)})
            return

        if parsed.path == "/api/metrics":
            self._send_json({"metrics": DATA.metrics()})
            return

        self.send_error(404, "Not found")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/api/recommend":
            self.send_error(404, "Not found")
            return

        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
        user_id = str(payload.get("user_id", ""))
        method = payload.get("method", "rank")
        top_n = max(1, min(int(payload.get("top_n", 10)), 50))

        try:
            if method == "rank":
                recommendations = DATA.rank_recommendations(user_id, top_n)
                method_label = "Rank-Based"
            elif method == "svd":
                recommendations = DATA.svd_recommendations(user_id, top_n)
                method_label = "SVD Collaborative Filtering"
            elif method == "hybrid":
                recommendations = DATA.hybrid_recommendations(user_id, top_n)
                method_label = "Hybrid SVD + Rank"
            else:
                recommendations = DATA.content_recommendations(user_id, top_n)
                method_label = "Content-Based"
        except RuntimeError as exc:
            self._send_json(
                {
                    "user_id": user_id,
                    "method": method,
                    "method_label": "Unavailable",
                    "recommendations": [],
                    "error": str(exc),
                },
                status=400,
            )
            return

        self._send_json(
            {
                "user_id": user_id,
                "method": method,
                "method_label": method_label,
                "recommendations": dataframe_to_records(recommendations),
            }
        )

    def log_message(self, format: str, *args) -> None:
        logger.info("%s - %s", self.address_string(), format % args)

    def _send_html(self, html: str) -> None:
        body = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-store, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("X-Demo-Version", APP_VERSION)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, payload: dict, status: int = 200) -> None:
        def sanitize(value):
            if value is None:
                return None
            if isinstance(value, dict):
                return {key: sanitize(item) for key, item in value.items()}
            if isinstance(value, list):
                return [sanitize(item) for item in value]
            if isinstance(value, tuple):
                return [sanitize(item) for item in value]
            if isinstance(value, float) and pd.isna(value):
                return None
            if pd.isna(value):
                return None
            if hasattr(value, "item"):
                return sanitize(value.item())
            return value

        body = json.dumps(sanitize(payload), ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("X-Demo-Version", APP_VERSION)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    server = None
    active_port = PORT
    for candidate_port in [PORT, *FALLBACK_PORTS]:
        try:
            server = ThreadingHTTPServer((HOST, candidate_port), DemoHandler)
            active_port = candidate_port
            break
        except OSError:
            continue

    if server is None:
        raise OSError(f"Could not bind demo server on ports: {[PORT, *FALLBACK_PORTS]}")

    print(f"Recommendation demo running at http://{HOST}:{active_port}")
    print(f"Demo version: {APP_VERSION}")
    print("Press Ctrl+C to stop.")
    server.serve_forever()


if __name__ == "__main__":
    main()
