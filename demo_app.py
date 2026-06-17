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
PROJECT_ROOT = Path(__file__).resolve().parent
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "processed_data.pkl"
PRODUCT_METADATA_PATH = PROJECT_ROOT / "data" / "processed" / "product_metadata.csv"
FINAL_SVD_MODEL_PATH = PROJECT_ROOT / "models" / "final_model_svd.pkl"
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
RAW_SAMPLE_ROWS_PER_FILE = 50_000
RELEVANCE_THRESHOLD = 4.5

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

    .summary {
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
      .summary { grid-template-columns: repeat(2, minmax(0, 1fr)); }
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
      <div class="status"><span class="dot"></span><span id="metadataStatus">Loading dataset</span></div>
    </div>
  </header>

  <main class="wrap">
    <section class="summary" id="summary"></section>

    <section class="layout">
      <aside class="panel controls">
        <div class="field">
          <label for="userSelect">User</label>
          <select id="userSelect"></select>
        </div>
        <div class="field">
          <label for="methodSelect">Method</label>
          <select id="methodSelect">
            <option value="content">Content-Based</option>
            <option value="rank">Rank-Based</option>
            <option value="svd">SVD Collaborative Filtering</option>
            <option value="hybrid">Hybrid SVD + Rank</option>
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
                <th data-col="rank">#</th>
                <th data-col="product">Product</th>
                <th data-col="category">Category</th>
                <th data-col="rating">Rating</th>
                <th data-col="count">Count</th>
                <th data-col="score">Score</th>
              </tr>
            </thead>
            <tbody id="resultsBody">
              <tr><td colspan="6" class="empty">Choose a user and generate recommendations.</td></tr>
            </tbody>
          </table>
        </div>
      </section>
    </section>
  </main>

  <script>
    const state = { users: [], summary: null };

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
        body.innerHTML = `<tr><td colspan="6" class="empty">No recommendations found for this user.</td></tr>`;
        return;
      }

      body.innerHTML = payload.recommendations.map((item, index) => `
        <tr>
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
      const [summary, users] = await Promise.all([
        api("/api/summary"),
        api("/api/users?limit=100"),
      ]);
      state.summary = summary;
      state.users = users.users;
      renderSummary(summary);
      renderUsers(state.users);
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
          `<tr><td colspan="6" class="empty">${error.message}</td></tr>`;
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
        highlightImportantColumns(document.getElementById("methodSelect").value);
        return generateRecommendations();
      })
      .catch((error) => {
        document.getElementById("metadataStatus").textContent = "Dataset failed to load";
        document.getElementById("warning").textContent = error.message;
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
        if PROCESSED_DATA_PATH.exists():
            try:
                df = pd.read_pickle(PROCESSED_DATA_PATH)
                return df[["user_id", "prod_id", "rating"]].copy(), "processed data"
            except Exception as exc:
                logger.warning("Could not read processed_data.pkl: %s", exc)

        csv_files = sorted(RAW_DATA_DIR.glob("amazon_product_ratings_part_*.csv"))
        if not csv_files:
            raise FileNotFoundError(
                "Could not load ratings data. Expected processed_data.pkl or raw CSV files."
            )

        frames = [
            pd.read_csv(
                csv_file,
                header=None,
                names=["user_id", "prod_id", "rating", "timestamp"],
                usecols=[0, 1, 2],
                nrows=RAW_SAMPLE_ROWS_PER_FILE,
            )
            for csv_file in csv_files
        ]
        df = pd.concat(frames, ignore_index=True)
        df = df.dropna(subset=["user_id", "prod_id", "rating"])
        df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
        df = df.dropna(subset=["rating"])

        active_users = df["user_id"].value_counts()
        active_products = df["prod_id"].value_counts()
        df = df[
            df["user_id"].isin(active_users[active_users >= 5].index)
            & df["prod_id"].isin(active_products[active_products >= 3].index)
        ].copy()

        return df[["user_id", "prod_id", "rating"]], "raw sample data"

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
        if PRODUCT_METADATA_PATH.exists():
            metadata = pd.read_csv(PRODUCT_METADATA_PATH)
            return metadata, "Using product metadata"

        metadata = (
            self.df.groupby("prod_id")
            .agg(avg_rating=("rating", "mean"), rating_count=("rating", "count"))
            .reset_index()
        )
        metadata["category"] = "Electronics"
        metadata["product_name"] = "Product " + metadata["prod_id"].astype(str)
        return metadata, "Using fallback metadata"

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
                "data/processed/product_metadata.csv with category, brand, title, "
                "and description for a stronger content-based demo."
                if not PRODUCT_METADATA_PATH.exists()
                else "Product metadata file detected."
            ),
            "cf_status": self.svd_status,
        }

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

    def svd_recommendations(self, user_id: str, top_n: int) -> pd.DataFrame:
        if self.svd_model is None:
            raise RuntimeError(
                "SVD Collaborative Filtering is unavailable. "
                f"{self.svd_status}. Run this demo in the project conda environment "
                "with scikit-surprise installed, or retrain/export the SVD model first."
            )

        seen_products = set(self.df.loc[self.df["user_id"] == user_id, "prod_id"])
        candidate_products = [
            prod_id
            for prod_id in self.product_metadata["prod_id"].drop_duplicates()
            if prod_id not in seen_products
        ]

        predictions = [
            {
                "prod_id": prod_id,
                "score": float(self.svd_model.predict(user_id, prod_id).est),
            }
            for prod_id in candidate_products
        ]
        recommendations = pd.DataFrame(predictions)
        recommendations = recommendations.sort_values("score", ascending=False).head(top_n)
        recommendations = self._merge_metadata(recommendations)
        recommendations["source"] = "SVD Collaborative Filtering"
        return recommendations

    def hybrid_recommendations(self, user_id: str, top_n: int) -> pd.DataFrame:
        svd_recommendations = self.svd_recommendations(user_id, top_n=len(self.product_metadata))
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
        recommendations = content_based_filtering(
            user_id=user_id,
            purchases=self.purchases,
            browsing_history=self.browsing_history,
            products=self.product_metadata,
            top_n=top_n,
            item_col="prod_id",
        )
        return recommendations


DATA = RecommendationData()


def dataframe_to_records(df: pd.DataFrame) -> list[dict]:
    safe = df.replace({pd.NA: None}).where(pd.notna(df), None)
    records = safe.to_dict(orient="records")
    return [
        {
            key: (
                round(float(value), 4)
                if isinstance(value, float)
                else int(value)
                if hasattr(value, "item") and isinstance(value.item(), int)
                else value.item()
                if hasattr(value, "item")
                else value
            )
            for key, value in record.items()
        }
        for record in records
    ]


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

        self.send_error(404, "Not found")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/api/recommend":
            self.send_error(404, "Not found")
            return

        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
        user_id = str(payload.get("user_id", ""))
        method = payload.get("method", "content")
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
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    server = ThreadingHTTPServer((HOST, PORT), DemoHandler)
    print(f"Recommendation demo running at http://{HOST}:{PORT}")
    print("Press Ctrl+C to stop.")
    server.serve_forever()


if __name__ == "__main__":
    main()
