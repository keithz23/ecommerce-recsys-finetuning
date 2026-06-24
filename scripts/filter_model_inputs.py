from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Optional, Set


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_INPUT = PROJECT_ROOT / "data" / "input"
DATA_OUTPUT = PROJECT_ROOT / "data" / "model_input"

TRAIN_FILE = DATA_INPUT / "interaction_train.csv"
TEST_FILE = DATA_INPUT / "interaction_test.csv"
PRODUCTS_FILE = DATA_INPUT / "products.csv"


PROFILES = {
    # KNN-based User-User and Item-Item CF need a deliberately small set.
    # Similarity matrices grow roughly with users^2 or products^2.
    "cf": {
        "max_users": 5_000,
        "max_products": 5_000,
        "min_user_interactions": 3,
        "min_product_interactions": 5,
        "max_train_rows": 250_000,
        "max_test_rows": 50_000,
    },
    # SVD can handle more rows than KNN, but full 18M rows is still heavy on Colab/Kaggle.
    "svd": {
        "max_users": 100_000,
        "max_products": 100_000,
        "min_user_interactions": 2,
        "min_product_interactions": 3,
        "max_train_rows": 1_000_000,
        "max_test_rows": 200_000,
    },
    # Content-based needs product metadata plus user history.
    "content_based": {
        "max_users": 30_000,
        "max_products": 50_000,
        "min_user_interactions": 2,
        "min_product_interactions": 3,
        "max_train_rows": 500_000,
        "max_test_rows": 100_000,
    },
    # Popularity can be larger because training is product aggregation, but evaluation still costs memory/time.
    "popularity": {
        "max_users": 100_000,
        "max_products": 100_000,
        "min_user_interactions": 1,
        "min_product_interactions": 2,
        "max_train_rows": 1_000_000,
        "max_test_rows": 200_000,
    },
}


def require_files() -> None:
    for path in [TRAIN_FILE, TEST_FILE, PRODUCTS_FILE]:
        if not path.exists():
            raise FileNotFoundError(f"Required input file not found: {path}")


def count_train_entities() -> tuple[Counter[str], Counter[str], int]:
    user_counts: Counter[str] = Counter()
    product_counts: Counter[str] = Counter()
    rows = 0

    with TRAIN_FILE.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows += 1
            user_id = (row.get("UserId") or "").strip()
            product_id = (row.get("ProductId") or "").strip()
            if user_id:
                user_counts[user_id] += 1
            if product_id:
                product_counts[product_id] += 1
            if rows % 1_000_000 == 0:
                print(f"Counted train rows: {rows:,}", flush=True)

    return user_counts, product_counts, rows


def choose_ids(
    counts: Counter[str],
    min_count: int,
    max_ids: int,
) -> Set[str]:
    chosen = [
        key
        for key, count in counts.most_common()
        if count >= min_count and key
    ]
    return set(chosen[:max_ids])


def open_writers(profile_names: Iterable[str], split: str):
    handles = {}
    writers = {}
    for profile in profile_names:
        out_dir = DATA_OUTPUT / profile
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"interaction_{split}.csv"
        handle = path.open("w", encoding="utf-8", newline="")
        handles[profile] = handle
        writers[profile] = None
    return handles, writers


def close_handles(handles) -> None:
    for handle in handles.values():
        handle.close()


def filter_interactions(
    source_file: Path,
    split: str,
    selected_users: Dict[str, Set[str]],
    selected_products: Dict[str, Set[str]],
    product_ids_written: Dict[str, Set[str]],
) -> Dict[str, int]:
    profile_names = list(PROFILES.keys())
    written_counts = {profile: 0 for profile in profile_names}
    max_rows = {
        profile: PROFILES[profile][f"max_{split}_rows"]
        for profile in profile_names
    }

    handles, writers = open_writers(profile_names, split)
    try:
        with source_file.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for profile in profile_names:
                writers[profile] = csv.DictWriter(handles[profile], fieldnames=reader.fieldnames)
                writers[profile].writeheader()

            scanned = 0
            for row in reader:
                scanned += 1
                user_id = (row.get("UserId") or "").strip()
                product_id = (row.get("ProductId") or "").strip()

                for profile in profile_names:
                    if written_counts[profile] >= max_rows[profile]:
                        continue
                    if user_id not in selected_users[profile]:
                        continue
                    if product_id not in selected_products[profile]:
                        continue

                    writers[profile].writerow(row)
                    written_counts[profile] += 1
                    product_ids_written[profile].add(product_id)

                if scanned % 1_000_000 == 0:
                    status = ", ".join(
                        f"{profile}:{written_counts[profile]:,}"
                        for profile in profile_names
                    )
                    print(f"Scanned {split} rows: {scanned:,} | {status}", flush=True)

                if all(written_counts[p] >= max_rows[p] for p in profile_names):
                    break
    finally:
        close_handles(handles)

    return written_counts


def parse_float(value: Optional[str], default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except ValueError:
        return default


def parse_date(value: Optional[str]) -> str:
    if not value:
        return ""
    return value


def build_popularity_train_dataset() -> int:
    source = DATA_OUTPUT / "popularity" / "interaction_train.csv"
    target = DATA_OUTPUT / "popularity" / "popularity_train_dataset.csv"

    stats = defaultdict(
        lambda: {
            "ViewCount": 0,
            "CartCount": 0,
            "PurchaseCount": 0.0,
            "RatingSum": 0.0,
            "RatingCount": 0,
            "LastTimestamp": float("-inf"),
            "LastDate": "",
        }
    )

    with source.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            product_id = (row.get("ProductId") or "").strip()
            if not product_id:
                continue

            rating = parse_float(row.get("Rating"), default=0.0)
            timestamp = parse_float(row.get("Timestamp"), default=float("-inf"))
            verified_purchase = parse_float(row.get("VerifiedPurchase"), default=0.0)
            date_value = parse_date(row.get("Date"))

            item = stats[product_id]
            item["ViewCount"] += 1
            item["PurchaseCount"] += verified_purchase
            item["RatingSum"] += rating
            item["RatingCount"] += 1
            if timestamp > item["LastTimestamp"]:
                item["LastTimestamp"] = timestamp
            if date_value > item["LastDate"]:
                item["LastDate"] = date_value

    fieldnames = [
        "ProductId",
        "ViewCount",
        "CartCount",
        "PurchaseCount",
        "AvgRating",
        "RatingCount",
        "LastTimestamp",
        "LastDate",
    ]

    with target.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for product_id, item in stats.items():
            rating_count = item["RatingCount"]
            writer.writerow(
                {
                    "ProductId": product_id,
                    "ViewCount": item["ViewCount"],
                    "CartCount": item["CartCount"],
                    "PurchaseCount": item["PurchaseCount"],
                    "AvgRating": item["RatingSum"] / rating_count if rating_count else "",
                    "RatingCount": rating_count,
                    "LastTimestamp": "" if item["LastTimestamp"] == float("-inf") else item["LastTimestamp"],
                    "LastDate": item["LastDate"],
                }
            )

    return len(stats)


def filter_products(product_ids: Set[str]) -> int:
    target_dir = DATA_OUTPUT / "content_based"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / "products.csv"
    written = 0

    with PRODUCTS_FILE.open("r", encoding="utf-8", newline="") as src, target.open(
        "w", encoding="utf-8", newline=""
    ) as dst:
        reader = csv.DictReader(src)
        writer = csv.DictWriter(dst, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            product_id = (row.get("ProductId") or "").strip()
            if product_id in product_ids:
                writer.writerow(row)
                written += 1
                if written % 50_000 == 0:
                    print(f"Filtered products: {written:,}", flush=True)

    return written


def write_summary(summary: dict) -> None:
    DATA_OUTPUT.mkdir(parents=True, exist_ok=True)
    summary_path = DATA_OUTPUT / "filter_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    markdown_path = DATA_OUTPUT / "README.md"
    lines = [
        "# Filtered Model Inputs",
        "",
        f"Generated at: {summary['generated_at']}",
        "",
        "| Model | Train file | Test file | Train rows | Test rows |",
        "|---|---|---|---:|---:|",
    ]
    for profile in PROFILES:
        lines.append(
            "| {profile} | `{train}` | `{test}` | {train_rows:,} | {test_rows:,} |".format(
                profile=profile,
                train=f"data/model_input/{profile}/interaction_train.csv",
                test=f"data/model_input/{profile}/interaction_test.csv",
                train_rows=summary["profiles"][profile]["train_rows"],
                test_rows=summary["profiles"][profile]["test_rows"],
            )
        )
    lines.extend(
        [
            "",
            "Additional files:",
            "",
            "- `data/model_input/popularity/popularity_train_dataset.csv`",
            "- `data/model_input/content_based/products.csv`",
        ]
    )
    markdown_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    require_files()
    DATA_OUTPUT.mkdir(parents=True, exist_ok=True)

    print("Counting train users/products...", flush=True)
    user_counts, product_counts, train_total_rows = count_train_entities()

    selected_users = {}
    selected_products = {}
    for profile, config in PROFILES.items():
        selected_users[profile] = choose_ids(
            user_counts,
            config["min_user_interactions"],
            config["max_users"],
        )
        selected_products[profile] = choose_ids(
            product_counts,
            config["min_product_interactions"],
            config["max_products"],
        )
        print(
            f"{profile}: selected {len(selected_users[profile]):,} users, "
            f"{len(selected_products[profile]):,} products",
            flush=True,
        )

    product_ids_written = {profile: set() for profile in PROFILES}

    print("Writing filtered train files...", flush=True)
    train_written = filter_interactions(
        TRAIN_FILE,
        "train",
        selected_users,
        selected_products,
        product_ids_written,
    )

    print("Writing filtered test files...", flush=True)
    test_written = filter_interactions(
        TEST_FILE,
        "test",
        selected_users,
        selected_products,
        product_ids_written,
    )

    print("Building popularity_train_dataset.csv...", flush=True)
    popularity_products = build_popularity_train_dataset()

    print("Filtering products.csv for content-based model...", flush=True)
    content_products = filter_products(product_ids_written["content_based"])

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source_files": {
            "train": str(TRAIN_FILE.relative_to(PROJECT_ROOT)),
            "test": str(TEST_FILE.relative_to(PROJECT_ROOT)),
            "products": str(PRODUCTS_FILE.relative_to(PROJECT_ROOT)),
            "train_total_rows": train_total_rows,
        },
        "profiles": {},
        "additional_outputs": {
            "popularity_products": popularity_products,
            "content_products": content_products,
        },
    }

    for profile in PROFILES:
        summary["profiles"][profile] = {
            **PROFILES[profile],
            "selected_users": len(selected_users[profile]),
            "selected_products": len(selected_products[profile]),
            "train_rows": train_written[profile],
            "test_rows": test_written[profile],
            "written_product_ids": len(product_ids_written[profile]),
        }

    write_summary(summary)
    print("Done. Summary written to data/model_input/filter_summary.json", flush=True)


if __name__ == "__main__":
    main()
