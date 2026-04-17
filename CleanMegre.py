import json
import pandas as pd
from collections import defaultdict
from pathlib import Path

DATA_DIR = Path(".")
OUTPUT_DIR = Path("./processed")
OUTPUT_DIR.mkdir(exist_ok=True)

CATEGORIES = [
    "All_Beauty",
    "Amazon_Fashion",
    "Appliances",
    "Arts_Crafts_and_Sewing"
]

MIN_USER_INTERACTIONS = 5
MIN_ITEM_INTERACTIONS = 5
MAX_SEQ_LEN = 50


#FILTER REVIEWS
def filter_reviews(category):
    path = DATA_DIR / f"{category}.jsonl"

    rows = []
    user_counts = defaultdict(int)
    item_counts = defaultdict(int)

    # count
    with open(path, "r") as f:
        for line in f:
            obj = json.loads(line)
            if obj.get("verified_purchase", False):
                user_counts[obj["user_id"]] += 1
                item_counts[obj["parent_asin"]] += 1

    # filter
    with open(path, "r") as f:
        for line in f:
            obj = json.loads(line)

            if not obj.get("verified_purchase", False):
                continue

            u = obj["user_id"]
            p = obj["parent_asin"]

            if (
                user_counts[u] >= MIN_USER_INTERACTIONS
                and item_counts[p] >= MIN_ITEM_INTERACTIONS
            ):
                rows.append({
                    "user_id": u,
                    "parent_asin": p,
                    "rating": obj.get("rating"),
                    "timestamp": obj.get("timestamp", 0)
                })

    return pd.DataFrame(rows)


#FILTER METADATA
def filter_metadata(category, valid_items):
    path = DATA_DIR / f"meta_{category}.jsonl"

    rows = []

    with open(path, "r") as f:
        for line in f:
            obj = json.loads(line)

            p = obj.get("parent_asin")

            if p in valid_items:
                rows.append({
                    "parent_asin": p,
                    "title": obj.get("title", "") or "",
                    "store": obj.get("store", "") or "",
                    "main_category": obj.get("main_category", "") or ""
                })

    return pd.DataFrame(rows)


#MERGE
def merge_data(reviews_df, meta_df):

    df = reviews_df.merge(meta_df, on="parent_asin", how="left")

    # BUILD TEXT FEATURE
    df["text"] = (
        df["title"].fillna("") + " " +
        df["store"].fillna("") + " " +
        df["main_category"].fillna("")
    )

    df = df.sort_values("timestamp")

    return df[[
        "user_id",
        "parent_asin",
        "text",
        "timestamp"
    ]]


#CREATE SPLITS
def build_splits(df):

    train_rows = []
    val_rows = []
    test_rows = []

    for user_id, g in df.groupby("user_id"):

        g = g.sort_values("timestamp")

        items = g["parent_asin"].tolist()
        texts = dict(zip(g["parent_asin"], g["text"]))

        if len(items) < 3:
            continue

        # TRAIN
        for item in items[:-2]:
            train_rows.append({
                "user_id": user_id,
                "parent_asin": item,
                "text": texts.get(item, "")
            })

        # VAL
        val_rows.append({
            "user_id": user_id,
            "history": " ".join(items[:-2][-MAX_SEQ_LEN:]),
            "target": items[-2]
        })

        # TEST
        test_rows.append({
            "user_id": user_id,
            "history": " ".join(items[:-1][-MAX_SEQ_LEN:]),
            "target": items[-1]
        })

    return (
        pd.DataFrame(train_rows),
        pd.DataFrame(val_rows),
        pd.DataFrame(test_rows)
    )



def main():

    all_reviews = []
    all_meta = []

    for cat in CATEGORIES:
        print(f"\nProcessing {cat}")

        reviews = filter_reviews(cat)
        items = set(reviews["parent_asin"].unique())

        meta = filter_metadata(cat, items)

        all_reviews.append(reviews)
        all_meta.append(meta)

    reviews_df = pd.concat(all_reviews, ignore_index=True)
    meta_df = pd.concat(all_meta, ignore_index=True)

    print("\nMerging...")
    df = merge_data(reviews_df, meta_df)

    print("Building splits...")
    train, val, test = build_splits(df)

    # SAVE
    train.to_parquet(OUTPUT_DIR / "train.parquet")
    val.to_parquet(OUTPUT_DIR / "val.parquet")
    test.to_parquet(OUTPUT_DIR / "test.parquet")

    print("\nSaved:")
    print("Train:", train.shape)
    print("Val:", val.shape)
    print("Test:", test.shape)

#MAKES IT SO WE DO NOT RUN ON IMPORT
if __name__ == "__main__":
    main()