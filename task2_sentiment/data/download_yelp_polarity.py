"""
Shared raw-data download for Task 2 (Yelp Polarity sentiment classification).
Downloads the full Yelp Polarity dataset (fancyzhx/yelp_polarity) from Hugging Face and
caches it as parquet so each member can independently subsample/split/preprocess it in
their own notebook (per the lab spec, preprocessing output is never shared).

Run once from the team repo root:
    python task2_sentiment/data/download_yelp_polarity.py
"""
import os

from datasets import load_dataset

OUT_DIR = os.path.dirname(__file__)
TRAIN_PATH = os.path.join(OUT_DIR, "yelp_polarity_train.parquet")
TEST_PATH = os.path.join(OUT_DIR, "yelp_polarity_test.parquet")


def main():
    if os.path.exists(TRAIN_PATH) and os.path.exists(TEST_PATH):
        print("[skip] Yelp Polarity parquet files already exist")
        return

    print("Downloading Yelp Polarity from Hugging Face (fancyzhx/yelp_polarity)...")
    ds = load_dataset("fancyzhx/yelp_polarity")
    ds["train"].to_parquet(TRAIN_PATH)
    ds["test"].to_parquet(TEST_PATH)
    print(f"train rows: {len(ds['train'])} -> {TRAIN_PATH}")
    print(f"test rows:  {len(ds['test'])} -> {TEST_PATH}")


if __name__ == "__main__":
    main()
