"""
Shared raw-data download for Task 1 (GPT from scratch).
Downloads TinyStories (roneneldan/TinyStories) from Hugging Face and writes a single
concatenated raw text corpus that each member then slices independently for their own
character-level train/validation split (see each member's src/ notebook).

Run once from the team repo root:
    python task1_llm/data/download_tinystories.py
"""
import os

from datasets import load_dataset

OUT_PATH = os.path.join(os.path.dirname(__file__), "tinystories_raw.txt")
NUM_STORIES = 4000  # enough raw characters (>1M) for every member to draw an independent
                     # non-overlapping 100K-train / 10K-val character slice


def main():
    if os.path.exists(OUT_PATH):
        print(f"[skip] {OUT_PATH} already exists ({os.path.getsize(OUT_PATH)} bytes)")
        return

    print("Downloading TinyStories (streaming, first %d stories)..." % NUM_STORIES)
    ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)

    texts = []
    total_chars = 0
    for i, row in enumerate(ds):
        if i >= NUM_STORIES:
            break
        texts.append(row["text"].strip())
        total_chars += len(row["text"])

    corpus = "\n<STORY_SEP>\n".join(texts)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write(corpus)

    print(f"Wrote {len(corpus)} characters from {len(texts)} stories to {OUT_PATH}")


if __name__ == "__main__":
    main()
