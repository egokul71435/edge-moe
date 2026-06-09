"""
download_corpus.py

Streams FineWeb-edu from HuggingFace and saves raw text to data/raw/.
Designed to never load the full dataset into RAM — streams and writes in chunks.

Usage:
    python download_corpus.py --sample_frac 0.01   # 1% for tokenizer training sample
    python download_corpus.py --sample_frac 1.0    # full corpus for encoding
    python download_corpus.py --max_chunks 100     # cap number of chunks (dev/debug)
"""

import os
import argparse
from pathlib import Path
from datasets import load_dataset

# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------

RAW_DIR = Path(__file__).parent / "data" / "raw"
CHUNK_SIZE = 10_000          # documents per output file
DATASET_NAME = "HuggingFaceFW/fineweb-edu"
DATASET_SPLIT = "train"
DATASET_CONFIG = "sample-10BT"  # 10B token sample — use "default" for full 96B
TEXT_FIELD = "text"              # field name in the dataset containing raw text


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def write_chunk(docs: list[str], chunk_idx: int, out_dir: Path) -> None:
    """Write a list of documents to a single .txt file."""
    out_path = out_dir / f"chunk_{chunk_idx:05d}.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        # separate docs with a blank line so boundaries are preserved
        f.write("\n\n".join(docs))
    print(f"  wrote {len(docs):,} docs → {out_path.name}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def download(sample_frac: float = 1.0, max_chunks: int | None = None) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print(f"streaming {DATASET_NAME} ({DATASET_CONFIG}) ...")
    print(f"sample_frac={sample_frac}, max_chunks={max_chunks or 'unlimited'}")
    print(f"output dir: {RAW_DIR}\n")

    dataset = load_dataset(
        DATASET_NAME,
        name=DATASET_CONFIG,
        split=DATASET_SPLIT,
        streaming=True,         # critical — do not remove
        trust_remote_code=True,
    )

    buffer: list[str] = []
    chunk_idx = 0
    total_docs = 0
    total_chars = 0

    for doc in dataset:
        # sample_frac: deterministic skip based on doc index
        # simple modulo sample — good enough for corpus prep
        if sample_frac < 1.0:
            if (total_docs % round(1 / sample_frac)) != 0:
                total_docs += 1
                continue

        text = doc[TEXT_FIELD].strip()
        if not text:
            continue

        buffer.append(text)
        total_chars += len(text)

        if len(buffer) >= CHUNK_SIZE:
            write_chunk(buffer, chunk_idx, RAW_DIR)
            chunk_idx += 1
            buffer = []

            if max_chunks is not None and chunk_idx >= max_chunks:
                print(f"\nreached max_chunks={max_chunks}, stopping.")
                break

        total_docs += 1

    # flush remaining buffer
    if buffer:
        write_chunk(buffer, chunk_idx, RAW_DIR)
        chunk_idx += 1

    print(f"\ndone.")
    print(f"  chunks written : {chunk_idx:,}")
    print(f"  total docs     : {total_docs:,}")
    print(f"  total chars    : {total_chars:,}")
    print(f"  approx tokens  : ~{total_chars // 4:,}  (rough estimate at 4 chars/token)")


# ---------------------------------------------------------------------------
# cli
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download FineWeb-edu corpus chunks.")
    parser.add_argument(
        "--sample_frac",
        type=float,
        default=1.0,
        help="Fraction of corpus to download (0.0–1.0). Use 0.01 for tokenizer training.",
    )
    parser.add_argument(
        "--max_chunks",
        type=int,
        default=None,
        help="Stop after this many chunks (for dev/debug).",
    )
    args = parser.parse_args()

    assert 0.0 < args.sample_frac <= 1.0, "--sample_frac must be in (0, 1]"

    download(sample_frac=args.sample_frac, max_chunks=args.max_chunks)