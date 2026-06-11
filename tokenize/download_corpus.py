"""
download_corpus.py

Streams FineWeb-edu from HuggingFace and saves raw text to data/raw/.
Never loads the full dataset into RAM — streams, filters, and writes in chunks.

Usage:
    python download_corpus.py                        # full corpus
    python download_corpus.py --sample_frac 0.02    # 2% random sample (tokenizer training)
    python download_corpus.py --max_chunks 10       # cap chunks (dev/debug)
    python download_corpus.py --seed 1337           # different random sample
"""

import argparse
import random
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------

RAW_DIR       = Path(__file__).parent / "data" / "raw"
CHUNK_SIZE    = 10_000                    # documents per output .txt file
MIN_DOC_CHARS = 200                       # discard docs shorter than this
DATASET_NAME  = "HuggingFaceFW/fineweb-edu"
DATASET_SPLIT = "train"
DATASET_CFG   = "sample-10BT"            # 10B token sample; "default" = full 96B
TEXT_FIELD    = "text"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def chunk_exists(chunk_idx: int, out_dir: Path) -> bool:
    return (out_dir / f"chunk_{chunk_idx:05d}.txt").exists()


def write_chunk(docs: list[str], chunk_idx: int, out_dir: Path) -> None:
    out_path = out_dir / f"chunk_{chunk_idx:05d}.txt"
    with open(out_path, "w", encoding="utf-8", errors="replace") as f:
        f.write("\n\n".join(docs))


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def download(
    sample_frac: float = 1.0,
    max_chunks: int | None = None,
    seed: int = 42,
    min_chars: int = MIN_DOC_CHARS,
) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # find first missing chunk so we can resume interrupted downloads
    chunk_idx = 0
    while chunk_exists(chunk_idx, RAW_DIR):
        chunk_idx += 1
    resume_from = chunk_idx
    if resume_from > 0:
        print(f"resuming from chunk {resume_from} (chunks 0–{resume_from - 1} already exist)")

    print(f"dataset      : {DATASET_NAME} ({DATASET_CFG})")
    print(f"sample_frac  : {sample_frac}")
    print(f"max_chunks   : {max_chunks or 'unlimited'}")
    print(f"seed         : {seed}")
    print(f"min_doc_chars: {min_chars}")
    print(f"output dir   : {RAW_DIR}\n")

    rng = random.Random(seed)

    dataset = load_dataset(
        DATASET_NAME,
        name=DATASET_CFG,
        split=DATASET_SPLIT,
        streaming=True,
        trust_remote_code=True,
    )

    # shuffle within a buffer before sampling — avoids taking only the
    # first N% of the dataset when sample_frac < 1.0
    if sample_frac < 1.0:
        dataset = dataset.shuffle(seed=seed, buffer_size=10_000)

    buffer: list[str] = []
    docs_seen    = 0   # total docs from the stream (including skipped)
    docs_kept    = 0   # docs that passed filter + sample
    total_chars  = 0
    chunks_written = 0

    # skip chunks already on disk
    docs_to_skip = resume_from * CHUNK_SIZE

    pbar = tqdm(desc="docs kept", unit=" docs")

    for doc in dataset:
        docs_seen += 1

        # --- sampling ---
        if sample_frac < 1.0 and rng.random() >= sample_frac:
            continue

        # --- length filter ---
        text = doc[TEXT_FIELD].strip()
        if len(text) < min_chars:
            continue

        # --- resume: skip docs that already landed in written chunks ---
        if docs_to_skip > 0:
            docs_to_skip -= 1
            continue

        buffer.append(text)
        total_chars += len(text)
        docs_kept += 1
        pbar.update(1)

        if len(buffer) >= CHUNK_SIZE:
            write_chunk(buffer, chunk_idx, RAW_DIR)
            chunks_written += 1
            pbar.write(f"  wrote chunk_{chunk_idx:05d}.txt  ({len(buffer):,} docs)")
            chunk_idx += 1
            buffer = []

            if max_chunks is not None and chunks_written >= max_chunks:
                pbar.write(f"\nreached max_chunks={max_chunks}, stopping.")
                break

    pbar.close()

    # flush any remaining docs
    if buffer:
        write_chunk(buffer, chunk_idx, RAW_DIR)
        chunks_written += 1
        print(f"  wrote chunk_{chunk_idx:05d}.txt  ({len(buffer):,} docs)")

    print(f"\ndone.")
    print(f"  chunks written : {chunks_written + resume_from:,}  ({resume_from} pre-existing + {chunks_written} new)")
    print(f"  docs kept      : {docs_kept:,}")
    print(f"  docs seen      : {docs_seen:,}")
    print(f"  total chars    : {total_chars:,}")
    print(f"  approx tokens  : ~{total_chars // 4:,}  (rough 4 chars/token estimate)")


# ---------------------------------------------------------------------------
# cli
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download FineWeb-edu corpus chunks.")
    parser.add_argument(
        "--sample_frac", type=float, default=1.0,
        help="Fraction of corpus to keep (0.0–1.0). Use 0.02 for tokenizer training sample.",
    )
    parser.add_argument(
        "--max_chunks", type=int, default=None,
        help="Stop after writing this many NEW chunks (dev/debug).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="RNG seed for shuffling and sampling (default: 42).",
    )
    parser.add_argument(
        "--min_chars", type=int, default=MIN_DOC_CHARS,
        help=f"Minimum document character length (default: {MIN_DOC_CHARS}).",
    )
    args = parser.parse_args()

    assert 0.0 < args.sample_frac <= 1.0, "--sample_frac must be in (0, 1]"

    download(
        sample_frac=args.sample_frac,
        max_chunks=args.max_chunks,
        seed=args.seed,
        min_chars=args.min_chars,
    )