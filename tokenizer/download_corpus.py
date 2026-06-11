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

import argparse # cli argument parsing
import random   # sampling
from pathlib import Path # filesystem paths
from datasets import load_dataset # streaming dataset access
from tqdm import tqdm # progress bar

"""
config --- adjust as needed, defaults work for tokenizer training sample

raw dir: where to save .txt chunks (created if missing)
chunk size: how many documents per .txt chunk file (adjust based on doc length and RAM)
min doc chars: filter out docs shorter than this (removes very short docs that add overhead without many tokens)
dataset name/split/cfg: which HuggingFace dataset to stream (see https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu for options)
text field: which field in the dataset contains the raw text to save
"""

RAW_DIR       = Path(__file__).parent / "data" / "raw"
CHUNK_SIZE    = 10_000         # documents per output .txt file
MIN_DOC_CHARS = 200            # discard docs shorter than this
DATASET_NAME  = "HuggingFaceFW/fineweb-edu"
DATASET_SPLIT = "train"
DATASET_CFG   = "sample-10BT" # 10B token sample; "default" = full 96B
TEXT_FIELD    = "text"


"""
helpers

chunk_exists: check if a chunk file already exists on disk (for resuming)
write_chunk: save a list of documents to a chunk file on disk
"""

def chunk_exists(chunk_idx: int, out_dir: Path) -> bool:
    return (out_dir / f"chunk_{chunk_idx:05d}.txt").exists()


def write_chunk(docs: list[str], chunk_idx: int, out_dir: Path) -> None:
    out_path = out_dir / f"chunk_{chunk_idx:05d}.txt"
    with open(out_path, "w", encoding="utf-8", errors="replace") as f:
        f.write("\n\n".join(docs))


"""
main

streams the dataset, applies sampling and length filtering, and writes out .txt chunk files.
sampling is done via rng.random() per doc — no shuffle buffer needed, avoids iterator stalls.
keeps track of how many docs seen, kept, and total chars for reporting at the end.
"""

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
        print(f"resuming from chunk {resume_from} (chunks 0-{resume_from - 1} already exist)")

    print(f"dataset      : {DATASET_NAME} ({DATASET_CFG})")
    print(f"sample_frac  : {sample_frac}")
    print(f"max_chunks   : {max_chunks or 'unlimited'}")
    print(f"seed         : {seed}")
    print(f"min_doc_chars: {min_chars}")
    print(f"output dir   : {RAW_DIR}\n")

    rng = random.Random(seed)

    # no trust_remote_code — deprecated in recent datasets versions
    dataset = load_dataset(
        DATASET_NAME,
        name=DATASET_CFG,
        split=DATASET_SPLIT,
        streaming=True,
    )

    # no shuffle buffer — rng.random() per doc gives uniform random sampling
    # without risking iterator stalls from large buffers on newer datasets versions

    buffer: list[str] = []
    docs_seen      = 0
    docs_kept      = 0
    total_chars    = 0
    chunks_written = 0

    # skip docs that already landed in chunks written in previous runs
    docs_to_skip = resume_from * CHUNK_SIZE

    pbar = tqdm(desc="docs kept", unit=" docs")

    for doc in dataset:
        docs_seen += 1

        # --- sampling: keep each doc independently at rate sample_frac ---
        if sample_frac < 1.0 and rng.random() >= sample_frac:
            continue

        # --- length filter ---
        text = doc[TEXT_FIELD].strip()
        if len(text) < min_chars:
            continue

        # --- resume: skip docs already written in previous runs ---
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

    # flush any remaining docs that didn't fill a full chunk
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


"""
cli

- sample_frac: keep only this fraction of docs (0.0-1.0). Use 0.02 for tokenizer training sample.
- max_chunks: stop after writing this many NEW chunks (dev/debug)
- seed: RNG seed for sampling reproducibility (default: 42)
- min_chars: minimum document character length to filter out very short docs (default: 200)
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download FineWeb-edu corpus chunks.")
    parser.add_argument(
        "--sample_frac", type=float, default=1.0,
        help="Fraction of corpus to keep (0.0-1.0). Use 0.02 for tokenizer training sample.",
    )
    parser.add_argument(
        "--max_chunks", type=int, default=None,
        help="Stop after writing this many NEW chunks (dev/debug).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="RNG seed for sampling (default: 42).",
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