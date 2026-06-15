# tokenizer walk-through

## high-level overview

Builds a Byte-Pair Encoding (BPE) tokenizer from scratch on top of a Rust core,
exposed to Python via PyO3/maturin bindings. Trained on a sample of FineWeb-edu
(a high-quality filtered web corpus from HuggingFace), then used to encode the
full corpus into binary token arrays for model training.

The tokenizer is framework-agnostic, both PyTorch (MPS) and MLX training runs
read the same `train.bin` / `val.bin` files, ensuring the corpus used is identical across experiments.

### why from scratch?
Full control over vocab structure, merge rules, and special tokens, no black-box
library behavior that could confound the research. The tokenizer is a fixed,
documented artifact of the experiment, not a dependency.

### why Rust?
BPE training on 500M+ tokens is CPU-bound. Rust with Rayon (parallel pair
counting) and a priority queue is 20-50x faster than pure Python. Compiled via
PyO3/maturin and called transparently from Python — nothing outside this
directory knows it's Rust underneath.

---

## directory structure

```
tokenizer/
│
├── tokenizer.md               ← this file
├── tokenizer.py               ← Python wrapper around Rust core
├── download_corpus.py         ← stream FineWeb-edu → data/raw/
├── train_tokenizer.py         ← train BPE on raw chunks → tokenizer.json
├── encode_corpus.py           ← encode full corpus → train.bin / val.bin
│
├── core/                      ← Rust BPE crate (compiled via maturin)
│   ├── Cargo.toml             ← crate manifest + dependencies
│   └── src/
│       ├── lib.rs             ← PyO3 bindings, exposes BPETokenizer to Python
│       ├── pretokenize.rs     ← Unicode-aware regex pre-tokenization
│       ├── bpe.rs             ← BPE training (priority queue + Rayon parallel)
│       ├── vocab.rs           ← vocab + special token management
│       └── serialize.rs       ← save/load tokenizer.json
│
└── data/
    ├── raw/                   ← raw .txt chunks from download_corpus.py
    │   └── chunk_00000.txt ... chunk_NNNNN.txt
    ├── tokenizer/             ← trained tokenizer artifact
    │   └── tokenizer.json     ← vocab (base64 bytes) + ordered merge rules
    └── encoded/               ← final binary token arrays for training
        ├── train.bin          ← int32 numpy array, 90% of corpus
        └── val.bin            ← int32 numpy array, 10% of corpus
```

---

## steps to run

run these in order, once each. after `encode_corpus.py` completes, this entire
directory is done — model training reads only from `data/encoded/`.

### 0. environment setup (one-time)

```bash
# install Rust (if not already installed)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env

# create and activate conda env
conda create -n edge-moe python=3.12
conda activate edge-moe

# install Python dependencies
pip install torch torchvision torchaudio
pip install mlx mlx-lm
pip install datasets tqdm regex numpy
pip install maturin

# verify ARM native Python (must print 'arm', not 'i386')
python -c "import platform; print(platform.processor())"
```

### 1. HuggingFace login (one-time)

FineWeb-edu is a gated dataset. You need to:
1. Accept terms at https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
2. Generate a Read token at https://huggingface.co/settings/tokens

```bash
pip install huggingface_hub
huggingface-cli login
# paste your Read token when prompted
```

### 2. download corpus

streams FineWeb-edu and saves raw text chunks to `data/raw/`.
safe to interrupt and resume — resumes from last complete chunk automatically.

```bash
# tokenizer training sample (5% ~ 570M tokens, ~2.3GB, ~19 min)
python tokenizer/download_corpus.py --sample_frac 0.05

# larger pull for model training encode pass (~20%, done separately later)
python tokenizer/download_corpus.py --sample_frac 0.20
```

key args:
- `--sample_frac` : fraction of corpus to keep (0.0-1.0)
- `--max_chunks`  : hard stop after N chunk files (dev/debug only)
- `--seed`        : RNG seed for reproducibility (default: 42)
- `--min_chars`   : minimum doc length in chars (default: 200)

### 3. compile Rust tokenizer core (one-time)

```bash
cd tokenizer/core
maturin develop --release   # --release is critical, debug builds are ~50x slower
cd ../..
```

verify it compiled:
```bash
python -c "from core import BPETokenizer; print('ok')"
```

### 4. train tokenizer

reads all chunks from `data/raw/`, trains BPE, saves `data/tokenizer/tokenizer.json`.
run once. takes 1-5 minutes with Rust core.

```bash
python tokenizer/train_tokenizer.py
```

output: `data/tokenizer/tokenizer.json` (~few MB)
this file is committed to git — it's the key reproducibility artifact.

### 5. encode corpus

encodes all raw chunks to binary token arrays. run once after tokenizer training.
takes 10-30 minutes depending on corpus size.

```bash
python tokenizer/encode_corpus.py
```

output:
- `data/encoded/train.bin` — 90% split, int32 numpy array (~1-2GB)
- `data/encoded/val.bin`   — 10% split, int32 numpy array (~100-200MB)

after this step, `data/raw/` can be deleted to save disk space if needed.
`train.bin` and `val.bin` are the only files model training needs.

---

## in-detail spec

### vocabulary structure

```
IDs   0 –   255 : raw byte tokens (base vocab — every possible UTF-8 byte)
IDs 256 –   511 : byte fallback tokens <0x00>–<0xFF> (lossless non-UTF-8 handling)
IDs 512 –   516 : special tokens: <PAD> <BOS> <EOS> <UNK> <SEP>
IDs 517 –   527 : reserved: <UNUSED_0>–<UNUSED_10> (future-proofing)
IDs 528 – 50303 : learned BPE merges (49,776 merges)
─────────────────
total vocab size : 50,304  (multiple of 64 — hardware-friendly matmul alignment)
```

### pre-tokenization

uses an upgraded version of GPT-2's regex pattern via Rust's `regex` crate,
which supports Unicode properties (`\p{L}`, `\p{N}`) unlike Python's `re`:

- contractions (`'s`, `'t`, `'re`, etc.) treated as atomic units
- digits split individually — prevents pathological number tokenizations
- emoji and Unicode symbols treated as atomic units
- optional whitespace prefix attached to following word (GPT-2 style)
- non-Latin scripts (CJK, Arabic, etc.) handled correctly via Unicode categories

pre-tokenization runs before BPE — BPE never merges across word boundaries.

### BPE training algorithm

1. pre-tokenize corpus into word-level chunks of byte IDs
2. count all adjacent pairs across all chunks (parallelized via Rayon)
3. find most frequent pair via BinaryHeap priority queue — O(log n)
4. merge pair everywhere using doubly-linked list representation — O(n) per merge
5. repeat for `num_merges = 49,776` steps
6. save merge rules in training order — order is semantically meaningful for encoding

### encoding

to encode a string:
1. pre-tokenize into chunks
2. for each chunk: replay learned merges in training order via priority queue
3. return flat list of token IDs

roundtrip guarantee: `decode(encode(text)) == text` for all valid UTF-8 input.
non-UTF-8 bytes are handled losslessly via byte fallback tokens.

### data flow

```
FineWeb-edu stream (HuggingFace)
        ↓
download_corpus.py
        ↓
data/raw/chunk_NNNNN.txt     ← intermediate, can delete after encoding
        ↓
train_tokenizer.py  →  data/tokenizer/tokenizer.json  ← commit this
        ↓
encode_corpus.py
        ↓
data/encoded/train.bin       ← model training reads these
data/encoded/val.bin         ← model training reads these
```

---

## corpus details

| property | value |
|---|---|
| dataset | HuggingFaceFW/fineweb-edu (sample-10BT) |
| tokenizer training sample | 5% (~483k docs, ~570M tokens, ~2.3GB) |
| model training sample | ~20% (separate download, done later) |
| doc length filter | min 200 chars |
| sampling seed | 42 |
| chunks | 49 files × 10,000 docs each + 1 remainder |

---

## required tools and resources

| tool | version | install |
|---|---|---|
| Python | 3.12 (ARM native) | `conda create -n edge-moe python=3.12` |
| Rust + Cargo | stable | `curl ... sh.rustup.rs` |
| maturin | ≥1.5.0 | `pip install maturin` |
| PyTorch | ≥2.3.0 | `pip install torch torchvision torchaudio` |
| MLX | ≥0.14.0 | `pip install mlx mlx-lm` |
| datasets | ≥2.19.0 | `pip install datasets` |
| tqdm | ≥4.66.0 | `pip install tqdm` |
| regex | ≥2024.4.0 | `pip install regex` |
| numpy | ≥1.26.0 | `pip install numpy` |
| huggingface_hub | any | `pip install huggingface_hub` |

hardware requirements:
- Apple Silicon Mac (M1/M2/M3) — required for MPS and MLX
- ≥16GB unified memory recommended
- ≥10GB free disk space for corpus + encoded files

HuggingFace access:
- account at https://huggingface.co
- dataset access approved at https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
- Read token generated at https://huggingface.co/settings/tokens
- logged in via `huggingface-cli login`




# archive: 





# tokenizer walk-through

## high-level overview

## steps-to run

python tokenizer/download_corpus.py --sample_frac 0.05

## in-detail spec

## required tools, resources

- hf auth login steps