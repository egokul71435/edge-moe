"""
tokenizer.py

Byte-Pair Encoding (BPE) tokenizer built from scratch.
No tiktoken, no HuggingFace tokenizers — full control over every step.

Architecture:
  - Base vocab: 256 UTF-8 bytes (IDs 0–255)
  - Special tokens appended after base vocab (IDs 256–259)
  - Learned merges appended after special tokens (IDs 260+)
  - Pre-tokenization splits text using GPT-2 regex before BPE runs

Usage:
    tok = BPETokenizer()
    tok.train(text, vocab_size=32768)
    tok.save("data/tokenizer.json")

    tok2 = BPETokenizer.load("data/tokenizer.json")
    ids = tok2.encode("hello world")
    text = tok2.decode(ids)
"""

import re
import json
import collections
from pathlib import Path


# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------

# GPT-2 pre-tokenization regex — splits on whitespace, punctuation, contractions.
# We use this verbatim; it's a standard tool, not a research contribution.
GPT2_PATTERN = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?[a-zA-Z]+| ?[0-9]+| ?[^\s\w]+|\s+(?!\S)|\s+""",
    re.UNICODE,
)

SPECIAL_TOKENS = {
    "<PAD>": 256,
    "<BOS>": 257,
    "<EOS>": 258,
    "<UNK>": 259,
}

BASE_VOCAB_SIZE = 256
SPECIAL_VOCAB_SIZE = len(SPECIAL_TOKENS)
MERGES_START_ID = BASE_VOCAB_SIZE + SPECIAL_VOCAB_SIZE  # 260


# ---------------------------------------------------------------------------
# BPETokenizer
# ---------------------------------------------------------------------------

class BPETokenizer:
    """
    Byte-Pair Encoding tokenizer.

    Attributes:
        vocab       : dict[int, bytes]  — ID → byte sequence
        merges      : dict[tuple[int,int], int]  — pair → merged ID
        str_to_id   : dict[str, int]  — special token string → ID
    """

    def __init__(self) -> None:
        # base vocab: each byte is its own token
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(BASE_VOCAB_SIZE)}

        # special tokens
        self.str_to_id: dict[str, int] = dict(SPECIAL_TOKENS)
        self.id_to_str: dict[int, str] = {v: k for k, v in SPECIAL_TOKENS.items()}
        for token, idx in SPECIAL_TOKENS.items():
            self.vocab[idx] = token.encode("utf-8")

        # learned merges: ordered dict preserves insertion order (Python 3.7+)
        self.merges: dict[tuple[int, int], int] = {}

    # -----------------------------------------------------------------------
    # pre-tokenization
    # -----------------------------------------------------------------------

    def _pre_tokenize(self, text: str) -> list[list[int]]:
        """
        Split text into word-level chunks using GPT-2 regex,
        then convert each chunk to a list of byte IDs.
        Returns a list of chunks, each chunk is a list of byte-level token IDs.
        """
        chunks = []
        for match in GPT2_PATTERN.finditer(text):
            word = match.group()
            byte_ids = list(word.encode("utf-8"))  # each byte → int in [0, 255]
            chunks.append(byte_ids)
        return chunks

    # -----------------------------------------------------------------------
    # training
    # -----------------------------------------------------------------------

    def _count_pairs(
        self, chunks: list[list[int]]
    ) -> collections.Counter:
        """Count all adjacent pairs across all chunks."""
        counts: collections.Counter = collections.Counter()
        for chunk in chunks:
            for pair in zip(chunk, chunk[1:]):
                counts[pair] += 1
        return counts

    def _merge_pair(
        self,
        chunks: list[list[int]],
        pair: tuple[int, int],
        new_id: int,
    ) -> list[list[int]]:
        """
        Replace every occurrence of `pair` in all chunks with `new_id`.
        Returns new chunks. O(total tokens) per merge step.
        """
        a, b = pair
        new_chunks = []
        for chunk in chunks:
            new_chunk = []
            i = 0
            while i < len(chunk):
                if i < len(chunk) - 1 and chunk[i] == a and chunk[i + 1] == b:
                    new_chunk.append(new_id)
                    i += 2
                else:
                    new_chunk.append(chunk[i])
                    i += 1
            new_chunks.append(new_chunk)
        return new_chunks

    def train(self, text: str, vocab_size: int = 32768, verbose: bool = True) -> None:
        """
        Train BPE on `text` until vocab reaches `vocab_size`.

        Args:
            text        : raw training text (str)
            vocab_size  : target vocabulary size (must be > 260)
            verbose     : print merge progress
        """
        assert vocab_size > MERGES_START_ID, (
            f"vocab_size must be > {MERGES_START_ID} "
            f"(256 bytes + {SPECIAL_VOCAB_SIZE} special tokens)"
        )

        num_merges = vocab_size - MERGES_START_ID
        if verbose:
            print(f"training BPE: vocab_size={vocab_size}, num_merges={num_merges}")

        # pre-tokenize into byte-level chunks
        chunks = self._pre_tokenize(text)
        if verbose:
            total_tokens = sum(len(c) for c in chunks)
            print(f"  pre-tokenized: {len(chunks):,} chunks, {total_tokens:,} tokens")

        for merge_idx in range(num_merges):
            counts = self._count_pairs(chunks)
            if not counts:
                print(f"  no more pairs to merge at step {merge_idx}, stopping early.")
                break

            # most frequent pair
            best_pair = max(counts, key=lambda p: counts[p])
            new_id = MERGES_START_ID + merge_idx

            # record merge
            self.merges[best_pair] = new_id
            self.vocab[new_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]

            # apply merge to all chunks
            chunks = self._merge_pair(chunks, best_pair, new_id)

            if verbose and (merge_idx % 500 == 0 or merge_idx == num_merges - 1):
                merged_str = self.vocab[new_id]
                print(
                    f"  merge {merge_idx+1:>6}/{num_merges}: "
                    f"{best_pair} → {new_id} "
                    f"(freq={counts[best_pair]:,}, "
                    f"token={merged_str!r})"
                )

        if verbose:
            print(f"done. vocab size: {len(self.vocab):,}")

    # -----------------------------------------------------------------------
    # encoding
    # -----------------------------------------------------------------------

    def _encode_chunk(self, byte_ids: list[int]) -> list[int]:
        """Apply learned merges to a single pre-tokenized chunk."""
        # iteratively apply merges in training order
        while len(byte_ids) >= 2:
            # find the merge with the lowest ID (earliest learned) among all pairs
            pairs = list(zip(byte_ids, byte_ids[1:]))
            best = min(pairs, key=lambda p: self.merges.get(p, float("inf")))
            if best not in self.merges:
                break  # no more applicable merges
            new_id = self.merges[best]
            byte_ids = self._merge_pair([byte_ids], best, new_id)[0]
        return byte_ids

    def encode(self, text: str, add_special: bool = False) -> list[int]:
        """
        Encode a string to a list of token IDs.

        Args:
            text        : input string
            add_special : if True, prepend BOS and append EOS
        """
        chunks = self._pre_tokenize(text)
        ids = []
        for chunk in chunks:
            ids.extend(self._encode_chunk(chunk))

        if add_special:
            ids = [self.str_to_id["<BOS>"]] + ids + [self.str_to_id["<EOS>"]]

        return ids

    def encode_batch(self, texts: list[str], add_special: bool = False) -> list[list[int]]:
        """Encode a list of strings."""
        return [self.encode(t, add_special=add_special) for t in texts]

    # -----------------------------------------------------------------------
    # decoding
    # -----------------------------------------------------------------------

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        """
        Decode a list of token IDs back to a string.

        Args:
            ids          : list of token IDs
            skip_special : if True, skip special token IDs
        """
        byte_parts = []
        for idx in ids:
            if idx in self.id_to_str:
                if not skip_special:
                    byte_parts.append(self.id_to_str[idx].encode("utf-8"))
                # else skip
            elif idx in self.vocab:
                byte_parts.append(self.vocab[idx])
            else:
                byte_parts.append(self.vocab[self.str_to_id["<UNK>"]])

        return b"".join(byte_parts).decode("utf-8", errors="replace")

    # -----------------------------------------------------------------------
    # save / load
    # -----------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """
        Save tokenizer to a JSON file.
        Merges saved as a list of [a, b] pairs in training order.
        Vocab saved as list of [id, base64-encoded bytes] pairs.
        """
        import base64

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # encode bytes values as base64 strings for JSON compatibility
        vocab_serialized = {
            str(k): base64.b64encode(v).decode("ascii")
            for k, v in self.vocab.items()
        }

        # merges as ordered list of pairs
        merges_serialized = [list(pair) for pair in self.merges.keys()]

        data = {
            "vocab_size": len(self.vocab),
            "special_tokens": self.str_to_id,
            "vocab": vocab_serialized,
            "merges": merges_serialized,
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        print(f"tokenizer saved → {path}  ({len(self.vocab):,} tokens)")

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        """Load a tokenizer from a saved JSON file."""
        import base64

        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        tok = cls.__new__(cls)

        # restore vocab
        tok.vocab = {
            int(k): base64.b64decode(v)
            for k, v in data["vocab"].items()
        }

        # restore special tokens
        tok.str_to_id = data["special_tokens"]
        tok.id_to_str = {v: k for k, v in tok.str_to_id.items()}

        # restore merges in training order → dict preserves insertion order
        tok.merges = {}
        for merge_idx, pair in enumerate(data["merges"]):
            a, b = pair
            new_id = MERGES_START_ID + merge_idx
            tok.merges[(a, b)] = new_id

        return tok

    # -----------------------------------------------------------------------
    # properties
    # -----------------------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def __repr__(self) -> str:
        return f"BPETokenizer(vocab_size={self.vocab_size:,}, merges={len(self.merges):,})"


# ---------------------------------------------------------------------------
# quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sample = (
        "Hello world! This is a test of the BPE tokenizer. "
        "It should handle punctuation, numbers like 42, and contractions like don't and it's. "
        "Repeated repeated repeated words should merge quickly. "
    ) * 20  # repeat to give BPE enough signal

    tok = BPETokenizer()
    tok.train(sample, vocab_size=300, verbose=True)

    test = "Hello world! don't repeat yourself."
    ids = tok.encode(test)
    decoded = tok.decode(ids)

    print(f"\noriginal : {test!r}")
    print(f"ids      : {ids}")
    print(f"decoded  : {decoded!r}")
    print(f"roundtrip: {test == decoded}")
    print(f"\ntokenizer: {tok}")