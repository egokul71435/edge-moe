// bpe training, encoding, decoding

// bpe.rs
//
// byte-pair encoding: training and encoding logic.
//
// naive implementation — correctness first, optimization later.
// optimization pass (rayon parallel counting, priority queue) comes
// after all tests pass on the naive version.
//
// training:
//   1. pretokenize corpus into chunks of byte IDs
//   2. count all adjacent pairs across all chunks
//   3. find most frequent pair
//   4. merge it everywhere, record merge rule
//   5. repeat for NUM_MERGES steps
//
// encoding:
//   1. pretokenize input into chunks of byte IDs
//   2. for each chunk: apply merges in training order
//   3. flatten chunks to single Vec<u32>

use std::collections::HashMap;
use crate::vocab::{Vocab, NUM_MERGES, BOS_ID, EOS_ID, MERGES_START};
use crate::pretokenize::pretokenize_to_ids;

// ---------------------------------------------------------------------------
// types
// ---------------------------------------------------------------------------

/// merge rules: (left_id, right_id) → new_id
/// stored in training order — encoding replays from lowest new_id upward
pub type MergeMap = HashMap<(u32, u32), u32>;

// ---------------------------------------------------------------------------
// pair counting
// ---------------------------------------------------------------------------

/// count all adjacent pairs across all chunks.
fn count_pairs(chunks: &[Vec<u32>]) -> HashMap<(u32, u32), u64> {
    let mut counts: HashMap<(u32, u32), u64> = HashMap::new();
    for chunk in chunks {
        for pair in chunk.windows(2) {
            *counts.entry((pair[0], pair[1])).or_insert(0) += 1;
        }
    }
    counts
}

/// find the most frequent pair.
/// returns None if no pairs exist (all chunks are length 1).
fn best_pair(counts: &HashMap<(u32, u32), u64>) -> Option<(u32, u32)> {
    counts
        .iter()
        .max_by_key(|(_, &freq)| freq)
        .map(|(&pair, _)| pair)
}

// ---------------------------------------------------------------------------
// merge application
// ---------------------------------------------------------------------------

/// replace every occurrence of `pair` in all chunks with `new_id`.
/// O(total tokens) per merge step.
fn apply_merge(chunks: &mut Vec<Vec<u32>>, pair: (u32, u32), new_id: u32) {
    let (a, b) = pair;
    for chunk in chunks.iter_mut() {
        let mut new_chunk: Vec<u32> = Vec::with_capacity(chunk.len());
        let mut i = 0;
        while i < chunk.len() {
            if i + 1 < chunk.len() && chunk[i] == a && chunk[i + 1] == b {
                new_chunk.push(new_id);
                i += 2;
            } else {
                new_chunk.push(chunk[i]);
                i += 1;
            }
        }
        *chunk = new_chunk;
    }
}

// ---------------------------------------------------------------------------
// training
// ---------------------------------------------------------------------------

/// train BPE on `text`.
///
/// runs up to NUM_MERGES (49,776) steps, updating `vocab` with each
/// new token and returning the full merge map.
///
/// `progress_every`: print a status line every N merges (0 = silent)
pub fn train(text: &str, vocab: &mut Vocab, progress_every: usize) -> MergeMap {
    let mut chunks = pretokenize_to_ids(text);
    let total_tokens: usize = chunks.iter().map(|c| c.len()).sum();
    let num_merges = NUM_MERGES as usize;

    if progress_every > 0 {
        println!(
            "bpe training: {} chunks, {} tokens, {} merges to run",
            chunks.len(), total_tokens, num_merges
        );
    }

    let mut merges: MergeMap = HashMap::with_capacity(num_merges);

    for step in 0..num_merges {
        let counts = count_pairs(&chunks);

        if counts.is_empty() {
            if progress_every > 0 {
                println!("  early stop at step {} — no more pairs", step);
            }
            break;
        }

        let pair = match best_pair(&counts) {
            Some(p) => p,
            None => break,
        };

        let new_id = vocab.add_merge(pair.0, pair.1);
        merges.insert(pair, new_id);
        apply_merge(&mut chunks, pair, new_id);

        if progress_every > 0 && (step % progress_every == 0 || step == num_merges - 1) {
            let token_bytes = vocab.get_bytes(new_id);
            let token_repr = String::from_utf8_lossy(token_bytes);
            println!(
                "  step {:>6}/{}: ({},{}) → {} | freq={} | token={:?}",
                step + 1, num_merges, pair.0, pair.1, new_id,
                counts[&pair], token_repr,
            );
        }
    }

    merges
}

// ---------------------------------------------------------------------------
// encoding
// ---------------------------------------------------------------------------

/// encode a single pre-tokenized chunk using learned merge rules.
/// finds the applicable merge with the lowest new_id (training order)
/// and applies it, repeating until no merges remain.
fn encode_chunk(chunk: Vec<u32>, merges: &MergeMap) -> Vec<u32> {
    let mut ids = chunk;

    loop {
        if ids.len() < 2 {
            break;
        }

        // find pair with lowest new_id — earliest learned merge takes priority
        let best = ids
            .windows(2)
            .filter_map(|w| {
                let pair = (w[0], w[1]);
                merges.get(&pair).map(|&new_id| (pair, new_id))
            })
            .min_by_key(|&(_, new_id)| new_id);

        match best {
            None => break,
            Some((pair, new_id)) => {
                let (a, b) = pair;
                let mut new_ids: Vec<u32> = Vec::with_capacity(ids.len());
                let mut i = 0;
                while i < ids.len() {
                    if i + 1 < ids.len() && ids[i] == a && ids[i + 1] == b {
                        new_ids.push(new_id);
                        i += 2;
                    } else {
                        new_ids.push(ids[i]);
                        i += 1;
                    }
                }
                ids = new_ids;
            }
        }
    }

    ids
}

/// encode `text` to a flat list of token IDs.
pub fn encode(text: &str, merges: &MergeMap) -> Vec<u32> {
    pretokenize_to_ids(text)
        .into_iter()
        .flat_map(|chunk| encode_chunk(chunk, merges))
        .collect()
}

/// encode with optional BOS/EOS tokens.
pub fn encode_with_special(
    text: &str,
    merges: &MergeMap,
    add_special: bool,
) -> Vec<u32> {
    let mut ids = encode(text, merges);
    if add_special {
        ids.insert(0, BOS_ID);
        ids.push(EOS_ID);
    }
    ids
}

// ---------------------------------------------------------------------------
// decoding
// ---------------------------------------------------------------------------

/// decode token IDs back to a UTF-8 string.
/// concatenates byte representations from vocab, decodes lossy UTF-8.
pub fn decode(ids: &[u32], vocab: &Vocab, skip_special: bool) -> String {
    let bytes: Vec<u8> = ids
        .iter()
        .filter(|&&id| !skip_special || !vocab.is_special(id))
        .flat_map(|&id| vocab.get_bytes(id).to_vec())
        .collect();
    String::from_utf8_lossy(&bytes).into_owned()
}

// ---------------------------------------------------------------------------
// tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vocab::Vocab;

    /// train on a small repeated corpus, return (vocab, merges)
    fn train_small() -> (Vocab, MergeMap) {
        let mut vocab = Vocab::new();
        let text = "hello world hello world hello world ".repeat(50);
        let merges = train(&text, &mut vocab, 0);
        (vocab, merges)
    }

    #[test]
    fn test_count_pairs_basic() {
        let chunks = vec![vec![1u32, 2, 1, 2, 3]];
        let counts = count_pairs(&chunks);
        assert_eq!(counts[&(1, 2)], 2);
        assert_eq!(counts[&(2, 1)], 1);
        assert_eq!(counts[&(2, 3)], 1);
    }

    #[test]
    fn test_best_pair_picks_most_frequent() {
        let mut counts = HashMap::new();
        counts.insert((1u32, 2u32), 10u64);
        counts.insert((3u32, 4u32), 5u64);
        counts.insert((5u32, 6u32), 20u64);
        assert_eq!(best_pair(&counts), Some((5, 6)));
    }

    #[test]
    fn test_apply_merge_basic() {
        let mut chunks = vec![vec![1u32, 2, 1, 2, 3]];
        apply_merge(&mut chunks, (1, 2), 99);
        assert_eq!(chunks[0], vec![99, 99, 3]);
    }

    #[test]
    fn test_apply_merge_no_overlap() {
        // [1,1,1] with pair (1,1) → [99, 1] not [99, 99] — no overlap
        let mut chunks = vec![vec![1u32, 1, 1]];
        apply_merge(&mut chunks, (1, 1), 99);
        assert_eq!(chunks[0], vec![99, 1]);
    }

    #[test]
    fn test_train_reduces_tokens() {
        let mut vocab = Vocab::new();
        let text = "hello world ".repeat(100);
        let original_len: usize = pretokenize_to_ids(&text)
            .iter().map(|c| c.len()).sum();
        let merges = train(&text, &mut vocab, 0);
        let encoded = encode(&text, &merges);
        assert!(encoded.len() < original_len);
    }

    #[test]
    fn test_train_adds_merges_to_vocab() {
        let (vocab, merges) = train_small();
        assert!(vocab.size > MERGES_START);
        assert!(!merges.is_empty());
    }

    #[test]
    fn test_encode_produces_valid_ids() {
        let (vocab, merges) = train_small();
        let ids = encode("hello world", &merges);
        for &id in &ids {
            assert!(id < vocab.size, "id {} >= vocab size {}", id, vocab.size);
        }
    }

    #[test]
    fn test_roundtrip() {
        let (vocab, merges) = train_small();
        let text = "hello world";
        let ids = encode(text, &merges);
        let decoded = decode(&ids, &vocab, true);
        assert_eq!(decoded, text);
    }

    #[test]
    fn test_encode_empty_string() {
        let (_, merges) = train_small();
        let ids = encode("", &merges);
        assert!(ids.is_empty());
    }

    #[test]
    fn test_decode_skip_special() {
        let (vocab, merges) = train_small();
        let mut ids = encode("hello", &merges);
        ids.insert(0, BOS_ID);
        ids.push(EOS_ID);
        let decoded = decode(&ids, &vocab, true);
        assert_eq!(decoded, "hello");
        let decoded_with = decode(&ids, &vocab, false);
        assert!(decoded_with.len() > decoded.len());
    }

    #[test]
    fn test_encode_with_special() {
        let (_, merges) = train_small();
        let ids = encode_with_special("hello", &merges, true);
        assert_eq!(ids[0], BOS_ID);
        assert_eq!(*ids.last().unwrap(), EOS_ID);
    }
}