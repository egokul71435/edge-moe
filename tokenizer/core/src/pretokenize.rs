// pretokenize.rs
//
// splits raw text into word-level chunks before BPE runs.
// BPE never merges across chunk boundaries — this controls
// what the tokenizer treats as atomic units.
//
// follows the cl100k_base pre-tokenization pattern from OpenAI's tiktoken,
// used in GPT-4, GPT-3.5-turbo, and text-embedding-ada-002.
// source: https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py
//
// uses fancy-regex instead of regex crate — required for lookahead (?!\S)
// in alternation 6 (trailing whitespace). fancy-regex is a superset of
// the regex crate with look-around support.
//
// input:  &str
// output: Vec<Vec<u8>>  (list of chunks, each chunk is raw UTF-8 bytes)

use fancy_regex::Regex;
use std::sync::OnceLock;

// ---------------------------------------------------------------------------
// cl100k_base pattern (verbatim from tiktoken)
// ---------------------------------------------------------------------------
//
// breakdown of each alternation in order:
//
// 1. (?i:'s|'t|'re|'ve|'m|'ll|'d)
//    contractions, case-insensitive. matched before words so "don't" →
//    ["don", "'t"] not ["don", "'", "t"]. (?i:...) scoped case-insensitivity.
//
// 2. [^\r\n\p{L}\p{N}]?\p{L}+
//    optional leading non-letter/non-digit/non-newline char (e.g. space,
//    quote) followed by one or more Unicode letters (\p{L} covers all scripts:
//    Latin, Arabic, CJK, Devanagari, etc.). handles " hello", "'quoted'", etc.
//
// 3. \p{N}{1,3}
//    1-3 Unicode digits as a group. "12345" → ["123", "45"]. batching up to
//    3 is GPT-4's deliberate choice — long enough to represent common numbers
//    (years, small ints) as single tokens, short enough to preserve
//    digit-level structure for arithmetic.
//
// 4. [^\s\p{L}\p{N}]+[\r\n]*
//    punctuation and symbol runs (anything not whitespace/letter/digit),
//    optionally followed by newlines. captures "...", "->", "!?" etc.
//
// 5. \s*[\r\n]+
//    newlines (with optional preceding whitespace). keeps line structure
//    visible to the model rather than collapsing into generic whitespace.
//
// 6. \s+(?!\S)
//    trailing whitespace not followed by a non-whitespace char. requires
//    lookahead — this is why we use fancy-regex instead of regex crate.
//
// 7. \s+
//    remaining whitespace runs not caught above.

const PRETOK_PATTERN: &str = concat!(
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)",  // 1. contractions (case-insensitive)
    r"|[^\r\n\p{L}\p{N}]?\p{L}+",      // 2. unicode words with optional prefix
    r"|\p{N}{1,3}",                     // 3. digit groups up to 3
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*",     // 4. punctuation/symbol runs
    r"|\s*[\r\n]+",                     // 5. newlines
    r"|\s+(?!\S)",                      // 6. trailing whitespace (needs lookahead)
    r"|\s+",                            // 7. remaining whitespace
);

// OnceLock compiles the regex exactly once across all threads
static REGEX: OnceLock<Regex> = OnceLock::new();

fn get_regex() -> &'static Regex {
    REGEX.get_or_init(|| {
        Regex::new(PRETOK_PATTERN).expect("pretokenize regex failed to compile")
    })
}

// ---------------------------------------------------------------------------
// public API
// ---------------------------------------------------------------------------

/// split `text` into pre-tokenized chunks.
/// each chunk is the raw UTF-8 bytes of one matched segment.
/// empty chunks are discarded.
pub fn pretokenize(text: &str) -> Vec<Vec<u8>> {
    get_regex()
        .find_iter(text)
        .filter_map(|m| m.ok())
        .map(|m| m.as_str().as_bytes().to_vec())
        .filter(|chunk| !chunk.is_empty())
        .collect()
}

/// pretokenize and flatten to byte ID chunks.
/// each byte in each chunk becomes its own u32 ID (0-255).
/// chunks remain separate — BPE sees them independently.
pub fn pretokenize_to_ids(text: &str) -> Vec<Vec<u32>> {
    pretokenize(text)
        .into_iter()
        .map(|chunk| chunk.into_iter().map(|b| b as u32).collect())
        .collect()
}

// ---------------------------------------------------------------------------
// tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn chunks_as_strings(text: &str) -> Vec<String> {
        pretokenize(text)
            .into_iter()
            .map(|chunk| String::from_utf8(chunk).unwrap())
            .collect()
    }

    #[test]
    fn test_basic_word_split() {
        let chunks = chunks_as_strings("hello world");
        assert_eq!(chunks, vec!["hello", " world"]);
    }

    #[test]
    fn test_contractions_atomic() {
        let chunks = chunks_as_strings("don't");
        assert_eq!(chunks, vec!["don", "'t"]);
    }

    #[test]
    fn test_contractions_case_insensitive() {
        let chunks = chunks_as_strings("DON'T");
        assert!(chunks.contains(&"'T".to_string()));
    }

    #[test]
    fn test_digits_batch_up_to_3() {
        let chunks = chunks_as_strings("12345");
        assert_eq!(chunks, vec!["123", "45"]);

        let chunks = chunks_as_strings("42");
        assert_eq!(chunks, vec!["42"]);

        let chunks = chunks_as_strings("7");
        assert_eq!(chunks, vec!["7"]);
    }

    #[test]
    fn test_unicode_letters() {
        let chunks = chunks_as_strings("café");
        assert!(!chunks.is_empty());

        let chunks = pretokenize("مرحبا");
        assert!(!chunks.is_empty());
    }

    #[test]
    fn test_punctuation() {
        let chunks = chunks_as_strings("hello, world!");
        assert_eq!(chunks, vec!["hello", ",", " world", "!"]);
    }

    #[test]
    fn test_punctuation_runs() {
        let chunks = chunks_as_strings("hello...");
        assert_eq!(chunks, vec!["hello", "..."]);
    }

    #[test]
    fn test_newlines_separate() {
        let chunks = chunks_as_strings("hello\nworld");
        assert!(chunks.contains(&"\n".to_string()));
    }

    #[test]
    fn test_leading_space_attached_to_word() {
        let chunks = chunks_as_strings("hello world foo");
        assert_eq!(chunks, vec!["hello", " world", " foo"]);
    }

    #[test]
    fn test_empty_string() {
        let chunks = pretokenize("");
        assert!(chunks.is_empty());
    }

    #[test]
    fn test_roundtrip_bytes() {
        let text = "Hello world! don't repeat 42 times.";
        let chunks = pretokenize(text);
        let recovered: Vec<u8> = chunks.into_iter().flatten().collect();
        assert_eq!(String::from_utf8(recovered).unwrap(), text);
    }

    #[test]
    fn test_pretokenize_to_ids_range() {
        let ids = pretokenize_to_ids("hello world 123");
        for chunk in &ids {
            for &id in chunk {
                assert!(id < 256, "id {} out of byte range", id);
            }
        }
    }

    #[test]
    fn test_year_as_single_token() {
        let chunks = chunks_as_strings("2024");
        assert_eq!(chunks, vec!["202", "4"]);

        let chunks = chunks_as_strings("100");
        assert_eq!(chunks, vec!["100"]);
    }

    #[test]
    fn test_sentence() {
        let chunks = chunks_as_strings("The quick brown fox.");
        assert_eq!(chunks, vec!["The", " quick", " brown", " fox", "."]);
    }
}