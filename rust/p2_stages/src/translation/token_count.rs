use std::sync::OnceLock;

use tiktoken_rs::CoreBPE;

static O200K: OnceLock<Result<CoreBPE, String>> = OnceLock::new();

fn encoding() -> Result<&'static CoreBPE, &'static str> {
    match O200K.get_or_init(|| tiktoken_rs::o200k_base().map_err(|e| e.to_string())) {
        Ok(encoding) => Ok(encoding),
        Err(_) => Err("failed to initialize o200k_base"),
    }
}

/// Matches Python `len(encoding_for_model("gpt-4o").encode(text, disallowed_special=()))`.
pub fn count_tokens(text: &str) -> usize {
    encoding().map_or(0, |encoding| encoding.encode_ordinary(text).len())
}

#[cfg(test)]
mod token_count_tests;
