//! The reference's pre-normalization valid-text predicate.
#[path = "valid_categories.rs"]
mod categories;
pub fn is_space(text: &str) -> bool {
    // Python's `$` also matches immediately before one final LF. The
    // reference uses ^[unicode_spaces]+$ for both projection and counting.
    let text = text.strip_suffix('\n').unwrap_or(text);
    !text.is_empty()
        && text.chars().all(|c| {
            matches!(
                c,
                ' ' | '\u{a0}' | '\u{1680}' | '\u{2000}'
                    ..='\u{200a}'
                        | '\u{202f}'
                        | '\u{205f}'
                        | '\u{3000}'
                        | '\u{200b}'
                        | '\u{2060}'
                        | '\t'
            )
        })
}

#[cfg(test)]
mod valid_text_tests;
pub fn collect(
    pages: &[Vec<String>],
    has_char: impl Fn(&str) -> bool,
) -> crate::Result<Vec<(i64, i64)>> {
    collect_pages(pages.iter().map(Vec::as_slice), has_char)
}

pub fn collect_pages<'a>(
    pages: impl Iterator<Item = &'a [String]>,
    has_char: impl Fn(&str) -> bool,
) -> crate::Result<Vec<(i64, i64)>> {
    static ENCODER: std::sync::LazyLock<std::result::Result<tiktoken_rs::CoreBPE, String>> =
        std::sync::LazyLock::new(|| tiktoken_rs::o200k_base().map_err(|e| e.to_string()));
    let encoder = ENCODER
        .as_ref()
        .map_err(|e| crate::Error::new("RuntimeError", e.clone()))?;
    Ok(pages
        .map(|page| {
            let mut text = String::new();
            for ch in page {
                if is_space(ch) {
                    text.push_str(ch);
                    continue;
                }
                let Some(first) = ch.chars().next() else {
                    continue;
                };
                let cp = first as u32;
                if categories::REJECT
                    .binary_search_by(|&(a, b)| {
                        if cp < a {
                            std::cmp::Ordering::Greater
                        } else if cp > b {
                            std::cmp::Ordering::Less
                        } else {
                            std::cmp::Ordering::Equal
                        }
                    })
                    .is_ok()
                {
                    continue;
                }
                if ch.contains("(cid:") {
                    continue;
                }
                if has_char(ch)
                    || (ch.chars().count() > 1 && ch.chars().all(|c| has_char(&c.to_string())))
                {
                    text.push_str(ch)
                }
            }
            let count = text.chars().count() as i64;
            let tokens = encoder.encode_ordinary(&text).len() as i64;
            (count, tokens)
        })
        .collect())
}
