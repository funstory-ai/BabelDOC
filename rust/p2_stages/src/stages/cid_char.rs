//! CID character ratio check over the M IR, kept separate from production wiring.
use crate::StageError;
use babeldoc_ir_core::{Document, ReadOnlyView};
use regex::Regex;
use std::sync::OnceLock;

const CID_THRESHOLD: f64 = 0.8;
const CID_ERROR_MESSAGE: &str = "The document contains too many CID chars.";

fn cid_pattern() -> &'static Regex {
    static PATTERN: OnceLock<Regex> = OnceLock::new();
    PATTERN.get_or_init(|| Regex::new(r"\A\(cid:\d+\)(?:\n)?\z").expect("static CID regex"))
}

/// Summary matching the 0.6.4 `check_cid_char` numerator, denominator and threshold.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CidCharReport {
    pub cid_chars: usize,
    pub total_chars: usize,
    /// `None` means there were no characters; 0.6.4 treats that case as not exceeded.
    pub ratio: Option<f64>,
    pub exceeds_threshold: bool,
}

/// Count CID strings in each page's top-level `pdf_character` list.
pub fn cid_char_report(document: &mut Document) -> Result<CidCharReport, StageError> {
    let view = document.view::<ReadOnlyView>();
    let root = view.document();
    let mut cid_chars = 0usize;
    let mut total_chars = 0usize;

    for page in root.page_iter(&view)? {
        let page = page?;
        total_chars += page.pdf_character_len(&view)?;
        for character in page.pdf_character_iter(&view)? {
            let character = character?;
            if character
                .char_unicode(&view)?
                .as_deref()
                .is_some_and(|unicode| cid_pattern().is_match(unicode))
            {
                cid_chars += 1;
            }
        }
    }

    let ratio = (total_chars != 0).then(|| cid_chars as f64 / total_chars as f64);
    // Mirror Python's `cid_count > len(chars) * 0.8` expression and strict boundary.
    let exceeds_threshold = cid_chars as f64 > total_chars as f64 * CID_THRESHOLD;
    Ok(CidCharReport {
        cid_chars,
        total_chars,
        ratio,
        exceeds_threshold,
    })
}

/// Return the same ExtractText failure used by 0.6.4 when the CID share is over 80%.
pub fn check_cid_char(document: &mut Document) -> Result<CidCharReport, StageError> {
    let report = cid_char_report(document)?;
    if report.exceeds_threshold {
        return Err(StageError::ExtractText(CID_ERROR_MESSAGE.to_owned()));
    }
    Ok(report)
}

#[cfg(test)]
mod cid_char_tests;
