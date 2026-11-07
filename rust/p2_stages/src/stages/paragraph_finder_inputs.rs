//! Inputs and debug-ID sources for paragraph finding.
use crate::StageError;
pub use babeldoc_font_core::{FontFace, FontFamily, FontMapper, PrimaryFontFamily};
use std::collections::BTreeSet;
const BASE58_ALPHABET: &str = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
/// ID source supplied by the differential harness or production random source.
pub trait DebugIdSource {
    fn next_id(&mut self) -> Result<String, StageError>;
}

#[derive(Clone, Debug, Default)]
pub struct RandomDebugIdSource;

impl DebugIdSource for RandomDebugIdSource {
    fn next_id(&mut self) -> Result<String, StageError> {
        let mut id = String::with_capacity(5);
        for _ in 0..5 {
            let index = fastrand::usize(..BASE58_ALPHABET.len());
            id.push(BASE58_ALPHABET.as_bytes()[index] as char);
        }
        Ok(id)
    }
}

#[derive(Clone, Debug, Default)]
pub struct ParagraphFinderConfig {
    pub formula_matcher:
        Option<std::sync::Arc<dyn crate::shared::formula_helper::FormulaPatternMatcher>>,
    pub debug: bool,
    pub ocr_workaround: bool,
    pub merge_alternating_line_numbers: Option<bool>,
    pub split_short_lines: bool,
    pub short_line_split_factor: f64,
    pub formula_font_pattern_hits: Option<BTreeSet<String>>,
}

/// Inputs that the caller must prepare before entering the GIL-free Rust stage.
pub struct ParagraphFinderInputs<'a> {
    pub config: &'a ParagraphFinderConfig,
    pub font_mapper: &'a FontMapper,
    pub debug_ids: &'a mut dyn DebugIdSource,
}
