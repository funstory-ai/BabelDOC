//! Configuration and locally resolved fonts for formula/style classification.
use crate::shared::formula_helper::FontCoverage;
use babeldoc_font_core::{FontFace, FontFamily, FontMapper, PrimaryFontFamily};
use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;
use std::sync::Arc;
/// Inputs that the Python/Rust boundary must provide for StylesAndFormulas.
///
/// `character_pattern_hits` is keyed by the complete character string passed
/// to `is_formulas_start_char`; `font_pattern_hits` contains the exact suffix
/// after the last `+` in each font name. `None` means no custom pattern was
/// configured, while an empty map/set means it was configured and matched
/// nothing.
#[derive(Clone)]
pub struct StylesAndFormulasConfig {
    pub formula_matcher: Option<Arc<dyn crate::shared::formula_helper::FormulaPatternMatcher>>,
    pub font_mapper: Arc<dyn FontCoverage + Send + Sync>,
    pub character_pattern_hits: Option<BTreeMap<String, bool>>,
    pub font_pattern_hits: Option<BTreeSet<String>>,
    pub skip_formula_offset_calculation: bool,
    pub ocr_workaround: bool,
    pub remove_non_formula_lines: bool,
    pub figure_table_protection_threshold: Option<f64>,
    pub non_formula_line_iou_threshold: Option<f64>,
}

impl<T: FontCoverage + ?Sized> FontCoverage for Arc<T> {
    fn has_char(&self, character: &str) -> bool {
        self.as_ref().has_char(character)
    }
}

impl StylesAndFormulasConfig {
    pub fn new(font_mapper: FontMapper) -> Self {
        Self {
            font_mapper: Arc::new(font_mapper),
            formula_matcher: None,
            character_pattern_hits: None,
            font_pattern_hits: None,
            skip_formula_offset_calculation: false,
            ocr_workaround: false,
            remove_non_formula_lines: false,
            figure_table_protection_threshold: None,
            non_formula_line_iou_threshold: None,
        }
    }

    /// Build the same mapper inputs used by the 0.6.4 Python adapter from
    /// already-resolved local font assets. This is exposed for the stage diff
    /// adapter; it performs no asset lookup or network access.
    pub fn from_font_paths(
        normal: Vec<(String, PathBuf)>,
        script: Vec<(String, PathBuf)>,
        fallback: Vec<(String, PathBuf)>,
        base: Vec<(String, PathBuf)>,
        primary_font_family: Option<&str>,
    ) -> Result<Self, String> {
        let mut loaded = BTreeMap::new();
        let mut load_group = |specs: Vec<(String, PathBuf)>| {
            specs
                .into_iter()
                .map(|(id, path)| {
                    if let Some(face) = loaded.get(&id) {
                        return Ok(Arc::clone(face));
                    }
                    let face =
                        FontFace::load(id.clone(), path).map_err(|error| error.to_string())?;
                    loaded.insert(id, Arc::clone(&face));
                    Ok(face)
                })
                .collect::<Result<Vec<_>, String>>()
        };
        let normal = load_group(normal)?;
        let script = load_group(script)?;
        let fallback = load_group(fallback)?;
        let base = load_group(base)?;
        let mapper = FontMapper::new(
            FontFamily {
                normal,
                script,
                fallback,
                base,
            },
            PrimaryFontFamily::from_option(primary_font_family),
        )
        .map_err(|error| error.to_string())?;
        Ok(Self::new(mapper))
    }
}
