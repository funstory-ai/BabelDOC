//! 0.6.4 formula primitives and their serial custom-pattern dependency.
//! Stage-diff callers can supply hit sets; production evaluates lazily.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::{LazyLock, OnceLock};

use babeldoc_font_core::FontMapper;
use babeldoc_il_types::schema::{Box as PdfBox, Page, PdfFormula};
use regex::Regex;

use super::layout_helper::formular_height_ignore_char;

#[path = "formula_patterns.rs"]
mod formula_patterns;

static FORMULA_CHARACTER_REGEX: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^[0-9\[\]•]").expect("literal formula character regex"));

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FormulaHelperError {
    EmptyCharacterList,
    MissingVisualBox,
    MissingCoordinate(&'static str),
    InvalidBase64,
    Base64PatternTypeError,
}

impl std::fmt::Display for FormulaHelperError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyCharacterList => f.write_str("min() iterable argument is empty"),
            Self::MissingVisualBox => f.write_str("formula character has no visual_bbox.box"),
            Self::MissingCoordinate(name) => write!(f, "formula helper input is missing {name}"),
            Self::InvalidBase64 => f.write_str("invalid base64-encoded formula font name"),
            Self::Base64PatternTypeError => {
                f.write_str("cannot use a string pattern on a bytes-like object")
            }
        }
    }
}

impl std::error::Error for FormulaHelperError {}

/// Evaluated only at serial consumer boundaries, after the reference guards.
/// Implementations must never be used by a CPU parallel worker.
pub trait FormulaPatternMatcher: Send + Sync + std::fmt::Debug {
    fn font(&self, name: &str) -> Result<bool, crate::StageError>;
    fn character(&self, text: &str) -> Result<bool, crate::StageError>;
    fn has_character_pattern(&self) -> bool {
        true
    }
}

pub fn start_char_with_matcher(
    text: &str,
    mapper: &impl FontCoverage,
    hit: Option<bool>,
    matcher: Option<&dyn FormulaPatternMatcher>,
) -> Result<bool, crate::StageError> {
    // These guards also decide whether Python may compile the custom regex.
    if matcher.is_some_and(|m| m.has_character_pattern())
        && !text.is_empty()
        && !text.contains("(cid:")
        && mapper.has_char(text)
    {
        if let Some(matcher) = matcher {
            return Ok(is_formulas_start_char(
                text,
                mapper,
                Some(matcher.character(text)?),
            ));
        }
    }
    Ok(is_formulas_start_char(text, mapper, hit))
}

pub fn font_with_matcher(
    name: &str,
    hits: Option<&BTreeSet<String>>,
    matcher: Option<&dyn FormulaPatternMatcher>,
) -> Result<bool, crate::StageError> {
    match matcher {
        Some(matcher) => matcher.font(name),
        None => is_formulas_font(name, hits).map_err(Into::into),
    }
}

pub trait FontCoverage {
    fn has_char(&self, character: &str) -> bool;
}

impl FontCoverage for FontMapper {
    fn has_char(&self, character: &str) -> bool {
        FontMapper::has_char(self, character)
    }
}

pub fn is_formulas_start_char(
    character: &str,
    font_mapper: &impl FontCoverage,
    character_pattern_hit: Option<bool>,
) -> bool {
    if character.is_empty() {
        return false;
    }
    if character.contains("(cid:") {
        return true;
    }
    if !font_mapper.has_char(character) {
        if character.chars().count() > 1
            && character
                .chars()
                .all(|ch| font_mapper.has_char(&ch.to_string()))
        {
            return false;
        }
        return true;
    }
    if character_pattern_hit == Some(true) {
        return true;
    }
    let first = character.chars().next().expect("nonempty checked");
    let cp = first as u32;
    if character != " "
        && (matches!(
            unicode_category(cp),
            Some("Mn" | "Sk" | "Sm" | "Zl" | "Zp" | "Zs" | "Co")
        ) || (0x370..0x400).contains(&cp))
    {
        return true;
    }
    FORMULA_CHARACTER_REGEX.is_match(character)
}

fn unicode_category(codepoint: u32) -> Option<&'static str> {
    let character = char::from_u32(codepoint)?;
    ["Mn", "Sk", "Sm", "Zl", "Zp", "Zs", "Co"]
        .into_iter()
        .find(|category| super::layout_helper::unicode::category_is(character, category))
}

pub fn is_formulas_middle_char(
    character: &str,
    font_mapper: &impl FontCoverage,
    character_pattern_hit: Option<bool>,
) -> bool {
    is_formulas_start_char(character, font_mapper, character_pattern_hit)
        || character.starts_with(',')
}

pub fn collect_page_formula_font_ids(
    page: &Page,
    custom_font_pattern_hits: Option<&BTreeSet<String>>,
) -> Result<(BTreeSet<String>, BTreeMap<i64, BTreeSet<String>>), FormulaHelperError> {
    let mut page_ids = BTreeSet::new();
    for font in &page.pdf_font {
        let is_formula = is_formulas_font(
            font.as_ref().name.as_deref().unwrap_or_default(),
            custom_font_pattern_hits,
        )?;
        if is_formula {
            if let Some(font_id) = font.as_ref().font_id.as_ref() {
                page_ids.insert(font_id.clone());
            }
        }
    }
    let mut xobject_ids = BTreeMap::new();
    for xobject in &page.pdf_xobject {
        let mut current = page_ids.clone();
        for font in &xobject.pdf_font {
            let font = font.as_ref();
            let id = font.font_id.clone().unwrap_or_default();
            if is_formulas_font(
                font.name.as_deref().unwrap_or_default(),
                custom_font_pattern_hits,
            )? {
                current.insert(id);
            } else {
                current.remove(&id);
            }
        }
        xobject_ids.insert(xobject.xobj_id.unwrap_or_default(), current);
    }
    Ok((page_ids, xobject_ids))
}

pub fn is_formulas_font(
    font_name: &str,
    custom_font_pattern_hits: Option<&BTreeSet<String>>,
) -> Result<bool, FormulaHelperError> {
    static PRECISE: OnceLock<Regex> = OnceLock::new();
    static EXCLUDE: OnceLock<Regex> = OnceLock::new();
    static DEFAULT_BROAD: OnceLock<Regex> = OnceLock::new();
    let precise = PRECISE.get_or_init(|| {
        Regex::new(formula_patterns::PRECISE_FORMULA_FONT_PATTERN)
            .expect("literal precise font regex")
    });
    let exclude = EXCLUDE.get_or_init(|| {
        Regex::new(formula_patterns::PATTERN_TEXT).expect("literal font exclusion regex")
    });
    let default_broad = DEFAULT_BROAD.get_or_init(|| {
        Regex::new(formula_patterns::DEFAULT_BROAD_PATTERN).expect("literal broad font regex")
    });

    let (font, base64_name) = if let Some(encoded) = font_name.strip_prefix("BASE64:") {
        let decoded = base64_decode_python_tolerant(encoded)?;
        let suffix = decoded
            .rsplit(|byte| *byte == b'+')
            .next()
            .unwrap_or_default();
        (String::from_utf8_lossy(suffix).into_owned(), true)
    } else {
        let suffix = font_name.rsplit('+').next().unwrap_or_default();
        (suffix.to_owned(), false)
    };
    if font.is_empty() {
        return Ok(false);
    }
    if base64_name {
        return Err(FormulaHelperError::Base64PatternTypeError);
    }
    if python_match(precise, &font) {
        return Ok(true);
    }
    if python_match(exclude, &font) {
        return Ok(false);
    }
    if let Some(hits) = custom_font_pattern_hits {
        return Ok(hits.contains(&font));
    }
    Ok(python_match(default_broad, &font))
}

fn python_match(pattern: &Regex, text: &str) -> bool {
    pattern
        .find(text)
        .is_some_and(|matched| matched.start() == 0)
        || text
            .strip_suffix('\n')
            .is_some_and(|without_final_newline| {
                pattern
                    .find(without_final_newline)
                    .is_some_and(|matched| matched.start() == 0)
            })
}

fn base64_decode_python_tolerant(input: &str) -> Result<Vec<u8>, FormulaHelperError> {
    let alphabet = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut cleaned = Vec::new();
    for byte in input.bytes() {
        if byte == b'=' || alphabet.contains(&byte) {
            cleaned.push(byte);
        }
    }
    let data_len = cleaned.iter().take_while(|byte| **byte != b'=').count();
    if data_len % 4 == 1 {
        return Err(FormulaHelperError::InvalidBase64);
    }
    let mut bits = 0u32;
    let mut bit_count = 0u8;
    let mut output = Vec::new();
    for byte in cleaned.into_iter().take(data_len) {
        let value = alphabet
            .iter()
            .position(|candidate| *candidate == byte)
            .ok_or(FormulaHelperError::InvalidBase64)? as u32;
        bits = (bits << 6) | value;
        bit_count += 6;
        if bit_count >= 8 {
            bit_count -= 8;
            output.push((bits >> bit_count) as u8);
        }
    }
    Ok(output)
}

pub fn update_formula_data(formula: &mut PdfFormula) -> Result<(), FormulaHelperError> {
    if formula.pdf_character.is_empty() {
        return Err(FormulaHelperError::EmptyCharacterList);
    }
    let mut xs = Vec::with_capacity(formula.pdf_character.len());
    let mut ys = Vec::with_capacity(formula.pdf_character.len());
    let mut eligible_ys = Vec::new();
    for character in &formula.pdf_character {
        let visual = character
            .visual_bbox
            .as_deref()
            .and_then(|bbox| bbox.r#box.as_deref())
            .ok_or(FormulaHelperError::MissingVisualBox)?;
        let x = visual
            .x
            .ok_or(FormulaHelperError::MissingCoordinate("visual_bbox.box.x"))?;
        let x2 = visual
            .x2
            .ok_or(FormulaHelperError::MissingCoordinate("visual_bbox.box.x2"))?;
        let y = visual
            .y
            .ok_or(FormulaHelperError::MissingCoordinate("visual_bbox.box.y"))?;
        let y2 = visual
            .y2
            .ok_or(FormulaHelperError::MissingCoordinate("visual_bbox.box.y2"))?;
        xs.push((x, x2));
        ys.push((y, y2));
        if !formular_height_ignore_char(character) {
            eligible_ys.push((y, y2));
        }
    }
    let min_x = xs.iter().map(|x| x.0).fold(f64::INFINITY, f64::min);
    let max_x = xs.iter().map(|x| x.1).fold(f64::NEG_INFINITY, f64::max);
    let ys_for_box = if eligible_ys.is_empty() {
        &ys
    } else {
        &eligible_ys
    };
    let min_y = ys_for_box.iter().map(|y| y.0).fold(f64::INFINITY, f64::min);
    let max_y = ys_for_box
        .iter()
        .map(|y| y.1)
        .fold(f64::NEG_INFINITY, f64::max);
    formula.r#box = Some(std::boxed::Box::new(PdfBox {
        x: Some(min_x),
        y: Some(min_y),
        x2: Some(max_x),
        y2: Some(max_y),
    }));
    if formula.y_offset.unwrap_or_default() == 0.0 {
        formula.y_offset = Some(0.0);
    }
    if formula.x_offset.unwrap_or_default() == 0.0 {
        formula.x_offset = Some(0.0);
    }
    if formula.x_advance.unwrap_or_default() == 0.0 {
        formula.x_advance = Some(0.0);
    }
    Ok(())
}

/// Python re.match(r"^\(cid:\d+\)$", text), including its final-LF anchor.
pub fn is_cid_character(text: &str) -> bool {
    static CID: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^\(cid:\d+\)$").unwrap());
    CID.is_match(text) || text.strip_suffix('\n').is_some_and(|s| CID.is_match(s))
}
