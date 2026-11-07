//! Pure Rust port of the 0.6.4 document IL layout helpers.
//!
//! `LayoutIndex` uses the libspatialindex 2.1.0 R*-tree compatibility port so
//! candidate order matches 0.6.4 for the shared layout-helper queries.

use std::sync::LazyLock;

use babeldoc_font_core::FontFace;
use babeldoc_il_types::schema::{
    Box as PdfBox, GraphicState, Page, PageLayout, PdfCharacter, PdfCurve, PdfParagraph,
    PdfParagraphComposition, PdfStyle,
};
use regex::Regex;
use rtree_compat::{RTree, Rect};

#[path = "unicode.rs"]
pub(crate) mod unicode;
#[path = "unicode_data.rs"]
pub(super) mod unicode_data;

pub const HEIGHT_NOT_USFUL_CHAR_IN_CHAR: &[Option<&str>] = &[None];
pub const LEFT_BRACKET: &[&str] = &["(cid:8)", "(", "(cid:16)", "{", "[", "(cid:104)", "(cid:2)"];
pub const RIGHT_BRACKET: &[&str] = &["(cid:9)", ")", "(cid:17)", "}", "]", "(cid:105)", "(cid:3)"];

const BULLET_PATTERN: &str =
    r"^[■•⚫⬤◆◇○●◦‣⁃▪▫∗†‡¹²³⁴⁵⁶⁷⁸⁹⁰₁₂₃₄₅₆₇₈₉₀ᵃᵇᶜᵈᵉᶠᵍʰⁱʲᵏˡᵐⁿᵒᵖʳˢᵗᵘᵛʷˣʸᶻ¶※⁑⁂⁕⁎⁜❧☙⁋‖‽·]";
pub static BULLET_POINT_PATTERN: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(BULLET_PATTERN).expect("literal bullet regex"));
pub static SPACE_REGEX: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\s+").expect("literal whitespace regex"));

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LayoutHelperError {
    MissingCoordinate(&'static str),
    EmptyFontMetricResult,
    DirectCharacterNotSubscriptable,
    DirectCharacterHasNoLen,
}

impl std::fmt::Display for LayoutHelperError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingCoordinate(name) => write!(f, "layout helper input is missing {name}"),
            Self::EmptyFontMetricResult => f.write_str("font char_lengths returned no widths"),
            Self::DirectCharacterNotSubscriptable => {
                f.write_str("'PdfCharacter' object is not subscriptable")
            }
            Self::DirectCharacterHasNoLen => {
                f.write_str("object of type 'PdfCharacter' has no len()")
            }
        }
    }
}

impl std::error::Error for LayoutHelperError {}

pub fn is_bullet_point(character: &PdfCharacter) -> bool {
    BULLET_POINT_PATTERN.is_match(character.char_unicode.as_deref().unwrap_or_default())
}

pub fn calculate_box_iou(
    box1: Option<&PdfBox>,
    box2: Option<&PdfBox>,
) -> Result<f64, LayoutHelperError> {
    let (Some(box1), Some(box2)) = (box1, box2) else {
        return Ok(0.0);
    };
    let a = coordinates(box1)?;
    let b = coordinates(box2)?;
    let left = a.0.max(b.0);
    let top = a.1.max(b.1);
    let right = a.2.min(b.2);
    let bottom = a.3.min(b.3);
    if left >= right || top >= bottom {
        return Ok(0.0);
    }
    let intersection = (right - left) * (bottom - top);
    let area1 = (a.2 - a.0) * (a.3 - a.1);
    let area2 = (b.2 - b.0) * (b.3 - b.1);
    let union = area1 + area2 - intersection;
    Ok(if union <= 0.0 {
        0.0
    } else {
        intersection / union
    })
}

pub fn formular_height_ignore_char(character: &PdfCharacter) -> bool {
    character.pdf_character_id.is_none()
        || HEIGHT_NOT_USFUL_CHAR_IN_CHAR
            .iter()
            .any(|value| value.is_some_and(|text| character.char_unicode.as_deref() == Some(text)))
}

pub fn box_to_tuple(box_: Option<&PdfBox>) -> Result<(f64, f64, f64, f64), LayoutHelperError> {
    match box_ {
        None => Ok((0.0, 0.0, 0.0, 0.0)),
        Some(box_) => coordinates(box_),
    }
}

fn coordinate(value: Option<f64>, name: &'static str) -> Result<f64, LayoutHelperError> {
    value.ok_or(LayoutHelperError::MissingCoordinate(name))
}

fn coordinates(box_: &PdfBox) -> Result<(f64, f64, f64, f64), LayoutHelperError> {
    Ok((
        coordinate(box_.x, "box.x")?,
        coordinate(box_.y, "box.y")?,
        coordinate(box_.x2, "box.x2")?,
        coordinate(box_.y2, "box.y2")?,
    ))
}

#[derive(Clone, Debug, PartialEq)]
pub struct Layout {
    pub id: Option<i64>,
    pub name: Option<String>,
}

impl Layout {
    pub fn new(id: Option<i64>, name: Option<String>) -> Self {
        Self { id, name }
    }

    pub fn is_newline(previous: Option<&PdfCharacter>, current: &PdfCharacter) -> bool {
        let Some(previous) = previous else {
            return false;
        };
        let Some(prev_box) = previous.r#box.as_deref() else {
            return false;
        };
        let Some(curr_box) = current.r#box.as_deref() else {
            return false;
        };
        let prev = coordinates(prev_box).unwrap_or_default();
        let curr = coordinates(curr_box).unwrap_or_default();
        let char_height = (curr.3 - curr.1).max(prev.3 - prev.1);
        let char_width = (curr.2 - curr.0).max(prev.2 - prev.0);
        let should_new_line = curr.3 < prev.1 || curr.2 < prev.0 - char_width * 10.0;
        if should_new_line
            && (formular_height_ignore_char(current) || formular_height_ignore_char(previous))
        {
            return false;
        }
        let _ = char_height; // retained because Python computes it before the predicate.
        should_new_line
    }
}

#[derive(Clone, Debug)]
pub struct LayoutIndex {
    tree: RTree,
    indexed_indices: Vec<usize>,
}

impl LayoutIndex {
    pub fn build(
        layouts: &[PageLayout],
    ) -> Result<(Self, Vec<Option<PageLayout>>), LayoutHelperError> {
        let mut tree = RTree::new();
        let mut indexed_indices = Vec::new();
        let mut layout_map = Vec::with_capacity(layouts.len());
        for (index, layout) in layouts.iter().enumerate() {
            layout_map.push(Some(layout.clone()));
            if let Some(box_) = layout.r#box.as_deref() {
                let (min_x, min_y, max_x, max_y) = coordinates(box_)?;
                let tree_id = indexed_indices.len() as i64;
                tree.insert(tree_id, Rect::new(min_x, min_y, max_x, max_y));
                indexed_indices.push(index);
            }
        }
        Ok((
            Self {
                tree,
                indexed_indices,
            },
            layout_map,
        ))
    }

    pub fn intersection(
        &self,
        query: (f64, f64, f64, f64),
    ) -> Result<Vec<usize>, LayoutHelperError> {
        Ok(self
            .tree
            .intersection(Rect::new(query.0, query.1, query.2, query.3))
            .into_iter()
            .map(|tree_id| self.indexed_indices[tree_id as usize])
            .collect())
    }

    pub fn len(&self) -> usize {
        self.indexed_indices.len()
    }

    pub fn is_empty(&self) -> bool {
        self.indexed_indices.is_empty()
    }
}

pub fn build_layout_index(
    page: &Page,
) -> Result<(LayoutIndex, Vec<Option<PageLayout>>), LayoutHelperError> {
    LayoutIndex::build(&page.page_layout)
}

pub fn calculate_iou_for_boxes(box1: &PdfBox, box2: &PdfBox) -> Result<f64, LayoutHelperError> {
    let a = coordinates(box1)?;
    let b = coordinates(box2)?;
    let left = a.0.max(b.0);
    let bottom = a.1.max(b.1);
    let right = a.2.min(b.2);
    let top = a.3.min(b.3);
    if right <= left || top <= bottom {
        return Ok(0.0);
    }
    let area = (right - left) * (top - bottom);
    let first_area = (a.2 - a.0) * (a.3 - a.1);
    Ok(if first_area <= 0.0 {
        0.0
    } else {
        area / first_area
    })
}

pub fn calculate_y_iou_for_boxes(box1: &PdfBox, box2: &PdfBox) -> Result<f64, LayoutHelperError> {
    let a = coordinates(box1)?;
    let b = coordinates(box2)?;
    let bottom = a.1.max(b.1);
    let top = a.3.min(b.3);
    if top <= bottom {
        return Ok(0.0);
    }
    let height = a.3 - a.1;
    Ok(if height <= 0.0 {
        0.0
    } else {
        (top - bottom) / height
    })
}

pub fn calculate_y_true_iou_for_boxes(
    box1: &PdfBox,
    box2: &PdfBox,
) -> Result<f64, LayoutHelperError> {
    let a = coordinates(box1)?;
    let b = coordinates(box2)?;
    let bottom = a.1.max(b.1);
    let top = a.3.min(b.3);
    if top <= bottom {
        return Ok(0.0);
    }
    let first_height = a.3 - a.1;
    let second_height = b.3 - b.1;
    let min_height = first_height.min(second_height);
    Ok(if first_height <= 0.0 {
        0.0
    } else {
        (top - bottom) / min_height
    })
}

const DEFAULT_LAYOUT_PRIORITY: &[&str] = &[
    "number",
    "reference",
    "reference_content",
    "algorithm",
    "formula_caption",
    "isolate_formula",
    "table_footnote",
    "table_caption",
    "figure_caption",
    "figure_title",
    "chart_title",
    "table_title",
    "table_cell_hybrid",
    "table_text",
    "wireless_table_cell",
    "wired_table_cell",
    "abandon",
    "title",
    "abstract",
    "paragraph_title",
    "content",
    "doc_title",
    "footnote",
    "header",
    "footer",
    "seal",
    "plain text",
    "tiny text",
    "author_info_hybrid",
    "list_item_hybrid",
    "text",
    "paragraph_hybrid",
    "paragraph",
    "table_cell",
    "figure_text",
    "list_item",
    "title",
    "caption",
    "footnote_hybrid",
    "footnote",
    "formula",
    "formula_hybrid",
    "page_header",
    "page_footer",
    "reference_hybrid",
    "document_hybrid",
    "academic_paper_hybrid",
    "form_or_table_hybrid",
    "presentation_slide_hybrid",
    "webpage_screenshot_hybrid",
    "manga_or_comic_hybrid",
    "advertisement_hybrid",
    "magazine_or_newspaper_hybrid",
    "other_hybrid",
    "table_cell_hybrid",
    "figure_text_hybrid",
    "title_hybrid",
    "caption_hybrid",
    "code_algo_hybrid",
    "line_number_hybrid",
    "page_header_hybrid",
    "page_footer_hybrid",
    "page_number_hybrid",
    "unknown_hybrid",
    "fallback_line",
    "table",
    "figure",
    "image",
];

pub fn get_character_layout(
    character: &PdfCharacter,
    index: &LayoutIndex,
    layout_map: &[Option<PageLayout>],
) -> Result<Option<Layout>, LayoutHelperError> {
    get_character_layout_with_options(character, index, layout_map, None, BboxMode::Auto)
}

pub fn get_character_layout_with_options(
    character: &PdfCharacter,
    index: &LayoutIndex,
    layout_map: &[Option<PageLayout>],
    layout_priority: Option<&[&str]>,
    _bbox_mode: BboxMode,
) -> Result<Option<Layout>, LayoutHelperError> {
    let priority = layout_priority.unwrap_or(DEFAULT_LAYOUT_PRIORITY);
    let visual_box = character
        .visual_bbox
        .as_deref()
        .and_then(|bbox| bbox.r#box.as_deref())
        .ok_or(LayoutHelperError::MissingCoordinate(
            "character.visual_bbox.box",
        ))?;
    let char_box = coordinates(visual_box)?;
    let candidate_ids = index.intersection(char_box)?;
    let mut matching: Vec<(usize, f64, Layout)> = Vec::new();
    for candidate_id in candidate_ids {
        let Some(Some(layout)) = layout_map.get(candidate_id) else {
            continue;
        };
        let Some(layout_box) = layout.r#box.as_deref() else {
            continue;
        };
        let layout_coords = coordinates(layout_box)?;
        let intersection = (0.0_f64
            .max(char_box.2.min(layout_coords.2) - char_box.0.max(layout_coords.0)))
            * (0.0_f64.max(char_box.3.min(layout_coords.3) - char_box.1.max(layout_coords.1)));
        let area = (char_box.2 - char_box.0) * (char_box.3 - char_box.1);
        if area > 0.0 {
            let iou = intersection / area;
            if iou > 0.0 {
                let name = layout.class_name.as_deref().unwrap_or_default();
                let priority_index = priority
                    .iter()
                    .position(|entry| *entry == name)
                    .unwrap_or(priority.len());
                matching.push((
                    priority_index,
                    iou,
                    Layout::new(layout.id, layout.class_name.clone()),
                ));
            }
        }
    }
    matching.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| b.1.total_cmp(&a.1)));
    Ok(matching.into_iter().next().map(|entry| entry.2))
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BboxMode {
    #[default]
    Auto,
    Visual,
    Box,
}

pub fn is_text_layout(layout: Option<&Layout>) -> bool {
    const TEXT: &[&str] = &[
        "plain text",
        "tiny text",
        "title",
        "abandon",
        "figure_caption",
        "table_caption",
        "table_text",
        "table_footnote",
        "title",
        "paragraph_title",
        "abstract",
        "content",
        "figure_title",
        "table_title",
        "doc_title",
        "footnote",
        "header",
        "footer",
        "seal",
        "text",
        "chart_title",
        "paragraph",
        "table_cell",
        "figure_text",
        "list_item",
        "title",
        "caption",
        "footnote",
        "page_header",
        "page_footer",
        "wired_table_cell",
        "wireless_table_cell",
        "paragraph_hybrid",
        "table_cell_hybrid",
        "caption_hybrid",
        "unknown_hybrid",
        "figure_text_hybrid",
        "list_item_hybrid",
        "title_hybrid",
        "fallback_line",
        "author_info_hybrid",
        "page_header_hybrid",
        "page_footer_hybrid",
        "footnote_hybrid",
    ];
    layout.is_some_and(|layout| {
        layout
            .name
            .as_deref()
            .is_some_and(|name| TEXT.contains(&name))
    })
}

pub fn is_character_in_formula_layout(
    character: &PdfCharacter,
    _page: &Page,
    index: &LayoutIndex,
    layout_map: &[Option<PageLayout>],
) -> Result<Option<i64>, LayoutHelperError> {
    let visual = character
        .visual_bbox
        .as_deref()
        .and_then(|bbox| bbox.r#box.as_deref())
        .ok_or(LayoutHelperError::MissingCoordinate(
            "character.visual_bbox.box",
        ))?;
    let char_box = character
        .r#box
        .as_deref()
        .ok_or(LayoutHelperError::MissingCoordinate("character.box"))?;
    let selected = if calculate_iou_for_boxes(visual, char_box)? < 0.2 {
        char_box
    } else {
        visual
    };
    for candidate_id in index.intersection(coordinates(selected)?)? {
        let Some(Some(layout)) = layout_map.get(candidate_id) else {
            continue;
        };
        if layout.class_name.as_deref() == Some("formula") {
            if let Some(layout_box) = layout.r#box.as_deref() {
                if calculate_iou_for_boxes(selected, layout_box)? > 0.4 {
                    return Ok(layout.id);
                }
            }
        }
    }
    Ok(None)
}

pub fn is_curve_in_figure_table_layout(
    curve: &PdfCurve,
    index: &LayoutIndex,
    layout_map: &[Option<PageLayout>],
) -> Result<bool, LayoutHelperError> {
    is_curve_in_figure_table_layout_with_threshold(curve, index, layout_map, 0.3)
}

pub fn is_curve_in_figure_table_layout_with_threshold(
    curve: &PdfCurve,
    index: &LayoutIndex,
    layout_map: &[Option<PageLayout>],
    protection_threshold: f64,
) -> Result<bool, LayoutHelperError> {
    let Some(curve_box) = curve.r#box.as_deref() else {
        return Ok(false);
    };
    const LABELS: &[&str] = &[
        "figure",
        "table",
        "figure_text",
        "table_text",
        "figure_caption",
        "table_caption",
        "figure_title",
        "table_title",
        "chart_title",
        "table_cell",
        "table_cell_hybrid",
        "wired_table_cell",
        "wireless_table_cell",
        "table_footnote",
    ];
    for candidate_id in index.intersection(coordinates(curve_box)?)? {
        let Some(Some(layout)) = layout_map.get(candidate_id) else {
            continue;
        };
        if layout
            .class_name
            .as_deref()
            .is_some_and(|name| LABELS.contains(&name))
        {
            if let Some(layout_box) = layout.r#box.as_deref() {
                if calculate_iou_for_boxes(curve_box, layout_box)? > protection_threshold {
                    return Ok(true);
                }
            }
        }
    }
    Ok(false)
}

pub fn is_curve_overlapping_with_paragraphs(
    curve: &PdfCurve,
    paragraphs: &[PdfParagraph],
) -> Result<bool, LayoutHelperError> {
    is_curve_overlapping_with_paragraphs_with_threshold(curve, paragraphs, 0.2)
}

pub fn is_curve_overlapping_with_paragraphs_with_threshold(
    curve: &PdfCurve,
    paragraphs: &[PdfParagraph],
    overlap_threshold: f64,
) -> Result<bool, LayoutHelperError> {
    let Some(curve_box) = curve.r#box.as_deref() else {
        return Ok(false);
    };
    for paragraph in paragraphs {
        if let Some(bounds) = get_paragraph_bounding_box(paragraph)? {
            if calculate_iou_for_boxes(curve_box, &bounds)? > overlap_threshold {
                return Ok(true);
            }
        }
    }
    Ok(false)
}

pub fn get_paragraph_bounding_box(
    paragraph: &PdfParagraph,
) -> Result<Option<PdfBox>, LayoutHelperError> {
    if paragraph.pdf_paragraph_composition.is_empty() {
        return Ok(None);
    }
    let mut min_x = f64::INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    let mut found = false;
    for composition in &paragraph.pdf_paragraph_composition {
        let box_ = composition_box(composition)?;
        if let Some(box_) = box_ {
            let c = coordinates(box_)?;
            min_x = min_x.min(c.0);
            min_y = min_y.min(c.1);
            max_x = max_x.max(c.2);
            max_y = max_y.max(c.3);
            found = true;
        }
    }
    Ok(found.then_some(PdfBox {
        x: Some(min_x),
        y: Some(min_y),
        x2: Some(max_x),
        y2: Some(max_y),
    }))
}

fn composition_box(
    composition: &PdfParagraphComposition,
) -> Result<Option<&PdfBox>, LayoutHelperError> {
    if let Some(line) = composition.pdf_line.as_deref() {
        if line.r#box.is_some() {
            return Ok(line.r#box.as_deref());
        }
    }
    if let Some(formula) = composition.pdf_formula.as_deref() {
        if formula.r#box.is_some() {
            return Ok(formula.r#box.as_deref());
        }
    }
    if let Some(same) = composition.pdf_same_style_characters.as_deref() {
        if same.r#box.is_some() {
            return Ok(same.r#box.as_deref());
        }
    }
    if let Some(character) = composition.pdf_character.as_deref() {
        let _ = character;
        return Err(LayoutHelperError::DirectCharacterHasNoLen);
    }
    Ok(None)
}

fn collapse_nfkc_spaces(text: &str) -> String {
    let normalized = unicode::nfkc(text);
    let mut output = String::new();
    let mut in_space = false;
    for ch in normalized.chars() {
        if unicode::is_whitespace(ch) {
            if !in_space {
                output.push(' ');
            }
            in_space = true;
        } else {
            output.push(ch);
            in_space = false;
        }
    }
    output
}

fn trim_python_whitespace(text: String) -> String {
    let output = text;
    let trimmed = output
        .trim_start_matches(unicode::is_python_strip_whitespace)
        .trim_end_matches(unicode::is_python_strip_whitespace);
    trimmed.to_owned()
}

pub fn normalize_unicode_nfkc(text: &str) -> String {
    unicode::nfkc(text)
}

pub enum UnicodePart<'a> {
    Character(&'a PdfCharacter),
    Text(&'a str),
}

pub fn get_char_unicode_string(parts: &[UnicodePart<'_>]) -> Result<String, LayoutHelperError> {
    let mut distances = Vec::new();
    for pair in parts.windows(2) {
        if let [
            UnicodePart::Character(first),
            UnicodePart::Character(second),
        ] = pair
        {
            let first_box = first
                .r#box
                .as_deref()
                .ok_or(LayoutHelperError::MissingCoordinate("character.box"))?;
            let second_box = second
                .r#box
                .as_deref()
                .ok_or(LayoutHelperError::MissingCoordinate("character.box"))?;
            let distance = coordinate(second_box.x, "box.x")? - coordinate(first_box.x2, "box.x2")?;
            if distance > 1.0 {
                distances.push(distance);
            }
        }
    }
    distances.sort_by(f64::total_cmp);
    distances.dedup_by(|a, b| a == b);
    let median = match distances.as_slice() {
        [] => 1.0,
        [one] => *one,
        many => many[1],
    };
    let mut pieces = Vec::new();
    for (index, part) in parts.iter().enumerate() {
        let UnicodePart::Character(character) = part else {
            if let UnicodePart::Text(text) = part {
                pieces.push((*text).to_owned());
            }
            continue;
        };
        let text = character.char_unicode.as_deref().unwrap_or_default();
        pieces.push(collapse_nfkc_spaces(text));
        if text == " " {
            continue;
        }
        if let Some(UnicodePart::Character(next)) = parts.get(index + 1) {
            let current_box = character
                .r#box
                .as_deref()
                .ok_or(LayoutHelperError::MissingCoordinate("character.box"))?;
            let next_box = next
                .r#box
                .as_deref()
                .ok_or(LayoutHelperError::MissingCoordinate("character.box"))?;
            let distance = coordinate(next_box.x, "box.x")? - coordinate(current_box.x2, "box.x2")?;
            if distance >= median || Layout::is_newline(Some(character), next) {
                pieces.push(" ".to_owned());
            }
        }
    }
    Ok(trim_python_whitespace(collapse_nfkc_spaces(
        &pieces.concat(),
    )))
}

pub fn get_paragraph_unicode(paragraph: &PdfParagraph) -> Result<String, LayoutHelperError> {
    let mut parts = Vec::new();
    for composition in &paragraph.pdf_paragraph_composition {
        if let Some(line) = composition.pdf_line.as_deref() {
            parts.extend(line.pdf_character.iter().map(UnicodePart::Character));
        } else if let Some(same) = composition.pdf_same_style_characters.as_deref() {
            parts.extend(same.pdf_character.iter().map(UnicodePart::Character));
        } else if let Some(same) = composition.pdf_same_style_unicode_characters.as_deref() {
            if let Some(unicode) = same.unicode.as_deref() {
                parts.push(UnicodePart::Text(unicode));
            }
        } else if let Some(formula) = composition.pdf_formula.as_deref() {
            parts.extend(formula.pdf_character.iter().map(UnicodePart::Character));
        } else if let Some(character) = composition.pdf_character.as_deref() {
            parts.push(UnicodePart::Character(character));
        }
    }
    get_char_unicode_string(&parts)
}

pub fn get_paragraph_max_height(paragraph: &PdfParagraph) -> Result<f64, LayoutHelperError> {
    let mut max_height: f64 = 0.0;
    for composition in &paragraph.pdf_paragraph_composition {
        if let Some(character) = composition.pdf_character.as_deref() {
            let _ = character;
            return Err(LayoutHelperError::DirectCharacterNotSubscriptable);
        } else if let Some(same) = composition.pdf_same_style_characters.as_deref() {
            for character in &same.pdf_character {
                max_height = max_height.max(character_height(character)?);
            }
        } else if let Some(same) = composition.pdf_same_style_unicode_characters.as_deref() {
            max_height = max_height.max(
                same.pdf_style
                    .as_deref()
                    .and_then(|s| s.font_size)
                    .unwrap_or_default(),
            );
        } else if let Some(line) = composition.pdf_line.as_deref() {
            for character in &line.pdf_character {
                max_height = max_height.max(character_height(character)?);
            }
        } else if let Some(formula) = composition.pdf_formula.as_deref() {
            if let Some(box_) = formula.r#box.as_deref() {
                let c = coordinates(box_)?;
                max_height = max_height.max(c.3 - c.1);
            }
        }
    }
    Ok(max_height)
}

fn character_height(character: &PdfCharacter) -> Result<f64, LayoutHelperError> {
    let box_ = character
        .r#box
        .as_deref()
        .ok_or(LayoutHelperError::MissingCoordinate("character.box"))?;
    let c = coordinates(box_)?;
    Ok(c.3 - c.1)
}

pub fn get_paragraph_length_except(
    paragraph: &PdfParagraph,
    except_chars: &str,
    font: &impl FontMetrics,
) -> Result<f64, LayoutHelperError> {
    let mut length = 0.0;
    for composition in &paragraph.pdf_paragraph_composition {
        if let Some(character) = composition.pdf_character.as_deref() {
            let _ = character;
            return Err(LayoutHelperError::DirectCharacterNotSubscriptable);
        } else if let Some(same) = composition.pdf_same_style_characters.as_deref() {
            for character in &same.pdf_character {
                let text = character.char_unicode.as_deref().unwrap_or_default();
                if except_chars.contains(text) {
                    continue;
                }
                let box_ = character
                    .r#box
                    .as_deref()
                    .ok_or(LayoutHelperError::MissingCoordinate("character.box"))?;
                let c = coordinates(box_)?;
                length += c.2 - c.0;
            }
        } else if let Some(same) = composition.pdf_same_style_unicode_characters.as_deref() {
            let size = same
                .pdf_style
                .as_deref()
                .and_then(|style| style.font_size)
                .unwrap_or_default();
            for ch in same.unicode.as_deref().unwrap_or_default().chars() {
                let text = ch.to_string();
                if except_chars.contains(&text) {
                    continue;
                }
                length += font
                    .char_length(&text, size)
                    .ok_or(LayoutHelperError::EmptyFontMetricResult)?;
            }
        } else if let Some(line) = composition.pdf_line.as_deref() {
            for character in &line.pdf_character {
                let text = character.char_unicode.as_deref().unwrap_or_default();
                if except_chars.contains(text) {
                    continue;
                }
                let box_ = character
                    .r#box
                    .as_deref()
                    .ok_or(LayoutHelperError::MissingCoordinate("character.box"))?;
                let c = coordinates(box_)?;
                length += c.2 - c.0;
            }
        } else if let Some(formula) = composition.pdf_formula.as_deref() {
            if let Some(box_) = formula.r#box.as_deref() {
                let c = coordinates(box_)?;
                length += c.2 - c.0;
            }
        }
    }
    Ok(length)
}

pub trait FontMetrics {
    fn char_length(&self, text: &str, font_size: f64) -> Option<f64>;
}

impl FontMetrics for FontFace {
    fn char_length(&self, text: &str, font_size: f64) -> Option<f64> {
        self.char_lengths(text, font_size).first().copied()
    }
}

pub fn is_same_style(style1: Option<&PdfStyle>, style2: Option<&PdfStyle>) -> bool {
    let (Some(one), Some(two)) = (style1, style2) else {
        return style1.is_none() && style2.is_none();
    };
    one.font_id == two.font_id
        && (one.font_size.unwrap_or_default() - two.font_size.unwrap_or_default()).abs() < 0.02
        && is_same_graphic_state(one.graphic_state.as_ref(), two.graphic_state.as_ref())
}

pub fn is_same_style_except_size(style1: Option<&PdfStyle>, style2: Option<&PdfStyle>) -> bool {
    let (Some(one), Some(two)) = (style1, style2) else {
        return style1.is_none() && style2.is_none();
    };
    let size2 = two.font_size.unwrap_or_default();
    one.font_id == two.font_id
        && 0.7 < (one.font_size.unwrap_or_default() / size2).abs()
        && (one.font_size.unwrap_or_default() / size2).abs() < 1.3
        && is_same_graphic_state(one.graphic_state.as_ref(), two.graphic_state.as_ref())
}

pub fn is_same_style_except_font(style1: Option<&PdfStyle>, style2: Option<&PdfStyle>) -> bool {
    let (Some(one), Some(two)) = (style1, style2) else {
        return style1.is_none() && style2.is_none();
    };
    (one.font_size.unwrap_or_default() - two.font_size.unwrap_or_default()).abs() < 0.02
        && is_same_graphic_state(one.graphic_state.as_ref(), two.graphic_state.as_ref())
}

pub fn is_same_graphic_state(
    state1: Option<&babeldoc_il_types::Shared<GraphicState>>,
    state2: Option<&babeldoc_il_types::Shared<GraphicState>>,
) -> bool {
    let (Some(one), Some(two)) = (state1, state2) else {
        return state1.is_none() && state2.is_none();
    };
    graphic_instruction(one.as_ref()) == graphic_instruction(two.as_ref())
}

fn graphic_instruction(state: &GraphicState) -> Option<String> {
    state
        .passthrough_per_char_instruction
        .as_ref()
        .map(|value| match value {
            babeldoc_il_types::PassthroughValue::Materialized(text) => text.clone(),
            babeldoc_il_types::PassthroughValue::View(view) => {
                let mut output = String::new();
                view.write_to(&mut output, None);
                output
            }
        })
}

pub fn add_space_dummy_chars(paragraph: &mut PdfParagraph) -> Result<(), LayoutHelperError> {
    for composition in &mut paragraph.pdf_paragraph_composition {
        if let Some(line) = composition.pdf_line.as_deref_mut() {
            add_space_dummy_chars_to_list(&mut line.pdf_character)?;
        } else if let Some(same) = composition.pdf_same_style_characters.as_deref_mut() {
            add_space_dummy_chars_to_list(&mut same.pdf_character)?;
        } else if composition.pdf_same_style_unicode_characters.is_some() {
            continue;
        } else if let Some(formula) = composition.pdf_formula.as_deref_mut() {
            add_space_dummy_chars_to_list(&mut formula.pdf_character)?;
        }
    }
    for index in 0..paragraph.pdf_paragraph_composition.len().saturating_sub(1) {
        let current_last = last_char(&paragraph.pdf_paragraph_composition[index]).cloned();
        let next_first = first_char(&paragraph.pdf_paragraph_composition[index + 1]).cloned();
        let (Some(current_last), Some(next_first)) = (current_last, next_first) else {
            continue;
        };
        let current_box = current_last
            .r#box
            .as_deref()
            .ok_or(LayoutHelperError::MissingCoordinate("character.box"))?;
        let next_box = next_first
            .r#box
            .as_deref()
            .ok_or(LayoutHelperError::MissingCoordinate("character.box"))?;
        let current = coordinates(current_box)?;
        let next = coordinates(next_box)?;
        let distance = next.0 - current.2;
        if distance > 1.0 {
            let box_ = PdfBox {
                x: Some(current.2),
                y: Some(current.1),
                x2: Some(current.2 + distance),
                y2: Some(current.3),
            };
            let space = dummy_char(&current_last, box_);
            let composition = &mut paragraph.pdf_paragraph_composition[index];
            if let Some(line) = composition.pdf_line.as_deref_mut() {
                line.pdf_character.push(space);
            } else if let Some(same) = composition.pdf_same_style_characters.as_deref_mut() {
                same.pdf_character.push(space);
            } else if let Some(formula) = composition.pdf_formula.as_deref_mut() {
                formula.pdf_character.push(space);
            }
        }
    }
    Ok(())
}

fn first_char(composition: &PdfParagraphComposition) -> Option<&PdfCharacter> {
    if let Some(line) = composition.pdf_line.as_deref() {
        line.pdf_character.first()
    } else if let Some(same) = composition.pdf_same_style_characters.as_deref() {
        same.pdf_character.first()
    } else if let Some(formula) = composition.pdf_formula.as_deref() {
        formula.pdf_character.first()
    } else {
        composition.pdf_character.as_deref()
    }
}

fn last_char(composition: &PdfParagraphComposition) -> Option<&PdfCharacter> {
    if let Some(line) = composition.pdf_line.as_deref() {
        line.pdf_character.last()
    } else if let Some(same) = composition.pdf_same_style_characters.as_deref() {
        same.pdf_character.last()
    } else if let Some(formula) = composition.pdf_formula.as_deref() {
        formula.pdf_character.last()
    } else {
        composition.pdf_character.as_deref()
    }
}

fn dummy_char(source: &PdfCharacter, box_: PdfBox) -> PdfCharacter {
    let advance = box_.x2.unwrap_or_default() - box_.x.unwrap_or_default();
    PdfCharacter {
        pdf_style: source.pdf_style.clone(),
        r#box: Some(std::boxed::Box::new(box_.clone())),
        char_unicode: Some(" ".to_owned()),
        scale: source.scale,
        advance: Some(advance),
        visual_bbox: Some(std::boxed::Box::new(
            babeldoc_il_types::schema::VisualBbox {
                r#box: Some(std::boxed::Box::new(box_)),
            },
        )),
        ..PdfCharacter::default()
    }
}

fn add_space_dummy_chars_to_list(chars: &mut Vec<PdfCharacter>) -> Result<(), LayoutHelperError> {
    if chars.is_empty() {
        return Ok(());
    }
    let mut distances = Vec::new();
    for pair in chars.windows(2) {
        let a = pair[0]
            .r#box
            .as_deref()
            .ok_or(LayoutHelperError::MissingCoordinate("character.box"))?;
        let b = pair[1]
            .r#box
            .as_deref()
            .ok_or(LayoutHelperError::MissingCoordinate("character.box"))?;
        let distance = coordinate(b.x, "box.x")? - coordinate(a.x2, "box.x2")?;
        if distance > 1.0 {
            distances.push(distance);
        }
    }
    distances.sort_by(f64::total_cmp);
    distances.dedup_by(|a, b| a == b);
    let median = match distances.as_slice() {
        [] => 1.0,
        [one] => *one,
        many => many[1],
    };
    let mut index = 0;
    while index + 1 < chars.len() {
        let current = chars[index].clone();
        let next = &chars[index + 1];
        let current_box = current
            .r#box
            .as_deref()
            .ok_or(LayoutHelperError::MissingCoordinate("character.box"))?;
        let next_box = next
            .r#box
            .as_deref()
            .ok_or(LayoutHelperError::MissingCoordinate("character.box"))?;
        let current_coords = coordinates(current_box)?;
        let next_coords = coordinates(next_box)?;
        let mut distance = next_coords.0 - current_coords.2;
        if distance >= median || Layout::is_newline(Some(&current), next) {
            if distance < 0.0 {
                distance = -distance;
            }
            let width = distance.min(median);
            let box_ = PdfBox {
                x: Some(current_coords.2),
                y: Some(current_coords.1),
                x2: Some(current_coords.2 + width),
                y2: Some(current_coords.3),
            };
            chars.insert(index + 1, dummy_char(&current, box_));
            index += 2;
        } else {
            index += 1;
        }
    }
    Ok(())
}
