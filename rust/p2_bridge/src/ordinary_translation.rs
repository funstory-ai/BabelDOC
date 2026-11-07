//! Ordinary paragraph translation over the native M document.
use crate::{
    callbacks::{self, CancelFlag, ProgressError, PythonProgress, PythonService, Recorder},
    font_mapper::FontMapperHandle,
    translation_service::TranslatorPoolPair,
};
use babeldoc_ir_core::{self as ir, Document, Permission, ReadOnlyView, TranslationView, TreeView};
use babeldoc_p2_stages::translation::ordinary_translation::PlaceholderRegex;
use babeldoc_p2_stages::{ProgressReporter, translation::ordinary_translation as stage};
use pyo3::{
    exceptions::{PyRuntimeError, PyTypeError},
    prelude::*,
    types::{PyDict, PyTuple},
};
use std::{
    collections::{HashMap, HashSet},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::Duration,
};
use tokio::sync::oneshot;

fn collect_children<T>(
    children: impl Iterator<Item = Result<T, babeldoc_ir_core::AccessError>>,
) -> PyResult<Vec<T>> {
    children.map(|child| child.map_err(access)).collect()
}

pub(crate) struct RegexService<'py> {
    pub(crate) service: Bound<'py, PyAny>,
}

pub(crate) fn title_hint_text(unicode: Option<String>) -> String {
    unicode.unwrap_or_else(|| "None".into())
}

fn initialize_first_title(
    first: &mut Option<(String, String)>,
    recent: &mut Option<(String, String)>,
    snapshot: babeldoc_p2_stages::runtime::shared_context::TitleContextSnapshot,
) -> Option<babeldoc_p2_stages::runtime::shared_context::TitleContextSnapshot> {
    if first.is_some() {
        return None;
    }
    let title = (
        snapshot.debug_id.clone().unwrap_or_default(),
        title_hint_text(snapshot.unicode.clone()),
    );
    *first = Some(title.clone());
    *recent = Some(title);
    Some(snapshot)
}

impl stage::PlaceholderRegex for RegexService<'_> {
    type Error = PyErr;

    fn compile(&mut self, pattern: &str, ignore_case: bool) -> PyResult<u64> {
        let flags = if ignore_case { 2 } else { 0 };
        self.service
            .call_method1("regex_compile", (pattern, flags))?
            .extract()
    }

    fn matches(&mut self, pattern: u64, text: &str) -> PyResult<bool> {
        Ok(!self
            .service
            .call_method1("regex_match", (pattern, text))?
            .is_none())
    }

    fn find_all(&mut self, pattern: u64, text: &str) -> PyResult<Vec<stage::RegexMatch>> {
        let values = self
            .service
            .call_method1("regex_finditer", (pattern, text))?;
        values
            .try_iter()?
            .map(|item| {
                let item = item?;
                let spans = item
                    .try_iter()?
                    .map(|span| {
                        let span = span?.cast_into::<PyTuple>()?;
                        let start = span.get_item(0)?.extract::<i64>()?;
                        let end = span.get_item(1)?.extract::<i64>()?;
                        Ok(if start < 0 || end < 0 {
                            None
                        } else {
                            Some((start as usize, end as usize))
                        })
                    })
                    .collect::<PyResult<Vec<_>>>()?;
                let Some((start, end)) = spans.first().and_then(|span| *span) else {
                    return Err(PyTypeError::new_err(
                        "Python regex returned no group zero span",
                    ));
                };
                let full = text.chars().skip(start).take(end - start).collect();
                Ok(stage::RegexMatch {
                    full,
                    start,
                    end,
                    groups: spans.into_iter().skip(1).collect(),
                })
            })
            .collect()
    }
}

fn placeholder_result(
    service: &Bound<'_, PyAny>,
    method: &str,
    id: &str,
) -> PyResult<(String, String)> {
    let value = service.call_method1(method, (id,))?;
    if let Ok(tuple) = value.cast::<PyTuple>() {
        Ok((tuple.get_item(0)?.extract()?, tuple.get_item(1)?.extract()?))
    } else {
        let token = value.extract::<String>()?;
        let pattern: String = service
            .py()
            .import("re")?
            .call_method1("escape", (&token,))?
            .extract()?;
        Ok((token, pattern))
    }
}

pub(crate) fn legacy_pattern(service: &Bound<'_, PyAny>, method: &str) -> PyResult<String> {
    let value = service.call_method1(method, (r"\d+",))?;
    if let Ok(tuple) = value.cast::<PyTuple>() {
        tuple.get_item(1)?.extract()
    } else {
        value
            .extract::<String>()?
            .chars()
            .nth(1)
            .map(|ch| ch.to_string())
            .ok_or_else(|| pyo3::exceptions::PyIndexError::new_err("string index out of range"))
    }
}

pub(crate) fn character_text(
    values: Vec<ir::Cursor<babeldoc_ir_core::types::PdfCharacter>>,
    view: &TreeView<'_, impl Permission>,
) -> PyResult<String> {
    let parts = character_fragments(values.into_iter().map(Ok), view)?;
    Ok(compose_input_text(&parts))
}

fn raw_character_text(
    characters: Vec<ir::Cursor<babeldoc_ir_core::types::PdfCharacter>>,
    view: &TreeView<'_, impl Permission>,
) -> PyResult<String> {
    characters
        .into_iter()
        .map(|character| {
            Ok(character
                .char_unicode(view)
                .map_err(access)?
                .clone()
                .unwrap_or_default())
        })
        .collect::<PyResult<Vec<_>>>()
        .map(|parts| parts.concat())
}

fn paragraph_unicode(
    paragraph: &ir::Cursor<babeldoc_ir_core::types::PdfParagraph>,
    view: &TreeView<'_, impl Permission>,
) -> PyResult<String> {
    let mut parts = Vec::new();
    for composition in paragraph
        .pdf_paragraph_composition_iter(view)
        .map_err(access)?
    {
        let composition = composition.map_err(access)?;
        if let Some(line) = composition.pdf_line(view).map_err(access)? {
            parts.extend(character_fragments(
                line.pdf_character_iter(view).map_err(access)?,
                view,
            )?);
        } else if let Some(same) = composition
            .pdf_same_style_characters(view)
            .map_err(access)?
        {
            parts.extend(character_fragments(
                same.pdf_character_iter(view).map_err(access)?,
                view,
            )?);
        } else if let Some(same) = composition
            .pdf_same_style_unicode_characters(view)
            .map_err(access)?
        {
            if let Some(unicode) = same.unicode(view).map_err(access)? {
                parts.push(InputTextPart::Text(unicode.clone()));
            }
        } else if let Some(formula) = composition.pdf_formula(view).map_err(access)? {
            parts.extend(character_fragments(
                formula.pdf_character_iter(view).map_err(access)?,
                view,
            )?);
        } else if let Some(character) = composition.pdf_character(view).map_err(access)? {
            parts.extend(character_fragments(std::iter::once(Ok(character)), view)?);
        }
    }
    Ok(compose_input_text(&parts))
}

fn instruction_equivalent(
    left: &Option<ir::Instructions>,
    right: &Option<ir::Instructions>,
    view: &babeldoc_ir_core::TreeView<'_, ReadOnlyView>,
) -> PyResult<bool> {
    Ok(view.render_instructions(left).map_err(access)?
        == view.render_instructions(right).map_err(access)?)
}

fn style_matches(
    left: Option<ir::Cursor<babeldoc_ir_core::types::PdfStyle>>,
    right: Option<ir::Cursor<babeldoc_ir_core::types::PdfStyle>>,
    view: &babeldoc_ir_core::TreeView<'_, ReadOnlyView>,
    except_size: bool,
    except_font: bool,
    mapper: &crate::font_mapper::FontMapperHandle,
    font_styles: &HashMap<String, babeldoc_p2_stages::font_mapper::SourceFontStyle>,
) -> PyResult<bool> {
    if left.is_none() || right.is_none() {
        return Ok(left.is_none() && right.is_none());
    }
    let left = left.expect("checked Some");
    let right = right.expect("checked Some");
    let left_id = left.font_id(view).map_err(access)?;
    let right_id = right.font_id(view).map_err(access)?;
    let left_size = left.font_size(view).map_err(access)?.unwrap_or_default();
    let right_size = right.font_size(view).map_err(access)?.unwrap_or_default();
    let left_graphic = left.graphic_state(view).map_err(access)?;
    let right_graphic = right.graphic_state(view).map_err(access)?;
    let same_graphic = match (left_graphic, right_graphic) {
        (None, None) => true,
        (Some(left), Some(right)) => {
            let left = left
                .passthrough_per_char_instruction(view)
                .map_err(access)?;
            let right = right
                .passthrough_per_char_instruction(view)
                .map_err(access)?;
            instruction_equivalent(left, right, view)?
        }
        _ => false,
    };
    let size_match = if except_size {
        let ratio = (left_size / right_size).abs();
        0.7 < ratio && ratio < 1.3
    } else {
        (left_size - right_size).abs() < 0.02
    };
    let font_match = if except_font {
        let source = |id: Option<String>| {
            id.and_then(|id| font_styles.get(&id).copied())
                .unwrap_or_default()
        };
        let left_mapped = mapper.mapped_font_id(
            source(left_id.clone()).bold,
            source(left_id.clone()).italic,
            source(left_id.clone()).monospaced,
            source(left_id.clone()).serif,
            '1',
        );
        let right_mapped = mapper.mapped_font_id(
            source(right_id.clone()).bold,
            source(right_id.clone()).italic,
            source(right_id.clone()).monospaced,
            source(right_id.clone()).serif,
            '1',
        );
        left_mapped.is_some() && left_mapped == right_mapped
    } else {
        left_id == right_id
    };
    Ok(font_match && size_match && same_graphic)
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum FontStyleLookupError {
    MissingId,
    MissingKey(String),
}

fn require_font_style(
    font_styles: &HashMap<String, babeldoc_p2_stages::font_mapper::SourceFontStyle>,
    font_id: Option<String>,
) -> Result<babeldoc_p2_stages::font_mapper::SourceFontStyle, FontStyleLookupError> {
    let Some(font_id) = font_id else {
        return Err(FontStyleLookupError::MissingId);
    };
    font_styles
        .get(&font_id)
        .copied()
        .ok_or(FontStyleLookupError::MissingKey(font_id))
}

impl FontStyleLookupError {
    pub(crate) fn key_error_message(&self) -> String {
        match self {
            Self::MissingId => "None".into(),
            Self::MissingKey(font_id) => python_string_repr(font_id),
        }
    }
}

pub(crate) fn build_input(
    paragraph: &ir::Cursor<babeldoc_ir_core::types::PdfParagraph>,
    view: &babeldoc_ir_core::TreeView<'_, ReadOnlyView>,
    service: &Bound<'_, PyAny>,
    regex: &mut RegexService<'_>,
    patterns: &stage::PlaceholderPatterns,
    allow_rich: bool,
    mapper: &crate::font_mapper::FontMapperHandle,
    font_styles: &HashMap<String, babeldoc_p2_stages::font_mapper::SourceFontStyle>,
    unicode_override: Option<&str>,
    deferred_missing_font_style: &mut Option<FontStyleLookupError>,
) -> PyResult<Option<stage::TranslateInput>> {
    let composition_len = paragraph
        .pdf_paragraph_composition_len(view)
        .map_err(access)?;
    if composition_len == 0 {
        return Ok(None);
    }
    let unicode = unicode_override
        .map_or_else(
            || {
                paragraph
                    .unicode(view)
                    .map(|value| value.clone())
                    .map_err(access)
            },
            |value| Ok(Some(value.to_owned())),
        )?
        .unwrap_or_default();
    if babeldoc_p2_stages::translation::paragraph_helpers::is_pure_numeric_text(&unicode) {
        return Ok(None);
    }
    let mut formula_only = composition_len != 0;
    for composition in paragraph
        .pdf_paragraph_composition_iter(view)
        .map_err(access)?
    {
        let composition = composition.map_err(access)?;
        formula_only &= composition.pdf_formula(view).map_err(access)?.is_some();
    }
    if formula_only
        || (!unicode.is_empty()
            && babeldoc_p2_stages::translation::python_semantics::python_isspace(&unicode))
    {
        return Ok(None);
    }
    let originals =
        stage::original_placeholder_tokens(regex, patterns, &unicode).map_err(access)?;
    if composition_len == 1 {
        let composition = paragraph
            .pdf_paragraph_composition_at(view, 0)
            .map_err(access)?;
        let plain = composition.pdf_line(view).map_err(access)?.is_some()
            || composition
                .pdf_same_style_characters(view)
                .map_err(access)?
                .is_some()
            || composition.pdf_character(view).map_err(access)?.is_some();
        if plain {
            return Ok(Some(stage::TranslateInput {
                text: unicode,
                placeholders: vec![],
                original_tokens: originals,
            }));
        }
        if composition.pdf_formula(view).map_err(access)?.is_some()
            || composition
                .pdf_same_style_unicode_characters(view)
                .map_err(access)?
                .is_some()
        {
            return Ok(None);
        }
        return Ok(None);
    }

    let mut parts = Vec::new();
    let mut placeholders = Vec::new();
    let mut next_id = 1i64;
    for (index, composition) in paragraph
        .pdf_paragraph_composition_iter(view)
        .map_err(access)?
        .enumerate()
    {
        let composition = composition.map_err(access)?;
        if let Some(line) = composition.pdf_line(view).map_err(access)? {
            parts.extend(character_fragments(
                line.pdf_character_iter(view).map_err(access)?,
                view,
            )?);
        } else if let Some(formula) = composition.pdf_formula(view).map_err(access)? {
            let formula_characters =
                collect_children(formula.pdf_character_iter(view).map_err(access)?)?;
            let cid_count = formula_characters
                .iter()
                .map(|character| character.char_unicode(view).map_err(access))
                .collect::<PyResult<Vec<_>>>()?
                .iter()
                .filter_map(|value| value.as_deref())
                .filter(|value| is_cid_placeholder(value))
                .count();
            let hint_source = cid_count as f64 <= formula_characters.len() as f64 * 0.8;
            let source_chars = character_text(formula_characters, view)?;
            let placeholder = stage::formula_placeholder(
                regex,
                &unicode,
                next_id,
                index,
                |id| placeholder_result(service, "get_formular_placeholder", &id.to_string()),
                source_chars,
                hint_source,
            )
            .map_err(access)?;
            next_id = placeholder.id + 1;
            parts.push(InputTextPart::Text(placeholder.token.clone()));
            placeholders.push(stage::InputPlaceholder::Formula(placeholder));
        } else if let Some(character) = composition.pdf_character(view).map_err(access)? {
            parts.extend(character_fragments(std::iter::once(Ok(character)), view)?);
        } else if let Some(group) = composition
            .pdf_same_style_characters(view)
            .map_err(access)?
        {
            let group_characters =
                collect_children(group.pdf_character_iter(view).map_err(access)?)?;
            let source_text = raw_character_text(group_characters.clone(), view)?;
            let composition_style = group.pdf_style(view).map_err(access)?;
            let paragraph_style = paragraph.pdf_style(view).map_err(access)?;
            if allow_rich {
                for style in [composition_style.as_ref(), paragraph_style.as_ref()]
                    .into_iter()
                    .flatten()
                {
                    let font_id = style.font_id(view).map_err(access)?;
                    if let Err(error) = require_font_style(font_styles, font_id.clone()) {
                        deferred_missing_font_style.get_or_insert(error);
                        return Ok(None);
                    }
                }
            }
            let equivalent = style_matches(
                composition_style.clone(),
                paragraph_style.clone(),
                view,
                false,
                false,
                mapper,
                font_styles,
            )? || style_matches(
                composition_style.clone(),
                paragraph_style.clone(),
                view,
                true,
                false,
                mapper,
                font_styles,
            )? || (style_matches(
                composition_style,
                paragraph_style,
                view,
                false,
                true,
                mapper,
                font_styles,
            )?);
            if allow_rich && !equivalent {
                let (left, left_pattern) = placeholder_result(
                    service,
                    "get_rich_text_left_placeholder",
                    &next_id.to_string(),
                )?;
                let (right, right_pattern) = placeholder_result(
                    service,
                    "get_rich_text_right_placeholder",
                    &next_id.to_string(),
                )?;
                let placeholder = stage::rich_placeholder(
                    regex,
                    &unicode,
                    next_id,
                    index,
                    |id| {
                        let (left, left_pattern) = placeholder_result(
                            service,
                            "get_rich_text_left_placeholder",
                            &id.to_string(),
                        )?;
                        let (right, right_pattern) = placeholder_result(
                            service,
                            "get_rich_text_right_placeholder",
                            &id.to_string(),
                        )?;
                        Ok((left, right, left_pattern, right_pattern))
                    },
                    source_text,
                )
                .map_err(access)?;
                next_id = placeholder.id + 2;
                parts.push(InputTextPart::Text(placeholder.left.clone()));
                parts.extend(character_fragments(
                    group_characters.into_iter().map(Ok),
                    view,
                )?);
                parts.push(InputTextPart::Text(placeholder.right.clone()));
                placeholders.push(stage::InputPlaceholder::Rich(placeholder));
                let _ = (left, left_pattern, right, right_pattern);
            } else {
                parts.extend(character_fragments(
                    group_characters.into_iter().map(Ok),
                    view,
                )?);
            }
        } else {
            return Ok(None);
        }
        if placeholders.len() > 40 && allow_rich {
            return build_input(
                paragraph,
                view,
                service,
                regex,
                patterns,
                false,
                mapper,
                font_styles,
                unicode_override,
                deferred_missing_font_style,
            );
        }
    }
    let translated =
        stage::prepare_translated_text(&[compose_input_text(&parts)], placeholders, originals);
    Ok(Some(translated))
}

const CONTENT_FILTER_HINT: &str =
    "翻译服务检测到内容可能包含不安全或敏感内容，请您避免翻译敏感内容，感谢您的配合。";

fn is_cid_placeholder(value: &str) -> bool {
    let Some(digits) = value
        .strip_prefix("(cid:")
        .and_then(|s| s.strip_suffix(')'))
    else {
        return false;
    };
    !digits.is_empty() && digits.chars().all(|ch| ch.to_digit(10).is_some())
}

fn python_string_repr(value: &str) -> String {
    let quote = if value.contains('\'') && !value.contains('"') {
        '"'
    } else {
        '\''
    };
    let mut result = String::with_capacity(value.len() + 2);
    result.push(quote);
    for ch in value.chars() {
        match ch {
            '\\' => result.push_str("\\\\"),
            '\n' => result.push_str("\\n"),
            '\r' => result.push_str("\\r"),
            '\t' => result.push_str("\\t"),
            ch if ch == quote => {
                result.push('\\');
                result.push(ch);
            }
            ch if ch.is_control() || ('\u{e000}'..='\u{f8ff}').contains(&ch) => {
                result.push_str(&format!("\\u{:04x}", ch as u32));
            }
            ch => result.push(ch),
        }
    }
    result.push(quote);
    result
}

pub(crate) fn formula_hint(input: &stage::TranslateInput) -> Option<String> {
    let entries = input
        .placeholders
        .iter()
        .filter_map(|placeholder| match placeholder {
            stage::InputPlaceholder::Formula(formula) if formula.hint_source => Some(format!(
                "{}: {}",
                python_string_repr(&formula.token),
                python_string_repr(&formula.source_chars)
            )),
            _ => None,
        })
        .collect::<Vec<_>>();
    (!entries.is_empty()).then(|| format!("{{{}}}", entries.join(", ")))
}

pub(crate) fn tracking_placeholders(
    input: &stage::TranslateInput,
) -> Vec<stage::PlaceholderRecord> {
    input
        .placeholders
        .iter()
        .map(|placeholder| match placeholder {
            stage::InputPlaceholder::Formula(formula) => stage::PlaceholderRecord::Formula {
                id: formula.id,
                text: formula.token.clone(),
                pattern: formula.pattern.clone(),
                chars: Some(formula.source_chars.clone()),
            },
            stage::InputPlaceholder::Rich(rich) => stage::PlaceholderRecord::RichText {
                id: rich.id,
                left: rich.left.clone(),
                right: rich.right.clone(),
                left_pattern: rich.left_pattern.clone(),
                right_pattern: rich.right_pattern.clone(),
                chars: Some(rich.source_text.clone()),
            },
        })
        .collect()
}

pub(crate) fn access(error: impl std::fmt::Display) -> PyErr {
    PyRuntimeError::new_err(error.to_string())
}

pub(crate) fn apply_llm_output(
    document: &mut Document,
    page_index: usize,
    paragraph_index: usize,
    input: &stage::TranslateInput,
    output: &str,
    regex_service: &Bound<'_, PyAny>,
    hallucination_ids: &[u64],
) -> PyResult<(String, bool, Vec<(String, usize)>)> {
    let text = stage::normalize_response_punctuation(output);
    let (parsed, full_match, removed) = Python::attach(|_py| -> PyResult<_> {
        let mut regex = RegexService {
            service: regex_service.clone(),
        };
        let mut removed = Vec::new();
        let (parsed, full) =
            stage::parse_output(&mut regex, input, &text, hallucination_ids, &mut removed)?;
        Ok((parsed, full, removed))
    })?;
    let mut view = document.view::<TranslationView>();
    let paragraph = view
        .document()
        .page_at(&view, page_index)
        .map_err(access)?
        .pdf_paragraph_at(&view, paragraph_index)
        .map_err(access)?;
    // Python 0.6.4 compares the output string with the TranslateInput object;
    // that branch is unreachable and the paragraph is always rebuilt.
    paragraph
        .set_unicode(&mut view, Some(text.clone()))
        .map_err(access)?;
    let mut compositions = Vec::new();
    for part in parsed {
        let composition = match part {
            stage::ParsedPart::Text(value) => {
                let style = paragraph
                    .pdf_style(&view)
                    .map_err(access)?
                    .map(|value| value.share(&mut view))
                    .transpose()
                    .map_err(access)?;
                let same = ir::PdfSameStyleUnicodeCharactersInit {
                    pdf_style: style,
                    unicode: Some(value),
                    debug_info: None,
                }
                .into_value();
                ir::PdfParagraphCompositionInit {
                    pdf_same_style_unicode_characters: Some(same),
                    ..Default::default()
                }
                .into_value()
            }
            stage::ParsedPart::Formula { composition_index } => {
                let old = paragraph
                    .pdf_paragraph_composition_at(&view, composition_index)
                    .map_err(access)?;
                let formula = old.pdf_formula(&view).map_err(access)?.ok_or_else(|| {
                    PyRuntimeError::new_err("formula placeholder source disappeared")
                })?;
                ir::PdfParagraphCompositionInit {
                    pdf_formula: Some(formula.share(&mut view).map_err(access)?),
                    ..Default::default()
                }
                .into_value()
            }
            stage::ParsedPart::Rich {
                composition_index,
                text: value,
                matches_source,
            } => {
                let old = paragraph
                    .pdf_paragraph_composition_at(&view, composition_index)
                    .map_err(access)?;
                let rich = old
                    .pdf_same_style_characters(&view)
                    .map_err(access)?
                    .ok_or_else(|| {
                        PyRuntimeError::new_err("rich placeholder source disappeared")
                    })?;
                if !matches_source {
                    let style = rich
                        .pdf_style(&view)
                        .map_err(access)?
                        .map(|value| value.share(&mut view))
                        .transpose()
                        .map_err(access)?;
                    let same = ir::PdfSameStyleUnicodeCharactersInit {
                        pdf_style: style,
                        unicode: Some(value),
                        debug_info: None,
                    }
                    .into_value();
                    ir::PdfParagraphCompositionInit {
                        pdf_same_style_unicode_characters: Some(same),
                        ..Default::default()
                    }
                    .into_value()
                } else {
                    ir::PdfParagraphCompositionInit {
                        pdf_same_style_characters: Some(rich.share(&mut view).map_err(access)?),
                        ..Default::default()
                    }
                    .into_value()
                }
            }
        };
        compositions.push(composition);
    }
    paragraph
        .set_pdf_paragraph_composition(&mut view, compositions)
        .map_err(access)?;
    Ok((text, full_match, removed))
}

fn add_page_font_styles(
    styles: &mut HashMap<String, babeldoc_p2_stages::font_mapper::SourceFontStyle>,
    fonts: impl Iterator<
        Item = Result<ir::Cursor<babeldoc_ir_core::types::PdfFont>, babeldoc_ir_core::AccessError>,
    >,
    view: &TreeView<'_, impl Permission>,
) -> PyResult<()> {
    for font in fonts {
        let font = font.map_err(access)?;
        let Some(font_id) = font.font_id(view).map_err(access)? else {
            continue;
        };
        styles.insert(
            font_id.clone(),
            babeldoc_p2_stages::font_mapper::SourceFontStyle {
                bold: font.bold(view).map_err(access)?.as_ref().is_some_and(|v| {
                    matches!(
                        v,
                        babeldoc_il_types::RuntimeBool::Bool(true)
                            | babeldoc_il_types::RuntimeBool::Int(1)
                    )
                }),
                italic: font
                    .italic(view)
                    .map_err(access)?
                    .as_ref()
                    .is_some_and(|v| {
                        matches!(
                            v,
                            babeldoc_il_types::RuntimeBool::Bool(true)
                                | babeldoc_il_types::RuntimeBool::Int(1)
                        )
                    }),
                monospaced: font
                    .monospace(view)
                    .map_err(access)?
                    .as_ref()
                    .is_some_and(|v| {
                        matches!(
                            v,
                            babeldoc_il_types::RuntimeBool::Bool(true)
                                | babeldoc_il_types::RuntimeBool::Int(1)
                        )
                    }),
                serif: font.serif(view).map_err(access)?.as_ref().is_some_and(|v| {
                    matches!(
                        v,
                        babeldoc_il_types::RuntimeBool::Bool(true)
                            | babeldoc_il_types::RuntimeBool::Int(1)
                    )
                }),
            },
        );
    }
    Ok(())
}

type FontStyles = HashMap<String, babeldoc_p2_stages::font_mapper::SourceFontStyle>;

#[derive(Clone, Default)]
struct PageFontMaps {
    fonts: FontStyles,
    xobjects: HashMap<Option<i64>, FontStyles>,
}

impl PageFontMaps {
    fn for_xobject(&self, xobj_id: Option<i64>) -> &FontStyles {
        self.xobjects.get(&xobj_id).unwrap_or(&self.fonts)
    }
}

fn merge_cross_page_font_maps(curr: &PageFontMaps, next: &PageFontMaps) -> PageFontMaps {
    let mut merged = curr.clone();
    merged.fonts.extend(next.fonts.clone());
    merged.xobjects.extend(next.xobjects.clone());
    merged
}

fn page_font_maps_for(
    view: &babeldoc_ir_core::TreeView<'_, ReadOnlyView>,
    page_index: usize,
    needed_xobjects: &HashSet<Option<i64>>,
) -> PyResult<PageFontMaps> {
    let page = view.document().page_at(view, page_index).map_err(access)?;
    let mut fonts = HashMap::new();
    add_page_font_styles(&mut fonts, page.pdf_font_iter(view).map_err(access)?, view)?;
    let mut xobjects = HashMap::new();
    for xobject in page.pdf_xobject_iter(view).map_err(access)? {
        let xobject = xobject.map_err(access)?;
        let id = xobject.xobj_id(view).map_err(access)?;
        if !needed_xobjects.contains(&id) {
            continue;
        }
        let mut xobject_fonts = fonts.clone();
        add_page_font_styles(
            &mut xobject_fonts,
            xobject.pdf_font_iter(view).map_err(access)?,
            view,
        )?;
        xobjects.insert(id, xobject_fonts);
    }
    Ok(PageFontMaps { fonts, xobjects })
}

/// Rebuild only inputs selected for cross-page LLM batches with the same
/// current-then-next page font maps that Python 0.6.4 passes to the batch.
pub(crate) fn rebuild_cross_page_inputs(
    document: &mut Document,
    jobs: &mut [Job],
    pairs: &[[(usize, usize); 2]],
    service: &Bound<'_, PyAny>,
    regex: &mut RegexService<'_>,
    patterns: &stage::PlaceholderPatterns,
    allow_rich: bool,
    mapper: &FontMapperHandle,
) -> PyResult<()> {
    if pairs.is_empty() {
        return Ok(());
    }

    let view = document.view::<ReadOnlyView>();
    let job_indices = jobs
        .iter()
        .enumerate()
        .map(|(index, job)| ((job.page, job.paragraph), index))
        .collect::<HashMap<_, _>>();

    for pair in pairs {
        let mut needed_xobjects = HashSet::new();
        for (page_index, paragraph_index) in pair {
            let page = view
                .document()
                .page_at(&view, *page_index)
                .map_err(access)?;
            let paragraph = page
                .pdf_paragraph_at(&view, *paragraph_index)
                .map_err(access)?;
            needed_xobjects.insert(paragraph.xobj_id(&view).map_err(access)?);
        }
        let current = page_font_maps_for(&view, pair[0].0, &needed_xobjects)?;
        let next = page_font_maps_for(&view, pair[1].0, &needed_xobjects)?;
        let merged = merge_cross_page_font_maps(&current, &next);

        for (page_index, paragraph_index) in pair {
            let page = view
                .document()
                .page_at(&view, *page_index)
                .map_err(access)?;
            let paragraph = page
                .pdf_paragraph_at(&view, *paragraph_index)
                .map_err(access)?;
            let is_vertical = paragraph.vertical(&view).map_err(access)?.unwrap_or(false);
            let Some(index) = job_indices.get(&(*page_index, *paragraph_index)) else {
                continue;
            };
            let xobj_id = paragraph.xobj_id(&view).map_err(access)?;
            let font_styles = merged.for_xobject(xobj_id);
            let unicode = paragraph.unicode(&view).map_err(access)?.clone();
            let tracker_unicode = jobs[*index].tracker_unicode.clone();
            rebuild_job_inputs(&mut jobs[*index], is_vertical, || {
                let mut deferred_missing_font_style = None;
                let input = build_input(
                    &paragraph,
                    &view,
                    service,
                    regex,
                    patterns,
                    allow_rich,
                    mapper,
                    font_styles,
                    unicode.as_deref(),
                    &mut deferred_missing_font_style,
                )?;
                let missing_font_style = deferred_missing_font_style;
                let mut deferred_fallback_font_style = None;
                let fallback_input = build_input(
                    &paragraph,
                    &view,
                    service,
                    regex,
                    patterns,
                    allow_rich,
                    mapper,
                    font_styles,
                    Some(&tracker_unicode),
                    &mut deferred_fallback_font_style,
                )?;
                Ok((
                    input,
                    fallback_input,
                    missing_font_style,
                    deferred_fallback_font_style,
                ))
            })?;
        }
    }
    Ok(())
}

#[derive(Clone)]
pub(crate) struct Job {
    pub(crate) page: usize,
    pub(crate) paragraph: usize,
    pub(crate) debug_id: Option<String>,
    pub(crate) text: String,
    pub(crate) tracker_unicode: String,
    pub(crate) vertical: bool,
    pub(crate) unicode_present: bool,
    pub(crate) layout_label: Option<String>,
    pub(crate) cid: bool,
    pub(crate) pure_numeric: bool,
    pub(crate) placeholder_only: bool,
    pub(crate) box_y2: Option<f64>,
    pub(crate) title_paragraph: bool,
    pub(crate) token_count: usize,
    pub(crate) title: Option<(String, String)>,
    pub(crate) local_title: Option<(String, String)>,
    pub(crate) eligible: bool,
    pub(crate) input: Option<stage::TranslateInput>,
    pub(crate) fallback_input: Option<stage::TranslateInput>,
    pub(crate) missing_font_style: Option<FontStyleLookupError>,
    pub(crate) fallback_missing_font_style: Option<FontStyleLookupError>,
}

fn rebuild_job_inputs(
    job: &mut Job,
    vertical: bool,
    build: impl FnOnce() -> PyResult<(
        Option<stage::TranslateInput>,
        Option<stage::TranslateInput>,
        Option<FontStyleLookupError>,
        Option<FontStyleLookupError>,
    )>,
) -> PyResult<()> {
    if vertical || job.vertical {
        job.eligible = false;
        job.input = None;
        job.fallback_input = None;
        job.missing_font_style = None;
        job.fallback_missing_font_style = None;
        return Ok(());
    }
    let (input, fallback_input, missing_font_style, fallback_missing_font_style) = build()?;
    job.eligible = input.is_some();
    job.input = input;
    job.fallback_input = fallback_input;
    job.missing_font_style = missing_font_style;
    job.fallback_missing_font_style = fallback_missing_font_style;
    Ok(())
}

#[derive(Clone)]
enum InputTextPart {
    Character {
        text: String,
        coords: Option<(f64, f64, f64, f64)>,
    },
    Text(String),
}

fn character_fragments(
    characters: impl IntoIterator<
        Item = Result<
            ir::Cursor<babeldoc_ir_core::types::PdfCharacter>,
            babeldoc_ir_core::AccessError,
        >,
    >,
    view: &TreeView<'_, impl Permission>,
) -> PyResult<Vec<InputTextPart>> {
    characters
        .into_iter()
        .map(|character| {
            let character = character.map_err(access)?;
            let text = character
                .char_unicode(view)
                .map_err(access)?
                .clone()
                .unwrap_or_default();
            let coords = character
                .r#box(view)
                .map_err(access)?
                .map(|box_| {
                    Ok::<_, PyErr>((
                        box_.x(view).map_err(access)?.unwrap_or_default(),
                        box_.y(view).map_err(access)?.unwrap_or_default(),
                        box_.x2(view).map_err(access)?.unwrap_or_default(),
                        box_.y2(view).map_err(access)?.unwrap_or_default(),
                    ))
                })
                .transpose()?;
            Ok(InputTextPart::Character { text, coords })
        })
        .collect()
}

fn python_whitespace(ch: char) -> bool {
    ch.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&ch)
}

fn normalize_character_text(text: &str) -> String {
    let normalized = babeldoc_p2_stages::shared::layout_helper::normalize_unicode_nfkc(text);
    let mut output = String::new();
    let mut in_space = false;
    for ch in normalized.chars() {
        if python_whitespace(ch) {
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

fn compose_input_text(parts: &[InputTextPart]) -> String {
    let mut distances = parts
        .windows(2)
        .filter_map(|pair| match (&pair[0], &pair[1]) {
            (
                InputTextPart::Character {
                    coords: Some((_, _, first_x2, _)),
                    ..
                },
                InputTextPart::Character {
                    coords: Some((second_x, _, _, _)),
                    ..
                },
            ) => {
                let distance = second_x - first_x2;
                (distance > 1.0).then_some(distance)
            }
            _ => None,
        })
        .collect::<Vec<_>>();
    distances.sort_by(f64::total_cmp);
    distances.dedup_by(|left, right| left == right);
    let median = match distances.as_slice() {
        [] => 1.0,
        [only] => *only,
        many => many[1],
    };
    let mut output = String::new();
    for (index, part) in parts.iter().enumerate() {
        match part {
            InputTextPart::Text(text) => output.push_str(text),
            InputTextPart::Character { text, coords } => {
                output.push_str(&normalize_character_text(text));
                if text == " " {
                    continue;
                }
                if let (
                    Some((x, y, x2, y2)),
                    Some(InputTextPart::Character {
                        coords: Some((next_x, next_y, next_x2, next_y2)),
                        ..
                    }),
                ) = (coords, parts.get(index + 1))
                {
                    let distance = next_x - x2;
                    let char_height = (y2 - y).max(next_y2 - next_y);
                    let char_width = (x2 - x).max(next_x2 - next_x);
                    let newline = *next_y2 < *y || *next_x2 < *x - char_width * 10.0;
                    let _ = char_height;
                    if distance >= median || newline {
                        output.push(' ');
                    }
                }
            }
        }
    }
    let normalized = babeldoc_p2_stages::shared::layout_helper::normalize_unicode_nfkc(&output);
    let mut collapsed = String::new();
    let mut in_space = false;
    for ch in normalized.chars() {
        if python_whitespace(ch) {
            if !in_space {
                collapsed.push(' ');
            }
            in_space = true;
        } else {
            collapsed.push(ch);
            in_space = false;
        }
    }
    babeldoc_p2_stages::translation::python_semantics::python_strip(&collapsed).to_owned()
}

pub(crate) fn collect(
    document: &mut Document,
    root: &babeldoc_p2_stages::runtime::shared_context::SharedContextRoot,
    service: &Bound<'_, PyAny>,
    regex: &mut RegexService<'_>,
    patterns: &stage::PlaceholderPatterns,
    allow_rich: bool,
    min_text_length: usize,
    llm_only_recent_title_admission: bool,
    mapper: &crate::font_mapper::FontMapperHandle,
) -> PyResult<(Vec<Job>, Vec<usize>, usize)> {
    let view = document.view::<ReadOnlyView>();
    let mut jobs = Vec::new();
    let mut count = 0;
    let mut page_lengths = Vec::new();
    let mut translated_ids = HashSet::new();
    let mut first = root.first_paragraph().map(|title| {
        (
            title.debug_id.unwrap_or_default(),
            title_hint_text(title.unicode),
        )
    });
    let mut recent = root.recent_title_paragraph().map(|title| {
        (
            title.debug_id.unwrap_or_default(),
            title_hint_text(title.unicode),
        )
    });
    if first.is_none() {
        'find_first: for page in view.document().page_iter(&view).map_err(access)? {
            let page = page.map_err(access)?;
            for paragraph in page.pdf_paragraph_iter(&view).map_err(access)? {
                let paragraph = paragraph.map_err(access)?;
                if paragraph.layout_label(&view).map_err(access)?.as_deref() == Some("title") {
                    let snapshot =
                        babeldoc_p2_stages::runtime::shared_context::TitleContextSnapshot {
                            debug_id: paragraph.debug_id(&view).map_err(access)?.clone(),
                            unicode: paragraph.unicode(&view).map_err(access)?.clone(),
                            layout_label: Some("title".into()),
                        };
                    let title = (
                        snapshot.debug_id.clone().unwrap_or_default(),
                        title_hint_text(snapshot.unicode.clone()),
                    );
                    // The initial title belongs to the document root, not just
                    // this part's job builder. Later parts must retain it.
                    root.set_first_paragraph(Some(snapshot.clone()));
                    root.set_recent_title_paragraph(Some(snapshot));
                    first = Some(title.clone());
                    recent = Some(title);
                    break 'find_first;
                }
            }
        }
    }
    for (page_index, page) in view
        .document()
        .page_iter(&view)
        .map_err(access)?
        .enumerate()
    {
        let page = page.map_err(access)?;
        let paragraph_len = page.pdf_paragraph_len(&view).map_err(access)?;
        page_lengths.push(paragraph_len);
        let mut font_styles = HashMap::new();
        add_page_font_styles(
            &mut font_styles,
            page.pdf_font_iter(&view).map_err(access)?,
            &view,
        )?;
        for (paragraph_index, paragraph) in
            page.pdf_paragraph_iter(&view).map_err(access)?.enumerate()
        {
            let paragraph = paragraph.map_err(access)?;
            count += 1;
            let unicode = paragraph.unicode(&view).map_err(access)?;
            let tracker_unicode = if unicode.is_some() {
                paragraph_unicode(&paragraph, &view)?
            } else {
                String::new()
            };
            let vertical = paragraph.vertical(&view).map_err(access)?.unwrap_or(false);
            let label = paragraph.layout_label(&view).map_err(access)?;
            let debug_id = paragraph.debug_id(&view).map_err(access)?.clone();
            let paragraph_box = paragraph.r#box(&view).map_err(access)?;
            let box_y2 = paragraph_box
                .map(|value| value.y2(&view).map_err(access))
                .transpose()?
                .flatten();
            let mut cid_count = 0usize;
            let mut cid_total = 0usize;
            let mut placeholder_only = unicode.as_deref().is_some_and(|text| !text.is_empty());
            let mut count_character = |character: &ir::Cursor<
                babeldoc_ir_core::types::PdfCharacter,
            >| {
                cid_total += 1;
                let text = character
                    .char_unicode(&view)
                    .ok()
                    .and_then(|value| value.as_deref());
                if text.as_deref().is_some_and(|value| {
                    value
                        .strip_prefix("(cid:")
                        .and_then(|value| value.strip_suffix(')'))
                        .is_some_and(|digits| {
                            !digits.is_empty() && digits.bytes().all(|byte| byte.is_ascii_digit())
                        })
                }) {
                    cid_count += 1;
                }
            };
            for composition in paragraph
                .pdf_paragraph_composition_iter(&view)
                .map_err(access)?
            {
                let composition = composition.map_err(access)?;
                if let Some(line) = composition.pdf_line(&view).map_err(access)? {
                    for character in line.pdf_character_iter(&view).map_err(access)? {
                        let character = character.map_err(access)?;
                        placeholder_only &= character
                            .char_unicode(&view)
                            .ok()
                            .and_then(|value| value.as_deref())
                            .as_deref()
                            .is_some_and(
                                babeldoc_p2_stages::translation::python_semantics::python_isspace,
                            );
                        count_character(&character);
                    }
                } else if let Some(group) = composition
                    .pdf_same_style_characters(&view)
                    .map_err(access)?
                {
                    for character in group.pdf_character_iter(&view).map_err(access)? {
                        let character = character.map_err(access)?;
                        placeholder_only &= character
                            .char_unicode(&view)
                            .ok()
                            .and_then(|value| value.as_deref())
                            .as_deref()
                            .is_some_and(
                                babeldoc_p2_stages::translation::python_semantics::python_isspace,
                            );
                        count_character(&character);
                    }
                } else if let Some(group) = composition
                    .pdf_same_style_unicode_characters(&view)
                    .map_err(access)?
                {
                    placeholder_only &= group
                        .unicode(&view)
                        .map_err(access)?
                        .as_deref()
                        .is_some_and(
                            babeldoc_p2_stages::translation::python_semantics::python_isspace,
                        );
                } else if let Some(formula) = composition.pdf_formula(&view).map_err(access)? {
                    for character in formula.pdf_character_iter(&view).map_err(access)? {
                        count_character(&character.map_err(access)?);
                    }
                } else if let Some(character) = composition.pdf_character(&view).map_err(access)? {
                    placeholder_only &= character
                        .char_unicode(&view)
                        .map_err(access)?
                        .as_deref()
                        .is_some_and(
                            babeldoc_p2_stages::translation::python_semantics::python_isspace,
                        );
                    count_character(&character);
                } else {
                    placeholder_only = false;
                }
            }
            let cid = cid_count * 5 > cid_total * 4;
            let title_paragraph = label.as_deref() == Some("title");
            let pure_numeric = unicode.as_deref().is_some_and(
                babeldoc_p2_stages::translation::paragraph_helpers::is_pure_numeric_text,
            );
            let mut deferred_missing_font_style = None;
            if title_paragraph && !llm_only_recent_title_admission {
                let title_id = debug_id.clone().unwrap_or_default();
                recent = Some((title_id.clone(), title_hint_text(unicode.clone())));
                root.set_recent_title_paragraph(Some(
                    babeldoc_p2_stages::runtime::shared_context::TitleContextSnapshot {
                        debug_id: Some(title_id),
                        unicode: unicode.clone(),
                        layout_label: label.clone(),
                    },
                ));
            }
            if first.is_none() && title_paragraph {
                let snapshot = babeldoc_p2_stages::runtime::shared_context::TitleContextSnapshot {
                    debug_id: debug_id.clone(),
                    unicode: unicode.clone(),
                    layout_label: label.clone(),
                };
                if let Some(first_snapshot) =
                    initialize_first_title(&mut first, &mut recent, snapshot)
                {
                    root.set_first_paragraph(Some(first_snapshot.clone()));
                    root.set_recent_title_paragraph(Some(first_snapshot));
                }
            }
            if let Some(text) = unicode.clone() {
                let (input, fallback_input, fallback_missing_font_style) = if vertical {
                    (None, None, None)
                } else {
                    let xobj_id = paragraph.xobj_id(&view).map_err(access)?;
                    let mut paragraph_font_styles = font_styles.clone();
                    for xobj in page.pdf_xobject_iter(&view).map_err(access)? {
                        let xobj = xobj.map_err(access)?;
                        if xobj.xobj_id(&view).map_err(access)? == xobj_id {
                            add_page_font_styles(
                                &mut paragraph_font_styles,
                                xobj.pdf_font_iter(&view).map_err(access)?,
                                &view,
                            )?;
                            break;
                        }
                    }
                    let input = build_input(
                        &paragraph,
                        &view,
                        service,
                        regex,
                        patterns,
                        allow_rich,
                        mapper,
                        &paragraph_font_styles,
                        Some(&text),
                        &mut deferred_missing_font_style,
                    )?;
                    let mut deferred_fallback_font_style = None;
                    let fallback_input = build_input(
                        &paragraph,
                        &view,
                        service,
                        regex,
                        patterns,
                        allow_rich,
                        mapper,
                        &paragraph_font_styles,
                        Some(&tracker_unicode),
                        &mut deferred_fallback_font_style,
                    )?;
                    (input, fallback_input, deferred_fallback_font_style)
                };
                let eligible = input.is_some();
                let passes_llm_filters = debug_id.is_some()
                    && text.chars().count() >= min_text_length
                    && !cid
                    && !pure_numeric
                    && !placeholder_only;
                let first_occurrence =
                    passes_llm_filters && translated_ids.insert(paragraph.identity());
                let admitted = first_occurrence;
                if title_paragraph && llm_only_recent_title_admission && admitted {
                    recent = Some((debug_id.clone().unwrap_or_default(), text.clone()));
                    root.set_recent_title_paragraph(Some(
                        babeldoc_p2_stages::runtime::shared_context::TitleContextSnapshot {
                            debug_id: debug_id.clone(),
                            unicode: Some(text.clone()),
                            layout_label: label.clone(),
                        },
                    ));
                }
                jobs.push(Job {
                    page: page_index,
                    paragraph: paragraph_index,
                    debug_id: paragraph.debug_id(&view).map_err(access)?.clone(),
                    token_count: babeldoc_p2_stages::translation::token_count::count_tokens(&text),
                    unicode_present: true,
                    layout_label: label.clone(),
                    cid,
                    pure_numeric,
                    placeholder_only,
                    box_y2,
                    title_paragraph,
                    text,
                    tracker_unicode,
                    vertical,
                    title: first.clone(),
                    local_title: recent.clone(),
                    eligible,
                    input,
                    fallback_input,
                    missing_font_style: deferred_missing_font_style,
                    fallback_missing_font_style,
                });
            } else {
                jobs.push(Job {
                    page: page_index,
                    paragraph: paragraph_index,
                    debug_id: paragraph.debug_id(&view).map_err(access)?.clone(),
                    token_count: 0,
                    unicode_present: false,
                    layout_label: label.clone(),
                    cid,
                    pure_numeric: false,
                    placeholder_only,
                    box_y2,
                    title_paragraph,
                    text: String::new(),
                    tracker_unicode: String::new(),
                    vertical,
                    title: first.clone(),
                    local_title: recent.clone(),
                    eligible: false,
                    input: None,
                    fallback_input: None,
                    missing_font_style: None,
                    fallback_missing_font_style: None,
                });
            }
        }
    }
    Ok((jobs, page_lengths, count))
}

pub fn process(
    py: Python<'_>,
    document: &mut Document,
    config: &Bound<'_, PyAny>,
    shared_context: &Bound<'_, PyAny>,
    mapper: &FontMapperHandle,
    dispatcher: Py<PyAny>,
    translator_service: Py<PyAny>,
    flag: CancelFlag,
    monitor: Py<PyAny>,
) -> PyResult<()> {
    let _native_font_mapper = &mapper.mapper;
    let root = crate::shared_context::root_from_handle(shared_context)?;
    let translator = translator_service.bind(py).clone();
    let supports_llm = translator
        .call_method0("probe_il_translator")?
        .extract::<bool>()?;
    let disable_rich = config
        .getattr("disable_rich_text_translate")?
        .extract::<bool>()?;
    let allow_rich = supports_llm && !disable_rich;
    let mut regex = RegexService {
        service: translator.clone(),
    };
    // BaseTranslator returns strings. Its [1] indexing is part of 0.6.4's
    // d10 behavior; OpenAI shaped returns expose the regex tuple member.
    let formula_pattern = legacy_pattern(&translator, "get_formular_placeholder")?;
    let left_pattern = legacy_pattern(&translator, "get_rich_text_left_placeholder")?;
    let right_pattern = legacy_pattern(&translator, "get_rich_text_right_placeholder")?;
    let patterns = stage::PlaceholderPatterns {
        formula: formula_pattern,
        rich_left: left_pattern,
        rich_right: right_pattern,
    };
    let hallucination_ids = [
        regex.compile(&patterns.formula, true)?,
        regex.compile(&patterns.rich_left, true)?,
        regex.compile(&patterns.rich_right, true)?,
    ];
    let min_length = config.getattr("min_text_length")?.extract::<usize>()?;
    let (jobs, page_lengths, total) = collect(
        document,
        &root,
        &translator,
        &mut regex,
        &patterns,
        allow_rich,
        min_length,
        false,
        mapper,
    )?;
    let workers = config
        .getattr("pool_max_workers")?
        .extract::<usize>()?
        .max(1);
    let lang_out = config.getattr("lang_out")?.extract::<String>()?;
    let custom_prompt = config
        .getattr("custom_system_prompt")
        .ok()
        .and_then(|v| v.extract::<String>().ok());
    let add_formula_hint = config
        .getattr("add_formula_placehold_hint")
        .and_then(|value| value.extract::<bool>())
        .unwrap_or(false);
    let auto_extract = config
        .getattr("auto_extract_glossary")?
        .extract::<bool>()
        .unwrap_or(false);
    let tracking_path = config
        .call_method1("get_working_file_path", ("translate_tracking.json",))?
        .str()?
        .to_str()?
        .to_owned();
    let write_tracking =
        config.getattr("debug")?.extract::<bool>()? || !config.getattr("working_dir")?.is_none();
    let glossaries = root.get_glossaries_for_translation(auto_extract);
    let roles = stage::role_block(custom_prompt.as_deref(), &lang_out);
    let regex_service_for_parse = translator_service.clone_ref(py);
    let recorder = Recorder::for_dispatcher(py, dispatcher.bind(py));
    let service = PythonService {
        dispatcher,
        recorder,
    };
    let pool = Arc::new(TranslatorPoolPair::new(
        service,
        stage::STAGE_NAME,
        workers,
        root,
    ));
    let result = callbacks::run(py, workers, move |runtime, progress_run| {
        let progress = progress_run
            .start(
                monitor,
                stage::STAGE_NAME,
                total as u64,
                Duration::from_millis(30),
            )
            .map_err(ProgressError::into_python)?;
        let result = runtime.block_on(async {
            let mut tasks = tokio::task::JoinSet::new();
            let mut tracking = stage::TranslationTracking {
                page: page_lengths
                    .iter()
                    .map(|count| stage::PageTracking {
                        paragraph: vec![stage::ParagraphTracking::default(); *count],
                    })
                    .collect(),
            };
            let mut page_error = None;
            let mut current_page = None;
            let mut submitted = 0usize;
            let mut translated = Vec::new();
            let mut admitted_tracking = Vec::new();
            for job in jobs {
                if current_page != Some(job.page) {
                    current_page = Some(job.page);
                    if let Err(error) = flag.checkpoint() {
                        page_error = Some(error);
                        break;
                    }
                }
                let page = job.page;
                let slot = &mut tracking.page[page].paragraph[job.paragraph];
                if job.missing_font_style.is_some() {
                    progress.report(1, None).map_err(|e| e.into_python())?;
                    continue;
                }
                if !job.eligible {
                    progress.report(1, None).map_err(|e| e.into_python())?;
                    continue;
                }
                let input = job
                    .input
                    .as_ref()
                    .expect("eligible job has a translate input");
                if input.text.chars().count() < min_length {
                    slot.pdf_unicode = Some(job.text.clone());
                    slot.input = Some(input.text.clone());
                    slot.original_placeholders = input.original_tokens.clone();
                    slot.placeholders = tracking_placeholders(input);
                    progress.report(1, None).map_err(|e| e.into_python())?;
                    continue;
                }
                let admitted = Arc::new(AtomicBool::new(false));
                admitted_tracking.push((
                    job.page,
                    job.paragraph,
                    job.text.clone(),
                    input.text.clone(),
                    input.original_tokens.clone(),
                    tracking_placeholders(input),
                    Arc::clone(&admitted),
                ));
                slot.llm_translate_trackers
                    .push(stage::LlmTracking::default());
                if supports_llm {
                    let context = stage::context_block(
                        job.title
                            .as_ref()
                            .map(|(id, text)| (id.as_str(), text.as_str())),
                        job.local_title
                            .as_ref()
                            .map(|(id, text)| (id.as_str(), text.as_str())),
                        add_formula_hint
                            .then(|| formula_hint(input))
                            .flatten()
                            .as_deref(),
                    );
                    let prompt = stage::render_prompt(
                        &roles,
                        &lang_out,
                        &stage::glossary_block(&input.text, &glossaries),
                        &context,
                        &input.text,
                    );
                    if let Some(track) = slot.llm_translate_trackers.last_mut() {
                        track.input = prompt.clone();
                    }
                    let admitted = spawn_request(
                        &mut tasks,
                        Arc::clone(&pool),
                        job,
                        prompt,
                        true,
                        flag.clone(),
                        progress.clone(),
                        admitted,
                    );
                    submitted += 1;
                    if submitted <= workers {
                        let _ = admitted.await;
                    }
                    tokio::task::yield_now().await;
                } else {
                    let admitted = spawn_request(
                        &mut tasks,
                        Arc::clone(&pool),
                        job.clone(),
                        input.text.clone(),
                        false,
                        flag.clone(),
                        progress.clone(),
                        admitted,
                    );
                    submitted += 1;
                    if submitted <= workers {
                        let _ = admitted.await;
                    }
                    tokio::task::yield_now().await;
                }
            }
            while let Some(joined) = tasks.join_next().await {
                match joined {
                    Ok(Ok(Some((job, response, content_filter)))) => {
                        translated.push((job, response, content_filter))
                    }
                    Ok(Ok(None)) => continue,
                    Ok(Err(error)) => return Err(error),
                    Err(error) => return Err(PyRuntimeError::new_err(error.to_string())),
                }
            }
            for (page, paragraph, pdf_unicode, input, originals, placeholders, admitted) in
                admitted_tracking
            {
                if admitted.load(Ordering::Acquire) {
                    let slot = &mut tracking.page[page].paragraph[paragraph];
                    slot.pdf_unicode = Some(pdf_unicode);
                    slot.input = Some(input);
                    slot.original_placeholders = originals;
                    slot.placeholders = placeholders;
                }
            }
            if let Some(error) = page_error {
                return Err(error);
            }
            // Apply completions after all worker jobs have finished. This keeps
            // the M tree serial while retaining independent paragraph requests.
            let mut view = document.view::<TranslationView>();
            for (job, response, content_filter) in translated {
                if content_filter {
                    let page = view.document().page_at(&view, job.page).map_err(access)?;
                    let paragraph = page
                        .pdf_paragraph_at(&view, job.paragraph)
                        .map_err(access)?;
                    let original_box = paragraph
                        .r#box(&view)
                        .map_err(|e| {
                            PyRuntimeError::new_err(format!("content filter paragraph box: {e}"))
                        })?
                        .ok_or_else(|| {
                            PyRuntimeError::new_err("content-filtered paragraph has no box")
                        })?;
                    let y2 = original_box
                        .y2(&view)
                        .map_err(|e| {
                            PyRuntimeError::new_err(format!("content filter box y2: {e}"))
                        })?
                        .ok_or_else(|| {
                            PyRuntimeError::new_err("content-filtered paragraph box has no y2")
                        })?;
                    let box_value = ir::BoxInit {
                        x: original_box.x(&view).map_err(|e| {
                            PyRuntimeError::new_err(format!("content filter box x: {e}"))
                        })?,
                        y: Some(y2),
                        x2: original_box.x2(&view).map_err(|e| {
                            PyRuntimeError::new_err(format!("content filter box x2: {e}"))
                        })?,
                        y2: Some(y2 + 1.1),
                    }
                    .into_value();
                    let hint_index = page.pdf_paragraph_len(&view).map_err(access)?;
                    let hint = ir::PdfParagraphInit {
                        r#box: Some(box_value),
                        pdf_style: Some(
                            ir::PdfStyleInit {
                                font_id: Some("base".into()),
                                font_size: Some(1.0),
                                graphic_state: Some(
                                    ir::GraphicStateInit {
                                        passthrough_per_char_instruction: Some(
                                            ir::Instructions::Materialized("0.80 g 0.80 G".into()),
                                        ),
                                    }
                                    .into_value(),
                                ),
                            }
                            .into_value(),
                        ),
                        xobj_id: Some(-1),
                        unicode: Some(CONTENT_FILTER_HINT.into()),
                        vertical: Some(false),
                        first_line_indent: Some(false),
                        ..Default::default()
                    }
                    .into_value();
                    page.push_pdf_paragraph(&mut view, hint).map_err(|e| {
                        PyRuntimeError::new_err(format!("content filter page append: {e}"))
                    })?;
                    let hint_paragraph =
                        page.pdf_paragraph_at(&view, hint_index).map_err(access)?;
                    let style = hint_paragraph
                        .pdf_style(&view)
                        .map_err(access)?
                        .ok_or_else(|| PyRuntimeError::new_err("content-filter style missing"))?
                        .share(&mut view)
                        .map_err(access)?;
                    let chars = ir::PdfSameStyleUnicodeCharactersInit {
                        unicode: Some(CONTENT_FILTER_HINT.into()),
                        pdf_style: Some(style),
                        debug_info: Some(true),
                    }
                    .into_value();
                    let composition = ir::PdfParagraphCompositionInit {
                        pdf_same_style_unicode_characters: Some(chars),
                        ..Default::default()
                    }
                    .into_value();
                    hint_paragraph
                        .set_pdf_paragraph_composition(&mut view, vec![composition])
                        .map_err(access)?;
                    continue;
                }
                if let Ok(text) = response {
                    let text = stage::normalize_response_punctuation(&text);
                    let Some(input) = job.input.as_ref() else {
                        continue;
                    };
                    let translation_unchanged = text == input.text;
                    let (parsed, placeholder_full_match, removed) =
                        Python::attach(|py| -> PyResult<_> {
                            let mut regex = RegexService {
                                service: regex_service_for_parse.bind(py).clone(),
                            };
                            let mut removed = Vec::new();
                            let (parsed, full) = stage::parse_output(
                                &mut regex,
                                input,
                                &text,
                                &hallucination_ids,
                                &mut removed,
                            )?;
                            Ok((parsed, full, removed))
                        })?;
                    let paragraph = view
                        .document()
                        .page_at(&view, job.page)
                        .map_err(access)?
                        .pdf_paragraph_at(&view, job.paragraph)
                        .map_err(access)?;
                    if text != input.text {
                        paragraph
                            .set_unicode(&mut view, Some(text.clone()))
                            .map_err(access)?;
                        let mut new_compositions = Vec::new();
                        for part in parsed {
                            let composition = match part {
                                stage::ParsedPart::Text(value) => {
                                    let style = paragraph
                                        .pdf_style(&view)
                                        .map_err(access)?
                                        .map(|c| c.share(&mut view))
                                        .transpose()
                                        .map_err(access)?;
                                    let same = ir::PdfSameStyleUnicodeCharactersInit {
                                        pdf_style: style,
                                        unicode: Some(value),
                                        debug_info: None,
                                    }
                                    .into_value();
                                    ir::PdfParagraphCompositionInit {
                                        pdf_same_style_unicode_characters: Some(same),
                                        ..Default::default()
                                    }
                                    .into_value()
                                }
                                stage::ParsedPart::Formula { composition_index } => {
                                    let old = paragraph
                                        .pdf_paragraph_composition_at(&view, composition_index)
                                        .map_err(access)?;
                                    let formula = old
                                        .pdf_formula(&view)
                                        .map_err(access)?
                                        .ok_or_else(|| {
                                            PyRuntimeError::new_err(
                                                "formula placeholder source disappeared",
                                            )
                                        })?;
                                    let formula = formula.share(&mut view).map_err(access)?;
                                    ir::PdfParagraphCompositionInit {
                                        pdf_formula: Some(formula),
                                        ..Default::default()
                                    }
                                    .into_value()
                                }
                                stage::ParsedPart::Rich {
                                    composition_index,
                                    text: value,
                                    matches_source,
                                } => {
                                    let old = paragraph
                                        .pdf_paragraph_composition_at(&view, composition_index)
                                        .map_err(access)?;
                                    let rich = old
                                        .pdf_same_style_characters(&view)
                                        .map_err(access)?
                                        .ok_or_else(|| {
                                            PyRuntimeError::new_err(
                                                "rich placeholder source disappeared",
                                            )
                                        })?;
                                    if !matches_source {
                                        let style = rich
                                            .pdf_style(&view)
                                            .map_err(access)?
                                            .map(|c| c.share(&mut view))
                                            .transpose()
                                            .map_err(access)?;
                                        let same = ir::PdfSameStyleUnicodeCharactersInit {
                                            pdf_style: style,
                                            unicode: Some(value),
                                            debug_info: None,
                                        }
                                        .into_value();
                                        ir::PdfParagraphCompositionInit {
                                            pdf_same_style_unicode_characters: Some(same),
                                            ..Default::default()
                                        }
                                        .into_value()
                                    } else {
                                        let rich = rich.share(&mut view).map_err(access)?;
                                        ir::PdfParagraphCompositionInit {
                                            pdf_same_style_characters: Some(rich),
                                            ..Default::default()
                                        }
                                        .into_value()
                                    }
                                }
                            };
                            new_compositions.push(composition);
                        }
                        paragraph
                            .set_pdf_paragraph_composition(&mut view, new_compositions)
                            .map_err(access)?;
                    }
                    if let Some(slot) = tracking
                        .page
                        .get_mut(job.page)
                        .and_then(|p| p.paragraph.get_mut(job.paragraph))
                    {
                        slot.output = Some(text);
                        slot.removed_hallucinated_placeholders = removed;
                        if let Some(last) = slot.llm_translate_trackers.last_mut() {
                            if supports_llm {
                                last.output = slot.output.clone().unwrap_or_default();
                            }
                            last.placeholder_full_match =
                                placeholder_full_match || translation_unchanged;
                        }
                    }
                }
            }
            Ok(tracking)
        });
        match result {
            Ok(tracking) => {
                progress.end(None).map_err(ProgressError::into_python)?;
                if write_tracking {
                    if let Some(parent) = std::path::Path::new(&tracking_path).parent() {
                        std::fs::create_dir_all(parent).map_err(access)?;
                    }
                    std::fs::write(&tracking_path, tracking.to_json()).map_err(access)
                } else {
                    Ok(())
                }
            }
            Err(error) => {
                let _ = progress.end(None);
                Err(error)
            }
        }
    });
    result
}

#[cfg(test)]
mod ordinary_translation_tests;

fn spawn_request(
    tasks: &mut tokio::task::JoinSet<PyResult<Option<(Job, PyResult<String>, bool)>>>,
    pool: Arc<TranslatorPoolPair>,
    job: Job,
    input: String,
    llm: bool,
    flag: CancelFlag,
    progress: PythonProgress,
    recorded: Arc<AtomicBool>,
) -> oneshot::Receiver<()> {
    let (admitted_tx, admitted_rx) = oneshot::channel();
    tasks.spawn(async move {
        let method = if llm { "llm_translate" } else { "translate" };
        let (args, kwargs) = Python::attach(|py| -> PyResult<_> {
            let args = PyTuple::new(py, [input.clone()])?.unbind();
            let params = PyDict::new(py);
            params.set_item("paragraph_token_count", job.token_count)?;
            let kwargs = PyDict::new(py);
            kwargs.set_item("rate_limit_params", params)?;
            Ok((args, kwargs.unbind()))
        })?;
        let priority = 1_048_576 - job.token_count as i64;
        let request_flag = flag.clone();
        let response = pool
            .request(
                method,
                format!("translate:{}:{}", job.page, job.paragraph),
                args,
                Some(kwargs),
                priority,
                move || {
                    let _ = admitted_tx.send(());
                    let allowed = !request_flag.is_cancelled();
                    if allowed {
                        recorded.store(true, Ordering::Release);
                    }
                    allowed
                },
            )
            .await;
        match response {
            Ok(Some(value)) => {
                progress
                    .report(1, None)
                    .map_err(|error| error.into_python())?;
                let result = Python::attach(|py| {
                    value
                        .bind(py)
                        .extract::<String>()
                        .map_err(|_| PyTypeError::new_err("translation response must be a string"))
                });
                Ok(Some((job, result, false)))
            }
            Ok(None) => Ok(None),
            Err(crate::callbacks::CallbackError::Python(error)) => {
                progress
                    .report(1, None)
                    .map_err(|error| error.into_python())?;
                let is_content_filter = Python::attach(|py| -> PyResult<bool> {
                    let class = py
                        .import("babeldoc.babeldoc_exception.BabelDOCException")?
                        .getattr("ContentFilterError")?;
                    Ok(error.is_instance(py, &class))
                })?;
                Ok(Some((job, Err(error), is_content_filter)))
            }
            Err(crate::callbacks::CallbackError::Bridge(message)) => {
                Err(crate::callbacks::BridgeError::new_err(message))
            }
        }
    });
    admitted_rx
}
