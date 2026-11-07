//! Serial formula/style classification on the M tree.
use super::styles_and_formulas_config::StylesAndFormulasConfig;
use super::tree_metrics::*;
use crate::shared::formula_helper as fh;
use crate::shared::layout_helper::{
    self as lh, LEFT_BRACKET, LayoutIndex, RIGHT_BRACKET, calculate_iou_for_boxes, is_bullet_point,
};
use crate::{StageContext, StageError};
use babeldoc_il_types::{Box as IlBox, PdfCharacter};
use babeldoc_ir_core::{self as ir, Cursor, StylesAndFormulasView, TreeView, types};
use log::debug;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::LazyLock;
type View<'a> = TreeView<'a, StylesAndFormulasView>;
type C = Cursor<types::PdfCharacter>;
type F = Cursor<types::PdfFormula>;
type Page = Cursor<types::Page>;
type S = Cursor<types::PdfStyle>;
type G = Cursor<types::GraphicState>;
const LOG_TARGET: &str = "babeldoc.format.pdf.document_il.midend.styles_and_formulas";
static COMMA_FORMULA_REMOVE_PATTERN: LazyLock<regex::Regex> =
    LazyLock::new(|| regex::Regex::new(r"[0-9\[\],\s]").expect("literal formula cleanup regex"));
static TRANSLATABLE_FORMULA_PATTERN: LazyLock<regex::Regex> = LazyLock::new(|| {
    regex::Regex::new(r"^[0-9, .]+$").expect("literal translatable formula regex")
});
fn failed(e: impl std::fmt::Display) -> StageError {
    StageError::failed(e.to_string())
}
fn box_coords(b: &IlBox, n: &str) -> Result<(f64, f64, f64, f64), StageError> {
    Ok((need(b.x, n)?, need(b.y, n)?, need(b.x2, n)?, need(b.y2, n)?))
}
fn visual_coords(c: &PdfCharacter) -> Result<(f64, f64, f64, f64), StageError> {
    box_coords(
        need(
            need(c.visual_bbox.as_deref(), "visual_bbox")?
                .r#box
                .as_deref(),
            "visual_bbox.box",
        )?,
        "visual",
    )
}
fn character_boxes(
    c: &PdfCharacter,
) -> Result<((f64, f64, f64, f64), (f64, f64, f64, f64)), StageError> {
    Ok((
        visual_coords(c)?,
        box_coords(need(c.r#box.as_deref(), "character.box")?, "box")?,
    ))
}
fn has_formula_layout(c: &PdfCharacter) -> bool {
    c.formula_layout_id.is_some_and(|id| id != 0)
}
fn classify_characters_in_composition(
    config: &StylesAndFormulasConfig,
    line: &babeldoc_il_types::PdfLine,
    formula_font_ids: &BTreeSet<String>,
    first_is_bullet_so_far: bool,
) -> Result<(Vec<(bool, bool)>, bool), StageError> {
    if line.pdf_character.is_empty() {
        return Ok((Vec::new(), first_is_bullet_so_far));
    }
    let mut first_is_bullet = first_is_bullet_so_far;
    let mut in_formula_state = false;
    let mut in_corner_mark_state = false;
    let mut is_formula_tags = Vec::with_capacity(line.pdf_character.len());
    let mut corner_mark_flags = Vec::with_capacity(line.pdf_character.len());
    for (index, character) in line.pdf_character.iter().enumerate() {
        let is_start_of_segment = index == 0
            || is_formula_tags
                .last()
                .is_some_and(|tag| *tag != in_formula_state);
        if !first_is_bullet && is_start_of_segment && is_bullet_point(character) {
            first_is_bullet = true;
        }
        let unicode = character.char_unicode.as_deref().unwrap_or_default();
        // 0.6.4 skips both character helpers when a formula layout is present;
        // the second helper is reached only if the first conjunction is false.
        let mut character_formula = has_formula_layout(character);
        if !character_formula {
            let hit = config
                .character_pattern_hits
                .as_ref()
                .and_then(|m| m.get(unicode).copied());
            let mapped = fh::start_char_with_matcher(
                unicode,
                &config.font_mapper,
                hit,
                config.formula_matcher.as_deref(),
            )?;
            character_formula = mapped && !in_formula_state;
            if !character_formula {
                let middle = fh::start_char_with_matcher(
                    unicode,
                    &config.font_mapper,
                    hit,
                    config.formula_matcher.as_deref(),
                )? || unicode.starts_with(',');
                character_formula = middle && in_formula_state;
            }
        }
        let (visual, actual) = character_boxes(character)?;
        let style = character.pdf_style.as_deref().ok_or_else(|| {
            StageError::Failed("StylesAndFormulas character has no pdf_style".into())
        })?;
        let mut is_formula = character_formula
            || style
                .font_id
                .as_ref()
                .is_some_and(|id| formula_font_ids.contains(id))
            || character.vertical.unwrap_or(false)
            || (character.char_unicode.is_none() && in_formula_state)
            || actual.0 > visual.2
            || actual.2 < visual.0
            || actual.1 > visual.3
            || actual.3 < visual.1;
        let previous = index.checked_sub(1).map(|idx| &line.pdf_character[idx]);
        let next = line.pdf_character.get(index + 1);
        let is_space = character
            .char_unicode
            .as_deref()
            .is_some_and(|value| !value.is_empty() && value.chars().all(char::is_whitespace));
        let previous_is_space = previous
            .and_then(|c| c.char_unicode.as_deref())
            .is_some_and(|value| !value.is_empty() && value.chars().all(char::is_whitespace));
        let current_size = style.font_size.unwrap_or(f64::NAN);
        let previous_size = previous
            .and_then(|c| c.pdf_style.as_deref())
            .and_then(|s| s.font_size)
            .unwrap_or(f64::NAN);
        let next_size = next
            .and_then(|c| c.pdf_style.as_deref())
            .and_then(|s| s.font_size)
            .unwrap_or(f64::NAN);
        let is_corner_mark = (previous.is_some()
            && !is_space
            && !previous_is_space
            && !first_is_bullet
            && current_size < previous_size * 0.79
            && !in_corner_mark_state)
            || (previous.is_some()
                && !is_space
                && !previous_is_space
                && !first_is_bullet
                && current_size < previous_size * 1.1
                && in_corner_mark_state)
            || (previous.is_none()
                && next.is_some()
                && !is_space
                && !previous_is_space
                && !first_is_bullet
                && current_size < next_size * 0.79
                && !in_corner_mark_state);
        is_formula |= is_corner_mark;
        if character.char_unicode.as_deref() == Some(" ") {
            is_formula = in_formula_state;
        }
        if is_formula != in_formula_state {
            in_formula_state = is_formula;
        }
        in_corner_mark_state = is_corner_mark;
        is_formula_tags.push(is_formula);
        corner_mark_flags.push(is_corner_mark);
    }
    Ok((
        is_formula_tags.into_iter().zip(corner_mark_flags).collect(),
        first_is_bullet,
    ))
}
fn expanded_box(formula_box: &IlBox, tolerance: f64) -> Result<IlBox, StageError> {
    let (x, y, x2, y2) = box_coords(formula_box, "formula.box coordinate")?;
    Ok(IlBox {
        x: Some(x - tolerance),
        y: Some(y - tolerance),
        x2: Some(x2 + tolerance),
        y2: Some(y2 + tolerance),
    })
}

fn calculate_element_formula_iou(
    element_box: Option<&IlBox>,
    formula_box: Option<&IlBox>,
    tolerance: f64,
) -> Result<f64, StageError> {
    let (Some(element_box), Some(formula_box)) = (element_box, formula_box) else {
        return Ok(0.0);
    };
    let expanded = expanded_box(formula_box, tolerance)?;
    calculate_iou_for_boxes(element_box, &expanded).map_err(failed)
}

fn is_element_contained_exact(
    element_box: Option<&IlBox>,
    formula_box: Option<&IlBox>,
    threshold: f64,
) -> Result<bool, StageError> {
    let (Some(element_box), Some(formula_box)) = (element_box, formula_box) else {
        return Ok(false);
    };
    Ok(calculate_iou_for_boxes(element_box, formula_box).map_err(failed)? >= threshold)
}

/// Local counterpart of spatial_analyzer.is_element_contained_in_formula.
fn is_element_contained_in_formula(
    element_box: Option<&IlBox>,
    formula_box: Option<&IlBox>,
) -> Result<bool, StageError> {
    Ok(calculate_element_formula_iou(element_box, formula_box, 2.0)? >= 0.95)
}

fn calculate_element_formula_distance(
    element_box: Option<&IlBox>,
    formula_box: Option<&IlBox>,
) -> Result<f64, StageError> {
    let (Some(element_box), Some(formula_box)) = (element_box, formula_box) else {
        return Ok(f64::INFINITY);
    };
    let (ex, ey, ex2, ey2) = box_coords(element_box, "element.box coordinate")?;
    let (fx, fy, fx2, fy2) = box_coords(formula_box, "formula.box coordinate")?;
    let dx = if ex2 < fx {
        fx - ex2
    } else if ex > fx2 {
        ex - fx2
    } else {
        0.0
    };
    let dy = if ey2 < fy {
        fy - ey2
    } else if ey > fy2 {
        ey - fy2
    } else {
        0.0
    };
    Ok((dx * dx + dy * dy).sqrt())
}
fn is_x_axis_contained(first: &IlBox, second: &IlBox) -> Result<bool, StageError> {
    let (x1, _, x12, _) = box_coords(first, "first box coordinate")?;
    let (x2, _, x22, _) = box_coords(second, "second box coordinate")?;
    Ok((x1 >= x2 && x12 <= x22) || (x2 >= x1 && x22 <= x12))
}

fn has_y_intersection(first: &IlBox, second: &IlBox) -> Result<bool, StageError> {
    let (_, y1, _, y12) = box_coords(first, "first box coordinate")?;
    let (_, y2, _, y22) = box_coords(second, "second box coordinate")?;
    Ok(!(y12 < y2 - 1.0 || y22 < y1 - 1.0))
}

fn is_x_axis_adjacent(first: &IlBox, second: &IlBox, tolerance: f64) -> Result<bool, StageError> {
    let (x1, _, x12, _) = box_coords(first, "first box coordinate")?;
    let (x2, _, x22, _) = box_coords(second, "second box coordinate")?;
    let intersection = !(x12 < x2 || x22 < x1);
    Ok(intersection || (x12 - x2).abs() <= tolerance || (x22 - x1).abs() <= tolerance)
}

/// Local counterpart of `calculate_y_iou_for_boxes` retained in the shard closure.
fn calculate_y_iou(first: &IlBox, second: &IlBox) -> Result<f64, StageError> {
    let (_, y1, _, y12) = box_coords(first, "first box coordinate")?;
    let (_, y2, _, y22) = box_coords(second, "second box coordinate")?;
    let intersection = (y12.min(y22) - y1.max(y2)).max(0.0);
    let union = (y12 - y1) + (y22 - y2) - intersection;
    Ok(if union <= 0.0 {
        0.0
    } else {
        intersection / union
    })
}

fn retained(cs: Vec<C>, v: &mut View<'_>) -> Result<Vec<C>, StageError> {
    cs.into_iter()
        .map(|c| c.retain(v).map_err(StageError::from))
        .collect()
}
fn formula_metrics(cs: &[C], v: &View<'_>) -> Result<(IlBox, f64, f64, f64), StageError> {
    let values = cs
        .iter()
        .map(|c| character(c, v))
        .collect::<Result<Vec<_>, _>>()?;
    let mut formula = babeldoc_il_types::PdfFormula {
        pdf_character: values,
        ..Default::default()
    };
    fh::update_formula_data(&mut formula).map_err(failed)?;
    Ok((
        *formula.r#box.unwrap(),
        formula.x_offset.unwrap_or_default(),
        formula.y_offset.unwrap_or_default(),
        formula.x_advance.unwrap_or_default(),
    ))
}
fn new_composition(
    cs: &[C],
    formula: bool,
    line_id: Option<i64>,
    corner: Option<bool>,
    v: &mut View<'_>,
) -> Result<ir::FieldValue<types::PdfParagraphComposition>, StageError> {
    let ch = cs
        .iter()
        .map(|c| c.share(v))
        .collect::<Result<Vec<_>, _>>()?;
    if formula {
        let (b, x, y, a) = formula_metrics(cs, v)?;
        Ok(ir::PdfParagraphCompositionInit {
            pdf_formula: Some(
                ir::PdfFormulaInit {
                    pdf_character: ch,
                    r#box: Some(box_value(b)),
                    x_offset: Some(x),
                    y_offset: Some(y),
                    x_advance: Some(a),
                    line_id,
                    is_corner_mark: corner,
                    ..Default::default()
                }
                .into_value(),
            ),
            ..Default::default()
        }
        .into_value())
    } else {
        if cs.is_empty() {
            return Err(StageError::failed("min() iterable argument is empty"));
        }
        let b = union(
            &cs.iter()
                .map(|c| visual(c, v))
                .collect::<Result<Vec<_>, _>>()?,
        )?;
        Ok(ir::PdfParagraphCompositionInit {
            pdf_line: Some(
                ir::PdfLineInit {
                    r#box: Some(box_value(b)),
                    pdf_character: ch,
                    ..Default::default()
                }
                .into_value(),
            ),
            ..Default::default()
        }
        .into_value())
    }
}
fn update_formula(f: &F, v: &mut View<'_>) -> Result<(), StageError> {
    let cs = f.pdf_character_iter(v)?.collect::<Result<Vec<_>, _>>()?;
    let (b, _, _, _) = formula_metrics(&cs, v)?;
    f.set_box(v, Some(box_value(b)))?;
    if f.x_offset(v)?.unwrap_or_default() == 0. {
        f.set_x_offset(v, Some(0.))?;
    }
    if f.y_offset(v)?.unwrap_or_default() == 0. {
        f.set_y_offset(v, Some(0.))?;
    }
    if f.x_advance(v)?.unwrap_or_default() == 0. {
        f.set_x_advance(v, Some(0.))?;
    }
    Ok(())
}
fn formula_box(f: &F, v: &View<'_>) -> Result<IlBox, StageError> {
    box_read(&need(f.r#box(v)?, "formula.box")?, v)
}
fn formula_text(cs: &[C], v: &View<'_>) -> Result<String, StageError> {
    let mut out = String::new();
    for (i, c) in cs.iter().enumerate() {
        let s = c.char_unicode(v)?.as_deref().ok_or_else(|| {
            StageError::failed(format!(
                "sequence item {i}: expected str instance, NoneType found"
            ))
        })?;
        out.push_str(&s);
    }
    Ok(out)
}
fn formula_layout(cs: &[C], v: &View<'_>) -> Result<bool, StageError> {
    for c in cs {
        if !c.formula_layout_id(v)?.is_some_and(|id| id != 0) {
            return Ok(false);
        }
    }
    Ok(true)
}
fn font_ids(
    page: &Page,
    v: &View<'_>,
    config: &StylesAndFormulasConfig,
) -> Result<(BTreeSet<String>, BTreeMap<i64, BTreeSet<String>>), StageError> {
    let mut page_ids = BTreeSet::new();
    for f in page.pdf_font_iter(v)? {
        let f = f?;
        if fh::font_with_matcher(
            f.name(v)?.as_deref().unwrap_or(""),
            config.font_pattern_hits.as_ref(),
            config.formula_matcher.as_deref(),
        )? {
            if let Some(id) = f.font_id(v)? {
                page_ids.insert(id.clone());
            }
        }
    }
    let mut xs = BTreeMap::new();
    for xo in page.pdf_xobject_iter(v)? {
        let xo = xo?;
        let mut ids = page_ids.clone();
        for f in xo.pdf_font_iter(v)? {
            let f = f?;
            let id = f.font_id(v)?.clone().unwrap_or_default();
            if fh::font_with_matcher(
                f.name(v)?.as_deref().unwrap_or(""),
                config.font_pattern_hits.as_ref(),
                config.formula_matcher.as_deref(),
            )? {
                ids.insert(id.clone());
            } else {
                ids.remove(&id);
            }
        }
        xs.insert(xo.xobj_id(v)?.unwrap_or_default(), ids);
    }
    Ok((page_ids, xs))
}
fn classify(
    page: &Page,
    v: &mut View<'_>,
    config: &StylesAndFormulasConfig,
) -> Result<(), StageError> {
    if page.pdf_paragraph_len(v)? == 0 {
        return Ok(());
    }
    let (page_ids, xs) = font_ids(page, v, config)?;
    for child_index in 0..page.pdf_paragraph_len(v)? {
        let p = page.pdf_paragraph_at(v, child_index)?;
        let mut out = vec![];
        let mut first_bullet = false;
        let xobj = p.xobj_id(v)?;
        let ids = xobj
            .filter(|id| *id != 0)
            .and_then(|id| xs.get(&id))
            .unwrap_or(&page_ids);
        for index in 0..p.pdf_paragraph_composition_len(v)? {
            let c = p.pdf_paragraph_composition_at(v, index)?;
            let Some(l) = c.pdf_line(v)? else {
                out.push(c.share(v)?);
                continue;
            };
            let raw = l.pdf_character_iter(v)?.collect::<Result<Vec<_>, _>>()?;
            let cs = retained(raw, v)?;
            let scalar_line = babeldoc_il_types::PdfLine {
                pdf_character: cs
                    .iter()
                    .map(|c| character(c, v))
                    .collect::<Result<Vec<_>, _>>()?,
                ..Default::default()
            };
            let (tags, bullet) =
                classify_characters_in_composition(config, &scalar_line, ids, first_bullet)?;
            first_bullet = bullet;
            if tags.is_empty() {
                out.push(c.share(v)?);
                continue;
            }
            let mut current = vec![];
            let mut corner = false;
            let mut tag = tags[0].0;
            for (ch, (t, co)) in cs.into_iter().zip(tags) {
                if t != tag {
                    out.push(new_composition(
                        &current,
                        tag,
                        Some(index as i64),
                        Some(corner),
                        v,
                    )?);
                    current.clear();
                    corner = false;
                    tag = t;
                }
                current.push(ch);
                corner |= co;
            }
            if !current.is_empty() {
                out.push(new_composition(
                    &current,
                    tag,
                    Some(index as i64),
                    Some(corner),
                    v,
                )?);
            }
        }
        p.set_pdf_paragraph_composition(v, out)?;
    }
    Ok(())
}
fn comma_formulas(page: &Page, v: &mut View<'_>) -> Result<(), StageError> {
    for child_index in 0..page.pdf_paragraph_len(v)? {
        let p = page.pdf_paragraph_at(v, child_index)?;
        let mut out = vec![];
        for child_index in 0..p.pdf_paragraph_composition_len(v)? {
            let c = p.pdf_paragraph_composition_at(v, child_index)?;
            let Some(f) = c.pdf_formula(v)? else {
                out.push(c.share(v)?);
                continue;
            };
            let raw = f.pdf_character_iter(v)?.collect::<Result<Vec<_>, _>>()?;
            let cs = retained(raw, v)?;
            let text = formula_text(&cs, v)?;
            if formula_layout(&cs, v)?
                || !text.contains(',')
                || COMMA_FORMULA_REMOVE_PATTERN
                    .replace_all(&text, "")
                    .is_empty()
            {
                out.push(c.share(v)?);
                continue;
            }
            let mut current = vec![];
            let mut level = 0usize;
            let line_id = f.line_id(v)?;
            for ch in cs {
                let text = ch.char_unicode(v)?;
                if text.as_deref().is_some_and(|s| LEFT_BRACKET.contains(&s)) {
                    level += 1;
                    current.push(ch);
                } else if text.as_deref().is_some_and(|s| RIGHT_BRACKET.contains(&s)) {
                    level = level.saturating_sub(1);
                    current.push(ch);
                } else if text.as_deref() == Some(",") && level == 0 {
                    if !current.is_empty() {
                        out.push(new_composition(&current, true, line_id, None, v)?);
                        out.push(new_composition(&[ch], false, None, None, v)?);
                        current.clear();
                    }
                } else {
                    current.push(ch);
                }
            }
            if !current.is_empty() {
                out.push(new_composition(&current, true, line_id, None, v)?);
            }
        }
        p.set_pdf_paragraph_composition(v, out)?;
    }
    Ok(())
}
fn layout_ids(f: &F, v: &View<'_>) -> Result<BTreeSet<i64>, StageError> {
    let mut out = BTreeSet::new();
    for c in f.pdf_character_iter(v)? {
        let c = c?;
        if c.char_unicode(v)?.as_deref() != Some(" ") {
            if let Some(id) = c.formula_layout_id(v)? {
                out.insert(id);
            }
        }
    }
    Ok(out)
}
fn merge_formulas(page: &Page, v: &mut View<'_>) -> Result<(), StageError> {
    for child_index in 0..page.pdf_paragraph_len(v)? {
        let p = page.pdf_paragraph_at(v, child_index)?;
        loop {
            let cs = p
                .pdf_paragraph_composition_iter(v)?
                .collect::<Result<Vec<_>, _>>()?;
            let mut layout_cache = std::iter::repeat_with(|| None)
                .take(cs.len())
                .collect::<Vec<Option<BTreeSet<i64>>>>();
            let mut merged = None;
            'outer: for i in 0..cs.len() {
                let Some(a) = cs[i].pdf_formula(v)? else {
                    continue;
                };
                for j in i + 1..cs.len() {
                    let Some(b) = cs[j].pdf_formula(v)? else {
                        continue;
                    };
                    let aa = formula_box(&a, v)?;
                    let bb = formula_box(&b, v)?;
                    if layout_cache[i].is_none() {
                        layout_cache[i] = Some(layout_ids(&a, v)?);
                    }
                    if layout_cache[j].is_none() {
                        layout_cache[j] = Some(layout_ids(&b, v)?);
                    }
                    let ai = layout_cache[i].as_ref().unwrap();
                    let bi = layout_cache[j].as_ref().unwrap();
                    if a.line_id(v)? == b.line_id(v)?
                        && ((j == i + 1
                            && ((is_x_axis_contained(&aa, &bb)? && has_y_intersection(&aa, &bb)?)
                                || (is_x_axis_adjacent(&aa, &bb, 2.)?
                                    && calculate_y_iou(&aa, &bb)? > 0.5)))
                            || (ai.len() == 1 && bi.len() == 1 && ai == bi)
                            || lh::calculate_iou_for_boxes(&aa, &bb)? > 0.8
                            || lh::calculate_iou_for_boxes(&bb, &aa)? > 0.8)
                    {
                        let a_raw = a.pdf_character_iter(v)?.collect::<Result<Vec<_>, _>>()?;
                        let mut chars = retained(a_raw, v)?;
                        let b_raw = b.pdf_character_iter(v)?.collect::<Result<Vec<_>, _>>()?;
                        chars.extend(retained(b_raw, v)?);
                        let replacement = new_composition(&chars, true, a.line_id(v)?, None, v)?;
                        merged = Some((i, j, replacement));
                        break 'outer;
                    }
                }
            }
            let Some((i, j, replacement)) = merged else {
                break;
            };
            let mut out = vec![];
            let mut replacement = Some(replacement);
            for (k, c) in cs.iter().enumerate() {
                if k == i {
                    out.push(replacement.take().unwrap());
                } else if k != j {
                    out.push(c.share(v)?);
                }
            }
            p.set_pdf_paragraph_composition(v, out)?;
        }
    }
    Ok(())
}
fn offsets(page: &Page, v: &mut View<'_>) -> Result<(), StageError> {
    for child_index in 0..page.pdf_paragraph_len(v)? {
        let p = page.pdf_paragraph_at(v, child_index)?;
        if p.debug_id(v)?.is_none() {
            continue;
        }
        let composition_len = p.pdf_paragraph_composition_len(v)?;
        for i in 0..composition_len {
            let c = p.pdf_paragraph_composition_at(v, i)?;
            let Some(f) = c.pdf_formula(v)? else { continue };
            let fb = formula_box(&f, v)?;
            let mut left = None;
            let mut right = None;
            for (side, previous) in [
                (true, i.checked_sub(1)),
                (
                    false,
                    if i + 1 < composition_len {
                        Some(i + 1)
                    } else {
                        None
                    },
                ),
            ] {
                if let Some(j) = previous {
                    if let Some(l) = p.pdf_paragraph_composition_at(v, j)?.pdf_line(v)? {
                        let character_len = l.pdf_character_len(v)?;
                        for step in 0..character_len {
                            let character_index =
                                if side { character_len - step - 1 } else { step };
                            let ch = l.pdf_character_at(v, character_index)?;
                            if ch.pdf_character_id(v)?.unwrap_or_default() == 0 {
                                continue;
                            }
                            let b = box_read(&need(ch.r#box(v)?, "character.box")?, v)?;
                            let iou = lh::calculate_y_true_iou_for_boxes(&fb, &b)?;
                            if iou > 0.6 {
                                let value = Some((box_coords(&b, "box")?, iou));
                                if side {
                                    left = value
                                } else {
                                    right = value
                                }
                                break;
                            }
                        }
                    }
                }
            }
            if let (Some((_, li)), Some((_, ri))) = (left, right) {
                if li < ri {
                    left = None;
                } else if ri < li {
                    right = None;
                }
            }
            let (x, y, _, _) = box_coords(&fb, "formula.box")?;
            let xo = left.map_or(0., |((_, _, x2, _), _)| x - x2);
            let xo = if xo.abs() < 0.1 || xo > 10. || xo < -5. {
                0.
            } else {
                xo
            };
            let yo = left.or(right).map_or(0., |((_, y1, _, _), _)| y - y1);
            let yo = if yo.abs() < 0.1 { 0. } else { yo };
            f.set_x_offset(v, Some(xo))?;
            f.set_y_offset(v, Some(yo))?;
        }
    }
    Ok(())
}
fn translatable(page: &Page, v: &mut View<'_>) -> Result<(), StageError> {
    for child_index in 0..page.pdf_paragraph_len(v)? {
        let p = page.pdf_paragraph_at(v, child_index)?;
        let mut out = vec![];
        for child_index in 0..p.pdf_paragraph_composition_len(v)? {
            let c = p.pdf_paragraph_composition_at(v, child_index)?;
            let Some(f) = c.pdf_formula(v)? else {
                out.push(c.share(v)?);
                continue;
            };
            let raw = f.pdf_character_iter(v)?.collect::<Result<Vec<_>, _>>()?;
            let cs = retained(raw, v)?;
            if !f.is_corner_mark(v)?.unwrap_or(false)
                && !formula_layout(&cs, v)?
                && f.y_offset(v)?.unwrap_or_default() <= 0.1
                && TRANSLATABLE_FORMULA_PATTERN.is_match(&formula_text(&cs, v)?)
            {
                out.push(new_composition(&cs, false, None, None, v)?);
            } else {
                out.push(c.share(v)?);
            }
        }
        p.set_pdf_paragraph_composition(v, out)?;
    }
    Ok(())
}
fn update_all(page: &Page, v: &mut View<'_>) -> Result<(), StageError> {
    for paragraph_index in 0..page.pdf_paragraph_len(v)? {
        let p = page.pdf_paragraph_at(v, paragraph_index)?;
        for composition_index in 0..p.pdf_paragraph_composition_len(v)? {
            let c = p.pdf_paragraph_composition_at(v, composition_index)?;
            if let Some(f) = c.pdf_formula(v)? {
                update_formula(&f, v)?;
            }
        }
    }
    Ok(())
}
fn assign_elements(page: &Page, v: &mut View<'_>) -> Result<(), StageError> {
    let mut formulas = vec![];
    for child_index in 0..page.pdf_paragraph_len(v)? {
        let p = page.pdf_paragraph_at(v, child_index)?;
        for child_index in 0..p.pdf_paragraph_composition_len(v)? {
            let c = p.pdf_paragraph_composition_at(v, child_index)?;
            if let Some(f) = c.pdf_formula(v)? {
                formulas.push((f, p.xobj_id(v)?));
            }
        }
    }
    let curve_len = page.pdf_curve_len(v)?;
    let form_len = page.pdf_form_len(v)?;
    let mut remove_curve = BTreeSet::new();
    let mut remove_form = BTreeSet::new();
    let mut assignments: BTreeMap<usize, (Vec<usize>, Vec<usize>)> = BTreeMap::new();
    let mut assignment_order = vec![];
    for (is_curve, n) in [(true, curve_len), (false, form_len)] {
        for i in 0..n {
            let (b, xobj) = if is_curve {
                let curve = page.pdf_curve_at(v, i)?;
                (curve.r#box(v)?, curve.xobj_id(v)?)
            } else {
                let form = page.pdf_form_at(v, i)?;
                (form.r#box(v)?, form.xobj_id(v)?)
            };
            let Some(b) = b else { continue };
            let b = box_read(&b, v)?;
            let mut candidates = vec![];
            for (j, (f, owner_xobj)) in formulas.iter().enumerate() {
                if owner_xobj.is_some() && xobj != *owner_xobj {
                    continue;
                }
                let Some(fb) = f.r#box(v)? else { continue };
                let fb = box_read(&fb, v)?;
                if is_element_contained_exact(Some(&b), Some(&fb), 0.95)? {
                    candidates.push((j, 0, lh::calculate_iou_for_boxes(&b, &fb)?));
                } else if is_element_contained_in_formula(Some(&b), Some(&fb))? {
                    let distance = calculate_element_formula_distance(Some(&b), Some(&fb))?;
                    candidates.push((j, 1, 0.5 + 0.4 * (1. - distance / 100.).max(0.)));
                }
            }
            candidates.sort_by(|a, b| a.1.cmp(&b.1).then_with(|| b.2.total_cmp(&a.2)));
            if let Some((j, _, _)) = candidates.first() {
                let entry = assignments.entry(*j).or_insert_with(|| {
                    assignment_order.push(*j);
                    (vec![], vec![])
                });
                if is_curve {
                    entry.0.push(i)
                } else {
                    entry.1.push(i)
                }
            }
        }
    }
    // Python applies its dict in first-assignment order. Distinct indices may
    // hold the same formula object, so sorting by index would reorder its curves.
    for j in assignment_order {
        let (cis, fis) = assignments
            .remove(&j)
            .ok_or_else(|| StageError::failed("formula assignment"))?;
        let f = &formulas[j].0;
        for i in cis {
            let value = page.pdf_curve_at(v, i)?.share(v)?;
            f.push_pdf_curve(v, value)?;
            remove_curve.insert(i);
        }
        for i in fis {
            let value = page.pdf_form_at(v, i)?.share(v)?;
            f.push_pdf_form(v, value)?;
            remove_form.insert(i);
        }
    }
    let mut remaining = Vec::with_capacity(curve_len - remove_curve.len());
    for i in 0..curve_len {
        if !remove_curve.contains(&i) {
            remaining.push(page.pdf_curve_at(v, i)?.share(v)?);
        }
    }
    page.set_pdf_curve(v, remaining)?;
    let mut remaining = Vec::with_capacity(form_len - remove_form.len());
    for i in 0..form_len {
        if !remove_form.contains(&i) {
            remaining.push(page.pdf_form_at(v, i)?.share(v)?);
        }
    }
    page.set_pdf_form(v, remaining)?;
    Ok(())
}
enum State {
    Selected(G),
    New(Option<ir::Instructions>),
}
enum Base {
    Selected(S),
    New {
        font_id: Option<String>,
        font_size: Option<f64>,
        state: Option<State>,
    },
}
impl Base {
    fn font_id(&self, v: &View<'_>) -> Result<Option<String>, StageError> {
        match self {
            Self::Selected(s) => Ok(s.font_id(v)?.clone()),
            Self::New { font_id, .. } => Ok(font_id.clone()),
        }
    }
    fn size(&self, v: &View<'_>) -> Result<Option<f64>, StageError> {
        match self {
            Self::Selected(s) => Ok(s.font_size(v)?),
            Self::New { font_size, .. } => Ok(*font_size),
        }
    }
    fn state(&self, v: &mut View<'_>) -> Result<Option<State>, StageError> {
        match self {
            Self::Selected(s) => Ok(s
                .graphic_state(v)?
                .map(|s| s.retain(v))
                .transpose()?
                .map(State::Selected)),
            Self::New { state, .. } => Ok(match state {
                Some(State::Selected(c)) => Some(State::Selected(c.clone())),
                Some(State::New(v)) => Some(State::New(v.clone())),
                None => None,
            }),
        }
    }
    fn value(self, v: &mut View<'_>) -> Result<ir::FieldValue<types::PdfStyle>, StageError> {
        match self {
            Self::Selected(c) => Ok(c.share(v)?),
            Self::New {
                font_id,
                font_size,
                state,
            } => {
                let state = match state {
                    Some(State::Selected(c)) => Some(c.share(v)?),
                    Some(State::New(instruction)) => Some(
                        ir::GraphicStateInit {
                            passthrough_per_char_instruction: instruction,
                            ..Default::default()
                        }
                        .into_value(),
                    ),
                    None => None,
                };
                Ok(ir::PdfStyleInit {
                    font_id,
                    font_size,
                    graphic_state: state,
                }
                .into_value())
            }
        }
    }
}
fn state_text(s: &State, v: &View<'_>) -> Result<Option<String>, StageError> {
    match s {
        State::Selected(c) => Ok(c.instruction_text(v)?),
        State::New(value) => Ok(v.instruction_text(value)?),
    }
}

fn state_instruction(s: &State, v: &View<'_>) -> Result<Option<ir::Instructions>, StageError> {
    match s {
        State::Selected(c) => Ok(c.passthrough_per_char_instruction(v)?.clone()),
        State::New(value) => Ok(value.clone()),
    }
}
fn merge_base(first: Base, second: S, v: &mut View<'_>) -> Result<Base, StageError> {
    let a = first.size(v)?;
    if a.is_none() {
        return Ok(Base::Selected(second));
    }
    let b = second.font_size(v)?;
    if b.is_none() {
        return Ok(first);
    }
    let fid = first.font_id(v)?;
    let sfid = second.font_id(v)?.clone();
    let state1 = first.state(v)?;
    let state2 = second
        .graphic_state(v)?
        .map(|s| s.retain(v))
        .transpose()?
        .map(State::Selected);
    let state = match (state1, state2) {
        (None, s) | (s, None) => s,
        (Some(a), Some(b)) => Some(State::New(if state_text(&a, v)? == state_text(&b, v)? {
            state_instruction(&a, v)?
        } else {
            None
        })),
    };
    Ok(Base::New {
        font_id: if fid == sfid { fid } else { None },
        font_size: if (a.unwrap() - b.unwrap()).abs() < 0.02 {
            a
        } else {
            None
        },
        state,
    })
}
fn mode<T: Clone + PartialEq>(values: &[T]) -> Option<T> {
    let mut best = None;
    let mut count = 0;
    for x in values {
        let n = values.iter().filter(|y| *y == x).count();
        if n > count {
            count = n;
            best = Some(x.clone());
        }
    }
    best
}
fn mode_option_string<T: std::borrow::Borrow<Option<String>>>(
    values: &[T],
) -> Option<Option<String>> {
    let mut counts = HashMap::<Option<&str>, (usize, usize)>::with_capacity(values.len());
    for (index, value) in values.iter().enumerate() {
        let entry = counts
            .entry(value.borrow().as_deref())
            .or_insert((0, index));
        entry.0 += 1;
    }
    counts
        .into_iter()
        .max_by_key(|(_, (count, first_index))| (*count, std::cmp::Reverse(*first_index)))
        .map(|(value, _)| value.map(str::to_owned))
}
fn same_style(a: &S, b: &S, v: &View<'_>) -> Result<bool, StageError> {
    if a.font_id(v)? != b.font_id(v)?
        || (a.font_size(v)?.unwrap_or_default() - b.font_size(v)?.unwrap_or_default()).abs() >= 0.02
    {
        return Ok(false);
    }
    match (a.graphic_state(v)?, b.graphic_state(v)?) {
        (Some(a), Some(b)) => Ok(a.instruction_text(v)? == b.instruction_text(v)?),
        (None, None) => Ok(true),
        _ => Ok(false),
    }
}
fn style_group(
    cs: &[C],
    style: &S,
    v: &mut View<'_>,
) -> Result<ir::FieldValue<types::PdfParagraphComposition>, StageError> {
    let b = union(
        &cs.iter()
            .map(|c| visual(c, v))
            .collect::<Result<Vec<_>, _>>()?,
    )?;
    let values = cs
        .iter()
        .map(|c| c.share(v))
        .collect::<Result<Vec<_>, _>>()?;
    let style = style.share(v)?;
    Ok(ir::PdfParagraphCompositionInit {
        pdf_same_style_characters: Some(
            ir::PdfSameStyleCharactersInit {
                r#box: Some(box_value(b)),
                pdf_style: Some(style),
                pdf_character: values,
            }
            .into_value(),
        ),
        ..Default::default()
    }
    .into_value())
}
fn styles(page: &Page, v: &mut View<'_>) -> Result<(), StageError> {
    for child_index in 0..page.pdf_paragraph_len(v)? {
        let p = page.pdf_paragraph_at(v, child_index)?;
        if p.pdf_paragraph_composition_len(v)? == 0 {
            continue;
        }
        let mut source = vec![];
        for composition_index in 0..p.pdf_paragraph_composition_len(v)? {
            let c = p.pdf_paragraph_composition_at(v, composition_index)?;
            if let Some(l) = c.pdf_line(v)? {
                for child_index in 0..l.pdf_character_len(v)? {
                    let ch = l.pdf_character_at(v, child_index)?;
                    source.push(need(ch.pdf_style(v)?, "character.style")?.retain(v)?);
                }
            }
        }
        let base = if let Some(first) = source.first() {
            let mut base = Base::Selected(first.clone());
            for style in source.iter().skip(1) {
                base = merge_base(base, style.clone(), v)?;
            }
            Some(base.value(v)?)
        } else {
            None
        };
        p.set_pdf_style(v, base)?;
        if let Some(base) = p.pdf_style(v)? {
            if base.font_id(v)?.is_none() {
                let values = source
                    .iter()
                    .map(|s| s.font_id(v))
                    .collect::<Result<Vec<_>, _>>()?;
                let selected = mode_option_string(&values).flatten();
                base.set_font_id(v, selected)?;
            }
            if base.font_size(v)?.is_none() {
                let values = source
                    .iter()
                    .map(|s| s.font_size(v))
                    .collect::<Result<Vec<_>, _>>()?;
                base.set_font_size(v, mode(&values).flatten())?;
            }
        }
        let mut out = vec![];
        let mut current = vec![];
        let mut current_style = None::<S>;
        for composition_index in 0..p.pdf_paragraph_composition_len(v)? {
            let c = p.pdf_paragraph_composition_at(v, composition_index)?;
            if c.pdf_formula(v)?.is_some() {
                if !current.is_empty() {
                    out.push(style_group(&current, current_style.as_ref().unwrap(), v)?);
                    current.clear();
                }
                out.push(c.share(v)?);
                continue;
            }
            let Some(l) = c.pdf_line(v)? else {
                out.push(c.share(v)?);
                continue;
            };
            for child_index in 0..l.pdf_character_len(v)? {
                let ch = l.pdf_character_at(v, child_index)?;
                let ch = ch.retain(v)?;
                let style = need(ch.pdf_style(v)?, "character.style")?.retain(v)?;
                if match current_style.as_ref() {
                    Some(s) => !same_style(&style, s, v)?,
                    None => false,
                } {
                    if !current.is_empty() {
                        out.push(style_group(&current, current_style.as_ref().unwrap(), v)?);
                        current.clear();
                    }
                    current_style = Some(style);
                } else if current_style.is_none() {
                    current_style = Some(style);
                }
                current.push(ch);
            }
        }
        if !current.is_empty() {
            out.push(style_group(&current, current_style.as_ref().unwrap(), v)?);
        }
        p.set_pdf_paragraph_composition(v, out)?;
    }
    Ok(())
}
fn remove_lines(
    page: &Page,
    v: &mut View<'_>,
    config: &StylesAndFormulasConfig,
) -> Result<(), StageError> {
    let layout = layouts(page, v)?;
    let (index, map) = LayoutIndex::build(&layout)?;
    let protection = config.figure_table_protection_threshold.unwrap_or(0.9);
    let threshold = config.non_formula_line_iou_threshold.unwrap_or(0.9);
    let mut boxes = vec![];
    for child_index in 0..page.pdf_paragraph_len(v)? {
        let p = page.pdf_paragraph_at(v, child_index)?;
        let mut comps = vec![];
        for child_index in 0..p.pdf_paragraph_composition_len(v)? {
            let c = p.pdf_paragraph_composition_at(v, child_index)?;
            let b = if let Some(l) = c.pdf_line(v)? {
                l.r#box(v)?
            } else if let Some(f) = c.pdf_formula(v)? {
                f.r#box(v)?
            } else if let Some(s) = c.pdf_same_style_characters(v)? {
                s.r#box(v)?
            } else {
                None
            };
            if let Some(b) = b {
                comps.push(box_read(&b, v)?);
            }
        }
        if !comps.is_empty() {
            boxes.push(union(&comps)?);
        }
    }
    let mut out = vec![];
    let mut removed = 0;
    for child_index in 0..page.pdf_curve_len(v)? {
        let curve = page.pdf_curve_at(v, child_index)?;
        let b = curve.r#box(v)?.map(|b| box_read(&b, v)).transpose()?;
        let sample = babeldoc_il_types::PdfCurve {
            r#box: b.clone().map(Box::new),
            ..Default::default()
        };
        let protected =
            lh::is_curve_in_figure_table_layout_with_threshold(&sample, &index, &map, protection)?;
        let mut overlaps = false;
        if let Some(b) = &b {
            for pb in &boxes {
                if lh::calculate_iou_for_boxes(b, pb)? > threshold {
                    overlaps = true;
                    break;
                }
            }
        }
        if protected || !overlaps {
            out.push(curve.share(v)?);
        } else {
            removed += 1;
        }
    }
    page.set_pdf_curve(v, out)?;
    if removed > 0 {
        debug!(target:LOG_TARGET,"Removed {removed} non-formula lines from paragraphs");
    }
    Ok(())
}
pub fn process(
    config: &StylesAndFormulasConfig,
    document: &mut ir::Document,
    context: &mut StageContext<'_>,
) -> Result<(), StageError> {
    let mut v = document.view::<StylesAndFormulasView>();
    let total = v.document().page_len(&v)? as u64;
    for page_index in 0..total as usize {
        let page = v.document().page_at(&v, page_index)?;
        context.check_cancelled()?;
        classify(&page, &mut v, config)?;
        comma_formulas(&page, &mut v)?;
        merge_formulas(&page, &mut v)?;
        if !config.skip_formula_offset_calculation {
            offsets(&page, &mut v)?;
        }
        translatable(&page, &mut v)?;
        update_all(&page, &mut v)?;
        if !config.ocr_workaround {
            assign_elements(&page, &mut v)?;
        }
        if config.remove_non_formula_lines {
            remove_lines(&page, &mut v, config)?;
        }
        if !config.skip_formula_offset_calculation {
            offsets(&page, &mut v)?;
        }
        update_all(&page, &mut v)?;
        styles(&page, &mut v)?;
        context.report_progress(1, Some(total))?;
    }
    drop(v);
    document.compact();
    Ok(())
}

#[cfg(test)]
mod styles_and_formulas_tests;
