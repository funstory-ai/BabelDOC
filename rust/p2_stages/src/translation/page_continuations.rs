use std::{collections::HashSet, ops::Range};

use babeldoc_ir_core::{AccessError, Cursor, Document, ReadOnlyView, TreeView, types};

use super::{
    llm_only::{ParagraphId, is_body},
    ordinary_translation::{InputPlaceholder, TranslateInput},
};
use crate::shared::layout_helper::BULLET_POINT_PATTERN;

type View<'a> = TreeView<'a, ReadOnlyView>;

struct Fragment {
    id: ParagraphId,
    text: String,
    indent: bool,
    font_size: f64,
    left_margin: f64,
    column: Option<Column>,
    top_fraction: f64,
    bottom_fraction: f64,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Column {
    Left,
    Right,
}

fn column(x: f64, x2: f64, center: f64) -> Option<Column> {
    if x2 <= center {
        Some(Column::Left)
    } else if x >= center {
        Some(Column::Right)
    } else {
        None
    }
}

struct PageEdges {
    number: Option<i64>,
    first: Option<Fragment>,
    last: Option<Fragment>,
    two_columns: bool,
}

fn coordinates(
    bounds: &Cursor<types::Box>,
    view: &View<'_>,
) -> Result<Option<[f64; 4]>, AccessError> {
    Ok(
        match [
            bounds.x(view)?,
            bounds.y(view)?,
            bounds.x2(view)?,
            bounds.y2(view)?,
        ] {
            [Some(x), Some(y), Some(x2), Some(y2)] => Some([x, y, x2, y2]),
            _ => None,
        },
    )
}

fn fragment(
    id: ParagraphId,
    paragraph: Cursor<types::PdfParagraph>,
    [page_x, page_y, page_right, page_top]: [f64; 4],
    view: &View<'_>,
) -> Result<Option<Fragment>, AccessError> {
    if !is_body(paragraph.layout_label(view)?.as_deref())
        || paragraph.vertical(view)?.unwrap_or(false)
        || paragraph.debug_id(view)?.is_none()
    {
        return Ok(None);
    }
    let (Some(text), Some(bounds), Some(style)) = (
        paragraph.unicode(view)?,
        paragraph.r#box(view)?,
        paragraph.pdf_style(view)?,
    ) else {
        return Ok(None);
    };
    let Some([x, y, x2, top]) = coordinates(&bounds, view)? else {
        return Ok(None);
    };
    let height = page_top - page_y;
    let text = text.trim();
    if height <= 0.0 || !text.chars().any(char::is_alphabetic) {
        return Ok(None);
    }
    let indent = paragraph.first_line_indent(view)?.unwrap_or(false);
    Ok(style
        .font_size(view)?
        .filter(|size| *size > 0.0)
        .map(|font_size| Fragment {
            id,
            text: text.to_owned(),
            indent,
            font_size,
            left_margin: x - page_x,
            column: column(x, x2, (page_x + page_right) * 0.5),
            top_fraction: (page_top - top) / height,
            bottom_fraction: (y - page_y) / height,
        }))
}

fn edges(page_index: usize, view: &View<'_>) -> Result<PageEdges, AccessError> {
    let page = view.document().page_at(view, page_index)?;
    let mut edges = PageEdges {
        number: page.page_number(view)?,
        first: None,
        last: None,
        two_columns: false,
    };
    let Some(cropbox) = page.cropbox(view)? else {
        return Ok(edges);
    };
    let Some(bounds) = cropbox.r#box(view)? else {
        return Ok(edges);
    };
    let Some(page_box) = coordinates(&bounds, view)? else {
        return Ok(edges);
    };
    let center = (page_box[0] + page_box[2]) * 0.5;
    let mut left_column = false;
    let mut right_column = false;
    let mut first = None;
    let mut last = None;
    for (index, paragraph) in page.pdf_paragraph_iter(view)?.enumerate() {
        let paragraph = paragraph?;
        let label = paragraph.layout_label(view)?;
        if matches!(
            label.as_deref(),
            Some(
                "header"
                    | "footer"
                    | "page_header"
                    | "page_footer"
                    | "page_number"
                    | "footnote"
                    | "page_header_hybrid"
                    | "page_footer_hybrid"
                    | "page_number_hybrid"
                    | "footnote_hybrid"
                    | "abandon"
            )
        ) || paragraph
            .unicode(view)?
            .as_deref()
            .is_none_or(|text| text.trim().is_empty())
        {
            continue;
        }
        if is_body(label.as_deref())
            && let Some(bounds) = paragraph.r#box(view)?
            && let (Some(x), Some(x2)) = (bounds.x(view)?, bounds.x2(view)?)
        {
            match column(x, x2, center) {
                Some(Column::Left) => left_column = true,
                Some(Column::Right) => right_column = true,
                None => {}
            }
        }
        let id = ParagraphId {
            page: page_index,
            index,
        };
        first.get_or_insert_with(|| (id, paragraph.clone()));
        last = Some((id, paragraph));
    }
    if let Some((id, paragraph)) = first {
        edges.first = fragment(id, paragraph, page_box, view)?;
    }
    if let Some((id, paragraph)) = last {
        edges.last = fragment(id, paragraph, page_box, view)?;
    }
    edges.two_columns = left_column && right_column;
    Ok(edges)
}

fn is_continuation(left: &Fragment, right: &Fragment, two_columns: bool) -> bool {
    let end = left.text.trim_end_matches([
        '\'', '"', ')', ']', '\u{2019}', '\u{201d}', '\u{300d}', '\u{300f}', '\u{ff09}',
    ]);
    let start = right
        .text
        .trim_start_matches(['\'', '"', '\u{2018}', '\u{201c}']);
    // PDF text has no paragraph delimiter at a page break. Require both the
    // geometry of a continuation and an unfinished sentence before joining.
    left.bottom_fraction <= 0.25
        && right.top_fraction <= 0.25
        && !right.indent
        && (left.font_size - right.font_size).abs() <= left.font_size * 0.10
        && ((left.left_margin - right.left_margin).abs() <= left.font_size * 1.5
            || (two_columns
                && left.column == Some(Column::Right)
                && right.column == Some(Column::Left)))
        && !end.ends_with([
            '.', '!', '?', '\u{3002}', '\u{ff01}', '\u{ff1f}', '\u{2026}',
        ])
        && !BULLET_POINT_PATTERN.is_match(start)
        && !start.split_whitespace().next().is_some_and(|word| {
            word.trim_end_matches(['.', ')']) != word
                && word
                    .trim_end_matches(['.', ')'])
                    .chars()
                    .all(|ch| ch.is_ascii_digit())
        })
}

pub fn find(document: &mut Document) -> Result<Vec<Vec<ParagraphId>>, AccessError> {
    let view = document.view::<ReadOnlyView>();
    let mut groups = Vec::<Vec<ParagraphId>>::new();
    let mut previous = None::<PageEdges>;
    for index in 0..view.document().page_len(&view)? {
        let current = edges(index, &view)?;
        if let Some(previous) = &previous
            && let (Some(a), Some(b)) = (previous.number, current.number)
            && a.checked_add(1) == Some(b)
            && let (Some(left), Some(right)) = (&previous.last, &current.first)
            && is_continuation(left, right, previous.two_columns && current.two_columns)
        {
            if let Some(group) = groups
                .last_mut()
                .filter(|group| group.last() == Some(&left.id))
            {
                group.push(right.id);
            } else {
                groups.push(vec![left.id, right.id]);
            }
        }
        previous = Some(current);
    }
    Ok(groups)
}

fn cjk_letter(ch: char) -> bool {
    ch.is_alphabetic()
        && matches!(ch as u32,
            0x3040..=0x30ff | 0x3400..=0x9fff | 0xac00..=0xd7af | 0xf900..=0xfaff | 0x20000..=0x323af
        )
}

pub fn join_texts(texts: &[&str]) -> String {
    let mut joined = String::new();
    for text in texts {
        let text = text.trim();
        if joined.ends_with('\u{ad}') {
            joined.pop();
        } else if !joined.is_empty()
            && !(joined.ends_with(cjk_letter) && text.starts_with(cjk_letter))
        {
            joined.push(' ');
        }
        joined.push_str(text);
    }
    joined
}

fn placeholder_positions<'a>(text: &str, tokens: &HashSet<&'a str>) -> Vec<(usize, &'a str)> {
    let mut positions = tokens
        .iter()
        .flat_map(|&token| {
            text.match_indices(token)
                .map(move |(offset, _)| (offset, token))
        })
        .collect::<Vec<_>>();
    positions.sort_unstable();
    positions
}

fn placeholder_ranges(
    text: &str,
    sources: &[&TranslateInput],
) -> Option<Vec<Option<Range<usize>>>> {
    let mut tokens = HashSet::new();
    for source in sources {
        for placeholder in &source.placeholders {
            match placeholder {
                InputPlaceholder::Formula(formula) => {
                    tokens.insert(formula.token.as_str());
                }
                InputPlaceholder::Rich(rich) => {
                    tokens.extend([rich.left.as_str(), rich.right.as_str()]);
                }
            }
        }
    }
    tokens.remove("");
    let mut translated = placeholder_positions(text, &tokens).into_iter();
    let mut previous_end = 0;
    let mut ranges = Vec::with_capacity(sources.len());
    for source in sources {
        let mut start = None;
        for (_, expected) in placeholder_positions(&source.text, &tokens) {
            let (offset, token) = translated.next()?;
            if token != expected || offset < previous_end {
                return None;
            }
            start.get_or_insert(offset);
            previous_end = offset + token.len();
        }
        ranges.push(start.map(|start| start..previous_end));
    }
    translated.next().is_none().then_some(ranges)
}

/// Fit the translation to the original pages at word or CJK character breaks.
/// Use the source placeholder order to identify which page owns each token.
pub fn split_translation(text: &str, sources: &[&TranslateInput]) -> Option<Vec<String>> {
    let text = text.trim();
    let source_lengths = sources
        .iter()
        .map(|input| input.text.trim().chars().count())
        .collect::<Vec<_>>();
    let total_source: usize = source_lengths.iter().sum();
    if total_source == 0 || text.is_empty() {
        return None;
    }
    let mut breaks = Vec::new();
    let mut previous = None;
    let mut length = 0;
    for (index, (offset, ch)) in text.char_indices().enumerate() {
        if previous.is_some_and(|prev: char| {
            (!ch.is_whitespace() && prev.is_whitespace()) || (cjk_letter(prev) && cjk_letter(ch))
        }) {
            breaks.push((index, offset));
        }
        previous = Some(ch);
        length = index + 1;
    }
    let protected = placeholder_ranges(text, sources)?;
    let mut lower_bounds = Vec::with_capacity(sources.len() - 1);
    let mut after = 0;
    for range in &protected[..sources.len() - 1] {
        if let Some(range) = range {
            after = range.end;
        }
        lower_bounds.push(after);
    }
    let mut before = text.len();
    let mut latest = breaks.len();
    let mut candidates_by_page = Vec::with_capacity(lower_bounds.len());
    // Work backwards to reserve a valid break for each later page.
    for (after, range) in lower_bounds
        .into_iter()
        .rev()
        .zip(protected[1..].iter().rev())
    {
        if let Some(range) = range {
            before = range.start;
        }
        let start = breaks.partition_point(|&(_, offset)| offset < after);
        let end = latest.min(breaks.partition_point(|&(_, offset)| offset <= before));
        if start >= end {
            return None;
        }
        candidates_by_page.push(start..end);
        latest = end - 1;
    }

    let mut translated = Vec::with_capacity(source_lengths.len());
    let mut next_break = 0;
    let mut start = 0;
    let mut source_count = 0;
    for (source_length, range) in source_lengths
        .iter()
        .zip(candidates_by_page.into_iter().rev())
    {
        source_count += source_length;
        let target = (length as f64 * source_count as f64 / total_source as f64).round() as usize;
        let first = range.start.max(next_break);
        let best = (first..range.end).min_by_key(|&index| breaks[index].0.abs_diff(target))?;
        let end = breaks[best].1;
        translated.push(text[start..end].trim().to_owned());
        start = end;
        next_break = best + 1;
    }
    translated.push(text[start..].trim().to_owned());
    translated
        .iter()
        .zip(sources)
        .all(|(part, source)| {
            source
                .placeholders
                .iter()
                .all(|placeholder| match placeholder {
                    InputPlaceholder::Formula(_) => true,
                    InputPlaceholder::Rich(rich) => part
                        .split_once(&rich.left)
                        .and_then(|(_, rest)| rest.split_once(&rich.right))
                        .is_some_and(|(content, _)| !content.contains(['\r', '\n'])),
                })
        })
        .then_some(translated)
}
