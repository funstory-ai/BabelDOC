//! 0.6.4 model-box conversion and fallback lines. No Python objects or callbacks.
use crate::{ProgressReporter, StageError};
use babeldoc_ir_core::{self as ir, Cursor, LayoutParserView, ReadOnlyView, TreeView, types};

#[derive(Clone, Debug)]
pub struct Character {
    pub visual_bbox_present: bool,
    pub bounds: Option<[Option<f64>; 4]>,
    pub unicode: Option<String>,
    pub vertical: Option<bool>,
}
#[derive(Clone, Debug)]
pub struct PageInput {
    pub page_number: Option<i64>,
    pub characters: Vec<Character>,
}
#[derive(Clone, Copy, Debug)]
pub enum Coordinate {
    F32(f32),
    F64(f64),
}
#[derive(Clone, Debug)]
pub struct ModelBox {
    pub xyxy: [Coordinate; 4],
    pub conf: f64,
    pub class_name: String,
}
#[derive(Clone, Debug)]
pub struct PageResult {
    pub index: Option<usize>,
    pub mediabox: Option<[f64; 2]>,
    pub boxes: Vec<ModelBox>,
}
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ConversionError {
    Nan,
    Infinite,
}
impl Coordinate {
    fn calc(self, h: f64, mode: usize) -> f64 {
        match self {
            Self::F32(v) => {
                (match mode {
                    0 => v - 1.,
                    1 => h as f32 - v - 1.,
                    2 => v + 1.,
                    _ => h as f32 - v + 1.,
                }) as f64
            }
            Self::F64(v) => match mode {
                0 => v - 1.,
                1 => h - v - 1.,
                2 => v + 1.,
                _ => h - v + 1.,
            },
        }
    }
}
pub fn convert_box(xyxy: [Coordinate; 4], size: [f64; 2]) -> Result<[f64; 4], ConversionError> {
    let [w, h] = size.map(f64::ceil);
    let a = [
        xyxy[0].calc(h, 0),
        xyxy[3].calc(h, 1),
        xyxy[2].calc(h, 2),
        xyxy[1].calc(h, 3),
    ];
    if a.iter().chain([&w, &h]).any(|v| v.is_nan()) {
        return Err(ConversionError::Nan);
    }
    if a.iter().chain([&w, &h]).any(|v| v.is_infinite()) {
        return Err(ConversionError::Infinite);
    }
    // numpy.clip applies the upper bound last, including inverted bounds.
    Ok(std::array::from_fn(|i| {
        a[i].trunc()
            .max(0.)
            .min(if i % 2 == 0 { w - 1. } else { h - 1. })
    }))
}
pub fn pairwise_sum(a: &[f64]) -> f64 {
    let n = a.len();
    if n < 8 {
        return a.iter().fold(-0.0, |r, x| r + x);
    }
    if n <= 128 {
        let mut r: [f64; 8] = a[..8].try_into().unwrap();
        let mut i = 8;
        while i < n - n % 8 {
            for j in 0..8 {
                r[j] += a[i + j];
            }
            i += 8;
        }
        let mut out = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        while i < n {
            out += a[i];
            i += 1;
        }
        return out;
    }
    let mid = (n / 2) / 8 * 8;
    pairwise_sum(&a[..mid]) + pairwise_sum(&a[mid..])
}
pub fn mean(a: &[f64]) -> f64 {
    pairwise_sum(a) / a.len() as f64
}
fn mean_or(a: &[f64], fallback: f64) -> f64 {
    if a.is_empty() { fallback } else { mean(a) }
}
fn groups(n: usize, neighbor: impl Fn(usize, usize) -> bool) -> Vec<Vec<usize>> {
    let mut labels = vec![usize::MAX; n];
    let mut label = 0;
    for i in 0..n {
        if labels[i] != usize::MAX {
            continue;
        }
        let mut stack = vec![i];
        while let Some(k) = stack.pop() {
            if labels[k] != usize::MAX {
                continue;
            }
            labels[k] = label;
            for q in 0..n {
                if labels[q] == usize::MAX && neighbor(k, q) {
                    stack.push(q);
                }
            }
        }
        label += 1;
    }
    let mut result = vec![Vec::new(); label];
    for (i, l) in labels.into_iter().enumerate() {
        result[l].push(i);
    }
    result
}
pub fn euclidean_groups(points: &[f64], eps: f64) -> Vec<Vec<usize>> {
    groups(points.len(), |k, q| {
        let (x, y) = (points[k], points[q]);
        if points.len() <= 11 {
            k == q || x * x - 2. * x * y + y * y <= eps * eps
        } else {
            (x - y).abs() <= eps
        }
    })
}
#[derive(Clone, Debug)]
struct Char {
    b: [f64; 4],
    vertical: bool,
}
type Line = Vec<Char>;
fn bbox(line: &Line) -> [f64; 4] {
    [
        line.iter().map(|c| c.b[0]).fold(f64::INFINITY, f64::min),
        line.iter().map(|c| c.b[1]).fold(f64::INFINITY, f64::min),
        line.iter()
            .map(|c| c.b[2])
            .fold(f64::NEG_INFINITY, f64::max),
        line.iter()
            .map(|c| c.b[3])
            .fold(f64::NEG_INFINITY, f64::max),
    ]
}
fn sorted(line: &mut Line, axis: usize) {
    line.sort_by(|a, b| a.b[axis].partial_cmp(&b.b[axis]).unwrap());
}
fn cluster(mut chars: Line, vertical: bool) -> Result<Vec<Line>, StageError> {
    let main = usize::from(vertical);
    let secondary = 1 - main;
    sorted(&mut chars, secondary);
    let mut bands: Vec<(Line, f64, f64)> = vec![];
    for c in chars {
        let start = c.b[secondary];
        let end = c.b[secondary + 2];
        let size = end - start;
        let mut best = None;
        let mut ratio = 0.5;
        for i in (0..bands.len()).rev() {
            if bands[i].2 < start {
                break;
            }
            let overlap = (end.min(bands[i].2) - start.max(bands[i].1)).max(0.);
            if size > 0. && overlap / size > ratio {
                ratio = overlap / size;
                best = Some(i);
            }
        }
        if let Some(i) = best {
            let mut band = bands.remove(i);
            band.0.push(c);
            band.1 = band.1.min(start);
            band.2 = band.2.max(end);
            bands.push(band);
        } else {
            bands.push((vec![c], start, end));
        }
    }
    let mut lines = vec![];
    for (band, _, _) in bands {
        let sizes: Vec<_> = band
            .iter()
            .map(|c| c.b[main + 2] - c.b[main])
            .filter(|s| *s > 0.)
            .collect();
        let eps = mean_or(&sizes, 10.) * 3.5;
        let centers: Vec<_> = band
            .iter()
            .map(|c| [(c.b[0] + c.b[2]) / 2., (c.b[1] + c.b[3]) / 2.])
            .collect();
        if !eps.is_finite() || eps <= 0. || centers.iter().flatten().any(|x| !x.is_finite()) {
            return Err(StageError::failed("invalid DBSCAN input"));
        }
        for g in groups(band.len(), |i, j| {
            (centers[i][0] - centers[j][0]).abs() + (centers[i][1] - centers[j][1]).abs() <= eps
        }) {
            let mut line: Line = g.iter().map(|i| band[*i].clone()).collect();
            sorted(&mut line, main);
            lines.push(line);
        }
    }
    let mut out = vec![];
    for line in lines {
        let b = bbox(&line);
        let sizes: Vec<_> = line
            .iter()
            .map(|c| c.b[secondary + 2] - c.b[secondary])
            .filter(|s| *s > 0.)
            .collect();
        let max_size = sizes.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        if !sizes.is_empty() && b[secondary + 2] - b[secondary] > max_size * 1.5 && line.len() > 1 {
            let centers: Vec<_> = line
                .iter()
                .map(|c| (c.b[secondary] + c.b[secondary + 2]) / 2.)
                .collect();
            let eps = max_size * 0.5;
            if !eps.is_finite() || eps <= 0. || centers.iter().any(|x| !x.is_finite()) {
                return Err(StageError::failed("invalid split DBSCAN input"));
            }
            for g in euclidean_groups(&centers, eps) {
                let mut sub: Line = g.iter().map(|i| line[*i].clone()).collect();
                sorted(&mut sub, main);
                out.push(sub);
            }
        } else {
            out.push(line);
        }
    }
    Ok(out)
}
fn merge(mut lines: Vec<Line>) -> Vec<Line> {
    let mut out = vec![];
    let mut skip = vec![false; lines.len()];
    for i in 0..lines.len() {
        if skip[i] {
            continue;
        }
        let mut b1 = bbox(&lines[i]);
        let heights: Vec<_> = lines[i]
            .iter()
            .map(|c| c.b[3] - c.b[1])
            .filter(|s| *s > 0.)
            .collect();
        let max_gap = mean_or(&heights, 0.) * 1.5;
        let mut merged = false;
        for j in i + 1..lines.len() {
            if skip[j] {
                continue;
            }
            let b2 = bbox(&lines[j]);
            if b1[1] - b2[3] > max_gap {
                break;
            }
            let inter = (b1[2].min(b2[2]) - b1[0].max(b2[0])).max(0.)
                * (b1[3].min(b2[3]) - b1[1].max(b2[1])).max(0.);
            let area = |b: [f64; 4]| {
                if b[2] > b[0] && b[3] > b[1] {
                    (b[2] - b[0]) * (b[3] - b[1])
                } else {
                    0.
                }
            };
            let (a1, a2) = (area(b1), area(b2));
            let mut join = false;
            if a2 > 0. && a1 >= a2 && inter / a2 > 0.6 {
                join = true;
            } else if a1 > 0. && a2 > a1 && inter / a1 > 0.6 {
                let old = lines[i].clone();
                lines[j].extend(old);
                lines.swap(i, j);
                skip[j] = true;
                merged = true;
                b1 = [
                    b1[0].min(b2[0]),
                    b1[1].min(b2[1]),
                    b1[2].max(b2[2]),
                    b1[3].max(b2[3]),
                ];
                continue;
            } else {
                let main = usize::from(lines[i][0].vertical);
                let secondary = 1 - main;
                let s1 = b1[secondary + 2] - b1[secondary];
                let s2 = b2[secondary + 2] - b2[secondary];
                if s1 > 0. && s2 > 0. {
                    let overlap = (b1[secondary + 2].min(b2[secondary + 2])
                        - b1[secondary].max(b2[secondary]))
                    .max(0.);
                    if overlap / s1 > 0.7 && overlap / s2 > 0.7 {
                        let gap = b1[main].max(b2[main]) - b1[main + 2].min(b2[main + 2]);
                        if gap >= 0. {
                            let sizes: Vec<_> = lines[i]
                                .iter()
                                .chain(&lines[j])
                                .map(|c| c.b[main + 2] - c.b[main])
                                .filter(|s| *s > 0.)
                                .collect();
                            let avg = mean_or(&sizes, 0.);
                            join = avg > 0. && gap < avg * 1.5;
                        }
                    }
                }
            }
            if join {
                let other = lines[j].clone();
                lines[i].extend(other);
                skip[j] = true;
                merged = true;
                b1 = [
                    b1[0].min(b2[0]),
                    b1[1].min(b2[1]),
                    b1[2].max(b2[2]),
                    b1[3].max(b2[3]),
                ];
            }
        }
        if merged {
            let axis = usize::from(lines[i][0].vertical);
            sorted(&mut lines[i], axis);
        }
        out.push(lines[i].clone());
    }
    out
}
pub fn fallback_lines(chars: &[Character]) -> Result<Vec<[f64; 4]>, StageError> {
    let mut horizontal = vec![];
    let mut vertical = vec![];
    for c in chars {
        let b = c
            .bounds
            .ok_or_else(|| StageError::failed("missing visual bbox or box"))?;
        let mut values = [0.; 4];
        for i in 0..4 {
            values[i] = b[i]
                .filter(|x| x.is_finite())
                .ok_or_else(|| StageError::failed("invalid coordinate"))?;
        }
        if c.unicode.is_none() {
            return Err(StageError::failed("missing unicode"));
        }
        let char = Char {
            b: values,
            vertical: c.vertical.unwrap_or(false),
        };
        if char.vertical {
            vertical.push(char);
        } else {
            horizontal.push(char);
        }
    }
    let mut lines = cluster(horizontal, false)?;
    lines.extend(cluster(vertical, true)?);
    let position = |line: &Line| {
        let y: Vec<_> = line.iter().map(|c| (c.b[1] + c.b[3]) / 2.).collect();
        let x: Vec<_> = line.iter().map(|c| (c.b[0] + c.b[2]) / 2.).collect();
        (-mean(&y), mean(&x))
    };
    lines.sort_by(|a, b| position(a).partial_cmp(&position(b)).unwrap());
    Ok(merge(lines).iter().map(bbox).collect())
}
pub fn page_inputs(
    document: &mut ir::Document,
    characters: bool,
) -> Result<Vec<PageInput>, StageError> {
    let v = document.view::<ReadOnlyView>();
    let mut out = vec![];
    for page in v.document().page_iter(&v)? {
        let page = page?;
        let mut chars = vec![];
        if characters {
            for character in page.pdf_character_iter(&v)? {
                let c = character?;
                let b = match c.visual_bbox(&v)? {
                    Some(vb) => vb.r#box(&v)?,
                    None => None,
                };
                let bounds = match b {
                    Some(b) => Some([b.x(&v)?, b.y(&v)?, b.x2(&v)?, b.y2(&v)?]),
                    None => None,
                };
                chars.push(Character {
                    visual_bbox_present: c.visual_bbox(&v)?.is_some(),
                    bounds,
                    unicode: c.char_unicode(&v)?.clone(),
                    vertical: c.vertical(&v)?,
                });
            }
        }
        out.push(PageInput {
            page_number: page.page_number(&v)?,
            characters: chars,
        });
    }
    Ok(out)
}
fn box_value(b: [f64; 4]) -> ir::FieldValue<types::Box> {
    ir::BoxInit {
        x: Some(b[0]),
        y: Some(b[1]),
        x2: Some(b[2]),
        y2: Some(b[3]),
    }
    .into_value()
}
pub fn apply_model(document: &mut ir::Document, result: &PageResult) -> Result<(), StageError> {
    // Conversion is deliberately complete before publication, including foreign pages.
    let mut layouts = vec![];
    for (i, b) in result.boxes.iter().enumerate() {
        let bounds = convert_box(
            b.xyxy,
            result
                .mediabox
                .ok_or_else(|| StageError::failed("missing mediabox"))?,
        )
        .map_err(|e| StageError::failed(format!("{e:?}")))?;
        layouts.push(
            ir::PageLayoutInit {
                id: Some(i as i64 + 1),
                r#box: Some(box_value(bounds)),
                conf: Some(b.conf),
                class_name: Some(b.class_name.clone()),
            }
            .into_value(),
        );
    }
    if let Some(index) = result.index {
        let mut v = document.view::<LayoutParserView>();
        let page = v.document().page_at(&v, index)?;
        page.set_page_layout(&mut v, layouts)?;
    }
    Ok(())
}
fn debug_boxes(
    page: &Cursor<types::Page>,
    v: &mut TreeView<'_, LayoutParserView>,
    green: &Cursor<types::GraphicState>,
) -> Result<(), StageError> {
    for layout_index in 0..page.page_layout_len(v)? {
        let layout = page.page_layout_at(v, layout_index)?;
        let b = layout
            .r#box(v)?
            .ok_or_else(|| StageError::failed("layout.box"))?;
        let values = [b.x(v)?, b.y(v)?, b.x2(v)?, b.y2(v)?];
        let class = layout.class_name(v)?.clone();
        let scale = if class.as_deref() == Some("fallback_line") {
            0.1
        } else {
            1.
        };
        let state = green.share(v)?;
        page.push_pdf_rectangle(
            v,
            ir::PdfRectangleInit {
                r#box: Some(
                    ir::BoxInit {
                        x: values[0],
                        y: values[1],
                        x2: values[2],
                        y2: values[3],
                    }
                    .into_value(),
                ),
                graphic_state: Some(state),
                debug_info: Some(true),
                line_width: Some(0.4 * scale),
                ..Default::default()
            }
            .into_value(),
        )?;
        let y2 = values[3].ok_or_else(|| StageError::failed("debug y2"))?;
        let state = green.share(v)?;
        let pi = page.pdf_paragraph_len(v)?;
        page.push_pdf_paragraph(
            v,
            ir::PdfParagraphInit {
                first_line_indent: Some(false),
                vertical: Some(false),
                xobj_id: Some(-1),
                unicode: class.clone(),
                r#box: Some(
                    ir::BoxInit {
                        x: values[0],
                        y: Some(y2),
                        x2: values[2],
                        y2: Some(y2 + 5.),
                    }
                    .into_value(),
                ),
                pdf_style: Some(
                    ir::PdfStyleInit {
                        font_id: Some("base".into()),
                        font_size: Some(4. * scale),
                        graphic_state: Some(state),
                    }
                    .into_value(),
                ),
                ..Default::default()
            }
            .into_value(),
        )?;
        let p = page.pdf_paragraph_at(v, pi)?;
        let style = p
            .pdf_style(v)?
            .ok_or_else(|| StageError::failed("debug style"))?
            .share(v)?;
        p.push_pdf_paragraph_composition(
            v,
            ir::PdfParagraphCompositionInit {
                pdf_same_style_unicode_characters: Some(
                    ir::PdfSameStyleUnicodeCharactersInit {
                        unicode: class,
                        pdf_style: Some(style),
                        debug_info: Some(true),
                    }
                    .into_value(),
                ),
                ..Default::default()
            }
            .into_value(),
        )?;
    }
    Ok(())
}
fn fallback_page(
    page: &Cursor<types::Page>,
    v: &mut TreeView<'_, LayoutParserView>,
    green: Option<&Cursor<types::GraphicState>>,
) -> Result<(), StageError> {
    let mut chars = vec![];
    for child_index in 0..page.pdf_character_len(v)? {
        let c = page.pdf_character_at(v, child_index)?;
        let b = c
            .visual_bbox(v)?
            .ok_or_else(|| StageError::failed("visual_bbox"))?
            .r#box(v)?
            .ok_or_else(|| StageError::failed("visual_bbox.box"))?;
        chars.push(Character {
            visual_bbox_present: true,
            bounds: Some([b.x(v)?, b.y(v)?, b.x2(v)?, b.y2(v)?]),
            unicode: c.char_unicode(v)?.clone(),
            vertical: c.vertical(v)?,
        });
    }
    if chars.is_empty() {
        return Ok(());
    }
    let mut layout_len = page.page_layout_len(v)?;
    for b in fallback_lines(&chars)? {
        let id = layout_len as i64 + 1;
        page.push_page_layout(
            v,
            ir::PageLayoutInit {
                id: Some(id),
                conf: Some(1.),
                class_name: Some("fallback_line".into()),
                r#box: Some(box_value(b)),
            }
            .into_value(),
        )?;
        layout_len += 1;
    }
    if let Some(green) = green {
        debug_boxes(page, v, green)?;
    }
    Ok(())
}
pub fn fallback<P: ProgressReporter>(
    document: &mut ir::Document,
    debug: bool,
    progress: &P,
) -> Result<(), StageError> {
    let green = if debug {
        Some(document.view::<LayoutParserView>().create_pool(ir::GraphicStateInit{passthrough_per_char_instruction:Some(ir::Instructions::Materialized("0.2039215686 0.7803921569 0.3490196078 rg 0.2039215686 0.7803921569 0.3490196078 RG".into())),..Default::default()}.into_value())?)
    } else {
        None
    };
    let mut v = document.view::<LayoutParserView>();
    let pages = v.document().page_iter(&v)?.collect::<Result<Vec<_>, _>>()?;
    // Errors mirror discarded Python futures; implementation panics propagate.
    // Preserve the old job ordering when a stored page appears more than once.
    let mut page_groups: Vec<(Cursor<types::Page>, usize)> = Vec::new();
    for page in pages {
        if let Some((_, count)) = page_groups
            .iter_mut()
            .find(|(grouped, _)| grouped.identity() == page.identity())
        {
            *count += 1;
        } else {
            page_groups.push((page, 1));
        }
    }
    for (page, count) in page_groups {
        for _ in 0..count {
            let _ = fallback_page(&page, &mut v, green.as_ref());
            let _ = progress.report(1, None);
        }
    }
    drop(v);
    document.compact();
    Ok(())
}

#[cfg(test)]
mod layout_parser_tests;
