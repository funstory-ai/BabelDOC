//! Type3 projection metrics, including the reference's f32 SVG geometry roundtrip.
use crate::{
    array, dict, resolve,
    tokens::{Lexer, Token},
};
use lopdf::{Dictionary, Document, Object};
use std::collections::{BTreeMap, HashMap};
use crate::fonts::{FontError, required};

#[derive(Default)]
struct Rectangles(Option<[f32; 4]>);
impl Rectangles {
    fn push(&mut self, r: [f32; 4]) {
        if let Some(bbox) = &mut self.0 {
            bbox[0] = bbox[0].min(r[0]);
            bbox[1] = bbox[1].min(r[1]);
            bbox[2] = bbox[2].max(r[2]);
            bbox[3] = bbox[3].max(r[3]);
        } else { self.0 = Some(r); }
    }
}
struct GlyphExpansion<'a> {
    guard: crate::expansion::Expansion,
    resources: &'a mut crate::fonts::FontResourceCache,
}
impl GlyphExpansion<'_> {
    fn outline(
        &mut self,
        doc: &Document,
        font: &Object,
        code: u8,
    ) -> Result<Option<std::sync::Arc<(Vec<[f32; 2]>, f32)>>, FontError> {
        // The parse-lifetime cache resolves aliases and keys the shared result by
        // the underlying FontFile object id plus every output-affecting parameter.
        crate::fonts::svg_glyph_outline_with_cache(doc, font, code, self.resources)
    }
}

#[derive(Clone, Debug)]
pub struct Metrics {
    pub matrix_text: String,
    pub bbox_text: String,
    pub em_height: f64,
    pub ascent: f64,
    pub descent: f64,
}
fn general12(value: f64) -> String {
    let scientific = format!("{value:.11e}");
    let (mantissa, exponent) = scientific.split_once('e').unwrap();
    let exponent: i32 = exponent.parse().unwrap();
    if !(-4..12).contains(&exponent) {
        format!(
            "{}e{}{:02}",
            mantissa.trim_end_matches('0').trim_end_matches('.'),
            if exponent >= 0 { "+" } else { "-" },
            exponent.abs()
        )
    } else {
        let s = format!("{:.*}", (11 - exponent).max(0) as usize, value);
        if s.contains('.') {
            s.trim_end_matches('0').trim_end_matches('.').into()
        } else {
            s
        }
    }
}
pub fn metrics(doc: &Document, spec: &Dictionary) -> Option<Metrics> {
    let m: [f64; 6] = array(doc, spec.get(b"FontMatrix").ok()?)?;
    let bbox: [f64; 4] = array(doc, spec.get(b"FontBBox").ok()?)?;
    if !m.iter().chain(bbox.iter()).all(|v| v.is_finite()) {
        return None;
    }
    let ys = [
        m[1] * bbox[0] + m[3] * bbox[1] + m[5],
        m[1] * bbox[0] + m[3] * bbox[3] + m[5],
        m[1] * bbox[2] + m[3] * bbox[1] + m[5],
        m[1] * bbox[2] + m[3] * bbox[3] + m[5],
    ];
    let low = ys.into_iter().fold(f64::INFINITY, f64::min);
    let high = ys.into_iter().fold(f64::NEG_INFINITY, f64::max);
    let em_height = high - low;
    if !em_height.is_finite() || em_height <= 0. {
        return None;
    }
    let scale = 1000. / em_height;
    Some(Metrics {
        matrix_text: m.into_iter().map(general12).collect::<Vec<_>>().join(" "),
        bbox_text: bbox
            .into_iter()
            .map(general12)
            .collect::<Vec<_>>()
            .join(" "),
        em_height,
        ascent: high * scale,
        descent: low * scale,
    })
}
type Matrix = [f32; 6];
fn concat(a: Matrix, b: Matrix) -> Matrix {
    [
        a[0] * b[0] + a[1] * b[2],
        a[0] * b[1] + a[1] * b[3],
        a[2] * b[0] + a[3] * b[2],
        a[2] * b[1] + a[3] * b[3],
        a[4] * b[0] + a[5] * b[2] + b[4],
        a[4] * b[1] + a[5] * b[3] + b[5],
    ]
}
fn point(m: Matrix, p: [f32; 2]) -> [f32; 2] {
    [
        p[0] * m[0] + p[1] * m[2] + m[4],
        p[0] * m[1] + p[1] * m[3] + m[5],
    ]
}
fn number(t: &Token) -> Option<f32> {
    match t {
        Token::Int(x) => Some(*x as f32),
        Token::Real(x) => Some(crate::object_numbers::lex_real(&crate::python_float_str(
            *x,
        ))),
        _ => None,
    }
}
fn image_rect(
    attrs: &[(String, Token)],
    ctm: Matrix,
    font_matrix: Matrix,
    rectangles: &mut Rectangles,
) {
    let get = |short: &str, long: &str| {
        attrs
            .iter()
            .find(|(k, _)| k == short || k == long)
            .map(|(_, v)| v)
    };
    if !matches!(get("IM","ImageMask"),Some(Token::Word(w)) if w=="true") {
        return;
    }
    let Some(w) = get("W", "Width").and_then(number) else {
        return;
    };
    let Some(h) = get("H", "Height").and_then(number) else {
        return;
    };
    image_mask_rect(w, h, ctm, font_matrix, rectangles);
}
fn image_mask_rect(w: f32, h: f32, ctm: Matrix, font_matrix: Matrix, rectangles: &mut Rectangles) {
    // PDF image samples are top-down. MuPDF applies this flip before the
    // SVG device normalizes the image to its pixel rectangle; its f32
    // cancellation is observable even though the real-valued bounds agree.
    let ctm = concat(concat([1., 0., 0., -1., 0., 1.], ctm), font_matrix);
    let local = concat([1. / w, 0., 0., 1. / h, 0., 0.], ctm);
    emit(&[[0., 0.], [w, 0.], [w, h], [0., h]], local, rectangles);
}
fn emit(path: &[[f32; 2]], ctm: Matrix, rectangles: &mut Rectangles) {
    if path.is_empty() {
        return;
    }
    let mut bbox = [
        f32::INFINITY,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NEG_INFINITY,
    ];
    // SVG reopens the glyph through a use with (1,0,0,-1,0,0).
    let svg_number =
        |v: f32| crate::object_numbers::lex_real(&crate::object_numbers::printed_real_text(v));
    let ctm = concat(ctm.map(svg_number), [1., 0., 0., -1., 0., 0.]);
    for &p in path {
        let p = point(ctm, p.map(svg_number));
        bbox[0] = bbox[0].min(p[0]);
        bbox[1] = bbox[1].min(p[1]);
        bbox[2] = bbox[2].max(p[0]);
        bbox[3] = bbox[3].max(p[1]);
    }
    rectangles.push(bbox);
}
fn drawings(
    doc: &Document,
    bytes: &[u8],
    matrix: Matrix,
    initial: Matrix,
    resources: Option<&Object>,
    active: &mut Vec<(u32, u16)>,
    rectangles: &mut Rectangles,
    expansion: &mut GlyphExpansion,
    depth: usize,
    source_key: usize,
    source_object: Option<lopdf::ObjectId>,
) -> Result<(), FontError> {
    // Operations belong to this invocation, just like its decoded bytes.
    expansion.guard.enter_program(source_key, bytes.len(), source_object)?;
    let ops = Lexer::for_type3(bytes).operations()?;
    let mut ctm = initial;
    let mut stack = Vec::new();
    let mut path = Vec::new();
    let mut start = [0.; 2];
    let mut last = [0.; 2];
    let mut text_matrix = [1., 0., 0., 1., 0., 0.];
    let mut line_matrix = text_matrix;
    let mut font_name = String::new();
    let mut font_size = 0.;
    let mut horizontal_scale = 1.;
    let mut rise = 0.;
    let mut char_space = 0.;
    let mut word_space = 0.;
    let mut leading = 0.;
    for op in ops.iter() {
        let v: Vec<f32> = op.args.iter().filter_map(number).collect();
        match op.op.as_str() {
            "q" => stack.push(ctm),
            "Q" => {
                if let Some(m) = stack.pop() {
                    ctm = m;
                }
            }
            "cm" if v.len() == 6 => ctm = concat(required(v.try_into().ok())?, ctm),
            "m" if v.len() == 2 => {
                start = [v[0], v[1]];
                last = start;
                // PyMuPDF get_drawings initializes its rectangle from the first
                // moveto, then ignores later subpath movetos when extending it.
                if path.is_empty() {
                    path.push(last);
                }
            }
            "l" if v.len() == 2 => {
                last = [v[0], v[1]];
                path.push(last);
            }
            "c" if v.len() == 6 => {
                for p in v.chunks_exact(2) {
                    path.push([p[0], p[1]]);
                }
                last = [v[4], v[5]];
            }
            "v" if v.len() == 4 => {
                path.push(last);
                path.push([v[0], v[1]]);
                last = [v[2], v[3]];
                path.push(last);
            }
            "y" if v.len() == 4 => {
                path.push([v[0], v[1]]);
                last = [v[2], v[3]];
                path.push(last);
                path.push(last);
            }
            "re" if v.len() == 4 => {
                let [x, y, w, h]: [f32; 4] = required(v.try_into().ok())?;
                path.extend([[x, y], [x + w, y], [x + w, y + h], [x, y + h]]);
                start = [x, y];
                last = start;
            }
            "h" => {
                // Closing a later subpath does not add its moveto to the
                // rectangle reported by PyMuPDF's SVG get_drawings path.
                last = start;
            }
            "S" | "s" | "f" | "F" | "f*" | "B" | "B*" | "b" | "b*" => {
                emit(&path, concat(ctm, matrix), rectangles);
                path.clear();
            }
            "n" => path.clear(),
            "INLINE_IMAGE" => {
                if let Some(Token::Inline(attrs, _)) = op.args.first() {
                    image_rect(attrs, ctm, matrix, rectangles);
                }
            }
            "BT" => {
                text_matrix = [1., 0., 0., 1., 0., 0.];
                line_matrix = text_matrix;
            }
            "Tf" => {
                font_name = required(op.args.first())?.name()?.to_owned();
                font_size = *required(v.last())?;
            }
            "Tm" if v.len() == 6 => {
                text_matrix = required(v.try_into().ok())?;
                line_matrix = text_matrix;
            }
            "Tz" if v.len() == 1 => horizontal_scale = v[0] / 100.,
            "Ts" if v.len() == 1 => rise = v[0],
            "Tc" if v.len() == 1 => char_space = v[0],
            "Tw" if v.len() == 1 => word_space = v[0],
            "TL" if v.len() == 1 => leading = v[0],
            "Td" | "TD" if v.len() == 2 => {
                if op.op == "TD" {
                    leading = -v[1];
                }
                line_matrix = concat([1., 0., 0., 1., v[0], v[1]], line_matrix);
                text_matrix = line_matrix;
            }
            "T*" => {
                line_matrix = concat([1., 0., 0., 1., 0., -leading], line_matrix);
                text_matrix = line_matrix;
            }
            "Tj" | "TJ" => {
                let font = dict(doc, dict(doc, required(resources)?)?.get(b"Font")?)?
                    .get(font_name.as_bytes())?;
                let values = match required(op.args.first())? {
                    Token::Array(a) => a.as_slice(),
                    token => std::slice::from_ref(token),
                };
                for value in values {
                    if let Token::Str(bytes, _) = value {
                        for &code in bytes {
                            if let Some(outline) = expansion.outline(doc, font, code)? {
                                let (points, advance) = outline.as_ref();
                                let trm = concat(
                                    concat(
                                        [font_size * horizontal_scale, 0., 0., font_size, 0., rise],
                                        text_matrix,
                                    ),
                                    concat(ctm, matrix),
                                );
                                emit(points, trm, rectangles);
                                let advance = (advance * font_size
                                    + char_space
                                    + if code == 32 { word_space } else { 0. })
                                    * horizontal_scale;
                                text_matrix = concat([1., 0., 0., 1., advance, 0.], text_matrix);
                            }
                        }
                    } else if let Some(adjust) = number(value) {
                        text_matrix = concat(
                            [
                                1.,
                                0.,
                                0.,
                                1.,
                                -adjust / 1000. * font_size * horizontal_scale,
                                0.,
                            ],
                            text_matrix,
                        );
                    }
                }
            }
            "Do" => {
                let name = required(op.args.first())?.name()?;
                let objects = dict(doc, required(resources)?)?.get(b"XObject")?;
                let object = dict(doc, objects)?.get(name.as_bytes())?;
                let id = if let Object::Reference(id) = object {
                    if active.contains(id) {
                        continue;
                    }
                    active.push(*id);
                    Some(*id)
                } else {
                    None
                };
                if let Ok(Object::Stream(stream)) = resolve(doc, object) {
                    let subtype = stream
                        .dict
                        .get(b"Subtype")
                        .ok()
                        .and_then(|o| o.as_name().ok());
                    if subtype == Some(b"Form") {
                        let child_depth = expansion.guard.child_depth(depth, id)?;
                        let key = stream.as_ref() as *const lopdf::Stream as usize;
                        let m = array(doc, stream.dict.get(b"Matrix").unwrap_or(&Object::Null))
                            .unwrap_or([1., 0., 0., 1., 0., 0.])
                            .map(|x| x as f32);
                        let data = expansion.guard.decode(doc, key, stream)
                            .map_err(|e| match id { Some(id) => e.at_object(id), None => e })?;
                        drawings(
                            doc,
                            &data,
                            matrix,
                            concat(m, ctm),
                            stream.dict.get(b"Resources").ok().or(resources),
                            active,
                            rectangles,
                            expansion,
                            child_depth,
                            key,
                            id,
                        )?;
                    } else if subtype == Some(b"Image")
                        && stream.dict.get(b"ImageMask").ok()
                            .and_then(|o| resolve(doc, o).ok())
                            .and_then(|o| o.as_bool().ok()) == Some(true)
                    {
                        let dimension = |name: &[u8]| {
                            stream.dict.get(name).ok()
                                .and_then(|o| resolve(doc, o).ok())
                                .and_then(crate::numeric)
                                .map(|n| n as f32)
                        };
                        image_mask_rect(required(dimension(b"Width"))?, required(dimension(b"Height"))?, ctm, matrix, rectangles);
                    }
                }
                if id.is_some() {
                    active.pop();
                }
            }
            _ => {}
        }
    }
    Ok(())
}
#[cfg(test)]
pub fn boxes(
    doc: &Document,
    spec: &Dictionary,
    metrics: Option<&Metrics>,
) -> Result<Vec<[f64; 4]>, FontError> {
    let mut resources = crate::fonts::FontResourceCache::default();
    boxes_with_resources(doc, spec, metrics, &mut resources)
}

pub(crate) fn boxes_with_resources(
    doc: &Document,
    spec: &Dictionary,
    metrics: Option<&Metrics>,
    resources: &mut crate::fonts::FontResourceCache,
) -> Result<Vec<[f64; 4]>, FontError> {
    boxes_with_execution(doc, spec, metrics, resources, |_| {})
}

fn boxes_with_execution(
    doc: &Document,
    spec: &Dictionary,
    metrics: Option<&Metrics>,
    resources: &mut crate::fonts::FontResourceCache,
    mut execute: impl FnMut(&[u8]),
) -> Result<Vec<[f64; 4]>, FontError> {
    let matrix: [f64; 6] = required(array(doc, spec.get(b"FontMatrix")?))?;
    let first = resolve(doc, spec.get(b"FirstChar")?)?.as_i64()?;
    let last = resolve(doc, spec.get(b"LastChar")?)?.as_i64()?;
    let procs = dict(doc, spec.get(b"CharProcs")?)?;
    let encoding = dict(doc, spec.get(b"Encoding")?)?;
    let differences = resolve(doc, encoding.get(b"Differences")?)?.as_array()?;
    let mut names = BTreeMap::new();
    let mut cid = 0i64;
    for item in differences {
        match item {
            Object::Integer(n) => cid = *n,
            Object::Name(name) => {
                names.insert(cid, name);
                cid = required(cid.checked_add(1))?;
            }
            _ => {}
        }
    }
    let mut result = vec![[0.; 4]; 256];
    if first > last { return Ok(result); }
    let mut glyph_boxes: HashMap<&[u8], Option<[f64; 4]>> = HashMap::new();
    for (&cid, name) in names.range(first..=last) {
        if !glyph_boxes.contains_key(name.as_slice()) {
            execute(name);
            // Undefined glyph names render no drawings in MuPDF; they do not
            // invalidate the bounding boxes of the font's defined CharProcs.
            let bbox = (|| -> Result<Option<[f64; 4]>, FontError> {
                let Some(object) = procs.get(name).ok() else { return Ok(None); };
                let Some(stream) = resolve(doc, object).ok() else {
                    return Ok(None);
                };
                let Object::Stream(stream) = stream else {
                    return Ok(None);
                };
                let mut rectangles = Rectangles::default();
                let mut expansion = GlyphExpansion {
                    guard: crate::expansion::Expansion::new(doc.parser_policy, "type3_glyph"),
                    resources,
                };
                let source = object.as_reference().ok();
                let key = stream.as_ref() as *const lopdf::Stream as usize;
                expansion.guard.depth(1, source)?;
                let data = expansion.guard.decode(doc, key, stream)?;
                drawings(
                    doc,
                    &data,
                    matrix.map(|x| x as f32),
                    [1., 0., 0., 1., 0., 0.],
                    spec.get(b"Resources").ok(),
                    &mut Vec::new(),
                    &mut rectangles,
                    &mut expansion,
                    1,
                    key,
                    source,
                )?;
                Ok(rectangles.0.map(|bbox| {
                    let scale = metrics.map(|m| 1000. / m.em_height).unwrap_or(1.);
                    [
                        f64::from(bbox[0]) * scale,
                        -f64::from(bbox[3]) * scale,
                        f64::from(bbox[2]) * scale,
                        -f64::from(bbox[1]) * scale,
                    ]
                }))
            })()?;
            glyph_boxes.insert(name.as_slice(), bbox);
        }
        if let Some(b) = glyph_boxes.get(name.as_slice()).copied().flatten() {
            *required(result.get_mut(usize::try_from(cid).map_err(|e| FontError::Pdf(e.to_string()))?))? = b;
        }
    }
    Ok(result)
}

#[cfg(test)]
mod type3_tests;

#[cfg(test)]
mod expansion_tests;
