use crate::{Result, WriterError, codecs, required};
use babeldoc_ir_core::{Cursor, ReadOnlyView, TreeView, types};
use std::collections::{BTreeMap, BTreeSet, HashMap};

type View<'a> = TreeView<'a, ReadOnlyView>;
type FontId = Option<String>;
type Scope = Option<i64>;
type Fonts = HashMap<FontId, Font>;
#[derive(Clone, Debug)]
struct Font {
    encoding: Option<i64>,
    subtype: Option<String>,
    em: Option<f64>,
}
#[derive(Clone, Debug, Default)]
pub struct AvailableFonts {
    pub page: BTreeSet<String>,
    /// Form entries are the union of page and local resource names, as in Python.
    pub forms: HashMap<Scope, BTreeSet<String>>,
}
#[derive(Clone, Debug, Default)]
pub struct BuildOptions {
    pub skip_char: bool,
    pub check_font_exists: bool,
    pub available: Option<AvailableFonts>,
    pub debug: bool,
    pub ocr_workaround: bool,
    pub skip_form_render: bool,
    pub skip_curve_render: bool,
}
#[derive(Clone, Debug)]
pub struct FormOutput {
    pub xobj_id: Scope,
    pub xref_id: i64,
    pub stream: Vec<u8>,
}
#[derive(Clone, Debug, Default)]
pub struct BuildCounts {
    pub units: usize,
    pub form_units: usize,
    pub inline_images: usize,
}
#[derive(Clone, Debug)]
pub struct PageOutput {
    pub page_number: i64,
    pub stream: Vec<u8>,
    pub forms: Vec<FormOutput>,
    pub counts: BuildCounts,
}
enum Unit {
    Character(Cursor<types::PdfCharacter>),
    Form(Cursor<types::PdfForm>),
    Rectangle(Cursor<types::PdfRectangle>, f64),
    Curve(Cursor<types::PdfCurve>),
}
struct Ordered {
    order: i64,
    sub: i64,
    scope: Scope,
    unit: Unit,
}
const LAST: i64 = 9_999_999_999_999_999;

struct FmtBytes<'a>(&'a mut Vec<u8>);

impl std::fmt::Write for FmtBytes<'_> {
    fn write_str(&mut self, value: &str) -> std::fmt::Result {
        self.0.extend_from_slice(value.as_bytes());
        Ok(())
    }
}

fn append_fmt(out: &mut Vec<u8>, args: std::fmt::Arguments<'_>) {
    std::fmt::write(&mut FmtBytes(out), args).expect("writing to Vec<u8> cannot fail");
}

fn append_graphic_state(out: &mut Vec<u8>, state: &str) {
    out.extend_from_slice(state.as_bytes());
    out.extend_from_slice(b" \n");
}

fn append_number(out: &mut Vec<u8>, n: f64) {
    if n.is_nan() {
        out.extend_from_slice(b"nan");
    } else if n == f64::INFINITY {
        out.extend_from_slice(b"inf");
    } else if n == f64::NEG_INFINITY {
        out.extend_from_slice(b"-inf");
    } else {
        append_fmt(out, format_args!("{n:.6}"));
    }
}

fn append_character_prefix(
    out: &mut Vec<u8>,
    font_id: &str,
    size: f64,
    orient: &str,
    x: f64,
    y: f64,
) {
    out.extend_from_slice(b"BT /");
    out.extend_from_slice(font_id.as_bytes());
    out.extend_from_slice(b" ");
    append_number(out, size);
    out.extend_from_slice(b" Tf ");
    out.extend_from_slice(orient.as_bytes());
    out.extend_from_slice(b" ");
    append_number(out, x);
    out.extend_from_slice(b" ");
    append_number(out, y);
    out.extend_from_slice(b" Tm ");
}

fn append_character_glyph(out: &mut Vec<u8>, id: i64, width: usize) {
    out.push(b'<');
    if id < 0 {
        out.push(b'-');
        append_fmt(
            out,
            format_args!(
                "{:0width$X}",
                id.unsigned_abs(),
                width = width.saturating_sub(1)
            ),
        );
    } else {
        append_fmt(out, format_args!("{id:0width$X}"));
    }
    out.extend_from_slice(b"> Tj ET Q \n");
}

pub fn number(n: f64, upper: bool) -> String {
    let s = if n.is_nan() {
        "nan".into()
    } else if n == f64::INFINITY {
        "inf".into()
    } else if n == f64::NEG_INFINITY {
        "-inf".into()
    } else {
        format!("{n:.6}")
    };
    if upper { s.to_uppercase() } else { s }
}
fn matrix(m: &[f64]) -> Vec<u8> {
    format!(
        " {} cm ",
        m.iter()
            .map(|x| number(*x, false))
            .collect::<Vec<_>>()
            .join(" ")
    )
    .into_bytes()
}
fn float_json(v: &babeldoc_ir_core::PythonValue, strings: bool) -> Option<f64> {
    match v {
        babeldoc_ir_core::PythonValue::Bool(b) => Some(if *b { 1. } else { 0. }),
        babeldoc_ir_core::PythonValue::String(s) if strings => s.trim().parse().ok(),
        babeldoc_ir_core::PythonValue::Bytes(b) | babeldoc_ir_core::PythonValue::ByteArray(b)
            if strings =>
        {
            std::str::from_utf8(b).ok()?.trim().parse().ok()
        }
        _ => v.as_f64(),
    }
}
fn relocation(out: &mut Vec<u8>, m: &[babeldoc_ir_core::PythonValue]) {
    if m.len() == 6 {
        if let Some(m) = m
            .iter()
            .map(|v| float_json(v, true))
            .collect::<Option<Vec<_>>>()
        {
            out.extend(matrix(&m));
        }
    }
}
fn instruction(
    v: &View<'_>,
    g: Option<Cursor<types::GraphicState>>,
    strict: bool,
) -> Result<String> {
    match g {
        Some(g) => match g.instruction_text(v)? {
            Some(s) => Ok(s),
            None if strict => Err(WriterError::Invalid(
                "missing passthrough instruction".into(),
            )),
            None => Ok(String::new()),
        },
        None if strict => Err(WriterError::Invalid("missing graphic state".into())),
        None => Ok(String::new()),
    }
}
fn collect_fonts(
    v: &View<'_>,
    fonts: impl Iterator<
        Item = std::result::Result<Cursor<types::PdfFont>, babeldoc_ir_core::AccessError>,
    >,
) -> Result<Fonts> {
    fonts
        .map(|f| {
            let f = f?;
            Ok((
                f.font_id(v)?.clone(),
                Font {
                    encoding: f.encoding_length(v)?,
                    subtype: f.font_subtype(v)?.clone(),
                    em: f.type3_em_height(v)?,
                },
            ))
        })
        .collect()
}
fn chars_and_formula(
    v: &View<'_>,
    page: &Cursor<types::Page>,
) -> Result<(
    Vec<Cursor<types::PdfCharacter>>,
    Vec<Cursor<types::PdfForm>>,
    Vec<Cursor<types::PdfCurve>>,
)> {
    let mut chars = page
        .pdf_character_iter(v)?
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let (mut forms, mut curves) = (vec![], vec![]);
    for paragraph in page.pdf_paragraph_iter(v)? {
        let p = paragraph?;
        for composition in p.pdf_paragraph_composition_iter(v)? {
            let c = composition?;
            if let Some(ch) = c.pdf_character(v)? {
                chars.push(ch);
            } else if let Some(f) = c.pdf_formula(v)? {
                chars.extend(
                    f.pdf_character_iter(v)?
                        .collect::<std::result::Result<Vec<_>, _>>()?,
                );
            } else {
                log::error!("Unknown composition type during PDF rendering");
            }
            if let Some(f) = c.pdf_formula(v)? {
                forms.extend(
                    f.pdf_form_iter(v)?
                        .collect::<std::result::Result<Vec<_>, _>>()?,
                );
                curves.extend(
                    f.pdf_curve_iter(v)?
                        .collect::<std::result::Result<Vec<_>, _>>()?,
                );
            }
        }
    }
    Ok((chars, forms, curves))
}
fn units(v: &View<'_>, p: &Cursor<types::Page>, o: &BuildOptions) -> Result<Vec<Ordered>> {
    let (chars, forms, curves) = chars_and_formula(v, p)?;
    let mut units = vec![];
    if !o.skip_char {
        for c in chars {
            units.push(Ordered {
                order: c.render_order(v)?.unwrap_or(LAST),
                sub: c.sub_render_order(v)?.unwrap_or(LAST),
                scope: c.xobj_id(v)?,
                unit: Unit::Character(c),
            });
        }
    }
    if !o.skip_form_render {
        for (i, form) in p
            .pdf_form_iter(v)?
            .chain(forms.into_iter().map(Ok))
            .enumerate()
        {
            let f = form?;
            units.push(Ordered {
                order: f.render_order(v)?.unwrap_or(LAST),
                sub: i as i64,
                scope: f.xobj_id(v)?,
                unit: Unit::Form(f),
            });
        }
    }
    for (i, rectangle) in p.pdf_rectangle_iter(v)?.enumerate() {
        let r = rectangle?;
        let debug = r.debug_info(v)?.unwrap_or(false);
        if (o.ocr_workaround && !debug && r.fill_background(v)?.unwrap_or(false))
            || (o.debug && debug)
        {
            units.push(Ordered {
                order: r.render_order(v)?.unwrap_or(LAST),
                sub: i as i64,
                scope: r.xobj_id(v)?,
                unit: Unit::Rectangle(r, if o.ocr_workaround { 0.1 } else { 0.4 }),
            });
        }
    }
    if !o.skip_curve_render {
        for (i, curve) in p
            .pdf_curve_iter(v)?
            .chain(curves.into_iter().map(Ok))
            .enumerate()
        {
            let c = curve?;
            let debug =
                c.debug_info(v)?
                    .as_ref()
                    .is_some_and(|d| match serde_json::to_value(d).ok() {
                        Some(serde_json::Value::Bool(b)) => b,
                        Some(serde_json::Value::String(s)) => !s.is_empty(),
                        _ => false,
                    });
            if c.passthrough_paint(v)?.unwrap_or(false) || debug || o.debug {
                units.push(Ordered {
                    order: c.render_order(v)?.unwrap_or(LAST),
                    sub: i as i64,
                    scope: c.xobj_id(v)?,
                    unit: Unit::Curve(c),
                });
            }
        }
    }
    units.sort_by_key(|u| (u.order, u.sub));
    Ok(units)
}

/// Pure IR navigation and stream construction. No PDF mutations or Python calls.
pub fn build_page_output(
    v: &View<'_>,
    page: &Cursor<types::Page>,
    o: &BuildOptions,
) -> Result<PageOutput> {
    let crop = required(
        required(page.cropbox(v)?, "page.cropbox")?.r#box(v)?,
        "cropbox.box",
    )?;
    let mut stream = matrix(&[
        1.,
        0.,
        0.,
        1.,
        -required(crop.x(v)?, "cropbox.x")?,
        -required(crop.y(v)?, "cropbox.y")?,
    ]);
    stream.extend(b" \n");
    let page_fonts = collect_fonts(v, page.pdf_font_iter(v)?)?;
    let mut all_encoding: HashMap<_, _> = page_fonts
        .iter()
        .map(|(k, f)| (k.clone(), f.encoding))
        .collect();
    let mut scoped = HashMap::new();
    let mut form_streams = BTreeMap::new();
    let mut form_refs = vec![];
    for x in page.pdf_xobject_iter(v)? {
        let x = x?;
        let id = x.xobj_id(v)?;
        let local = collect_fonts(v, x.pdf_font_iter(v)?)?;
        all_encoding.extend(local.iter().map(|(k, f)| (k.clone(), f.encoding)));
        let mut fonts = page_fonts.clone();
        fonts.extend(local.clone());
        // Encoding lengths intentionally merge in the opposite order to font metrics.
        let mut encoding: HashMap<_, _> =
            local.iter().map(|(k, f)| (k.clone(), f.encoding)).collect();
        encoding.extend(page_fonts.iter().map(|(k, f)| (k.clone(), f.encoding)));
        scoped.insert(id, (fonts, encoding));
        let form_stream =
            codecs::operations(&required(x.base_operations(v)?, "base_operations")?.value(v)?)?;
        form_streams.insert(id, form_stream.clone());
        form_refs.push((id, required(x.xref_id(v)?, "xref_id")?));
    }
    let mut counts = BuildCounts::default();
    for u in units(v, page, o)? {
        counts.units += 1;
        let target = if let Some(out) = form_streams.get_mut(&u.scope) {
            counts.form_units += 1;
            out
        } else {
            &mut stream
        };
        match u.unit {
            Unit::Character(c) => render_character(
                v,
                &c,
                u.scope,
                target,
                &page_fonts,
                &scoped,
                &all_encoding,
                o,
                false,
            )?,
            Unit::Form(f) => {
                if render_form(v, &f, target)? {
                    counts.inline_images += 1;
                }
            }
            Unit::Rectangle(r, w) => render_rectangle(v, &r, target, w, false)?,
            Unit::Curve(c) => render_curve(v, &c, target)?,
        }
    }
    let forms = form_refs
        .into_iter()
        .map(|(xobj_id, xref_id)| {
            Ok(FormOutput {
                xobj_id,
                xref_id,
                stream: form_streams[&xobj_id].clone(),
            })
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(PageOutput {
        page_number: required(page.page_number(v)?, "page_number")?,
        stream,
        forms,
        counts,
    })
}

fn render_character(
    v: &View<'_>,
    c: &Cursor<types::PdfCharacter>,
    scope: Scope,
    out: &mut Vec<u8>,
    page: &Fonts,
    scoped: &HashMap<Scope, (Fonts, HashMap<FontId, Option<i64>>)>,
    all: &HashMap<FontId, Option<i64>>,
    o: &BuildOptions,
    debug: bool,
) -> Result<()> {
    if c.char_unicode(v)?.as_deref() == Some("\n") {
        return Ok(());
    }
    let Some(id) = c.pdf_character_id(v)? else {
        return Ok(());
    };
    let style = required(c.pdf_style(v)?, "char.pdf_style")?;
    let font_id = style.font_id(v)?;
    let mut size = required(style.font_size(v)?, "font_size")?;
    if o.check_font_exists || debug {
        let available = required(o.available.as_ref(), "available font names")?;
        let names = available.forms.get(&scope).unwrap_or(&available.page);
        if !font_id.as_ref().is_some_and(|f| names.contains(f)) {
            return Ok(());
        }
    }
    if !debug {
        if let Some(font) = scoped
            .get(&scope)
            .and_then(|(m, _)| m.get(&font_id))
            .or_else(|| page.get(&font_id))
        {
            if font.subtype.as_deref() == Some("Type3") {
                if let Some(em) = font.em.filter(|e| e.is_finite() && *e > 0.) {
                    size /= em;
                }
            }
        }
    }
    out.extend(b"q ");
    let state = instruction(v, style.graphic_state(v)?, false)?;
    if !state.is_empty() {
        append_graphic_state(out, &state);
    }
    let b = required(c.r#box(v)?, "char.box")?;
    let x = if c.vertical(v)?.unwrap_or(false) {
        required(b.x2(v)?, "box.x2")?
    } else {
        required(b.x(v)?, "box.x")?
    };
    let y = required(b.y(v)?, "box.y")?;
    let orient = if c.vertical(v)?.unwrap_or(false) {
        "0 1 -1 0"
    } else {
        "1 0 0 1"
    };
    append_character_prefix(
        out,
        font_id.as_deref().unwrap_or("None"),
        size,
        orient,
        x,
        y,
    );
    let encoding = if debug {
        page.get(&font_id).and_then(|f| f.encoding)
    } else {
        scoped
            .get(&scope)
            .map(|(_, e)| e.get(&font_id).copied().flatten())
            .unwrap_or_else(|| page.get(&font_id).and_then(|f| f.encoding))
            .or_else(|| all.get(&font_id).copied().flatten())
    };
    let Some(encoding) = encoding else {
        if debug {
            return Err(WriterError::Invalid("missing debug encoding length".into()));
        }
        return Ok(());
    };
    let width = usize::try_from((encoding * 2).max(0))
        .map_err(|_| WriterError::Invalid("encoding width".into()))?;
    append_character_glyph(out, id, width);
    Ok(())
}

fn render_rectangle(
    v: &View<'_>,
    r: &Cursor<types::PdfRectangle>,
    out: &mut Vec<u8>,
    fallback: f64,
    debug: bool,
) -> Result<()> {
    let b = required(r.r#box(v)?, "rectangle.box")?;
    let x = required(b.x(v)?, "box.x")?;
    let y = required(b.y(v)?, "box.y")?;
    let width = required(b.x2(v)?, "box.x2")? - x;
    let height = required(b.y2(v)?, "box.y2")? - y;
    out.extend(if debug {
        b"q ".as_slice()
    } else {
        b"q n ".as_slice()
    });
    out.extend(instruction(v, r.graphic_state(v)?, true)?.as_bytes());
    let line = r.line_width(v)?.unwrap_or(fallback);
    if line > 0. {
        out.extend(format!(" {} w ", number(line, false)).as_bytes());
    }
    out.extend(
        format!(
            "{} {} {} {} re ",
            number(x, false),
            number(y, false),
            number(width, false),
            number(height, false)
        )
        .as_bytes(),
    );
    out.extend(if r.fill_background(v)?.unwrap_or(false) {
        b" f "
    } else {
        b" S "
    });
    out.extend(if debug {
        b" n Q\n".as_slice()
    } else {
        b"Q\n".as_slice()
    });
    Ok(())
}
fn render_curve(v: &View<'_>, c: &Cursor<types::PdfCurve>, out: &mut Vec<u8>) -> Result<()> {
    out.extend(b"q n ");
    relocation(out, c.relocation_transform(v)?);
    out.extend(b" ");
    let ctm = c.ctm(v)?;
    if ctm.len() == 6 {
        let m = ctm
            .iter()
            .map(|n| {
                float_json(n.as_ref(), false)
                    .ok_or_else(|| WriterError::Invalid("invalid curve CTM number".into()))
            })
            .collect::<Result<Vec<_>>>()?;
        out.extend(
            format!(
                "{} cm ",
                m.iter()
                    .map(|x| number(*x, false))
                    .collect::<Vec<_>>()
                    .join(" ")
            )
            .as_bytes(),
        );
    }
    out.extend(b" ");
    out.extend(instruction(v, c.graphic_state(v)?, true)?.as_bytes());
    if c.passthrough_paint(v)?.unwrap_or(false) {
        out.extend(b" Q\n");
        return Ok(());
    }
    out.extend(b" ");
    let mut path = b" ".to_vec();
    let mut primitive = false;
    if let Some(p) = c.pdf_original_path_primitive(v)? {
        let a = p.args(v)?;
        if p.op(v)?.as_deref() == Some("re") && a.len() == 4 && a.iter().all(|x| x.is_finite()) {
            path.extend(
                format!(
                    "{} re ",
                    a.iter()
                        .map(|x| number(*x, true))
                        .collect::<Vec<_>>()
                        .join(" ")
                )
                .as_bytes(),
            );
            primitive = true;
        }
    }
    if !primitive {
        for original_path in c.pdf_original_path_iter(v)? {
            let p = original_path?;
            let p = required(p.pdf_path(v)?, "original_path.pdf_path")?;
            if p.has_xy(v)?.unwrap_or(false) {
                path.extend(
                    format!(
                        "{} {} {} ",
                        number(required(p.x(v)?, "path.x")?, true),
                        number(required(p.y(v)?, "path.y")?, true),
                        p.op(v)?.as_deref().unwrap_or("None")
                    )
                    .as_bytes(),
                );
            } else {
                path.extend(format!("{} ", p.op(v)?.as_deref().unwrap_or("None")).as_bytes());
            }
        }
    }
    if c.fill_background(v)?.unwrap_or(false) {
        out.extend(&path);
        out.extend(b" f");
    }
    out.extend(if c.evenodd(v)?.unwrap_or(false) {
        b"* ".as_slice()
    } else {
        b" ".as_slice()
    });
    if c.stroke_path(v)?.unwrap_or(false) {
        out.extend(&path);
        out.extend(b"S ");
    }
    out.extend(b" n Q\n");
    Ok(())
}
fn render_form(v: &View<'_>, f: &Cursor<types::PdfForm>, out: &mut Vec<u8>) -> Result<bool> {
    out.extend(b"q ");
    let m = required(f.pdf_matrix(v)?, "form.pdf_matrix")?;
    relocation(out, f.relocation_transform(v)?);
    out.extend(matrix(&[
        required(m.a(v)?, "matrix.a")?,
        required(m.b(v)?, "matrix.b")?,
        required(m.c(v)?, "matrix.c")?,
        required(m.d(v)?, "matrix.d")?,
        required(m.e(v)?, "matrix.e")?,
        required(m.f(v)?, "matrix.f")?,
    ]));
    out.extend(b" ");
    out.extend(instruction(v, f.graphic_state(v)?, true)?.as_bytes());
    out.extend(b" ");
    let sub = required(f.pdf_form_subtype(v)?, "form.pdf_form_subtype")?;
    let mut inline = false;
    if let Some(x) = sub.pdf_xobj_form(v)? {
        out.extend(format!(" /{} Do ", x.do_args(v)?.as_deref().unwrap_or("None")).as_bytes());
    } else if let Some(image) = sub.pdf_inline_form(v)? {
        inline = true;
        out.extend(b" BI ");
        if let Some(params) = image
            .image_parameters(v)?
            .as_deref()
            .filter(|s| !s.is_empty())
        {
            // Parse the original JSON string without sorting its dictionary keys.
            if let Ok(token) = JsonToken::parse(&params) {
                let JsonToken::Map(entries) = token else {
                    return Err(WriterError::Invalid(
                        "inline image parameters must have .items()".into(),
                    ));
                };
                for (key, value) in entries {
                    out.extend(
                        format!(
                            "/{} {} ",
                            key.strip_prefix('/').unwrap_or(&key),
                            value.pdf(0)?
                        )
                        .as_bytes(),
                    );
                }
            }
        }
        out.extend(b"ID ");
        if let Some(data) = image.form_data(v)? {
            if let Some(data) = codecs::base64(&data) {
                out.extend(data);
            }
        }
        out.extend(b" EI ");
    }
    out.extend(b" Q\n");
    Ok(inline)
}

/// Debug drawing uses the existing first stream, unadjusted Type3 sizes and no sort.
pub fn build_debug_stream(
    v: &View<'_>,
    p: &Cursor<types::Page>,
    base: &[u8],
    available: AvailableFonts,
) -> Result<Vec<u8>> {
    let mut out = b"q ".to_vec();
    out.extend(base);
    out.extend(b" Q ");
    let b = required(required(p.cropbox(v)?, "cropbox")?.r#box(v)?, "cropbox.box")?;
    out.extend(
        format!(
            "q Q 1 0 0 1 {} {} cm \n",
            number(required(b.x(v)?, "box.x")?, false),
            number(required(b.y(v)?, "box.y")?, false)
        )
        .as_bytes(),
    );
    let fonts = collect_fonts(v, p.pdf_font_iter(v)?)?;
    let o = BuildOptions {
        available: Some(available),
        ..Default::default()
    };
    for c in chars_and_formula(v, p)?.0 {
        if c.debug_info(v)?.unwrap_or(false) {
            render_character(
                v,
                &c,
                None,
                &mut out,
                &fonts,
                &HashMap::new(),
                &HashMap::new(),
                &o,
                true,
            )?;
        }
    }
    for rectangle in p.pdf_rectangle_iter(v)? {
        let r = rectangle?;
        if r.debug_info(v)?.unwrap_or(false) {
            render_rectangle(v, &r, &mut out, 0.4, true)?;
        }
    }
    Ok(out)
}

// Ordered Python JSON subset, including NaN / Infinity and duplicate-key replacement.
enum JsonToken {
    Atom(String),
    Float(f64),
    String(String),
    List(Vec<Self>),
    Map(Vec<(String, Self)>),
}
impl JsonToken {
    fn parse(s: &str) -> std::result::Result<Self, ()> {
        fn ws(s: &str, i: &mut usize) {
            while s
                .as_bytes()
                .get(*i)
                .is_some_and(|c| matches!(c, b' ' | b'\n' | b'\r' | b'\t'))
            {
                *i += 1;
            }
        }
        fn value(s: &str, i: &mut usize, depth: usize) -> std::result::Result<JsonToken, ()> {
            if depth > 512 {
                return Err(());
            }
            ws(s, i);
            let start = *i;
            let c = *s.as_bytes().get(*i).ok_or(())?;
            if c == b'"' {
                *i += 1;
                while *i < s.len() {
                    match s.as_bytes()[*i] {
                        b'\\' => *i += 2,
                        b'"' => {
                            *i += 1;
                            return serde_json::from_str::<String>(&s[start..*i])
                                .map(JsonToken::String)
                                .map_err(|_| ());
                        }
                        _ => *i += 1,
                    }
                }
                return Err(());
            }
            if c == b'[' || c == b'{' {
                *i += 1;
                let end = if c == b'[' { b']' } else { b'}' };
                let (mut list, mut map) = (vec![], vec![]);
                ws(s, i);
                if s.as_bytes().get(*i) == Some(&end) {
                    *i += 1;
                    return Ok(if c == b'[' {
                        JsonToken::List(list)
                    } else {
                        JsonToken::Map(map)
                    });
                }
                loop {
                    if c == b'[' {
                        list.push(value(s, i, depth + 1)?);
                    } else {
                        let JsonToken::String(k) = value(s, i, depth + 1)? else {
                            return Err(());
                        };
                        ws(s, i);
                        if s.as_bytes().get(*i) != Some(&b':') {
                            return Err(());
                        }
                        *i += 1;
                        let v = value(s, i, depth + 1)?;
                        if let Some((_, old)) = map.iter_mut().find(|(key, _)| key == &k) {
                            *old = v;
                        } else {
                            map.push((k, v));
                        }
                    }
                    ws(s, i);
                    match s.as_bytes().get(*i) {
                        Some(x) if *x == end => {
                            *i += 1;
                            break;
                        }
                        Some(b',') => *i += 1,
                        _ => return Err(()),
                    }
                }
                return Ok(if c == b'[' {
                    JsonToken::List(list)
                } else {
                    JsonToken::Map(map)
                });
            }
            while s
                .as_bytes()
                .get(*i)
                .is_some_and(|c| !matches!(c, b' ' | b'\n' | b'\r' | b'\t' | b',' | b']' | b'}'))
            {
                *i += 1;
            }
            let text = &s[start..*i];
            match text {
                "NaN" => Ok(JsonToken::Float(f64::NAN)),
                "Infinity" => Ok(JsonToken::Float(f64::INFINITY)),
                "-Infinity" => Ok(JsonToken::Float(f64::NEG_INFINITY)),
                "null" | "true" | "false" => Ok(JsonToken::Atom(text.into())),
                _ => {
                    if !json_number(text) {
                        return Err(());
                    }
                    if text.contains(['.', 'e', 'E']) {
                        Ok(JsonToken::Float(text.parse().map_err(|_| ())?))
                    } else {
                        Ok(JsonToken::Atom(if text == "-0" {
                            "0".into()
                        } else {
                            text.into()
                        }))
                    }
                }
            }
        }
        let mut i = 0;
        let result = value(s, &mut i, 0)?;
        ws(s, &mut i);
        if i == s.len() { Ok(result) } else { Err(()) }
    }
    fn pdf(&self, depth: usize) -> Result<String> {
        if depth > 128 {
            return Err(WriterError::Invalid(
                "PDF token serialization exceeded max depth 128".into(),
            ));
        }
        Ok(match self {
            Self::Atom(s) | Self::String(s) => s.clone(),
            Self::Float(f) => number(*f, false),
            Self::List(a) => format!(
                "[{}]",
                a.iter()
                    .map(|v| v.pdf(depth + 1))
                    .collect::<Result<Vec<_>>>()?
                    .join(" ")
            ),
            Self::Map(m) => format!(
                "<< {} >>",
                m.iter()
                    .map(|(k, v)| Ok(format!(
                        "/{} {}",
                        k.strip_prefix('/').unwrap_or(k),
                        v.pdf(depth + 1)?
                    )))
                    .collect::<Result<Vec<_>>>()?
                    .join(" ")
            ),
        })
    }
}

// Validate JSON number syntax without imposing JSON's finite-f64 range on
// Python json.loads (which accepts exponent overflow and arbitrary integers).
fn json_number(s: &str) -> bool {
    let b = s.as_bytes();
    let mut i = usize::from(b.first() == Some(&b'-'));
    match b.get(i) {
        Some(b'0') => i += 1,
        Some(b'1'..=b'9') => {
            i += 1;
            while b.get(i).is_some_and(u8::is_ascii_digit) {
                i += 1;
            }
        }
        _ => return false,
    }
    if b.get(i) == Some(&b'.') {
        i += 1;
        let start = i;
        while b.get(i).is_some_and(u8::is_ascii_digit) {
            i += 1;
        }
        if i == start {
            return false;
        }
    }
    if matches!(b.get(i), Some(b'e' | b'E')) {
        i += 1;
        if matches!(b.get(i), Some(b'+' | b'-')) {
            i += 1;
        }
        let start = i;
        while b.get(i).is_some_and(u8::is_ascii_digit) {
            i += 1;
        }
        if i == start {
            return false;
        }
    }
    i == b.len()
}

#[cfg(test)]
mod character_format_tests;
