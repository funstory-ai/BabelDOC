//! Prepared-PDF parser. Reading uses lopdf; projection creates owned Rust IR directly.
// Keep leftovers in place: zero-operand operations never move the prefix.
fn operation_args(stack: &mut Vec<Token>, op: &str, operands: Vec<Token>) -> Result<Option<Vec<Token>>> {
    stack.extend(operands);
    let expected = match op {
        "q" | "Q" | "h" | "S" | "s" | "f" | "F" | "f*" | "B" | "B*" | "b" | "b*" | "n"
        | "W" | "W*" | "BT" | "ET" | "T*" => Some(0),
        "cm" | "Tm" | "c" => Some(6),
        "m" | "l" | "Td" | "TD" | "Tf" | "d" => Some(2),
        "re" | "v" | "y" => Some(4),
        "w" | "i" | "J" | "j" | "M" | "ri" | "gs" | "sh" | "CS" | "cs" | "Do" | "Tc"
        | "Tw" | "Tz" | "TL" | "Ts" | "Tr" | "Tj" | "TJ" | "'" | "INLINE_IMAGE" => Some(1),
        "\"" => Some(3),
        _ => None,
    };
    let Some(n) = expected else { return Ok(Some(std::mem::take(stack))); };
    if stack.len() < n {
        let got = stack.len();
        stack.clear();
        if ["m", "l", "c", "v", "y", "re"].contains(&op) { return Ok(None); }
        return Err(Error::new("ValueError", format!("{op} expected {n} operands, got {got}.")));
    }
    if stack.len() == n { return Ok(Some(std::mem::take(stack))); }
    Ok(Some(stack.split_off(stack.len() - n)))
}

#[cfg(test)]
#[allow(dead_code)]
#[path = "resource_v3_oracle/tokens_b20.rs"]
mod tokens_b20;
#[cfg(test)]
#[allow(dead_code)]
#[path = "resource_v3_oracle/page_access_b20.rs"]
mod page_access_b20;
#[cfg(test)]
#[allow(dead_code)]
#[path = "resource_v3_oracle/page_access_v2.rs"]
mod page_access_v2;
#[cfg(test)]
#[allow(dead_code)]
#[path = "resource_v3_oracle/type3_b20.rs"]
mod type3_b20;
#[cfg(test)]
mod resource_v3_tests;

pub mod fonts;
pub mod object_numbers;
mod page_access;
mod expansion;
mod pool;
mod projection;
mod tokens;
mod type3;
pub mod valid_text;
use babeldoc_ir_core as ir;
use ir::{Edge, Node, NodeData, PythonValue};
use lopdf::{Dictionary, Document as Pdf, Object, ObjectId};
pub use lopdf::resource::ParserDiagnostics;
use serde::Deserialize;
use std::collections::{HashMap, HashSet};
use std::path::Path;
use tokens::{Lexer, Token};
#[cfg(feature = "memory-profile")]
pub fn memory_checkpoint(phase: &str, details: serde_json::Value) {
    #[cfg(target_env = "gnu")]
    {
        #[repr(C)]
        struct AllocatorInfo {
            values: [usize; 10],
        }
        unsafe extern "C" {
            fn mallinfo2() -> AllocatorInfo;
        }
        // glibc's diagnostic API only reads allocator counters. Non-default
        // profiling feature; never changes allocator settings or parse output.
        let info = unsafe { mallinfo2() }.values;
        eprintln!(
            "ALLOC_PROFILE {}",
            serde_json::json!({"phase":phase,"arena":info[0],"mmap":info[4],"live_heap":info[7],"free_heap":info[8]})
        );
    }
    let status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    let mut values = serde_json::Map::new();
    for key in ["VmRSS:", "VmHWM:"] {
        if let Some(line) = status.lines().find(|line| line.starts_with(key)) {
            values.insert(
                key.trim_end_matches(':').into(),
                serde_json::json!(
                    line.split_whitespace()
                        .nth(1)
                        .unwrap()
                        .parse::<u64>()
                        .unwrap()
                        * 1024
                ),
            );
        }
    }
    eprintln!(
        "MEMORY_PROFILE {}",
        serde_json::json!({"phase":phase,"memory":values,"details":details})
    );
}
pub use lopdf::resource::{Policy, ResourceLimitExceeded, DISABLED_WARNING};
pub type Result<T> = std::result::Result<T, Error>;
#[derive(Debug)]
pub struct Error {
    pub resource: Option<Box<ResourceLimitExceeded>>,
    pub kind: String,
    pub message: String,
}
impl Error {
    fn at_page(mut self, page: usize) -> Self {
        if let Some(resource) = &mut self.resource { resource.page.get_or_insert(page); }
        self
    }
    pub fn new(kind: &str, message: impl Into<String>) -> Self {
        Self {
            resource: None,
            kind: kind.into(),
            message: message.into(),
        }
    }
}
impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.kind, self.message)
    }
}
impl std::error::Error for Error {}
impl From<fonts::FontError> for Error {
    fn from(e: fonts::FontError) -> Self {
        match e {
            fonts::FontError::Resource(e) => e.into(),
            other => Self::new(other.kind(), other.to_string()),
        }
    }
}
impl From<lopdf::Error> for Error {
    fn from(e: lopdf::Error) -> Self {
        if let Some(resource) = e.resource_limit() { return resource.clone().into(); }
        Self::new("PDFError", e.to_string())
    }
}
impl From<ResourceLimitExceeded> for Error {
    fn from(resource: ResourceLimitExceeded) -> Self {
        Box::new(resource).into()
    }
}
impl From<Box<ResourceLimitExceeded>> for Error {
    fn from(resource: Box<ResourceLimitExceeded>) -> Self {
        Self { kind: "ParserResourceLimitError".into(), message: resource.to_string(), resource: Some(resource) }
    }
}
pub fn catch_parser_unwind<T>(run: impl FnOnce() -> Result<T>) -> Result<T> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(run)).unwrap_or_else(|panic| {
        let text = panic
            .downcast_ref::<String>()
            .map(String::as_str)
            .or_else(|| panic.downcast_ref::<&str>().copied())
            .unwrap_or("Rust parser panicked");
        Err(Error::new("RuntimeError", text))
    })
}

#[derive(Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Config {
    pub disable_parser_resource_limits: bool,
    pub pages: Option<Vec<usize>>,
    pub enable_graphic_element_process: bool,
    pub ocr_workaround: bool,
    pub visual_roundtrip_mode: bool,
}
impl Default for Config {
    fn default() -> Self {
        Self {
            disable_parser_resource_limits: false,
            pages: None,
            enable_graphic_element_process: true,
            ocr_workaround: false,
            visual_roundtrip_mode: false,
        }
    }
}
pub type Matrix = [f64; 6];
pub const ID: Matrix = [1., 0., 0., 1., 0., 0.];
pub fn mul(l: Matrix, r: Matrix) -> Matrix {
    let [a1, b1, c1, d1, e1, f1] = l;
    let [a0, b0, c0, d0, e0, f0] = r;
    [
        a0 * a1 + c0 * b1,
        b0 * a1 + d0 * b1,
        a0 * c1 + c0 * d1,
        b0 * c1 + d0 * d1,
        a0 * e1 + c0 * f1 + e0,
        b0 * e1 + d0 * f1 + f0,
    ]
}
pub fn inv(m: Matrix) -> Matrix {
    let [a, b, c, d, e, f] = m;
    let det = a * d - b * c;
    if det.abs() < 1e-10 {
        return ID;
    }
    [
        d / det,
        -b / det,
        -c / det,
        a / det,
        (c * f - d * e) / det,
        (b * e - a * f) / det,
    ]
}
pub fn pt(m: Matrix, p: [f64; 2]) -> [f64; 2] {
    [
        m[0] * p[0] + m[2] * p[1] + m[4],
        m[1] * p[0] + m[3] * p[1] + m[5],
    ]
}
pub fn translate(m: Matrix, p: [f64; 2]) -> Matrix {
    [
        m[0],
        m[1],
        m[2],
        m[3],
        p[0] * m[0] + p[1] * m[2] + m[4],
        p[0] * m[1] + p[1] * m[3] + m[5],
    ]
}
pub fn edge(n: NodeData) -> Edge {
    Edge::Inline(Box::new(Node::new(n)))
}
fn font_edge_identity(edge: &Edge) -> usize {
    match edge {
        Edge::Shared(key) => *key,
        Edge::Inline(node) => std::ptr::from_ref(node.as_ref()) as usize,
        Edge::ImmutableBaseOperations(node) => std::sync::Arc::as_ptr(node) as usize,
        Edge::ImmutableBox(node) => std::sync::Arc::as_ptr(node) as usize,
        Edge::ImmutableGraphicState(node) => std::sync::Arc::as_ptr(node) as usize,
        Edge::ImmutablePdfAffineTransform(node) => std::sync::Arc::as_ptr(node) as usize,
        Edge::ImmutablePdfFontCharBoundingBox(node) => std::sync::Arc::as_ptr(node) as usize,
        Edge::ImmutablePdfInlineForm(node) => std::sync::Arc::as_ptr(node) as usize,
        Edge::ImmutablePdfMatrix(node) => std::sync::Arc::as_ptr(node) as usize,
        Edge::ImmutablePdfPath(node) => std::sync::Arc::as_ptr(node) as usize,
        Edge::ImmutablePdfXobjForm(node) => std::sync::Arc::as_ptr(node) as usize,
        Edge::ImmutableCropbox(node) => std::sync::Arc::as_ptr(node) as usize,
        Edge::ImmutableMediabox(node) => std::sync::Arc::as_ptr(node) as usize,
        Edge::ImmutablePageLayout(node) => std::sync::Arc::as_ptr(node) as usize,
        Edge::ImmutablePdfFigure(node) => std::sync::Arc::as_ptr(node) as usize,
        Edge::ImmutablePdfFont(node) => std::sync::Arc::as_ptr(node) as usize,
        Edge::ImmutablePdfFormSubtype(node) => std::sync::Arc::as_ptr(node) as usize,
        Edge::ImmutablePdfOriginalPath(node) => std::sync::Arc::as_ptr(node) as usize,
        Edge::ImmutablePdfOriginalPathPrimitive(node) => std::sync::Arc::as_ptr(node) as usize,
        Edge::ImmutablePdfRectangle(node) => std::sync::Arc::as_ptr(node) as usize,
        Edge::ImmutablePdfStyle(node) => std::sync::Arc::as_ptr(node) as usize,
        Edge::ImmutableVisualBbox(node) => std::sync::Arc::as_ptr(node) as usize,
        Edge::ImmutablePdfCharacter(node) => std::sync::Arc::as_ptr(node) as usize,
        Edge::ImmutablePdfCurve(node) => std::sync::Arc::as_ptr(node) as usize,
        Edge::ImmutablePdfForm(node) => std::sync::Arc::as_ptr(node) as usize,
        Edge::ImmutablePdfSameStyleUnicodeCharacters(node) => {
            std::sync::Arc::as_ptr(node) as usize
        }
        Edge::ImmutablePdfXobject(node) => std::sync::Arc::as_ptr(node) as usize,
        Edge::ImmutablePdfFormula(node) => std::sync::Arc::as_ptr(node) as usize,
        Edge::ImmutablePdfLine(node) => std::sync::Arc::as_ptr(node) as usize,
        Edge::ImmutablePdfSameStyleCharacters(node) => std::sync::Arc::as_ptr(node) as usize,
        Edge::ImmutablePdfParagraphComposition(node) => std::sync::Arc::as_ptr(node) as usize,
        Edge::ImmutablePdfParagraph(node) => std::sync::Arc::as_ptr(node) as usize,
        Edge::ImmutablePage(node) => std::sync::Arc::as_ptr(node) as usize,
        Edge::ImmutableDocument(node) => std::sync::Arc::as_ptr(node) as usize,
    }
}
pub fn bounds(points: impl Iterator<Item = [f64; 2]>) -> [f64; 4] {
    let mut b = [
        f64::INFINITY,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NEG_INFINITY,
    ];
    for p in points {
        b[0] = b[0].min(p[0]);
        b[1] = b[1].min(p[1]);
        b[2] = b[2].max(p[0]);
        b[3] = b[3].max(p[1])
    }
    if b[0] == f64::INFINITY { [0.; 4] } else { b }
}
pub fn box_edge(b: [f64; 4]) -> Edge {
    edge(NodeData::Box(ir::DataBox {
        x: Some(b[0]),
        y: Some(b[1]),
        x2: Some(b[2]),
        y2: Some(b[3]),
    }))
}
pub fn py_matrix(m: Matrix) -> Vec<PythonValue> {
    m.into_iter().map(PythonValue::Float).collect()
}
pub fn cm(m: Matrix) -> String {
    m.iter()
        .map(|x| format!("{x:.6}"))
        .collect::<Vec<_>>()
        .join(" ")
}
pub fn resolve<'a>(pdf: &'a Pdf, o: &'a Object) -> Result<&'a Object> {
    let mut o = o;
    let mut seen = HashSet::new();
    while let Object::Reference(id) = o {
        if !seen.insert(*id) {
            return Err(Error::new("PDFError", "Cyclic indirect reference"));
        }
        o = pdf.get_object(*id)?
    }
    Ok(o)
}
pub fn dict<'a>(pdf: &'a Pdf, o: &'a Object) -> Result<&'a Dictionary> {
    let o = resolve(pdf, o)?;
    match o {
        Object::Dictionary(d) => Ok(d),
        Object::Stream(s) => Ok(&s.dict),
        _ => Err(Error::new("PDFError", "Expected dictionary")),
    }
}
pub fn numeric(o: &Object) -> Option<f64> {
    match o {
        Object::Integer(n) => Some(*n as f64),
        Object::Real(x) => Some(object_numbers::printed_real(*x)),
        _ => None,
    }
}
pub fn array<const N: usize>(pdf: &Pdf, o: &Object) -> Option<[f64; N]> {
    let a = resolve(pdf, o).ok()?.as_array().ok()?;
    if a.len() < N {
        return None;
    }
    let mut b = [0.; N];
    for i in 0..N {
        b[i] = numeric(resolve(pdf, &a[i]).ok()?)?
    }
    Some(b)
}
fn object_truthy(o: &Object) -> bool {
    match o {
        Object::Null => false,
        Object::Boolean(b) => *b,
        Object::Integer(n) => *n != 0,
        Object::Real(n) => *n != 0.,
        Object::Array(a) => !a.is_empty(),
        Object::Dictionary(d) => !d.is_empty(),
        Object::String(s, _) => !s.is_empty(),
        _ => true,
    }
}

#[derive(Default)]
struct InheritedObjectCache {
    values: HashMap<(ObjectId, Vec<u8>), Option<std::sync::Arc<Object>>>,
    objects: HashMap<ObjectId, std::sync::Arc<Object>>,
    object_ids: HashMap<usize, ObjectId>,
}

fn inherited_cached(
    pdf: &Pdf,
    id: ObjectId,
    key: &[u8],
    cache: &mut InheritedObjectCache,
) -> Result<Option<std::sync::Arc<Object>>> {
    let mut id = id;
    let mut path = Vec::new();
    let mut seen = HashSet::new();
    let result = loop {
        if let Some(cached) = cache.values.get(&(id, key.to_vec())) {
            break cached.clone();
        }
        if !seen.insert(id) {
            break None;
        }
        path.push(id);
        let dictionary = pdf.get_dictionary(id)?;
        if let Ok(object) = dictionary.get(key)
            && !matches!(object, Object::Null)
        {
            break Some(match object {
                Object::Reference(object_id) => {
                    if let Some(object) = cache.objects.get(object_id) {
                        std::sync::Arc::clone(object)
                    } else {
                        let object = match pdf.get_object(*object_id) {
                            Ok(object) => std::sync::Arc::new(object.clone()),
                            Err(error) if error.is_resource_limit() => return Err(error.into()),
                            Err(_) => std::sync::Arc::new(Object::Reference(*object_id)),
                        };
                        cache
                            .objects
                            .insert(*object_id, std::sync::Arc::clone(&object));
                        cache.object_ids.insert(
                            std::sync::Arc::as_ptr(&object) as usize,
                            *object_id,
                        );
                        object
                    }
                }
                _ => std::sync::Arc::new(object.clone()),
            });
        }
        if let Ok(Object::Reference(parent)) = dictionary.get(b"Parent") {
            id = *parent;
        } else {
            break None;
        }
    };
    for node in path {
        cache.values.insert((node, key.to_vec()), result.clone());
    }
    Ok(result)
}

#[cfg(test)]
fn inherited<'a>(pdf: &'a Pdf, id: ObjectId, key: &[u8]) -> Result<Option<&'a Object>> {
    let mut id = id;
    let mut seen = HashSet::new();
    while seen.insert(id) {
        let d = pdf.get_dictionary(id)?;
        if let Ok(o) = d.get(key)
            && !matches!(o, Object::Null)
        {
            return Ok(Some(o));
        }
        if let Ok(Object::Reference(p)) = d.get(b"Parent") {
            id = *p
        } else {
            break;
        }
    }
    Ok(None)
}
fn streams(pdf: &Pdf, o: &Object, out: &mut Vec<u8>) -> Result<()> {
    if matches!(o, Object::Null) {
        return Ok(());
    }
    // Current pymupdf_page_view_access.read_many recursively flattens arrays.
    // An explicit stack retains that behavior without risking a process abort.
    let mut stack = vec![(o, false)];
    let mut active = HashSet::new();
    let mut expansion = expansion::Expansion::new(pdf.parser_policy, "contents");
    let mut decoded: HashMap<usize, std::ops::Range<usize>> = HashMap::new();
    let mut first = true;
    while let Some((value, leaving)) = stack.pop() {
        let source_id = if let Object::Reference(id) = value { Some(*id) } else { None };
        let indirect = source_id.is_some();
        let value = resolve(pdf, value).map_err(|error| {
            // MuPDF returns null for free/missing entries inside the xref table.
            // stream_value then raises TypeError; an out-of-table xref still
            // raises the object reader's RuntimeError.
            if matches!(value, Object::Reference(id) if id.0 <= pdf.max_id)
                && error.message.contains("not found")
            {
                Error::new(
                    "TypeError",
                    "Expected stream-like object, got <class 'NoneType'>",
                )
            } else {
                error
            }
        })?;
        let key = value as *const Object as usize;
        if leaving {
            active.remove(&key);
            continue;
        }
        match value {
            Object::Array(items) => {
                if !active.insert(key) {
                    return Err(Error::new(
                        "RecursionError",
                        "maximum recursion depth exceeded",
                    ));
                }
                expansion.structural_program(key, items.len(), source_id)?;
                expansion.work(1, source_id)?;
                stack.push((value, true));
                if !items.is_empty() {
                    stack.extend(items.iter().rev().map(|item| (item, false)));
                }
            }
            Object::Stream(s) => {
                if !first {
                    out.push(b'\n');
                }
                first = false;
                if let Some(range) = decoded.get(&key).cloned() {
                    expansion.enter_program(key, range.end - range.start, source_id)?;
                    out.extend_from_within(range);
                } else {
                    let bytes = pdf.decode_stream(s).map_err(|e| match source_id { Some(id) => e.at_object(id), None => e })?;
                    expansion.enter_program(key, bytes.len(), source_id)?;
                    let start = out.len();
                    out.extend_from_slice(&bytes);
                    decoded.insert(key, start..out.len());
                }
            }
            other => {
                let kind = match other {
                    Object::Null => "NoneType",
                    Object::Boolean(_) => "bool",
                    Object::Integer(_) => "int",
                    Object::Real(_) => "float",
                    Object::Name(_) => "str",
                    Object::String(..) => "bytes",
                    Object::Dictionary(_) if indirect => {
                        "babeldoc.format.pdf.new_parser.object_model.PdfObjectDict"
                    }
                    Object::Dictionary(_) => "dict",
                    _ => unreachable!(),
                };
                return Err(Error::new(
                    "TypeError",
                    format!("Expected stream-like object, got <class '{kind}'>"),
                ));
            }
        }
    }
    Ok(())
}
#[derive(Clone, Default)]
struct Snapshot {
    tail: Option<usize>,
    latest: Vec<(String, u64)>,
    next: u64,
    count: u64,
}
#[derive(Clone)]
struct Clip {
    path: std::sync::Arc<[Segment]>,
    ctm: Matrix,
    evenodd: bool,
}
struct ClipList {
    parent: Option<std::sync::Arc<ClipList>>,
    clip: std::sync::Arc<Clip>,
    formatted: std::sync::Mutex<Option<FormattedClip>>,
    #[cfg(test)]
    format_misses: std::sync::atomic::AtomicUsize,
}
struct FormattedClip {
    target_bits: [u64; 6],
    text: Option<String>,
}
#[derive(Clone)]
struct TextClip {
    ops: String,
    ctm: Matrix,
}
#[derive(Clone)]
struct Segment {
    op: String,
    coords: Vec<f64>,
}
#[derive(Clone)]
struct State {
    ctm: Matrix,
    font: Option<String>,
    size: f64,
    tc: f64,
    tw: f64,
    tz: f64,
    leading: f64,
    rise: f64,
    mode: i64,
    tm: Matrix,
    line: [f64; 2],
    in_text: bool,
    text_clip: Option<std::sync::Arc<TextClip>>,
}
impl Default for State {
    fn default() -> Self {
        Self {
            ctm: ID,
            font: None,
            size: 0.,
            tc: 0.,
            tw: 0.,
            tz: 100.,
            leading: 0.,
            rise: 0.,
            mode: 0,
            tm: ID,
            line: [0.; 2],
            in_text: false,
            text_clip: None,
        }
    }
}
#[derive(Clone)]
struct FontEntry {
    name: String,
    xref: i64,
    identity: i64,
    backend: std::sync::Arc<fonts::FontBackend>,
    legacy_descent: f64,
}
#[derive(Clone)]
struct ScopeFonts {
    entries: Vec<FontEntry>,
    exact: HashMap<String, usize>,
    resolved: HashMap<String, Option<usize>>,
}
struct Parser {
    pdf: Pdf,
    config: Config,
    table: ir::Table,
    history: ir::History,
    pool: pool::GraphicPool,
    curve_ctms: HashMap<[u64; 6], ir::CurveCtm>,
    font_cache: HashMap<ObjectId, std::sync::Arc<fonts::FontBackend>>,
    font_resources: fonts::FontResourceCache,
    diagnostics: ParserDiagnostics,
    fallback_fonts: HashMap<String, FontEntry>,
    next_direct_font: i64,
    inherited_objects: InheritedObjectCache,
    root_resources: std::sync::Arc<Object>,
    prepared_form_resources:
        std::sync::Arc<HashMap<ObjectId, std::sync::Arc<Object>>>,
    prepared_resource_graphs:
        HashMap<ObjectId, std::sync::Arc<HashMap<ObjectId, std::sync::Arc<Object>>>>,
    scope_font_cache: HashMap<Vec<String>, ScopeFonts>,
    emitted_fonts: HashSet<(i64, String)>,
    projected_descents: HashMap<(i64, String), f64>,
    box_cache: HashMap<(i64, String, u64), Vec<Edge>>,
    bbox_maps: Vec<projection::BboxMap>,
    bbox_map_cache: HashMap<(i64, String, u64), usize>,
    page_bbox_maps: HashMap<Option<i64>, usize>,
    legacy_cache: HashMap<(i64, String), f64>,
    font_names: Vec<Vec<String>>,
    available_fonts: Vec<(String, Edge)>,
    render: i64,
    xinc: i64,
    page: ir::DataPage,
    xobjects: Vec<ir::DataPdfXobject>,
    scope_path: Vec<String>,
    xobject_paths: Vec<Vec<String>>,
    end_operations: HashMap<Vec<String>, String>,
    valid_text: Vec<Vec<String>>,
}
pub struct Parsed {
    pub document: ir::Document,
    pub page_text: Vec<Vec<String>>,
    pub diagnostics: ParserDiagnostics,
    /// Parser-selected source-page objects in full-document ordinal order.
    ///
    /// Runtime writer metadata only; this is intentionally excluded from the
    /// IR document and artifact/hash.
    pub page_objects: Vec<ObjectId>,
    #[cfg(test)]
    pub(crate) resource_graph_cache_len: usize,
}
/// Serial page lifecycle shared by the harness and product adapter.
pub trait ParseObserver {
    fn resource_limits_disabled(&mut self) -> Result<()> {
        log::warn!(target: "babeldoc.parser", "{}", lopdf::resource::DISABLED_WARNING);
        Ok(())
    }

    fn start(&mut self, _selected_pages: usize) -> Result<()> {
        Ok(())
    }
    fn page_start(&mut self) -> Result<()> {
        Ok(())
    }
    fn page_end(&mut self, _text: &[String]) -> Result<()> {
        Ok(())
    }
    fn retain_page_text(&self) -> bool {
        true
    }
}
struct NoObserver;
impl ParseObserver for NoObserver {}

pub fn parse(path: impl AsRef<Path>, config: Config) -> Result<Parsed> {
    parse_observed(path, config, &mut NoObserver)
}

pub fn parse_observed(
    path: impl AsRef<Path>,
    config: Config,
    observer: &mut impl ParseObserver,
) -> Result<Parsed> {
    if config.disable_parser_resource_limits {
        observer.resource_limits_disabled()?;
    }
    let policy = if config.disable_parser_resource_limits { Policy::disabled() } else { Policy::default() };
    let pdf = Pdf::load_with_object_loader(
        path,
        lopdf::LoadOptions {
            real_parser: Some(object_numbers::lex_real),
            parser_policy: Some(policy),
            ..Default::default()
        },
        |reader| page_access::load_selected(reader, config.pages.as_deref()),
    )
    .map_err(|e| if e.is_resource_limit() { e.into() } else { Error::new("FileDataError", e.to_string()) })?;
    let diagnostics = pdf.parser_diagnostics;
    parse_loaded_with_diagnostics(pdf, config, observer, diagnostics)
}
#[cfg(test)]
fn parse_loaded(pdf: Pdf, config: Config, observer: &mut impl ParseObserver) -> Result<Parsed> {
    parse_loaded_with_diagnostics(pdf, config, observer, ParserDiagnostics::default())
}
fn parse_loaded_with_diagnostics(
    pdf: Pdf,
    config: Config,
    observer: &mut impl ParseObserver,
    diagnostics: ParserDiagnostics,
) -> Result<Parsed> {
    #[cfg(feature = "memory-profile")]
    {
        let mut slots = 0usize;
        let mut unused_slots = 0usize;
        let mut stream_bytes = 0usize;
        let mut stack: Vec<_> = pdf.objects.values().collect();
        while let Some(object) = stack.pop() {
            let dictionary = match object {
                Object::Array(values) => {
                    slots += values.capacity();
                    unused_slots += values.capacity() - values.len();
                    stack.extend(values);
                    None
                }
                Object::Dictionary(d) => Some(d),
                Object::Stream(s) => {
                    stream_bytes += s.content.capacity();
                    Some(&s.dict)
                }
                _ => None,
            };
            if let Some(d) = dictionary {
                slots += d.as_hashmap().capacity();
                unused_slots += d.as_hashmap().capacity() - d.len();
                stack.extend(d.iter().map(|(_, v)| v));
            }
        }
        memory_checkpoint(
            "loaded_pdf",
            serde_json::json!({"objects":pdf.objects.len(),"object_size":std::mem::size_of::<Object>(),"container_slots":slots,"unused_slots":unused_slots,"stream_bytes":stream_bytes}),
        );
    }
    let mut p = Parser {
        pdf,
        config,
        table: ir::Table::default(),
        history: ir::History::default(),
        pool: pool::GraphicPool::default(),
        curve_ctms: HashMap::new(),
        font_cache: HashMap::new(),
        font_resources: fonts::FontResourceCache::default(),
        diagnostics,
        fallback_fonts: HashMap::new(),
        next_direct_font: 0,
        inherited_objects: InheritedObjectCache::default(),
        root_resources: std::sync::Arc::new(Object::Null),
        prepared_form_resources: std::sync::Arc::new(HashMap::new()),
        prepared_resource_graphs: HashMap::new(),
        scope_font_cache: HashMap::new(),
        emitted_fonts: HashSet::new(),
        projected_descents: HashMap::new(),
        box_cache: HashMap::new(),
        bbox_maps: Vec::new(),
        bbox_map_cache: HashMap::new(),
        page_bbox_maps: HashMap::new(),
        legacy_cache: HashMap::new(),
        font_names: vec![],
        available_fonts: vec![],
        render: 0,
        xinc: 0,
        page: Default::default(),
        xobjects: vec![],
        scope_path: vec![],
        xobject_paths: vec![],
        end_operations: HashMap::new(),
        valid_text: vec![],
    };
    let result = p.run(observer);
    #[cfg(test)]
    let resource_graph_cache_len = p.prepared_resource_graphs.len();
    let diagnostics = p.diagnostics;
    drop(p);
    #[cfg(feature = "memory-profile")]
    memory_checkpoint("after_parser_drop", serde_json::json!({}));
    let (root, table, history, page_text, page_objects) = result?;
    let document = ir::Document::from_parts(root, table, history)
        .map_err(|e| Error::new("IRError", e.to_string()))?;
    #[cfg(feature = "memory-profile")]
    memory_checkpoint("after_attach", document.memory_profile());
    Ok(Parsed {
        document,
        page_text,
        diagnostics,
        page_objects,
        #[cfg(test)]
        resource_graph_cache_len,
    })
}
impl Parser {
    fn charge_clip_output(
        &self,
        clips: Option<&ClipList>,
        expansion: &mut expansion::Expansion,
        object: Option<ObjectId>,
    ) -> Result<()> {
        let mut current = clips;
        while let Some(node) = current {
            expansion.clip_node(
                std::ptr::from_ref(node) as usize,
                node.clip.path.len().saturating_add(1),
                object,
            )?;
            current = node.parent.as_deref();
        }
        Ok(())
    }
    fn order(&mut self) -> i64 {
        self.render += 1;
        self.render
    }
    fn base(&self, s: &str) -> Result<Edge> {
        let bytes = zstd::encode_all(s.as_bytes(), 0)
            .map_err(|e| Error::new("ZstdError", e.to_string()))?;
        let value = base85(&bytes);
        Ok(edge(NodeData::BaseOperations(ir::DataBaseOperations {
            value,
        })))
    }
    fn run(
        &mut self,
        observer: &mut impl ParseObserver,
    ) -> Result<(Node, ir::Table, ir::History, Vec<Vec<String>>, Vec<ObjectId>)> {
        let ids = page_access::page_ids(&self.pdf)?;
        let page_objects = ids.iter().map(|(_, id)| *id).collect();
        let mut views = Vec::new();
        for (one, id) in ids {
            let n = (one - 1) as usize;
            if self.config.pages.as_ref().is_some_and(|p| !p.contains(&n)) {
                continue;
            }
            if id.0 > self.pdf.max_id {
                return Err(Error::new("ValueError", "bad xref"));
            }
            let resources = inherited_cached(
                &self.pdf,
                id,
                b"Resources",
                &mut self.inherited_objects,
            )?
            .unwrap_or_else(|| {
                std::sync::Arc::new(Object::Dictionary(Dictionary::new()))
            });
            let raw =
                page_access::cropbox(&self.pdf, id, &mut self.inherited_objects)?;
            let rotate = inherited_cached(
                &self.pdf,
                id,
                b"Rotate",
                &mut self.inherited_objects,
            )?
                .and_then(|o| resolve(&self.pdf, o.as_ref()).ok().and_then(numeric))
                .unwrap_or(0.) as i64;
            let mut data = Vec::new();
            if let Ok(o) = self.pdf.get_dictionary(id)?.get(b"Contents") {
                streams(&self.pdf, o, &mut data).map_err(|e| e.at_page(n))?
            }
            views.push((n, raw, rotate, resources, data));
        }
        // load_page_views materializes every selected page before any backend
        // construction. Preserve its failure precedence on malformed documents.
        if views.is_empty() {
            return Err(Error::new("AssertionError", "total_pages > 0"));
        }
        observer.start(views.len())?;
        let mut pages = Vec::new();
        for (n, raw, rotate, resources, data) in views {
            observer.page_start()?;
            self.table.begin_page_scope();
            let crop = if !self
                .config
                .visual_roundtrip_mode
                && [90, 270].contains(&rotate.rem_euclid(360))
            {
                [raw[1], raw[2], raw[3], raw[0]]
            } else {
                raw
            };
            self.page = ir::DataPage {
                page_number: Some(n as i64),
                unit: Some("point".into()),
                cropbox: Some(edge(NodeData::Cropbox(ir::DataCropbox {
                    r#box: Some(box_edge(crop)),
                }))),
                mediabox: Some(edge(NodeData::Mediabox(ir::DataMediabox {
                    r#box: Some(box_edge(raw)),
                }))),
                ..Default::default()
            };
            self.xobjects.clear();
            self.scope_path.clear();
            self.xobject_paths.clear();
            self.end_operations.clear();
            self.legacy_cache.clear();
            self.emitted_fonts.clear();
            self.projected_descents.clear();
            self.page_bbox_maps.clear();
            self.font_names = vec![vec![]];
            self.valid_text.push(vec![]);
            self.root_resources = std::sync::Arc::clone(&resources);
            self.prepare_form_resources(&resources);
            self.scope_font_cache.clear();
            self.fallback_fonts.clear();
            let fs = self.fonts(resources.as_ref())?;
            self.install_fonts(&fs, None, false)?;
            let mut active = HashSet::new();
            let mut expansion = expansion::Expansion::new(self.pdf.parser_policy, "form");
            let inner = self.scope(
                &data,
                resources.as_ref(),
                fs,
                State::default(),
                0,
                None,
                &mut active,
                &mut expansion,
                0,
                None,
            ).map_err(|e| e.at_page(n))?;
            let wrapped = format!(
                "q {inner} Q {} cm",
                cm([1., 0., 0., 1., -crop[0], -crop[1]])
            );
            self.page.base_operations = Some(self.base(&wrapped)?);
            for i in 0..self.xobjects.len() {
                if let Some(text) = self.end_operations.get(&self.xobject_paths[i]) {
                    self.xobjects[i].base_operations = Some(self.base(text)?);
                }
            }
            self.page.pdf_xobject = std::mem::take(&mut self.xobjects)
                .into_iter()
                .map(|x| self.table.attach(Node::new(NodeData::PdfXobject(x))))
                .collect();
            pages.push(
                self.table
                    .attach(Node::new(NodeData::Page(std::mem::take(&mut self.page)))),
            );
            observer.page_end(self.valid_text.last().unwrap())?;
            self.table.finish_page_scope();
            if !observer.retain_page_text() {
                self.valid_text.pop();
            }
            #[cfg(feature = "memory-profile")]
            memory_checkpoint(
                "page_assembled",
                serde_json::json!({"page":n,"characters":self.valid_text.last().map(Vec::len),"table":self.table.memory_profile(None),"bbox_maps":self.bbox_maps.iter().map(|m|m.memory_profile()).collect::<Vec<_>>()}),
            );
        }
        if pages.is_empty() {
            return Err(Error::new("AssertionError", "total_pages > 0"));
        }
        let count = pages.len() as i64;
        let root = Node::new(NodeData::Document(ir::DataDocument {
            page: pages,
            total_pages: Some(count),
        }));
        #[cfg(feature = "memory-profile")]
        memory_checkpoint("before_attach", self.table.memory_profile(Some(&root)));
        #[cfg(feature = "memory-profile")]
        memory_checkpoint("history_before_attach", self.history.memory_profile());
        Ok((
            root,
            std::mem::take(&mut self.table),
            std::mem::take(&mut self.history),
            std::mem::take(&mut self.valid_text),
            page_objects,
        ))
    }

    fn history_push(&mut self, s: &mut Snapshot, op: &str, arg: ir::HistoryArg, replace: bool) {
        let version = if replace { s.next.max(1) } else { 0 };
        let key = replace.then(|| op.to_string());
        let idx = self.history.push(ir::HistoryNode {
            parent: s.tail,
            op: op.into(),
            arg,
            replace_key: key,
            version,
            length: s.count + 1,
        });
        s.tail = Some(idx);
        s.count += 1;
        if replace {
            s.latest.retain(|(k, _)| k != op);
            s.latest.push((op.into(), version));
            s.next = version + 1
        }
    }
    fn scope(
        &mut self,
        data: &[u8],
        resources: &Object,
        mut fonts: ScopeFonts,
        mut state: State,
        xid: i64,
        xslot: Option<usize>,
        active: &mut HashSet<ObjectId>,
        expansion: &mut expansion::Expansion,
        depth: usize,
        source: Option<(usize, ObjectId)>,
    ) -> Result<String> {
        // Validate the complete stream before execution, preserving tokenizer
        // failure precedence, without keeping every operation beside the IR.
        let inner = if depth == 0 {
            let inner = base_inner(data)?;
            expansion.enter_program(0, data.len(), None)?;
            inner
        } else {
            let key = source.map_or(0, |(key, _)| key);
            expansion.validate_entered_program(key, data)?;
            String::new() // Child return strings are discarded; only validation is observed.
        };
        let mut lexer = Lexer::new(data);
        let mut args_stack = Vec::new();
        let mut stack = Vec::new();
        let mut snap = Snapshot::default();
        let mut clips: Option<std::sync::Arc<ClipList>> = None;
        let mut path = Vec::<Segment>::new();
        let mut primitive = None;
        let mut pending = None;
        let mut buffered = Vec::new();
        let mut text_ops = Vec::new();
        let mut clip_text = false;
        let mut clip_ctm = ID;
        while let Some(oper) = lexer.operation()? {
            let op = oper.op.as_str();
            if state.in_text && op != "BT" {
                text_ops.push(serialize_op(op, &oper.args))
            }
            let Some(args) = operation_args(&mut args_stack, op, oper.args)? else {
                continue;
            };
            let nums = || args.iter().map(Token::number).collect::<Result<Vec<_>>>();
            match op {
                "q" => stack.push((state.clone(), snap.clone(), clips.clone())),
                "Q" => {
                    if let Some((st, ss, cc)) = stack.pop() {
                        state = st;
                        snap = ss;
                        clips = cc
                    } else {
                        snap = Snapshot::default();
                        clips = None
                    }
                }
                "cm" => state.ctm = mul(nums()?.try_into().unwrap(), state.ctm),
                "BT" => {
                    state.in_text = true;
                    state.tm = ID;
                    state.line = [0.; 2];
                    text_ops = vec!["BT".into()];
                    clip_text = false
                }
                "ET" => {
                    state.in_text = false;
                    state.tm = ID;
                    state.line = [0.; 2];
                    if clip_text {
                        state.text_clip = Some(std::sync::Arc::new(TextClip {
                            ops: text_ops.join(" "),
                            ctm: clip_ctm,
                        }))
                    }
                }
                "Tf" => {
                    state.font = Some(tokens::decode_name(args[0].name()?));
                    state.size = args[1].number()?
                }
                "Tc" => state.tc = args[0].number()?,
                "Tw" => state.tw = args[0].number()?,
                "Tz" => state.tz = args[0].number()?,
                "TL" => state.leading = -args[0].number()?,
                "Ts" => state.rise = args[0].number()?,
                "Tr" => state.mode = args[0].number()? as i64,
                "Tm" => {
                    state.tm = nums()?.try_into().unwrap();
                    state.line = [0.; 2]
                }
                "Td" | "TD" => {
                    let v = nums()?;
                    state.tm = translate(state.tm, [v[0], v[1]]);
                    state.line = [0.; 2];
                    if op == "TD" {
                        state.leading = v[1]
                    }
                }
                "T*" => {
                    state.tm = translate(state.tm, [0., state.leading]);
                    state.line = [0.; 2]
                }
                "Tj" | "TJ" | "'" | "\"" => {
                    if !state.in_text {
                        return Err(Error::new(
                            "ValueError",
                            format!("{op} used outside a text object."),
                        ));
                    }
                    if op == "'" || op == "\"" {
                        if op == "\"" {
                            state.tw = args[0].number()?;
                            state.tc = args[1].number()?
                        }
                        state.tm = translate(state.tm, [0., state.leading]);
                        state.line = [0.; 2]
                    }
                    let seq = if op == "TJ" {
                        if let Token::Array(a) = &args[0] {
                            a.clone()
                        } else {
                            return Err(Error::new(
                                "ValueError",
                                "TJ expects a single array operand.",
                            ));
                        }
                    } else {
                        vec![args.last().unwrap().clone()]
                    };
                    if state.mode == 7 {
                        clip_text = true;
                        clip_ctm = state.ctm
                    }
                    self.ensure_text_font(&state, &mut fonts)?;
                    self.text(
                        &state,
                        &seq,
                        &mut fonts,
                        xid,
                        &snap,
                        clips.as_deref(),
                        &mut buffered,
                    )?;
                    self.advance(&mut state, &seq, &mut fonts)?;
                }
                "w" | "i" | "J" | "j" | "M" | "ri" | "gs" | "cs" | "CS" | "g" | "G" | "rg"
                | "RG" | "k" | "K" | "sc" | "SC" | "scn" | "SCN" | "d" => {
                    let text = if op == "d" {
                        let a = if let Token::Array(v) = &args[0] {
                            format!(
                                "[{}]",
                                v.iter()
                                    .map(Token::python_str)
                                    .collect::<Vec<_>>()
                                    .join(" ")
                            )
                        } else {
                            args[0].pdf()
                        };
                        format!("{a} {}", args[1].python_str())
                    } else {
                        args.iter().map(Token::pdf).collect::<Vec<_>>().join(" ")
                    };
                    let arg = if op == "gs" {
                        ir::HistoryArg::CtmAware {
                            value: text,
                            ctm: PythonValue::Tuple(py_matrix(state.ctm)),
                        }
                    } else {
                        ir::HistoryArg::Text(text)
                    };
                    self.history_push(&mut snap, op, arg, op != "gs")
                }
                "m" | "l" | "c" | "v" | "y" | "re" => {
                    let Ok(v) = nums() else { continue };
                    if op == "re" {
                        primitive = if path.is_empty() {
                            Some(v.clone())
                        } else {
                            None
                        };
                        let (x, y, w, h) = (v[0], v[1], v[2], v[3]);
                        for (o, c) in [
                            ("m", vec![x, y]),
                            ("l", vec![x + w, y]),
                            ("l", vec![x + w, y + h]),
                            ("l", vec![x, y + h]),
                            ("h", vec![]),
                        ] {
                            path.push(Segment {
                                op: o.into(),
                                coords: c,
                            })
                        }
                    } else {
                        if op != "m" || !path.is_empty() {
                            primitive = None
                        }
                        path.push(Segment {
                            op: op.into(),
                            coords: v,
                        })
                    }
                }
                "h" => {
                    primitive = None;
                    path.push(Segment {
                        op: "h".into(),
                        coords: vec![],
                    })
                }
                "W" | "W*" => pending = Some(op == "W*"),
                "S" | "s" | "f" | "F" | "f*" | "B" | "B*" | "b" | "b*" | "n" => {
                    if ["s", "b", "b*"].contains(&op) {
                        path.push(Segment {
                            op: "h".into(),
                            coords: vec![],
                        })
                    }
                    if !path.is_empty() {
                        let path: std::sync::Arc<[Segment]> =
                            std::mem::take(&mut path).into();
                        if let Some(evenodd) = pending.take() {
                            clips = Some(std::sync::Arc::new(ClipList {
                                parent: clips,
                                clip: std::sync::Arc::new(Clip {
                                    path: std::sync::Arc::clone(&path),
                                    ctm: state.ctm,
                                    evenodd,
                                }),
                                formatted: std::sync::Mutex::new(None),
                                #[cfg(test)]
                                format_misses: std::sync::atomic::AtomicUsize::new(0),
                            }));
                        }
                        if op != "n" && path[0].op == "m" {
                            let order = self.order();
                            buffered.push(projection::Buffered::Curve {
                                path,
                                primitive: primitive.clone(),
                                ctm: state.ctm,
                                xid,
                                order,
                                snap: snap.clone(),
                                clips: clips.clone(),
                                stroke: ["S", "s", "B", "B*", "b", "b*"].contains(&op),
                                fill: !["S", "s"].contains(&op),
                                evenodd: op.ends_with('*'),
                            })
                        }
                    }
                    primitive = None
                }
                "sh" => {
                    let order = self.order();
                    buffered.push(projection::Buffered::Shading {
                        name: args[0].pdf(),
                        ctm: state.ctm,
                        xid,
                        order,
                        snap: snap.clone(),
                        clips: clips.clone(),
                    })
                }
                "Do" => {
                    let name = args[0].name()?.to_string();
                    let xo = dict(&self.pdf, resources)
                        .ok()
                        .and_then(|r| r.get(b"XObject").ok())
                        .and_then(|o| dict(&self.pdf, o).ok())
                        .and_then(|d| d.get(name.as_bytes()).ok())
                        .cloned();
                    if let Some(o) = xo {
                        let id = if let Object::Reference(i) = o {
                            Some(i)
                        } else {
                            None
                        };
                        let Ok(d) = dict(&self.pdf, &o) else {
                            continue;
                        };
                        let subtype = d
                            .get(b"Subtype")
                            .ok()
                            .and_then(|o| o.as_name().ok())
                            .map(|b| String::from_utf8_lossy(b).to_string())
                            .unwrap_or_default();
                        if subtype == "Image" {
                            self.charge_clip_output(clips.as_deref(), expansion, id)?;
                            if let Some(clip) = state.text_clip.as_deref() {
                                expansion.text_clip(
                                    std::ptr::from_ref(clip) as usize,
                                    clip.ops.len().saturating_add(1),
                                    id,
                                )?;
                            }
                            let extra = state.text_clip.as_deref().map(|clip| {
                                format!(
                                    "{} cm {} {} cm",
                                    cm(mul(clip.ctm, inv(state.ctm))),
                                    clip.ops,
                                    cm(mul(state.ctm, inv(clip.ctm)))
                                )
                            });
                            self.form(
                                state.ctm,
                                xid,
                                id.map(|x| x.0 as i64).unwrap_or(-1),
                                "image",
                                &name,
                                [0., 0., 1., 1.],
                                ID,
                                &snap,
                                clips.as_deref(),
                                extra,
                            )?
                        } else if subtype == "Form" {
                            if let Some(id) = id {
                                if depth >= 64 || !active.insert(id) {
                                    continue;
                                }
                                let target = d as *const Dictionary as usize;
                                let bbox = d
                                    .get(b"BBox")
                                    .ok()
                                    .and_then(|o| array(&self.pdf, o))
                                    .unwrap_or([0.; 4]);
                                let matrix = d
                                    .get(b"Matrix")
                                    .ok()
                                    .and_then(|o| array(&self.pdf, o))
                                    .unwrap_or(ID);
                                let res = self
                                    .prepared_form_resources
                                    .get(&id)
                                    .cloned()
                                    .unwrap_or_else(|| {
                                        std::sync::Arc::new(
                                            d.get(b"Resources")
                                                .ok()
                                                // Prepared resource building falls back for a
                                                // false direct dictionary as well as a missing key.
                                                .filter(|o| object_truthy(o))
                                                .cloned()
                                                .unwrap_or_else(|| resources.clone()),
                                        )
                                    });
                                let data = match resolve(&self.pdf, &o)? {
                                    Object::Stream(stream) => expansion.decode(&self.pdf, target, stream).map_err(|e| e.at_object(id))?,
                                    _ => std::rc::Rc::new(Vec::new()),
                                };
                                expansion.enter_program(target, data.len(), Some(id))?;
                                self.charge_clip_output(clips.as_deref(), expansion, Some(id))?;
                                for (_, edge) in &self.available_fonts {
                                    expansion.font_edge(font_edge_identity(edge), Some(id))?;
                                }
                                self.form(
                                    state.ctm,
                                    xid,
                                    id.0 as i64,
                                    "form",
                                    &name,
                                    bbox,
                                    matrix,
                                    &snap,
                                    clips.as_deref(),
                                    None,
                                )?;
                                self.xinc += 1;
                                let child_id = self.xinc;
                                let combined = mul(matrix, state.ctm);
                                let b = bounds(
                                    [
                                        [bbox[0], bbox[1]],
                                        [bbox[2], bbox[1]],
                                        [bbox[0], bbox[3]],
                                        [bbox[2], bbox[3]],
                                    ]
                                    .into_iter()
                                    .map(|p| pt(combined, p)),
                                );
                                let slot = self.xobjects.len();
                                let parent_available: Vec<_> = self
                                    .available_fonts
                                    .iter()
                                    .map(|(name, edge)| (name.clone(), edge.share()))
                                    .collect();
                                let mut obj = ir::DataPdfXobject {
                                    r#box: Some(box_edge(b)),
                                    xobj_id: Some(child_id),
                                    xref_id: Some(id.0 as i64),
                                    ..Default::default()
                                };
                                obj.pdf_font = self.inherited_font_edges(&fonts, xslot);
                                self.xobjects.push(obj);
                                self.scope_path.push(name.clone());
                                self.xobject_paths.push(self.scope_path.clone());
                                self.font_names.push(
                                    self.available_fonts
                                        .iter()
                                        .map(|(name, _)| name.clone())
                                        .collect(),
                                );
                                let child_fonts = self.fonts(&res)?;
                                self.install_fonts(&child_fonts, Some(slot), false)?;
                                self.scope(
                                    &data,
                                    &res,
                                    child_fonts,
                                    State {
                                        ctm: combined,
                                        ..Default::default()
                                    },
                                    child_id,
                                    Some(slot),
                                    active,
                                    expansion,
                                    depth + 1,
                                    Some((target, id)),
                                )?;
                                self.available_fonts = parent_available;
                                let [a, b, c, d, e, f] = combined;
                                let det = a * d - b * c;
                                let end = if det == 0. {
                                    " ".into()
                                } else {
                                    let ia = d / det;
                                    let ib = -b / det;
                                    let ic = -c / det;
                                    let id = a / det;
                                    format!(
                                        "{} cm ",
                                        cm([
                                            ia,
                                            ib,
                                            ic,
                                            id,
                                            -(e * ia + f * ic),
                                            -(e * ib + f * id)
                                        ])
                                    )
                                };
                                self.xobjects[slot].base_operations = Some(self.base(&end)?);
                                self.end_operations.insert(self.scope_path.clone(), end);
                                self.scope_path.pop();
                                self.xinc += 1;
                                active.remove(&id);
                            }
                        }
                    }
                }
                "INLINE_IMAGE" => {
                    if let Token::Inline(params, data) = &args[0] {
                        self.charge_clip_output(clips.as_deref(), expansion, None)?;
                        self.inline_image(
                            state.ctm,
                            xid,
                            &snap,
                            clips.as_deref(),
                            params,
                            data,
                        )?
                    }
                }
                "MP" | "DP" | "BMC" | "BDC" | "EMC" | "BX" | "EX" => {}
                _ => return Err(Error::new("UnsupportedOperatorError", op)),
            }
            if buffered.len() >= 2048 {
                self.flush(&mut buffered, expansion)?
            }
        }
        self.flush(&mut buffered, expansion)?;
        Ok(inner)
    }
}
pub fn python_float_str(value: f64) -> String {
    if value.is_nan() {
        return "nan".into();
    }
    if value == f64::INFINITY {
        return "inf".into();
    }
    if value == f64::NEG_INFINITY {
        return "-inf".into();
    }
    if value == 0. && value.is_sign_negative() {
        return "-0.0".into();
    }
    if value.abs() >= 1e16 || (value != 0. && value.abs() < 1e-4) {
        let text = format!("{value:e}");
        let (m, e) = text.split_once('e').unwrap();
        let e: i32 = e.parse().unwrap();
        return format!(
            "{m}e{}{e:02}",
            if e >= 0 { "+" } else { "-" },
            e = e.unsigned_abs()
        );
    }
    let text = value.to_string();
    if text.contains('.') || text.contains('e') {
        text
    } else {
        format!("{text}.0")
    }
}
fn serialize_op(op: &str, args: &[Token]) -> String {
    if args.is_empty() {
        op.into()
    } else {
        format!(
            "{} {op}",
            args.iter().map(Token::pdf).collect::<Vec<_>>().join(" ")
        )
    }
}
fn base_inner(data: &[u8]) -> Result<String> {
    let mut lexer = Lexer::new(data);
    let mut out = String::new();
    let mut first = true;
    while let Some(o) = lexer.operation()? {
        let op = o.op.as_str();
        if [
            "m", "l", "c", "v", "y", "re", "h", "S", "s", "f", "f*", "F", "B", "B*", "b", "b*",
            "n", "Do",
        ]
        .contains(&op)
            || op.starts_with('T')
            || ["\"", "'", "EI", "MP", "DP", "BMC", "BDC"].contains(&op)
        {
            continue;
        }
        if o.args.is_empty() && ["BI", "ID", "EMC", "INLINE_IMAGE"].contains(&op) {
            continue;
        }
        if op == "d" && o.args.len() != 2 {
            continue;
        }
        if !first {
            out.push(' ');
        }
        first = false;
        out.push_str(&serialize_op(op, &o.args).replace('\'', ""));
    }
    Ok(out)
}
fn base85(data: &[u8]) -> String {
    const A: &[u8] =
        b"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!#$%&()*+-;<=>?@^_`{|}~";
    let mut out = String::new();
    for ch in data.chunks(4) {
        let mut b = [0u8; 4];
        b[..ch.len()].copy_from_slice(ch);
        let mut n = u32::from_be_bytes(b) as u64;
        let mut s = [0u8; 5];
        for x in s.iter_mut().rev() {
            *x = A[(n % 85) as usize];
            n /= 85
        }
        out.push_str(std::str::from_utf8(&s[..ch.len() + 1]).unwrap())
    }
    out
}

#[cfg(test)]
mod panic_boundary_tests;

#[cfg(test)]
mod resource_decode_tests;
