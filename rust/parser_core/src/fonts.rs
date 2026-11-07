//! PDF font construction and metrics for the Rust parser.
//!
//! The public shape deliberately exposes both decoded PDF codes and the
//! projection-oriented data (bbox ordering and cmap key ordering).

use lopdf::{Dictionary, Document, Object, ObjectId, Stream};
use std::borrow::Cow;
use std::collections::{BTreeMap, HashMap};
use std::rc::Rc;
use std::sync::{Arc, LazyLock};

include!("font_cmap_data.rs");
mod maps;
#[cfg(test)]
mod maps_tests;
use lopdf::resource::Policy;
pub(crate) use maps::projection_codes;
use maps::*;

static FONT_METRICS: LazyLock<serde_json::Value> =
    LazyLock::new(|| serde_json::from_str(font_asset_json("fontmetrics.json")).unwrap_or_default());
static GLYPH_LIST: LazyLock<serde_json::Value> =
    LazyLock::new(|| serde_json::from_str(font_asset_json("glyphlist.json")).unwrap_or_default());
static LATIN_ENCODING: LazyLock<serde_json::Value> =
    LazyLock::new(|| serde_json::from_str(font_asset_json("latin_enc.json")).unwrap_or_default());
static TYPE1_ENCODINGS: LazyLock<serde_json::Value> = LazyLock::new(|| {
    serde_json::from_str(font_asset_json("type1_encoding.json")).unwrap_or_default()
});
static BASE14_TABLES: LazyLock<serde_json::Value> =
    LazyLock::new(|| serde_json::from_str(font_asset_json("base14.json")).unwrap_or_default());
static WIN_CORE: LazyLock<serde_json::Value> =
    LazyLock::new(|| serde_json::from_str(font_asset_json("win_core.json")).unwrap_or_default());

#[derive(Debug)]
pub enum FontError {
    Resource(Box<lopdf::resource::ResourceLimitExceeded>),
    Pdf(String),
    Unsupported(String),
    FreeType(String),
    Typed(&'static str, String),
}

impl std::fmt::Display for FontError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Resource(e) => e.fmt(f),
            Self::Pdf(s) | Self::Unsupported(s) | Self::FreeType(s) | Self::Typed(_, s) => {
                f.write_str(s)
            }
        }
    }
}
impl std::error::Error for FontError {}
impl FontError {
    pub fn kind(&self) -> &'static str {
        match self {
            Self::Resource(_) => "ParserResourceLimitError",
            Self::Typed(kind, _) => kind,
            Self::Pdf(_) => "PDFTypeError",
            Self::Unsupported(_) | Self::FreeType(_) => "FontError",
        }
    }
}

impl From<lopdf::Error> for FontError {
    fn from(e: lopdf::Error) -> Self {
        match e.resource_limit() {
            Some(resource) => Self::Resource(Box::new(resource.clone())),
            None => Self::Pdf(e.to_string()),
        }
    }
}
impl From<freetype::Error> for FontError {
    fn from(e: freetype::Error) -> Self {
        Self::FreeType(e.to_string())
    }
}
impl From<crate::Error> for FontError {
    fn from(e: crate::Error) -> Self {
        match e.resource {
            Some(resource) => Self::Resource(resource),
            None => Self::Pdf(e.message),
        }
    }
}
pub(crate) fn required<T>(o: Option<T>) -> Result<T, FontError> {
    o.ok_or_else(|| FontError::Pdf(String::new()))
}
pub(crate) fn optional<T>(r: Result<T, FontError>) -> Result<Option<T>, FontError> {
    match r {
        Ok(v) => Ok(Some(v)),
        Err(e @ FontError::Resource(_)) => Err(e),
        Err(_) => Ok(None),
    }
}
fn decode_optional(doc: &Document, stream: Option<&Stream>) -> Result<Option<Vec<u8>>, FontError> {
    match stream {
        Some(stream) => optional(doc.decode_stream(stream).map_err(Into::into)),
        None => Ok(None),
    }
}

#[derive(Clone)]
enum DecodedStreamCache {
    Decoded(Rc<Vec<u8>>),
    Ignored,
}

type CachedWidthValue = Result<f64, &'static str>;

#[derive(Default)]
pub(crate) struct FontResourceCache {
    decoded_streams: HashMap<ObjectId, DecodedStreamCache>,
    width_tables: HashMap<ObjectId, Arc<Vec<CachedWidthValue>>>,
    unicode_tables: HashMap<ObjectId, Option<Arc<ParsedUnicodeTable>>>,
    encoding_maps: HashMap<ObjectId, Option<Arc<EmbeddedCmap>>>,
    outlines: HashMap<OutlineKey, Option<Arc<(Vec<[f32; 2]>, f32)>>>,
    #[cfg(test)]
    pub(crate) decoded_stream_misses: usize,
    #[cfg(test)]
    pub(crate) width_table_misses: usize,
    #[cfg(test)]
    pub(crate) unicode_table_misses: usize,
    #[cfg(test)]
    pub(crate) encoding_map_misses: usize,
    #[cfg(test)]
    pub(crate) outline_misses: usize,
    #[cfg(test)]
    pub(crate) outline_cache_hits: usize,
}

#[derive(Clone, Default)]
struct ParsedUnicodeTable {
    unicode: Arc<BTreeMap<u32, String>>,
    ranges: Arc<Vec<(u32, u32, UnicodeTarget, usize)>>,
    order: Arc<BTreeMap<u32, usize>>,
    projection: Option<Vec<u32>>,
}

fn overlay_unicode_map(
    target: &mut Arc<BTreeMap<u32, String>>,
    source: &Arc<BTreeMap<u32, String>>,
) {
    if target.is_empty() {
        *target = Arc::clone(source);
    } else {
        Arc::make_mut(target).extend(source.iter().map(|(code, text)| (*code, text.clone())));
    }
}

fn overlay_unicode_ranges(
    target: &mut Arc<Vec<(u32, u32, UnicodeTarget, usize)>>,
    source: &Arc<Vec<(u32, u32, UnicodeTarget, usize)>>,
) {
    if target.is_empty() {
        *target = Arc::clone(source);
    } else {
        Arc::make_mut(target).extend(source.iter().cloned());
    }
}

fn overlay_unicode_order(
    target: &mut Arc<BTreeMap<u32, usize>>,
    source: &Arc<BTreeMap<u32, usize>>,
) {
    if target.is_empty() {
        *target = Arc::clone(source);
    } else {
        Arc::make_mut(target).extend(source.iter().map(|(code, order)| (*code, *order)));
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct OutlineKey {
    font: OutlineFontIdentity,
    code: u8,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum OutlineFontIdentity {
    Indirect(ObjectId),
    Direct(usize),
}

impl FontResourceCache {
    fn decode_font_file_object(
        &mut self,
        doc: &Document,
        object: &Object,
    ) -> Result<Option<Rc<Vec<u8>>>, FontError> {
        let Some(id) = object.as_reference().ok() else {
            return decode_optional(doc, object.as_stream().ok()).map(|bytes| bytes.map(Rc::new));
        };
        if let Some(cached) = self.decoded_streams.get(&id) {
            return Ok(match cached {
                DecodedStreamCache::Ignored => None,
                DecodedStreamCache::Decoded(bytes) => Some(Rc::clone(bytes)),
            });
        }
        #[cfg(test)]
        {
            self.decoded_stream_misses += 1;
        }
        let stream = doc.get_object(id).ok().and_then(|object| object.as_stream().ok());
        let decoded = match decode_optional(doc, stream) {
            Ok(Some(bytes)) => {
                let bytes = Rc::new(bytes);
                self.decoded_streams
                    .insert(id, DecodedStreamCache::Decoded(Rc::clone(&bytes)));
                Some(bytes)
            }
            Ok(None) => {
                self.decoded_streams.insert(id, DecodedStreamCache::Ignored);
                None
            }
            Err(error @ FontError::Resource(_)) => return Err(error),
            Err(_) => {
                self.decoded_streams.insert(id, DecodedStreamCache::Ignored);
                None
            }
        };
        Ok(decoded)
    }

    fn decode_projection_font_file_object(
        &self,
        doc: &Document,
        object: &Object,
    ) -> Result<Option<Rc<Vec<u8>>>, FontError> {
        if let Ok(id) = object.as_reference()
            && let Some(cached) = self.decoded_streams.get(&id)
        {
            return Ok(match cached {
                DecodedStreamCache::Ignored => None,
                DecodedStreamCache::Decoded(bytes) => Some(Rc::clone(bytes)),
            });
        }
        // Bounding-box projection uses the bytes only for this call. Share the
        // backend-retained program when it is the same object, but never make a
        // projection-only FontFile resident for the rest of the parse.
        self.decode_transient_stream_object(doc, object)
    }

    fn decode_transient_stream_object(
        &self,
        doc: &Document,
        object: &Object,
    ) -> Result<Option<Rc<Vec<u8>>>, FontError> {
        let stream = object
            .as_reference()
            .ok()
            .and_then(|id| doc.get_object(id).ok())
            .and_then(|object| object.as_stream().ok())
            .or_else(|| object.as_stream().ok());
        decode_optional(doc, stream).map(|bytes| bytes.map(Rc::new))
    }

    #[cfg(test)]
    pub(crate) fn retained_font_file_ids(&self) -> Vec<ObjectId> {
        let mut ids: Vec<_> = self.decoded_streams.keys().copied().collect();
        ids.sort_unstable();
        ids
    }

    fn width_table(
        &mut self,
        doc: &Document,
        object: Option<&Object>,
    ) -> Result<Arc<Vec<CachedWidthValue>>, FontError> {
        let Some(object) = object else {
            return Ok(Arc::new(Vec::new()));
        };
        let Some(id) = object.as_reference().ok() else {
            return parse_width_table(doc, object).map(Arc::new);
        };
        if let Some(values) = self.width_tables.get(&id) {
            return Ok(Arc::clone(values));
        }
        #[cfg(test)]
        {
            self.width_table_misses += 1;
        }
        let values = Arc::new(parse_width_table(doc, object)?);
        self.width_tables.insert(id, Arc::clone(&values));
        Ok(values)
    }

    fn encoding_map(
        &mut self,
        doc: &Document,
        object: &Object,
    ) -> Result<Option<Arc<EmbeddedCmap>>, FontError> {
        let id = object.as_reference().ok();
        if let Some(id) = id
            && let Some(map) = self.encoding_maps.get(&id)
        {
            return Ok(map.clone());
        }
        #[cfg(test)]
        {
            self.encoding_map_misses += 1;
        }
        let map = self
            .decode_transient_stream_object(doc, object)?
            .and_then(|bytes| EmbeddedCmap::parse_internal(&bytes))
            .map(Arc::new);
        if let Some(id) = id {
            self.encoding_maps.insert(id, map.clone());
        }
        Ok(map)
    }

    fn unicode_table(
        &mut self,
        doc: &Document,
        object: &Object,
    ) -> Result<Option<Arc<ParsedUnicodeTable>>, FontError> {
        let id = object.as_reference().ok();
        if let Some(id) = id
            && let Some(table) = self.unicode_tables.get(&id)
        {
            return Ok(table.clone());
        }
        #[cfg(test)]
        {
            self.unicode_table_misses += 1;
        }
        let Some(content) = self.decode_transient_stream_object(doc, object)? else {
            if let Some(id) = id {
                self.unicode_tables.insert(id, None);
            }
            return Ok(None);
        };
        let mut unicode = BTreeMap::new();
        let mut ranges = Vec::new();
        let mut order = BTreeMap::new();
        parse_tounicode_with_policy(
            &content,
            &mut unicode,
            &mut ranges,
            &mut order,
            doc.parser_policy.as_ref(),
        )?;
        let table = ParsedUnicodeTable {
            unicode: Arc::new(unicode),
            ranges: Arc::new(ranges),
            order: Arc::new(order),
            projection: parse_projection_intervals(&content, doc.parser_policy.as_ref())?,
        };
        let table = Arc::new(table);
        if let Some(id) = id {
            self.unicode_tables.insert(id, Some(Arc::clone(&table)));
        }
        Ok(Some(table))
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct FontStyleFlags {
    pub mono: bool,
    pub monospace: bool,
    pub serif: bool,
    pub bold: bool,
    pub italic: bool,
    pub embed: bool,
    pub never_embed: bool,
    pub has_opentype: bool,
}

// CharacterMap's custom stream decoder tries range widths longest first and
// falls back to the ordinary CMap unless every decoded CID is strictly positive.
// Keep numeric ranges instead of propagating every possible source code.
#[derive(Clone, Debug, Default)]
struct EmbeddedCmap {
    ranges: Vec<(usize, u32, u32, u32)>,
    lengths: Vec<usize>,
}
impl EmbeddedCmap {
    #[cfg(test)]
    fn parse(bytes: &[u8]) -> Option<Self> {
        Self::parse_internal(bytes)
    }
    fn parse_internal(bytes: &[u8]) -> Option<Self> {
        let Ok(text) = std::str::from_utf8(bytes) else {
            return None;
        };
        // Unknown usecmap resources make the reference constructor fail.
        if text.contains("usecmap") {
            return None;
        }
        let mut map = Self::default();
        for block in projection_cmap_blocks(text, "begincidrange", "endcidrange", true) {
            let tokens = cmap_tokens(block.as_bytes());
            let mut ranges = Vec::new();
            for group in tokens.chunks(3) {
                if group.len() != 3 {
                    ranges.clear();
                    break;
                }
                let Some((lo, width)) = parse_hex_token(&group[0]) else {
                    ranges.clear();
                    break;
                };
                let Some((hi, hi_width)) = parse_hex_token(&group[1]) else {
                    ranges.clear();
                    break;
                };
                let Ok(cid) = group[2].parse::<u32>() else {
                    ranges.clear();
                    break;
                };
                if width == hi_width {
                    ranges.push((width / 2, lo, hi, cid));
                }
            }
            map.ranges.extend(ranges);
        }
        // Discarded malformed blocks remain ordinary recovery.
        // Widths outside 1..=4 are never queried by the legacy decoder.
        map.ranges.retain(|r| (1..=4).contains(&r.0) && r.1 <= r.2);
        let mut compiled = Vec::new();
        let mut by_width: BTreeMap<usize, Vec<_>> = BTreeMap::new();
        for &range in &map.ranges {
            by_width.entry(range.0).or_default().push(range);
        }
        for (width, original) in by_width {
            for (lo, hi, index) in overlay(
                &original.iter().map(|r| (r.1, r.2)).collect::<Vec<_>>(),
                true,
            ) {
                let (_, base, _, cid) = original[index];
                let first = u64::from(cid) + u64::from(lo - base);
                if first > u32::MAX as u64 {
                    compiled.push((width, lo, hi, 0));
                } else {
                    let valid_hi = u64::from(lo) + u64::from(u32::MAX) - first;
                    let end = hi.min(valid_hi.min(u32::MAX as u64) as u32);
                    if first == 0 {
                        compiled.push((width, lo, lo, 0));
                        if lo < end {
                            compiled.push((width, lo + 1, end, 1));
                        }
                    } else {
                        compiled.push((width, lo, end, first as u32));
                    }
                    if valid_hi < u64::from(hi) {
                        compiled.push((width, (valid_hi + 1) as u32, hi, 0));
                    }
                }
            }
        }
        map.ranges = compiled;
        map.lengths = map.ranges.iter().map(|r| r.0).collect();
        map.lengths.sort_unstable_by(|a, b| b.cmp(a));
        map.lengths.dedup();
        Some(map)
    }
    fn decode(&self, bytes: &[u8]) -> Option<Vec<u32>> {
        let mut codes = Vec::new();
        let mut position = 0;
        while position < bytes.len() {
            let mut found = None;
            for &length in &self.lengths {
                if length == 0 || length > 4 || position + length > bytes.len() {
                    continue;
                }
                let code = bytes[position..position + length]
                    .iter()
                    .fold(0u32, |v, b| (v << 8) | u32::from(*b));
                let end = self
                    .ranges
                    .partition_point(|r| (r.0, r.1) <= (length, code));
                if let Some((_, lo, _, cid)) = end
                    .checked_sub(1)
                    .and_then(|i| self.ranges.get(i))
                    .filter(|r| r.0 == length && code <= r.2)
                {
                    found = if *cid == 0 {
                        None
                    } else {
                        cid.checked_add(code - lo).map(|cid| (cid, length))
                    };
                    break;
                }
            }
            let (cid, length) = found?;
            if cid == 0 {
                return None;
            }
            codes.push(cid);
            position += length;
        }
        Some(codes)
    }
}

#[derive(Clone, Debug)]
pub struct FontBackend {
    pub font_subtype: String,
    pub subtype: String,
    pub font_name: String,
    pub ascent: f64,
    pub descent: f64,
    pub default_width: f64,
    pub font_bbox: [f64; 4],
    pub vertical: bool,
    pub multibyte: bool,
    pub encoding_length: usize,
    pub identity_encoding: bool,
    pub identity_unicode: bool,
    pub style: Option<FontStyleFlags>,
    pub monospace: Option<bool>,
    pub legacy_descent: f64,
    pub type3_metrics: Option<crate::type3::Metrics>,
    /// Font program bytes after lopdf stream decompression, if embedded.
    pub embedded_font: Option<Rc<Vec<u8>>>,
    /// BBoxes in glyph-code order used by the IL font projection.
    pub projection_bboxes: Vec<[f64; 4]>,
    /// Packed inclusive [lo, hi] intervals in first-key encounter order.
    pub projection_cmap: Vec<u32>,
    widths: BTreeMap<u32, f64>,
    width_errors: BTreeMap<u32, &'static str>,
    displacement_errors: BTreeMap<u32, &'static str>,
    width_ranges: Vec<(u32, u32, f64)>,
    metric_widths: BTreeMap<String, f64>,
    unicode: Arc<BTreeMap<u32, String>>,
    unicode_ranges: Arc<Vec<(u32, u32, UnicodeTarget, usize)>>,
    unicode_order: Arc<BTreeMap<u32, usize>>,
    cmap: Arc<BTreeMap<Vec<u8>, u32>>,
    cid_encoding: Option<Arc<EmbeddedCmap>>,
    displacements: BTreeMap<u32, (Option<f64>, f64)>,
    displacement_ranges: Vec<(u32, u32, (Option<f64>, f64))>,
    default_displacement: (Option<f64>, f64),
    code_bytes: usize,
    width_scale: f64,
}

impl FontBackend {
    pub fn from_document(doc: &Document, font_id: ObjectId) -> Result<Self, FontError> {
        let obj = doc
            .get_object(font_id)
            .map_err(|e| FontError::Pdf(e.to_string()))?;
        Self::from_object(doc, obj)
    }

    pub fn from_object(doc: &Document, object: &Object) -> Result<Self, FontError> {
        let mut resources = FontResourceCache::default();
        Self::from_object_with_cache(doc, object, &mut resources)
    }

    pub(crate) fn from_object_with_cache(
        doc: &Document,
        object: &Object,
        resources: &mut FontResourceCache,
    ) -> Result<Self, FontError> {
        let font = resolve_dict(doc, object)?;
        let subtype = name(doc, font.get(b"Subtype").ok(), "")?;
        if subtype == "Type0" {
            if font.get(b"DescendantFonts").is_err() {
                return Err(FontError::Typed("KeyError", "DescendantFonts".into()));
            }
            let descendants = array(doc, font.get(b"DescendantFonts").ok())?;
            let desc = descendants
                .first()
                .ok_or_else(|| FontError::Typed("IndexError", "list index out of range".into()))?;
            let mut child = match resolve(doc, desc)? {
                Object::Dictionary(d) => d,
                _ => Dictionary::new(),
            };
            for key in [b"Encoding".as_slice(), b"ToUnicode".as_slice()] {
                if let Ok(v) = font.get(key) {
                    child.set(key, v.clone());
                }
            }
            let mut backend = Self::from_dict(doc, &child, true, resources)?;
            let (boxes, cmap) =
                optional(type0_projection(doc, &font, resources))?.unwrap_or_default();
            backend.projection_bboxes = boxes;
            backend.projection_cmap = cmap;
            return Ok(backend);
        }
        Self::from_dict(doc, &font, false, resources)
    }

    pub fn decode(&self, data: &[u8]) -> Vec<u32> {
        if !self.multibyte {
            return data.iter().map(|b| u32::from(*b)).collect();
        }
        if let Some(decoded) = self.cid_encoding.as_ref().and_then(|map| map.decode(data)) {
            return decoded;
        }
        if self.identity_encoding && self.code_bytes == 1 {
            return data.iter().map(|b| u32::from(*b)).collect();
        }
        if self.identity_encoding {
            return data
                .chunks_exact(2)
                .map(|c| u32::from(u16::from_be_bytes([c[0], c[1]])))
                .collect();
        }
        let mut out = Vec::new();
        let mut prefix = Vec::new();
        for byte in data {
            prefix.push(*byte);
            if let Some(cid) = self.cmap.get(&prefix) {
                out.push(*cid);
                prefix.clear();
            } else if !self
                .cmap
                .range(prefix.clone()..)
                .next()
                .is_some_and(|(key, _)| key.starts_with(&prefix))
            {
                prefix.clear();
            }
        }
        out
    }

    #[cfg(test)]
    pub fn unicode(&self, code: u32) -> Option<Cow<'_, str>> {
        self.unicode_with_policy(code, None).unwrap()
    }
    pub fn unicode_with_policy(
        &self,
        code: u32,
        policy: Option<&Policy>,
    ) -> Result<Option<Cow<'_, str>>, FontError> {
        unicode_map_lookup_with_policy(
            &self.unicode,
            &self.unicode_ranges,
            &self.unicode_order,
            code,
            policy,
        )
    }
    #[cfg(test)]
    pub fn unicode_text(&self, code: u32, fallback: &str) -> String {
        self.unicode_text_with_policy(code, fallback, None).unwrap()
    }
    pub fn unicode_text_with_policy(
        &self,
        code: u32,
        fallback: &str,
        policy: Option<&Policy>,
    ) -> Result<String, FontError> {
        Ok(self
            .unicode_with_policy(code, policy)?
            .map(Cow::into_owned)
            .or_else(|| {
                self.identity_unicode
                    .then(|| char::from_u32(code).map(|c| c.to_string()))
                    .flatten()
            })
            .unwrap_or_else(|| fallback.to_owned()))
    }
    /// Standalone numeric witness; the product parser passes its policy to width_with_policy.
    pub fn width(&self, code: u32) -> Result<f64, FontError> {
        self.width_with_policy(code, None)
    }
    pub fn width_with_policy(&self, code: u32, policy: Option<&Policy>) -> Result<f64, FontError> {
        if let Some(kind) = self.width_errors.get(&code) {
            return Err(FontError::Typed(kind, "invalid font width".into()));
        }
        let width = self
            .widths
            .get(&code)
            .copied()
            .or_else(|| lookup(&self.width_ranges, code, |r| (r.0, r.1)).map(|r| r.2));
        let width = match width {
            Some(width) => width,
            None => self
                .unicode_with_policy(code, policy)?
                .as_deref()
                .and_then(|text| self.metric_widths.get(text))
                .copied()
                .unwrap_or(self.default_width),
        };
        Ok(width * self.width_scale)
    }
    pub fn displacement(&self, code: u32) -> Result<(Option<f64>, f64), FontError> {
        if let Some(kind) = self.displacement_errors.get(&code) {
            return Err(FontError::Typed(kind, "invalid font displacement".into()));
        }
        Ok(self
            .displacements
            .get(&code)
            .copied()
            .or_else(|| lookup(&self.displacement_ranges, code, |r| (r.0, r.1)).map(|r| r.2))
            .unwrap_or(self.default_displacement))
    }
    pub fn is_multibyte(&self) -> bool {
        self.multibyte
    }
    pub fn is_vertical(&self) -> bool {
        self.vertical
    }

    fn from_dict(
        doc: &Document,
        spec: &Dictionary,
        was_type0: bool,
        resources: &mut FontResourceCache,
    ) -> Result<Self, FontError> {
        let subtype = name(doc, spec.get(b"Subtype").ok(), "Type1")?;
        let projection_descriptor = spec
            .get(b"FontDescriptor")
            .ok()
            .and_then(|v| resolve_dict(doc, v).ok())
            .unwrap_or_default();
        let descriptor = spec
            .get(b"FontDescriptor")
            .ok()
            .and_then(|v| resolve(doc, v).ok())
            .and_then(|v| {
                if let Object::Dictionary(d) = v {
                    Some(d)
                } else {
                    None
                }
            })
            .unwrap_or_default();
        let font_name = name(
            doc,
            descriptor
                .get(b"FontName")
                .ok()
                .or_else(|| spec.get(b"BaseFont").ok()),
            "unknown",
        )?;
        let mut out = Self {
            subtype: subtype.clone(),
            font_name: font_name.clone(),
            ascent: num(doc, descriptor.get(b"Ascent").ok(), 0.0)?,
            descent: num(doc, descriptor.get(b"Descent").ok(), 0.0)?,
            default_width: num(doc, descriptor.get(b"MissingWidth").ok(), 0.0)?,
            font_bbox: bbox(doc, descriptor.get(b"FontBBox").ok()),
            vertical: false,
            multibyte: matches!(subtype.as_str(), "CIDFontType0" | "CIDFontType2"),
            encoding_length: 1,
            identity_encoding: false,
            identity_unicode: false,
            font_subtype: if was_type0 {
                "Type0".into()
            } else {
                subtype.clone()
            },
            style: None,
            monospace: None,
            legacy_descent: num(doc, descriptor.get(b"Descent").ok(), 0.0)?,
            type3_metrics: None,
            embedded_font: None,
            projection_bboxes: Vec::new(),
            projection_cmap: Vec::new(),
            widths: BTreeMap::new(),
            width_errors: BTreeMap::new(),
            displacement_errors: BTreeMap::new(),
            width_ranges: Vec::new(),
            metric_widths: BTreeMap::new(),
            unicode: Arc::new(BTreeMap::new()),
            unicode_ranges: Arc::new(Vec::new()),
            unicode_order: Arc::new(BTreeMap::new()),
            cmap: Arc::new(BTreeMap::new()),
            cid_encoding: None,
            displacements: BTreeMap::new(),
            default_displacement: (None, 0.0),
            displacement_ranges: Vec::new(),
            code_bytes: 1,
            width_scale: 0.001,
        };
        if out.descent > 0.0 {
            out.descent = -out.descent;
        }
        let encname = encoding_name(doc, spec.get(b"Encoding").ok())?;
        let ordering = spec
            .get(b"CIDSystemInfo")
            .ok()
            .and_then(|o| resolve_dict(doc, o).ok())
            .and_then(|d| {
                name(doc, d.get(b"Registry").ok(), "")
                    .ok()
                    .zip(name(doc, d.get(b"Ordering").ok(), "").ok())
                    .map(|(r, o)| format!("{}-{}", r.trim(), o.trim()))
            })
            .unwrap_or_default();
        if subtype == "Type3" {
            let first = integer(doc, spec.get(b"FirstChar").ok(), 0)?;
            if spec.get(b"Widths").is_err() {
                store_default_widths(&mut out, first, 256);
            } else {
                let widths = resources.width_table(doc, spec.get(b"Widths").ok())?;
                apply_width_table(&mut out, first, &widths);
            }
            out.multibyte = false;
            let m = array(doc, spec.get(b"FontMatrix").ok())?;
            if m.len() != 6 {
                return Err(FontError::Typed(
                    "ValueError",
                    "not enough values to unpack FontMatrix".into(),
                ));
            }
            if m[..4]
                .iter()
                .any(|o| !matches!(o, Object::Integer(_) | Object::Real(_) | Object::Boolean(_)))
            {
                return Err(FontError::Typed(
                    "TypeError",
                    "non-numeric FontMatrix".into(),
                ));
            }
            out.width_scale = number(&m[0])? + number(&m[2])?;
            out.font_bbox = bbox(
                doc,
                if spec.get(b"FontDescriptor").is_ok() {
                    descriptor.get(b"FontBBox").ok()
                } else {
                    spec.get(b"FontBBox").ok()
                },
            );
            out.ascent = out.font_bbox[3];
            out.descent = out.font_bbox[1];
        } else if out.multibyte {
            out.vertical = encname.ends_with("-V") || encname == "Identity-V";
            out.encoding_length = if encname.starts_with("Identity-") {
                2
            } else {
                1
            };
            out.code_bytes = out.encoding_length;
            if out.vertical {
                let dw2 = array(doc, spec.get(b"DW2").ok())?;
                out.default_width = dw2.get(1).map(number).transpose()?.unwrap_or(-1000.0);
                out.default_displacement =
                    (None, dw2.first().map(number).transpose()?.unwrap_or(880.0));
                parse_cid_widths(doc, spec, &mut out, true)?;
            } else {
                out.default_width = num(doc, spec.get(b"DW").ok(), 1000.0)?;
                parse_cid_widths(doc, spec, &mut out, false)?;
            }
            if encname.starts_with("OneByteIdentity") {
                out.identity_encoding = true;
                out.code_bytes = 1;
            } else if encname.starts_with("Identity-") {
                out.encoding_length = 2;
                out.identity_encoding = out.multibyte;
            } else if let Some(table) = builtin_cmap(&encname) {
                out.vertical = table.vertical.unwrap_or(out.vertical);
                out.cmap = Arc::clone(&table.code2cid);
            }
        } else {
            let first = integer(doc, spec.get(b"FirstChar").ok(), 0)?;
            if spec.get(b"Widths").is_err() {
                store_default_widths(&mut out, first, 256);
            } else {
                let widths = resources.width_table(doc, spec.get(b"Widths").ok())?;
                apply_width_table(&mut out, first, &widths);
            }
        }
        if out.multibyte {
            out.cid_encoding = match spec.get(b"Encoding").ok() {
                Some(object) => resources.encoding_map(doc, object)?,
                None => None,
            };
        }
        let metrics_replaced = if !out.multibyte {
            populate_simple_tables(doc, spec, &mut out)?
        } else {
            false
        };
        if let Some(stream_obj) = projection_descriptor
            .get(b"FontFile")
            .ok()
            .or_else(|| projection_descriptor.get(b"FontFile2").ok())
            .or_else(|| projection_descriptor.get(b"FontFile3").ok())
        {
            out.embedded_font = resources.decode_font_file_object(doc, stream_obj)?;
        }
        // The active Python backend replaces its simple encoding map with the
        // Encoding vector embedded in a Type1 program when the PDF omits /Encoding.
        if !metrics_replaced && spec.get(b"Encoding").is_err() {
            if let Some(font_file_obj) = descriptor.get(b"FontFile").ok() {
                // fontfile["Length1"] raises KeyError when absent; the reference
                // catches it and retains its original simple encoding map.
                if let Some(length1_value) = stream_length1(doc, font_file_obj) {
                    if let Ok(length1) = integer(doc, Some(length1_value), 0) {
                        if let Some(data) =
                            resources.decode_projection_font_file_object(doc, font_file_obj)?
                        {
                            let n = (length1.max(0) as usize).min(data.len());
                            out.unicode = Arc::new(type1_header_encoding(&data[..n]));
                        }
                    }
                }
            }
        }
        if let Some(program) = out.embedded_font.as_deref() {
            if let Ok((style, glyph_boxes)) = freetype_details(program, "", false) {
                out.monospace = Some(style.mono);
                out.style = Some(style);
                out.projection_bboxes = glyph_boxes;
            }
            if !was_type0 && (subtype == "Type1" || subtype == "MMType1" || subtype == "TrueType") {
                // Projection visits every FontFile key in order; unlike the
                // backend/style program selection, its last present file wins.
                for key in [b"FontFile".as_slice(), b"FontFile2", b"FontFile3"] {
                    if let Ok(Object::Reference(id)) = projection_descriptor.get(key) {
                        out.projection_bboxes = resources
                            .decode_projection_font_file_object(
                                doc,
                                &Object::Reference((id.0, 0)),
                            )?
                            .map(|bytes| {
                                simple_font_boxes(&bytes, doc, spec, subtype == "TrueType")
                            })
                            .unwrap_or_default();
                    }
                }
            }
        } else if !out.multibyte {
            out.projection_bboxes = base14_boxes(doc, spec);
        }
        if out
            .embedded_font
            .as_ref()
            .is_none_or(|bytes| bytes.is_empty())
        {
            // PyMuPDF Font(fontbuffer=b"") follows JM_get_font's Noto fallback.
            out.style = Some(FontStyleFlags {
                mono: false,
                monospace: false,
                serif: true,
                bold: false,
                italic: false,
                embed: true,
                never_embed: false,
                has_opentype: false,
            });
            out.monospace = Some(false);
        }
        if subtype == "Type3" {
            out.type3_metrics = crate::type3::metrics(doc, spec);
            out.projection_bboxes = optional(crate::type3::boxes_with_resources(
                doc,
                spec,
                out.type3_metrics.as_ref(),
                resources,
            ))?
            .unwrap_or_default();
        }
        let to_unicode_object = spec.get(b"ToUnicode").ok();
        let to_unicode = to_unicode_object.and_then(|o| resolve(doc, o).ok());
        if matches!(to_unicode.as_ref(), Some(Object::Stream(_)))
            && let Some(object) = to_unicode_object
        {
            if let Some(table) = resources.unicode_table(doc, object)? {
                overlay_unicode_map(&mut out.unicode, &table.unicode);
                overlay_unicode_ranges(&mut out.unicode_ranges, &table.ranges);
                overlay_unicode_order(&mut out.unicode_order, &table.order);
                if let Some(keys) = &table.projection {
                    out.projection_cmap = keys.clone();
                } else {
                    out.projection_cmap.clear();
                    out.projection_bboxes.clear();
                }
            }
        }
        if out.multibyte && to_unicode.is_none() {
            if ordering == "Adobe-Identity" || ordering == "Adobe-UCS" {
                if let Some(stream_obj) = descriptor.get(b"FontFile2").ok() {
                    if let Some(bytes) =
                        resources.decode_projection_font_file_object(doc, stream_obj)?
                    {
                        out.unicode = Arc::new(truetype_gid_unicode(&bytes));
                    }
                }
            } else if let Some(table) = builtin_cmap(&format!("to-unicode-{ordering}")) {
                overlay_unicode_map(
                    &mut out.unicode,
                    if out.vertical {
                        &table.unicode_v
                    } else {
                        &table.unicode_h
                    },
                );
            }
        } else if out.multibyte && to_unicode.is_some() {
            // In the Python reference's non-strict mode, stream_value() turns a
            // non-stream ToUnicode value into an empty stream. That empty map
            // shadows the IdentityUnicodeMap name fallback.
            out.identity_unicode = false;
        }
        if out.multibyte
            && out.encoding_length == 1
            && (out.unicode.keys().any(|k| *k > 255)
                || out.unicode_ranges.iter().any(|(_, hi, _, _)| *hi > 255))
        {
            out.encoding_length = 2;
        }
        out.legacy_descent = out.descent;
        Ok(out)
    }
}

fn encoding_name(doc: &Document, value: Option<&Object>) -> Result<String, FontError> {
    let Some(value) = value else {
        return Ok(String::new());
    };
    match resolve(doc, value)? {
        Object::Name(n) => Ok(String::from_utf8_lossy(&n).into_owned()),
        Object::Dictionary(d) => {
            name(doc, d.get(b"CMapName").ok(), "").map(|n| normalize_cmap_name(&n))
        }
        Object::Stream(s) => {
            name(doc, s.dict.get(b"CMapName").ok(), "").map(|n| normalize_cmap_name(&n))
        }
        _ => Ok(String::new()),
    }
}
fn normalize_cmap_name(name: &str) -> String {
    match name {
        "DLIdent-H" => "Identity-H".into(),
        "DLIdent-V" => "Identity-V".into(),
        _ => name.to_owned(),
    }
}

fn truetype_gid_unicode(bytes: &[u8]) -> BTreeMap<u32, String> {
    // TrueTypeFont reads the SFNT directory from BytesIO, then passes the same
    // unrewound file object to freetype.Face. Its read() consumes only this
    // suffix. Reproduce the current reference bug, including CMapNotFound.
    let Some(header) = bytes.get(4..12) else {
        return BTreeMap::new();
    };
    let count = u16::from_be_bytes([header[0], header[1]]) as usize;
    let mut position = 12usize;
    let mut cmap = false;
    for _ in 0..count {
        let end = (position + 16).min(bytes.len());
        let entry = &bytes[position..end];
        position = end;
        if entry.len() != 16 {
            break;
        }
        cmap |= &entry[..4] == b"cmap";
    }
    if !cmap {
        return BTreeMap::new();
    }
    let bytes = &bytes[position..];
    let Ok(lib) = freetype::Library::init() else {
        return BTreeMap::new();
    };
    let Ok(mut face) = lib.new_memory_face(bytes.to_vec(), 0) else {
        return BTreeMap::new();
    };
    let mut result = BTreeMap::new();
    let mut gid = 0u32;
    let mut code = unsafe { freetype::ffi::FT_Get_First_Char(face.raw_mut(), &mut gid) };
    while gid != 0 {
        if let Some(ch) = char::from_u32(code as u32) {
            result.insert(gid, ch.to_string());
        }
        code = unsafe { freetype::ffi::FT_Get_Next_Char(face.raw_mut(), code, &mut gid) };
    }
    result
}

fn populate_simple_tables(
    doc: &Document,
    spec: &Dictionary,
    out: &mut FontBackend,
) -> Result<bool, FontError> {
    let mut metrics_replaced = false;
    if out.subtype != "Type3" {
        {
            if let Some(entry) = FONT_METRICS.get(&out.font_name).and_then(|v| v.as_array()) {
                metrics_replaced = true;
                out.widths.clear();
                out.width_errors.clear();
                if let Some(props) = entry.first().and_then(|v| v.as_object()) {
                    if let Some(name) = props.get("FontName").and_then(|v| v.as_str()) {
                        out.font_name = name.to_owned();
                    }
                    out.ascent = props.get("Ascent").and_then(|v| v.as_f64()).unwrap_or(0.);
                    out.descent = props.get("Descent").and_then(|v| v.as_f64()).unwrap_or(0.);
                    if out.descent > 0. {
                        out.descent = -out.descent;
                    }
                    out.default_width = props
                        .get("MissingWidth")
                        .and_then(|v| v.as_f64())
                        .unwrap_or(0.);
                    out.font_bbox = props
                        .get("FontBBox")
                        .and_then(|v| v.as_array())
                        .filter(|v| v.len() == 4)
                        .map(|v| {
                            [
                                v[0].as_f64().unwrap_or(0.),
                                v[1].as_f64().unwrap_or(0.),
                                v[2].as_f64().unwrap_or(0.),
                                v[3].as_f64().unwrap_or(0.),
                            ]
                        })
                        .unwrap_or([0.; 4]);
                }
                if let Some(widths) = entry.get(1).and_then(|v| v.as_object()) {
                    for (k, v) in widths {
                        if let Some(w) = v.as_f64() {
                            out.metric_widths.insert(k.clone(), w);
                        }
                    }
                }
            }
        }
    }
    let encoding = spec
        .get(b"Encoding")
        .ok()
        .map(|v| resolve(doc, v))
        .transpose()?;
    let (base_name, differences) = match encoding.as_ref() {
        Some(Object::Dictionary(d)) => (
            name(doc, d.get(b"BaseEncoding").ok(), "StandardEncoding")?,
            array(doc, d.get(b"Differences").ok())?,
        ),
        Some(Object::Name(n)) => (String::from_utf8_lossy(n).into_owned(), Vec::new()),
        _ => ("StandardEncoding".into(), Vec::new()),
    };
    let column = match base_name.as_str() {
        "MacRomanEncoding" => 2,
        "WinAnsiEncoding" => 3,
        "PDFDocEncoding" => 4,
        _ => 1,
    };
    let mut unicode = BTreeMap::<u32, String>::new();
    if let Some(rows) = LATIN_ENCODING.as_array() {
        for row in rows {
            let Some(values) = row.as_array() else {
                continue;
            };
            if values.len() < 5 {
                continue;
            }
            let glyph = values[0].as_str().unwrap_or("");
            if let Some(cid) = values[column].as_u64() {
                if let Some(text) = GLYPH_LIST.get(glyph).and_then(|v| v.as_str()) {
                    unicode.insert(cid as u32, text.to_owned());
                }
            }
        }
    }
    let mut cid = 0i128;
    for item in differences {
        if let Object::Integer(n) = item {
            cid = n as i128;
        } else {
            let glyph = name(doc, Some(&item), "")?;
            if let Some(text) = glyph_name_unicode(&glyph) {
                if let Ok(code) = u32::try_from(cid) {
                    unicode.insert(code, text);
                }
            }
            cid += 1;
        }
    }
    out.unicode = Arc::new(unicode);
    Ok(metrics_replaced)
}

fn stream_length1<'a>(doc: &'a Document, font_file: &'a Object) -> Option<&'a Object> {
    let stream = resolve_stream(doc, font_file).ok()?;
    stream.dict.get(b"Length1").ok()
}

fn type1_header_encoding(data: &[u8]) -> BTreeMap<u32, String> {
    #[derive(Clone)]
    enum Token {
        Integer(u32),
        Name(String),
        Other,
    }
    let mut tokens = Vec::new();
    let mut result = BTreeMap::new();
    let mut i = 0;
    while i < data.len() {
        let b = data[i];
        if b.is_ascii_whitespace() {
            i += 1;
            continue;
        }
        if b == b'%' {
            while i < data.len() && data[i] != b'\n' && data[i] != b'\r' {
                i += 1
            }
            continue;
        }
        if b == b'/' {
            i += 1;
            let start = i;
            while i < data.len() && !is_pdf_delimiter(data[i]) {
                i += 1
            }
            tokens.push(Token::Name(
                String::from_utf8_lossy(&data[start..i]).into_owned(),
            ));
            continue;
        }
        if b.is_ascii_digit()
            || ((b == b'-' || b == b'+') && data.get(i + 1).is_some_and(u8::is_ascii_digit))
        {
            let start = i;
            i += 1;
            while i < data.len() && data[i].is_ascii_digit() {
                i += 1
            }
            if let Ok(n) = std::str::from_utf8(&data[start..i])
                .unwrap_or("")
                .parse::<i64>()
            {
                tokens.push(if n >= 0 && n <= u32::MAX as i64 {
                    Token::Integer(n as u32)
                } else {
                    Token::Other
                })
            } else {
                tokens.push(Token::Other)
            }
            continue;
        }
        if b.is_ascii_alphabetic() {
            let start = i;
            i += 1;
            while i < data.len() && (data[i].is_ascii_alphanumeric() || b"._-".contains(&data[i])) {
                i += 1
            }
            if &data[start..i] == b"put" {
                if tokens.len() >= 2 {
                    let value = tokens.pop().unwrap();
                    let key = tokens.pop().unwrap();
                    if let (Token::Integer(code), Token::Name(name)) = (key, value) {
                        if let Some(unicode) = glyph_name_unicode(&name) {
                            result.insert(code, unicode);
                        }
                    }
                }
            } else {
                tokens.push(Token::Other)
            }
            continue;
        }
        i += 1;
    }
    result
}

fn is_pdf_delimiter(b: u8) -> bool {
    b.is_ascii_whitespace() || b"()<>[]{}/%".contains(&b)
}

fn glyph_name_unicode(name: &str) -> Option<String> {
    let name = name.split('.').next()?;
    if name.contains('_') {
        let mut out = String::new();
        for part in name.split('_') {
            out.push_str(&glyph_name_unicode(part)?)
        }
        return Some(out);
    }
    if let Some(value) = GLYPH_LIST.get(name).and_then(|v| v.as_str()) {
        return Some(value.to_owned());
    }
    if let Some(hex) = name.strip_prefix("uni") {
        if !hex.is_empty() && hex.len() % 4 == 0 && hex.bytes().all(|b| b.is_ascii_hexdigit()) {
            let mut out = String::new();
            for part in hex.as_bytes().chunks_exact(4) {
                let code = u16::from_str_radix(std::str::from_utf8(part).ok()?, 16).ok()?;
                out.push(char::from_u32(code as u32)?)
            }
            return Some(out);
        }
    }
    if let Some(hex) = name.strip_prefix('u') {
        if (4..=6).contains(&hex.len()) && hex.bytes().all(|b| b.is_ascii_hexdigit()) {
            return char::from_u32(u32::from_str_radix(hex, 16).ok()?).map(|c| c.to_string());
        }
    }
    None
}

fn simple_font_boxes(
    bytes: &[u8],
    doc: &Document,
    spec: &Dictionary,
    _is_truetype: bool,
) -> Vec<[f64; 4]> {
    let encoding = spec
        .get(b"Encoding")
        .ok()
        .and_then(|v| resolve(doc, v).ok());
    let (enc_name, differences) = match encoding.as_ref() {
        Some(Object::Dictionary(d)) => {
            let base = d
                .get(b"BaseEncoding")
                .ok()
                .and_then(|v| resolve(doc, v).ok());
            let enc = match base.as_ref() {
                Some(Object::Name(n)) => String::from_utf8_lossy(n).into_owned(),
                _ => "Custom".into(),
            };
            (
                enc,
                match d.get(b"Differences").ok() {
                    Some(Object::Array(values)) => values.clone(),
                    _ => Vec::new(),
                },
            )
        }
        Some(Object::Name(n)) => (String::from_utf8_lossy(n).into_owned(), Vec::new()),
        _ => ("Custom".into(), Vec::new()),
    };
    let encodings = &*TYPE1_ENCODINGS;
    let (enc_key, arr_key) = match enc_name.as_str() {
        "WinAnsiEncoding" => ("win_ansi", "WinAnsiEncoding"),
        "MacRomanEncoding" => ("mac_roman", "WinAnsiEncoding"),
        "MacExpertEncoding" => ("mac_expert", "WinAnsiEncoding"),
        _ => ("adobe_standard", "WinAnsiEncoding"),
    };
    let default = encodings
        .get(enc_key)
        .or_else(|| encodings.get("adobe_standard"));
    let mut names: Vec<Option<String>> = default
        .and_then(|v| v.as_array())
        .map(|a| a.iter().map(|v| v.as_str().map(str::to_owned)).collect())
        .unwrap_or_else(|| vec![None; 256]);
    if names.len() < 256 {
        names.resize(256, None)
    }
    let mut cid = 0i128;
    for item in differences {
        if let Object::Integer(n) = item {
            cid = n as i128;
        } else {
            if let Ok(index) = usize::try_from(cid) {
                if index < names.len() {
                    names[index] = Some(name(doc, Some(&item), "").unwrap_or_default());
                }
            }
            cid += 1;
        }
    }
    let lib = match freetype::Library::init() {
        Ok(l) => l,
        Err(_) => return Vec::new(),
    };
    let face = match lib.new_memory_face(bytes.to_vec(), 0) {
        Ok(f) => f,
        Err(_) => return Vec::new(),
    };
    let face_is_truetype = face_is_truetype(&face);
    set_bbox_charmap(&face, face_is_truetype, &enc_name);
    let units = (face.raw().units_per_EM.max(1)) as f64;
    let scale = 1000.0 / units;
    if face_is_truetype && enc_name == "WinAnsiEncoding" {
        let codes = encodings
            .get(arr_key)
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default();
        return codes
            .iter()
            .map(|c| {
                c.as_u64()
                    .map(|c| glyph_bbox(&face, face.get_char_index(c as usize).unwrap_or(0), scale))
                    .unwrap_or([0.; 4])
            })
            .collect();
    }
    if face_is_truetype && enc_name == "Custom" {
        return (0..256)
            .map(|code| glyph_bbox(&face, face.get_char_index(code).unwrap_or(0), scale))
            .collect();
    }
    let Some(glyph_name_set) = face_glyph_names(&face) else {
        // freetype-py raises on the first failed glyph-name query; the outer
        // projection catches it and abandons the complete font bbox map.
        return Vec::new();
    };
    let mut boxes = Vec::with_capacity(256);
    for (code, glyph) in names.iter().enumerate() {
        let bbox = if let Some(g) = glyph
            .as_ref()
            .filter(|g| glyph_name_set.contains(g.as_str()))
        {
            face_name_index(&face, g)
                .map(|gid| glyph_bbox(&face, gid, scale))
                .unwrap_or([0.; 4])
        } else {
            glyph_bbox(&face, face.get_char_index(code).unwrap_or(0), scale)
        };
        boxes.push(bbox);
    }
    for (code, name) in parse_differences(spec, doc) {
        if code < boxes.len() {
            if let Some(gid) = face_name_index(&face, &name) {
                boxes[code] = glyph_bbox(&face, gid, scale)
            }
        }
    }
    boxes
}

/// FreeType's name lookup returns glyph index zero for `.notdef` and for an
/// unresolved name. PyMuPDF keeps that value; freetype-rs maps zero to `None`.
fn face_name_index(face: &freetype::Face, name: &str) -> Option<u32> {
    use std::ffi::CString;
    let name = CString::new(name).ok()?;
    let raw = face.raw() as *const _ as freetype::ffi::FT_Face;
    if raw.is_null() {
        return None;
    }
    Some(unsafe { freetype::ffi::FT_Get_Name_Index(raw, name.as_ptr().cast()) })
}

/// MuPDF's SVG text outlines load at 65536 units, then divide the resulting
/// hinted outline by 65536 before serializing its path and transform separately.
#[cfg(test)]
pub(crate) fn svg_glyph_outline(
    doc: &Document,
    object: &Object,
    code: u8,
) -> Result<Option<(Vec<[f32; 2]>, f32)>, FontError> {
    let mut resources = FontResourceCache::default();
    Ok(svg_glyph_outline_with_cache(doc, object, code, &mut resources)?
        .map(|value| value.as_ref().clone()))
}

pub(crate) fn svg_glyph_outline_with_cache(
    doc: &Document,
    object: &Object,
    code: u8,
    resources: &mut FontResourceCache,
) -> Result<Option<Arc<(Vec<[f32; 2]>, f32)>>, FontError> {
    let key = OutlineKey {
        font: outline_font_identity(doc, object)?,
        code,
    };
    if let Some(value) = resources.outlines.get(&key) {
        #[cfg(test)]
        {
            resources.outline_cache_hits += 1;
        }
        return Ok(value.clone());
    }
    #[cfg(test)]
    {
        resources.outline_misses += 1;
    }
    let computed = (|| {
        let spec = resolve_dict(doc, object)?;
        let descriptor = resolve_dict(doc, spec.get(b"FontDescriptor")?)?;
        let mut program = None;
        for file_key in [b"FontFile".as_slice(), b"FontFile2", b"FontFile3"] {
            let Some(file_object) = descriptor.get(file_key).ok() else {
                continue;
            };
            if let Some(bytes) = resources.decode_font_file_object(doc, file_object)? {
                program = Some(bytes);
                break;
            }
        }
        let program = required(program)?;
        let glyph = parse_differences(&spec, doc)
            .into_iter()
            .find(|(value, _)| *value == usize::from(code))
            .map(|(_, name)| name);
        svg_glyph_outline_inner(
            doc,
            &spec,
            &descriptor,
            program,
            glyph.as_deref(),
            code,
        )
    })();
    let value = optional(computed)?.map(Arc::new);
    resources.outlines.insert(key, value.clone());
    Ok(value)
}

fn outline_font_identity(
    doc: &Document,
    object: &Object,
) -> Result<OutlineFontIdentity, FontError> {
    let Object::Reference(reference) = object else {
        return Ok(OutlineFontIdentity::Direct(object as *const Object as usize));
    };
    let initial = *reference;
    let mut id = initial;
    let mut seen = std::collections::HashSet::new();
    while seen.insert(id) {
        match doc.objects.get(&id) {
            Some(Object::Reference(next)) => id = *next,
            Some(_) => return Ok(OutlineFontIdentity::Indirect(id)),
            None => return Ok(OutlineFontIdentity::Indirect(initial)),
        }
    }
    Ok(OutlineFontIdentity::Indirect(initial))
}

fn svg_glyph_outline_inner(
    doc: &Document,
    spec: &Dictionary,
    descriptor: &Dictionary,
    program: Rc<Vec<u8>>,
    glyph: Option<&str>,
    code: u8,
) -> Result<(Vec<[f32; 2]>, f32), FontError> {
    use freetype::outline::Curve;
    let library = freetype::Library::init()?;
    let face = library.new_memory_face(program, 0)?;
    let flags = integer(doc, descriptor.get(b"Flags").ok(), 0)?;
    let symbolic = flags & 4 != 0 && flags & 32 == 0;
    let maps: Vec<_> = (0..face.num_charmaps())
        .map(|i| face.get_charmap(i as isize))
        .collect();
    let selected = if face_is_truetype(&face) {
        maps.iter()
            .find(|m| symbolic && m.platform_id() == 3 && m.encoding_id() == 0)
            .or_else(|| {
                maps.iter()
                    .find(|m| m.platform_id() == 3 && m.encoding_id() == 1)
            })
            .or_else(|| {
                maps.iter()
                    .find(|m| m.platform_id() == 1 && m.encoding_id() == 0)
            })
            .or(maps.first())
    } else {
        maps.first()
    };
    if let Some(map) = selected {
        face.set_charmap(map)?;
    }
    let mut gid = face.get_char_index(code as usize).unwrap_or(0);
    if gid == 0
        && selected.is_some_and(|map| map.encoding() == freetype::ffi::FT_ENCODING_MS_SYMBOL)
    {
        gid = face.get_char_index(0xf000 | usize::from(code)).unwrap_or(0);
    }
    if let Some(glyph) = glyph {
        if let Some(named) = face_name_index(&face, glyph).filter(|g| *g > 0) {
            gid = named;
        } else if let Some(text) = glyph_name_unicode(glyph) {
            if let Some(ch) = text.chars().next() {
                gid = face.get_char_index(ch as usize).unwrap_or(gid);
            }
        }
    }
    face.set_char_size(65536, 65536, 72, 72)?;
    let loaded = face
        .load_glyph(gid, freetype::face::LoadFlag::IGNORE_TRANSFORM)
        .ok()
        .or_else(|| {
            face.load_glyph(
                gid,
                freetype::face::LoadFlag::IGNORE_TRANSFORM | freetype::face::LoadFlag::NO_HINTING,
            )
            .ok()
        });
    required(loaded)?;
    let outline = required(face.glyph().outline())?;
    let to_point = |p: &freetype::Vector| [p.x as f32 / 65536., p.y as f32 / 65536.];
    let mut points = Vec::new();
    // freetype-rs' contour iterator computes an end pointer at count - 1;
    // do not construct it for an empty outline (e.g. a space glyph).
    if face.glyph().raw().outline.n_contours > 0 {
        for contour in outline.contours_iter() {
            let mut last = to_point(contour.start());
            if points.is_empty() {
                points.push(last);
            }
            for curve in contour {
                match curve {
                    Curve::Line(p) => {
                        last = to_point(&p);
                        points.push(last);
                    }
                    Curve::Bezier2(control, end) => {
                        let c = to_point(&control);
                        let end = to_point(&end);
                        points.push([(last[0] + c[0] * 2.) / 3., (last[1] + c[1] * 2.) / 3.]);
                        points.push([(c[0] * 2. + end[0]) / 3., (c[1] * 2. + end[1]) / 3.]);
                        points.push(end);
                        last = end;
                    }
                    Curve::Bezier3(c1, c2, end) => {
                        points.extend([to_point(&c1), to_point(&c2), to_point(&end)]);
                        last = to_point(&end);
                    }
                }
            }
        }
    }
    let first = integer(doc, spec.get(b"FirstChar").ok(), 0)?;
    let widths = array(doc, spec.get(b"Widths").ok())?;
    let advance = usize::try_from(i64::from(code) - first)
        .ok()
        .and_then(|i| widths.get(i))
        .and_then(|v| number(v).ok())
        .map(|w| w as f32 / 1000.)
        .unwrap_or(face.glyph().advance().x as f32 / 65536.);
    Ok((points, advance))
}

fn face_is_truetype(face: &freetype::Face) -> bool {
    use std::ffi::CStr;
    unsafe extern "C" {
        fn FT_Get_Font_Format(face: freetype::ffi::FT_Face) -> *const std::ffi::c_char;
    }
    let raw = face.raw() as *const _ as freetype::ffi::FT_Face;
    if raw.is_null() {
        return false;
    }
    let format = unsafe { FT_Get_Font_Format(raw) };
    !format.is_null() && unsafe { CStr::from_ptr(format) }.to_bytes() == b"TrueType"
}

fn face_glyph_names(face: &freetype::Face) -> Option<std::collections::HashSet<String>> {
    use std::ffi::CStr;
    let mut names = std::collections::HashSet::new();
    let raw = face.raw() as *const _ as freetype::ffi::FT_Face;
    if unsafe { !freetype::ffi::FT_HAS_GLYPH_NAMES(raw) } {
        return None;
    }
    for glyph in 0..face.num_glyphs().max(0) as u32 {
        let mut buffer = [0 as std::ffi::c_char; 256];
        let error = unsafe {
            freetype::ffi::FT_Get_Glyph_Name(
                raw,
                glyph,
                buffer.as_mut_ptr().cast(),
                buffer.len() as u32,
            )
        };
        if error != 0 {
            return None;
        }
        let name = unsafe { CStr::from_ptr(buffer.as_ptr()) }.to_str().ok()?;
        names.insert(name.to_owned());
    }
    Some(names)
}

fn set_bbox_charmap(face: &freetype::Face, is_truetype: bool, enc_name: &str) {
    let count = face.num_charmaps().max(0) as usize;
    let mut unicode = None;
    let mut non_unicode = None;
    for index in 0..count {
        let map = face.get_charmap(index as isize);
        if map.encoding() == freetype::ffi::FT_ENCODING_UNICODE {
            if unicode.is_none() {
                unicode = Some(map)
            }
        } else if non_unicode.is_none() {
            non_unicode = Some(map)
        }
    }
    let preferred = if is_truetype && enc_name == "Custom" {
        unicode.or(non_unicode)
    } else if is_truetype && enc_name == "WinAnsiEncoding" {
        None
    } else if matches!(
        enc_name,
        "Custom"
            | "MacRomanEncoding"
            | "StandardEncoding"
            | "WinAnsiEncoding"
            | "MacExpertEncoding"
    ) {
        non_unicode
    } else {
        None
    };
    if let Some(map) = preferred {
        let _ = face.set_charmap(&map);
    }
}

fn base14_boxes(doc: &Document, spec: &Dictionary) -> Vec<[f64; 4]> {
    let mut family = name(doc, spec.get(b"BaseFont").ok(), "").unwrap_or_default();
    let tables = &*BASE14_TABLES;
    if let Some(alias) = tables
        .get("base14_alias")
        .and_then(|v| v.get(&family))
        .and_then(|v| v.as_str())
    {
        family = alias.to_owned();
    }
    let encs = &*TYPE1_ENCODINGS;
    let names = encs.get("win_ansi").and_then(|v| v.as_array());
    let core = &*WIN_CORE;
    let base = tables.get("base14_bbox").and_then(|v| v.get(&family));
    let win = core.get("win_core").and_then(|v| v.get(&family));
    (0..256)
        .map(|i| {
            let Some(glyph) = names.and_then(|a| a.get(i)).and_then(|v| v.as_str()) else {
                return [0.; 4];
            };
            let v = base
                .and_then(|m| m.get(glyph))
                .or_else(|| win.and_then(|m| m.get(glyph)))
                .and_then(|v| v.as_array());
            v.filter(|a| a.len() == 4)
                .map(|a| {
                    [
                        a[0].as_f64().unwrap_or(0.),
                        a[1].as_f64().unwrap_or(0.),
                        a[2].as_f64().unwrap_or(0.),
                        a[3].as_f64().unwrap_or(0.),
                    ]
                })
                .unwrap_or([0.; 4])
        })
        .collect()
}

/// Projection reads the original Type0 dictionary, independently of backend
/// construction from its descendant. A projection exception clears both maps.
fn type0_projection(
    doc: &Document,
    font: &Dictionary,
    resources: &mut FontResourceCache,
) -> Result<(Vec<[f64; 4]>, Vec<u32>), FontError> {
    let mut boxes = Vec::new();
    if let Ok(descriptor) = font.get(b"FontDescriptor") {
        let descriptor = resolve_dict(doc, descriptor)?;
        for key in [b"FontFile".as_slice(), b"FontFile2", b"FontFile3"] {
            if let Ok(Object::Reference(id)) = descriptor.get(key) {
                let bytes = resources
                    .decode_projection_font_file_object(doc, &Object::Reference((id.0, 0)))?
                    .ok_or_else(|| FontError::Pdf(String::new()))?;
                let library = freetype::Library::init()?;
                library.new_memory_face(Rc::clone(&bytes), 0)?;
                boxes = simple_font_boxes(&bytes, doc, font, false);
            }
        }
    }
    let cmap = if let Ok(object @ Object::Reference(_)) = font.get(b"ToUnicode") {
        let table = resources
            .unicode_table(doc, object)?
            .ok_or_else(|| FontError::Pdf(String::new()))?;
        required(table.projection.clone())?
    } else {
        Vec::new()
    };
    if boxes.is_empty() {
        boxes = base14_boxes(doc, font);
    }
    if let Some(cid) = cid_projection_bboxes(doc, font, resources)? {
        if !cid.is_empty() {
            boxes = cid;
        }
    }
    Ok((boxes, cmap))
}

/// Reproduce `cidfont.get_descendant_fonts`: MuPDF exposes DescendantFonts as
/// printable text and the Python helper selects the first decimal run in it.
/// Inline dictionaries therefore sometimes select a subtype digit instead of
/// the actual descendant xref; that oddity affects projection boxes only.
fn cid_projection_bboxes(
    doc: &Document,
    font: &Dictionary,
    resources: &mut FontResourceCache,
) -> Result<Option<Vec<[f64; 4]>>, FontError> {
    let Ok(value) = font.get(b"DescendantFonts") else {
        return Ok(None);
    };
    let resolved = resolve(doc, value)?;
    // xref_get_key prints a direct array tightly, while xref_object prints an
    // indirect array non-tightly. Only tight dictionaries prioritize Type/Subtype.
    let text = projection_object_text(&resolved, !matches!(value, Object::Reference(_)));
    let Some(descendant_id) = first_decimal_run(&text) else {
        return Ok(None);
    };
    let desc = doc
        .get_object((descendant_id, 0))
        .map_err(|e| FontError::Pdf(e.to_string()))?;
    let Ok(desc) = resolve_dict(doc, desc) else {
        return Ok(None);
    };
    // get_xref accepts only indirect entries, even if a direct value could resolve.
    let Ok(Object::Reference(descriptor_id)) = desc.get(b"FontDescriptor") else {
        return Ok(None);
    };
    let descriptor = doc
        .get_object((descriptor_id.0, 0))
        .map_err(|e| FontError::Pdf(e.to_string()))?;
    let Ok(descriptor) = resolve_dict(doc, descriptor) else {
        return Ok(None);
    };
    let Some(file) = [b"FontFile".as_slice(), b"FontFile2", b"FontFile3"]
        .iter()
        .find_map(|key| match descriptor.get(key).ok()? {
            Object::Reference(id) => Some(*id),
            _ => None,
        })
    else {
        return Ok(None);
    };
    let bytes = resources
        .decode_projection_font_file_object(doc, &Object::Reference((file.0, 0)))?
        .ok_or_else(|| FontError::Pdf(String::new()))?;
    if bytes.is_empty() {
        return Ok(None);
    }
    let lib = freetype::Library::init().map_err(|e| FontError::FreeType(e.to_string()))?;
    let face = lib
        .new_memory_face(Rc::clone(&bytes), 0)
        .map_err(|e| FontError::FreeType(e.to_string()))?;
    let units = face.raw().units_per_EM as f64;
    if units == 0. {
        return Err(FontError::FreeType("zero units_per_EM".into()));
    }
    let scale = 1000.0 / units;
    Ok(Some(
        (0..face.num_glyphs().max(0) as u32)
            .map(|gid| glyph_bbox(&face, gid, scale))
            .collect(),
    ))
}

/// The reference's decimal regex can create an object dependency which is not
/// a PDF indirect reference (for example a number in an inline descendant).
pub(crate) fn projection_descendant_id(doc: &Document, font: &Dictionary) -> Option<ObjectId> {
    let value = font.get(b"DescendantFonts").ok()?;
    let resolved = resolve(doc, value).ok()?;
    let text = projection_object_text(&resolved, !matches!(value, Object::Reference(_)));
    first_decimal_run(&text).map(|id| (id, 0))
}

// MuPDF 1.27.2 object print spelling relevant to cidfont.py's text regex.
// Whitespace between tokens cannot change a decimal run, so arrays/dicts use
// separators while retaining the tight printer's Type/Subtype ordering.
fn projection_object_text(object: &Object, tight: bool) -> String {
    match object {
        Object::Null => "null".into(),
        Object::Boolean(value) => value.to_string(),
        Object::Integer(value) => value.to_string(),
        Object::Real(value) => crate::object_numbers::printed_real_text(*value),
        Object::Reference(id) => format!("{} {} R", id.0, id.1),
        Object::Name(name) => {
            let mut text = String::from("/");
            for &byte in name.iter().take_while(|&&b| b != 0) {
                if byte <= 32 || byte >= 127 || b"()<>[]{}/%#".contains(&byte) {
                    text.push_str(&format!("#{byte:02X}"));
                } else {
                    text.push(byte as char);
                }
            }
            text
        }
        Object::String(bytes, _) => {
            let escaped_length: usize = bytes
                .iter()
                .map(|&b| {
                    if b >= 127 || b == 0 || b < 32 && !b"\n\r\t\x08\x0c".contains(&b) {
                        4
                    } else if b"\n\r\t\x08\x0c()\\".contains(&b) {
                        2
                    } else {
                        1
                    }
                })
                .sum();
            if bytes.starts_with(&[0xfe, 0xff])
                || bytes.starts_with(&[0xff, 0xfe])
                || escaped_length > bytes.len() * 2
            {
                return format!(
                    "<{}>",
                    bytes.iter().map(|b| format!("{b:02X}")).collect::<String>()
                );
            }
            let mut text = String::from("(");
            for &byte in bytes {
                match byte {
                    b'\n' => text.push_str("\\n"),
                    b'\r' => text.push_str("\\r"),
                    b'\t' => text.push_str("\\t"),
                    8 => text.push_str("\\b"),
                    12 => text.push_str("\\f"),
                    b'(' | b')' | b'\\' => {
                        text.push('\\');
                        text.push(byte as char);
                    }
                    0..=31 | 127..=255 => text.push_str(&format!("\\{byte:03o}")),
                    _ => text.push(byte as char),
                }
            }
            text.push(')');
            text
        }
        Object::Array(items) => format!(
            "[{}]",
            items
                .iter()
                .map(|o| projection_object_text(o, tight))
                .collect::<Vec<_>>()
                .join(" ")
        ),
        Object::Dictionary(dictionary) => {
            let mut fields = Vec::new();
            let priority = tight
                && dictionary
                    .get(b"Type")
                    .is_ok_and(|v| !matches!(v, Object::Null));
            let subtype = priority
                && dictionary
                    .get(b"Subtype")
                    .is_ok_and(|v| !matches!(v, Object::Null));
            if priority {
                fields.push((b"Type".as_slice(), dictionary.get(b"Type").unwrap()));
            }
            if subtype {
                fields.push((b"Subtype".as_slice(), dictionary.get(b"Subtype").unwrap()));
            }
            fields.extend(
                dictionary
                    .iter()
                    .filter(|(k, _)| {
                        !(priority && k.as_slice() == b"Type"
                            || subtype && k.as_slice() == b"Subtype")
                    })
                    .map(|(k, v)| (k.as_slice(), v)),
            );
            format!(
                "<<{}>>",
                fields
                    .into_iter()
                    .map(|(k, v)| format!(
                        "{} {}",
                        projection_object_text(&Object::Name(k.to_vec()), tight),
                        projection_object_text(v, tight)
                    ))
                    .collect::<Vec<_>>()
                    .join(" ")
            )
        }
        Object::Stream(stream) => {
            projection_object_text(&Object::Dictionary(stream.dict.clone()), tight)
        }
    }
}

fn first_decimal_run(text: &str) -> Option<u32> {
    let digits = text.bytes().position(|b| b.is_ascii_digit())?;
    let end = digits
        + text[digits..]
            .bytes()
            .take_while(|b| b.is_ascii_digit())
            .count();
    text[digits..end].parse().ok()
}

fn parse_differences(spec: &Dictionary, doc: &Document) -> Vec<(usize, String)> {
    let Some(enc) = spec
        .get(b"Encoding")
        .ok()
        .and_then(|v| resolve(doc, v).ok())
    else {
        return Vec::new();
    };
    let Object::Dictionary(d) = enc else {
        return Vec::new();
    };
    // xref_get_key returns an indirect Differences value as "N 0 R";
    // the reference regex does not dereference it.
    let Ok(Object::Array(values)) = d.get(b"Differences") else {
        return Vec::new();
    };
    let mut result = Vec::new();
    let mut code = 0usize;
    for v in values {
        if let Object::Integer(i) = v {
            code = (*i).max(0) as usize
        } else {
            result.push((code, name(doc, Some(&v), "").unwrap_or_default()));
            code += 1
        }
    }
    result
}

fn glyph_bbox(face: &freetype::Face, gid: u32, scale: f64) -> [f64; 4] {
    if face
        .load_glyph(gid, freetype::face::LoadFlag::NO_SCALE)
        .is_err()
    {
        return [0.; 4];
    }
    let slot = face.glyph().raw();
    if slot.outline.n_contours <= 0 {
        return [0.; 4];
    }
    let mut bbox = freetype::ffi::FT_BBox {
        xMin: 0,
        yMin: 0,
        xMax: 0,
        yMax: 0,
    };
    if unsafe { freetype::ffi::FT_Outline_Get_BBox(&slot.outline, &mut bbox) }
        != freetype::ffi::FT_Err_Ok
    {
        return [0.; 4];
    }
    [
        bbox.xMin as f64 * scale,
        bbox.yMin as f64 * scale,
        bbox.xMax as f64 * scale,
        bbox.yMax as f64 * scale,
    ]
}

fn freetype_details(
    bytes: &[u8],
    requested_name: &str,
    need_boxes: bool,
) -> Result<(FontStyleFlags, Vec<[f64; 4]>), FontError> {
    let lib = freetype::Library::init().map_err(|e| FontError::FreeType(e.to_string()))?;
    let mut face = lib
        .new_memory_face(bytes.to_vec(), 0)
        .map_err(|e| FontError::FreeType(e.to_string()))?;
    let (units, b, ascender, descender) = {
        let raw = face.raw();
        (
            (raw.units_per_EM.max(1)) as f64,
            raw.bbox,
            raw.ascender as f64,
            raw.descender as f64,
        )
    };
    let family = face.family_name();
    let style = face.style_name();
    let default_name = match (&family, &style) {
        (None, Some(s)) => s.clone(),
        (Some(f), None) => f.clone(),
        (Some(f), Some(s)) if s.starts_with(f) => s.clone(),
        (Some(f), Some(s)) => {
            // MuPDF assembles family + style in char namebuf[32] before
            // applying its case-sensitive style-name heuristics.
            let combined = format!("{f} {s}");
            String::from_utf8_lossy(&combined.as_bytes()[..combined.len().min(31)]).into_owned()
        }
        (None, None) => requested_name.to_owned(),
    };
    let chosen_name = if requested_name.is_empty() {
        default_name
    } else {
        requested_name.to_owned()
    };
    let mut flags = FontStyleFlags {
        mono: face.is_fixed_width(),
        monospace: face.is_fixed_width(),
        serif: true,
        bold: face.style_flags().contains(freetype::face::StyleFlag::BOLD),
        italic: face
            .style_flags()
            .contains(freetype::face::StyleFlag::ITALIC),
        embed: true,
        never_embed: false,
        has_opentype: false,
    };
    if face.is_sfnt() {
        // OS/2 sFamilyClass == 8 (sans-serif) maps to MuPDF's non-serif flag.
        if let Some(os2) = freetype::tt_os2::TrueTypeOS2Table::from_face(&mut face) {
            flags.serif = os2.s_family_class() & 0x0800 == 0;
            if os2.fs_type() & (0x0002 | 0x0200) != 0 {
                flags.never_embed = true;
                flags.embed = false;
            }
        }
        for tag in [*b"GDEF", *b"GPOS", *b"GSUB"] {
            let mut length = 0;
            let err = unsafe {
                freetype::ffi::FT_Load_Sfnt_Table(
                    face.raw_mut(),
                    u32::from_be_bytes(tag) as _,
                    0,
                    std::ptr::null_mut(),
                    &mut length,
                )
            };
            if err == freetype::ffi::FT_Err_Ok {
                flags.has_opentype = true;
                break;
            }
        }
    }
    if !flags.bold {
        flags.bold = chosen_name.contains("Semibold") || chosen_name.contains("Bold");
    }
    if !flags.italic {
        flags.italic = chosen_name.contains("Italic") || chosen_name.contains("Oblique");
    }
    // MuPDF uses these trust limits before retaining FreeType ascent/descent.
    let asc = ascender / units;
    let desc = descender / units;
    let _normalized = (asc, desc);
    let _font_bbox = [
        b.xMin as f64 / units,
        b.yMin as f64 / units,
        b.xMax as f64 / units,
        b.yMax as f64 / units,
    ];
    let mut glyph_boxes = Vec::new();
    if need_boxes {
        let count = face.num_glyphs().max(0) as u32;
        glyph_boxes.reserve(count as usize);
        for gid in 0..count {
            glyph_boxes.push(glyph_bbox(&face, gid, 1000.0 / units));
        }
    }
    Ok((flags, glyph_boxes))
}

fn float_value(doc: &Document, o: &Object) -> Result<f64, &'static str> {
    match resolve(doc, o).map_err(|_| "TypeError")? {
        Object::Integer(n) => Ok(n as f64),
        Object::Real(n) => Ok(crate::object_numbers::printed_real(n)),
        Object::Boolean(b) => Ok(if b { 1. } else { 0. }),
        Object::String(s, _) => std::str::from_utf8(&s)
            .ok()
            .and_then(|s| s.trim().parse().ok())
            .ok_or("ValueError"),
        _ => Err("TypeError"),
    }
}
fn parse_width_table(
    doc: &Document,
    object: &Object,
) -> Result<Vec<CachedWidthValue>, FontError> {
    Ok(array(doc, Some(object))?
        .iter()
        .map(|value| float_value(doc, value))
        .collect())
}
fn apply_width_table(out: &mut FontBackend, first: i64, widths: &[CachedWidthValue]) {
    for (index, value) in widths.iter().enumerate() {
        let Ok(cid) = u32::try_from(i128::from(first) + index as i128) else {
            continue;
        };
        store_width_value(out, cid, *value);
    }
}
fn store_default_widths(out: &mut FontBackend, first: i64, count: usize) {
    for index in 0..count {
        let Ok(cid) = u32::try_from(i128::from(first) + index as i128) else {
            continue;
        };
        store_width_value(out, cid, Ok(0.));
    }
}
fn store_width_value(out: &mut FontBackend, cid: u32, value: CachedWidthValue) {
    out.widths.remove(&cid);
    out.width_errors.remove(&cid);
    match value {
        Ok(width) => {
            out.widths.insert(cid, width);
        }
        Err(kind) => {
            out.width_errors.insert(cid, kind);
        }
    }
}
fn store_width(doc: &Document, out: &mut FontBackend, cid: u32, o: &Object) {
    store_width_value(out, cid, float_value(doc, o));
}
fn parse_cid_widths(
    doc: &Document,
    spec: &Dictionary,
    out: &mut FontBackend,
    vertical: bool,
) -> Result<(), FontError> {
    let vals = array(
        doc,
        spec.get(if vertical {
            b"W2".as_slice()
        } else {
            b"W".as_slice()
        })
        .ok(),
    )?;
    let mut pending: Vec<Object> = Vec::new();
    let mut order = 0usize;
    let mut single_orders = BTreeMap::new();
    let mut range_orders = Vec::new();
    for item in vals {
        let value = resolve(doc, &item)?;
        if let Object::Array(items) = value {
            if let Some(start) = pending.last() {
                let start = number(start)?;
                let stride = if vertical { 3 } else { 1 };
                for (i, chunk) in items.chunks_exact(stride).enumerate() {
                    let cid = start + i as f64;
                    if cid.fract() != 0. || !(0. ..=u32::MAX as f64).contains(&cid) {
                        continue;
                    }
                    let cid = cid as u32;
                    store_width(doc, out, cid, &chunk[0]);
                    single_orders.insert(cid, order);
                    order += 1;
                    if vertical {
                        out.displacement_errors.remove(&cid);
                        match (float_value(doc, &chunk[1]), float_value(doc, &chunk[2])) {
                            (Ok(x), Ok(y)) => {
                                out.displacements.insert(cid, (Some(x), y));
                            }
                            _ => {
                                out.displacement_errors.insert(cid, "TypeError");
                            }
                        }
                    }
                }
                pending.clear();
            }
        } else if matches!(
            value,
            Object::Integer(_) | Object::Real(_) | Object::Boolean(_)
        ) {
            pending.push(value);
            if pending.len() == if vertical { 5 } else { 3 } {
                let integral = |o: &Object| matches!(o, Object::Integer(_) | Object::Boolean(_));
                if !integral(&pending[0]) || !integral(&pending[1]) {
                    if vertical {
                        return Err(FontError::Typed(
                            "TypeError",
                            "float object cannot be interpreted as an integer".into(),
                        ));
                    }
                } else {
                    let start = number(&pending[0])?.max(0.);
                    let end = number(&pending[1])?.min(u32::MAX as f64);
                    if start <= end {
                        let lo = start as u32;
                        let hi = end as u32;
                        let w = number(&pending[2])?;
                        range_orders.push(order);
                        order += 1;
                        out.width_ranges.push((lo, hi, w));
                        if vertical {
                            out.displacement_ranges.push((
                                lo,
                                hi,
                                (Some(number(&pending[3])?), number(&pending[4])?),
                            ));
                        }
                    }
                }
                pending.clear();
            }
        }
    }
    // Resolve all range/single priorities once, rather than scanning singles per range.
    let segments = overlay(
        &out.width_ranges
            .iter()
            .map(|r| (r.0, r.1))
            .collect::<Vec<_>>(),
        true,
    );
    let keep_single = |cid: &u32| {
        lookup(&segments, *cid, |r| (r.0, r.1)).is_none_or(|range| {
            range_orders[range.2] < single_orders.get(cid).copied().unwrap_or(0)
        })
    };
    out.widths.retain(|cid, _| keep_single(cid));
    out.width_errors.retain(|cid, _| keep_single(cid));
    if vertical {
        out.displacements.retain(|cid, _| keep_single(cid));
        out.displacement_errors.retain(|cid, _| keep_single(cid));
        out.displacement_ranges = segments
            .iter()
            .map(|&(lo, hi, index)| (lo, hi, out.displacement_ranges[index].2))
            .collect();
    }
    out.width_ranges = segments
        .into_iter()
        .map(|(lo, hi, index)| (lo, hi, out.width_ranges[index].2))
        .collect();
    Ok(())
}

#[cfg(test)]
fn unicode_map_lookup<'a>(
    singles: &'a BTreeMap<u32, String>,
    ranges: &[(u32, u32, UnicodeTarget, usize)],
    orders: &BTreeMap<u32, usize>,
    code: u32,
) -> Option<Cow<'a, str>> {
    unicode_map_lookup_with_policy(singles, ranges, orders, code, None).unwrap()
}
fn unicode_map_lookup_with_policy<'a>(
    singles: &'a BTreeMap<u32, String>,
    ranges: &[(u32, u32, UnicodeTarget, usize)],
    orders: &BTreeMap<u32, usize>,
    code: u32,
    policy: Option<&Policy>,
) -> Result<Option<Cow<'a, str>>, FontError> {
    if let Some((_, _, raw, order)) = lookup(ranges, code, |r| (r.0, r.1)) {
        if orders.get(&code).copied().unwrap_or(0) <= *order {
            check_mapping(policy, raw.bytes.len())?;
            return Ok(Some(Cow::Owned(decode_mapping(
                &incremented_bytes(&raw.bytes, code - raw.origin),
                policy,
            )?)));
        }
    }
    if let Some(text) = singles.get(&code) {
        check_mapping(policy, text.len())?;
        Ok(Some(Cow::Borrowed(text.as_str())))
    } else {
        Ok(None)
    }
}
#[cfg(test)]
fn parse_tounicode(bytes: &[u8], unicode: &mut BTreeMap<u32, String>) {
    let mut ranges = Vec::new();
    let mut orders = BTreeMap::new();
    parse_tounicode_lazy(bytes, unicode, &mut ranges, &mut orders).unwrap();
    for (lo, hi, raw, _) in ranges {
        for code in lo..=hi {
            unicode.insert(
                code,
                unicode_from_incremented_utf16(&raw.bytes, code - raw.origin),
            );
        }
    }
}
#[cfg(test)]
fn parse_tounicode_lazy(
    bytes: &[u8],
    unicode: &mut BTreeMap<u32, String>,
    ranges: &mut Vec<(u32, u32, UnicodeTarget, usize)>,
    orders: &mut BTreeMap<u32, usize>,
) -> Result<(), FontError> {
    parse_tounicode_with_policy(bytes, unicode, ranges, orders, None)
}
fn parse_tounicode_with_policy(
    bytes: &[u8],
    unicode: &mut BTreeMap<u32, String>,
    ranges: &mut Vec<(u32, u32, UnicodeTarget, usize)>,
    orders: &mut BTreeMap<u32, usize>,
    policy: Option<&Policy>,
) -> Result<(), FontError> {
    let tokens = cmap_tokens(bytes);
    let mut i = 0;
    let mut order = 1usize;
    while i < tokens.len() {
        if tokens[i] == "begincidrange" {
            i += 1;
            while i < tokens.len() && tokens[i] != "endcidrange" {
                if i + 2 >= tokens.len() {
                    break;
                }
                let Some(start_bytes) = hex_string_bytes(&tokens[i]) else {
                    i += 3;
                    continue;
                };
                let Some(end_bytes) = hex_string_bytes(&tokens[i + 1]) else {
                    i += 3;
                    continue;
                };
                let prefix_len = start_bytes.len().saturating_sub(4);
                let start_prefix = &start_bytes[..prefix_len];
                let end_prefix_len = end_bytes.len().saturating_sub(4);
                let end_prefix = &end_bytes[..end_prefix_len];
                let lo = start_bytes[prefix_len..]
                    .iter()
                    .fold(0u32, |n, b| (n << 8) | u32::from(*b));
                let hi = end_bytes[end_prefix_len..]
                    .iter()
                    .fold(0u32, |n, b| (n << 8) | u32::from(*b));
                let Ok(cid) = tokens[i + 2].parse::<u32>() else {
                    i += 3;
                    continue;
                };
                if lo <= hi
                    && start_bytes.len() == end_bytes.len()
                    && start_prefix == end_prefix
                    && cid.checked_add(hi - lo).is_some()
                {
                    check_mapping(policy, start_bytes.len())?;
                    decode_mapping(&start_bytes, policy)?;
                    ranges.push((
                        cid,
                        cid + (hi - lo),
                        UnicodeTarget::new(start_bytes, cid),
                        order,
                    ));
                    order += 1;
                }
                i += 3;
            }
        } else if tokens[i] == "beginbfchar" {
            i += 1;
            while i < tokens.len() && tokens[i] != "endbfchar" {
                if i + 1 < tokens.len() {
                    if let Some((code, text)) = mapping_pair(&tokens[i], &tokens[i + 1], policy)? {
                        unicode.insert(code, text);
                        orders.insert(code, order);
                        order += 1;
                    }
                    i += 2;
                } else {
                    break;
                }
            }
        } else if tokens[i] == "beginbfrange" {
            i += 1;
            while i < tokens.len() && tokens[i] != "endbfrange" {
                if i + 2 >= tokens.len() {
                    break;
                }
                let Some((lo, src_width)) = parse_unicode_code(&tokens[i]) else {
                    i += 1;
                    continue;
                };
                let Some((hi, hi_width)) = parse_unicode_code(&tokens[i + 1]) else {
                    i += 1;
                    continue;
                };
                if src_width != hi_width {
                    i += 3;
                    continue;
                }
                let mut dst = i + 2;
                if tokens[dst] == "[" {
                    dst += 1;
                    for cid in lo..=hi {
                        if dst >= tokens.len() || tokens[dst] == "]" {
                            break;
                        }
                        // The source is already a validated u32. Reformatting it to
                        // src_width hex digits would replay a long zero prefix per item.
                        if let Some(raw) = mapping_bytes(&tokens[dst], policy)? {
                            unicode.insert(cid, decode_mapping(&raw, policy)?);
                            orders.insert(cid, order);
                            order += 1;
                        }
                        dst += 1;
                    }
                    while dst < tokens.len() && tokens[dst] != "]" {
                        dst += 1;
                    }
                    i = (dst + 1).min(tokens.len());
                } else {
                    if let Some(raw) = mapping_bytes(&tokens[dst], policy)? {
                        let base = raw[raw.len().saturating_sub(4)..]
                            .iter()
                            .fold(0u64, |value, b| (value << 8) | u64::from(*b));
                        if hi >= lo && base + u64::from(hi - lo) > u32::MAX as u64 {
                            return Err(FontError::Typed(
                                "StructError",
                                "argument out of range".into(),
                            ));
                        }
                        if hi >= lo && hi - lo > 0 {
                            decode_mapping(&raw, policy)?;
                            ranges.push((lo, hi, UnicodeTarget::new(raw, lo), order));
                            order += 1;
                        } else if hi == lo {
                            unicode.insert(lo, decode_mapping(&raw, policy)?);
                            orders.insert(lo, order);
                            order += 1;
                        }
                    }
                    i = dst + 1;
                }
            }
        }
        i += 1;
    }
    compile_unicode(ranges);
    Ok(())
}
fn hex_decode(s: &str) -> Vec<u8> {
    (0..s.len())
        .step_by(2)
        .filter_map(|i| u8::from_str_radix(&s[i..(i + 2).min(s.len())], 16).ok())
        .collect()
}
fn parse_hex_token(token: &str) -> Option<(u32, usize)> {
    let x = token.strip_prefix('<')?.strip_suffix('>')?;
    Some((u32::from_str_radix(x, 16).ok()?, x.len()))
}
fn parse_unicode_code(token: &str) -> Option<(u32, usize)> {
    let bytes = hex_string_bytes(token)?;
    let mut code = 0u32;
    for byte in &bytes {
        code = code.checked_mul(256)?.checked_add(u32::from(*byte))?;
    }
    Some((code, bytes.len() * 2))
}
fn hex_string_bytes(token: &str) -> Option<Vec<u8>> {
    let mut digits: String = token
        .strip_prefix('<')?
        .strip_suffix('>')?
        .chars()
        .filter(|c| !matches!(c, '\0' | '\t' | '\n' | '\x0c' | '\r' | ' '))
        .collect();
    if !digits.bytes().all(|b| b.is_ascii_hexdigit()) {
        return None;
    }
    if digits.len() % 2 != 0 {
        digits.push('0');
    }
    Some(hex_decode(&digits))
}
#[cfg(test)]
fn unicode_pair(src: &str, dst: &str) -> Option<(u32, String)> {
    let (code, _) = parse_unicode_code(src)?;
    let bytes = hex_string_bytes(dst)?;
    Some((code, decode_utf16_ignoring_errors(&bytes)))
}

/// Match the intentionally small projection parser in active_font_support:
/// ranges are read before chars, and the range target is repeated for each key.
#[cfg(test)]
fn parse_projection_cmap(bytes: &[u8]) -> Option<Vec<u32>> {
    let packed = parse_projection_intervals(bytes, None).unwrap()?;
    Some(packed.chunks_exact(2).flat_map(|r| r[0]..=r[1]).collect())
}
// Projection ignores malformed UTF-16 targets. Validate the input token without
// allocating raw mapping bytes, so ignored entries cannot trip their byte cap.
fn projection_target_is_valid_utf16(token: &str) -> bool {
    let Some(body) = token.strip_prefix('<').and_then(|s| s.strip_suffix('>')) else { return false; };
    if body.len() % 4 != 0 || !body.bytes().all(|b| b.is_ascii_hexdigit()) { return false; }
    let words = body.as_bytes().chunks_exact(4).map(|hex| {
        hex.iter().fold(0u16, |word, &byte| {
            let digit = match byte { b'0'..=b'9' => byte-b'0', b'a'..=b'f' => byte-b'a'+10, _ => byte-b'A'+10 };
            (word << 4) | u16::from(digit)
        })
    });
    char::decode_utf16(words).all(|c| c.is_ok())
}

fn parse_projection_intervals(
    bytes: &[u8],
    policy: Option<&Policy>,
) -> Result<Option<Vec<u32>>, FontError> {
    let mut ordered = Vec::new();
    let Ok(text) = std::str::from_utf8(bytes) else {
        return Ok(None);
    };
    for (begin, end, stride, requires_trailing_ws) in [
        ("beginbfrange", "endbfrange", 3, true),
        ("beginbfchar", "endbfchar", 2, false),
    ] {
        for body in projection_cmap_blocks(text, begin, end, requires_trailing_ws) {
            let Some(hex) = projection_hex_tokens(body) else {
                continue;
            };
            if hex.len() % stride != 0 {
                return Ok(None);
            }
            for group in hex.chunks_exact(stride) {
                if stride == 2 {
                    let Some((code, _)) = parse_hex_token(group[0]) else {
                        continue;
                    };
                    if !projection_target_is_valid_utf16(group[1]) {
                        continue;
                    }
                    let Some(raw) = mapping_bytes(group[1], policy)? else {
                        continue;
                    };
                    if raw.len() % 2 == 0
                        && char::decode_utf16(
                            raw.chunks_exact(2)
                                .map(|b| u16::from_be_bytes([b[0], b[1]])),
                        )
                        .all(|c| c.is_ok())
                    {
                        ordered.push((code, code));
                    }
                } else {
                    let Some((lo, _)) = parse_hex_token(group[0]) else {
                        continue;
                    };
                    let Some((hi, _)) = parse_hex_token(group[1]) else {
                        continue;
                    };
                    if !projection_target_is_valid_utf16(group[2]) {
                        continue;
                    }
                    // b20 ignores reversed ranges before materializing their
                    // target; ignored declarations must not consume a limit.
                    if hi < lo { continue; }
                    let Some(raw) = mapping_bytes(group[2], policy)? else {
                        continue;
                    };
                    if raw.len() % 2 != 0
                        || !char::decode_utf16(
                            raw.chunks_exact(2)
                                .map(|b| u16::from_be_bytes([b[0], b[1]])),
                        )
                        .all(|c| c.is_ok())
                    {
                        continue;
                    }
                    ordered.push((lo, hi));
                }
            }
        }
    }
    let mut segments = overlay(&ordered, false);
    segments.sort_unstable_by_key(|&(lo, _, order)| (order, lo));
    Ok(Some(
        segments
            .into_iter()
            .flat_map(|(lo, hi, _)| [lo, hi])
            .collect(),
    ))
}

fn projection_cmap_blocks<'a>(
    text: &'a str,
    begin: &str,
    end: &str,
    requires_trailing_ws: bool,
) -> Vec<&'a str> {
    let mut blocks = Vec::new();
    let mut cursor = 0;
    while cursor < text.len() {
        let Some(relative) = text[cursor..].find(begin) else {
            break;
        };
        let start = cursor + relative;
        let before = text[..start].chars().next_back();
        let after = &text[start + begin.len()..];
        if !before.is_some_and(char::is_whitespace) {
            cursor = start + begin.len();
            continue;
        }
        let Some(end_relative) = after.find(end) else {
            break;
        };
        let end_start = start + begin.len() + end_relative;
        let end_after = end_start + end.len();
        if requires_trailing_ws
            && !text[end_after..]
                .chars()
                .next()
                .is_some_and(char::is_whitespace)
        {
            cursor = end_after;
            continue;
        }
        blocks.push(&text[start + begin.len()..end_start]);
        cursor = end_after;
    }
    blocks
}

fn projection_hex_tokens(body: &str) -> Option<Vec<&str>> {
    let bytes = body.as_bytes();
    let mut tokens = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i].is_ascii_whitespace() {
            i += 1;
            continue;
        }
        if bytes[i] != b'<' {
            return None;
        }
        let start = i;
        i += 1;
        let digits = i;
        while i < bytes.len() && bytes[i].is_ascii_hexdigit() {
            i += 1
        }
        if i == digits || i >= bytes.len() || bytes[i] != b'>' {
            return None;
        }
        i += 1;
        tokens.push(&body[start..i]);
    }
    Some(tokens)
}

#[cfg(test)]
fn valid_utf16_hex(token: &str) -> bool {
    // The legacy bounding-box projection uses strict b16decode, unlike the
    // text backend's PDF hex-string decoder, which pads a trailing nibble.
    if token.len().saturating_sub(2) % 2 != 0 {
        return false;
    }
    let Some(bytes) = hex_string_bytes(token) else {
        return false;
    };
    if bytes.len() % 2 != 0 {
        return false;
    }
    let words: Vec<u16> = bytes
        .chunks_exact(2)
        .map(|pair| u16::from_be_bytes([pair[0], pair[1]]))
        .collect();
    String::from_utf16(&words).is_ok()
}

fn cmap_tokens(bytes: &[u8]) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i].is_ascii_whitespace() {
            i += 1;
            continue;
        }
        if bytes[i] == b'<' {
            let start = i;
            i += 1;
            while i < bytes.len() && bytes[i] != b'>' {
                i += 1
            }
            if i < bytes.len() {
                i += 1
            }
            tokens.push(String::from_utf8_lossy(&bytes[start..i]).into_owned());
            continue;
        }
        if bytes[i] == b'[' || bytes[i] == b']' {
            tokens.push((bytes[i] as char).to_string());
            i += 1;
            continue;
        }
        let start = i;
        while i < bytes.len()
            && !bytes[i].is_ascii_whitespace()
            && bytes[i] != b'<'
            && bytes[i] != b'>'
            && bytes[i] != b'['
            && bytes[i] != b']'
        {
            i += 1
        }
        if start < i {
            tokens.push(String::from_utf8_lossy(&bytes[start..i]).into_owned())
        } else {
            i += 1
        }
    }
    tokens
}
#[cfg(test)]
fn unicode_from_incremented_utf16(bytes: &[u8], increment: u32) -> String {
    let n = bytes.len().min(4);
    let split = bytes.len() - n;
    let base = bytes[split..]
        .iter()
        .fold(0u32, |value, b| (value << 8) | u32::from(*b));
    let value = base.wrapping_add(increment).to_be_bytes();
    let mut mapped = bytes[..split].to_vec();
    mapped.extend_from_slice(&value[4 - n..]);
    decode_utf16_ignoring_errors(&mapped)
}
#[cfg(test)]
fn decode_utf16_ignoring_errors(bytes: &[u8]) -> String {
    char::decode_utf16(
        bytes
            .chunks_exact(2)
            .map(|b| u16::from_be_bytes([b[0], b[1]])),
    )
    .filter_map(std::result::Result::ok)
    .collect()
}

fn resolve(doc: &Document, o: &Object) -> Result<Object, FontError> {
    match o {
        Object::Reference(id) => doc.get_object(*id).cloned().map_err(|e| match e {
            lopdf::Error::ObjectNotFound(_) => FontError::Typed("RuntimeError", e.to_string()),
            _ => FontError::Pdf(e.to_string()),
        }),
        _ => Ok(o.clone()),
    }
}
fn resolve_dict(doc: &Document, o: &Object) -> Result<Dictionary, FontError> {
    match resolve(doc, o)? {
        Object::Dictionary(d) => Ok(d),
        Object::Stream(s) => Ok(s.dict),
        _ => Err(FontError::Pdf("font object is not a dictionary".into())),
    }
}
fn resolve_stream<'a>(doc: &'a Document, o: &'a Object) -> Result<&'a Stream, FontError> {
    let id = match o {
        Object::Reference(id) => Some(*id),
        _ => None,
    };
    if let Some(id) = id {
        doc.get_object(id)
            .map_err(|e| FontError::Pdf(e.to_string()))?
            .as_stream()
            .map_err(|e| FontError::Pdf(e.to_string()))
    } else {
        o.as_stream().map_err(|e| FontError::Pdf(e.to_string()))
    }
}
fn array<'a>(doc: &'a Document, o: Option<&'a Object>) -> Result<Vec<Object>, FontError> {
    let Some(o) = o else { return Ok(Vec::new()) };
    match resolve(doc, o)? {
        Object::Array(a) => Ok(a),
        _ => Ok(Vec::new()),
    }
}
fn number(o: &Object) -> Result<f64, FontError> {
    match o {
        Object::Integer(n) => Ok(*n as f64),
        Object::Real(n) => Ok(crate::object_numbers::printed_real(*n)),
        Object::Boolean(b) => Ok(if *b { 1. } else { 0. }),
        _ => Err(FontError::Pdf("expected numeric font value".into())),
    }
}
fn num(doc: &Document, o: Option<&Object>, default: f64) -> Result<f64, FontError> {
    let Some(o) = o else { return Ok(default) };
    number(&resolve(doc, o)?).or(Ok(default))
}
fn integer(doc: &Document, o: Option<&Object>, default: i64) -> Result<i64, FontError> {
    Ok(match o.and_then(|o| resolve(doc, o).ok()) {
        Some(Object::Integer(n)) => n,
        Some(Object::Boolean(b)) => i64::from(b),
        _ => default,
    })
}
fn name(doc: &Document, o: Option<&Object>, default: &str) -> Result<String, FontError> {
    let Some(o) = o else {
        return Ok(default.into());
    };
    match resolve(doc, o)? {
        Object::Name(n) => Ok(normalize_pdf_name(&n)),
        Object::String(s, _) => Ok(normalize_pdf_name(&s)),
        _ => Ok(default.into()),
    }
}
fn normalize_pdf_name(bytes: &[u8]) -> String {
    if let Ok(text) = std::str::from_utf8(bytes) {
        return text.to_owned();
    }
    crate::tokens::python_bytes_repr(bytes)
}
fn bbox(doc: &Document, o: Option<&Object>) -> [f64; 4] {
    let vals = array(doc, o).unwrap_or_default();
    if vals.len() < 4 {
        return [0.0; 4];
    }
    [0, 1, 2, 3].map(|i| number(&vals[i]).unwrap_or(0.0))
}
#[cfg(test)]
mod fonts_tests;
