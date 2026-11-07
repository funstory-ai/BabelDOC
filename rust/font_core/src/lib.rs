#![forbid(unsafe_code)]

//! Thread-safe font assets and deterministic FontMapper decisions for the
//! parser-after Rust pipeline. This crate deliberately has no Python binding.

use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::sync::{Arc, OnceLock};
mod pdf_font_flags;

use sha3::{Digest, Sha3_256};
use skrifa::instance::Size;
use skrifa::outline::{DrawSettings, OutlinePen};
use skrifa::{
    FontRef, MetadataProvider,
    raw::{TableProvider, types::GlyphId},
};

#[derive(Debug)]
pub enum FontError {
    Io(std::io::Error),
    InvalidFont,
    UnknownFontId(String),
    AssetProfileMismatch(String),
    AssetChecksumMismatch(String),
    UnprofiledPdfFlags(String),
}

impl std::fmt::Display for FontError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(err) => write!(f, "font I/O failed: {err}"),
            Self::InvalidFont => f.write_str("invalid or unsupported font data"),
            Self::UnknownFontId(id) => write!(f, "unknown font id: {id}"),
            Self::UnprofiledPdfFlags(sha) => {
                write!(f, "unprofiled PDF font flags for SHA3-256 {sha}")
            }
            Self::AssetProfileMismatch(id) => {
                write!(
                    f,
                    "font asset does not match the embedded PyMuPDF profile: {id}"
                )
            }
            Self::AssetChecksumMismatch(id) => {
                write!(f, "font asset checksum mismatch: {id}")
            }
        }
    }
}

impl std::error::Error for FontError {}

impl From<std::io::Error> for FontError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FontTraits {
    pub bold: bool,
    pub italic: bool,
    pub monospaced: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PdfFontFlags {
    pub bold: bool,
    pub italic: bool,
    pub monospace: bool,
    pub serif: bool,
}

/// Immutable font data. `Arc` makes the face safe to share between the
/// paragraph workers; per-call Skrifa views borrow the immutable bytes.
pub struct FontFace {
    id: String,
    path: std::path::PathBuf,
    data: Arc<[u8]>,
    cmap: HashMap<u32, u32>,
    advances_em: Vec<f32>,
    glyph_count: u32,
    traits: FontTraits,
    asset_index: Option<u8>,
    sha3_digest: OnceLock<[u8; 32]>,
}

impl std::fmt::Debug for FontFace {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FontFace")
            .field("id", &self.id)
            .field("path", &self.path)
            .field("glyph_count", &self.glyph_count)
            .field("traits", &self.traits)
            .finish()
    }
}

impl FontFace {
    /// Exact PyMuPDF flags for the approved finite asset set, keyed by bytes.
    /// These are deliberately separate from the mapper's selection traits.
    pub fn pdf_font_flags(&self) -> Result<PdfFontFlags, FontError> {
        let digest = self.sha3_digest.get_or_init(|| sha3_256(&self.data));
        pdf_font_flags::lookup(&sha3_hex(digest))
    }
    /// Immutable data for the Type0 / Identity-H PDF writer. Ownership and
    /// mapper lifetime remain with the caller of the writer.
    pub fn pdf_embedding(&self) -> Result<PdfEmbedding<'_>, FontError> {
        let font = FontRef::new(&self.data).map_err(|_| FontError::InvalidFont)?;
        let head = font.head().map_err(|_| FontError::InvalidFont)?;
        let upem = head.units_per_em() as f32;
        let postscript_name = font
            .localized_strings(skrifa::string::StringId::POSTSCRIPT_NAME)
            .english_or_first()
            .ok_or(FontError::InvalidFont)?
            .to_string();
        let mut unicode = std::collections::BTreeMap::new();
        // MuPDF enumerates codepoints in ascending order, overwriting aliases
        // for the same glyph. U+0000 and the missing glyph are not emitted.
        for (&code, &gid) in &self.cmap {
            if code != 0 && gid != 0 {
                unicode
                    .entry(gid)
                    .and_modify(|u: &mut u32| *u = (*u).max(code))
                    .or_insert(code);
            }
        }
        Ok(PdfEmbedding {
            data: &self.data,
            postscript_name,
            unicode,
            widths: self
                .advances_em
                .iter()
                .map(|w| (w * 1000.) as i64)
                .collect(),
            bbox: [head.x_min(), head.y_min(), head.x_max(), head.y_max()]
                .map(|v| v as f32 / upem * 1000.),
        })
    }
    pub fn load(id: impl Into<String>, path: impl AsRef<Path>) -> Result<Arc<Self>, FontError> {
        let id = id.into();
        let path = path.as_ref().to_path_buf();
        let data: Arc<[u8]> = fs::read(&path)?.into();
        Self::load_data(id, path, data, None)
    }

    /// Reads an approved asset once, verifies its content, and keeps the
    /// verified digest for later PDF-font flag lookup.
    pub fn load_verified(
        id: impl Into<String>,
        path: impl AsRef<Path>,
        expected_sha3: &str,
    ) -> Result<Arc<Self>, FontError> {
        let id = id.into();
        let path = path.as_ref().to_path_buf();
        let data: Arc<[u8]> = fs::read(&path)?.into();
        let digest = sha3_256(&data);
        if sha3_hex(&digest) != expected_sha3 {
            return Err(FontError::AssetChecksumMismatch(id));
        }
        Self::load_data(id, path, data, Some(digest))
    }

    fn load_data(
        id: String,
        path: std::path::PathBuf,
        data: Arc<[u8]>,
        loaded_digest: Option<[u8; 32]>,
    ) -> Result<Arc<Self>, FontError> {
        let sha3_digest = loaded_digest.map(OnceLock::from).unwrap_or_default();
        let font = FontRef::new(&data).map_err(|_| FontError::InvalidFont)?;
        let cmap = font
            .charmap()
            .mappings()
            .map(|(codepoint, glyph_id)| (codepoint, glyph_id.to_u32()))
            .collect::<HashMap<_, _>>();
        let upem = f32::from(
            font.head()
                .map_err(|_| FontError::InvalidFont)?
                .units_per_em(),
        );
        let glyph_metrics = font.glyph_metrics(Size::new(upem), &[][..]);
        let glyph_count = glyph_metrics.glyph_count();
        let advances_em: Vec<f32> = (0..glyph_count)
            .map(|glyph_id| {
                glyph_metrics
                    .advance_width(GlyphId::new(glyph_id))
                    .unwrap_or_default()
                    / upem
            })
            .collect();
        let attrs = font.attributes();
        let post_fixed = font
            .post()
            .map(|post| post.is_fixed_pitch() != 0)
            .unwrap_or(false);
        let italic = matches!(
            attrs.style,
            skrifa::attribute::Style::Italic | skrifa::attribute::Style::Oblique(_)
        );
        let traits = FontTraits {
            bold: attrs.weight >= skrifa::attribute::Weight::BOLD,
            italic,
            monospaced: post_fixed || id.to_ascii_lowercase().contains("mono"),
        };
        let asset_index = missing_width_table().profile_index(&id);
        if let Some(index) = asset_index {
            let profile = &missing_width_table().profiles[index as usize];
            let actual_hash = *sha3_digest.get_or_init(|| sha3_256(&data));
            if actual_hash != profile.sha3
                || advances_em.first().copied().unwrap_or_default().to_bits()
                    != profile.default_width_em.to_bits()
            {
                return Err(FontError::AssetProfileMismatch(id));
            }
        }
        Ok(Arc::new(Self {
            id,
            path,
            data,
            cmap,
            advances_em,
            glyph_count,
            traits,
            asset_index,
            sha3_digest,
        }))
    }

    pub fn id(&self) -> &str {
        &self.id
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn traits(&self) -> FontTraits {
        self.traits
    }

    pub fn glyph_count(&self) -> u32 {
        self.glyph_count
    }

    pub fn outline_flags(&self) -> Result<Vec<bool>, FontError> {
        let font = FontRef::new(&self.data).map_err(|_| FontError::InvalidFont)?;
        let outlines = font.outline_glyphs();
        let mut flags = Vec::with_capacity(self.glyph_count as usize);
        let mut pen = ContourCounter::default();
        for glyph_id in 0..self.glyph_count {
            pen.contours = 0;
            let has_outline = outlines.get(GlyphId::new(glyph_id)).is_some_and(|outline| {
                outline
                    .draw(DrawSettings::unhinted(Size::unscaled(), &[][..]), &mut pen)
                    .is_ok()
                    && pen.contours > 0
            });
            flags.push(has_outline);
        }
        Ok(flags)
    }

    /// Equivalent to PyMuPDF `Font.has_glyph(codepoint)`: return the output
    /// glyph id, using zero for an unmapped character.
    pub fn has_glyph(&self, codepoint: u32) -> u32 {
        self.cmap.get(&codepoint).copied().unwrap_or(0)
    }

    /// PyMuPDF `char_lengths` returns one advance per Unicode scalar, without
    /// shaping or kerning. Asset-font misses use the PyMuPDF fallback-width
    /// profile; non-asset fonts use their own .notdef advance.
    pub fn char_lengths(&self, text: &str, font_size: f64) -> Vec<f64> {
        text.chars()
            .map(|ch| {
                let codepoint = ch as u32;
                let glyph_id = self.has_glyph(codepoint) as usize;
                let advance = if glyph_id == 0 {
                    self.missing_advance_em(codepoint)
                } else {
                    self.advances_em.get(glyph_id).copied().unwrap_or_default()
                };
                f64::from(advance) * font_size
            })
            .collect()
    }

    fn missing_advance_em(&self, codepoint: u32) -> f32 {
        let default = self.advances_em.first().copied().unwrap_or_default();
        self.asset_index
            .and_then(|font_index| missing_width_table().width(codepoint, font_index))
            .unwrap_or(default)
    }

    /// Mirrors the `freetype-py` test `glyph.outline.contours` being nonempty.
    pub fn has_outline(&self, glyph_id: u32) -> bool {
        let Ok(font) = FontRef::new(&self.data) else {
            return false;
        };
        if glyph_id >= self.glyph_count {
            return false;
        }
        let Some(outline) = font.outline_glyphs().get(GlyphId::new(glyph_id)) else {
            return false;
        };
        let mut pen = ContourCounter::default();
        outline
            .draw(DrawSettings::unhinted(Size::unscaled(), &[][..]), &mut pen)
            .is_ok()
            && pen.contours > 0
    }
}

pub struct PdfEmbedding<'a> {
    pub data: &'a [u8],
    pub postscript_name: String,
    pub unicode: std::collections::BTreeMap<u32, u32>,
    pub widths: Vec<i64>,
    pub bbox: [f32; 4],
}

#[cfg(test)]
mod lib_tests;

#[derive(Default)]
struct ContourCounter {
    contours: usize,
}

impl OutlinePen for ContourCounter {
    fn move_to(&mut self, _: f32, _: f32) {
        self.contours += 1;
    }
    fn line_to(&mut self, _: f32, _: f32) {}
    fn quad_to(&mut self, _: f32, _: f32, _: f32, _: f32) {}
    fn curve_to(&mut self, _: f32, _: f32, _: f32, _: f32, _: f32, _: f32) {}
    fn close(&mut self) {}
}

struct AssetWidthProfile {
    sha3: [u8; 32],
    default_width_em: f32,
}

struct MissingWidthTable {
    profiles: Vec<AssetWidthProfile>,
    profile_indices: HashMap<String, u8>,
    widths: HashMap<u32, Vec<(u64, f32)>>,
}

impl MissingWidthTable {
    fn parse(mut input: &[u8]) -> Result<Self, &'static str> {
        if take(&mut input, 5)? != b"I5MW\x01" {
            return Err("invalid missing-width table header");
        }
        let version_length = take_u8(&mut input)? as usize;
        let version = take(&mut input, version_length)?;
        if version != b"1.27.2.3" {
            return Err("unsupported PyMuPDF table source version");
        }
        let profile_count = take_u16(&mut input)? as usize;
        if profile_count > u8::MAX as usize {
            return Err("too many asset font profiles");
        }
        let mut profiles = Vec::with_capacity(profile_count);
        let mut profile_indices = HashMap::with_capacity(profile_count);
        for index in 0..profile_count {
            let name_length = take_u8(&mut input)? as usize;
            let id = std::str::from_utf8(take(&mut input, name_length)?)
                .map_err(|_| "invalid UTF-8 asset font id")?
                .to_owned();
            let sha3: [u8; 32] = take(&mut input, 32)?
                .try_into()
                .map_err(|_| "invalid asset font SHA3 length")?;
            let default_width_em = f32::from_bits(take_u32(&mut input)?);
            if !default_width_em.is_finite() || default_width_em < 0.0 {
                return Err("invalid asset default advance");
            }
            if profile_indices.insert(id.clone(), index as u8).is_some() {
                return Err("duplicate asset font profile");
            }
            profiles.push(AssetWidthProfile {
                sha3,
                default_width_em,
            });
        }

        let codepoint_count = take_u32(&mut input)? as usize;
        let mut widths = HashMap::with_capacity(codepoint_count);
        let mut codepoint = 0u32;
        for _ in 0..codepoint_count {
            codepoint = codepoint
                .checked_add(take_varint(&mut input)?)
                .ok_or("missing-width codepoint overflow")?;
            let width_count = take_u8(&mut input)? as usize;
            let mut entries = Vec::with_capacity(width_count);
            for _ in 0..width_count {
                let width = f32::from_bits(take_u32(&mut input)?);
                let font_mask = take_u64(&mut input)?;
                if !width.is_finite() || width < 0.0 || font_mask == 0 {
                    return Err("invalid missing-width entry");
                }
                entries.push((font_mask, width));
            }
            if widths.insert(codepoint, entries).is_some() {
                return Err("duplicate missing-width codepoint");
            }
        }
        if !input.is_empty() {
            return Err("trailing bytes in missing-width table");
        }
        Ok(Self {
            profiles,
            profile_indices,
            widths,
        })
    }

    fn profile_index(&self, id: &str) -> Option<u8> {
        self.profile_indices.get(id).copied()
    }

    fn width(&self, codepoint: u32, font_index: u8) -> Option<f32> {
        self.widths
            .get(&codepoint)?
            .iter()
            .find_map(|(mask, width)| (mask & (1u64 << font_index) != 0).then_some(*width))
    }
}

fn missing_width_table() -> &'static MissingWidthTable {
    static TABLE: OnceLock<MissingWidthTable> = OnceLock::new();
    TABLE.get_or_init(|| {
        MissingWidthTable::parse(include_bytes!("missing_widths.bin"))
            .expect("embedded missing-width table must match its generated format")
    })
}

fn take<'a>(input: &mut &'a [u8], length: usize) -> Result<&'a [u8], &'static str> {
    if input.len() < length {
        return Err("truncated missing-width table");
    }
    let (prefix, suffix) = input.split_at(length);
    *input = suffix;
    Ok(prefix)
}

fn take_u8(input: &mut &[u8]) -> Result<u8, &'static str> {
    Ok(take(input, 1)?[0])
}

fn take_u16(input: &mut &[u8]) -> Result<u16, &'static str> {
    Ok(u16::from_le_bytes(
        take(input, 2)?.try_into().map_err(|_| "invalid u16")?,
    ))
}

fn take_u32(input: &mut &[u8]) -> Result<u32, &'static str> {
    Ok(u32::from_le_bytes(
        take(input, 4)?.try_into().map_err(|_| "invalid u32")?,
    ))
}

fn take_u64(input: &mut &[u8]) -> Result<u64, &'static str> {
    Ok(u64::from_le_bytes(
        take(input, 8)?.try_into().map_err(|_| "invalid u64")?,
    ))
}

fn take_varint(input: &mut &[u8]) -> Result<u32, &'static str> {
    let mut value = 0u32;
    for shift in (0..35).step_by(7) {
        let byte = take_u8(input)?;
        value |= u32::from(byte & 0x7F)
            .checked_shl(shift)
            .ok_or("invalid missing-width varint")?;
        if byte & 0x80 == 0 {
            return Ok(value);
        }
    }
    Err("overlong missing-width varint")
}

fn sha3_256(input: &[u8]) -> [u8; 32] {
    Sha3_256::digest(input).into()
}

fn sha3_hex(digest: &[u8; 32]) -> String {
    use std::fmt::Write as _;

    let mut output = String::with_capacity(64);
    for byte in digest {
        write!(&mut output, "{byte:02x}").expect("writing to a String cannot fail");
    }
    output
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrimaryFontFamily {
    Serif,
    SansSerif,
    Script,
    None,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FontType {
    Normal,
    Script,
    Fallback,
    Base,
}

impl PrimaryFontFamily {
    pub fn from_option(value: Option<&str>) -> Self {
        match value {
            Some("serif") => Self::Serif,
            Some("sans-serif") => Self::SansSerif,
            Some("script") => Self::Script,
            _ => Self::None,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SourceFontStyle {
    pub bold: bool,
    pub italic: bool,
    pub monospaced: bool,
    pub serif: bool,
}

#[derive(Clone, Debug, Default)]
pub struct FontFamily {
    pub normal: Vec<Arc<FontFace>>,
    pub script: Vec<Arc<FontFace>>,
    pub fallback: Vec<Arc<FontFace>>,
    pub base: Vec<Arc<FontFace>>,
}

/// A document/run-scoped mapper. All contained data is immutable and shareable
/// across threads. Cache policy is intentionally left to callers; this mapper
/// has no cache whose state could affect a decision.
#[derive(Clone, Debug)]
pub struct FontMapper {
    primary: PrimaryFontFamily,
    normal: Vec<Arc<FontFace>>,
    script: Vec<Arc<FontFace>>,
    fallback: Vec<Arc<FontFace>>,
    base_group: Vec<Arc<FontFace>>,
    all_fonts: Vec<Arc<FontFace>>,
    by_id: HashMap<String, Arc<FontFace>>,
    base: Arc<FontFace>,
}

impl FontMapper {
    pub fn new(family: FontFamily, primary: PrimaryFontFamily) -> Result<Self, FontError> {
        let mut all_fonts = Vec::new();
        let mut by_id = HashMap::new();
        for font in family
            .normal
            .iter()
            .chain(&family.script)
            .chain(&family.fallback)
            .chain(&family.base)
        {
            if !by_id.contains_key(font.id()) {
                by_id.insert(font.id().to_owned(), Arc::clone(font));
                all_fonts.push(Arc::clone(font));
            }
        }
        let base_id = family.base.first().ok_or(FontError::InvalidFont)?.id();
        let base = by_id.get(base_id).cloned().ok_or(FontError::InvalidFont)?;
        by_id.insert("base".to_owned(), Arc::clone(&base));
        Ok(Self {
            primary,
            normal: family.normal,
            script: family.script,
            fallback: family.fallback,
            base_group: vec![Arc::clone(&base)],
            all_fonts,
            by_id,
            base,
        })
    }

    pub fn has_char(&self, text: &str) -> bool {
        let mut chars = text.chars();
        let Some(ch) = chars.next() else { return false };
        if chars.next().is_some() {
            return false;
        }
        self.all_fonts
            .iter()
            .any(|font| font.has_glyph(ch as u32) != 0)
    }

    pub fn map(&self, source: SourceFontStyle, ch: char) -> Option<Arc<FontFace>> {
        let (bold, mut italic, _monospaced, mut serif) =
            (source.bold, source.italic, source.monospaced, source.serif);
        match self.primary {
            PrimaryFontFamily::Serif => serif = true,
            PrimaryFontFamily::SansSerif => serif = false,
            PrimaryFontFamily::Script => {
                serif = false;
                italic = true;
            }
            PrimaryFontFamily::None => {}
        }
        let cp = ch as u32;
        if italic {
            if let Some(font) =
                self.map_in_type_with_flags(FontType::Script, bold, italic, serif, cp)
            {
                return Some(font);
            }
        }
        if italic {
            if let Some(font) = self.script.iter().find(|font| font.has_glyph(cp) != 0) {
                return Some(Arc::clone(font));
            }
        }
        if let Some(font) = self.map_in_type_with_flags(FontType::Normal, bold, italic, serif, cp) {
            return Some(font);
        }
        if let Some(font) = self.map_in_type_with_flags(FontType::Fallback, bold, italic, serif, cp)
        {
            return Some(font);
        }
        self.fallback
            .iter()
            .find(|font| font.has_glyph(cp) != 0)
            .cloned()
    }

    /// Direct counterpart of Python `map_in_type`; normal/fallback mapping
    /// intentionally ignores the italic and monospaced flags as 0.6.4 does.
    pub fn map_in_type(
        &self,
        source: SourceFontStyle,
        ch: char,
        font_type: FontType,
    ) -> Option<Arc<FontFace>> {
        if font_type == FontType::Script && !source.italic {
            return None;
        }
        self.map_in_type_with_flags(
            font_type,
            source.bold,
            source.italic,
            source.serif,
            ch as u32,
        )
    }

    fn map_in_type_with_flags(
        &self,
        font_type: FontType,
        bold: bool,
        _italic: bool,
        serif: bool,
        codepoint: u32,
    ) -> Option<Arc<FontFace>> {
        let fonts = match font_type {
            FontType::Normal => &self.normal,
            FontType::Script => &self.script,
            FontType::Fallback => &self.fallback,
            FontType::Base => &self.base_group,
        };
        fonts
            .iter()
            .find(|font| {
                font.has_glyph(codepoint) != 0
                    && bold == font.traits.bold
                    && serif == font.id().to_ascii_lowercase().contains("serif")
            })
            .cloned()
    }

    pub fn font_by_id(&self, id: &str) -> Option<Arc<FontFace>> {
        self.by_id.get(id).cloned()
    }

    pub fn base_font(&self) -> Arc<FontFace> {
        Arc::clone(&self.base)
    }

    /// Resolves resources needed by the later writer/IR `add_font` operation.
    /// This crate does not mutate a PDF or IR.
    pub fn font_install_plan<I, S>(&self, font_ids: I) -> Result<FontInstallPlan, FontError>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let mut fonts = Vec::new();
        let mut seen = std::collections::HashSet::new();
        for font_id in font_ids {
            let font_id = font_id.as_ref();
            if seen.insert(font_id.to_owned()) {
                fonts.push(
                    self.font_by_id(font_id)
                        .ok_or_else(|| FontError::UnknownFontId(font_id.to_owned()))?,
                );
            }
        }
        Ok(FontInstallPlan { fonts })
    }
}

pub struct FontInstallPlan {
    fonts: Vec<Arc<FontFace>>,
}

impl FontInstallPlan {
    pub fn fonts(&self) -> &[Arc<FontFace>] {
        &self.fonts
    }
}
