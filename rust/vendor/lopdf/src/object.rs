use crate::encodings;
use crate::encodings::cmap::ToUnicodeCMap;
use crate::encodings::{Differences, Encoding, Glyph};
use crate::error::DecompressError;
use crate::{Document, Error, Result};
use indexmap::IndexMap;
use log::warn;
use std::cmp::max;
use std::collections::HashSet;
use std::fmt;
use std::str;

/// Object identifier consists of two parts: object number and generation number.
pub type ObjectId = (u32, u16);

/// Dictionary object.
#[derive(Clone, Default, PartialEq)]
// Keep the map header out of every Object slot (including numeric array items).
// This private indirection preserves the public dictionary/stream representation.
pub struct Dictionary(Box<IndexMap<Vec<u8>, Object>>);

/// Stream object
/// Warning - all streams must be indirect objects, while
/// the stream dictionary may be a direct object
#[derive(Debug, Clone, PartialEq)]
pub struct Stream {
    /// Associated stream dictionary
    pub dict: Dictionary,
    /// Contents of the stream in bytes
    pub content: StreamContent,
    /// Can the stream be compressed by the `Document::compress()` function?
    /// Font streams may not be compressed, for example
    pub allows_compression: bool,
    /// Stream data's position in PDF file.
    pub start_position: Option<usize>,
}

pub use crate::stream_content::StreamContent;

/// Basic PDF object types defined in an enum.
#[derive(Clone, PartialEq)]
pub enum Object {
    Null,
    Boolean(bool),
    Integer(i64),
    Real(f32),
    Name(Vec<u8>),
    String(Vec<u8>, StringFormat),
    Array(Vec<Object>),
    Dictionary(Dictionary),
    Stream(Box<Stream>),
    Reference(ObjectId),
}

/// String objects can be written in two formats.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum StringFormat {
    #[default]
    Literal,
    Hexadecimal,
}

impl From<bool> for Object {
    fn from(value: bool) -> Self {
        Object::Boolean(value)
    }
}

impl From<i64> for Object {
    fn from(number: i64) -> Self {
        Object::Integer(number)
    }
}

macro_rules! from_smaller_ints {
	($( $Int: ty )+) => {
		$(
			impl From<$Int> for Object {
				fn from(number: $Int) -> Self {
					Object::Integer(i64::from(number))
				}
			}
		)+
	}
}

from_smaller_ints! {
    i8 i16 i32
    u8 u16 u32
}

impl From<f64> for Object {
    fn from(number: f64) -> Self {
        Object::Real(number as f32)
    }
}

impl From<f32> for Object {
    fn from(number: f32) -> Self {
        Object::Real(number)
    }
}

impl From<String> for Object {
    fn from(name: String) -> Self {
        Object::Name(name.into_bytes())
    }
}

impl<'a> From<&'a str> for Object {
    fn from(name: &'a str) -> Self {
        Object::Name(name.as_bytes().to_vec())
    }
}

impl From<Vec<Object>> for Object {
    fn from(array: Vec<Object>) -> Self {
        Object::Array(array)
    }
}

impl From<Dictionary> for Object {
    fn from(dict: Dictionary) -> Self {
        Object::Dictionary(dict)
    }
}

impl From<Stream> for Object {
    fn from(stream: Stream) -> Self {
        Object::Stream(Box::new(stream))
    }
}

impl From<ObjectId> for Object {
    fn from(id: ObjectId) -> Self {
        Object::Reference(id)
    }
}

impl Object {
    pub fn string_literal<S: Into<Vec<u8>>>(s: S) -> Self {
        Object::String(s.into(), StringFormat::Literal)
    }

    pub fn is_null(&self) -> bool {
        matches!(*self, Object::Null)
    }

    pub fn as_bool(&self) -> Result<bool> {
        match self {
            Object::Boolean(value) => Ok(*value),
            _ => Err(Error::ObjectType {
                expected: "Boolean",
                found: self.enum_variant(),
            }),
        }
    }

    pub fn as_i64(&self) -> Result<i64> {
        match self {
            Object::Integer(value) => Ok(*value),
            _ => Err(Error::ObjectType {
                expected: "Integer",
                found: self.enum_variant(),
            }),
        }
    }

    pub fn as_f32(&self) -> Result<f32> {
        match self {
            Object::Real(value) => Ok(*value),
            _ => Err(Error::ObjectType {
                expected: "Real",
                found: self.enum_variant(),
            }),
        }
    }

    /// Get the object value as a float.
    /// Unlike [`Object::as_f32`] this will also cast an Integer to a Real.
    pub fn as_float(&self) -> Result<f32> {
        match self {
            Object::Integer(value) => Ok(*value as f32),
            Object::Real(value) => Ok(*value),
            _ => Err(Error::ObjectType {
                expected: "Integer or Real",
                found: self.enum_variant(),
            }),
        }
    }

    pub fn as_name(&self) -> Result<&[u8]> {
        match self {
            Object::Name(name) => Ok(name),
            _ => Err(Error::ObjectType {
                expected: "Name",
                found: self.enum_variant(),
            }),
        }
    }

    pub fn as_str(&self) -> Result<&[u8]> {
        match self {
            Object::String(string, _) => Ok(string),
            _ => Err(Error::ObjectType {
                expected: "String",
                found: self.enum_variant(),
            }),
        }
    }

    pub fn as_str_mut(&mut self) -> Result<&mut Vec<u8>> {
        match self {
            Object::String(string, _) => Ok(string),
            _ => Err(Error::ObjectType {
                expected: "String",
                found: self.enum_variant(),
            }),
        }
    }

    pub fn as_reference(&self) -> Result<ObjectId> {
        match self {
            Object::Reference(id) => Ok(*id),
            _ => Err(Error::ObjectType {
                expected: "Reference",
                found: self.enum_variant(),
            }),
        }
    }

    pub fn as_array(&self) -> Result<&Vec<Object>> {
        match self {
            Object::Array(arr) => Ok(arr),
            _ => Err(Error::ObjectType {
                expected: "Array",
                found: self.enum_variant(),
            }),
        }
    }

    pub fn as_array_mut(&mut self) -> Result<&mut Vec<Object>> {
        match self {
            Object::Array(arr) => Ok(arr),
            _ => Err(Error::ObjectType {
                expected: "Array",
                found: self.enum_variant(),
            }),
        }
    }

    pub fn as_dict(&self) -> Result<&Dictionary> {
        match self {
            Object::Dictionary(dict) => Ok(dict),
            _ => Err(Error::ObjectType {
                expected: "Dictionary",
                found: self.enum_variant(),
            }),
        }
    }

    pub fn as_dict_mut(&mut self) -> Result<&mut Dictionary> {
        match self {
            Object::Dictionary(dict) => Ok(dict),
            _ => Err(Error::ObjectType {
                expected: "Dictionary",
                found: self.enum_variant(),
            }),
        }
    }

    pub fn as_stream(&self) -> Result<&Stream> {
        match self {
            Object::Stream(stream) => Ok(stream),
            _ => Err(Error::ObjectType {
                expected: "Stream",
                found: self.enum_variant(),
            }),
        }
    }

    pub fn as_stream_mut(&mut self) -> Result<&mut Stream> {
        match self {
            Object::Stream(stream) => Ok(stream),
            _ => Err(Error::ObjectType {
                expected: "Stream",
                found: self.enum_variant(),
            }),
        }
    }

    // TODO: maybe remove
    pub fn type_name(&self) -> Result<&[u8]> {
        match self {
            Object::Dictionary(dict) => dict.get_type(),
            Object::Stream(stream) => stream.dict.get_type(),
            obj => Err(Error::ObjectType {
                expected: "Dictionary or Stream",
                found: obj.enum_variant(),
            }),
        }
    }

    pub fn enum_variant(&self) -> &'static str {
        match self {
            Object::Null => "Null",
            Object::Boolean(_) => "Boolean",
            Object::Integer(_) => "Integer",
            Object::Real(_) => "Real",
            Object::Name(_) => "Name",
            Object::String(_, _) => "String",
            Object::Array(_) => "Array",
            Object::Dictionary(_) => "Dictionary",
            Object::Stream(_) => "Stream",
            Object::Reference(_) => "Reference",
        }
    }
}

impl fmt::Debug for Object {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Object::Null => write!(f, "Null"),
            Object::Boolean(value) => write!(f, "{value}"),
            Object::Integer(value) => write!(f, "{value}"),
            Object::Real(value) => write!(f, "{value}"),
            Object::Name(name) => write!(f, "/{}", String::from_utf8_lossy(name)),
            Object::String(text, StringFormat::Literal) => write!(f, "({})", String::from_utf8_lossy(text)),
            Object::String(text, StringFormat::Hexadecimal) => {
                write!(f, "<")?;
                for b in text {
                    write!(f, "{b:02x}")?
                }
                write!(f, ">")
            }
            Object::Array(array) => {
                let items = array.iter().map(|item| format!("{item:?}")).collect::<Vec<String>>();
                write!(f, "[{}]", items.join(" "))
            }
            Object::Dictionary(dict) => write!(f, "{dict:?}"),
            Object::Stream(stream) => write!(f, "{:?}stream...endstream", stream.dict),
            Object::Reference(id) => write!(f, "{} {} R", id.0, id.1),
        }
    }
}

impl Dictionary {
    pub fn new() -> Dictionary {
        Dictionary(Box::new(IndexMap::new()))
    }

    pub fn has(&self, key: &[u8]) -> bool {
        self.0.contains_key(key)
    }

    pub fn get(&self, key: &[u8]) -> Result<&Object> {
        self.0
            .get(key)
            .ok_or(Error::DictKey(String::from_utf8_lossy(key).to_string()))
    }

    /// Extract object from dictionary, dereferencing
    /// the object if it is a reference.
    pub fn get_deref<'a>(&'a self, key: &[u8], doc: &'a Document) -> Result<&'a Object> {
        doc.dereference(self.get(key)?).map(|(_, object)| object)
    }

    pub fn get_mut(&mut self, key: &[u8]) -> Result<&mut Object> {
        self.0
            .get_mut(key)
            .ok_or(Error::DictKey(String::from_utf8_lossy(key).to_string()))
    }

    pub fn set<K, V>(&mut self, key: K, value: V)
    where
        K: Into<Vec<u8>>,
        V: Into<Object>,
    {
        self.0.insert(key.into(), value.into());
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.len() == 0
    }

    pub fn remove(&mut self, key: &[u8]) -> Option<Object> {
        self.0.swap_remove(key)
    }

    pub fn has_type(&self, type_name: &[u8]) -> bool {
        self.get(b"Type").and_then(|s| s.as_name()).ok() == Some(type_name)
    }

    pub fn get_type(&self) -> Result<&[u8]> {
        self.get(b"Type")
            .and_then(Object::as_name)
            .or_else(|_| self.get(b"Linearized").and(Ok(b"Linearized")))
    }

    pub fn iter(&'_ self) -> indexmap::map::Iter<'_, Vec<u8>, Object> {
        self.0.iter()
    }

    pub fn iter_mut(&'_ mut self) -> indexmap::map::IterMut<'_, Vec<u8>, Object> {
        self.0.iter_mut()
    }

    pub fn get_font_encoding<'a>(&'a self, doc: &'a Document) -> Result<Encoding<'a>> {
        self.get_font_encoding_inner(doc, None)
    }

    /// Resolve this font's encoding, bounding any decompression it performs (for
    /// example decoding a `/ToUnicode` CMap stream) to `max_decompressed_size`
    /// bytes. This is the decompression-bomb-safe counterpart to
    /// [`Dictionary::get_font_encoding`]: it is used by
    /// [`crate::Document::extract_text_with_limit`] so a crafted font stream
    /// cannot inflate without limit. Unlike other encoding errors (which fall
    /// back to standard encoding), a detected size-limit violation is propagated
    /// as [`DecompressError::MemoryLimitExceeded`].
    pub fn get_font_encoding_with_limit<'a>(
        &'a self, doc: &'a Document, max_decompressed_size: usize,
    ) -> Result<Encoding<'a>> {
        self.get_font_encoding_inner(doc, Some(max_decompressed_size))
    }

    fn get_font_encoding_inner<'a>(&'a self, doc: &'a Document, limit: Option<usize>) -> Result<Encoding<'a>> {
        if !self.has_type(b"Font") {
            return Err(Error::DictType {
                expected: "Font",
                found: String::from_utf8_lossy(self.get_type().unwrap_or(b"None")).to_string(),
            });
        }

        // Note: currently not all encodings are handled, not implemented:
        // - TrueType cmap tables
        // - DescendantFonts in CID-Keyed fonts
        // - Predefined CJK CMAP other than indicated in SimpleEncoding
        // - Deciding what should be the fallback font if no such encoding is defined in difference encoding (see Table
        //   114 in 9.6.6.1 General under `BaseEncoding`).
        let result = (|| {
            if let Ok(object) = self.get(b"Encoding") {
                return self.get_base_encoding(object, doc, limit);
            }

            if let Ok(stream) = self.get_deref(b"ToUnicode", doc).and_then(Object::as_stream) {
                return self.get_to_unicode_encoding(stream, limit);
            }

            Ok(Encoding::OneByteEncoding(&encodings::STANDARD_ENCODING))
        })();

        match result {
            Ok(encoding) => Ok(encoding),
            // A detected decompression bomb must surface, not be swallowed into a
            // fallback encoding — otherwise the bounded caller's guard is silently
            // defeated. Every other encoding error stays lenient, as before.
            Err(err @ Error::Decompress(DecompressError::MemoryLimitExceeded { .. })) => Err(err),
            Err(err) => {
                warn!(
                    "Could not parse the encoding, error: {err:#?}\nFont: {self:#?}. Using standard encoding as a fallback!"
                );
                Ok(Encoding::OneByteEncoding(&encodings::STANDARD_ENCODING))
            }
        }
    }

    /// Get a simple encoding from the /Encoding entry of a font dictionary.
    fn get_base_encoding<'a>(
        &'a self, mut object: &'a Object, doc: &'a Document, limit: Option<usize>,
    ) -> Result<Encoding<'a>> {
        // Set of visited to detect circular references.
        let mut visited = HashSet::new();

        loop {
            match *object {
                Object::Name(ref name) => {
                    return self.base_encoding(doc, name, limit);
                }
                Object::Reference(id) => {
                    if !visited.insert(id) {
                        return Err(Error::ReferenceCycle(id));
                    }

                    let Ok(o) = doc.get_object(id) else {
                        return Err(Error::ObjectNotFound(id));
                    };

                    object = o;
                }
                Object::Dictionary(ref dict) => {
                    let ty = dict.get(b"Type")?.as_name()?;

                    match ty {
                        b"Encoding" => {
                            let mut base = None;

                            if let Ok(base_encoding) = dict.get(b"BaseEncoding")
                                && let Ok(name) = base_encoding.as_name()
                            {
                                base = Some(self.base_encoding(doc, name, limit)?);
                            }

                            let base = match base {
                                Some(base) => base,
                                None => Encoding::OneByteEncoding(&encodings::STANDARD_ENCODING),
                            };

                            let differences = dict.get(b"Differences")?.as_array()?;
                            let differences = self.differences(base, differences)?;
                            return Ok(Encoding::Differences(differences));
                        }
                        _ => {
                            return Err(Error::ObjectType {
                                expected: "Encoding Dictionary",
                                found: "Dictionary with Type other than /Encoding",
                            });
                        }
                    }
                }
                ref object => {
                    return Err(Error::ObjectType {
                        expected: "Name or Reference or Dictionary",
                        found: object.enum_variant(),
                    });
                }
            }
        }
    }

    fn base_encoding<'a>(&'a self, doc: &'a Document, name: &'a [u8], limit: Option<usize>) -> Result<Encoding<'a>> {
        match name {
            b"StandardEncoding" => Ok(Encoding::OneByteEncoding(&encodings::STANDARD_ENCODING)),
            b"MacRomanEncoding" => Ok(Encoding::OneByteEncoding(&encodings::MAC_ROMAN_ENCODING)),
            b"MacExpertEncoding" => Ok(Encoding::OneByteEncoding(&encodings::MAC_EXPERT_ENCODING)),
            b"WinAnsiEncoding" => Ok(Encoding::OneByteEncoding(&encodings::WIN_ANSI_ENCODING)),
            b"PDFDocEncoding" => {
                log::warn!("PDFDocEncoding is not a valid character encoding for a font");
                Ok(Encoding::OneByteEncoding(&encodings::PDF_DOC_ENCODING))
            }
            b"Identity-H" | b"Identity-V" => {
                let stream = self.get_deref(b"ToUnicode", doc)?.as_stream()?;
                self.get_to_unicode_encoding(stream, limit)
            }
            name => Ok(Encoding::SimpleEncoding(name)),
        }
    }

    fn differences<'a>(&'a self, base: Encoding<'a>, array: &[Object]) -> Result<Differences<'a>> {
        let mut map = IndexMap::new();
        let mut inverse = IndexMap::new();
        let mut current_code = 0;

        for obj in array {
            match *obj {
                Object::Integer(code) => {
                    if !(0..=255).contains(&code) {
                        return Err(Error::InvalidEncodingDifferenceCode { code });
                    }

                    current_code = code as u8;
                }
                Object::Name(ref name) => {
                    let Some(glyph) = Glyph::from_name(name) else {
                        return Err(Error::InvalidEncodingDifferenceGlyph {
                            name: String::from_utf8_lossy(name).into_owned(),
                        });
                    };

                    map.insert(current_code, glyph);
                    inverse.insert(glyph, current_code);
                    current_code = current_code.wrapping_add(1);
                }
                _ => {
                    return Err(Error::ObjectType {
                        expected: "Integer or Name",
                        found: obj.enum_variant(),
                    });
                }
            }
        }

        Ok(Differences {
            base: Box::new(base),
            map,
            inverse,
        })
    }

    fn get_to_unicode_encoding(&'_ self, stream: &Stream, limit: Option<usize>) -> Result<Encoding<'_>> {
        let content = match limit {
            Some(max) => stream.get_plain_content_with_limit(max)?,
            None => stream.get_plain_content()?,
        };
        let cmap = ToUnicodeCMap::parse(content)?;
        Ok(Encoding::UnicodeMapEncoding(cmap))
    }

    pub fn extend(&mut self, other: &Dictionary) {
        let keep_both_objects =
            |new_dict: &mut IndexMap<Vec<u8>, Object>, key: &Vec<u8>, value: &Object, old_value: Object| {
                let mut final_array;

                match value {
                    Object::Array(array) => {
                        final_array = Vec::with_capacity(array.len() + 1);
                        final_array.push(old_value);
                        final_array.extend(array.to_owned());
                    }
                    _ => {
                        final_array = vec![value.to_owned(), old_value];
                    }
                }

                new_dict.insert(key.to_owned(), Object::Array(final_array));
            };

        let mut new_dict = std::mem::take(&mut self.0);
        new_dict.reserve_exact(other.0.len());

        for (key, value) in other.0.iter() {
            if let Some(old_value) = new_dict.get(key) {
                let old_value = old_value.to_owned();
                match (&old_value, value) {
                    (Object::Dictionary(old_dict), Object::Dictionary(dict)) => {
                        let mut replaced_dict = old_dict.to_owned();
                        replaced_dict.extend(dict);
                        new_dict.insert(key.to_owned(), Object::Dictionary(replaced_dict));
                    }
                    (Object::Array(old_array), Object::Array(array)) => {
                        let mut replaced_array = old_array.to_owned();
                        replaced_array.extend(array.to_owned());
                        new_dict.insert(key.to_owned(), Object::Array(replaced_array));
                    }
                    (Object::Integer(old_id), Object::Integer(id)) => {
                        let array = vec![Object::Integer(*old_id), Object::Integer(*id)];
                        new_dict.insert(key.to_owned(), Object::Array(array));
                    }
                    (Object::Real(old_id), Object::Real(id)) => {
                        let array = vec![Object::Real(*old_id), Object::Real(*id)];
                        new_dict.insert(key.to_owned(), Object::Array(array));
                    }
                    (Object::String(old_ids, old_format), Object::String(ids, format)) => {
                        let array = vec![
                            Object::String(old_ids.to_owned(), old_format.to_owned()),
                            Object::String(ids.to_owned(), format.to_owned()),
                        ];
                        new_dict.insert(key.to_owned(), Object::Array(array));
                    }
                    (Object::Reference(old_object_id), Object::Reference(object_id)) => {
                        let array = vec![Object::Reference(*old_object_id), Object::Reference(*object_id)];
                        new_dict.insert(key.to_owned(), Object::Array(array));
                    }
                    (Object::Null, _) | (Object::Boolean(_), _) | (Object::Name(_), _) | (Object::Stream(_), _) => {
                        new_dict.insert(key.to_owned(), old_value);
                    }
                    (_, _) => keep_both_objects(&mut new_dict, key, value, old_value),
                }
            } else {
                new_dict.insert(key.to_owned(), value.to_owned());
            }
        }

        self.0 = new_dict;
    }

    /// Return a reference to the inner  [`IndexMap`].
    pub fn as_hashmap(&self) -> &IndexMap<Vec<u8>, Object> {
        &self.0
    }

    /// Return a mut reference to the inner [`IndexMap`].
    pub fn as_hashmap_mut(&mut self) -> &mut IndexMap<Vec<u8>, Object> {
        &mut self.0
    }
}

#[macro_export]
macro_rules! dictionary {
	() => {
		$crate::Dictionary::new()
	};
	($( $key: expr => $value: expr ),+ ,) => {
		dictionary!( $($key => $value),+ )
	};
	($( $key: expr => $value: expr ),*) => {{
		let mut dict = $crate::Dictionary::new();
		$(
			dict.set($key, $value);
		)*
		dict
	}}
}

impl fmt::Debug for Dictionary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let entries = self
            .into_iter()
            .map(|(key, value)| format!("/{} {:?}", String::from_utf8_lossy(key), value))
            .collect::<Vec<String>>();
        write!(f, "<<{}>>", entries.concat())
    }
}

impl IntoIterator for Dictionary {
    type Item = (Vec<u8>, Object);
    type IntoIter = indexmap::map::IntoIter<Vec<u8>, Object>;

    fn into_iter(self) -> Self::IntoIter {
        (*self.0).into_iter()
    }
}

impl<'a> IntoIterator for &'a Dictionary {
    type Item = (&'a Vec<u8>, &'a Object);
    type IntoIter = indexmap::map::Iter<'a, Vec<u8>, Object>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

impl<'a> IntoIterator for &'a mut Dictionary {
    type Item = (&'a Vec<u8>, &'a mut Object);
    type IntoIter = indexmap::map::IterMut<'a, Vec<u8>, Object>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.iter_mut()
    }
}

use std::iter::FromIterator;
use crate::resource::{Policy, Limit};
impl<K: Into<Vec<u8>>> FromIterator<(K, Object)> for Dictionary {
    fn from_iter<I: IntoIterator<Item = (K, Object)>>(iter: I) -> Self {
        let mut dict = Dictionary::new();
        for (k, v) in iter {
            dict.set(k, v);
        }
        dict
    }
}

/// A [`std::io::Write`] adapter that appends to a `Vec<u8>` but refuses to grow
/// it past `limit` bytes. Used to bound decoders (e.g. LZW) that write into a
/// sink rather than being read from, so a decompression bomb cannot allocate an
/// unbounded amount of memory. On overflow it fills up to exactly `limit` bytes
/// and then returns an error, so the caller can detect the overflow by length.
struct LimitedWriter<'a> {
    inner: &'a mut Vec<u8>,
    limit: usize,
    policy: Option<Policy>,
}

impl<'a> LimitedWriter<'a> {
    fn new(inner: &'a mut Vec<u8>, limit: usize) -> Self {
        LimitedWriter { inner, limit, policy: None }
    }
}

impl std::io::Write for LimitedWriter<'_> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        if let Some(policy) = &self.policy {
            policy.append(self.inner.len(), buf.len()).map_err(crate::resource::into_io)?;
        }
        let remaining = self.limit.saturating_sub(self.inner.len());
        if buf.len() > remaining {
            // Fill up to the limit so the caller's `len() > max` check trips,
            // then signal that no more may be written.
            self.inner.extend_from_slice(&buf[..remaining]);
            return Err(std::io::Error::new(
                std::io::ErrorKind::WriteZero,
                "decompression output exceeded limit",
            ));
        }
        self.inner.extend_from_slice(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl Stream {

    #[cfg(test)]
    fn decode_ascii85(input: &[u8], limit: Option<usize>) -> Result<Vec<u8>> { Self::decode_ascii85_policy(input, limit, None) }
    #[cfg(test)]
    fn decode_ascii_hex(input: &[u8], limit: Option<usize>) -> Result<Vec<u8>> { Self::decode_ascii_hex_policy(input, limit, None) }
    #[cfg(test)]
    fn decode_run_length(input: &[u8], limit: Option<usize>) -> Result<Vec<u8>> { Self::decode_run_length_policy(input, limit, None) }
    #[cfg(test)]
    fn decompress_zlib(input: &[u8], params: Option<&Dictionary>, limit: Option<usize>) -> Result<Vec<u8>> { Self::decompress_zlib_policy(input, params, limit, None) }
    #[cfg(test)]
    fn decompress_predictor(data: Vec<u8>, params: Option<&Dictionary>) -> Result<Vec<u8>> { Self::decompress_predictor_policy(data, params, None) }

    pub fn new(dict: Dictionary, content: Vec<u8>) -> Stream {
        Self::with_content(dict, content.into())
    }

    pub(crate) fn with_content(mut dict: Dictionary, content: StreamContent) -> Stream {
        dict.set("Length", content.len() as i64);
        Stream {
            dict,
            content,
            allows_compression: true,
            start_position: None,
        }
    }

    pub fn with_position(dict: Dictionary, position: usize) -> Stream {
        Stream {
            dict,
            content: vec![].into(),
            allows_compression: true,
            start_position: Some(position),
        }
    }

    /// Default is that the stream may be compressed. On font streams,
    /// set this to false, otherwise the font will be corrupt
    #[inline]
    pub fn with_compression(mut self, allows_compression: bool) -> Stream {
        self.allows_compression = allows_compression;
        self
    }

    pub fn filters(&self) -> Result<Vec<&[u8]>> {
        let filter = self.dict.get(b"Filter")?;

        if let Ok(name) = filter.as_name() {
            Ok(vec![name])
        } else if let Ok(names) = filter.as_array() {
            names.iter().map(Object::as_name).collect()
        } else {
            Err(Error::ObjectType {
                expected: "Name or Array",
                found: filter.enum_variant(),
            })
        }
    }

    pub fn set_content(&mut self, content: Vec<u8>) {
        self.content = content.into();
        self.dict.set("Length", self.content.len() as i64);
    }

    pub fn set_plain_content(&mut self, content: Vec<u8>) {
        self.dict.remove(b"DecodeParms");
        self.dict.remove(b"Filter");
        self.dict.set("Length", content.len() as i64);
        self.content = content.into();
    }

    pub fn decompressed_content_with_policy(&self, policy: Option<&Policy>, legacy_limit: Option<usize>) -> Result<Vec<u8>> {
        self.decode_filters_policy(legacy_limit, policy)
    }

    pub fn get_plain_content_with_policy(&self, policy: Option<&Policy>, legacy_limit: Option<usize>) -> Result<Vec<u8>> {
        match self.filters() {
            Ok(filters) if !filters.is_empty() => self.decode_filters_policy(legacy_limit, policy),
            _ => match legacy_limit {
                Some(max) if self.content.len() > max => Err(DecompressError::MemoryLimitExceeded { limit: max }.into()),
                _ => Ok(self.content.to_vec()),
            }
        }
    }

    pub fn get_plain_content(&self) -> Result<Vec<u8>> {
        match self.filters() {
            Ok(vec) if !vec.is_empty() => self.decompressed_content(),
            _ => Ok(self.content.to_vec()),
        }
    }

    /// Bomb-safe counterpart to [`Stream::get_plain_content`]: decode the stream,
    /// rejecting it with [`DecompressError::MemoryLimitExceeded`] if the output
    /// would exceed `max_output` bytes. An uncompressed stream whose raw content
    /// already exceeds the limit is rejected too.
    pub fn get_plain_content_with_limit(&self, max_output: usize) -> Result<Vec<u8>> {
        match self.filters() {
            Ok(vec) if !vec.is_empty() => self.decompressed_content_with_limit(max_output),
            _ => {
                if self.content.len() > max_output {
                    return Err(DecompressError::MemoryLimitExceeded { limit: max_output }.into());
                }
                Ok(self.content.to_vec())
            }
        }
    }

    pub fn compress(&mut self) -> Result<()> {
        use flate2::Compression;
        use flate2::write::ZlibEncoder;
        use std::io::prelude::*;

        if self.dict.get(b"Filter").is_err() {
            let mut encoder = ZlibEncoder::new(Vec::new(), Compression::best());
            encoder.write_all(self.content.as_slice())?;
            let compressed = encoder.finish()?;
            if compressed.len() + 19 < self.content.len() {
                self.dict.set("Filter", "FlateDecode");
                self.set_content(compressed);
            }
        }
        Ok(())
    }

    /// Decode this stream's content, applying its filters in order.
    ///
    /// # Warning: unbounded output
    ///
    /// This method places **no limit** on the size of the decompressed output.
    /// A small compressed stream can inflate to an enormous size (a
    /// "decompression bomb"), so calling this on untrusted input can exhaust
    /// all available memory. When processing PDFs from an untrusted source,
    /// prefer [`Stream::decompressed_content_with_limit`] or
    /// [`Stream::decompress_to_writer`], and load documents with
    /// [`crate::LoadOptions::max_decompressed_size`] set.
    pub fn decompressed_content(&self) -> Result<Vec<u8>> {
        self.decode_filters_policy(None, None)
    }

    /// Decode this stream's content, rejecting the stream with
    /// [`DecompressError::MemoryLimitExceeded`] if the decoded output would
    /// exceed `max_output` bytes.
    ///
    /// This is the bomb-safe counterpart to [`Stream::decompressed_content`].
    /// Each filter in the chain is bounded individually, so a stream with nested
    /// filters (e.g. `/Filter [FlateDecode FlateDecode]`) can never allocate
    /// more than roughly `max_output` bytes per layer before being rejected,
    /// rather than expanding without limit.
    pub fn decompressed_content_with_limit(&self, max_output: usize) -> Result<Vec<u8>> {
        self.decode_filters_policy(Some(max_output), None)
    }

    /// Decode this stream's content into a caller-provided writer, rejecting the
    /// stream if the decoded output would exceed `max_output` bytes.
    ///
    /// The result is decoded with the same bomb-safe bound as
    /// [`Stream::decompressed_content_with_limit`] (internal buffering is capped
    /// at roughly `max_output` bytes) and then written to `writer`, which lets
    /// callers direct the output straight into a file or a fixed-capacity buffer
    /// of their choosing. Returns the number of bytes written on success, or
    /// [`DecompressError::MemoryLimitExceeded`] if the limit would be exceeded.
    pub fn decompress_to_writer<W: std::io::Write>(&self, writer: &mut W, max_output: usize) -> Result<usize> {
        let data = self.decompressed_content_with_limit(max_output)?;
        writer.write_all(&data)?;
        Ok(data.len())
    }

    /// Shared decoder for [`Stream::decompressed_content`] and its bounded
    /// variants. `limit` is `None` to decode without a size limit, or
    /// `Some(max)` to cap the decoded output at `max` bytes per filter layer.
    fn decode_filters_policy(&self, limit: Option<usize>, policy: Option<&Policy>) -> Result<Vec<u8>> {
        let params = self.dict.get(b"DecodeParms").and_then(Object::as_dict).ok();
        let filters = match self.filters() {
            Ok(f) => f,
            // No /Filter key means the stream is uncompressed. The raw content is
            // already in memory, but still honor the caller's limit.
            Err(_) => {
                if let Some(max) = limit
                    && self.content.len() > max
                {
                    return Err(DecompressError::MemoryLimitExceeded { limit: max }.into());
                }
                return Ok(self.content.to_vec());
            }
        };

        let mut input = self.content.as_slice();
        let mut output = vec![];

        if let Some(policy) = policy { policy.check(Limit::FilterLayers, filters.len())?; }
        // Filters are in decoding order.
        for filter in filters {
            output = match filter {
                b"FlateDecode" => Self::decompress_zlib_policy(input, params, limit, policy)?,
                b"BrotliDecode" => Self::decompress_brotli_policy(input, limit, policy)?,
                b"LZWDecode" => Self::decompress_lzw_policy(input, params, limit, policy)?,
                b"ASCII85Decode" => Self::decode_ascii85_policy(input, limit, policy)?,
                b"ASCIIHexDecode" => Self::decode_ascii_hex_policy(input, limit, policy)?,
                b"RunLengthDecode" => Self::decode_run_length_policy(input, limit, policy)?,
                _ => return Err(Error::Unimplemented("decompression algorithms")),
            };
            if let Some(max) = limit
                && output.len() > max
            {
                return Err(DecompressError::MemoryLimitExceeded { limit: max }.into());
            }
            input = &output;
        }
        Ok(output)
    }

    fn decompress_brotli_policy(input: &[u8], limit: Option<usize>, policy: Option<&Policy>) -> Result<Vec<u8>> {
        use brotli_decompressor::Decompressor;
        if let Some(policy) = policy {
            if let Some(bits) = crate::resource::brotli_window_bits(input) {
                policy.check(Limit::BrotliWindowBytes, 1usize << bits)?;
            }
        }

        // Unlike the Flate path there is no fallback (e.g. raw deflate), so a
        // truncated or corrupt stream fails hard instead of yielding partial
        // output — intentional for a new filter, not an oversight.
        let initial_capacity = match limit {
            Some(max) => input.len().saturating_mul(2).min(max.saturating_add(1)),
            None => input.len().saturating_mul(2),
        };
        let capacity = initial_capacity.min(8192).min(policy.filter(|p| !p.is_disabled())
            .map_or(usize::MAX, |p| p.value(Limit::DecodedStreamBytes)));
        let mut output = Vec::with_capacity(capacity);
        Self::read_capped(Decompressor::new(input, 4096), &mut output, limit, policy)
            .map_err(|err| {
                let err = Error::from_decode_io(err);
                if err.is_decode_limit() { err } else {
                    let text = match &err { Error::IO(e) => e.to_string(), _ => err.to_string() };
                    Error::InvalidStream(format!("Brotli decompression failed: {text}"))
                }
            })?;
        if let Some(max) = limit
            && output.len() > max
        {
            return Err(DecompressError::MemoryLimitExceeded { limit: max }.into());
        }
        // /DecodeParms predictors are deliberately NOT applied here. The PDF
        // Association's prototype files never pair /BrotliDecode with a
        // /Predictor (their Brotli xref streams store raw W entries), pdf.js
        // and pypdf ignore the parameters too; only MuPDF applies them.
        // Applying them would also corrupt [/BrotliDecode /FlateDecode]
        // chains, since the shared dict-form /DecodeParms is passed to every
        // filter and a predictor there belongs to the Flate layer.
        Ok(output)
    }

    /// Read `reader` to end, appending to `output`. When `limit` is `Some(max)`,
    /// at most `max + 1` bytes are read, so an oversized (bomb) stream is
    /// detected via the `> max` check by the caller without ever allocating the
    /// full decompressed output.
    fn check_decode_append(current: usize, additional: usize, limit: Option<usize>, policy: Option<&Policy>) -> Result<()> {
        if let Some(policy) = policy { policy.append(current, additional)?; }
        let next = current.checked_add(additional).ok_or_else(|| Error::NumericCast("decoded size overflow".into()))?;
        if let Some(max) = limit { if next > max { return Err(DecompressError::MemoryLimitExceeded { limit: max }.into()); } }
        Ok(())
    }

    fn read_capped<R: std::io::Read>(reader: R, output: &mut Vec<u8>, limit: Option<usize>, policy: Option<&Policy>) -> std::io::Result<()> {
        use std::io::Read;
        // Keep std's read_to_end scheduling, including its probe/growth behavior:
        // Flate discards bytes written during a failing read, so read boundaries
        // are observable in the legacy partial-output fallback.
        struct CheckedReader<'a, R> {
            inner: R,
            produced: usize,
            policy: &'a Policy,
        }
        impl<R: Read> Read for CheckedReader<'_, R> {
            fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
                // Keep the original decoder read boundary, even near the cap.
                // A narrower slice can turn one failing Flate read into a
                // successful prefix and falsely reject b20's retained output.
                // The spare buffer follows std/input-linear reservation; only
                // successfully returned bytes become decoded output below.
                let n = self.inner.read(buf)?;
                // read_to_end cannot append this read before the limit accepts it.
                self.policy.append(self.produced, n).map_err(crate::resource::into_io)?;
                self.produced += n;
                Ok(n)
            }
        }
        fn read_legacy<R: Read>(reader: R, output: &mut Vec<u8>, limit: Option<usize>) -> std::io::Result<()> {
            match limit {
                Some(max) => Read::take(reader, (max as u64).saturating_add(1)).read_to_end(output).map(|_| ()),
                None => { let mut reader = reader; reader.read_to_end(output).map(|_| ()) }
            }
        }
        match policy.filter(|p| !p.is_disabled()) {
            Some(policy) => {
                let checked = CheckedReader { inner: reader, produced: output.len(), policy };
                read_legacy(checked, output, limit)
            }
            None => read_legacy(reader, output, limit),
        }
    }

    fn decompress_lzw_policy(input: &[u8], params: Option<&Dictionary>, limit: Option<usize>, policy: Option<&Policy>) -> Result<Vec<u8>> {
        use weezl::{BitOrder, decode::Decoder};
        const MIN_BITS: u8 = 9;

        let early_change = params
            .and_then(|p| p.get(b"EarlyChange").ok())
            .and_then(|p| Object::as_i64(p).ok())
            .map(|v| v != 0)
            .unwrap_or(true);

        let mut decoder = if early_change {
            Decoder::with_tiff_size_switch(BitOrder::Msb, MIN_BITS - 1)
        } else {
            Decoder::new(BitOrder::Msb, MIN_BITS - 1)
        };

        let output = Self::decompress_lzw_loop_policy(input, &mut decoder, limit, policy)?;
        if let Some(max) = limit
            && output.len() > max
        {
            return Err(DecompressError::MemoryLimitExceeded { limit: max }.into());
        }
        Self::decompress_predictor_policy(output, params, policy)
    }

    fn decompress_lzw_loop_policy(input: &[u8], decoder: &mut weezl::decode::Decoder, limit: Option<usize>, policy: Option<&Policy>) -> Result<Vec<u8>> {
        let mut output = vec![];

        // When a limit is set, decode into a writer that stops accepting bytes
        // once `max + 1` have been produced, so a bomb cannot allocate the full
        // (potentially enormous) output; the caller rejects it via the `> max`
        // check. Without a limit, decode straight into the output vector.
        let status = if let Some(policy) = policy.filter(|p| !p.is_disabled()) {
            let mut sink = LimitedWriter::new(&mut output, limit.map_or(usize::MAX, |m| m.saturating_add(1)));
            sink.policy = Some(*policy);
            let status = decoder.into_stream(&mut sink).decode_all(input).status;
            status
        } else {
            match limit {
                Some(max) => {
                    let mut sink = LimitedWriter::new(&mut output, max.saturating_add(1));
                    decoder.into_stream(&mut sink).decode_all(input).status
                }
                None => decoder.into_stream(&mut output).decode_all(input).status,
            }
        };
        if let Err(err) = status {
            let err = Error::from_decode_io(err);
            if err.is_resource_limit() { return Err(err); }
            warn!("{err}");
        }

        Ok(output)
    }

    fn decompress_zlib_policy(input: &[u8], params: Option<&Dictionary>, limit: Option<usize>, policy: Option<&Policy>) -> Result<Vec<u8>> {
        use flate2::read::ZlibDecoder;

        // Reserve a starting capacity, but never pre-allocate beyond the limit so
        // a bomb cannot force a huge allocation up front.
        let initial_capacity = match limit {
            Some(max) => input.len().saturating_mul(2).min(max.saturating_add(1)),
            None => input.len().saturating_mul(2),
        };
        // This reservation is linear in encoded input, not decoded expansion.
        // Preserve b20's capacity as well as std's read_to_end: both determine
        // which bytes survive a failed Flate read (including when limits are off).
        let mut output = Vec::with_capacity(initial_capacity);

        if !input.is_empty()
            && let Err(err) = Self::read_capped(ZlibDecoder::new(input), &mut output, limit, policy)
        {
            let err = Error::from_decode_io(err);
            if err.is_decode_limit() { return Err(err); }
            warn!("{err}");
            // Zlib decompression failed (e.g. corrupt adler32 checksum in
            // encrypted PDFs). Retry with raw deflate, skipping the 2-byte
            // zlib header and ignoring the checksum.
            if output.is_empty() && input.len() > 2 {
                use flate2::read::DeflateDecoder;
                if let Err(raw_err) = Self::read_capped(DeflateDecoder::new(&input[2..]), &mut output, limit, policy) {
                    let raw_err = Error::from_decode_io(raw_err);
                    if raw_err.is_decode_limit() { return Err(raw_err); }
                    warn!("raw deflate fallback also failed: {raw_err}");
                }
            }
        }
        if let Some(max) = limit
            && output.len() > max
        {
            return Err(DecompressError::MemoryLimitExceeded { limit: max }.into());
        }
        Self::decompress_predictor_policy(output, params, policy)
    }

    fn decode_ascii85_policy(input: &[u8], limit: Option<usize>, policy: Option<&Policy>) -> Result<Vec<u8>> {
        let mut output = vec![];
        let mut buffer: u32 = 0;
        let mut count = 0;
        // Check for EOD marker
        let input_no_eod = if input.len() >= 2 && &input[input.len() - 2..] == b"~>" {
            &input[..input.len() - 2]
        } else {
            log::warn!("ASCII85 stream is missing its EOD marker");
            input
        };
        for &ch in input_no_eod {
            // ASCII85 can amplify up to 4x (each `z` expands to four zero bytes),
            // so bound the output here rather than only after the fact; each
            // iteration appends at most 4 bytes, so output never exceeds max + 4.
            if let Some(max) = limit
                && output.len() > max
            {
                return Err(DecompressError::MemoryLimitExceeded { limit: max }.into());
            }
            if ch == b'z' {
                if count != 0 {
                    return Err(DecompressError::Ascii85("z character is not allowed in the middle of a group").into());
                }
                Self::check_decode_append(output.len(), 4, limit, policy)?;
                output.extend_from_slice(&[0, 0, 0, 0]);
                continue;
            }

            if ch.is_ascii_whitespace() {
                continue;
            }

            if !(b'!'..=b'u').contains(&ch) {
                break;
            }
            // A base-85 group encodes a value in 0..2^32; reject any group whose
            // value reaches 2^32. Both the multiply and the add can carry past
            // u32::MAX (2^32-1 == 85 * 50529027, so a prefix of 50529027 leaves
            // the multiply in range yet the add still overflows), so guard both.
            buffer = buffer
                .checked_mul(85)
                .and_then(|b| b.checked_add((ch - b'!') as u32))
                .ok_or(DecompressError::Ascii85("group value out of range"))?;
            count += 1;

            if count == 5 {
                Self::check_decode_append(output.len(), 4, limit, policy)?;
                output.extend_from_slice(&buffer.to_be_bytes());
                buffer = 0;
                count = 0;
            }
        }

        if count > 0 {
            for _ in count..5 {
                buffer = buffer
                    .checked_mul(85)
                    .and_then(|b| b.checked_add(84))
                    .ok_or(DecompressError::Ascii85("group value out of range"))?;
            }

            let bytes = buffer.to_be_bytes();
            Self::check_decode_append(output.len(), count - 1, limit, policy)?;
            output.extend_from_slice(&bytes[..count - 1]);
        }

        Ok(output)
    }

    /// ASCIIHexDecode (PDF 32000-1:2008, 7.4.2): each byte is two hexadecimal
    /// digits, ASCII whitespace is ignored, and `>` ends the data. A final odd
    /// digit is treated as if followed by `0`; any other non-hex byte is an error.
    fn decode_ascii_hex_policy(input: &[u8], limit: Option<usize>, policy: Option<&Policy>) -> Result<Vec<u8>> {
        let mut output = vec![];
        let mut high: Option<u8> = None;
        for &ch in input {
            if let Some(max) = limit
                && output.len() > max
            {
                return Err(DecompressError::MemoryLimitExceeded { limit: max }.into());
            }
            if ch == b'>' {
                break;
            }
            if ch.is_ascii_whitespace() {
                continue;
            }
            let nibble = (ch as char)
                .to_digit(16)
                .ok_or(DecompressError::AsciiHex("invalid hexadecimal character"))? as u8;
            match high.take() {
                Some(h) => { Self::check_decode_append(output.len(), 1, limit, policy)?; output.push((h << 4) | nibble); },
                None => high = Some(nibble),
            }
        }
        if let Some(h) = high {
            Self::check_decode_append(output.len(), 1, limit, policy)?;
            output.push(h << 4);
        }
        Ok(output)
    }

    /// RunLengthDecode (PDF 32000-1:2008, 7.4.5): a length byte `l` where
    /// `0..=127` copies the next `l + 1` bytes literally, `129..=255` repeats the
    /// next byte `257 - l` times, and `128` marks the end of data. A stream that
    /// ends mid-run is decoded best-effort rather than treated as fatal.
    fn decode_run_length_policy(input: &[u8], limit: Option<usize>, policy: Option<&Policy>) -> Result<Vec<u8>> {
        let mut output = vec![];
        let mut i = 0;
        while i < input.len() {
            if let Some(max) = limit
                && output.len() > max
            {
                return Err(DecompressError::MemoryLimitExceeded { limit: max }.into());
            }
            let length = input[i];
            i += 1;
            match length {
                128 => break,
                0..=127 => {
                    let end = (i + length as usize + 1).min(input.len());
                    Self::check_decode_append(output.len(), end - i, limit, policy)?;
                    output.extend_from_slice(&input[i..end]);
                    i = end;
                }
                _ => {
                    let Some(&byte) = input.get(i) else { break };
                    Self::check_decode_append(output.len(), 257 - length as usize, limit, policy)?;
                    output.resize(output.len() + (257 - length as usize), byte);
                    i += 1;
                }
            }
        }
        Ok(output)
    }

    fn decompress_predictor_policy(mut data: Vec<u8>, params: Option<&Dictionary>, policy: Option<&Policy>) -> Result<Vec<u8>> {
        use crate::filters::png;

        if let Some(params) = params {
            let predictor = params.get(b"Predictor").and_then(Object::as_i64).unwrap_or(1);
            if predictor == 2 {
                // TIFF Predictor 2 (horizontal differencing). Distinct from the PNG
                // predictors below and previously ignored, so `/Predictor 2` streams
                // silently decoded to the un-differenced (wrong) bytes.
                let columns = max(1, params.get(b"Columns").and_then(Object::as_i64).unwrap_or(1)) as usize;
                let colors = max(1, params.get(b"Colors").and_then(Object::as_i64).unwrap_or(1)) as usize;
                let bits = max(1, params.get(b"BitsPerComponent").and_then(Object::as_i64).unwrap_or(8)) as usize;
                data = Self::reverse_tiff_predictor2_policy(data, columns, colors, bits, policy)?;
            } else if (10..=15).contains(&predictor) {
                if data.is_empty() { return Ok(data); }
                // PNG predictors 10-15. Rows are packed to a byte boundary, so a
                // sub-byte component depth still occupies ceil(Columns*Colors*bits/8)
                // bytes per row; the old max(8, bits) width over-read and failed on
                // valid 1/2/4-bit images. The filter's left reference is whole bytes,
                // rounded up to one.
                let columns = max(1, params.get(b"Columns").and_then(Object::as_i64).unwrap_or(1)) as usize;
                let colors = max(1, params.get(b"Colors").and_then(Object::as_i64).unwrap_or(1)) as usize;
                let bits = max(1, params.get(b"BitsPerComponent").and_then(Object::as_i64).unwrap_or(8)) as usize;
                let arithmetic = policy.copied().unwrap_or_else(Policy::disabled);
                let bytes_per_row = arithmetic.product(Limit::PredictorRowBytes, &[columns, colors, bits])?.div_ceil(8);
                let bpp = arithmetic.product(Limit::PredictorRowBytes, &[colors, bits])?.div_ceil(8);
                data = png::decode_frame_with_policy(data.as_slice(), bpp, bytes_per_row, policy).map_err(Error::from_decode_io)?;
            }
            Ok(data)
        } else {
            Ok(data)
        }
    }

    /// Reverse TIFF Predictor 2 (horizontal differencing): each sample is the
    /// running sum of the encoded differences of the same colour component along
    /// its row (PDF 32000-1:2008 Table 10). Rows are `Columns * Colors` samples of
    /// `BitsPerComponent` bits; sums wrap modulo `2^BitsPerComponent`. The 1-, 2-,
    /// 4-, 8- and 16-bit depths PDF allows are all supported; sub-byte samples are
    /// unpacked, summed and repacked (see [`reverse_tiff_predictor2_subbyte`]).
    fn reverse_tiff_predictor2_policy(mut data: Vec<u8>, columns: usize, colors: usize, bits: usize, policy: Option<&Policy>) -> Result<Vec<u8>> {
        if !matches!(bits, 1 | 2 | 4 | 8 | 16) {
            return Err(DecompressError::Predictor("TIFF Predictor 2 supports only 1-, 2-, 4-, 8- or 16-bit components").into());
        }
        let arithmetic = policy.copied().unwrap_or_else(Policy::disabled);
        let row_bytes = if arithmetic.is_disabled() {
            // b20 computed byte strides directly for 8/16-bit TIFF rows. A
            // wrapped bit product divided by eight is not equivalent to that.
            match bits {
                8 => arithmetic.product(Limit::PredictorRowBytes, &[columns, colors])?,
                16 => arithmetic.product(Limit::PredictorRowBytes, &[columns, colors, 2])?,
                _ => arithmetic.product(Limit::PredictorRowBytes, &[columns, colors, bits])?.div_ceil(8),
            }
        } else {
            arithmetic.product(Limit::PredictorRowBytes, &[columns, colors, bits])?.div_ceil(8)
        };
        if let Some(policy) = policy { policy.check(Limit::PredictorRowBytes, row_bytes)?; }
        match bits {
            8 => {
                let stride = row_bytes;
                if stride > 0 {
                    for row in data.chunks_mut(stride) {
                        for i in colors..row.len() {
                            row[i] = row[i].wrapping_add(row[i - colors]);
                        }
                    }
                }
            }
            16 => {
                let stride = row_bytes;
                if stride > 0 {
                    for row in data.chunks_mut(stride) {
                        let samples = row.len() / 2;
                        for s in colors..samples {
                            let prev = u16::from_be_bytes([row[(s - colors) * 2], row[(s - colors) * 2 + 1]]);
                            let cur = u16::from_be_bytes([row[s * 2], row[s * 2 + 1]]);
                            row[s * 2..s * 2 + 2].copy_from_slice(&cur.wrapping_add(prev).to_be_bytes());
                        }
                    }
                }
            }
            1 | 2 | 4 => {
                let stride = row_bytes;
                if stride > 0 {
                    for row in data.chunks_mut(stride) {
                        Self::reverse_tiff_predictor2_subbyte_policy(row, columns, colors, bits, policy)?;
                    }
                }
            }
            _ => {
                return Err(DecompressError::Predictor(
                    "TIFF Predictor 2 supports only 1-, 2-, 4-, 8- or 16-bit components",
                )
                .into());
            }
        }
        Ok(data)
    }

    /// Reverse TIFF Predictor 2 for one sub-byte (1/2/4-bit) packed row in place.
    /// Samples are unpacked MSB-first, summed per colour component modulo
    /// `2^bits`, then repacked MSB-first; the row's trailing padding bits are kept.
    fn reverse_tiff_predictor2_subbyte_policy(row: &mut [u8], columns: usize, colors: usize, bits: usize, policy: Option<&Policy>) -> Result<()> {
        let mask = (1u16 << bits) - 1;
        let arithmetic = policy.copied().unwrap_or_else(Policy::disabled);
        let declared_samples = arithmetic.product(Limit::PredictorRowBytes, &[columns, colors])?;
        let actual_samples = arithmetic.product(Limit::PredictorRowBytes, &[row.len(), 8])? / bits;
        let samples = declared_samples.min(actual_samples);
        let accumulator_len = colors.min(samples);
        let accumulator_bytes = arithmetic.product(Limit::PredictorRowBytes, &[accumulator_len, 2])?;
        if let Some(policy) = policy { policy.check(Limit::PredictorRowBytes, accumulator_bytes)?; }
        let mut acc = vec![0u16; accumulator_len];
        let mut out = vec![0u8; row.len()];
        let mut in_bit = 0usize;
        let mut out_buf = 0u32;
        let mut out_bits = 0u32;
        let mut out_idx = 0usize;
        for s in 0..samples {
            let byte = in_bit / 8;
            let off = (in_bit % 8) as u32;
            let window = ((row[byte] as u16) << 8) | *row.get(byte + 1).unwrap_or(&0) as u16;
            let sample = (window >> (16 - off - bits as u32)) & mask;
            in_bit += bits;

            let c = s % colors;
            acc[c] = (acc[c] + sample) & mask;

            out_buf = (out_buf << bits) | acc[c] as u32;
            out_bits += bits as u32;
            if out_bits >= 8 {
                out[out_idx] = (out_buf >> (out_bits - 8)) as u8;
                out_idx += 1;
                out_bits -= 8;
            }
        }
        if out_bits > 0 && out_idx < out.len() {
            let keep = 8 - out_bits;
            out[out_idx] = ((out_buf << keep) as u8) | (row[out_idx] & ((1u8 << keep) - 1));
        }
        row.copy_from_slice(&out);
        Ok(())
    }

    pub fn decompress(&mut self) -> Result<()> {
        let data = self.decompressed_content()?;
        self.dict.remove(b"DecodeParms");
        self.dict.remove(b"Filter");
        self.set_content(data);
        Ok(())
    }

    /// Decompress this stream in place like [`Stream::decompress`], but reject it
    /// with [`DecompressError::MemoryLimitExceeded`] if the decoded output would
    /// exceed `max_output` bytes. Used on the load path to bound the memory a
    /// single object/xref stream can consume.
    pub fn decompress_with_limit(&mut self, max_output: usize) -> Result<()> {
        let data = self.decompressed_content_with_limit(max_output)?;
        self.dict.remove(b"DecodeParms");
        self.dict.remove(b"Filter");
        self.set_content(data);
        Ok(())
    }

    pub fn is_compressed(&self) -> bool {
        self.dict.get(b"Filter").is_ok()
    }
}

#[cfg(test)]
mod test {
    use crate::{Error, error::DecompressError};

    use super::Stream;

    #[test]
    fn test_decode_ascii85() {
        let input = r#"9jqo^BlbD-BleB1DJ+*+F(f,q/0JhKF<GL>Cj@.4Gp$d7F!,L7@<6@)/0JDEF<G%<+EV:2F!,O<
            DJ+*.@<*K0@<6L(Df-\0Ec5e;DffZ(EZee.Bl.9pF"AGXBPCsi+DGm>@3BB/F*&OCAfu2/AKYi(
            DIb:@FD,*)+C]U=@3BN#EcYf8ATD3s@q?d$AftVqCh[NqF<G:8+EV:.+Cf>-FD5W8ARlolDIal(
            DId<j@<?3r@:F%a+D58'ATD4$Bl@l3De:,-DJs`8ARoFb/0JMK@qB4^F!,R<AKZ&-DfTqBG%G>u
            D.RTpAKYo'+CT/5+Cei#DII?(E,9)oF*2M7/c~>"#;
        let expected = "Man is distinguished, not only by his reason, but by this singular passion from other animals, which is a lust of the mind, that by a perseverance of delight in the continued and indefatigable generation of knowledge, exceeds the short vehemence of any carnal pleasure.";
        let output = Stream::decode_ascii85(input.as_bytes(), None).unwrap();
        println!("{}", String::from_utf8(output.clone()).unwrap());
        assert_eq!(&output, expected.as_bytes());
    }

    #[test]
    fn test_decode_ascii85_overflow() {
        let input = b"uuuuu~>";
        let output = Stream::decode_ascii85(input, None);
        // let expected: Result<Vec<u8>, Error> = Err(Error::ContentDecode);
        assert!(matches!(output, Err(Error::Decompress(DecompressError::Ascii85(_)))));
    }

    #[test]
    fn test_decode_ascii85_group_value_out_of_range() {
        // A base-85 group must be < 2^32. 85 * 50529027 == u32::MAX, so the prefix
        // "s8W-" reaches the multiply ceiling and the 5th digit overflows the add;
        // the multiply-only guard missed this. The value == u32::MAX case ("s8W-!")
        // is in range and must still decode.
        assert_eq!(
            Stream::decode_ascii85(b"s8W-!~>", None).unwrap(),
            vec![255, 255, 255, 255]
        );

        for input in [
            &b"s8W-\"~>"[..], // value == 2^32 (issue #442)
            &b"s8W-~>"[..],   // same overflow via the partial-group padding
            &b"uuuuu~>"[..],  // overflow caught by the multiply
        ] {
            let out = Stream::decode_ascii85(input, None);
            assert!(
                matches!(out, Err(Error::Decompress(DecompressError::Ascii85(_)))),
                "expected {input:?} to be rejected, got {out:?}"
            );
        }
    }

    #[test]
    fn test_decode_ascii85_valid_vectors() {
        let cases: &[(&[u8], &[u8])] = &[
            (b"88/~>", &[72, 105]),
            (b"87cURD_*#4DfTZ)+T~>", b"Hello, World!"),
            (b"z~>", &[0, 0, 0, 0]),
            (b":ddb~>", b"PDF"),
            (
                b"!!*-'\"9eu7#RLhG$k3[W&.oNg~>",
                &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
            ),
        ];
        for (input, expected) in cases {
            assert_eq!(
                &Stream::decode_ascii85(input, None).unwrap(),
                expected,
                "input {input:?}"
            );
        }
    }

    #[test]
    fn test_decompress_tiff_predictor2() {
        use crate::Dictionary;

        fn params(colors: i64, columns: i64, bits: i64) -> Dictionary {
            let mut p = Dictionary::new();
            p.set("Predictor", 2i64);
            p.set("Colors", colors);
            p.set("Columns", columns);
            p.set("BitsPerComponent", bits);
            p
        }

        // 8-bit single component: running sum along the row, wrapping mod 256.
        assert_eq!(
            Stream::decompress_predictor(vec![10, 5, 250, 1], Some(&params(1, 4, 8))).unwrap(),
            vec![10, 15, 9, 10]
        );
        // 8-bit, 3 colours: differencing is per component.
        assert_eq!(
            Stream::decompress_predictor(vec![255, 0, 128, 1, 2, 3], Some(&params(3, 2, 8))).unwrap(),
            vec![255, 0, 128, 0, 2, 131]
        );
        // 8-bit: rows reconstruct independently.
        assert_eq!(
            Stream::decompress_predictor(vec![1, 2, 3, 10, 20, 30], Some(&params(1, 3, 8))).unwrap(),
            vec![1, 3, 6, 10, 30, 60]
        );
        // 16-bit big-endian samples, sums wrap mod 2^16.
        assert_eq!(
            Stream::decompress_predictor(vec![0xFF, 0xFF, 0x00, 0x02], Some(&params(1, 2, 16))).unwrap(),
            vec![0xFF, 0xFF, 0x00, 0x01]
        );
        // Sub-byte depths (1/2/4-bit) are now reconstructed instead of rejected;
        // values verified against pdf.js. 1-bit, 8 columns: running sum mod 2.
        assert_eq!(
            Stream::decompress_predictor(vec![0b1010_1010], Some(&params(1, 8, 1))).unwrap(),
            vec![0b1100_1100]
        );
        // 1-bit, 16 columns, two rows.
        assert_eq!(
            Stream::decompress_predictor(vec![255, 170, 8, 255], Some(&params(1, 16, 1))).unwrap(),
            vec![170, 204, 15, 85]
        );
        // 2-bit and 4-bit, single component (rows have trailing padding bits).
        assert_eq!(
            Stream::decompress_predictor(vec![85, 179], Some(&params(1, 6, 2))).unwrap(),
            vec![108, 147]
        );
        assert_eq!(
            Stream::decompress_predictor(vec![25, 16], Some(&params(1, 3, 4))).unwrap(),
            vec![26, 176]
        );
        // 4-bit, 2 colours: differencing stays per component.
        assert_eq!(
            Stream::decompress_predictor(vec![18, 34], Some(&params(2, 2, 4))).unwrap(),
            vec![18, 52]
        );
        // A depth PDF never uses (not 1/2/4/8/16) is still rejected.
        assert!(matches!(
            Stream::decompress_predictor(vec![0, 0, 0], Some(&params(1, 4, 3))),
            Err(Error::Decompress(DecompressError::Predictor(_)))
        ));
    }

    #[test]
    fn test_decompress_png_predictor_subbyte() {
        use crate::Dictionary;

        // Predictor 15 (PNG). Each row carries a leading filter-type byte. Decoded
        // bytes verified against pdf.js and a real libpng-encoded 1-bit PNG.
        fn params(colors: i64, columns: i64, bits: i64) -> Dictionary {
            let mut p = Dictionary::new();
            p.set("Predictor", 15i64);
            p.set("Colors", colors);
            p.set("Columns", columns);
            p.set("BitsPerComponent", bits);
            p
        }

        // 1-bit grayscale, 16 columns: 2 packed bytes/row. Before the fix this
        // decoded with a 16-byte row width and failed with UnexpectedEof on a
        // valid image. Filters None/Sub/Up.
        assert_eq!(
            Stream::decompress_predictor(vec![0, 170, 204, 1, 15, 70, 2, 225, 65], Some(&params(1, 16, 1))).unwrap(),
            vec![170, 204, 15, 85, 240, 150]
        );
        // 2-bit grayscale, 6 columns (Sub, Average).
        assert_eq!(
            Stream::decompress_predictor(vec![1, 108, 36, 3, 144, 141], Some(&params(1, 6, 2))).unwrap(),
            vec![108, 144, 198, 56]
        );
        // 4-bit grayscale, 3 columns (Up, Paeth).
        assert_eq!(
            Stream::decompress_predictor(vec![2, 26, 176, 4, 171, 107], Some(&params(1, 3, 4))).unwrap(),
            vec![26, 176, 197, 48]
        );
        // 2-bit RGB (3 colours), 4 columns: 3 bytes/row, left reference = 1 byte.
        // The old bytes_per_pixel * pixels_per_row factorisation could not express
        // this row width.
        assert_eq!(
            Stream::decompress_predictor(vec![1, 108, 48, 72, 4, 175, 202, 241], Some(&params(3, 4, 2))).unwrap(),
            vec![108, 156, 228, 27, 54, 141]
        );
    }

    #[test]
    fn test_decompress_zlib_corrupt_checksum() {
        use flate2::Compression;
        use flate2::write::ZlibEncoder;
        use std::io::Write;

        let original = b"BT /F1 12 Tf (Hello World) Tj ET";

        // Compress with valid zlib
        let mut encoder = ZlibEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(original).unwrap();
        let mut compressed = encoder.finish().unwrap();

        // Corrupt the adler32 checksum (last 4 bytes)
        let len = compressed.len();
        assert!(len >= 4);
        for byte in &mut compressed[len - 4..] {
            *byte ^= 0xFF;
        }

        // Normal zlib should fail, but our fallback should recover
        let result = Stream::decompress_zlib(&compressed, None, None).unwrap();
        assert_eq!(result, original);
    }

    #[test]
    fn test_uncompressed_stream_returns_raw_content() {
        use crate::Dictionary;

        // A stream with no /Filter should return its raw content from decompressed_content()
        let content = b"/FullPage Do
"
        .to_vec();
        let mut dict = Dictionary::new();
        dict.set("Length", content.len() as i64);
        let stream = Stream::new(dict, content.clone());

        let result = stream
            .decompressed_content()
            .expect("should succeed for uncompressed stream");
        assert_eq!(result, content);
    }

    #[test]
    fn test_uncompressed_stream_honors_limit() {
        use crate::Dictionary;

        // Even with no filter, an over-limit raw stream must be rejected so the
        // caller's memory bound is never silently exceeded.
        let content = vec![0u8; 1024];
        let mut dict = Dictionary::new();
        dict.set("Length", content.len() as i64);
        let stream = Stream::new(dict, content);

        assert!(matches!(
            stream.decompressed_content_with_limit(512),
            Err(Error::Decompress(DecompressError::MemoryLimitExceeded { limit: 512 }))
        ));
        assert!(stream.decompressed_content_with_limit(4096).is_ok());
    }

    #[test]
    fn test_limited_writer_fills_to_limit_then_errors() {
        use super::LimitedWriter;
        use std::io::Write;

        // The push-based (LZW) guard: the sink accepts bytes up to `limit`, then
        // fills the remaining room and refuses further writes, so the caller can
        // detect the overflow via `len() > max` without unbounded allocation.
        let mut buf = Vec::new();
        let mut writer = LimitedWriter::new(&mut buf, 4);

        assert_eq!(writer.write(b"ab").unwrap(), 2);
        // This write would cross the limit: it fills to exactly 4 bytes, errors.
        let err = writer.write(b"cdef").unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::WriteZero);

        assert_eq!(buf, b"abcd");
    }

    #[test]
    fn test_ascii85_honors_limit() {
        // Each `z` expands to four zero bytes, so 1 MiB of `z` decodes to 4 MiB.
        // Without a limit that full 4x amplification is allocated; with a limit
        // the decoder must stop rather than expand past it (guards the chained
        // `[FlateDecode ASCII85Decode]` bomb vector on the load path).
        let mut input = vec![b'z'; 1024 * 1024];
        input.extend_from_slice(b"~>");

        let full = Stream::decode_ascii85(&input, None).unwrap();
        assert_eq!(full.len(), 4 * 1024 * 1024, "unbounded ASCII85 amplifies 4x");

        assert!(matches!(
            Stream::decode_ascii85(&input, Some(1024 * 1024)),
            Err(Error::Decompress(DecompressError::MemoryLimitExceeded { limit })) if limit == 1024 * 1024
        ));
    }

    #[test]
    fn test_lzw_bomb_rejected_with_limit() {
        use crate::Dictionary;
        use weezl::{BitOrder, encode::Encoder};

        // A tiny LZW stream that decodes back to 8 MiB of zeros. The encoder must
        // match lopdf's default decoder (`with_tiff_size_switch`, code size 8).
        let plain = vec![0u8; 8 * 1024 * 1024];
        let compressed = Encoder::with_tiff_size_switch(BitOrder::Msb, 8).encode(&plain).unwrap();
        assert!(compressed.len() < 128 * 1024, "LZW bomb input should be tiny");

        let mut dict = Dictionary::new();
        dict.set("Filter", "LZWDecode");
        let stream = Stream::new(dict, compressed);

        // Bounded: the push-based LZW path (LimitedWriter + weezl) rejects it.
        assert!(matches!(
            stream.decompressed_content_with_limit(1024 * 1024),
            Err(Error::Decompress(DecompressError::MemoryLimitExceeded { limit })) if limit == 1024 * 1024
        ));
        // Unbounded default still round-trips fully (proves the pairing is valid,
        // so the rejection above is real and not a decode failure).
        assert_eq!(stream.decompressed_content().unwrap().len(), 8 * 1024 * 1024);
    }

    #[test]
    fn test_decode_ascii_hex() {
        use crate::Dictionary;

        // Whole-stream path: a `/Filter /ASCIIHexDecode` stream decodes to bytes.
        let mut dict = Dictionary::new();
        dict.set("Filter", "ASCIIHexDecode");
        let stream = Stream::new(dict, b"48656C6C6F>".to_vec());
        assert_eq!(stream.decompressed_content().unwrap(), b"Hello");

        // Spec vectors (PDF 32000-1 7.4.2). Whitespace is ignored, `>` ends the
        // data, and a trailing odd digit is padded with 0.
        let cases: &[(&[u8], &[u8])] = &[
            (b"48 65 6C\n6C6F>", b"Hello"),
            (b"901FA3>", &[0x90, 0x1F, 0xA3]),
            (b"4A5>", &[0x4A, 0x50]),
            (b"4>", &[0x40]),
            (b">", &[]),
            (b"4865>ffff", b"He"),
        ];
        for (input, expected) in cases {
            assert_eq!(
                &Stream::decode_ascii_hex(input, None).unwrap(),
                expected,
                "input {input:?}"
            );
        }

        // A byte that is neither a hex digit, whitespace, nor `>` is rejected.
        assert!(matches!(
            Stream::decode_ascii_hex(b"48XY>", None),
            Err(Error::Decompress(DecompressError::AsciiHex(_)))
        ));
    }

    #[test]
    fn test_decode_run_length() {
        use crate::Dictionary;

        // Whole-stream path: literal run of 5, then 3 copies of 'A', then EOD.
        let mut dict = Dictionary::new();
        dict.set("Filter", "RunLengthDecode");
        let mut data = vec![0x04];
        data.extend_from_slice(b"Hello");
        data.extend_from_slice(&[0xFE, b'A', 0x80]);
        let stream = Stream::new(dict, data);
        assert_eq!(stream.decompressed_content().unwrap(), b"HelloAAA");

        // Spec vectors (PDF 32000-1 7.4.5). Length 0x81 = 128 copies, 0xFF = 2,
        // 0x80 = end of data (trailing bytes are dropped), and a run that ends
        // without an EOD marker is decoded best-effort.
        let cases: &[(&[u8], Vec<u8>)] = &[
            (&[0x81, b'Z', 0x80], vec![b'Z'; 128]),
            (&[0xFF, b'Q', 0x00, b'!', 0x80], b"QQ!".to_vec()),
            (&[0x02, b'a', b'b', b'c', 0x80, b'x', b'y'], b"abc".to_vec()),
            (&[0x01, b'H', b'i'], b"Hi".to_vec()),
        ];
        for (input, expected) in cases {
            assert_eq!(
                &Stream::decode_run_length(input, None).unwrap(),
                expected,
                "input {input:?}"
            );
        }

        // Run-length can amplify up to 128x, so an over-limit stream must be
        // rejected instead of allocating the full expansion (bomb guard).
        let bomb: Vec<u8> = std::iter::repeat_n([0x81u8, b'x'], 4096).flatten().collect();
        assert_eq!(Stream::decode_run_length(&bomb, None).unwrap().len(), 128 * 4096);
        assert!(matches!(
            Stream::decode_run_length(&bomb, Some(64 * 1024)),
            Err(Error::Decompress(DecompressError::MemoryLimitExceeded { limit })) if limit == 64 * 1024
        ));
    }
}
