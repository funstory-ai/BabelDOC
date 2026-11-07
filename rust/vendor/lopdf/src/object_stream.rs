use crate::parser;
use crate::{Document, Error, Object, ObjectId, Result, Stream};
use std::collections::{BTreeMap, HashMap};
use std::num::TryFromIntError;
use std::str::FromStr;

use log::warn;
#[cfg(feature = "rayon")]
use rayon::prelude::*;

#[derive(Debug)]
pub struct ObjectStream {
    pub objects: BTreeMap<ObjectId, Object>,
    /// Valid member pairs rejected by segmented parsing, split by reason.
    pub diagnostics: ObjectStreamDiagnostics,
    max_objects: usize,
    compression_level: u32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ObjectStreamDiagnostics {
    pub out_of_bounds_or_duplicate: usize,
    pub segment_b20_would_accept: usize,
    pub segment_both_fail: usize,
    /// Segment failures beyond the bounded legacy-classification budget.
    pub segment_unclassified: usize,
}

impl ObjectStreamDiagnostics {
    pub fn invalid_members(self) -> usize {
        self.out_of_bounds_or_duplicate
            + self.segment_b20_would_accept
            + self.segment_both_fail
            + self.segment_unclassified
    }
}

impl std::ops::AddAssign for ObjectStreamDiagnostics {
    fn add_assign(&mut self, rhs: Self) {
        self.out_of_bounds_or_duplicate += rhs.out_of_bounds_or_duplicate;
        self.segment_b20_would_accept += rhs.segment_b20_would_accept;
        self.segment_both_fail += rhs.segment_both_fail;
        self.segment_unclassified += rhs.segment_unclassified;
    }
}

#[derive(Clone, Copy, Debug)]
enum ObjectStreamFailure {
    OutOfBoundsOrDuplicate,
    SegmentNeedsClassification(usize),
}

const MAX_SEGMENT_FAILURE_CLASSIFICATIONS: usize = 64;

fn classify_segment_failure(
    legacy_slice: &[u8],
    real_parser: Option<fn(&str) -> f32>,
) -> bool {
    // This is the b20 member parser: it reads from the current member start
    // through the end of the decoded stream.
    let legacy_succeeds = parser::with_real_parser(real_parser, || {
        parser::direct_object(legacy_slice)
    })
    .is_some();
    legacy_succeeds
}

#[derive(Debug, Clone)]
pub struct ObjectStreamBuilder {
    max_objects: usize,
    compression_level: u32,
}

#[derive(Debug, Clone)]
pub struct ObjectStreamConfig {
    pub max_objects_per_stream: usize,
    pub compression_level: u32,
}

impl Default for ObjectStreamConfig {
    fn default() -> Self {
        Self {
            max_objects_per_stream: 100,
            compression_level: 6,
        }
    }
}

impl ObjectStream {
    /// Parse an existing object stream without modifying its encoded content or
    /// filter dictionary.
    ///
    /// This decodes the stream without any size limit. For untrusted input,
    /// prefer [`ObjectStream::new_with_limit`] to guard against decompression
    /// bombs.
    pub fn new(stream: &Stream) -> Result<ObjectStream> {
        Self::new_with_limit(stream, None)
    }

    /// Parse an existing object stream without modifying it, rejecting the
    /// decoded content if it would exceed `max_decompressed_size` bytes. `None`
    /// means no limit (the behavior of [`ObjectStream::new`]).
    pub fn new_with_limit(stream: &Stream, max_decompressed_size: Option<usize>) -> Result<ObjectStream> {
        Self::new_with_policy(stream, max_decompressed_size, None)
    }

    pub fn new_with_policy(stream: &Stream, max_decompressed_size: Option<usize>, policy: Option<&crate::resource::Policy>) -> Result<ObjectStream> {
        let content = stream.get_plain_content_with_policy(policy, max_decompressed_size)?;

        if content.is_empty() {
            return Ok(ObjectStream {
                objects: BTreeMap::new(),
                diagnostics: ObjectStreamDiagnostics::default(),
                max_objects: 100,
                compression_level: 6,
            });
        }

        let first_offset = stream
            .dict
            .get(b"First")
            .and_then(Object::as_i64)?
            .try_into()
            .map_err(|e: TryFromIntError| Error::NumericCast(e.to_string()))?;
        let index_block = content.get(..first_offset).ok_or(Error::InvalidOffset(first_offset))?;

        let numbers_str = std::str::from_utf8(index_block).map_err(|e| Error::InvalidObjectStream(e.to_string()))?;
        let numbers: Vec<_> = numbers_str
            .split_whitespace()
            .map(|number| u32::from_str(number).ok())
            .collect();
        let len = numbers.len() / 2 * 2; // Ensure only pairs.

        let n = stream.dict.get(b"N").and_then(Object::as_i64)?;
        if numbers.len().try_into().ok() != n.checked_mul(2) {
            warn!("object stream: the object stream dictionary specifies a wrong number of objects")
        }

        // Each offset names the start of a distinct object. Duplicate offsets
        // are malformed and invalidate every member using that offset.
        let mut offsets: Vec<u32> = numbers[..len]
            .chunks_exact(2)
            .filter_map(|chunk| chunk[1])
            .collect();
        offsets.sort_unstable();
        offsets.dedup();
        let mut next_by_offset = HashMap::with_capacity(offsets.len());
        for (index, &relative) in offsets.iter().enumerate() {
            let end = offsets
                .get(index + 1)
                .map_or(content.len(), |&next| {
                    first_offset.saturating_add(next as usize).min(content.len())
                });
            next_by_offset.insert(relative, end);
        }
        let mut occurrences = HashMap::with_capacity(offsets.len());
        for relative in numbers[..len].chunks_exact(2).filter_map(|chunk| chunk[1]) {
            *occurrences.entry(relative).or_insert(0usize) += 1;
        }

        let real_parser = parser::real_parser();
        let parse_member = |chunk: &[_]| {
            let (Some(id), Some(relative)) = (chunk[0], chunk[1]) else {
                return (None, None);
            };
            let offset = first_offset.saturating_add(relative as usize);
            if offset >= content.len() || occurrences.get(&relative).copied() != Some(1) {
                warn!("out-of-bounds or duplicate offset in object stream");
                return (None, Some(ObjectStreamFailure::OutOfBoundsOrDuplicate));
            }
            let end = *next_by_offset.get(&relative).unwrap_or(&content.len());
            let mut start = offset;
            while start < end && content[start].is_ascii_whitespace() {
                start += 1;
            }
            if start >= end {
                warn!("only whitespace after offset in object stream");
                return (None, Some(ObjectStreamFailure::SegmentNeedsClassification(offset)));
            }
            let object = parser::with_real_parser(real_parser, || {
                parser::direct_object(&content[start..end])
            });
            match object {
                Some(object) => (Some(((id, 0), object)), None),
                None => (None, Some(ObjectStreamFailure::SegmentNeedsClassification(offset))),
            }
        };
        #[cfg(feature = "rayon")]
        let parsed: Vec<_> = numbers[..len].par_chunks(2).map(parse_member).collect();
        #[cfg(not(feature = "rayon"))]
        let parsed: Vec<_> = numbers[..len].chunks(2).map(parse_member).collect();
        let mut diagnostics = ObjectStreamDiagnostics::default();
        let mut segment_failures = 0usize;
        for (_, failure) in &parsed {
            match failure {
                None => {}
                Some(ObjectStreamFailure::OutOfBoundsOrDuplicate) => {
                    diagnostics.out_of_bounds_or_duplicate += 1
                }
                Some(ObjectStreamFailure::SegmentNeedsClassification(offset)) => {
                    if segment_failures < MAX_SEGMENT_FAILURE_CLASSIFICATIONS {
                        // The legacy parser starts after leading whitespace and
                        // runs to end-of-stream. Compute that start only for the
                        // bounded classification attempts, never per member.
                        let mut legacy_start = *offset;
                        while legacy_start < content.len()
                            && content[legacy_start].is_ascii_whitespace()
                        {
                            legacy_start += 1;
                        }
                        if classify_segment_failure(&content[legacy_start..], real_parser) {
                            diagnostics.segment_b20_would_accept += 1;
                        } else {
                            diagnostics.segment_both_fail += 1;
                        }
                    } else {
                        diagnostics.segment_unclassified += 1
                    }
                    segment_failures += 1
                }
            }
        }
        let objects = parsed.into_iter().filter_map(|(object, _)| object).collect();

        Ok(ObjectStream {
            objects,
            diagnostics,
            max_objects: 100,
            compression_level: 6,
        })
    }

    /// Create a builder for constructing new object streams
    pub fn builder() -> ObjectStreamBuilder {
        ObjectStreamBuilder {
            max_objects: 100,
            compression_level: 6,
        }
    }

    /// Add an object to the stream
    pub fn add_object(&mut self, id: ObjectId, obj: Object) -> Result<()> {
        // Check if object can be added to stream
        if matches!(obj, Object::Stream(_)) {
            return Err(Error::InvalidObjectStream(
                "Stream objects cannot be stored in object streams".into(),
            ));
        }

        // Check capacity
        if self.objects.len() >= self.max_objects {
            return Err(Error::InvalidObjectStream(format!(
                "Object stream has reached maximum capacity of {} objects",
                self.max_objects
            )));
        }

        self.objects.insert(id, obj);
        Ok(())
    }

    /// Get the number of objects in the stream
    pub fn object_count(&self) -> usize {
        self.objects.len()
    }

    /// Build the stream content in the format required by PDF spec
    pub fn build_stream_content(&self) -> Result<Vec<u8>> {
        if self.objects.is_empty() {
            return Ok(Vec::new());
        }

        // Sort objects by ID for consistent output
        let mut sorted_objects: Vec<_> = self.objects.iter().collect();
        sorted_objects.sort_by_key(|(id, _)| *id);

        // First build the offset table to know its size
        let mut offset_entries = Vec::new();
        let mut current_offset = 0;

        for ((obj_num, _gen), obj) in &sorted_objects {
            // Store the object number and its offset
            offset_entries.push(format!("{obj_num} {current_offset}"));

            // Calculate size of this object's serialization
            let mut obj_bytes = Vec::new();
            crate::writer::Writer::write_object(&mut obj_bytes, obj)?;
            current_offset += obj_bytes.len() + 1; // +1 for space separator
        }

        // Build the complete offset table with proper spacing
        let offset_table = offset_entries.join(" ") + " ";

        // Now build the final content
        let mut content = Vec::new();
        content.extend_from_slice(offset_table.as_bytes());

        // Add serialized objects with space separators
        for ((_, _), obj) in &sorted_objects {
            let mut obj_bytes = Vec::new();
            crate::writer::Writer::write_object(&mut obj_bytes, obj)?;
            content.extend_from_slice(&obj_bytes);
            content.push(b' '); // Space separator between objects
        }

        Ok(content)
    }

    /// Convert to a Stream object ready for insertion into a PDF
    pub fn to_stream_object(&self) -> Result<Stream> {
        let content = self.build_stream_content()?;

        // Calculate where the first object starts
        // We need to find the size of the offset table
        let mut sorted_objects: Vec<_> = self.objects.iter().collect();
        sorted_objects.sort_by_key(|(id, _)| *id);

        // Build the offset entries to calculate exact size
        let mut offset_entries = Vec::new();
        let mut current_offset = 0;

        for ((obj_num, _gen), obj) in &sorted_objects {
            offset_entries.push(format!("{obj_num} {current_offset}"));

            // Calculate size of this object's serialization
            let mut obj_bytes = Vec::new();
            crate::writer::Writer::write_object(&mut obj_bytes, obj)?;
            current_offset += obj_bytes.len() + 1; // +1 for space separator
        }

        // The offset table is joined with spaces and has a trailing space
        let offset_table = offset_entries.join(" ") + " ";
        let first_offset = offset_table.len();

        let dict = dictionary! {
            "Type" => "ObjStm",
            "N" => self.objects.len() as i64,
            "First" => first_offset as i64,
        };

        let mut stream = Stream::new(dict, content);

        // Apply compression - object streams should always be compressed
        if self.compression_level > 0 {
            // Force compression by setting Filter directly
            use flate2::Compression;
            use flate2::write::ZlibEncoder;
            use std::io::prelude::*;

            let compression = match self.compression_level {
                0 => Compression::none(),
                1..=3 => Compression::fast(),
                4..=6 => Compression::default(),
                _ => Compression::best(),
            };

            let mut encoder = ZlibEncoder::new(Vec::new(), compression);
            encoder.write_all(&stream.content)?;
            let compressed = encoder.finish()?;

            stream.dict.set("Filter", "FlateDecode");
            stream.set_content(compressed);
        }

        Ok(stream)
    }

    /// Check if an object can be compressed into an object stream
    pub fn can_be_compressed(id: ObjectId, obj: &Object, doc: &Document) -> bool {
        // Rule 1: Stream objects cannot be compressed
        if matches!(obj, Object::Stream(_)) {
            return false;
        }

        // Rule 2: Objects with non-zero generation cannot be compressed
        if id.1 != 0 {
            return false;
        }

        // Rule 3: Only encryption dictionary cannot be compressed from trailer references
        if let Ok(Object::Reference(encrypt_ref)) = doc.trailer.get(b"Encrypt")
            && id == *encrypt_ref
        {
            return false;
        }

        // Rule 4: Specific object types that cannot be compressed
        if let Object::Dictionary(dict) = obj
            && let Ok(type_obj) = dict.get(b"Type")
            && let Ok(type_name) = type_obj.as_name()
        {
            match type_name {
                // Cross-reference streams and object streams cannot be compressed
                b"XRef" => return false,
                b"ObjStm" => return false,

                // Catalog can only be excluded in linearized PDFs
                b"Catalog" if Self::is_linearized(doc) => {
                    return false;
                }
                b"Catalog" => {}

                // Page, Pages, and all other types CAN be compressed
                _ => {}
            }
        }

        // Default: Allow compression
        true
    }

    /// Check if a PDF document is linearized
    fn is_linearized(doc: &Document) -> bool {
        // In a linearized PDF, the first object after the header should be a
        // linearization dictionary with /Linearized entry
        // For simplicity, we check if any object has a /Linearized entry
        for obj in doc.objects.values() {
            if let Object::Dictionary(dict) = obj
                && dict.has(b"Linearized")
            {
                return true;
            }
        }
        false
    }
}

impl ObjectStreamBuilder {
    /// Set the maximum number of objects per stream
    pub fn max_objects(mut self, max: usize) -> Self {
        self.max_objects = max;
        self
    }

    /// Set the compression level (0-9)
    pub fn compression_level(mut self, level: u32) -> Self {
        self.compression_level = level;
        self
    }

    /// Build the ObjectStream
    pub fn build(self) -> ObjectStream {
        ObjectStream {
            objects: BTreeMap::new(),
            diagnostics: ObjectStreamDiagnostics::default(),
            max_objects: self.max_objects,
            compression_level: self.compression_level,
        }
    }

    /// Get the current max_objects setting
    pub fn get_max_objects(&self) -> usize {
        self.max_objects
    }

    /// Get the current compression_level setting
    pub fn get_compression_level(&self) -> u32 {
        self.compression_level
    }
}
