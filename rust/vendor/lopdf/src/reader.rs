use log::{error, warn};
use std::cmp;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::convert::TryInto;
#[cfg(not(feature = "async"))]
use std::fs::File;
#[cfg(not(feature = "async"))]
use std::io::Read;
use std::ops::Range;
use std::path::Path;
use std::sync::Arc;
use std::sync::Mutex;

#[cfg(feature = "rayon")]
use rayon::prelude::*;
#[cfg(feature = "async")]
use tokio::fs::File;
#[cfg(feature = "async")]
use tokio::io::{AsyncRead, AsyncReadExt};
#[cfg(feature = "async")]
use tokio::pin;

use crate::common_data_structures;
use crate::encryption::{self, EncryptionState};
use crate::error::{ParseError, XrefError};
use crate::load_options::{FilterFunc, LoadOptions};
use crate::object_stream::{ObjectStream, ObjectStreamDiagnostics};
use crate::parser;
use crate::xref::{Xref, XrefEntry, XrefType};
use crate::{Dictionary, Document, Error, IncrementalDocument, Object, ObjectId, Result};

const MAX_DIAGNOSTIC_FAILURE_CLASSIFICATIONS: usize = 64;

#[cfg(test)]
static OBJECT_STREAM_PARSE_COUNT: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

#[cfg(not(feature = "async"))]
impl Document {
    /// Load with an application-selected object graph. The loader can use
    /// `Reader::load_object` after xrefs/trailer have been resolved. Encrypted
    /// documents retain the normal decryption path.
    pub fn load_with_object_loader<P: AsRef<Path>, F>(path: P, options: LoadOptions, loader: F) -> Result<Document>
    where
        F: FnOnce(&mut Reader<'_>) -> Result<()>,
    {
        let mut file = File::open(path)?;
        let mut input = Vec::with_capacity(file.metadata()?.len() as usize);
        file.read_to_end(&mut input)?;
        let buffer = Arc::new(input);
        Reader {
            buffer: &buffer,
            shared_buffer: Some(buffer.clone()),
            document: Document::new(),
            encryption_state: None,
            raw_objects: BTreeMap::new(),
            password: options.password,
            strict: options.strict,
            max_decompressed_size: options.max_decompressed_size,
            parser_policy: options.parser_policy,
            real_parser: options.real_parser,
            normal_offsets: Vec::new(),
            loaded_object_stream_containers: HashSet::new(),
        }
        .read_with_loader(options.filter, Some(loader))
    }
    /// Load a PDF document from a specified file path.
    #[inline]
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Document> {
        Self::load_with_options(path, LoadOptions::default())
    }

    /// Load a PDF document from a specified file path with the given options.
    #[inline]
    pub fn load_with_options<P: AsRef<Path>>(path: P, options: LoadOptions) -> Result<Document> {
        let file = File::open(path)?;
        let capacity = Some(file.metadata()?.len() as usize);
        Self::load_internal(file, capacity, options)
    }

    /// Load a PDF document from a specified file path with a password for encrypted PDFs.
    #[inline]
    pub fn load_with_password<P: AsRef<Path>>(path: P, password: &str) -> Result<Document> {
        Self::load_with_options(path, LoadOptions::with_password(password))
    }

    #[deprecated(since = "0.41.0", note = "Use load_with_options instead")]
    #[inline]
    pub fn load_filtered<P: AsRef<Path>>(path: P, filter_func: FilterFunc) -> Result<Document> {
        Self::load_with_options(path, LoadOptions::with_filter(filter_func))
    }

    /// Load a PDF document from an arbitrary source.
    #[inline]
    pub fn load_from<R: Read>(source: R) -> Result<Document> {
        Self::load_from_with_options(source, LoadOptions::default())
    }

    /// Load a PDF document from an arbitrary source with the given options.
    #[inline]
    pub fn load_from_with_options<R: Read>(source: R, options: LoadOptions) -> Result<Document> {
        Self::load_internal(source, None, options)
    }

    /// Load a PDF document from an arbitrary source with a password for encrypted PDFs.
    #[deprecated(since = "0.41.0", note = "Use load_from_with_options instead")]
    #[inline]
    pub fn load_from_with_password<R: Read>(source: R, password: &str) -> Result<Document> {
        Self::load_from_with_options(source, LoadOptions::with_password(password))
    }

    fn load_internal<R: Read>(mut source: R, capacity: Option<usize>, options: LoadOptions) -> Result<Document> {
        let mut buffer = capacity.map(Vec::with_capacity).unwrap_or_default();
        source.read_to_end(&mut buffer)?;

        Reader {
            buffer: &buffer,
            shared_buffer: None,
            document: Document::new(),
            encryption_state: None,
            raw_objects: BTreeMap::new(),
            password: options.password,
            strict: options.strict,
            max_decompressed_size: options.max_decompressed_size,
            parser_policy: options.parser_policy,
            real_parser: options.real_parser,
            normal_offsets: Vec::new(),
            loaded_object_stream_containers: HashSet::new(),
        }
        .read(options.filter)
    }

    /// Load a PDF document from a memory slice.
    pub fn load_mem(buffer: &[u8]) -> Result<Document> {
        Self::load_mem_with_options(buffer, LoadOptions::default())
    }

    /// Load a PDF document from a memory slice with the given options.
    pub fn load_mem_with_options(buffer: &[u8], options: LoadOptions) -> Result<Document> {
        Reader {
            buffer,
            shared_buffer: None,
            document: Document::new(),
            encryption_state: None,
            raw_objects: BTreeMap::new(),
            password: options.password,
            strict: options.strict,
            max_decompressed_size: options.max_decompressed_size,
            parser_policy: options.parser_policy,
            real_parser: options.real_parser,
            normal_offsets: Vec::new(),
            loaded_object_stream_containers: HashSet::new(),
        }
        .read(options.filter)
    }

    /// Load a PDF document from a memory slice with a password for encrypted PDFs.
    #[deprecated(since = "0.41.0", note = "Use load_mem_with_options instead")]
    pub fn load_mem_with_password(buffer: &[u8], password: &str) -> Result<Document> {
        Self::load_mem_with_options(buffer, LoadOptions::with_password(password))
    }

    /// Load PDF metadata (title and page count) without loading the entire document.
    /// This is much faster for large PDFs when you only need basic information.
    #[inline]
    pub fn load_metadata<P: AsRef<Path>>(path: P) -> Result<PdfMetadata> {
        let file = File::open(path)?;
        let capacity = Some(file.metadata()?.len() as usize);
        Self::load_metadata_internal(file, capacity, None)
    }

    /// Load PDF metadata from a file path with a password for encrypted PDFs.
    #[inline]
    pub fn load_metadata_with_password<P: AsRef<Path>>(path: P, password: &str) -> Result<PdfMetadata> {
        let file = File::open(path)?;
        let capacity = Some(file.metadata()?.len() as usize);
        Self::load_metadata_internal(file, capacity, Some(password.to_string()))
    }

    /// Load PDF metadata from an arbitrary source without loading the entire document.
    #[inline]
    pub fn load_metadata_from<R: Read>(source: R) -> Result<PdfMetadata> {
        Self::load_metadata_internal(source, None, None)
    }

    /// Load PDF metadata from an arbitrary source with a password for encrypted PDFs.
    #[inline]
    pub fn load_metadata_from_with_password<R: Read>(source: R, password: &str) -> Result<PdfMetadata> {
        Self::load_metadata_internal(source, None, Some(password.to_string()))
    }

    /// Load PDF metadata from a memory slice without loading the entire document.
    #[inline]
    pub fn load_metadata_mem(buffer: &[u8]) -> Result<PdfMetadata> {
        Reader {
            buffer,
            shared_buffer: None,
            document: Document::new(),
            encryption_state: None,
            raw_objects: BTreeMap::new(),
            password: None,
            strict: false,
            max_decompressed_size: None,
            parser_policy: None,
            real_parser: None,
            normal_offsets: Vec::new(),
            loaded_object_stream_containers: HashSet::new(),
        }
        .read_metadata()
    }

    /// Load PDF metadata from a memory slice with a password for encrypted PDFs.
    #[inline]
    pub fn load_metadata_mem_with_password(buffer: &[u8], password: &str) -> Result<PdfMetadata> {
        Reader {
            buffer,
            shared_buffer: None,
            document: Document::new(),
            encryption_state: None,
            raw_objects: BTreeMap::new(),
            password: Some(password.to_string()),
            strict: false,
            max_decompressed_size: None,
            parser_policy: None,
            real_parser: None,
            normal_offsets: Vec::new(),
            loaded_object_stream_containers: HashSet::new(),
        }
        .read_metadata()
    }

    fn load_metadata_internal<R: Read>(
        mut source: R, capacity: Option<usize>, password: Option<String>,
    ) -> Result<PdfMetadata> {
        let mut buffer = capacity.map(Vec::with_capacity).unwrap_or_default();
        source.read_to_end(&mut buffer)?;

        Reader {
            buffer: &buffer,
            shared_buffer: None,
            document: Document::new(),
            encryption_state: None,
            raw_objects: BTreeMap::new(),
            password,
            strict: false,
            max_decompressed_size: None,
            parser_policy: None,
            real_parser: None,
            normal_offsets: Vec::new(),
            loaded_object_stream_containers: HashSet::new(),
        }
        .read_metadata()
    }
}

#[cfg(feature = "async")]
impl Document {
    pub async fn load<P: AsRef<Path>>(path: P) -> Result<Document> {
        Self::load_with_options(path, LoadOptions::default()).await
    }

    /// Load a PDF document from a specified file path with the given options.
    pub async fn load_with_options<P: AsRef<Path>>(path: P, options: LoadOptions) -> Result<Document> {
        let file = File::open(path).await?;
        let metadata = file.metadata().await?;
        let capacity = Some(metadata.len() as usize);
        Self::load_internal(file, capacity, options).await
    }

    /// Load a PDF document from a specified file path with a password for encrypted PDFs.
    pub async fn load_with_password<P: AsRef<Path>>(path: P, password: &str) -> Result<Document> {
        Self::load_with_options(path, LoadOptions::with_password(password)).await
    }

    #[deprecated(since = "0.41.0", note = "Use load_with_options instead")]
    pub async fn load_filtered<P: AsRef<Path>>(path: P, filter_func: FilterFunc) -> Result<Document> {
        Self::load_with_options(path, LoadOptions::with_filter(filter_func)).await
    }

    async fn load_internal<R: AsyncRead>(source: R, capacity: Option<usize>, options: LoadOptions) -> Result<Document> {
        pin!(source);

        let mut buffer = capacity.map(Vec::with_capacity).unwrap_or_default();
        source.read_to_end(&mut buffer).await?;

        Reader {
            buffer: &buffer,
            shared_buffer: None,
            document: Document::new(),
            encryption_state: None,
            raw_objects: BTreeMap::new(),
            password: options.password,
            strict: options.strict,
            max_decompressed_size: options.max_decompressed_size,
            parser_policy: options.parser_policy,
            real_parser: options.real_parser,
            normal_offsets: Vec::new(),
            loaded_object_stream_containers: HashSet::new(),
        }
        .read(options.filter)
    }

    /// Load a PDF document from a memory slice.
    pub fn load_mem(buffer: &[u8]) -> Result<Document> {
        Self::load_mem_with_options(buffer, LoadOptions::default())
    }

    /// Load a PDF document from a memory slice with the given options.
    pub fn load_mem_with_options(buffer: &[u8], options: LoadOptions) -> Result<Document> {
        Reader {
            buffer,
            shared_buffer: None,
            document: Document::new(),
            encryption_state: None,
            raw_objects: BTreeMap::new(),
            password: options.password,
            strict: options.strict,
            max_decompressed_size: options.max_decompressed_size,
            parser_policy: options.parser_policy,
            real_parser: options.real_parser,
            normal_offsets: Vec::new(),
            loaded_object_stream_containers: HashSet::new(),
        }
        .read(options.filter)
    }

    /// Load PDF metadata (title and page count) without loading the entire document.
    /// This is much faster for large PDFs when you only need basic information.
    #[inline]
    pub async fn load_metadata<P: AsRef<Path>>(path: P) -> Result<PdfMetadata> {
        let file = File::open(path).await?;
        let metadata = file.metadata().await?;
        let capacity = Some(metadata.len() as usize);
        Self::load_metadata_internal(file, capacity, None).await
    }

    /// Load PDF metadata from a file path with a password for encrypted PDFs.
    #[inline]
    pub async fn load_metadata_with_password<P: AsRef<Path>>(path: P, password: &str) -> Result<PdfMetadata> {
        let file = File::open(path).await?;
        let metadata = file.metadata().await?;
        let capacity = Some(metadata.len() as usize);
        Self::load_metadata_internal(file, capacity, Some(password.to_string())).await
    }

    /// Load PDF metadata from an arbitrary source without loading the entire document.
    #[inline]
    pub async fn load_metadata_from<R: AsyncRead>(source: R) -> Result<PdfMetadata> {
        Self::load_metadata_internal(source, None, None).await
    }

    /// Load PDF metadata from an arbitrary source with a password for encrypted PDFs.
    #[inline]
    pub async fn load_metadata_from_with_password<R: AsyncRead>(source: R, password: &str) -> Result<PdfMetadata> {
        Self::load_metadata_internal(source, None, Some(password.to_string())).await
    }

    /// Load PDF metadata from a memory slice without loading the entire document.
    #[inline]
    pub fn load_metadata_mem(buffer: &[u8]) -> Result<PdfMetadata> {
        Reader {
            buffer,
            shared_buffer: None,
            document: Document::new(),
            encryption_state: None,
            raw_objects: BTreeMap::new(),
            password: None,
            strict: false,
            max_decompressed_size: None,
            parser_policy: None,
            real_parser: None,
            normal_offsets: Vec::new(),
            loaded_object_stream_containers: HashSet::new(),
        }
        .read_metadata()
    }

    /// Load PDF metadata from a memory slice with a password for encrypted PDFs.
    #[inline]
    pub fn load_metadata_mem_with_password(buffer: &[u8], password: &str) -> Result<PdfMetadata> {
        Reader {
            buffer,
            shared_buffer: None,
            document: Document::new(),
            encryption_state: None,
            raw_objects: BTreeMap::new(),
            password: Some(password.to_string()),
            strict: false,
            max_decompressed_size: None,
            parser_policy: None,
            real_parser: None,
            normal_offsets: Vec::new(),
            loaded_object_stream_containers: HashSet::new(),
        }
        .read_metadata()
    }

    async fn load_metadata_internal<R: AsyncRead>(
        source: R, capacity: Option<usize>, password: Option<String>,
    ) -> Result<PdfMetadata> {
        pin!(source);

        let mut buffer = capacity.map(Vec::with_capacity).unwrap_or_default();
        source.read_to_end(&mut buffer).await?;

        Reader {
            buffer: &buffer,
            shared_buffer: None,
            document: Document::new(),
            encryption_state: None,
            raw_objects: BTreeMap::new(),
            password,
            strict: false,
            max_decompressed_size: None,
            parser_policy: None,
            real_parser: None,
            normal_offsets: Vec::new(),
            loaded_object_stream_containers: HashSet::new(),
        }
        .read_metadata()
    }
}

impl TryInto<Document> for &[u8] {
    type Error = Error;

    fn try_into(self) -> Result<Document> {
        Reader {
            buffer: self,
            shared_buffer: None,
            document: Document::new(),
            encryption_state: None,
            raw_objects: BTreeMap::new(),
            password: None,
            strict: false,
            max_decompressed_size: None,
            parser_policy: None,
            real_parser: None,
            normal_offsets: Vec::new(),
            loaded_object_stream_containers: HashSet::new(),
        }
        .read(None)
    }
}

#[cfg(not(feature = "async"))]
impl IncrementalDocument {
    /// Load a PDF document from a specified file path.
    #[inline]
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path)?;
        let capacity = Some(file.metadata()?.len() as usize);
        Self::load_internal(file, capacity)
    }

    /// Load a PDF document from an arbitrary source.
    #[inline]
    pub fn load_from<R: Read>(source: R) -> Result<Self> {
        Self::load_internal(source, None)
    }

    fn load_internal<R: Read>(mut source: R, capacity: Option<usize>) -> Result<Self> {
        let mut buffer = capacity.map(Vec::with_capacity).unwrap_or_default();
        source.read_to_end(&mut buffer)?;

        let document = Reader {
            buffer: &buffer,
            shared_buffer: None,
            document: Document::new(),
            encryption_state: None,
            raw_objects: BTreeMap::new(),
            password: None,
            strict: false,
            max_decompressed_size: None,
            parser_policy: None,
            real_parser: None,
            normal_offsets: Vec::new(),
            loaded_object_stream_containers: HashSet::new(),
        }
        .read(None)?;

        Ok(IncrementalDocument::create_from(buffer, document))
    }

    /// Load a PDF document from a memory slice.
    pub fn load_mem(buffer: &[u8]) -> Result<Document> {
        buffer.try_into()
    }
}

#[cfg(feature = "async")]
impl IncrementalDocument {
    /// Load a PDF document from a specified file path.
    #[inline]
    pub async fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path).await?;
        let metadata = file.metadata().await?;
        let capacity = Some(metadata.len() as usize);
        Self::load_internal(file, capacity).await
    }

    /// Load a PDF document from an arbitrary source.
    #[inline]
    pub async fn load_from<R: AsyncRead>(source: R) -> Result<Self> {
        Self::load_internal(source, None).await
    }

    async fn load_internal<R: AsyncRead>(source: R, capacity: Option<usize>) -> Result<Self> {
        pin!(source);

        let mut buffer = capacity.map(Vec::with_capacity).unwrap_or_default();
        source.read_to_end(&mut buffer).await?;

        let document = Reader {
            buffer: &buffer,
            shared_buffer: None,
            document: Document::new(),
            encryption_state: None,
            raw_objects: BTreeMap::new(),
            password: None,
            strict: false,
            max_decompressed_size: None,
            parser_policy: None,
            real_parser: None,
            normal_offsets: Vec::new(),
            loaded_object_stream_containers: HashSet::new(),
        }
        .read(None)?;

        Ok(IncrementalDocument::create_from(buffer, document))
    }

    /// Load a PDF document from a memory slice.
    pub fn load_mem(buffer: &[u8]) -> Result<Document> {
        buffer.try_into()
    }
}

impl TryInto<IncrementalDocument> for &[u8] {
    type Error = Error;

    fn try_into(self) -> Result<IncrementalDocument> {
        let document = Reader {
            buffer: self,
            shared_buffer: None,
            document: Document::new(),
            encryption_state: None,
            raw_objects: BTreeMap::new(),
            password: None,
            strict: false,
            max_decompressed_size: None,
            parser_policy: None,
            real_parser: None,
            normal_offsets: Vec::new(),
            loaded_object_stream_containers: HashSet::new(),
        }
        .read(None)?;

        Ok(IncrementalDocument::create_from(self.to_vec(), document))
    }
}

pub struct Reader<'a> {
    pub parser_policy: Option<crate::resource::Policy>,
    real_parser: Option<fn(&str) -> f32>,
    pub buffer: &'a [u8],
    shared_buffer: Option<Arc<Vec<u8>>>,
    pub document: Document,
    pub encryption_state: Option<EncryptionState>,
    pub raw_objects: BTreeMap<ObjectId, Range<usize>>, // Input ranges for encrypted objects
    pub password: Option<String>,                 // Password for encrypted PDFs
    pub strict: bool,                             // Reject non-conforming PDFs when true
    /// Maximum decompressed size, in bytes, allowed for any single stream
    /// (object streams and cross-reference streams) decoded during loading.
    /// `None` (the default) applies no limit; set it to guard against
    /// decompression bombs. See [`crate::LoadOptions`].
    pub max_decompressed_size: Option<usize>,
    /// Sorted unique byte offsets of every `XrefEntry::Normal` entry in the
    /// final cross-reference table. Built once when the table is complete so
    /// object-boundary lookups can binary-search the successor offset instead
    /// of scanning all xref entries on each call.
    normal_offsets: Vec<usize>,
    loaded_object_stream_containers: HashSet<u32>,
}

fn collect_encrypted_object_ranges(
    buffer: &[u8],
    object_offsets: Vec<usize>,
) -> (
    BTreeMap<ObjectId, Range<usize>>,
    usize,
    Vec<usize>,
) {
    let mut occurrences = BTreeMap::<usize, usize>::new();
    for offset in &object_offsets {
        *occurrences.entry(*offset).or_insert(0) += 1;
    }
    let mut distinct: Vec<_> = occurrences.keys().copied().collect();
    distinct.sort_unstable();

    let mut ranges = BTreeMap::new();
    let mut offset_failures = 0usize;
    let mut segment_failures = Vec::new();
    for offset in object_offsets {
        if occurrences.get(&offset).copied() != Some(1)
            || offset >= buffer.len()
        {
            offset_failures += 1;
            continue;
        }
        let next = distinct.partition_point(|&candidate| candidate <= offset);
        let end = distinct
            .get(next)
            .copied()
            .unwrap_or(buffer.len())
            .min(buffer.len());
        if offset >= end {
            offset_failures += 1;
            continue;
        }
        match extract_raw_object_range(buffer, offset, end) {
            Ok((id, range)) => {
                ranges.insert(id, range);
            }
            Err(_) => segment_failures.push(offset),
        }
    }
    (ranges, offset_failures, segment_failures)
}

fn extract_raw_object_range(
    buffer: &[u8],
    offset: usize,
    end: usize,
) -> Result<(ObjectId, Range<usize>)> {
    if offset >= end || end > buffer.len() {
        return Err(Error::InvalidOffset(offset));
    }
    let slice = &buffer[offset..end];
    let mut pos = 0usize;
    while pos < slice.len() && slice[pos].is_ascii_whitespace() {
        pos += 1;
    }
    let id = Reader::parse_object_header(&slice[pos..])
        .ok_or(Error::Parse(ParseError::InvalidXref))?;

    let endobj = b"endobj";
    let relative_end = slice
        .windows(endobj.len())
        .position(|window| window == endobj)
        .map(|position| position + endobj.len())
        .ok_or(Error::Parse(ParseError::InvalidXref))?;
    Ok((id, offset..offset + relative_end))
}

impl Reader<'_> {
    fn record_object_stream_diagnostics(&mut self, diagnostics: ObjectStreamDiagnostics) {
        self.document.parser_diagnostics.invalid_objstm_members +=
            diagnostics.invalid_members();
        self.document
            .parser_diagnostics
            .invalid_objstm_offset_members += diagnostics.out_of_bounds_or_duplicate;
        self.document
            .parser_diagnostics
            .invalid_objstm_segment_b20_would_accept +=
            diagnostics.segment_b20_would_accept;
        self.document
            .parser_diagnostics
            .invalid_objstm_segment_both_fail += diagnostics.segment_both_fail;
    }

    fn legacy_encrypted_object_succeeds(&self, offset: usize) -> bool {
        let Ok((_, range)) = extract_raw_object_range(self.buffer, offset, self.buffer.len())
        else {
            return false;
        };
        matches!(
            crate::resource::optional(self.parse_raw_object(&self.buffer[range])),
            Ok(Some(_))
        )
    }

    fn record_encrypted_segment_failure(
        &mut self,
        offset: usize,
        classifications_used: &mut usize,
    ) {
        self.document.parser_diagnostics.invalid_encrypted_objects += 1;
        if *classifications_used < MAX_DIAGNOSTIC_FAILURE_CLASSIFICATIONS {
            if self.legacy_encrypted_object_succeeds(offset) {
                self.document
                    .parser_diagnostics
                    .invalid_encrypted_segment_b20_would_accept += 1;
            } else {
                self.document
                    .parser_diagnostics
                    .invalid_encrypted_segment_both_fail += 1;
            }
            *classifications_used += 1;
        }
    }
}

/// Maximum allowed embedding of literal strings.
pub const MAX_BRACKET: usize = 100;

pub const MAX_NESTING_DEPTH: usize = 100;

/// Cap on reconstructed cross-reference entries, bounding memory on hostile inputs.
const MAX_RECONSTRUCTED_OBJECTS: usize = 1_000_000;

/// Cap on how many trailing `trailer` dictionaries are inspected for a `/Root`.
const MAX_TRAILER_CANDIDATES: usize = 16;

/// PDF metadata extracted without loading the entire document.
/// This is useful for quickly getting basic information about large PDFs.
#[derive(Debug, Clone)]
pub struct PdfMetadata {
    /// Document title from Info dictionary
    pub title: Option<String>,
    /// Document author from Info dictionary
    pub author: Option<String>,
    /// Document subject from Info dictionary
    pub subject: Option<String>,
    /// Document keywords from Info dictionary
    pub keywords: Option<String>,
    /// Application that created the document
    pub creator: Option<String>,
    /// Application that produced the document
    pub producer: Option<String>,
    /// Document creation date (PDF date format: D:YYYYMMDDHHmmSSOHH'mm')
    pub creation_date: Option<String>,
    /// Document modification date (PDF date format: D:YYYYMMDDHHmmSSOHH'mm')
    pub modification_date: Option<String>,
    /// Custom Info dictionary entries that are not part of the standard set above.
    ///
    /// Producers commonly use the Info dictionary to attach application-specific
    /// metadata (for example, Microsoft Information Protection labels stored as
    /// `MSIP_Label_{GUID}_{Property}` keys). Such entries used to be discarded
    /// by the metadata-only loader; they are now preserved here as raw `Object`
    /// values so callers can inspect or decode them as needed.
    pub custom: HashMap<Vec<u8>, Object>,
    /// Number of pages in the document
    pub page_count: u32,
    /// PDF version
    pub version: String,
    /// Whether the document declares encryption (an `/Encrypt` entry in the
    /// trailer). When `true`, string and stream contents are encrypted; the
    /// standard Info fields above are populated only if the document could be
    /// decrypted — e.g. an empty user password, or a password supplied via a
    /// `*_with_password` loader.
    pub encrypted: bool,
}

struct InfoMetadata {
    title: Option<String>,
    author: Option<String>,
    subject: Option<String>,
    keywords: Option<String>,
    creator: Option<String>,
    producer: Option<String>,
    creation_date: Option<String>,
    modification_date: Option<String>,
    custom: HashMap<Vec<u8>, Object>,
}

impl InfoMetadata {
    fn empty() -> Self {
        Self {
            title: None,
            author: None,
            subject: None,
            keywords: None,
            creator: None,
            producer: None,
            creation_date: None,
            modification_date: None,
            custom: HashMap::new(),
        }
    }
}

/// Standard Info dictionary keys defined in PDF 1.7 §14.3.3 ("Document
/// Information Dictionary"). Any other key is preserved on
/// `PdfMetadata::custom` rather than dropped.
const STANDARD_INFO_KEYS: &[&[u8]] = &[
    b"Title",
    b"Author",
    b"Subject",
    b"Keywords",
    b"Creator",
    b"Producer",
    b"CreationDate",
    b"ModDate",
    b"Trapped",
];

impl Reader<'_> {
    /// Read metadata (title and page count) without loading the entire document.
    /// This is much faster for large PDFs when you only need basic information.
    ///
    /// For encrypted PDFs, use `Document::load_metadata_with_password()` instead.
    pub fn read_metadata(mut self) -> Result<PdfMetadata> {
        let offset = self.buffer.windows(5).position(|w| w == b"%PDF-").unwrap_or(0);
        self.buffer = &self.buffer[offset..];

        let version = parser::header(self.buffer, self.strict).ok_or(ParseError::InvalidFileHeader)?;

        self.document.parser_policy = self.parser_policy;
        let (xref, trailer) = match self.resolve_xref_and_trailer() {
            Ok(resolved) => resolved,
            Err(err) if err.is_resource_limit() => return Err(err),
            Err(err) => match self.reconstruct_xref_and_trailer() {
                Some(reconstructed) => {
                    warn!("cross-reference resolution failed ({err}); recovered by scanning for indirect objects");
                    reconstructed
                }
                None => return Err(err),
            },
        };

        self.set_reference_table(xref);
        self.document.trailer = trailer.clone();
        let encrypted = self.document.trailer.get(b"Encrypt").is_ok();
        // For encrypted PDFs, set up decryption so the Info dictionary can be
        // read. If the document is protected by a password we cannot supply,
        // still report what we have (notably `encrypted`) rather than failing
        // the whole call; callers needing the Info fields should use a
        // `*_with_password` loader.
        let needs_password = if encrypted {
            match self.setup_encryption_for_metadata() {
                Ok(()) => false,
                // No usable password was available (empty password failed and
                // none was supplied): report `encrypted` without the Info
                // fields. A wrong supplied password still surfaces as an error.
                Err(Error::Unimplemented(_)) => true,
                Err(e) => return Err(e),
            }
        } else {
            false
        };

        let info_metadata = if needs_password {
            InfoMetadata::empty()
        } else {
            self.extract_info_metadata()?
        };
        let page_count = if needs_password { 0 } else { self.extract_page_count()? };

        Ok(PdfMetadata {
            title: info_metadata.title,
            author: info_metadata.author,
            subject: info_metadata.subject,
            keywords: info_metadata.keywords,
            creator: info_metadata.creator,
            producer: info_metadata.producer,
            creation_date: info_metadata.creation_date,
            modification_date: info_metadata.modification_date,
            custom: info_metadata.custom,
            page_count,
            version,
            encrypted,
        })
    }

    fn extract_info_metadata(&self) -> Result<InfoMetadata> {
        let info_ref = match self.document.trailer.get(b"Info") {
            Ok(obj) => obj.as_reference().ok(),
            Err(_) => return Ok(InfoMetadata::empty()),
        };

        let info_id = match info_ref {
            Some(id) => id,
            None => return Ok(InfoMetadata::empty()),
        };

        let mut already_seen = HashSet::new();
        let info_obj = match self.get_object(info_id, &mut already_seen) {
            Ok(obj) => obj,
            Err(_) => return Ok(InfoMetadata::empty()),
        };

        let info_dict = match info_obj.as_dict() {
            Ok(dict) => dict,
            Err(_) => return Ok(InfoMetadata::empty()),
        };

        let mut custom = HashMap::new();
        for (key, value) in info_dict.iter() {
            if STANDARD_INFO_KEYS.contains(&key.as_slice()) {
                continue;
            }
            custom.insert(key.clone(), value.clone());
        }

        Ok(InfoMetadata {
            title: Self::extract_string_field(info_dict, b"Title"),
            author: Self::extract_string_field(info_dict, b"Author"),
            subject: Self::extract_string_field(info_dict, b"Subject"),
            keywords: Self::extract_string_field(info_dict, b"Keywords"),
            creator: Self::extract_string_field(info_dict, b"Creator"),
            producer: Self::extract_string_field(info_dict, b"Producer"),
            creation_date: Self::extract_string_field(info_dict, b"CreationDate"),
            modification_date: Self::extract_string_field(info_dict, b"ModDate"),
            custom,
        })
    }

    fn extract_string_field(dict: &Dictionary, key: &[u8]) -> Option<String> {
        match dict.get(key) {
            Ok(obj) => match obj {
                Object::String(_bytes, _) => common_data_structures::decode_text_string(obj).ok(),
                _ => None,
            },
            Err(_) => None,
        }
    }

    fn extract_page_count(&self) -> Result<u32> {
        let root_ref = match self.document.trailer.get(b"Root").and_then(Object::as_reference) {
            Ok(id) => id,
            Err(_) => return Ok(0),
        };

        let mut already_seen = HashSet::new();
        let catalog_obj = match self.get_object(root_ref, &mut already_seen) {
            Ok(obj) => obj,
            Err(_) => return Ok(0),
        };

        let catalog_dict = match catalog_obj.as_dict() {
            Ok(dict) => dict,
            Err(_) => return Ok(0),
        };

        let pages_ref = match catalog_dict.get(b"Pages").and_then(Object::as_reference) {
            Ok(id) => id,
            Err(_) => return Ok(0),
        };

        self.get_pages_tree_count(pages_ref, &mut HashSet::new(), 0).or(Ok(0))
    }

    fn get_pages_tree_count(&self, pages_id: ObjectId, seen: &mut HashSet<ObjectId>, depth: usize) -> Result<u32> {
        // `seen` catches a repeated node (a cycle); it does not bound the depth of a long,
        // non-cyclic /Pages chain with no /Count, which recurses once per level. Cap that
        // separately, matching this crate's existing MAX_NESTING_DEPTH budget.
        if depth >= MAX_NESTING_DEPTH {
            return Ok(0);
        }
        if seen.contains(&pages_id) {
            return Err(Error::ReferenceCycle(pages_id));
        }
        seen.insert(pages_id);

        let mut already_seen = HashSet::new();
        let pages_obj = match self.get_object(pages_id, &mut already_seen) {
            Ok(obj) => obj,
            Err(_) => return Ok(0),
        };

        let pages_dict = match pages_obj.as_dict() {
            Ok(dict) => dict,
            Err(_) => return Ok(0),
        };

        match pages_dict.get_type() {
            Ok(type_name) if type_name == b"Page" => Ok(1),
            Ok(type_name) if type_name == b"Pages" => {
                if let Ok(count_obj) = pages_dict.get(b"Count")
                    && let Ok(count) = count_obj.as_i64()
                    && count >= 0
                {
                    return Ok(count as u32);
                }

                let kids = match pages_dict.get(b"Kids").and_then(Object::as_array) {
                    Ok(arr) => arr,
                    Err(_) => return Ok(0),
                };

                let mut total = 0u32;
                for kid in kids.iter() {
                    if let Ok(kid_ref) = kid.as_reference()
                        && let Ok(count) = self.get_pages_tree_count(kid_ref, seen, depth + 1)
                    {
                        total += count;
                    }
                }
                Ok(total)
            }
            _ => Ok(1),
        }
    }

    /// Read whole document.
    pub fn read(self, filter_func: Option<FilterFunc>) -> Result<Document> {
        self.read_with_loader(filter_func, None::<fn(&mut Reader<'_>) -> Result<()>>)
    }

    fn read_with_loader<F>(mut self, filter_func: Option<FilterFunc>, loader: Option<F>) -> Result<Document>
    where
        F: FnOnce(&mut Reader<'_>) -> Result<()>,
    {
        let offset = self.buffer.windows(5).position(|w| w == b"%PDF-").unwrap_or(0);
        self.buffer = &self.buffer[offset..];

        // The document structure can be expressed in PEG as:
        //   document <- header indirect_object* xref trailer xref_start
        let version = parser::header(self.buffer, self.strict).ok_or(ParseError::InvalidFileHeader)?;

        //The binary_mark is in line 2 after the pdf version. If at other line number, then will be declared as invalid
        // pdf.
        if let Some(pos) = self.buffer.iter().position(|&byte| byte == b'\n')
            && let Some(binary_mark) = parser::binary_mark(&self.buffer[pos + 1..])
            && binary_mark.iter().all(|&byte| byte >= 128)
        {
            self.document.binary_mark = binary_mark;
        }

        // Fall back to reconstruction only after all standard resolution fails.
        self.document.parser_policy = self.parser_policy;
        let (xref, trailer) = match self.resolve_xref_and_trailer() {
            Ok(resolved) => resolved,
            Err(err) if err.is_resource_limit() => return Err(err),
            Err(err) => match self.reconstruct_xref_and_trailer() {
                Some(reconstructed) => {
                    warn!("cross-reference resolution failed ({err}); recovered by scanning for indirect objects");
                    reconstructed
                }
                None => return Err(err),
            },
        };

        self.document.version = version;
        self.document.max_id = xref.size - 1;
        self.document.trailer = trailer;
        self.set_reference_table(xref);

        // Check if encrypted
        let is_encrypted = self.document.trailer.get(b"Encrypt").is_ok();

        if is_encrypted {
            // For encrypted PDFs, use a special loading strategy
            self.load_encrypted_document(filter_func)?;
        } else {
            // For non-encrypted PDFs, use the normal loading
            if let Some(loader) = loader {
                loader(&mut self)?;
            } else {
                self.load_objects_raw(filter_func)?;
            }
        }

        // Object-stream members join `objects` only during loading, after
        // `max_id` was derived from the xref size. Keep the ceiling at least
        // at the highest loaded object so new ids cannot collide with live ones.
        if let Some(&max_loaded_id) = self.document.objects.keys().next_back() {
            self.document.max_id = self.document.max_id.max(max_loaded_id.0);
        }

        Ok(self.document)
    }

    /// Populate one object (or the matching members of its compressed container).
    /// Failed reads are left absent, as in lenient whole-document loading.
    pub fn load_object(&mut self, id: ObjectId) -> Result<()> {
        self.load_object_notifying(id, None)
    }

    /// As `load_object`, reporting every actually inserted object, including
    /// compressed siblings. The caller owns and drains this compact journal.
    pub fn load_object_with_new_ids(&mut self, id: ObjectId, added: &mut Vec<ObjectId>) -> Result<()> {
        self.load_object_notifying(id, Some(added))
    }

    fn load_object_notifying(&mut self, id: ObjectId, mut added: Option<&mut Vec<ObjectId>>) -> Result<()> {
        if self.document.objects.contains_key(&id) {
            return Ok(());
        }
        if let Some(XrefEntry::Compressed { container, .. }) = self.document.reference_table.get(id.0).cloned() {
            let container_id = (container, 0);
            if !self.loaded_object_stream_containers.insert(container) {
                return Ok(());
            }
            // Object-stream containers must be normal entries. Eager loading
            // ignores invalid compressed-container chains rather than recursing.
            if !matches!(
                self.document.reference_table.get(container),
                Some(XrefEntry::Normal { .. })
            ) {
                return Err(Error::MissingXrefEntry);
            }
            self.load_object_notifying(container_id, added.as_deref_mut())?;
            let stream = self.document.get_object(container_id)?.as_stream()?;
            #[cfg(test)]
            OBJECT_STREAM_PARSE_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let objects = parser::with_real_parser(self.real_parser, || {
                ObjectStream::new_with_policy(stream, self.max_decompressed_size, self.parser_policy.as_ref())
            }).map_err(|e| e.at_object(container_id))?;
            self.record_object_stream_diagnostics(objects.diagnostics);
            for (member, value) in objects.objects {
                if matches!(self.document.reference_table.get(member.0),Some(XrefEntry::Compressed{container:c,..}) if *c==container)
                {
                    if let std::collections::btree_map::Entry::Vacant(entry) = self.document.objects.entry(member) {
                        entry.insert(value);
                        if let Some(ids) = &mut added { ids.push(member); }
                    }
                }
            }
        } else {
            let object = self.get_object(id, &mut HashSet::new())?;
            self.document.objects.insert(id, object);
            if let Some(ids) = &mut added { ids.push(id); }
        }
        Ok(())
    }

    fn load_encrypted_document(&mut self, _filter_func: Option<FilterFunc>) -> Result<()> {
        // First, record raw object ranges without copying their payload bytes.
        let entries: Vec<_> = self
            .document
            .reference_table
            .entries
            .iter()
            .map(|(k, v)| (*k, v.clone()))
            .collect();

        let mut object_streams = Vec::new();
        let normal_offsets: Vec<_> = entries
            .iter()
            .filter_map(|(_, entry)| match entry {
                XrefEntry::Normal { offset, .. } => Some(*offset as usize),
                _ => None,
            })
            .collect();
        let (raw_objects, offset_failures, segment_failures) =
            collect_encrypted_object_ranges(self.buffer, normal_offsets);
        self.raw_objects = raw_objects;
        self.document.parser_diagnostics.invalid_encrypted_objects += offset_failures;
        self.document
            .parser_diagnostics
            .invalid_encrypted_offset_objects += offset_failures;
        let mut encrypted_classifications_used = 0usize;
        for offset in segment_failures {
            self.record_encrypted_segment_failure(offset, &mut encrypted_classifications_used);
        }

        for (obj_num, entry) in entries {
            match entry {
                XrefEntry::Normal { .. } => {}
                XrefEntry::Compressed { container, index } => {
                    // Store compressed object info for later processing
                    object_streams.push((obj_num, container, index));
                }
                XrefEntry::Free | XrefEntry::UnusableFree => {
                    // Skip free entries
                }
            }
        }

        self.parse_encryption_dictionary()?;

        if self.authenticate_and_setup_encryption(false)?.is_none() {
            return Ok(());
        }

        let mut post_auth_segment_failures = Vec::new();
        let mut post_auth_objstm_diagnostics = ObjectStreamDiagnostics::default();
        if let Some(ref state) = self.encryption_state {
            let encrypt_ref = self
                .document
                .trailer
                .get(b"Encrypt")
                .ok()
                .and_then(|o| o.as_reference().ok());

            let raw_entries: Vec<_> = self
                .raw_objects
                .iter()
                .map(|(&object_id, range)| (object_id, range.clone()))
                .collect();
            for (obj_id, raw_range) in raw_entries {
                if let Some(enc_ref) = encrypt_ref
                    && obj_id == enc_ref
                {
                    continue;
                }

                let raw_offset = raw_range.start;
                let raw_bytes = &self.buffer[raw_range];
                match crate::resource::optional(self.parse_raw_object(raw_bytes))? {
                    Some((id, mut obj)) => {
                        let _ = encryption::decrypt_object(state, obj_id, &mut obj);
                        self.document.objects.insert(id, obj);
                    }
                    None => post_auth_segment_failures.push(raw_offset),
                }
            }

            let mut streams_to_process: std::collections::HashMap<u32, Vec<(u32, u16)>> =
                std::collections::HashMap::new();
            for (obj_num, container_id, index) in object_streams {
                streams_to_process
                    .entry(container_id)
                    .or_default()
                    .push((obj_num, index));
            }

            for (container_id, objects_in_stream) in streams_to_process {
                if let Some(container_obj) = self.document.objects.get(&(container_id, 0))
                    && let Ok(stream) = container_obj.as_stream()
                {
                    match parser::with_real_parser(self.real_parser, || {
                        ObjectStream::new_with_policy(stream, self.max_decompressed_size, self.parser_policy.as_ref())
                    }) {
                        Ok(object_stream) => {
                            post_auth_objstm_diagnostics += object_stream.diagnostics;
                            for (obj_num, _index) in objects_in_stream {
                                let obj_id = (obj_num, 0);
                                if let Some(obj) = object_stream.objects.get(&obj_id) {
                                    self.document.objects.insert(obj_id, obj.clone());
                                }
                            }
                        }
                        Err(e) if e.is_resource_limit() => return Err(e),
                        Err(_e) => {}
                    }
                }
            }

            let mut stored_state = state.clone();
            // Record the /Encrypt dictionary's object id so that a later
            // incremental save can restore `/Encrypt N G R` in the appended
            // trailer. See `IncrementalDocument::save_internal`.
            stored_state.encrypt_object_id = encrypt_ref;
            self.document.encryption_state = Some(stored_state);

            if let Some(enc_ref) = encrypt_ref {
                self.document.objects.remove(&enc_ref);
            }
            self.document.trailer.remove(b"Encrypt");
        }
        for offset in post_auth_segment_failures {
            self.record_encrypted_segment_failure(offset, &mut encrypted_classifications_used);
        }
        self.record_object_stream_diagnostics(post_auth_objstm_diagnostics);

        Ok(())
    }

    fn parse_raw_object(&self, raw_bytes: &[u8]) -> Result<(ObjectId, Object)> {
        // Parse the raw bytes as an indirect object
        parser::with_real_parser(self.real_parser, || {
            parser::indirect_object(raw_bytes, 0, None, self, &mut HashSet::new(), None)
        })
    }

    /// Install a fully merged cross-reference table and derive its sorted
    /// offset index. All later object-boundary lookups go through that index,
    /// so it must be rebuilt whenever the table is replaced.
    fn set_reference_table(&mut self, xref: Xref) {
        let mut normal_offsets: Vec<usize> = xref
            .entries
            .values()
            .filter_map(|entry| match entry {
                XrefEntry::Normal { offset, .. } => Some(*offset as usize),
                _ => None,
            })
            .collect();
        normal_offsets.sort_unstable();
        normal_offsets.dedup();
        self.normal_offsets = normal_offsets;
        self.document.reference_table = xref;
    }

    fn load_objects_raw(&mut self, filter_func: Option<FilterFunc>) -> Result<()> {
        let is_encrypted = self.document.trailer.get(b"Encrypt").is_ok();
        let zero_length_streams = Mutex::new(vec![]);
        let object_streams = Mutex::new(vec![]);
        let invalid_objstm_members = Mutex::new(ObjectStreamDiagnostics::default());

        // Build a map of which container each compressed object belongs to
        // according to the xref. This prevents stale ObjStm copies (e.g., from
        // linearization first-page sections) from overriding the correct version.
        let compressed_obj_containers: BTreeMap<u32, u32> = self
            .document
            .reference_table
            .entries
            .iter()
            .filter_map(|(&id, entry)| {
                if let XrefEntry::Compressed { container, .. } = entry {
                    Some((id, *container))
                } else {
                    None
                }
            })
            .collect();
        let normal_offsets = &self.normal_offsets;

        let entries_filter_map = |(_, entry): (&_, &_)| -> Result<Option<(ObjectId, Object)>> {
            if let XrefEntry::Normal { offset, .. } = *entry {
                // read_object now handles decryption internally
                let offset = offset as usize;
                let next_index = normal_offsets.partition_point(|&next| next <= offset);
                let next_object = normal_offsets.get(next_index).copied();
                let end = self.object_end(offset, next_object);
                let result = self.read_object_to(offset, end, None, &mut HashSet::new());
                let (object_id, mut object) = match result {
                    Ok(obj) => obj,
                    Err(e) => {
                        if e.is_resource_limit() { return Err(e); }
                        if is_encrypted {
                            // Expected for some encrypted objects - but log which
                            // ones. These failures stay non-fatal even in strict
                            // mode because they do not indicate a malformed file.
                            warn!("Skipping encrypted object at offset {}: {:?}", offset, e);
                            return Ok(None);
                        }
                        // Lenient loading logs and skips malformed objects; strict loading fails.
                        error!("Object load error at offset {}: {e:?}", offset);
                        return if self.strict { Err(e) } else { Ok(None) };
                    }
                };
                if let Some(filter_func) = filter_func
                    && filter_func(object_id, &mut object).is_none()
                {
                    return Ok(None);
                }

                if let Ok(stream) = object.as_stream() {
                    if stream.dict.has_type(b"ObjStm") && !is_encrypted {
                        let obj_stream = match parser::with_real_parser(self.real_parser, || {
                            ObjectStream::new_with_policy(stream, self.max_decompressed_size, self.parser_policy.as_ref())
                        }) {
                            Ok(obj_stream) => obj_stream,
                            // Not an encryption-related loss, so it obeys the same
                            // strict-mode policy as any other load error.
                            Err(e) => {
                                if e.is_resource_limit() { return Err(e.at_object(object_id)); }
                                error!("Object stream load error for {object_id:?}: {e:?}");
                                return if self.strict { Err(e) } else { Ok(None) };
                            }
                        };
                        *invalid_objstm_members.lock().unwrap() += obj_stream.diagnostics;
                        let container_id = object_id.0;
                        let mut object_streams = object_streams.lock().unwrap();
                        if let Some(filter_func) = filter_func {
                            let objects: BTreeMap<(u32, u16), Object> = obj_stream
                                .objects
                                .into_iter()
                                .filter(|((obj_num, _), _)| {
                                    compressed_obj_containers
                                        .get(obj_num)
                                        .is_none_or(|&c| c == container_id)
                                })
                                .filter_map(|(object_id, mut object)| filter_func(object_id, &mut object))
                                .collect();
                            object_streams.extend(objects);
                        } else {
                            object_streams.extend(obj_stream.objects.into_iter().filter(|((obj_num, _), _)| {
                                compressed_obj_containers
                                    .get(obj_num)
                                    .is_none_or(|&c| c == container_id)
                            }));
                        }
                    } else if stream.content.is_empty() {
                        let mut zero_length_streams = zero_length_streams.lock().unwrap();
                        zero_length_streams.push(object_id);
                    }
                }

                Ok(Some((object_id, object)))
            } else {
                Ok(None)
            }
        };

        #[cfg(feature = "rayon")]
        {
            self.document.objects = self
                .document
                .reference_table
                .entries
                .par_iter()
                .map(entries_filter_map)
                .filter_map(Result::transpose)
                .collect::<Result<BTreeMap<_, _>>>()?;
        }
        #[cfg(not(feature = "rayon"))]
        {
            self.document.objects = self
                .document
                .reference_table
                .entries
                .iter()
                .map(entries_filter_map)
                .filter_map(Result::transpose)
                .collect::<Result<BTreeMap<_, _>>>()?;
        }

        // Only add entries, but never replace entries
        for (id, entry) in object_streams.into_inner().unwrap() {
            self.document.objects.entry(id).or_insert(entry);
        }

        for object_id in zero_length_streams.into_inner().unwrap() {
            let _ = self.read_stream_content(object_id);
        }
        let invalid_objstm_members = invalid_objstm_members.into_inner().unwrap();
        self.document.parser_diagnostics.invalid_objstm_members +=
            invalid_objstm_members.invalid_members();
        self.document
            .parser_diagnostics
            .invalid_objstm_offset_members +=
            invalid_objstm_members.out_of_bounds_or_duplicate;
        self.document
            .parser_diagnostics
            .invalid_objstm_segment_b20_would_accept +=
            invalid_objstm_members.segment_b20_would_accept;
        self.document
            .parser_diagnostics
            .invalid_objstm_segment_both_fail += invalid_objstm_members.segment_both_fail;

        Ok(())
    }

    fn read_stream_content(&mut self, object_id: ObjectId) -> Result<()> {
        let length = self.get_stream_length(object_id)?;
        let stream = self
            .document
            .get_object_mut(object_id)
            .and_then(Object::as_stream_mut)?;
        let start = stream
            .start_position
            .ok_or(Error::InvalidStream("missing start position".to_string()))?;

        if length < 0 {
            return Err(Error::InvalidStream("negative stream length.".to_string()));
        }

        let length = usize::try_from(length).map_err(|e| Error::NumericCast(e.to_string()))?;
        let end = start + length;

        if end > self.buffer.len() {
            return Err(Error::InvalidStream("stream extends after document end.".to_string()));
        }

        stream.set_content(self.buffer[start..end].to_vec());
        Ok(())
    }

    fn get_stream_length(&self, object_id: ObjectId) -> Result<i64> {
        let object = self.document.get_object(object_id)?;
        let stream = object.as_stream()?;
        stream
            .dict
            .get(b"Length")
            .and_then(|value| self.document.dereference(value))
            .and_then(|(_id, obj)| match obj.as_i64() {
                Ok(length) => Ok(length),
                // ISO 32000-1 s7.3.8.2 requires /Length to be an integer, but some producers
                // write it as a real ("42." instead of "42"). Accept one whose value is
                // integral and in range rather than dropping the stream's content.
                Err(err) => match obj.as_f32() {
                    Ok(value) if value.fract() == 0.0 && value >= -(2f32.powi(63)) && value < 2f32.powi(63) => {
                        Ok(value as i64)
                    }
                    _ => Err(err),
                },
            })
            .inspect_err(|_err| {
                error!(
                    "stream dictionary of '{} {} R' is missing the Length entry",
                    object_id.0, object_id.1
                );
            })
    }

    /// Get object offset by object ID.
    fn get_offset(&self, id: ObjectId) -> Result<u32> {
        let entry = self.document.reference_table.get(id.0).ok_or(Error::MissingXrefEntry)?;
        match *entry {
            XrefEntry::Normal { offset, generation } if generation == id.1 => Ok(offset),
            _ => Err(Error::MissingXrefEntry),
        }
    }

    /// Load a compressed object from an object stream (for lightweight metadata extraction)
    fn get_compressed_object(&self, id: ObjectId) -> Result<Object> {
        let entry = self.document.reference_table.get(id.0).ok_or(Error::MissingXrefEntry)?;

        let container_id = match entry {
            XrefEntry::Compressed { container, .. } => *container,
            _ => return Err(Error::MissingXrefEntry),
        };

        let container_id = (container_id, 0);
        let mut already_seen = HashSet::new();
        let container_obj = self.get_object(container_id, &mut already_seen)?;
        let container_stream = container_obj.as_stream()?;
        let object_stream = parser::with_real_parser(self.real_parser, || {
            ObjectStream::new_with_policy(container_stream, self.max_decompressed_size, self.parser_policy.as_ref())
        }).map_err(|e| e.at_object(container_id))?;
        object_stream.objects.get(&id).cloned().ok_or(Error::MissingXrefEntry)
    }

    /// Borrow only a verified range of this reader's unmodified input. Parsing
    /// and decryption retain their existing order; transformed bytes stay owned.
    pub(crate) fn stream_content(&self, data: &[u8]) -> crate::StreamContent {
        if self.encryption_state.is_none() {
            if let Some(bytes) = &self.shared_buffer {
                let base = bytes.as_ptr() as usize;
                if let Some(start) = (data.as_ptr() as usize).checked_sub(base) {
                    if let Some(end) = start.checked_add(data.len()) {
                        if end <= bytes.len() && bytes[start..end] == *data {
                            let mut content = crate::StreamContent::from(Vec::new());
                            content.set_shared(bytes.clone(), start..end);
                            return content;
                        }
                    }
                }
            }
        }
        data.to_vec().into()
    }

    pub fn get_object(&self, id: ObjectId, already_seen: &mut HashSet<ObjectId>) -> Result<Object> {
        if already_seen.contains(&id) {
            warn!("reference cycle detected resolving object {} {}", id.0, id.1);
            return Err(Error::ReferenceCycle(id));
        }
        already_seen.insert(id);

        if let Some(entry) = self.document.reference_table.get(id.0)
            && matches!(entry, XrefEntry::Compressed { .. })
        {
            return self.get_compressed_object(id);
        }

        let offset = self.get_offset(id)?;
        let (_, mut obj) = self.read_object(offset as usize, Some(id), already_seen)?;

        if let Some(ref state) = self.encryption_state {
            let encrypt_ref = self
                .document
                .trailer
                .get(b"Encrypt")
                .ok()
                .and_then(|o| o.as_reference().ok());
            if let Some(enc_ref) = encrypt_ref
                && id != enc_ref
            {
                encryption::decrypt_object(state, id, &mut obj).map_err(Error::Decryption)?;
            }
        }

        // Share only byte-identical parsed streams. This deliberately leaves
        // decrypted, repaired, and otherwise transformed stream bytes owned.
        if let Some(bytes) = &self.shared_buffer {
            if let Ok(stream) = obj.as_stream_mut() {
                if let Some(start) = stream.start_position.and_then(|start| {
                    (self.buffer.as_ptr() as usize)
                        .checked_sub(bytes.as_ptr() as usize)
                        .and_then(|offset| offset.checked_add(start))
                }) {
                    if let Some(end) = start.checked_add(stream.content.len()) {
                        if end <= bytes.len() && bytes[start..end] == *stream.content.as_slice() {
                            stream.content.set_shared(bytes.clone(), start..end);
                        }
                    }
                }
            }
        }

        Ok(obj)
    }

    fn parse_encryption_dictionary(&mut self) -> Result<()> {
        if let Ok(encrypt_ref) = self.document.trailer.get(b"Encrypt").and_then(|o| o.as_reference()) {
            if self.raw_objects.is_empty() {
                let offset = self.get_offset(encrypt_ref)?;
                let (_, encrypt_obj) = self.read_object(offset as usize, Some(encrypt_ref), &mut HashSet::new())?;
                self.document.objects.insert(encrypt_ref, encrypt_obj);
            } else if let Some(raw_range) = self.raw_objects.get(&encrypt_ref).cloned()
                && let Ok((_, obj)) = self.parse_raw_object(&self.buffer[raw_range])
            {
                self.document.objects.insert(encrypt_ref, obj);
            }
        }
        Ok(())
    }

    fn authenticate_and_setup_encryption(&mut self, require_password: bool) -> Result<Option<String>> {
        let password_to_use: Option<String> = if self.document.authenticate_password("").is_ok() {
            Some(String::new())
        } else if let Some(ref pwd) = self.password {
            if self.document.authenticate_password(pwd).is_ok() {
                Some(pwd.clone())
            } else if require_password {
                return Err(Error::InvalidPassword);
            } else {
                warn!("Invalid password provided for encrypted PDF");
                return Err(Error::InvalidPassword);
            }
        } else if require_password {
            return Err(Error::Unimplemented(
                "PDF is encrypted and requires a password. Use Document::load_metadata_with_password() instead.",
            ));
        } else {
            warn!("PDF is encrypted and requires a password");
            return Ok(None);
        };

        if let Some(ref password) = password_to_use {
            let state = EncryptionState::decode(&self.document, password)?;
            self.encryption_state = Some(state);
        }

        Ok(password_to_use)
    }

    fn setup_encryption_for_metadata(&mut self) -> Result<()> {
        self.parse_encryption_dictionary()?;
        self.authenticate_and_setup_encryption(true)?;
        Ok(())
    }

    fn read_object(
        &self, offset: usize, expected_id: Option<ObjectId>, already_seen: &mut HashSet<ObjectId>,
    ) -> Result<(ObjectId, Object)> {
        let next_index = self.normal_offsets.partition_point(|&next| next <= offset);
        let end = self.object_end(offset, self.normal_offsets.get(next_index).copied());
        self.read_object_to(offset, end, expected_id, already_seen)
    }

    fn object_end(&self, offset: usize, next_object: Option<usize>) -> usize {
        let xref_start = (self.document.xref_start > offset).then_some(self.document.xref_start);
        next_object
            .into_iter()
            .chain(xref_start)
            .min()
            .unwrap_or(self.buffer.len())
            .min(self.buffer.len())
    }

    fn read_object_to(
        &self, offset: usize, end: usize, expected_id: Option<ObjectId>, already_seen: &mut HashSet<ObjectId>,
    ) -> Result<(ObjectId, Object)> {
        if offset > end || end > self.buffer.len() {
            return Err(Error::InvalidOffset(offset));
        }

        // Objects are parsed against the full buffer so a wrong *neighbor*
        // xref offset cannot truncate a well-formed object; `end` only limits
        // how far malformed-stream length recovery may scan.
        parser::with_real_parser(self.real_parser, || {
            parser::indirect_object(self.buffer, offset, expected_id, self, already_seen, Some(end))
        })
    }

    /// Parse the cross-reference section recorded at `offset`, first correcting
    /// the offset if it is slightly miswritten (lenient mode only).
    fn xref_and_trailer_at(&self, offset: usize) -> Result<(Xref, Dictionary)> {
        let offset = self.correct_xref_offset(offset);
        parser::with_real_parser(self.real_parser, || {
            parser::xref_and_trailer(&self.buffer[offset..], self)
        })
    }

    /// Resolve the cross-reference table/stream and trailer, including the
    /// `/Prev` chain, and record the resolved start offset on the document.
    fn resolve_xref_and_trailer(&mut self) -> Result<(Xref, Dictionary)> {
        let xref_start = Self::get_xref_start(self.buffer)?;
        if xref_start > self.buffer.len() {
            return Err(Error::Xref(XrefError::Start));
        }
        let xref_start = self.correct_xref_offset(xref_start);
        self.document.xref_start = xref_start;

        let (mut xref, mut trailer) = self.xref_and_trailer_at(xref_start)?;

        // Read previous Xrefs of linearized or incremental updated document.
        let mut already_seen = HashSet::new();
        let mut prev_xref_start = trailer.remove(b"Prev");
        while let Some(prev) = prev_xref_start.and_then(|offset| offset.as_i64().ok()) {
            if already_seen.contains(&prev) {
                break;
            }
            already_seen.insert(prev);
            if prev < 0 || prev as usize > self.buffer.len() {
                return Err(Error::Xref(XrefError::PrevStart));
            }

            let (prev_xref, prev_trailer) = self.xref_and_trailer_at(prev as usize)?;
            xref.merge(prev_xref);

            // Read xref stream in hybrid-reference file
            let prev_xref_stream_start = trailer.remove(b"XRefStm");
            if let Some(prev) = prev_xref_stream_start.and_then(|offset| offset.as_i64().ok()) {
                if prev < 0 || prev as usize > self.buffer.len() {
                    return Err(Error::Xref(XrefError::StreamStart));
                }

                let (prev_xref, _) = self.xref_and_trailer_at(prev as usize)?;
                xref.merge(prev_xref);
            }

            prev_xref_start = prev_trailer.get(b"Prev").cloned().ok();
        }
        let xref_entry_count = xref.max_id().checked_add(1).ok_or(ParseError::InvalidXref)?;
        if xref.size != xref_entry_count {
            warn!(
                "Size entry of trailer dictionary is {}, correct value is {}.",
                xref.size, xref_entry_count
            );
            xref.size = xref_entry_count;
        }

        Ok((xref, trailer))
    }

    /// Last-resort recovery: rebuild the cross-reference table by scanning the
    /// raw bytes for indirect-object headers and locating a trailer with a
    /// usable `/Root`. Returns `None` (caller keeps the original error) when
    /// strict mode forbids recovery or nothing usable was found.
    fn reconstruct_xref_and_trailer(&mut self) -> Option<(Xref, Dictionary)> {
        if self.strict {
            return None;
        }
        u32::try_from(self.buffer.len()).ok()?;

        let markers = Self::scan_object_markers(self.buffer);
        if markers.is_empty() {
            return None;
        }

        let mut xref = Xref::new(markers.len() as u32, XrefType::CrossReferenceTable);
        for (offset, (number, generation)) in markers {
            // Incremental updates append, so later revisions win.
            xref.insert(number, XrefEntry::Normal { offset, generation });
        }
        // Normalize like `resolve_xref_and_trailer`: size spans object numbers
        // up to the highest, not physical copies across incremental updates.
        xref.size = xref.max_id().saturating_add(1);

        let (_trailer_pos, trailer) = self.find_latest_trailer(&xref)?;
        // deliberate: no on-disk table exists to point at. Zero marks the
        // offset "unknown" so `Document::new_from_prev` omits `/Prev` instead
        // of recording end-of-file; `object_end` clamps to the buffer either way.
        self.document.xref_start = 0;

        warn!(
            "reconstructed cross-reference table with {} objects by scanning for indirect objects",
            xref.entries.len()
        );
        Some((xref, trailer))
    }

    /// Collect `(offset, id)` of every indirect-object header in one pass.
    /// Only headers starting a line (optional leading blanks allowed) count,
    /// stream payloads are skipped wholesale, and object numbers beyond
    /// [`MAX_RECONSTRUCTED_OBJECTS`] are rejected so a forged header can
    /// neither shadow a genuine entry nor poison the reconstructed size.
    fn scan_object_markers(buffer: &[u8]) -> Vec<(u32, ObjectId)> {
        const STREAM_KEYWORD: &[u8] = b"stream";
        const END_STREAM_KEYWORD: &[u8] = b"endstream";

        let index = RecoveryIndex::new(buffer);
        let mut markers = Vec::new();
        let mut oversized_number_warned = false;
        let mut at_line_start = true;
        let mut pos = 0;
        while pos < buffer.len() {
            // Skip raw stream data: an uncompressed payload may embed
            // convincing `N G obj` lines whose later offsets would otherwise
            // override the genuine entries for those object numbers.
            if buffer[pos..].starts_with(STREAM_KEYWORD)
                && !buffer[..pos].ends_with(b"end")
                && matches!(buffer.get(pos + STREAM_KEYWORD.len()), Some(b'\r' | b'\n'))
            {
                let after_keyword = pos + STREAM_KEYWORD.len();
                let next = index.endstreams.partition_point(|&offset| offset < after_keyword);
                if let Some(&offset) = index.endstreams.get(next) {
                    pos = offset + END_STREAM_KEYWORD.len();
                    at_line_start = false;
                    continue;
                }
                // Damaged stream without terminator: fall back to the
                // dictionary's /Length hint so the payload cannot hide
                // line-start pseudo headers, while objects written after it
                // stay reachable.
                if let Some(resume) = index.payload_end(buffer, pos) {
                    pos = resume;
                    at_line_start = false;
                    continue;
                }
                // Unusable /Length too: nothing bounds the payload, so keep
                // scanning byte by byte rather than dropping what follows.
            }
            // Anchor at line starts so pseudo headers buried after another
            // token (string literal, comment) are never mistaken for markers.
            if at_line_start
                && buffer[pos].is_ascii_digit()
                && let Some(id) = Self::parse_object_header(&buffer[pos..])
            {
                // A huge bogus number would inflate `size` (and thus
                // `max_id`) via `Xref::insert`; genuine numbering stays
                // within the same cap as the marker count.
                if id.0 > MAX_RECONSTRUCTED_OBJECTS as u32 {
                    if !oversized_number_warned {
                        warn!("ignoring object headers numbered above {MAX_RECONSTRUCTED_OBJECTS}");
                        oversized_number_warned = true;
                    }
                } else {
                    if markers.len() == MAX_RECONSTRUCTED_OBJECTS {
                        warn!(
                            "object marker scan stopped at the {MAX_RECONSTRUCTED_OBJECTS}-marker cap; reconstruction may be incomplete"
                        );
                        break;
                    }
                    markers.push((pos as u32, id));
                }
            }
            match buffer[pos] {
                b'\r' | b'\n' => at_line_start = true,
                b' ' | b'\t' => {}
                _ => at_line_start = false,
            }
            pos += 1;
        }
        markers
    }

    /// Resume offset past a stream payload according to a *direct* `/Length`
    /// integer in the dictionary preceding the `stream` keyword at
    /// `stream_pos`. The search is scoped to the current object (after the
    /// nearest preceding `obj` keyword) so a missing `/Length` cannot latch
    /// onto a previous object's value. Returns `None` when the hint is
    /// absent, indirect, or points outside the buffer — damaged files tend
    /// to carry wrong `/Length` values, so it must stay a hint, never a hard
    /// boundary.
    /// Parse an `N G obj` header; the byte after `obj` must end the token.
    fn parse_object_header(input: &[u8]) -> Option<ObjectId> {
        // deliberate: caps at u32/u16 width bound work on pathological padding.
        fn digits(input: &[u8], cap: usize) -> Option<(&[u8], &[u8])> {
            let n = input.iter().take_while(|b| b.is_ascii_digit()).count();
            (0 < n && n <= cap).then(|| input.split_at(n))
        }
        fn spaces(input: &[u8]) -> Option<&[u8]> {
            let n = input
                .iter()
                .take_while(|&&b| matches!(b, b' ' | b'\t' | b'\r' | b'\n'))
                .count();
            (n > 0).then(|| &input[n..])
        }

        let (number, rest) = digits(input, 10)?;
        let number: u32 = std::str::from_utf8(number).ok()?.parse().ok()?;
        let rest = spaces(rest)?;
        let (generation, rest) = digits(rest, 5)?;
        let generation: u16 = std::str::from_utf8(generation).ok()?.parse().ok()?;
        let rest = spaces(rest)?;
        let tail = rest.strip_prefix(b"obj")?;
        match tail.first() {
            None => {}
            Some(&byte) if !byte.is_ascii_alphanumeric() => {}
            _ => return None,
        }
        Some((number, generation))
    }

    /// Newest backwards-scanned `trailer` whose `/Root` references a scanned object.
    fn find_latest_trailer(&self, xref: &Xref) -> Option<(usize, Dictionary)> {
        let mut bound = self.buffer.len();
        for _ in 0..MAX_TRAILER_CANDIDATES {
            let Some(pos) = Reader::search_substring(&self.buffer[..bound], b"trailer", 0) else {
                break;
            };
            bound = pos;

            let after_keyword = &self.buffer[pos + b"trailer".len()..];
            let Some(dict_start) = after_keyword.iter().position(|byte| !byte.is_ascii_whitespace()) else {
                continue;
            };
            let Ok((_, dict)) =
                parser::with_real_parser(self.real_parser, || parser::dictionary(&after_keyword[dict_start..]))
            else {
                continue;
            };
            if dict
                .get(b"Root")
                .and_then(Object::as_reference)
                .is_ok_and(|root| xref.entries.contains_key(&root.0))
            {
                return Some((pos, dict));
            }
        }
        None
    }

    /// Some generators write `startxref` (or trailer `Prev`) values that are
    /// slightly off — most commonly the offset of the line *after* the `xref`
    /// keyword instead of the keyword itself. Such files are otherwise intact,
    /// and common readers (Acrobat, poppler, mupdf, pdf.js) recover from them.
    /// In lenient mode, if `offset` does not point at a cross-reference table
    /// or an indirect object (cross-reference stream), scan a small window
    /// around it for the `xref` keyword and use the nearest match instead.
    fn correct_xref_offset(&self, offset: usize) -> usize {
        const RECOVERY_WINDOW: usize = 64;

        if self.strict || offset >= self.buffer.len() {
            return offset;
        }
        let rest = &self.buffer[offset..];
        if rest.starts_with(b"xref") || Self::starts_indirect_object(rest) {
            return offset;
        }

        let window_start = offset.saturating_sub(RECOVERY_WINDOW);
        let window_end = cmp::min(self.buffer.len(), offset + RECOVERY_WINDOW);
        let mut corrected: Option<usize> = None;
        for pos in window_start..window_end.saturating_sub(4) {
            if !self.buffer[pos..].starts_with(b"xref") {
                continue;
            }
            // `startxref` contains `xref`; never match inside it.
            if pos >= 5 && &self.buffer[pos - 5..pos] == b"start" {
                continue;
            }
            if corrected.is_none_or(|best: usize| pos.abs_diff(offset) < best.abs_diff(offset)) {
                corrected = Some(pos);
            }
        }
        match corrected {
            Some(pos) => {
                warn!(
                    "Cross-reference offset {} does not point at an xref section; using nearby offset {} instead.",
                    offset, pos
                );
                pos
            }
            None => offset,
        }
    }

    /// Whether `input` begins with an indirect-object header (`N G obj`), the
    /// form a cross-reference stream starts with.
    fn starts_indirect_object(input: &[u8]) -> bool {
        Self::parse_object_header(input).is_some()
    }

    fn get_xref_start(buffer: &[u8]) -> Result<usize> {
        let seek_pos = buffer.len() - cmp::min(buffer.len(), 512);
        Self::search_substring(buffer, b"%%EOF", seek_pos)
            .filter(|&eof_pos| eof_pos > 25)
            .and_then(|eof_pos| Self::search_substring(&buffer[..eof_pos], b"startxref", eof_pos - 25))
            .ok_or(Error::Xref(XrefError::Start))
            .and_then(|xref_pos| {
                if xref_pos <= buffer.len() {
                    match parser::xref_start(&buffer[xref_pos..]) {
                        Some(startxref) => Ok(startxref as usize),
                        None => Err(Error::Xref(XrefError::Start)),
                    }
                } else {
                    Err(Error::Xref(XrefError::Start))
                }
            })
    }

    fn search_substring(buffer: &[u8], pattern: &[u8], start_pos: usize) -> Option<usize> {
        buffer
            .get(start_pos..)?
            .windows(pattern.len())
            .rposition(|window| window == pattern)
            .map(|pos| start_pos + pos)
    }
}

#[cfg(all(test, not(feature = "async")))]
#[test]
fn load_document() {
    let mut doc = Document::load("assets/example.pdf").unwrap();
    assert_eq!(doc.version, "1.5");

    // Create temporary folder to store file.
    let temp_dir = tempfile::tempdir().unwrap();
    let file_path = temp_dir.path().join("test_2_load.pdf");
    doc.save(file_path).unwrap();
}

#[cfg(all(test, feature = "async"))]
#[tokio::test]
async fn load_document() {
    let mut doc = Document::load("assets/example.pdf").await.unwrap();
    assert_eq!(doc.version, "1.5");

    // Create temporary folder to store file.
    let temp_dir = tempfile::tempdir().unwrap();
    let file_path = temp_dir.path().join("test_2_load.pdf");
    doc.save(file_path).unwrap();
}

#[test]
#[should_panic(expected = "Xref(Start)")]
fn load_short_document() {
    let _doc = Document::load_mem(b"%PDF-1.5\n%%EOF\n").unwrap();
}

#[test]
fn load_document_with_preceding_bytes() {
    let mut content = Vec::new();
    content.extend(b"garbage");
    content.extend(include_bytes!("../assets/example.pdf"));
    let doc = Document::load_mem(&content).unwrap();
    assert_eq!(doc.version, "1.5");
}

#[test]
fn load_many_shallow_brackets() {
    let content: String = std::iter::repeat_n("()", MAX_BRACKET * 10)
        .flat_map(|x| x.chars())
        .collect();
    const STREAM_CRUFT: usize = 33;
    let doc = format!(
        "%PDF-1.5
1 0 obj<</Type/Pages/Kids[5 0 R]/Count 1/Resources 3 0 R/MediaBox[0 0 595 842]>>endobj
2 0 obj<</Type/Font/Subtype/Type1/BaseFont/Courier>>endobj
3 0 obj<</Font<</F1 2 0 R>>>>endobj
5 0 obj<</Type/Page/Parent 1 0 R/Contents[4 0 R]>>endobj
6 0 obj<</Type/Catalog/Pages 1 0 R>>endobj
4 0 obj<</Length {}>>stream
BT
/F1 48 Tf
100 600 Td
({}) Tj
ET
endstream endobj\n",
        content.len() + STREAM_CRUFT,
        content
    );
    let doc = format!(
        "{}xref
0 7
0000000000 65535 f 
0000000009 00000 n 
0000000096 00000 n 
0000000155 00000 n 
0000000291 00000 n 
0000000191 00000 n 
0000000248 00000 n 
trailer
<</Root 6 0 R/Size 7>>
startxref
{}
%%EOF",
        doc,
        doc.len()
    );

    let _doc = Document::load_mem(doc.as_bytes()).unwrap();
}

#[test]
fn load_too_deep_brackets() {
    let content: Vec<u8> = std::iter::repeat_n(b'(', MAX_BRACKET + 1)
        .chain(std::iter::repeat_n(b')', MAX_BRACKET + 1))
        .collect();
    let content = String::from_utf8(content).unwrap();
    const STREAM_CRUFT: usize = 33;
    let doc = format!(
        "%PDF-1.5
1 0 obj<</Type/Pages/Kids[5 0 R]/Count 1/Resources 3 0 R/MediaBox[0 0 595 842]>>endobj
2 0 obj<</Type/Font/Subtype/Type1/BaseFont/Courier>>endobj
3 0 obj<</Font<</F1 2 0 R>>>>endobj
5 0 obj<</Type/Page/Parent 1 0 R/Contents[7 0 R 4 0 R]>>endobj
6 0 obj<</Type/Catalog/Pages 1 0 R>>endobj
7 0 obj<</Length 45>>stream
BT /F1 48 Tf 100 600 Td (Hello World!) Tj ET
endstream
endobj
4 0 obj<</Length {}>>stream
BT
/F1 48 Tf
100 600 Td
({}) Tj
ET
endstream endobj\n",
        content.len() + STREAM_CRUFT,
        content
    );
    let doc = format!(
        "{}xref
0 7
0000000000 65535 f 
0000000009 00000 n 
0000000096 00000 n 
0000000155 00000 n 
0000000387 00000 n 
0000000191 00000 n 
0000000254 00000 n 
0000000297 00000 n 
trailer
<</Root 6 0 R/Size 7>>
startxref
{}
%%EOF",
        doc,
        doc.len()
    );

    let doc = Document::load_mem(doc.as_bytes()).unwrap();
    let pages = doc.get_pages().keys().cloned().collect::<Vec<_>>();
    assert_eq!("Hello World!\n", doc.extract_text(&pages).unwrap());
}

#[cfg(all(test, not(feature = "async")))]
#[test]
fn search_substring_finds_last_occurrence() {
    assert_eq!(Reader::search_substring(b"hello world", b"xyz", 0), None);
    assert_eq!(Reader::search_substring(b"hello world", b"world", 0), Some(6));

    let buffer = b"%%EOF\ntest%%EOF\nend";
    assert_eq!(Reader::search_substring(buffer, b"%%EOF", 0), Some(10));
    assert_eq!(Reader::search_substring(buffer, b"%%EOF", 6), Some(10));
    assert_eq!(Reader::search_substring(buffer, b"%%EOF", 15), None);
    assert_eq!(Reader::search_substring(b"%%EOF", b"%%EOF", 0), Some(0));

    let buffer_with_many_percents = b"%%%PDF-1.3%%%comment%%%more%%EOF";
    assert_eq!(
        Reader::search_substring(buffer_with_many_percents, b"%%EOF", 0),
        Some(27)
    );
}

#[cfg(all(test, not(feature = "async")))]
#[test]
fn get_xref_start_ignores_startxref_past_eof() {
    // Simulate a PDF with two revisions where the second has a corrupted %%EOF.
    // The valid %%EOF is at a known position; a second startxref appears after it
    // but belongs to the corrupted revision. get_xref_start must pick the
    // startxref *before* the valid %%EOF.
    let mut buf = Vec::new();
    // Padding so the buffer is large enough
    buf.extend_from_slice(&[b' '; 200]);
    // First (correct) startxref + xref offset + valid %%EOF
    let xref_offset = 100usize;
    let startxref_block = format!("startxref\n{}\n%%EOF\n", xref_offset);
    let _startxref_pos = buf.len();
    buf.extend_from_slice(startxref_block.as_bytes());
    // Second (corrupted) revision: another startxref pointing elsewhere, with %%EO\0
    let bad_block = "startxref\n999\n%%EO\x00\n".to_string();
    buf.extend_from_slice(bad_block.as_bytes());

    let result = Reader::get_xref_start(&buf).unwrap();
    // Should find the xref_offset from the first startxref (before valid %%EOF)
    assert_eq!(result, xref_offset);
    // Verify it did NOT pick up 999 from the corrupted revision
    assert_ne!(result, 999);
}

#[cfg(all(test, not(feature = "async")))]
#[test]
fn recovers_from_miswritten_startxref_offset() {
    // Some generators write the offset of the line *after* the `xref` keyword
    // into `startxref` instead of the offset of the keyword itself. Lenient
    // loading must recover; strict loading must still reject the file.
    let header = "%PDF-1.5\n";
    let obj1 = "1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n";
    let obj2 = "2 0 obj<</Type/Pages/Kids[]/Count 0>>endobj\n";
    let o1 = header.len();
    let o2 = o1 + obj1.len();
    let body = format!("{header}{obj1}{obj2}");
    let xref_pos = body.len();
    let doc = format!(
        "{body}xref
0 3
0000000000 65535 f 
{o1:010} 00000 n 
{o2:010} 00000 n 
trailer
<</Root 1 0 R/Size 3>>
startxref
{}
%%EOF",
        xref_pos + 4 // past the `xref` keyword, onto the line after it
    );

    let loaded = Document::load_mem(doc.as_bytes()).unwrap();
    assert!(loaded.trailer.get(b"Root").is_ok());
    assert_eq!(loaded.xref_start, xref_pos);

    let strict = Document::load_mem_with_options(
        doc.as_bytes(),
        LoadOptions {
            strict: true,
            ..Default::default()
        },
    );
    assert!(strict.is_err());
}

#[cfg(all(test, not(feature = "async")))]
#[test]
fn stream_length_written_as_real() {
    // ISO 32000-1 s7.3.8.2 requires /Length to be an integer, but some generators
    // emit "/Length 42." instead of "/Length 42". Such a stream must still resolve
    // to its content rather than silently loading as empty.
    let obj4 = "4 0 obj\n<< /Length 42. >>\nstream\nBT /F1 12 Tf 20 100 Td (Hello World) Tj ET\nendstream\nendobj\n";
    let header = "%PDF-1.7\n";
    let obj1 = "1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n";
    let obj2 = "2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n";
    let obj3 = "3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] /Contents 4 0 R >>\nendobj\n";
    let o1 = header.len();
    let o2 = o1 + obj1.len();
    let o3 = o2 + obj2.len();
    let o4 = o3 + obj3.len();
    let body = format!("{header}{obj1}{obj2}{obj3}{obj4}");
    let xref_pos = body.len();
    let doc = format!(
        "{body}xref
0 5
0000000000 65535 f 
{o1:010} 00000 n 
{o2:010} 00000 n 
{o3:010} 00000 n 
{o4:010} 00000 n 
trailer
<< /Size 5 /Root 1 0 R >>
startxref
{xref_pos}
%%EOF
"
    );

    let loaded = Document::load_mem(doc.as_bytes()).unwrap();
    let stream = loaded.get_object((4, 0)).unwrap().as_stream().unwrap();
    assert_eq!(stream.content, b"BT /F1 12 Tf 20 100 Td (Hello World) Tj ET");
}

#[cfg(all(test, not(feature = "async")))]
#[test]
fn object_marker_parsing_accepts_only_real_headers() {
    assert_eq!(Reader::parse_object_header(b"12 0 obj<<"), Some((12, 0)));
    assert_eq!(Reader::parse_object_header(b"007 0 obj\n"), Some((7, 0)));
    assert_eq!(Reader::parse_object_header(b"5 2 obj>>"), Some((5, 2)));
    assert_eq!(Reader::parse_object_header(b"12\t3\r\nobj["), Some((12, 3)));
    assert_eq!(Reader::parse_object_header(b"2 0 objects"), None);
    assert_eq!(Reader::parse_object_header(b"99 88 objx"), None);
    assert_eq!(Reader::parse_object_header(b"12345678901 0 obj"), None);
    assert_eq!(Reader::parse_object_header(b"1 70000 obj"), None);
    assert_eq!(Reader::parse_object_header(b"1 0obj"), None);
    assert_eq!(Reader::parse_object_header(b"1  obj"), None);
    assert_eq!(Reader::parse_object_header(b"x 0 obj"), None);
}

// Local to one recovery scan; never retains a copy of the input.
struct RecoveryIndex {
    endstreams: Vec<usize>,
    objects: Vec<usize>,
    lengths: Vec<(usize, Option<(usize, usize, usize, Option<usize>)>)>,
}
impl RecoveryIndex {
    fn new(bytes: &[u8]) -> Self { Self::new_observed(bytes, |_| {}) }
    fn new_observed(bytes: &[u8], mut read: impl FnMut(usize)) -> Self {
        fn positions(bytes: &[u8], key: &[u8], read: &mut impl FnMut(usize)) -> Vec<usize> {
            bytes.windows(key.len()).enumerate()
                .filter_map(|(pos, window)| { read(1); (window == key).then_some(pos) }).collect()
        }
        let mut out = Self {
            endstreams: positions(bytes, b"endstream", &mut read),
            objects: positions(bytes, b"obj", &mut read),
            lengths: Vec::new(),
        };
        let mut cursor = 0;
        let mut digit = None;
        for key in positions(bytes, b"/Length", &mut read) {
            let start = key + b"/Length".len();
            if digit.is_none_or(|(begin, _, _, _)| begin < start) {
                cursor = cursor.max(start);
                while cursor < bytes.len() && !bytes[cursor].is_ascii_digit() { read(1); cursor += 1; }
                let begin = cursor;
                while cursor < bytes.len() && bytes[cursor].is_ascii_digit() { read(1); cursor += 1; }
                let end = cursor;
                while cursor < bytes.len() && bytes[cursor].is_ascii_whitespace() { read(1); cursor += 1; }
                let value = (end - begin <= 10).then(|| {
                    std::str::from_utf8(&bytes[begin..end]).ok()?.parse::<usize>().ok()
                }).flatten();
                digit = Some((begin, end, cursor, value));
            }
            out.lengths.push((key, digit));
        }
        out
    }
    fn payload_end(&self, bytes: &[u8], stream_pos: usize) -> Option<usize> {
        let obj_slot = self.objects.partition_point(|&pos| pos + b"obj".len() <= stream_pos);
        let obj = *self.objects.get(obj_slot.checked_sub(1)?)?;
        let key_slot = self.lengths.partition_point(|&(pos, _)| pos + b"/Length".len() <= stream_pos);
        let &(key, hint) = self.lengths.get(key_slot.checked_sub(1)?)?;
        if key < obj + b"obj".len() { return None; }
        let (begin, end, tail, value) = hint?;
        let value = value?;
        if begin >= stream_pos || end > stream_pos || tail >= stream_pos
            || !matches!(bytes[tail], b'/' | b'>') { return None; }
        let after = stream_pos + b"stream".len();
        let eol = if bytes.get(after) == Some(&b'\r') && bytes.get(after + 1) == Some(&b'\n') { 2 } else { 1 };
        let end = (after + eol).checked_add(value)?;
        (end <= bytes.len()).then_some(end)
    }
}
#[cfg(test)]
#[path = "reader_resource_v3_tests.rs"]
mod resource_v3_tests;
