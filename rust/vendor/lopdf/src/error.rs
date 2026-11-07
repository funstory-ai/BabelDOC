use thiserror::Error;

use crate::encodings::cmap::UnicodeCMapError;
use crate::{ObjectId, encryption};

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Error)]
pub enum Error {
    #[error(transparent)]
    ResourceLimitExceeded(#[from] crate::resource::ResourceLimitExceeded),
    /// Lopdf does not (yet) implement a needed feature.
    #[error(
        "missing feature of lopdf: {0}; please open an issue at https://github.com/J-F-Liu/lopdf/ to let the developers know of your usecase"
    )]
    Unimplemented(&'static str),

    /// An Object has the wrong type, e.g. the Object is an Array where a Name would be expected.
    #[error("object has wrong type; expected type {expected} but found type {found}")]
    ObjectType {
        expected: &'static str,
        found: &'static str,
    },
    #[error("dictionary has wrong type: ")]
    DictType { expected: &'static str, found: String },
    /// PDF document is already encrypted.
    #[error("PDF document is already encrypted")]
    AlreadyEncrypted,
    /// The encountered character encoding is invalid.
    #[error("invalid character encoding")]
    CharacterEncoding,
    /// The stream couldn't be decompressed.
    #[error("couldn't decompress stream")]
    Decompress(#[from] DecompressError),
    /// Failed to parse input.
    #[error("couldn't parse input")]
    Parse(#[from] ParseError),
    /// Error when decrypting the contents of the file
    #[error("decryption error")]
    Decryption(#[from] encryption::DecryptionError),
    /// Dictionary key was not found.
    #[error("missing required dictionary key \"{0}\"")]
    DictKey(String),
    /// Invalid inline image.
    #[error("invalid inline image: {0}")]
    InvalidInlineImage(String),
    /// Invalid document outline.
    #[error("invalid document outline: {0}")]
    InvalidOutline(String),
    /// Invalid stream.
    #[error("invalid stream: {0}")]
    InvalidStream(String),
    /// Invalid object stream.
    #[error("invalid object stream: {0}")]
    InvalidObjectStream(String),
    /// Byte offset in stream or file is invalid.
    #[error("invalid byte offset")]
    InvalidOffset(usize),
    /// IO error
    #[error("IO error")]
    IO(#[from] std::io::Error),
    // TODO: Maybe remove, as outline is not required in spec.
    /// PDF document has no outline.
    #[error("PDF document does not have an outline")]
    NoOutline,
    /// PDF document is not encrypted.
    #[error("PDF document is not encrypted")]
    NotEncrypted,
    /// Invalid password provided for encrypted PDF.
    #[error("invalid password for encrypted PDF")]
    InvalidPassword,
    /// Missing xref entry.
    #[error("missing xref entry")]
    MissingXrefEntry,
    /// The Object ID was not found.
    #[error("object ID {} {} not found", .0.0, .0.1)]
    ObjectNotFound(ObjectId),
    /// Dereferencing object failed due to a reference cycle.
    #[error("reference cycle with object ID {} {}", .0.0, .0.1)]
    ReferenceCycle(ObjectId),
    /// Page number was not found in document.
    #[error("page number not found")]
    PageNumberNotFound(u32),
    /// Numeric type cast failed.
    #[error("numberic type cast failed: {0}")]
    NumericCast(String),
    /// Dereferencing object reached the limit.
    /// This might indicate a reference loop.
    #[error("dereferencing object reached limit, may indicate a reference cycle")]
    ReferenceLimit,
    /// Traversal of the document's object graph (e.g. a `/Kids`, `/First`, or `/Parent` chain)
    /// exceeded the supported nesting depth.
    /// This might indicate a reference cycle or a maliciously deep structure.
    #[error("object graph traversal reached the nesting-depth limit, may indicate a reference cycle")]
    RecursionLimit,
    /// Decoding text string failed.
    #[error("decoding text string failed")]
    TextStringDecode,
    /// Error while parsing cross reference table.
    #[error("failed parsing cross reference table")]
    Xref(#[from] XrefError),
    /// Invalid indirect object while parsing at offset.
    #[error("invalid indirect object at byte offset {offset}")]
    IndirectObject { offset: usize },
    /// Found object ID does not match expected object ID.
    #[error("found object ID does not match expected object ID")]
    ObjectIdMismatch,
    /// Error when handling images.
    #[cfg(feature = "embed_image")]
    #[error("image error")]
    Image(#[from] image::ImageError),
    /// Syntax error while processing the content stream.
    #[error("syntax error in content stream: {0}")]
    Syntax(String),
    /// Could not parse ToUnicodeCMap.
    #[error("failed parsing ToUnicode CMap")]
    ToUnicodeCMap(#[from] UnicodeCMapError),
    #[error("converting integer")]
    TryFromInt(#[from] std::num::TryFromIntError),
    /// Encountered an unsupported security handler.
    #[error("unsupported security handler")]
    UnsupportedSecurityHandler(Vec<u8>),
    /// Encountered when a differences code is out of bounds.
    #[error("invalid encoding difference code: {code}")]
    InvalidEncodingDifferenceCode { code: i64 },
    /// Encountered when a differences glyph name is invalid.
    #[error("invalid encoding difference glyph name: {name}")]
    InvalidEncodingDifferenceGlyph { name: String },
}

#[derive(Error, Debug)]
pub enum DecompressError {
    #[error("decoding ASCII85 failed: {0}")]
    Ascii85(&'static str),
    #[error("decoding ASCIIHex failed: {0}")]
    AsciiHex(&'static str),
    #[error("applying the stream predictor failed: {0}")]
    Predictor(&'static str),
    /// The decompressed output exceeded the allowed size limit. This guards
    /// against decompression bombs: a small compressed stream that inflates to
    /// an enormous size (potentially exhausting memory). The `limit` is the
    /// maximum number of output bytes that were permitted.
    #[error("decompressed output exceeded the {limit}-byte limit (possible decompression bomb)")]
    MemoryLimitExceeded { limit: usize },
}

#[derive(Error, Debug)]
pub enum ParseError {
    #[error("unexpected end of input")]
    EndOfInput,
    #[error("invalid content stream")]
    InvalidContentStream,
    #[error("invalid file header")]
    InvalidFileHeader,
    #[error("invalid file trailer")]
    InvalidTrailer,
    #[error("invalid cross reference table")]
    InvalidXref,
}

#[derive(Debug, Error)]
pub enum XrefError {
    /// Could not find start of cross reference table.
    #[error("invalid start value")]
    Start,
    /// The trailer's "Prev" field was invalid.
    #[error("invalid start value in Prev field")]
    PrevStart,
    /// The trailer's "XRefStm" field was invalid.
    #[error("invalid start value of XRefStm")]
    StreamStart,
}
