//! PDF content construction and serial application, without Python callbacks.
#![forbid(unsafe_code)]

mod codecs;
mod content;
mod fonts;
mod pdf;
mod retry;
pub use retry::{WriteAttempt, write_with_retry};
/// Python-compatible inline-image decoder; exposed for differential unit tests.
pub use codecs::base64 as decode_inline_data;
pub use content::*;
pub use fonts::*;
pub use lopdf::Document as PdfDocument;
pub use pdf::*;

#[derive(Debug)]
pub enum WriterError {
    Ir(babeldoc_ir_core::AccessError),
    Pdf(lopdf::Error),
    Invalid(String),
    Io(std::io::Error),
}
impl std::fmt::Display for WriterError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Ir(e) => e.fmt(f),
            Self::Pdf(e) => e.fmt(f),
            Self::Invalid(s) => s.fmt(f),
            Self::Io(e) => e.fmt(f),
        }
    }
}
impl std::error::Error for WriterError {}
impl From<babeldoc_ir_core::AccessError> for WriterError {
    fn from(e: babeldoc_ir_core::AccessError) -> Self {
        Self::Ir(e)
    }
}
impl From<lopdf::Error> for WriterError {
    fn from(e: lopdf::Error) -> Self {
        Self::Pdf(e)
    }
}
impl From<std::io::Error> for WriterError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}
pub type Result<T> = std::result::Result<T, WriterError>;
fn required<T>(v: Option<T>, field: &str) -> Result<T> {
    v.ok_or_else(|| WriterError::Invalid(format!("missing {field}")))
}

pub use codecs::base85 as decode_base85;
