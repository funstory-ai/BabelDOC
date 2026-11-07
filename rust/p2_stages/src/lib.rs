pub use babeldoc_il_types as il_types;
pub use babeldoc_il_types::schema::*;

pub mod font_mapper;
pub mod runtime;
pub mod shared;
pub mod stages;
pub mod translation;

use std::error::Error;
use std::fmt;

pub use runtime::{Cancellation, NoopCancellation, NoopProgress, ProgressReporter, StageContext};

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum StageError {
    Cancelled,
    InvalidConfig(String),
    ScannedPdf(String),
    ExtractText(String),
    ContentFilter(String),
    Failed(String),
}

impl fmt::Display for StageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cancelled => Ok(()),
            Self::InvalidConfig(message)
            | Self::ScannedPdf(message)
            | Self::ExtractText(message)
            | Self::ContentFilter(message)
            | Self::Failed(message) => f.write_str(message),
        }
    }
}

impl StageError {
    pub fn failed(message: impl Into<String>) -> Self {
        Self::Failed(message.into())
    }

    pub fn category(&self) -> StageErrorCategory {
        match self {
            Self::Cancelled => StageErrorCategory::Cancelled,
            Self::InvalidConfig(_) => StageErrorCategory::ValueError,
            Self::ScannedPdf(_) => StageErrorCategory::ScannedPdf,
            Self::ExtractText(_) => StageErrorCategory::ExtractText,
            Self::ContentFilter(_) => StageErrorCategory::ContentFilter,
            Self::Failed(_) => StageErrorCategory::Failed,
        }
    }
}

impl From<String> for StageError {
    fn from(message: String) -> Self {
        Self::Failed(message)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StageErrorCategory {
    ScannedPdf,
    ExtractText,
    ValueError,
    Cancelled,
    ContentFilter,
    Failed,
}

impl Error for StageError {}

#[cfg(test)]
mod test_logging_tests;

impl From<babeldoc_ir_core::AccessError> for StageError {
    fn from(error: babeldoc_ir_core::AccessError) -> Self {
        Self::Failed(error.to_string())
    }
}

impl From<shared::layout_helper::LayoutHelperError> for StageError {
    fn from(error: shared::layout_helper::LayoutHelperError) -> Self {
        Self::Failed(error.to_string())
    }
}

impl From<shared::formula_helper::FormulaHelperError> for StageError {
    fn from(error: shared::formula_helper::FormulaHelperError) -> Self {
        Self::Failed(error.to_string())
    }
}
