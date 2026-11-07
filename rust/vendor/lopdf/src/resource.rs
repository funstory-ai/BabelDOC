//! Immutable, local amplification limits. No counters or ambient policy state.
use crate::{Error, Result};

pub const POLICY_VERSION: u32 = 5;
pub const MAX_DECODED_STREAM_BYTES: usize = 64 * 1024 * 1024;
pub const MAX_ROOT_DECODED_BYTES: usize = 256 * 1024 * 1024;
pub const MAX_FILTER_LAYERS: usize = 16;
pub const MAX_PREDICTOR_ROW_BYTES: usize = 16 * 1024 * 1024;
pub const MAX_BROTLI_WINDOW_BYTES: usize = 16 * 1024 * 1024;
pub const MAX_UNICODE_MAPPING_BYTES: usize = 256;
pub const MAX_REFERENCE_EXPANSION_AMPLIFICATION: usize = 16;
pub const MAX_REFERENCE_EXPANSION_DEPTH: usize = 128;
pub const MAX_FORM_EXPANSION_DEPTH: usize = 64;
pub const DISABLED_WARNING: &str = "Parser resource limits are disabled. Malicious PDFs may cause excessive memory or CPU use; isolate this parse with external process limits.";

/// Parser-only observations collected during document loading.
///
/// These counters are diagnostics, not resource budgets, and are never written
/// to the PDF or serialized into parser IR artifacts.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ParserDiagnostics {
    /// Total rejected ObjStm members across all three reason buckets.
    pub invalid_objstm_members: usize,
    /// Out-of-bounds or duplicate start offsets rejected before parsing.
    pub invalid_objstm_offset_members: usize,
    /// Segment-bounded parsing failed, while the legacy unbounded parse succeeded.
    pub invalid_objstm_segment_b20_would_accept: usize,
    /// Segment-bounded parsing failed and the legacy unbounded parse also failed.
    pub invalid_objstm_segment_both_fail: usize,
    /// Total rejected encrypted raw objects across all three reason buckets.
    pub invalid_encrypted_objects: usize,
    /// Out-of-bounds or duplicate xref offsets rejected before parsing.
    pub invalid_encrypted_offset_objects: usize,
    /// Segment-bounded extraction/parsing failed, while the legacy path succeeded.
    pub invalid_encrypted_segment_b20_would_accept: usize,
    /// Segment-bounded extraction/parsing failed and the legacy path also failed.
    pub invalid_encrypted_segment_both_fail: usize,
    /// Font lookup names skipped because they exceed the legacy alias bound.
    pub long_font_alias_skips: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResourceLimitExceeded {
    pub limit_name: &'static str,
    pub limit: u64,
    pub observed_or_requested: u64,
    pub unit: &'static str,
    pub scope: &'static str,
    pub phase: &'static str,
    pub page: Option<usize>,
    pub object_ref: Option<crate::ObjectId>,
    pub policy_version: u32,
}
impl std::fmt::Display for ResourceLimitExceeded {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}: requested/observed {} {}, limit {} (policy {}, phase {})",
            self.limit_name,
            self.observed_or_requested,
            self.unit,
            self.limit,
            self.policy_version,
            self.phase
        )
    }
}
impl std::error::Error for ResourceLimitExceeded {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Policy {
    disabled: bool,
    decoded: usize,
    layers: usize,
    row: usize,
    window: usize,
    unicode_bytes: usize,
    root_decoded_bytes: usize,
    expansion_amplification: usize,
    depth: usize,
}
impl Default for Policy {
    fn default() -> Self {
        Self {
            disabled: false,
            decoded: MAX_DECODED_STREAM_BYTES,
            layers: MAX_FILTER_LAYERS,
            row: MAX_PREDICTOR_ROW_BYTES,
            window: MAX_BROTLI_WINDOW_BYTES,
            unicode_bytes: MAX_UNICODE_MAPPING_BYTES,
            root_decoded_bytes: MAX_ROOT_DECODED_BYTES,
            expansion_amplification: MAX_REFERENCE_EXPANSION_AMPLIFICATION,
            depth: MAX_REFERENCE_EXPANSION_DEPTH,
        }
    }
}
#[derive(Clone, Copy, Debug)]
pub enum Limit {
    DecodedStreamBytes,
    FilterLayers,
    PredictorRowBytes,
    BrotliWindowBytes,
    UnicodeMappingBytes,
    RootDecodedBytes,
    ReferenceExpansionAmplification,
    ReferenceExpansionDepth,
    PageTreePageAmplification,
}
impl Policy {
    pub fn disabled() -> Self {
        Self {
            disabled: true,
            ..Self::default()
        }
    }
    pub fn is_disabled(&self) -> bool {
        self.disabled
    }
    /// Scope-specific immutable policy copy; the legacy Form cutoff runs first.
    #[doc(hidden)]
    pub fn for_form_expansion(mut self) -> Self {
        self.depth = self.depth.min(MAX_FORM_EXPANSION_DEPTH);
        self
    }
    /// Internal deterministic-test injection; never deserialized from product configuration.
    #[doc(hidden)]
    pub fn for_test(decoded: usize, layers: usize, row: usize, window: usize) -> Self {
        Self {
            disabled: false,
            decoded,
            layers,
            row,
            window,
            ..Self::default()
        }
    }
    /// Internal font-map test thresholds; not a product configuration surface.
    #[doc(hidden)]
    pub fn for_font_test(unicode_bytes: usize) -> Self {
        Self {
            unicode_bytes,
            ..Self::default()
        }
    }
    /// Internal expansion-root aggregate decode threshold.
    #[doc(hidden)]
    pub fn for_expansion_decoded_test(root_decoded_bytes: usize) -> Self {
        Self {
            root_decoded_bytes,
            ..Self::default()
        }
    }
    /// Internal expansion-root test thresholds; never product configuration.
    #[doc(hidden)]
    pub fn for_expansion_test(expansion_amplification: usize, depth: usize) -> Self {
        Self { expansion_amplification, depth, ..Self::default() }
    }
    pub fn value(&self, kind: Limit) -> usize {
        match kind {
            Limit::DecodedStreamBytes => self.decoded,
            Limit::FilterLayers => self.layers,
            Limit::PredictorRowBytes => self.row,
            Limit::BrotliWindowBytes => self.window,
            Limit::UnicodeMappingBytes => self.unicode_bytes,
            Limit::RootDecodedBytes => self.root_decoded_bytes,
            Limit::ReferenceExpansionAmplification => self.expansion_amplification,
            Limit::ReferenceExpansionDepth => self.depth,
            Limit::PageTreePageAmplification => self.expansion_amplification,
        }
    }
    fn error(&self, kind: Limit, requested: u64) -> Error {
        let (name, unit, scope) = match kind {
            Limit::DecodedStreamBytes => ("max_decoded_stream_bytes", "bytes", "stream_layer"),
            Limit::FilterLayers => ("max_filter_layers", "layers", "stream"),
            Limit::PredictorRowBytes => ("max_predictor_row_bytes", "bytes", "row"),
            Limit::BrotliWindowBytes => ("max_brotli_window_bytes", "bytes", "stream_layer"),
            Limit::UnicodeMappingBytes => ("max_unicode_mapping_bytes", "bytes", "mapping"),
            Limit::RootDecodedBytes => ("max_root_decoded_bytes", "bytes", "expansion_root"),
            Limit::ReferenceExpansionAmplification => ("max_reference_expansion_amplification", "weighted_work_per_distinct_program", "expansion_root"),
            Limit::ReferenceExpansionDepth => ("max_reference_expansion_depth", "frames", "expansion_root"),
            Limit::PageTreePageAmplification => ("max_page_tree_page_amplification", "weighted_work_per_distinct_program", "page_tree"),
        };
        ResourceLimitExceeded {
            limit_name: name,
            limit: self.value(kind) as u64,
            observed_or_requested: requested,
            unit,
            scope,
            phase: match kind {
                Limit::UnicodeMappingBytes => "font_map",
                Limit::ReferenceExpansionAmplification
                | Limit::ReferenceExpansionDepth
                | Limit::PageTreePageAmplification
                | Limit::RootDecodedBytes => "reference_expansion",
                _ => "decode",
            },
            page: None,
            object_ref: None,
            policy_version: POLICY_VERSION,
        }
        .into()
    }
    pub fn check(&self, kind: Limit, requested: usize) -> Result<()> {
        self.check_u64(kind, requested as u64)
    }
    pub fn check_u64(&self, kind: Limit, requested: u64) -> Result<()> {
        if !self.disabled && requested > self.value(kind) as u64 {
            return Err(self.error(kind, requested));
        }
        Ok(())
    }
    /// An unrepresentable local count is a resource error even at a MAX test cap.
    #[doc(hidden)]
    pub fn overflow(&self, kind: Limit) -> Error {
        self.error(kind, u64::MAX)
    }
    /// Predictor declarations retain b20 release arithmetic when limits are disabled.
    /// Allocation sizes must still be validated against actual data by the caller.
    pub fn product(&self, kind: Limit, factors: &[usize]) -> Result<usize> {
        if self.disabled {
            return Ok(factors.iter().fold(1usize, |n, &v| n.wrapping_mul(v)));
        }
        factors
            .iter()
            .try_fold(1usize, |n, &v| n.checked_mul(v))
            .ok_or_else(|| self.error(kind, u64::MAX))
    }
    pub fn append(&self, current: usize, additional: usize) -> Result<usize> {
        let next = current.checked_add(additional).ok_or_else(|| {
            if self.disabled {
                Error::NumericCast("parser size arithmetic overflow".into())
            } else {
                self.error(Limit::DecodedStreamBytes, u64::MAX)
            }
        })?;
        self.check(Limit::DecodedStreamBytes, next)?;
        Ok(next)
    }
}

/// Recover only ordinary PDF errors. Resource failures must remain typed.
pub fn optional<T>(value: Result<T>) -> Result<Option<T>> {
    match value {
        Ok(v) => Ok(Some(v)),
        Err(e) if e.is_resource_limit() => Err(e),
        Err(_) => Ok(None),
    }
}
impl Error {
    pub fn is_resource_limit(&self) -> bool {
        self.resource_limit().is_some()
    }
    pub fn resource_limit(&self) -> Option<&ResourceLimitExceeded> {
        match self {
            Self::ResourceLimitExceeded(e) => Some(e),
            Self::IO(e) => e
                .get_ref()
                .and_then(|e| e.downcast_ref::<ResourceLimitExceeded>()),
            _ => None,
        }
    }
    pub fn at_object(mut self, id: crate::ObjectId) -> Self {
        if let Self::ResourceLimitExceeded(e) = &mut self {
            e.object_ref.get_or_insert(id);
        }
        self
    }
    pub fn is_decode_limit(&self) -> bool {
        self.is_resource_limit()
            || matches!(
                self,
                Self::Decompress(crate::DecompressError::MemoryLimitExceeded { .. })
            )
    }
    pub fn from_decode_io(e: std::io::Error) -> Self {
        if e.get_ref().is_some_and(|inner| inner.is::<Error>()) {
            return *e.into_inner().unwrap().downcast::<Error>().unwrap();
        }
        if let Some(resource) = e
            .get_ref()
            .and_then(|e| e.downcast_ref::<ResourceLimitExceeded>())
        {
            Self::ResourceLimitExceeded(resource.clone())
        } else {
            Self::IO(e)
        }
    }
}
pub(crate) fn into_io(e: Error) -> std::io::Error {
    match e {
        Error::ResourceLimitExceeded(e) => std::io::Error::other(e),
        other => std::io::Error::other(other),
    }
}

/// Normal-window grammar accepted by brotli-decompressor (large_window=false).
/// Invalid/truncated headers are left to the original decoder.
pub(crate) fn brotli_window_bits(input: &[u8]) -> Option<u32> {
    let first = *input.first()?;
    if first & 1 == 0 {
        return Some(16);
    }
    let n = (first >> 1) & 7;
    if n != 0 {
        return Some(17 + u32::from(n));
    }
    let n = (first >> 4) & 7;
    if n != 1 {
        return Some(if n == 0 { 17 } else { 8 + u32::from(n) });
    }
    // The extension marker is invalid with the decoder's large_window=false.
    // Do not turn its second byte into a resource error: the original decoder
    // must retain its ordinary malformed-header error and recovery behavior.
    None
}
