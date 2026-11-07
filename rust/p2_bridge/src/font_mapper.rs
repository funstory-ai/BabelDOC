//! Read-only Python handle to the existing, immutable Rust asset mapper.
use babeldoc_p2_stages::font_mapper::{
    FontError, FontMapper, FontMapperAssetsError, font_mapper_from_asset_paths,
    font_mapper_with_source_styles_from_verified_asset_paths,
};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::sync::Arc;

#[pyclass(frozen, module = "babeldoc._rust_il")]
pub struct FontMapperHandle {
    pub mapper: Arc<FontMapper>,
}
impl FontMapperHandle {
    pub fn from_shared(mapper: Arc<FontMapper>) -> Self {
        Self { mapper }
    }
}
#[pymethods]
impl FontMapperHandle {
    pub fn has_char(&self, py: Python<'_>, text: &str) -> bool {
        py.detach(|| self.mapper.has_char(text))
    }

    pub fn mapped_font_id(
        &self,
        bold: bool,
        italic: bool,
        monospaced: bool,
        serif: bool,
        ch: char,
    ) -> Option<String> {
        self.mapper
            .map(
                babeldoc_p2_stages::font_mapper::SourceFontStyle {
                    bold,
                    italic,
                    monospaced,
                    serif,
                },
                ch,
            )
            .map(|font| font.id().to_owned())
    }

    pub fn font_source_styles(&self, font_id: &str) -> PyResult<(bool, bool, bool, bool)> {
        let face = self.mapper.font_by_id(font_id).ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(format!("unknown font asset {font_id}"))
        })?;
        let traits = face.traits();
        let flags = face
            .pdf_font_flags()
            .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?;
        Ok((traits.bold, traits.italic, traits.monospaced, flags.serif))
    }
}
/// Local resolved assets only. No downloads or Python font mapper is constructed.
#[pyfunction]
#[pyo3(signature=(family_json, paths_json, primary=None))]
pub fn font_mapper_handle(
    family_json: &str,
    paths_json: &str,
    primary: Option<&str>,
) -> PyResult<FontMapperHandle> {
    let family = serde_json::from_str(family_json)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    let paths = serde_json::from_str(paths_json)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    let mapper = font_mapper_from_asset_paths(&family, &paths, primary)
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    Ok(FontMapperHandle::from_shared(mapper))
}

fn map_verified_assets_error(error: FontMapperAssetsError) -> PyErr {
    match error {
        FontMapperAssetsError::InvalidOrMismatchedAsset(path) => {
            pyo3::exceptions::PyFileNotFoundError::new_err(format!(
                "cached FontMapper asset is absent or invalid: {path}"
            ))
        }
        FontMapperAssetsError::Font(FontError::Io(error)) => match error.kind() {
            std::io::ErrorKind::PermissionDenied => {
                pyo3::exceptions::PyPermissionError::new_err(error.to_string())
            }
            _ => pyo3::exceptions::PyOSError::new_err(error.to_string()),
        },
        FontMapperAssetsError::Font(error) => {
            pyo3::exceptions::PyValueError::new_err(error.to_string())
        }
        FontMapperAssetsError::InvalidInput(error) => {
            pyo3::exceptions::PyValueError::new_err(error)
        }
    }
}

/// Builds the production mapper and returns all source styles in one Python
/// boundary crossing. Each asset's checked digest is reused by the flag lookup.
#[pyfunction]
pub fn font_mapper_handle_with_source_styles(
    py: Python<'_>,
    family_json: &str,
    paths_json: &str,
    expected_sha3_json: &str,
    primary: Option<&str>,
) -> PyResult<(FontMapperHandle, Py<PyDict>)> {
    let family = serde_json::from_str(family_json)
        .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?;
    let paths = serde_json::from_str(paths_json)
        .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?;
    let expected_sha3 = serde_json::from_str(expected_sha3_json)
        .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?;
    let assets = font_mapper_with_source_styles_from_verified_asset_paths(
        &family,
        &paths,
        &expected_sha3,
        primary,
    )
    .map_err(map_verified_assets_error)?;

    let styles = PyDict::new(py);
    for (font_id, style) in assets.source_styles {
        let item = PyDict::new(py);
        item.set_item("bold", style.bold)?;
        item.set_item("italic", style.italic)?;
        item.set_item("monospaced", style.monospaced)?;
        item.set_item("serif", style.serif)?;
        styles.set_item(font_id, item)?;
    }
    Ok((
        FontMapperHandle::from_shared(assets.mapper),
        styles.unbind(),
    ))
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<FontMapperHandle>()?;
    module.add_function(wrap_pyfunction!(font_mapper_handle, module)?)?;
    module.add_function(wrap_pyfunction!(
        font_mapper_handle_with_source_styles,
        module
    )?)?;
    Ok(())
}
