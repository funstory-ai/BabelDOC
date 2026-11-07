//! Local font-asset construction shared by typesetting adapters.
pub use babeldoc_font_core::FontError;
pub use babeldoc_font_core::FontMapper;
pub use babeldoc_font_core::SourceFontStyle;
use babeldoc_font_core::{FontFace, FontFamily, PrimaryFontFamily};
use std::sync::Arc;

pub struct FontMapperWithSourceStyles {
    pub mapper: Arc<FontMapper>,
    pub source_styles: Vec<(String, SourceFontStyle)>,
}

pub enum FontMapperAssetsError {
    InvalidOrMismatchedAsset(String),
    Font(FontError),
    InvalidInput(String),
}

impl std::fmt::Display for FontMapperAssetsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidOrMismatchedAsset(path) => {
                write!(f, "cached FontMapper asset is absent or invalid: {path}")
            }
            Self::Font(error) => write!(f, "{error}"),
            Self::InvalidInput(error) => write!(f, "{error}"),
        }
    }
}

fn verified_font_error(path: &str, error: FontError) -> FontMapperAssetsError {
    match error {
        FontError::AssetChecksumMismatch(_) => {
            FontMapperAssetsError::InvalidOrMismatchedAsset(path.to_owned())
        }
        FontError::Io(error) if error.kind() == std::io::ErrorKind::NotFound => {
            FontMapperAssetsError::InvalidOrMismatchedAsset(path.to_owned())
        }
        error => FontMapperAssetsError::Font(error),
    }
}

fn build_font_mapper(
    family: &serde_json::Value,
    paths: &serde_json::Value,
    expected_sha3: Option<&serde_json::Value>,
    primary_font_family: Option<&str>,
) -> Result<(Arc<FontMapper>, Vec<(String, Arc<FontFace>)>), FontMapperAssetsError> {
    let mut loaded = std::collections::HashMap::<String, Arc<FontFace>>::new();
    let mut loaded_order = Vec::<String>::new();
    let mut load_group = |name: &str| -> Result<Vec<Arc<FontFace>>, FontMapperAssetsError> {
        let ids = family
            .get(name)
            .and_then(serde_json::Value::as_array)
            .ok_or_else(|| {
                FontMapperAssetsError::InvalidInput(format!("font family has no {name} list"))
            })?;
        ids.iter()
            .map(|id| {
                let id = id.as_str().ok_or_else(|| {
                    FontMapperAssetsError::InvalidInput(format!(
                        "font family {name} contains a non-string id"
                    ))
                })?;
                if let Some(face) = loaded.get(id) {
                    return Ok(Arc::clone(face));
                }
                let path = paths
                    .get(id)
                    .and_then(serde_json::Value::as_str)
                    .ok_or_else(|| {
                        FontMapperAssetsError::InvalidInput(format!(
                            "font asset path is missing for {id}"
                        ))
                    })?;
                let face = if let Some(expected_sha3) = expected_sha3 {
                    let expected = expected_sha3
                        .get(id)
                        .and_then(serde_json::Value::as_str)
                        .ok_or_else(|| {
                            FontMapperAssetsError::InvalidInput(format!(
                                "font asset SHA3 is missing for {id}"
                            ))
                        })?;
                    FontFace::load_verified(id, path, expected)
                        .map_err(|error| verified_font_error(path, error))?
                } else {
                    FontFace::load(id, path).map_err(FontMapperAssetsError::Font)?
                };
                loaded_order.push(id.to_owned());
                loaded.insert(id.to_owned(), Arc::clone(&face));
                Ok(face)
            })
            .collect()
    };
    let family = FontFamily {
        normal: load_group("normal")?,
        script: load_group("script")?,
        fallback: load_group("fallback")?,
        base: load_group("base")?,
    };
    drop(load_group);

    let mapper = Arc::new(
        FontMapper::new(family, PrimaryFontFamily::from_option(primary_font_family))
            .map_err(FontMapperAssetsError::Font)?,
    );
    let faces = loaded_order
        .into_iter()
        .map(|id| {
            let face = loaded.get(&id).expect("every loaded font id has a face");
            (id, Arc::clone(face))
        })
        .collect();
    Ok((mapper, faces))
}

/// Builds the immutable asset mapper for the stage-diff adapter from the
/// resolved font family and local asset paths supplied at the Python boundary.
#[doc(hidden)]
pub fn font_mapper_from_asset_paths(
    family: &serde_json::Value,
    paths: &serde_json::Value,
    primary_font_family: Option<&str>,
) -> Result<Arc<FontMapper>, String> {
    build_font_mapper(family, paths, None, primary_font_family)
        .map(|(mapper, _)| mapper)
        .map_err(|error| error.to_string())
}

/// Loads approved font assets once, verifies each loaded byte buffer, and
/// returns the source-style metadata computed from those same faces.
#[doc(hidden)]
pub fn font_mapper_with_source_styles_from_verified_asset_paths(
    family: &serde_json::Value,
    paths: &serde_json::Value,
    expected_sha3: &serde_json::Value,
    primary_font_family: Option<&str>,
) -> Result<FontMapperWithSourceStyles, FontMapperAssetsError> {
    let (mapper, faces) =
        build_font_mapper(family, paths, Some(expected_sha3), primary_font_family)?;
    let mut source_styles = Vec::with_capacity(faces.len());
    for (id, face) in faces {
        let traits = face.traits();
        let flags = face.pdf_font_flags().map_err(FontMapperAssetsError::Font)?;
        source_styles.push((
            id,
            SourceFontStyle {
                bold: traits.bold,
                italic: traits.italic,
                monospaced: traits.monospaced,
                serif: flags.serif,
            },
        ));
    }
    Ok(FontMapperWithSourceStyles {
        mapper,
        source_styles,
    })
}
