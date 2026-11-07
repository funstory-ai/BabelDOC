use babeldoc_p2_stages::runtime::shared_context::{SharedContextRoot, TitleContextSnapshot};
use babeldoc_p2_stages::translation::glossary::{Glossary, GlossaryEntry};
use pyo3::prelude::*;

#[pyclass(name = "SharedContextCrossSplitPart")]
pub struct PySharedContextCrossSplitPart {
    pub root: SharedContextRoot,
}

#[pymethods]
impl PySharedContextCrossSplitPart {
    #[new]
    #[pyo3(signature = (initial_glossaries=None))]
    fn new(initial_glossaries: Option<Vec<Py<PyAny>>>, py: Python<'_>) -> PyResult<Self> {
        let glossaries = initial_glossaries
            .unwrap_or_default()
            .iter()
            .map(|glossary| glossary_from_python(py, glossary.bind(py)))
            .collect::<PyResult<Vec<_>>>()?;
        Ok(Self {
            root: SharedContextRoot::new(glossaries),
        })
    }

    fn initialize_glossaries(
        &self,
        glossaries: Option<Vec<Py<PyAny>>>,
        py: Python<'_>,
    ) -> PyResult<()> {
        let converted = glossaries
            .unwrap_or_default()
            .iter()
            .map(|glossary| glossary_from_python(py, glossary.bind(py)))
            .collect::<PyResult<Vec<_>>>()?;
        self.root.initialize_glossaries(converted);
        Ok(())
    }

    fn snapshot_title_paragraph(
        &self,
        py: Python<'_>,
        paragraph: Option<Bound<'_, PyAny>>,
    ) -> PyResult<Option<Py<PyAny>>> {
        let Some(paragraph) = paragraph else {
            return Ok(None);
        };
        let snapshot = TitleContextSnapshot {
            debug_id: optional_attr(&paragraph, "debug_id")?,
            unicode: optional_attr(&paragraph, "unicode")?,
            layout_label: optional_attr(&paragraph, "layout_label")?,
        };
        title_snapshot_to_python(py, &snapshot).map(Some)
    }

    fn add_raw_extracted_term_pair(&self, src: String, tgt: String) {
        self.root.add_raw_extracted_term_pair(src, tgt);
    }

    fn contains_term(&self, term: &str) -> bool {
        self.root.contains_term(term)
    }

    fn finalize_auto_extracted_glossary(&self) -> PyResult<()> {
        self.root
            .finalize_auto_extracted_glossary()
            .map_err(pyo3::exceptions::PyValueError::new_err)
    }

    fn get_glossaries(&self, py: Python<'_>) -> PyResult<Vec<Py<PyAny>>> {
        self.root
            .get_glossaries()
            .iter()
            .map(|g| glossary_to_python(py, g))
            .collect()
    }

    fn get_glossaries_for_translation(
        &self,
        auto_extract_enabled: bool,
        py: Python<'_>,
    ) -> PyResult<Vec<Py<PyAny>>> {
        self.root
            .get_glossaries_for_translation(auto_extract_enabled)
            .iter()
            .map(|g| glossary_to_python(py, g))
            .collect()
    }

    fn add_valid_counts(&self, char_count: i64, token_count: i64) {
        self.root.add_valid_counts(char_count, token_count);
    }

    #[getter]
    fn first_paragraph(&self, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        self.root
            .first_paragraph()
            .as_ref()
            .map(|s| title_snapshot_to_python(py, s))
            .transpose()
    }
    #[setter]
    fn set_first_paragraph(&self, value: Option<Bound<'_, PyAny>>) -> PyResult<()> {
        self.root
            .set_first_paragraph(value.as_ref().map(snapshot_from_python).transpose()?);
        Ok(())
    }

    #[getter]
    fn recent_title_paragraph(&self, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        self.root
            .recent_title_paragraph()
            .as_ref()
            .map(|s| title_snapshot_to_python(py, s))
            .transpose()
    }
    #[setter]
    fn set_recent_title_paragraph(&self, value: Option<Bound<'_, PyAny>>) -> PyResult<()> {
        self.root
            .set_recent_title_paragraph(value.as_ref().map(snapshot_from_python).transpose()?);
        Ok(())
    }

    #[getter]
    fn auto_enabled_ocr_workaround(&self) -> bool {
        self.root.auto_enabled_ocr_workaround()
    }
    #[setter]
    fn set_auto_enabled_ocr_workaround(&self, value: bool) {
        self.root.set_auto_enabled_ocr_workaround(value);
    }

    #[getter]
    fn valid_char_count_total(&self) -> i64 {
        self.root.valid_counts().0
    }
    #[getter]
    fn total_valid_text_token_count(&self) -> i64 {
        self.root.valid_counts().1
    }

    #[getter]
    fn raw_extracted_terms(&self) -> Vec<(String, String)> {
        self.root.raw_extracted_terms()
    }
    #[getter]
    fn user_glossaries(&self, py: Python<'_>) -> PyResult<Vec<Py<PyAny>>> {
        self.root
            .user_glossaries()
            .iter()
            .map(|g| glossary_to_python(py, g))
            .collect()
    }
    #[getter]
    fn auto_extracted_glossary(&self, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        self.root
            .auto_extracted_glossary()
            .as_ref()
            .map(|g| glossary_to_python(py, g))
            .transpose()
    }
    #[getter]
    fn unique_name(&self) -> String {
        self.root.unique_name()
    }
}

pub fn root_from_handle(value: &Bound<'_, PyAny>) -> PyResult<SharedContextRoot> {
    Ok(value
        .extract::<PyRef<'_, PySharedContextCrossSplitPart>>()?
        .root
        .clone())
}

fn glossary_from_python(py: Python<'_>, glossary: &Bound<'_, PyAny>) -> PyResult<Glossary> {
    let name: String = glossary.getattr("name")?.extract()?;
    let entries = glossary
        .getattr("entries")?
        .try_iter()?
        .map(|entry| {
            let entry = entry?;
            Ok(GlossaryEntry {
                source: entry.getattr("source")?.extract()?,
                target: entry.getattr("target")?.extract()?,
                target_language: entry.getattr("target_language")?.extract()?,
            })
        })
        .collect::<PyResult<Vec<_>>>()?;
    let _ = py;
    Glossary::new(name, entries)
        .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))
}

fn glossary_to_python(py: Python<'_>, glossary: &Glossary) -> PyResult<Py<PyAny>> {
    let module = py.import("babeldoc.glossary")?;
    let entry_type = module.getattr("GlossaryEntry")?;
    let entries = glossary
        .entries
        .iter()
        .map(|entry| {
            entry_type.call1((
                &entry.source,
                &entry.target,
                entry.target_language.as_deref(),
            ))
        })
        .collect::<PyResult<Vec<_>>>()?;
    Ok(module
        .getattr("Glossary")?
        .call1((&glossary.name, entries))?
        .unbind())
}

fn optional_attr(object: &Bound<'_, PyAny>, name: &str) -> PyResult<Option<String>> {
    match object.getattr(name) {
        Ok(value) if !value.is_none() => value.extract().map(Some),
        Ok(_) => Ok(None),
        Err(error) if error.is_instance_of::<pyo3::exceptions::PyAttributeError>(object.py()) => {
            Ok(None)
        }
        Err(error) => Err(error),
    }
}

fn snapshot_from_python(value: &Bound<'_, PyAny>) -> PyResult<TitleContextSnapshot> {
    Ok(TitleContextSnapshot {
        debug_id: optional_attr(value, "debug_id")?,
        unicode: optional_attr(value, "unicode")?,
        layout_label: optional_attr(value, "layout_label")?,
    })
}

fn title_snapshot_to_python(
    py: Python<'_>,
    snapshot: &TitleContextSnapshot,
) -> PyResult<Py<PyAny>> {
    let ty = py
        .import("babeldoc.format.pdf.translation_config")?
        .getattr("TitleContextSnapshot")?;
    Ok(ty
        .call1((
            snapshot.debug_id.as_deref(),
            snapshot.unicode.as_deref(),
            snapshot.layout_label.as_deref(),
        ))?
        .unbind())
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PySharedContextCrossSplitPart>()
}
