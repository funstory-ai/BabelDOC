use babeldoc_p2_stages::translation::{
    glossary::{self, Glossary, GlossaryEntry},
    token_count,
};
use pyo3::{exceptions::PyValueError, prelude::*};

#[pyfunction]
fn translation_count_tokens(text: &str) -> usize {
    token_count::count_tokens(text)
}

#[pyfunction]
fn translation_glossary_entries(
    name: String,
    entries: Vec<(String, String, Option<String>)>,
) -> PyResult<Vec<(String, String, Option<String>)>> {
    let entries = entries
        .into_iter()
        .map(|(source, target, target_language)| GlossaryEntry {
            source,
            target,
            target_language,
        })
        .collect();
    let glossary =
        Glossary::new(name, entries).map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok(glossary
        .entries
        .into_iter()
        .map(|entry| (entry.source, entry.target, entry.target_language))
        .collect())
}

#[pyfunction]
fn translation_glossary_matches(
    name: String,
    entries: Vec<(String, String, Option<String>)>,
    text: &str,
) -> PyResult<Vec<(String, String)>> {
    let entries = entries
        .into_iter()
        .map(|(source, target, target_language)| GlossaryEntry {
            source,
            target,
            target_language,
        })
        .collect();
    let glossary =
        Glossary::new(name, entries).map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok(glossary.active_entries_for_text(text))
}

#[pyfunction]
fn translation_finalize_terms(
    name: String,
    raw_terms: Vec<(String, String)>,
) -> PyResult<Option<(String, Vec<(String, String, Option<String>)>, String)>> {
    glossary::finalize_terms(name, &raw_terms)
        .map(|glossary| glossary.map(|glossary| as_tuple(glossary)))
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

fn as_tuple(glossary: Glossary) -> (String, Vec<(String, String, Option<String>)>, String) {
    let csv = glossary.to_csv();
    let entries = glossary
        .entries
        .into_iter()
        .map(|entry| (entry.source, entry.target, entry.target_language))
        .collect();
    (glossary.name, entries, csv)
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(translation_count_tokens, module)?)?;
    module.add_function(wrap_pyfunction!(translation_glossary_entries, module)?)?;
    module.add_function(wrap_pyfunction!(translation_glossary_matches, module)?)?;
    module.add_function(wrap_pyfunction!(translation_finalize_terms, module)?)?;
    Ok(())
}
