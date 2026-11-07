use crate::callbacks::{CancelFlag, ProgressError, ProgressRun, PythonProgress};
use crate::font_mapper::FontMapperHandle;
use babeldoc_ir_core::Document;
use babeldoc_p2_stages::font_mapper::SourceFontStyle;
use babeldoc_p2_stages::stages::paragraph_finder_inputs::{
    ParagraphFinderConfig, ParagraphFinderInputs, RandomDebugIdSource,
};
use babeldoc_p2_stages::stages::styles_and_formulas_config::StylesAndFormulasConfig;
use babeldoc_p2_stages::stages::typesetting::Typesetting;
use babeldoc_p2_stages::{StageContext, StageError};
use pyo3::prelude::*;
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::time::Duration;

struct ProgressAdapter(PythonProgress);
impl babeldoc_p2_stages::ProgressReporter for ProgressAdapter {
    type Error = StageError;
    fn report(&self, amount: u64, total: Option<u64>) -> Result<(), Self::Error> {
        self.0.report(amount, total).map_err(progress_error)
    }
    fn checkpoint(&self) -> Result<(), Self::Error> {
        self.0.checkpoint().map_err(progress_error)
    }
}

fn progress_error(error: ProgressError) -> StageError {
    match error {
        ProgressError::Cancelled => StageError::Cancelled,
        other => StageError::Failed(other.into_python().to_string()),
    }
}

fn finish_stage(progress: &PythonProgress, result: Result<(), StageError>) -> PyResult<()> {
    let error = result.err();
    let error = error.map(|error| {
        let python_error = Python::attach(|py| crate::stage_error::to_python(py, &error));
        ProgressError::from(python_error)
    });
    progress.end(error).map_err(ProgressError::into_python)
}

fn finish_pattern_stage(
    progress: &PythonProgress,
    result: Result<(), StageError>,
    matcher: Option<&crate::formula_patterns::PythonFormulaMatcher>,
) -> PyResult<()> {
    if let Some(error) = matcher.and_then(|m| m.take_error()) {
        return progress
            .end(Some(ProgressError::from(error)))
            .map_err(ProgressError::into_python);
    }
    finish_stage(progress, result)
}

fn field<'a>(value: &'a Value, name: &str) -> Option<&'a Value> {
    value.get(name)
}
fn bool_field(value: &Value, name: &str, default: bool) -> bool {
    field(value, name)
        .and_then(Value::as_bool)
        .unwrap_or(default)
}
fn string_set(value: &Value, name: &str) -> PyResult<Option<BTreeSet<String>>> {
    let Some(values) = field(value, name).and_then(Value::as_array) else {
        return Ok(None);
    };
    values
        .iter()
        .map(|value| {
            value.as_str().map(str::to_owned).ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(format!("{name} must contain strings"))
            })
        })
        .collect::<PyResult<BTreeSet<_>>>()
        .map(Some)
}

fn start<'a>(
    progress: &'a ProgressRun,
    monitor: &Py<PyAny>,
    name: &str,
    total: u64,
) -> PyResult<(PythonProgress, ProgressAdapter)> {
    let progress = progress
        .start(
            Python::attach(|py| monitor.clone_ref(py)),
            name,
            total,
            Duration::from_millis(100),
        )
        .map_err(ProgressError::into_python)?;
    Ok((progress.clone(), ProgressAdapter(progress)))
}

pub fn paragraph_finder(
    document: &mut Document,
    config: &Value,
    mapper: &FontMapperHandle,
    monitor: Py<PyAny>,
    cancellation: &CancelFlag,
) -> PyResult<()> {
    let total = {
        let view = document.view::<babeldoc_ir_core::ReadOnlyView>();
        view.document()
            .page_len(&view)
            .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?
            as u64
    };
    let mut run = ProgressRun::new()?;
    let (progress, mut reporter) = start(&run, &monitor, "Parse Paragraphs", total)?;
    let formula_font_pattern_hits = string_set(config, "formula_font_pattern_hits")?;
    let matcher = crate::formula_patterns::PythonFormulaMatcher::from_config(config);
    let cfg = ParagraphFinderConfig {
        formula_matcher: Some(matcher.clone()),
        debug: bool_field(config, "debug", false),
        ocr_workaround: bool_field(config, "ocr_workaround", false),
        merge_alternating_line_numbers: field(config, "merge_alternating_line_numbers")
            .and_then(Value::as_bool),
        split_short_lines: bool_field(config, "split_short_lines", false),
        short_line_split_factor: field(config, "short_line_split_factor")
            .and_then(Value::as_f64)
            .unwrap_or(0.8),
        formula_font_pattern_hits,
    };
    let mut ids = RandomDebugIdSource;
    let mut inputs = ParagraphFinderInputs {
        config: &cfg,
        font_mapper: &mapper.mapper,
        debug_ids: &mut ids,
    };
    let mut context = StageContext::new(cancellation, &mut reporter);
    let result = babeldoc_p2_stages::stages::paragraph_finder::process_with(
        &mut inputs,
        document,
        &mut context,
    );
    finish_pattern_stage(&progress, result, Some(matcher.as_ref()))?;
    run.close()?;
    Ok(())
}

pub fn styles_and_formulas(
    document: &mut Document,
    config: &Value,
    mapper: &FontMapperHandle,
    monitor: Py<PyAny>,
    cancellation: &CancelFlag,
) -> PyResult<()> {
    let total = {
        let view = document.view::<babeldoc_ir_core::ReadOnlyView>();
        view.document()
            .page_len(&view)
            .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?
            as u64
    };
    let mut run = ProgressRun::new()?;
    let (progress, mut reporter) = start(&run, &monitor, "Parse Formulas and Styles", total)?;
    let config_fields = field(config, "config_fields").unwrap_or(config);
    let font_pattern_hits = string_set(config, "font_pattern_hits")?;
    let character_pattern_hits = field(config, "character_pattern_hits")
        .and_then(Value::as_object)
        .map(|values| {
            values
                .iter()
                .map(|(key, value)| {
                    value
                        .as_bool()
                        .map(|value| (key.clone(), value))
                        .ok_or_else(|| {
                            pyo3::exceptions::PyValueError::new_err(
                                "character_pattern_hits values must be bool",
                            )
                        })
                })
                .collect::<PyResult<BTreeMap<_, _>>>()
        })
        .transpose()?;
    let matcher = crate::formula_patterns::PythonFormulaMatcher::from_config(config_fields);
    let cfg = StylesAndFormulasConfig {
        formula_matcher: Some(matcher.clone()),
        font_mapper: mapper.mapper.clone(),
        character_pattern_hits,
        font_pattern_hits,
        skip_formula_offset_calculation: bool_field(
            config_fields,
            "skip_formula_offset_calculation",
            false,
        ),
        ocr_workaround: bool_field(config_fields, "ocr_workaround", false),
        remove_non_formula_lines: bool_field(config_fields, "remove_non_formula_lines", false),
        figure_table_protection_threshold: field(
            config_fields,
            "figure_table_protection_threshold",
        )
        .and_then(Value::as_f64),
        non_formula_line_iou_threshold: field(config_fields, "non_formula_line_iou_threshold")
            .and_then(Value::as_f64),
    };
    let mut context = StageContext::new(cancellation, &mut reporter);
    let result =
        babeldoc_p2_stages::stages::styles_and_formulas::process(&cfg, document, &mut context);
    finish_pattern_stage(&progress, result, Some(matcher.as_ref()))?;
    run.close()?;
    Ok(())
}

pub fn add_debug_information(document: &mut Document, debug: bool) -> PyResult<()> {
    babeldoc_p2_stages::stages::add_debug_information::process(debug, document)
        .map_err(|error| Python::attach(|py| crate::stage_error::to_python(py, &error)))?;
    Ok(())
}

pub fn typesetting(
    document: &mut Document,
    config: &Value,
    mapper: &FontMapperHandle,
    monitor: Py<PyAny>,
    cancellation: &CancelFlag,
) -> PyResult<()> {
    let fields = field(config, "config_fields").unwrap_or(config);
    let total = {
        let view = document.view::<babeldoc_ir_core::ReadOnlyView>();
        view.document()
            .page_len(&view)
            .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?
            as u64
    };
    let mut run = ProgressRun::new()?;
    let (progress, mut reporter) = start(&run, &monitor, "Typesetting", total)?;
    let mut styles = HashMap::new();
    if let Some(values) = field(config, "font_source_styles").and_then(Value::as_object) {
        for (id, value) in values {
            let read = |key: &str| {
                value.get(key).and_then(Value::as_bool).ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err(format!(
                        "missing font source style {id}.{key}"
                    ))
                })
            };
            styles.insert(
                id.clone(),
                SourceFontStyle {
                    bold: read("bold")?,
                    italic: read("italic")?,
                    monospaced: read("monospaced")?,
                    serif: read("serif")?,
                },
            );
        }
    }
    let lang = field(fields, "lang_out")
        .and_then(Value::as_str)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("missing lang_out"))?;
    let has_progress = bool_field(config, "progress_monitor_present", true);
    let stage = Typesetting::new(lang, mapper.mapper.clone(), has_progress, styles);
    let mut context = StageContext::new(cancellation, &mut reporter);
    let result = stage.process(document, &mut context);
    finish_stage(&progress, result)?;
    run.close()?;
    Ok(())
}

#[cfg(test)]
mod production_stages_tests;
