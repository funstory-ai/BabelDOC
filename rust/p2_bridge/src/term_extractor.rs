//! Python bridge for the Rust automatic-term extraction stage.
use crate::{
    callbacks::{self, CallbackError, CancelFlag, ProgressError, PythonService, Recorder},
    translation_service::TranslatorPoolPair,
};
use babeldoc_ir_core::{Document, ReadOnlyView, TreeView, types};
use babeldoc_p2_stages::{
    ProgressReporter,
    translation::{
        automatic_term_extractor as stage,
        automatic_term_extractor::TermParagraphFacts,
        paragraph_helpers::is_pure_numeric_text,
        python_semantics::{PythonJsonValue, json_loads},
    },
};
use pyo3::{
    exceptions::PyRuntimeError,
    prelude::*,
    types::{PyDict, PyTuple},
};
use std::{
    sync::{Arc, Mutex},
    time::Duration,
};

fn ir_error(error: impl std::fmt::Display) -> PyErr {
    PyRuntimeError::new_err(error.to_string())
}

fn character_unicodes(
    characters: impl IntoIterator<
        Item = Result<babeldoc_ir_core::Cursor<types::PdfCharacter>, babeldoc_ir_core::AccessError>,
    >,
    view: &TreeView<'_, ReadOnlyView>,
) -> Result<Vec<Option<String>>, babeldoc_ir_core::AccessError> {
    characters
        .into_iter()
        .map(|character| character?.char_unicode(view).cloned())
        .collect()
}

fn paragraph_facts(
    paragraph: &babeldoc_ir_core::Cursor<types::PdfParagraph>,
    view: &TreeView<'_, ReadOnlyView>,
) -> Result<TermParagraphFacts, babeldoc_ir_core::AccessError> {
    let unicode = paragraph.unicode(view)?;
    let debug_id = paragraph.debug_id(view)?;
    let mut characters = Vec::new();
    let mut placeholder_only = unicode.as_deref().is_some_and(|text| !text.is_empty());
    for composition in paragraph.pdf_paragraph_composition_iter(view)? {
        let composition = composition?;
        if let Some(line) = composition.pdf_line(view)? {
            let parts = character_unicodes(
                (0..line.pdf_character_len(view)?).map(|index| line.pdf_character_at(view, index)),
                view,
            )?;
            placeholder_only &= parts.iter().all(|part| {
                part.as_deref()
                    .is_some_and(babeldoc_p2_stages::translation::python_semantics::python_isspace)
            });
            characters.extend(parts);
        } else if let Some(group) = composition.pdf_same_style_characters(view)? {
            let parts = character_unicodes(
                (0..group.pdf_character_len(view)?)
                    .map(|index| group.pdf_character_at(view, index)),
                view,
            )?;
            placeholder_only &= parts.iter().all(|part| {
                part.as_deref()
                    .is_some_and(babeldoc_p2_stages::translation::python_semantics::python_isspace)
            });
            characters.extend(parts);
        } else if composition
            .pdf_same_style_unicode_characters(view)?
            .is_some()
        {
            let group = composition
                .pdf_same_style_unicode_characters(view)?
                .expect("checked Some");
            placeholder_only &= group
                .unicode(view)?
                .as_deref()
                .is_some_and(babeldoc_p2_stages::translation::python_semantics::python_isspace);
        } else if let Some(formula) = composition.pdf_formula(view)? {
            characters.extend(character_unicodes(formula.pdf_character_iter(view)?, view)?);
        } else if let Some(character) = composition.pdf_character(view)? {
            let unicode = character.char_unicode(view)?;
            placeholder_only &= unicode
                .as_deref()
                .is_some_and(babeldoc_p2_stages::translation::python_semantics::python_isspace);
            characters.push(unicode.clone());
        } else {
            placeholder_only = false;
        }
    }
    let cid_count = characters
        .iter()
        .filter(|text| text.as_deref().is_some_and(is_cid_text))
        .count();
    let cid = cid_count * 5 > characters.len() * 4;
    let pure_numeric = unicode.as_deref().is_some_and(is_pure_numeric_text);
    Ok(TermParagraphFacts {
        debug_id: debug_id.clone(),
        unicode: unicode.clone(),
        cid,
        pure_numeric,
        placeholder_only,
    })
}

fn is_cid_text(text: &str) -> bool {
    text.strip_prefix("(cid:")
        .and_then(|text| text.strip_suffix(')'))
        .is_some_and(|digits| {
            !digits.is_empty() && digits.bytes().all(|byte| byte.is_ascii_digit())
        })
}

fn document_pages(
    document: &mut Document,
) -> Result<Vec<Vec<TermParagraphFacts>>, babeldoc_ir_core::AccessError> {
    let view = document.view::<ReadOnlyView>();
    view.document()
        .page_iter(&view)?
        .map(|page| {
            let page = page?;
            let mut facts = Vec::new();
            for paragraph in page.pdf_paragraph_iter(&view)? {
                facts.push(paragraph_facts(&paragraph?, &view)?);
            }
            Ok(facts)
        })
        .collect()
}

fn usage_counters(service: &Bound<'_, PyAny>) -> PyResult<[i64; 4]> {
    let values = service
        .call_method0("usage_counters")?
        .cast_into::<PyTuple>()?;
    Ok([
        values.get_item(0)?.extract()?,
        values.get_item(1)?.extract()?,
        values.get_item(2)?.extract()?,
        values.get_item(3)?.extract()?,
    ])
}

fn output_path(config: &Bound<'_, PyAny>, name: &str) -> PyResult<Option<String>> {
    if !config.getattr("debug")?.extract::<bool>()? && config.getattr("working_dir")?.is_none() {
        return Ok(None);
    }
    let path = config.call_method1("get_working_file_path", (name,))?;
    Ok(Some(path.str()?.to_str()?.to_owned()))
}

fn python_output_value(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<PythonJsonValue> {
    let json = py.import("json")?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("ensure_ascii", false)?;
    let encoded = json
        .call_method("dumps", (value,), Some(&kwargs))?
        .extract::<String>()?;
    json_loads(&encoded).map_err(|error| PyRuntimeError::new_err(error.to_string()))
}

fn write_output(path: &str, bytes: &[u8]) -> PyResult<()> {
    if let Some(parent) = std::path::Path::new(path).parent() {
        std::fs::create_dir_all(parent).map_err(ir_error)?;
    }
    std::fs::write(path, bytes).map_err(ir_error)
}

pub fn process(
    py: Python<'_>,
    document: &mut Document,
    config: &Bound<'_, PyAny>,
    shared_context: &Bound<'_, PyAny>,
    dispatcher: Py<PyAny>,
    translator_service: Py<PyAny>,
    flag: CancelFlag,
    monitor: Py<PyAny>,
) -> PyResult<()> {
    let pages = document_pages(document).map_err(ir_error)?;
    let root = crate::shared_context::root_from_handle(shared_context)?;
    let lang_out = config.getattr("lang_out")?.extract::<String>()?;
    let workers = config
        .getattr("term_pool_max_workers")?
        .extract::<usize>()?
        .max(1);
    let paths = [
        output_path(config, "term_extractor_tracking.json")?,
        output_path(config, "term_extractor_freq.json")?,
        output_path(config, "auto_extractor_glossary.csv")?,
    ];
    let start = usage_counters(translator_service.bind(py))?;
    let glossaries = root.user_glossaries();
    let translator = translator_service.clone_ref(py);
    let recorder = Recorder::for_dispatcher(py, dispatcher.bind(py));
    let service = PythonService {
        dispatcher,
        recorder,
    };
    callbacks::run(py, workers, move |runtime, progress_run| {
        let progress = progress_run
            .start(
                monitor,
                stage::STAGE_NAME,
                pages.iter().map(Vec::len).sum::<usize>() as u64,
                Duration::from_millis(30),
            )
            .map_err(ProgressError::into_python)?;
        let tracking = stage::TermTracking {
            page: vec![Default::default(); pages.len()],
        };
        let tracking = Arc::new(Mutex::new(tracking));
        let pool = Arc::new(TranslatorPoolPair::new(
            service,
            stage::STAGE_NAME,
            workers,
            root.clone(),
        ));
        let result = runtime.block_on(async {
            let mut tasks = tokio::task::JoinSet::new();
            let mut submitted_count = 0;
            let mut page_error = None;
            'pages: for (page_index, page) in pages.iter().enumerate() {
                if let Err(error) = flag.checkpoint() {
                    Python::attach(|py| {
                        error.value(py).setattr("_p2_term_page_index", page_index)
                    })?;
                    page_error = Some(error);
                    break 'pages;
                }
                let actions = {
                    let mut tracking = tracking.lock().unwrap_or_else(|error| error.into_inner());
                    stage::plan_page_actions(
                        page_index,
                        page,
                        &lang_out,
                        &glossaries,
                        &mut tracking,
                    )
                };
                for action in actions {
                    let stage::PageAction::Batch(batch) = action else {
                        progress
                            .report(1, None)
                            .map_err(|error| error.into_python())?;
                        tokio::task::yield_now().await;
                        continue;
                    };
                    let request_pool = Arc::clone(&pool);
                    let progress = progress.clone();
                    let tracking = Arc::clone(&tracking);
                    let admission_flag = flag.clone();
                    let admission_tracking = Arc::clone(&tracking);
                    let admission_batch = batch.clone();
                    tasks.spawn(async move {
                        let priority = stage::batch_priority(batch.token_count);
                        let request_args = Python::attach(|py| {
                            let kwargs = PyDict::new(py);
                            kwargs.set_item("paragraph_token_count", batch.token_count)?;
                            kwargs.set_item("request_json_mode", true)?;
                            let args = PyTuple::new(
                                py,
                                [
                                    batch.prompt.clone().into_pyobject(py)?.into_any(),
                                    kwargs.into_any(),
                                ],
                            )?;
                            PyResult::Ok(args.unbind())
                        })?;
                        let response = request_pool
                            .request(
                                "llm_translate",
                                format!("term:{}:{}", batch.page_index, batch.tracking_index),
                                request_args,
                                None,
                                priority,
                                move || {
                                    if admission_flag.is_cancelled() {
                                        return false;
                                    }
                                    stage::record_input(
                                        &mut admission_tracking
                                            .lock()
                                            .unwrap_or_else(|error| error.into_inner()),
                                        &admission_batch,
                                    );
                                    true
                                },
                            )
                            .await;
                        let response = match response {
                            Ok(Some(response)) => response,
                            Ok(None) => return Ok::<(), PyErr>(()),
                            Err(CallbackError::Python(_)) => {
                                progress
                                    .report(batch.paragraphs.len() as u64, None)
                                    .map_err(|error| error.into_python())?;
                                return Ok::<(), PyErr>(());
                            }
                            Err(CallbackError::Bridge(error)) => {
                                progress
                                    .report(batch.paragraphs.len() as u64, None)
                                    .map_err(|error| error.into_python())?;
                                return Err(PyRuntimeError::new_err(error));
                            }
                        };
                        let parsed = Python::attach(
                            |py| -> PyResult<
                                Option<(PythonJsonValue, Option<Vec<(String, String)>>)>,
                            > {
                                let value = response.bind(py);
                                let stored = python_output_value(py, value)?;
                                let text = value.extract::<String>().ok();
                                let terms = text
                                    .as_deref()
                                    .and_then(|text| stage::parse_terms(text).ok());
                                Ok(Some((stored, terms)))
                            },
                        );
                        if let Ok(Some((output, terms))) = parsed {
                            stage::record_output(
                                &mut tracking.lock().unwrap_or_else(|error| error.into_inner()),
                                &batch,
                                output,
                            );
                            if let Some(terms) = terms {
                                for (source, target) in terms {
                                    request_pool
                                        .shared_context()
                                        .add_raw_extracted_term_pair(source, target);
                                }
                            }
                        } else if let Err(error) = parsed {
                            progress
                                .report(batch.paragraphs.len() as u64, None)
                                .map_err(|e| e.into_python())?;
                            return Err(error);
                        }
                        progress
                            .report(batch.paragraphs.len() as u64, None)
                            .map_err(|error| error.into_python())?;
                        Ok::<(), PyErr>(())
                    });
                    submitted_count += 1;
                    while {
                        let readings = pool.readings();
                        readings.admission_order.len() + readings.queued < submitted_count
                    } {
                        tokio::task::yield_now().await;
                    }
                }
            }
            let mut task_error = None;
            while let Some(result) = tasks.join_next().await {
                match result {
                    Ok(Ok(())) => {}
                    Ok(Err(error)) => {
                        task_error.get_or_insert(error);
                    }
                    Err(error) => {
                        task_error
                            .get_or_insert_with(|| PyRuntimeError::new_err(error.to_string()));
                    }
                };
            }
            if let Some(error) = task_error.or(page_error) {
                let tracking_json = tracking
                    .lock()
                    .unwrap_or_else(|error| error.into_inner())
                    .to_json();
                Python::attach(|py| {
                    error
                        .value(py)
                        .setattr("_p2_term_tracking_json", tracking_json)
                })?;
                Err(error)
            } else {
                Ok(())
            }
        });
        if let Err(error) = result {
            let _ = progress.end(None);
            return Err(error);
        }
        progress.end(None).map_err(ProgressError::into_python)?;

        root.finalize_auto_extracted_glossary()
            .map_err(PyRuntimeError::new_err)?;
        let end = Python::attach(|py| usage_counters(translator.bind(py)))?;
        let deltas = std::array::from_fn::<_, 4, _>(|index| end[index] - start[index]);
        Python::attach(|py| {
            translator.bind(py).call_method1(
                "record_term_extraction_usage",
                (deltas[0], deltas[1], deltas[2], deltas[3]),
            )?;
            Ok::<(), PyErr>(())
        })?;

        let tracking_json = tracking
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .to_json();
        if let Some(path) = &paths[0] {
            write_output(path, tracking_json.as_bytes())?;
        }
        let raw_terms = root.raw_extracted_terms();
        if let Some(path) = &paths[1] {
            write_output(path, stage::frequency_json(&raw_terms).as_bytes())?;
        }
        let csv = root
            .auto_extracted_glossary()
            .map_or_else(String::new, |glossary| glossary.to_csv());
        if let Some(path) = &paths[2] {
            let mut bytes = Vec::new();
            if !csv.is_empty() {
                bytes.extend_from_slice(&[0xef, 0xbb, 0xbf]);
                bytes.extend_from_slice(csv.as_bytes());
            }
            write_output(path, &bytes)?;
        }
        Ok(())
    })
}
