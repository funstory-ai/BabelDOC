//! Native LLM-only batch planner and translator bridge.
use crate::{
    callbacks::{self, CancelFlag, ProgressError, PythonService, Recorder},
    font_mapper::FontMapperHandle,
    ordinary_translation::{self as ordinary, RegexService},
    translation_service::TranslatorPoolPair,
};
use babeldoc_ir_core::Document;
use babeldoc_p2_stages::{
    ProgressReporter,
    runtime::shared_context::SharedContextRoot,
    translation::{
        llm_only as stage,
        ordinary_translation::{self as ordinary_stage, PlaceholderRegex},
        token_count::count_tokens,
    },
};
use pyo3::{
    exceptions::PyTypeError,
    prelude::*,
    types::{PyDict, PyTuple},
};
use std::{
    collections::HashMap,
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
    time::Duration,
};

#[derive(Clone)]
struct Work {
    batch: stage::BatchPlan,
    jobs: Vec<ordinary::Job>,
    prompt: String,
    title_context: Option<(String, String)>,
    local_title_context: Option<(String, String)>,
    inputs: Vec<String>,
    prepare_error: Option<String>,
    admitted: Arc<AtomicBool>,
    tracker_group: usize,
    skipped_count: usize,
}

#[derive(Clone)]
struct Output {
    job: ordinary::Job,
    input: ordinary_stage::TranslateInput,
    text: String,
    fallback: bool,
    fallback_prompt: Option<String>,
    tracker_kind: stage::GroupKind,
    tracker_group: usize,
    tracker_index: usize,
}

struct FallbackRequest {
    input: ordinary_stage::TranslateInput,
    text: String,
    formula_hint: Option<String>,
    paragraph_token_count: usize,
}

enum ScheduledWork {
    Checkpoint,
    Batch(Work),
}

fn py_error(error: impl std::fmt::Display) -> PyErr {
    pyo3::exceptions::PyRuntimeError::new_err(error.to_string())
}

fn fallback_request(
    job: &ordinary::Job,
    include_formula_hint: bool,
    batch_failed: bool,
) -> Option<FallbackRequest> {
    let input = job.fallback_input.clone()?;
    let paragraph_token_text = if batch_failed {
        &job.tracker_unicode
    } else {
        &job.text
    };
    Some(FallbackRequest {
        text: input.text.clone(),
        formula_hint: include_formula_hint
            .then(|| ordinary::formula_hint(&input))
            .flatten(),
        // A batch-level exception restores paragraph.unicode before counting;
        // an item-level response fallback counts it before the restore.
        paragraph_token_count: count_tokens(paragraph_token_text),
        input,
    })
}

fn request_args(
    py: Python<'_>,
    text: &str,
    count: usize,
    json_mode: bool,
) -> PyResult<(Py<PyTuple>, Py<PyDict>)> {
    let args = PyTuple::new(py, [text])?.unbind();
    let params = PyDict::new(py);
    params.set_item("paragraph_token_count", count)?;
    if json_mode {
        params.set_item("request_json_mode", true)?;
    }
    let kwargs = PyDict::new(py);
    kwargs.set_item("rate_limit_params", params)?;
    Ok((args, kwargs.unbind()))
}

fn pages_and_jobs(
    document: &mut Document,
    root: &SharedContextRoot,
    translator: &Bound<'_, PyAny>,
    patterns: &ordinary_stage::PlaceholderPatterns,
    allow_rich: bool,
    min_text_length: usize,
    mapper: &FontMapperHandle,
) -> PyResult<(
    Vec<Vec<stage::ParagraphFacts>>,
    Vec<ordinary::Job>,
    Vec<usize>,
)> {
    let mut regex = RegexService {
        service: translator.clone(),
    };
    let (jobs, page_lengths, _) = ordinary::collect(
        document,
        root,
        translator,
        &mut regex,
        patterns,
        allow_rich,
        min_text_length,
        true,
        mapper,
    )?;
    let mut pages = page_lengths
        .iter()
        .map(|length| vec![None; *length])
        .collect::<Vec<Vec<Option<stage::ParagraphFacts>>>>();
    for job in &jobs {
        pages[job.page][job.paragraph] = Some(stage::ParagraphFacts {
            debug_id: job.debug_id.clone(),
            unicode: job.unicode_present.then(|| job.text.clone()),
            layout_label: job.layout_label.clone(),
            cid: job.cid,
            pure_numeric: job.pure_numeric,
            placeholder_only: job.placeholder_only,
            box_y2: job.box_y2,
            title: job.title_paragraph,
        });
    }
    let facts = pages
        .into_iter()
        .map(|page| {
            page.into_iter()
                .map(|value| value.unwrap_or_default())
                .collect()
        })
        .collect();
    Ok((facts, jobs, page_lengths))
}

fn role_block(custom_prompt: Option<&str>, lang_out: &str) -> String {
    if let Some(prompt) = custom_prompt.filter(|value| !value.trim().is_empty()) {
        let mut value = prompt.trim().to_owned();
        if !value.contains("Follow all rules strictly.") {
            if !value.ends_with('\n') {
                value.push('\n');
            }
            value.push_str("Follow all rules strictly.");
        }
        value
    } else {
        format!(
            "You are a professional {lang_out} native translator who needs to fluently translate text into {lang_out}.\n\nFollow all rules strictly."
        )
    }
}

/// Run a complete LLM-only stage on the native document. The regular translator
/// stage remains separately callable for the fallback worker.
pub fn process(
    py: Python<'_>,
    document: &mut Document,
    config: &Bound<'_, PyAny>,
    shared_context: &Bound<'_, PyAny>,
    mapper: &FontMapperHandle,
    dispatcher_a: Py<PyAny>,
    dispatcher_b: Py<PyAny>,
    translator_service: Py<PyAny>,
    flag: CancelFlag,
    monitor: Py<PyAny>,
) -> PyResult<(usize, usize, usize)> {
    let _native_mapper = &mapper.mapper;
    let root = crate::shared_context::root_from_handle(shared_context)?;
    let initial_first = root.first_paragraph().map(|title| {
        (
            title.debug_id.unwrap_or_default(),
            ordinary::title_hint_text(title.unicode),
        )
    });
    let initial_recent = root.recent_title_paragraph().map(|title| {
        (
            title.debug_id.unwrap_or_default(),
            ordinary::title_hint_text(title.unicode),
        )
    });
    let translator = translator_service.bind(py).clone();
    translator.call_method0("probe_il_translator")?;
    let formula = ordinary::legacy_pattern(&translator, "get_formular_placeholder")?;
    let left = ordinary::legacy_pattern(&translator, "get_rich_text_left_placeholder")?;
    let right = ordinary::legacy_pattern(&translator, "get_rich_text_right_placeholder")?;
    let patterns = ordinary_stage::PlaceholderPatterns {
        formula,
        rich_left: left,
        rich_right: right,
    };
    let supports_llm = translator
        .call_method0("probe_il_translator")?
        .extract::<bool>()?;
    if !supports_llm {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "LLM translator not supported",
        ));
    }
    let disable_rich = config
        .getattr("disable_rich_text_translate")?
        .extract::<bool>()?;
    let min_length = config.getattr("min_text_length")?.extract::<usize>()?;
    let (facts, mut jobs, page_lengths) = pages_and_jobs(
        document,
        &root,
        &translator,
        &patterns,
        !disable_rich,
        min_length,
        mapper,
    )?;
    let first_title = jobs.iter().find(|job| job.title_paragraph).map(|job| {
        (
            job.debug_id.clone().unwrap_or_default(),
            ordinary::title_hint_text(job.unicode_present.then(|| job.text.clone())),
        )
    });
    let cross_title = initial_first.or_else(|| first_title.clone());
    let cross_recent = initial_recent.or(first_title);
    let plans = stage::plan_batches(&facts, min_length);
    let cross_page_pairs = plans
        .iter()
        .filter(|plan| plan.kind == stage::GroupKind::CrossPage)
        .filter_map(|plan| match plan.paragraphs.as_slice() {
            [left, right] => Some([(left.page, left.index), (right.page, right.index)]),
            _ => None,
        })
        .collect::<Vec<_>>();
    let mut regex = RegexService {
        service: translator.clone(),
    };
    ordinary::rebuild_cross_page_inputs(
        document,
        &mut jobs,
        &cross_page_pairs,
        &translator,
        &mut regex,
        &patterns,
        !disable_rich,
        mapper,
    )?;
    let workers = config
        .getattr("pool_max_workers")?
        .extract::<usize>()?
        .max(1);
    let lang = config.getattr("lang_out")?.extract::<String>()?;
    let custom = config
        .getattr("custom_system_prompt")
        .ok()
        .and_then(|v| v.extract::<String>().ok());
    let formula_hints = config
        .getattr("add_formula_placehold_hint")
        .and_then(|v| v.extract::<bool>())
        .unwrap_or(false);
    let auto_extract = config
        .getattr("auto_extract_glossary")
        .and_then(|v| v.extract::<bool>())
        .unwrap_or(false);
    let glossaries = root.get_glossaries_for_translation(auto_extract);
    let roles = role_block(custom.as_deref(), &lang);
    let mut by_id = HashMap::new();
    for job in jobs {
        by_id.insert((job.page, job.paragraph), job);
    }
    let mut works = Vec::new();
    let mut tracking = stage::DocumentTracking::new(&page_lengths);
    let mut kind_offsets = HashMap::<stage::GroupKind, usize>::new();
    for batch in plans {
        let work_jobs = batch
            .paragraphs
            .iter()
            .filter_map(|id| by_id.get(&(id.page, id.index)).cloned())
            .collect::<Vec<_>>();
        let mut inputs = Vec::new();
        let mut json_items = Vec::new();
        let mut skipped_count = 0;
        let mut prepare_error = None;
        for job in &work_jobs {
            if let Some(error) = &job.missing_font_style {
                prepare_error = Some(error.key_error_message());
                break;
            }
            let Some(input) = &job.input else {
                skipped_count += 1;
                continue;
            };
            if input.text.chars().count() < min_length {
                skipped_count += 1;
                continue;
            }
            let label = job.layout_label.clone();
            let hint = formula_hints
                .then(|| ordinary::formula_hint(input))
                .flatten();
            inputs.push(input.text.clone());
            json_items.push((input.text.clone(), label, hint));
        }
        let serialized = stage::batch_json(&json_items, formula_hints);
        let matching_text = json_items
            .iter()
            .map(|(input, _, _)| input.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        let first = work_jobs.first();
        let last = work_jobs.last();
        let (title_value, local_title_value) = if batch.kind == stage::GroupKind::Page {
            (
                first.and_then(|job| job.title.as_ref()),
                last.and_then(|job| job.local_title.as_ref()),
            )
        } else {
            (cross_title.as_ref(), cross_recent.as_ref())
        };
        let title = title_value.map(|(id, text)| (id.as_str(), text.as_str()));
        let local_title = local_title_value.map(|(id, text)| (id.as_str(), text.as_str()));
        let prompt = stage::render_batch_prompt(
            &roles,
            &lang,
            &serialized,
            &matching_text,
            title,
            local_title,
            &glossaries,
        );
        let title_context = title_value.cloned();
        let local_title_context = local_title_value.cloned();
        let group = if batch.kind == stage::GroupKind::Page {
            batch.paragraphs.first().map_or(0, |id| id.page)
        } else {
            let offset = kind_offsets.entry(batch.kind).or_default();
            let result = *offset;
            *offset += 1;
            result
        };
        if batch.kind != stage::GroupKind::Page {
            tracking.new_group(batch.kind, batch.paragraphs.len());
        }
        let admitted = Arc::new(AtomicBool::new(false));
        works.push(Work {
            batch,
            jobs: work_jobs,
            prompt,
            title_context,
            local_title_context,
            inputs,
            prepare_error,
            admitted,
            tracker_group: group,
            skipped_count,
        });
        let _ = group;
    }
    let mut schedule = Vec::new();
    schedule.push(ScheduledWork::Checkpoint); // process_cross_page_paragraph entry
    schedule.extend(
        works
            .iter()
            .filter(|work| work.batch.kind == stage::GroupKind::CrossPage)
            .cloned()
            .map(ScheduledWork::Batch),
    );
    for page in 0..page_lengths.len() {
        schedule.push(ScheduledWork::Checkpoint); // process_cross_column_paragraph entry
        schedule.extend(
            works
                .iter()
                .filter(|work| {
                    work.batch.kind == stage::GroupKind::CrossColumn
                        && work
                            .batch
                            .paragraphs
                            .first()
                            .is_some_and(|id| id.page == page)
                })
                .cloned()
                .map(ScheduledWork::Batch),
        );
    }
    for page in 0..page_lengths.len() {
        schedule.push(ScheduledWork::Checkpoint); // process_page entry
        schedule.extend(
            works
                .iter()
                .filter(|work| {
                    work.batch.kind == stage::GroupKind::Page
                        && work
                            .batch
                            .paragraphs
                            .first()
                            .is_some_and(|id| id.page == page)
                })
                .cloned()
                .map(ScheduledWork::Batch),
        );
    }
    let planned = works
        .iter()
        .flat_map(|work| work.batch.paragraphs.iter().copied())
        .collect::<std::collections::HashSet<_>>();
    let pre_skipped = facts
        .iter()
        .enumerate()
        .flat_map(|(page, values)| {
            values
                .iter()
                .enumerate()
                .map(move |(index, value)| (page, index, value))
        })
        .filter(|(page, index, facts)| {
            facts.debug_id.is_some()
                && facts.unicode.is_some()
                && !planned.contains(&stage::ParagraphId {
                    page: *page,
                    index: *index,
                })
        })
        .count();
    let tracking_path = config
        .call_method1("get_working_file_path", ("translate_tracking.json",))?
        .str()?
        .to_str()?
        .to_owned();
    let write_tracking =
        config.getattr("debug")?.extract::<bool>()? || !config.getattr("working_dir")?.is_none();
    let recorder_a = Recorder::for_dispatcher(py, dispatcher_a.bind(py));
    let recorder_b = Recorder::for_dispatcher(py, dispatcher_b.bind(py));
    let a_service = PythonService {
        dispatcher: dispatcher_a,
        recorder: recorder_a,
    };
    let b_service = PythonService {
        dispatcher: dispatcher_b,
        recorder: recorder_b,
    };
    let pool_a = Arc::new(TranslatorPoolPair::new(
        a_service,
        stage::STAGE_NAME,
        workers,
        root.clone(),
    ));
    let pool_b = Arc::new(TranslatorPoolPair::new(
        b_service,
        stage::STAGE_NAME,
        workers,
        root,
    ));
    let translator_for_regex = translator_service.clone_ref(py);
    let counters = (
        Arc::new(AtomicUsize::new(0)),
        Arc::new(AtomicUsize::new(0)),
        Arc::new(AtomicUsize::new(0)),
    );
    let counters_out = counters.clone();
    let result = callbacks::run(py, workers, move |runtime, progress_run| {
        let progress = progress_run
            .start(
                monitor,
                stage::STAGE_NAME,
                stage::stage_total(&facts) as u64,
                Duration::from_millis(30),
            )
            .map_err(ProgressError::into_python)?;
        if pre_skipped > 0 {
            progress
                .report(pre_skipped as u64, None)
                .map_err(|error| error.into_python())?;
        }
        let mut result_tracking = tracking;
        let mut outputs = Vec::<Output>::new();
        let run = runtime.block_on(async {
            // A jobs are all submitted and joined before B is drained; fallback
            // requests are submitted to their independent equal-capacity pool.
            let mut a_tasks = tokio::task::JoinSet::new();
            let mut b_tasks = tokio::task::JoinSet::new();
            let mut by_batch = Vec::new();
            for scheduled in schedule {
                let work = match scheduled {
                    ScheduledWork::Checkpoint => {
                        flag.checkpoint()?;
                        continue;
                    }
                    ScheduledWork::Batch(work) => work,
                };
                if work.inputs.is_empty() {
                    if work.skipped_count > 0 {
                        progress
                            .report(work.skipped_count as u64, None)
                            .map_err(|e| e.into_python())?;
                    }
                    continue;
                }
                if let Some(error) = work.prepare_error.clone() {
                    work.admitted.store(true, Ordering::Release);
                    let prompt = work.prompt.clone();
                    let inputs = work.inputs.clone();
                    let id = by_batch.len();
                    a_tasks
                        .spawn(async move { Ok((id, prompt, inputs, Some(Err((error, None))))) });
                    by_batch.push(work);
                    continue;
                }
                let admitted = Arc::clone(&work.admitted);
                let req_flag = flag.clone();
                let (args, kwargs) = Python::attach(|py| {
                    request_args(py, &work.prompt, work.batch.token_count, true)
                })?;
                let pool = Arc::clone(&pool_a);
                let prompt = work.prompt.clone();
                let batch = work.batch.clone();
                let inputs = work.inputs.clone();
                let id = by_batch.len();
                a_tasks.spawn(async move {
                    let response = pool
                        .request(
                            "llm_translate",
                            format!("llm-only:{}", batch.id),
                            args,
                            Some(kwargs),
                            1_048_576 - batch.token_count as i64,
                            move || {
                                if req_flag.is_cancelled() {
                                    return false;
                                }
                                admitted.store(true, Ordering::Release);
                                true
                            },
                        )
                        .await;
                    let value = match response {
                        Ok(Some(value)) => {
                            let raw = Python::attach(|py| {
                                value
                                    .bind(py)
                                    .str()
                                    .map(|value| value.to_string_lossy().into_owned())
                            });
                            match raw {
                                Err(error) => {
                                    let message = Python::attach(|py| {
                                        error
                                            .value(py)
                                            .str()
                                            .map(|message| message.to_string_lossy().into_owned())
                                    })
                                    .unwrap_or_else(|_| {
                                        "translator response conversion failed".into()
                                    });
                                    Err((message, None))
                                }
                                Ok(raw) => {
                                    match Python::attach(|py| -> PyResult<_> {
                                        value
                                            .bind(py)
                                            .call_method0("strip")?
                                            .extract::<String>()
                                            .map_err(Into::into)
                                    }) {
                                        Ok(value) => Ok((value, Some(raw))),
                                        Err(error) => {
                                            let message = Python::attach(|py| {
                                                error.value(py).str().map(|message| {
                                                    message.to_string_lossy().into_owned()
                                                })
                                            })
                                            .unwrap_or_else(|_| {
                                                "translator response strip failed".into()
                                            });
                                            Err((message, Some(raw)))
                                        }
                                    }
                                }
                            }
                        }
                        Ok(None) => return Ok((id, prompt, inputs, None)),
                        Err(crate::callbacks::CallbackError::Python(error)) => {
                            let message = Python::attach(|py| {
                                error
                                    .value(py)
                                    .str()
                                    .map(|message| message.to_string_lossy().into_owned())
                            })
                            .unwrap_or_else(|_| "translator request failed".into());
                            return Ok((id, prompt, inputs, Some(Err((message, None)))));
                        }
                        Err(crate::callbacks::CallbackError::Bridge(error)) => {
                            return Err(py_error(error));
                        }
                    };
                    Ok((id, prompt, inputs, Some(value)))
                });
                by_batch.push(work);
            }
            while let Some(joined) = a_tasks.join_next().await {
                let (batch_index, _prompt, inputs, response) =
                    joined.map_err(|e| py_error(e))??;
                let work = &by_batch[batch_index];
                if !work.admitted.load(Ordering::Acquire) {
                    continue;
                }
                let (response, tracker_output, request_error) = match response {
                    Some(Ok((value, tracker_output))) => (value, tracker_output, None),
                    Some(Err((error, tracker_output))) => {
                        (String::new(), tracker_output, Some(error))
                    }
                    None => (String::new(), None, None),
                };
                let parsed = stage::interpret_response(&response, &inputs, false);
                let outer_error = request_error.is_some() || parsed.error.is_some();
                counters_out.0.fetch_add(parsed.ok_count, Ordering::Relaxed);
                counters_out.1.fetch_add(
                    parsed.fallback_count + if outer_error { inputs.len() } else { 0 },
                    Ordering::Relaxed,
                );
                counters_out.2.fetch_add(
                    parsed.processed + if outer_error { inputs.len() } else { 0 },
                    Ordering::Relaxed,
                );
                let group_index = work.tracker_group;
                let mut item_outputs = Vec::new();
                for item in parsed.items {
                    item_outputs.push((item.id, item.output));
                }
                let llm_error = request_error.or(parsed.error);
                let fallback_errors = parsed.fallback_errors;
                let fallback_indices = if llm_error.is_some() {
                    (0..inputs.len()).collect()
                } else {
                    parsed.fallback_indices
                };
                let mut input_position = 0usize;
                for (tracker_index, id) in work.batch.paragraphs.iter().enumerate() {
                    let Some(job) = work
                        .jobs
                        .iter()
                        .find(|job| job.page == id.page && job.paragraph == id.index)
                        .cloned()
                    else {
                        continue;
                    };
                    if job.missing_font_style.is_some() {
                        break;
                    }
                    let Some(input) = job.input.clone() else {
                        progress.report(1, None).map_err(|e| e.into_python())?;
                        continue;
                    };
                    if input.text.chars().count() < min_length {
                        let slot = match work.batch.kind {
                            stage::GroupKind::Page => {
                                &mut result_tracking.page[id.page].paragraph[id.index]
                            }
                            stage::GroupKind::CrossPage => {
                                &mut result_tracking.cross_page[work.tracker_group].paragraph
                                    [tracker_index]
                            }
                            stage::GroupKind::CrossColumn => {
                                &mut result_tracking.cross_column[work.tracker_group].paragraph
                                    [tracker_index]
                            }
                        };
                        slot.pdf_unicode = Some(job.text.clone());
                        slot.input = Some(input.text.clone());
                        slot.original_placeholders = input.original_tokens.clone();
                        slot.placeholders = ordinary::tracking_placeholders(&input);
                        progress.report(1, None).map_err(|e| e.into_python())?;
                        continue;
                    }
                    let pos = input_position;
                    input_position += 1;
                    let (tracker_kind, tracker_group) = if work.batch.kind == stage::GroupKind::Page
                    {
                        (work.batch.kind, id.page)
                    } else {
                        (work.batch.kind, group_index)
                    };
                    let slot = match tracker_kind {
                        stage::GroupKind::Page => {
                            &mut result_tracking.page[tracker_group].paragraph[id.index]
                        }
                        stage::GroupKind::CrossPage => {
                            &mut result_tracking.cross_page[tracker_group].paragraph[tracker_index]
                        }
                        stage::GroupKind::CrossColumn => {
                            &mut result_tracking.cross_column[tracker_group].paragraph
                                [tracker_index]
                        }
                    };
                    if slot.input.is_none() {
                        slot.pdf_unicode = Some(job.text.clone());
                        slot.input = Some(input.text.clone());
                        slot.original_placeholders = input.original_tokens.clone();
                        slot.placeholders = ordinary::tracking_placeholders(&input);
                        slot.multi_paragraph_id = Some(work.batch.id);
                        slot.multi_paragraph_index = Some(pos);
                        let mut llm_tracker = ordinary_stage::LlmTracking {
                            input: if work.prepare_error.is_some() {
                                String::new()
                            } else {
                                work.prompt.clone()
                            },
                            output: tracker_output.clone().unwrap_or_default(),
                            ..Default::default()
                        };
                        if let Some(error) = &llm_error {
                            llm_tracker.has_error = true;
                            llm_tracker.error_message =
                                format!("Error {error} during translation. try fallback");
                            llm_tracker.fallback_to_translate = true;
                        } else if let Some((_, error, full_match)) =
                            fallback_errors.iter().find(|(index, _, _)| *index == pos)
                        {
                            llm_tracker.has_error = true;
                            llm_tracker.error_message = error.clone();
                            llm_tracker.placeholder_full_match = *full_match;
                            llm_tracker.fallback_to_translate = true;
                        } else if fallback_indices.contains(&pos) {
                            llm_tracker.fallback_to_translate = true;
                        }
                        slot.llm_translate_trackers.push(llm_tracker);
                    }
                    let matching_outputs = item_outputs
                        .iter()
                        .filter(|(index, _)| *index == pos)
                        .map(|(_, text)| text.clone())
                        .collect::<Vec<_>>();
                    for text in matching_outputs {
                        outputs.push(Output {
                            job: job.clone(),
                            input: input.clone(),
                            text,
                            fallback: false,
                            fallback_prompt: None,
                            tracker_kind,
                            tracker_group,
                            tracker_index,
                        });
                        progress.report(1, None).map_err(|e| e.into_python())?;
                    }
                    let fallback_times = fallback_indices
                        .iter()
                        .filter(|index| **index == pos)
                        .count();
                    if fallback_times > 0 {
                        // Python's LLM-only tracker keeps paragraph.unicode for the
                        // direct request, then its ordinary fallback path rebuilds
                        // unicode from the paragraph compositions before tracking.
                        slot.pdf_unicode = Some(job.tracker_unicode.clone());
                        if let Some(fallback_input) = &job.fallback_input {
                            slot.input = Some(fallback_input.text.clone());
                            slot.original_placeholders = fallback_input.original_tokens.clone();
                            slot.placeholders = ordinary::tracking_placeholders(fallback_input);
                        }
                    }
                    for _ in 0..fallback_times {
                        let Some(request) =
                            fallback_request(&job, formula_hints, llm_error.is_some())
                        else {
                            progress.report(1, None).map_err(|e| e.into_python())?;
                            continue;
                        };
                        let text = request.text.clone();
                        if text.chars().count() < min_length {
                            progress.report(1, None).map_err(|e| e.into_python())?;
                            continue;
                        }
                        let (fallback_title, fallback_local) = (
                            work.title_context.as_ref(),
                            work.local_title_context.as_ref(),
                        );
                        let context = ordinary_stage::context_block(
                            fallback_title.map(|(id, t)| (id.as_str(), t.as_str())),
                            fallback_local.map(|(id, t)| (id.as_str(), t.as_str())),
                            request.formula_hint.as_deref(),
                        );
                        let prompt = ordinary_stage::render_prompt(
                            &roles,
                            &lang,
                            &ordinary_stage::glossary_block(&text, &glossaries),
                            &context,
                            &text,
                        );
                        let fallback_token_count = request.paragraph_token_count;
                        let (args, kwargs) = Python::attach(|py| {
                            request_args(py, &prompt, fallback_token_count, false)
                        })?;
                        let pool = Arc::clone(&pool_b);
                        let flag = flag.clone();
                        let p = progress.clone();
                        let job_id = format!("fallback:{}:{}", job.page, job.paragraph);
                        let job_for_task = job.clone();
                        let input_for_task = request.input;
                        let fallback_prompt = prompt.clone();
                        let indices = (tracker_kind, tracker_group, tracker_index);
                        b_tasks.spawn(async move {
                            let response = pool
                                .request(
                                    "llm_translate",
                                    job_id,
                                    args,
                                    Some(kwargs),
                                    1_048_576 - fallback_token_count as i64,
                                    move || !flag.is_cancelled(),
                                )
                                .await;
                            match response {
                                Ok(Some(value)) => {
                                    let text = Python::attach(|py| {
                                        value.bind(py).extract::<String>().map_err(|_| {
                                            PyTypeError::new_err(
                                                "fallback llm_translate response must be a string",
                                            )
                                        })
                                    })?;
                                    p.report(1, None).map_err(|e| e.into_python())?;
                                    Ok(Some(Output {
                                        job: job_for_task,
                                        input: input_for_task,
                                        text,
                                        fallback: true,
                                        fallback_prompt: Some(fallback_prompt),
                                        tracker_kind: indices.0,
                                        tracker_group: indices.1,
                                        tracker_index: indices.2,
                                    }))
                                }
                                Ok(None) => Ok(None),
                                Err(crate::callbacks::CallbackError::Python(_)) => Ok(None),
                                Err(crate::callbacks::CallbackError::Bridge(error)) => {
                                    Err(py_error(error))
                                }
                            }
                        });
                    }
                }
            }
            while let Some(joined) = b_tasks.join_next().await {
                if let Some(output) = joined.map_err(|e| py_error(e))?? {
                    outputs.push(output);
                }
            }
            Ok::<_, PyErr>(outputs)
        });
        let outputs = match run {
            Ok(value) => value,
            Err(error) => {
                let _ = progress.end(None);
                return Err(error);
            }
        };
        let hallucination_ids = Python::attach(|py| -> PyResult<Vec<u64>> {
            let mut regex = RegexService {
                service: translator_for_regex.bind(py).clone(),
            };
            [
                patterns.formula.as_str(),
                patterns.rich_left.as_str(),
                patterns.rich_right.as_str(),
            ]
            .into_iter()
            .map(|pattern| regex.compile(pattern, true))
            .collect()
        })?;
        for output in outputs {
            let (text, full_match, removed) = Python::attach(|py| {
                ordinary::apply_llm_output(
                    document,
                    output.job.page,
                    output.job.paragraph,
                    &output.input,
                    &output.text,
                    translator_for_regex.bind(py),
                    &hallucination_ids,
                )
            })?;
            let slot = match output.tracker_kind {
                stage::GroupKind::Page => {
                    &mut result_tracking.page[output.tracker_group].paragraph[output.job.paragraph]
                }
                stage::GroupKind::CrossPage => {
                    &mut result_tracking.cross_page[output.tracker_group].paragraph
                        [output.tracker_index]
                }
                stage::GroupKind::CrossColumn => {
                    &mut result_tracking.cross_column[output.tracker_group].paragraph
                        [output.tracker_index]
                }
            };
            slot.output = Some(text);
            slot.removed_hallucinated_placeholders = removed;
            if output.fallback {
                slot.llm_translate_trackers
                    .push(ordinary_stage::LlmTracking {
                        input: output.fallback_prompt.unwrap_or_default(),
                        output: output.text.clone(),
                        placeholder_full_match: full_match || output.text == output.input.text,
                        ..Default::default()
                    });
            } else if let Some(last) = slot.llm_translate_trackers.last_mut() {
                last.placeholder_full_match = full_match || output.text == output.input.text;
            }
        }
        progress.end(None).map_err(ProgressError::into_python)?;
        if write_tracking {
            if let Some(parent) = std::path::Path::new(&tracking_path).parent() {
                std::fs::create_dir_all(parent).map_err(py_error)?;
            }
            std::fs::write(&tracking_path, result_tracking.to_json()).map_err(py_error)?;
        }
        Ok::<_, PyErr>(())
    });
    result?;
    Ok((
        counters.0.load(Ordering::Relaxed),
        counters.1.load(Ordering::Relaxed),
        counters.2.load(Ordering::Relaxed),
    ))
}

#[cfg(test)]
mod llm_only_translation_tests;
