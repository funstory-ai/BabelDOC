//! Serial write orchestration. CPU content/font calls remain in writer_core;
//! only orchestration boundaries submit E2 file requests or report progress.
use crate::callbacks::{
    self, CallbackService, CancelFlag, ProgressError, ProgressRun, PythonProgress, PythonService,
    Recorder, Request,
};
use crate::font_mapper::FontMapperHandle;
use babeldoc_ir_core::{Document, ReadOnlyView, WriterView};
use babeldoc_p2_stages::ProgressReporter;
use babeldoc_writer_core as core;
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use serde::Deserialize;
use serde_json::{Value, json};
use std::borrow::Cow;
use std::time::Duration;

#[derive(Deserialize)]
pub struct WriteConfig {
    pub source: String,
    pub translated: String,
    pub debug_original: String,
    pub mono: Option<String>,
    pub dual: Option<String>,
    #[serde(default)]
    pub debug: bool,
    #[serde(default)]
    pub ocr_workaround: bool,
    #[serde(default)]
    pub skip_clean: bool,
    #[serde(default)]
    pub skip_form_render: bool,
    #[serde(default)]
    pub skip_curve_render: bool,
    #[serde(default)]
    pub only_include_translated_page: bool,
    #[serde(default)]
    pub keep_pages: Vec<i64>,
    #[serde(default)]
    pub use_alternating_pages_dual: bool,
    #[serde(default)]
    pub dual_translate_first: bool,
}
#[derive(Deserialize)]
pub struct AssetSpec {
    id: String,
    metadata: Metadata,
}
#[derive(Deserialize)]
struct Metadata {
    font_name: String,
    ascent: f64,
    descent: f64,
    encoding_length: i64,
}

/// The file layer is replaceable for fault injection; it never owns Rust IR.
pub trait FileLayer {
    fn call<A>(&self, stage: &str, method: &str, args: A) -> PyResult<Py<PyAny>>
    where
        A: for<'py> IntoPyObject<'py, Target = PyTuple>;
}

struct E2FileLayer<'a> {
    runtime: &'a tokio::runtime::Runtime,
    service: PythonService,
}
impl FileLayer for E2FileLayer<'_> {
    fn call<A>(&self, stage: &str, method: &str, args: A) -> PyResult<Py<PyAny>>
    where
        A: for<'py> IntoPyObject<'py, Target = PyTuple>,
    {
        // This is an I/O boundary, outside all content/font CPU functions.
        let args = Python::attach(|py| callbacks::into_py_tuple(py, args))?;
        self.runtime
            .block_on(self.service.request(Request {
                stage: stage.into(),
                job: method.into(),
                service: "writer".into(),
                method: method.into(),
                args,
                kwargs: None,
            }))
            .map_err(|e| e.into_python())
    }
}
fn ordinary(error: &PyErr) -> bool {
    Python::attach(|py| error.is_instance_of::<pyo3::exceptions::PyException>(py))
}
fn core_error(error: impl std::fmt::Display) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(error.to_string())
}
fn stage<T>(
    run: &ProgressRun,
    monitor: &Py<PyAny>,
    name: &str,
    total: u64,
    body: impl FnOnce(&PythonProgress) -> PyResult<T>,
) -> PyResult<T> {
    let progress = run
        .start(
            Python::attach(|py| monitor.clone_ref(py)),
            name,
            total,
            Duration::from_millis(100),
        )
        .map_err(ProgressError::into_python)?;
    match body(&progress) {
        Ok(value) => {
            progress.end(None).map_err(ProgressError::into_python)?;
            Ok(value)
        }
        Err(error) => {
            progress
                .end(Some(error.into()))
                .map_err(ProgressError::into_python)?;
            unreachable!("stage end propagates primary error")
        }
    }
}
fn fonts(
    ir: &mut Document,
    pdf: &mut core::PdfDocument,
    page_objects: &core::PageObjectMap,
    assets: &[core::FontAsset<'_>],
    progress: &ProgressRun,
    monitor: &Py<PyAny>,
) -> PyResult<()> {
    let pages = ir
        .view::<WriterView>()
        .document()
        .page_len(&ir.view::<WriterView>())
        .map_err(core_error)?;
    let used = core::used_font_ids(&ir.view::<WriterView>()).map_err(core_error)?;
    let total = pdf.max_id as u64
        + 2 * assets.iter().filter(|a| used.contains(a.id)).count() as u64
        + pages as u64;
    stage(progress, monitor, "Add Fonts", total, |p| {
        let mut callback_error = None;
        let result = core::add_fonts_with_used_ids_from_page_objects(
            pdf,
            &mut ir.view::<WriterView>(),
            assets,
            &used,
            page_objects,
            |n| {
                p.report(n, None).map_err(|e| {
                    callback_error = Some(e.into_python());
                    core::WriterError::Invalid("progress callback failed".into())
                })
            },
        );
        if let Some(error) = callback_error {
            return Err(error);
        }
        result.map_err(core_error)?;
        Ok(())
    })
}
fn drawing(
    ir: &mut Document,
    pdf: &mut core::PdfDocument,
    page_objects: &core::PageObjectMap,
    cfg: &WriteConfig,
    checked: bool,
    flag: &CancelFlag,
    progress: &ProgressRun,
    monitor: &Py<PyAny>,
) -> PyResult<()> {
    let v = ir.view::<ReadOnlyView>();
    let page_count = v.document().page_len(&v).map_err(core_error)?;
    stage(
        progress,
        monitor,
        "Generate drawing instructions",
        page_count as u64,
        |p| {
            for page in v.document().page_iter(&v).map_err(core_error)? {
                let page = page.map_err(core_error)?;
                flag.checkpoint()?;
                let mut opts = core::BuildOptions {
                    check_font_exists: checked,
                    debug: cfg.debug,
                    ocr_workaround: cfg.ocr_workaround,
                    skip_form_render: cfg.skip_form_render,
                    skip_curve_render: cfg.skip_curve_render,
                    ..Default::default()
                };
                if checked {
                    let forms = page
                        .pdf_xobject_iter(&v)
                        .map_err(core_error)?
                        .map(|x| {
                            let x = x.map_err(core_error)?;
                            Ok(core::FormOutput {
                                xobj_id: x.xobj_id(&v).map_err(core_error)?,
                                xref_id: x
                                    .xref_id(&v)
                                    .map_err(core_error)?
                                    .ok_or_else(|| core_error("missing xref"))?,
                                stream: vec![],
                            })
                        })
                        .collect::<PyResult<Vec<_>>>()?;
                    let empty = core::PageOutput {
                        page_number: page
                            .page_number(&v)
                            .map_err(core_error)?
                            .ok_or_else(|| core_error("missing page number"))?,
                        stream: vec![],
                        forms,
                        counts: Default::default(),
                    };
                    opts.available = Some(
                        core::available_fonts_with_page_objects(pdf, &empty, page_objects)
                            .map_err(core_error)?,
                    );
                }
                let output = core::build_page_output(&v, &page, &opts).map_err(core_error)?;
                core::apply_page_output_with_page_objects(pdf, &output, page_objects)
                    .map_err(core_error)?;
                p.report(1, None).map_err(ProgressError::into_python)?;
            }
            Ok(())
        },
    )
}
fn debug<F: FileLayer>(
    ir: &mut Document,
    page_objects: Option<&core::PageObjectMap>,
    cfg: &WriteConfig,
    assets: &[core::FontAsset<'_>],
    flag: &CancelFlag,
    progress: &ProgressRun,
    monitor: &Py<PyAny>,
    file: &F,
) -> PyResult<()> {
    let mut pdf = core::PdfDocument::load(&cfg.source).map_err(core_error)?;
    let page_objects = page_objects_for(page_objects, &pdf);
    fonts(ir, &mut pdf, &page_objects, assets, progress, monitor)?;
    let v = ir.view::<ReadOnlyView>();
    for page in v.document().page_iter(&v).map_err(core_error)? {
        let page = page.map_err(core_error)?;
        flag.checkpoint()?;
        let number = page
            .page_number(&v)
            .map_err(core_error)?
            .ok_or_else(|| core_error("missing page number"))?;
        let (id, base) = core::first_content_stream_with_page_objects(&pdf, number, &page_objects)
            .map_err(core_error)?;
        let available = core::AvailableFonts {
            page: core::available_font_names(
                &pdf,
                core::page_object_with_map(&pdf, &page_objects, number).map_err(core_error)?,
            ),
            ..Default::default()
        };
        let stream = core::build_debug_stream(&v, &page, &base, available).map_err(core_error)?;
        core::replace_stream(&mut pdf, id, stream).map_err(core_error)?;
    }
    flag.checkpoint()?;
    pdf.save(&cfg.debug_original).map_err(core_error)?;
    file.call("Save PDF", "open_original", (cfg.debug_original.clone(),))?;
    if !cfg.skip_clean {
        file.call("Save PDF", "subset_debug", ())?;
    }
    Ok(())
}
fn page_objects_for<'a>(
    page_objects: Option<&'a core::PageObjectMap>,
    pdf: &core::PdfDocument,
) -> Cow<'a, core::PageObjectMap> {
    match page_objects {
        Some(page_objects) => Cow::Borrowed(page_objects),
        None => Cow::Owned(core::legacy_page_objects(pdf)),
    }
}
fn attempt<F: FileLayer>(
    ir: &mut Document,
    page_objects: Option<&core::PageObjectMap>,
    cfg: &WriteConfig,
    assets: &[core::FontAsset<'_>],
    checked: bool,
    flag: &CancelFlag,
    progress: &ProgressRun,
    monitor: &Py<PyAny>,
    file: &F,
) -> PyResult<Py<PyAny>> {
    let mut pdf = core::PdfDocument::load(&cfg.source).map_err(core_error)?;
    let page_objects = page_objects_for(page_objects, &pdf);
    fonts(ir, &mut pdf, &page_objects, assets, progress, monitor)?;
    drawing(
        ir,
        &mut pdf,
        &page_objects,
        cfg,
        checked,
        flag,
        progress,
        monitor,
    )?;
    flag.checkpoint()?;
    pdf.save(&cfg.translated).map_err(core_error)?;
    file.call("write", "open_translated", (cfg.translated.clone(),))?;
    flag.checkpoint()?;
    let garbage = if cfg.ocr_workaround { 4 } else { 1 };
    stage(progress, monitor, "Subset font", 1, |p| {
        if !cfg.skip_clean {
            file.call("Subset font", "subset_mono", ())?;
        }
        p.report(1, None).map_err(ProgressError::into_python)
    })?;
    file.call("write", "restore_media_box", ())?;
    if cfg.only_include_translated_page {
        file.call("write", "keep_translated", (cfg.keep_pages.clone(),))?;
    }
    stage(progress, monitor, "Save PDF", 2, |p| {
        if let Some(path) = &cfg.mono {
            if cfg.debug {
                flag.checkpoint()?;
                file.call(
                    "Save PDF",
                    "save_decompressed_mono",
                    (format!("{path}.decompressed.pdf"),),
                )?;
            }
            flag.checkpoint()?;
            file.call(
                "Save PDF",
                "save_mono",
                (path.clone(), garbage, !cfg.skip_clean),
            )?;
        }
        p.report(1, None).map_err(ProgressError::into_python)?;
        if let Some(path) = &cfg.dual {
            flag.checkpoint()?;
            // Opening the resource lies outside the ordinary debug fallback.
            file.call("Save PDF", "open_original", (cfg.source.clone(),))?;
            if cfg.debug {
                flag.checkpoint()?;
                if let Err(error) = debug(
                    ir,
                    Some(page_objects.as_ref()),
                    cfg,
                    assets,
                    flag,
                    progress,
                    monitor,
                    file,
                ) {
                    if !ordinary(&error) {
                        return Err(error);
                    }
                    log::warn!("Failed to write debug info to dual PDF: {error}");
                    progress.acknowledge_ended_errors();
                    // Ex8: an ordinary debug failure retains an unmodified source.
                    file.call("Save PDF", "open_original", (cfg.source.clone(),))?;
                }
            }
            if cfg.only_include_translated_page {
                file.call("Save PDF", "keep_original", (cfg.keep_pages.clone(),))?;
            }
            if cfg.use_alternating_pages_dual {
                file.call("Save PDF", "make_alternating", (cfg.dual_translate_first,))?;
            } else {
                file.call(
                    "Save PDF",
                    "make_side_by_side",
                    (cfg.dual_translate_first, path.clone()),
                )?;
            }
            file.call(
                "Save PDF",
                "save_dual",
                (path.clone(), garbage, !cfg.skip_clean),
            )?;
            if cfg.debug {
                flag.checkpoint()?;
                file.call(
                    "Save PDF",
                    "save_decompressed_dual",
                    (format!("{path}.decompressed.pdf"),),
                )?;
            }
        }
        p.report(1, None).map_err(ProgressError::into_python)
    })?;
    file.call("write", "glossary_hook", ())
}

/// Compatibility entry for `p2_stage_diff::writer_probe`; its frozen probe is
/// the only caller allowed to discover pages through lopdf.
#[doc(hidden)]
pub fn process(
    py: Python<'_>,
    ir: &mut Document,
    cfg: WriteConfig,
    mapper: &FontMapperHandle,
    specs: Vec<AssetSpec>,
    dispatcher: Py<PyAny>,
    flag: CancelFlag,
    monitor: Py<PyAny>,
) -> PyResult<Value> {
    process_impl(py, ir, None, cfg, mapper, specs, dispatcher, flag, monitor)
}

pub fn process_with_page_objects(
    py: Python<'_>,
    ir: &mut Document,
    page_objects: &core::PageObjectMap,
    cfg: WriteConfig,
    mapper: &FontMapperHandle,
    specs: Vec<AssetSpec>,
    dispatcher: Py<PyAny>,
    flag: CancelFlag,
    monitor: Py<PyAny>,
) -> PyResult<Value> {
    process_impl(
        py,
        ir,
        Some(page_objects),
        cfg,
        mapper,
        specs,
        dispatcher,
        flag,
        monitor,
    )
}

fn validate_page_objects(page_objects: Option<&core::PageObjectMap>) -> PyResult<()> {
    if page_objects.is_some_and(core::PageObjectMap::is_empty) {
        return Err(core_error("page object map is empty"));
    }
    Ok(())
}

fn process_impl(
    py: Python<'_>,
    ir: &mut Document,
    page_objects: Option<&core::PageObjectMap>,
    cfg: WriteConfig,
    mapper: &FontMapperHandle,
    specs: Vec<AssetSpec>,
    dispatcher: Py<PyAny>,
    flag: CancelFlag,
    monitor: Py<PyAny>,
) -> PyResult<Value> {
    validate_page_objects(page_objects)?;
    // Borrow faces from the same immutable mapper used by the layout handle.
    let faces = specs
        .iter()
        .map(|a| {
            mapper
                .mapper
                .font_by_id(&a.id)
                .ok_or_else(|| core_error(format!("mapper has no font {}", a.id)))
        })
        .collect::<PyResult<Vec<_>>>()?;
    let metadata = specs
        .iter()
        .map(|a| core::FontMetadata {
            font_name: a.metadata.font_name.clone(),
            ascent: a.metadata.ascent,
            descent: a.metadata.descent,
            encoding_length: a.metadata.encoding_length,
        })
        .collect::<Vec<_>>();
    let recorder = Recorder::for_dispatcher(py, dispatcher.bind(py));
    callbacks::run(py, 1, move |runtime, progress| {
        let assets = specs
            .iter()
            .enumerate()
            .map(|(i, a)| core::FontAsset {
                id: &a.id,
                face: &faces[i],
                metadata: &metadata[i],
            })
            .collect::<Vec<_>>();
        let file = E2FileLayer {
            runtime,
            service: PythonService {
                dispatcher,
                recorder: recorder.clone(),
            },
        };
        struct Execution<'a, 'b, F: FileLayer> {
            ir: &'a mut Document,
            page_objects: Option<&'a core::PageObjectMap>,
            cfg: &'a WriteConfig,
            assets: &'a [core::FontAsset<'b>],
            flag: &'a CancelFlag,
            progress: &'a ProgressRun,
            monitor: &'a Py<PyAny>,
            file: &'a F,
        }
        impl<F: FileLayer> core::WriteAttempt for Execution<'_, '_, F> {
            type Value = Option<String>;
            type Error = PyErr;
            fn attempt(&mut self, checked: bool) -> PyResult<Self::Value> {
                let value = attempt(
                    self.ir,
                    self.page_objects,
                    self.cfg,
                    self.assets,
                    checked,
                    self.flag,
                    self.progress,
                    self.monitor,
                    self.file,
                )?;
                Python::attach(|py| value.bind(py).extract::<Option<String>>())
            }
            fn close(&mut self) -> PyResult<()> {
                self.file.call("write", "close", ()).map(|_| ())
            }
            fn ordinary(&self, error: &PyErr) -> bool {
                ordinary(error)
            }
            fn failed(&mut self, error: &PyErr) {
                log::error!("Failed to create PDF: {}: {error}", self.cfg.source);
            }
            fn recovered(&mut self) {
                self.progress.acknowledge_ended_errors();
            }
        }
        let (glossary, attempts) = core::write_with_retry(&mut Execution {
            ir,
            page_objects,
            cfg: &cfg,
            assets: &assets,
            flag: &flag,
            progress,
            monitor: &monitor,
            file: &file,
        })?;
        let requests = recorder
            .take()
            .into_iter()
            .map(|r| {
                json!({"sequence":r.sequence,
            "stage":r.request.stage,"method":r.request.method,"thread":r.execution_thread,
            "passed":r.outcome.is_ok()})
            })
            .collect::<Vec<_>>();
        Ok(
            json!({"attempts":attempts,"mono":cfg.mono,"dual":cfg.dual,"glossary":glossary,"requests":requests}),
        )
    })
}

#[cfg(test)]
mod writer_tests;
