//! Direct Python-to-owned-IR handoff. Address memos are local to this loader only.
use crate::callbacks::{self, CallbackService, CancelFlag, PythonService, Recorder, Request};
use babeldoc_ir_core::{self as ir, *};
#[cfg(feature = "stage-capture")]
use pyo3::exceptions::PyOSError;
use pyo3::{
    exceptions::{PyRuntimeError, PyTypeError, PyValueError},
    prelude::*,
    types::{PyDict, PyList, PyString, PyTuple},
};
use std::{
    collections::{HashMap, HashSet},
    hash::{BuildHasherDefault, Hasher},
};
#[cfg(feature = "stage-capture")]
use std::{io::Write, path::Path};

#[derive(Clone, Copy, Default)]
struct ScanContext {
    page: Option<usize>,
    pool: bool,
}
struct AliasRecord {
    kind: Kind,
    count: usize,
    path: PathRange,
    priority: i32,
    page: Option<usize>,
    cross_page: bool,
    pool: bool,
}
#[derive(Clone, Copy)]
struct PathRange {
    start: usize,
    end: usize,
}
/// Handoff records use Python object addresses as keys; mix those addresses directly.
#[derive(Default)]
struct FastPointerHasher(u64);
impl Hasher for FastPointerHasher {
    fn finish(&self) -> u64 {
        let mut value = self.0;
        value ^= value >> 30;
        value = value.wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value ^= value >> 27;
        value = value.wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }

    fn write(&mut self, bytes: &[u8]) {
        let mut value = self.0;
        for byte in bytes {
            value = (value ^ u64::from(*byte)).wrapping_mul(0x100_0000_01b3);
        }
        self.0 = value;
    }

    fn write_usize(&mut self, value: usize) {
        self.0 = value as u64;
    }
}
type PointerHashState = BuildHasherDefault<FastPointerHasher>;

struct MBuildContext {
    records: HashMap<usize, AliasRecord, PointerHashState>,
    loaded: HashMap<usize, Edge>,
    building: HashSet<usize>,
    path_buffer: String,
    table: Table,
    history: History,
    history_keys: HashMap<usize, usize>,
    scalar_context: crate::PyBuildContext,
    cancel_flag: Option<CancelFlag>,
}
fn access_error(error: AccessError) -> PyErr {
    PyValueError::new_err(error.to_string())
}
impl MBuildContext {
    fn new() -> Self {
        Self {
            records: HashMap::default(),
            loaded: HashMap::new(),
            building: HashSet::new(),
            path_buffer: String::new(),
            table: Table::default(),
            history: History::default(),
            history_keys: HashMap::new(),
            scalar_context: crate::PyBuildContext::default(),
            cancel_flag: None,
        }
    }
    fn checkpoint(&self, py: Python<'_>) -> PyResult<()> {
        if let Some(flag) = self.cancel_flag.as_ref() {
            // Let the Python cancellation watcher run between pages, including
            // the ownership pre-scan, before checking the native atomic flag.
            py.detach(|| flag.checkpoint())?;
        }
        Ok(())
    }
    fn observe(
        &mut self,
        value: &Bound<'_, PyAny>,
        kind: Kind,
        context: ScanContext,
        path: &str,
        priority: i32,
        pool: bool,
    ) -> PyResult<(ScanContext, bool)> {
        if kind == Kind::Page {
            self.checkpoint(value.py())?;
        }
        if value.get_type().name()?.to_str()? != kind.name() {
            return Err(PyTypeError::new_err(format!(
                "{}: expected {}",
                path,
                kind.name()
            )));
        }
        let ptr = value.as_ptr() as usize;
        let mut context = context;
        if kind == Kind::Page {
            context.page = Some(ptr);
        }
        context.pool |= pool;
        if let Some(record) = self.records.get_mut(&ptr) {
            if record.kind != kind {
                return Err(access_error(AccessError::WrongType));
            }
            record.count += 1;
            if let Some(page) = context.page {
                if record.page.is_some_and(|first| first != page) {
                    record.cross_page = true;
                }
                record.page.get_or_insert(page);
            }
            record.pool |= context.pool;
            if priority > record.priority {
                record.priority = priority;
                let start = self.path_buffer.len();
                self.path_buffer.push_str(path);
                record.path = PathRange {
                    start,
                    end: self.path_buffer.len(),
                };
            }
            return Ok((context, false));
        }
        let start = self.path_buffer.len();
        self.path_buffer.push_str(path);
        let path = PathRange {
            start,
            end: self.path_buffer.len(),
        };
        self.records.insert(
            ptr,
            AliasRecord {
                kind,
                count: 1,
                path,
                priority,
                page: context.page,
                cross_page: false,
                pool: context.pool,
            },
        );
        Ok((context, true))
    }
    fn validate(&self) -> PyResult<()> {
        for record in self.records.values() {
            if !record.pool && record.cross_page {
                return Err(access_error(AccessError::CrossPage {
                    kind: record.kind.name().into(),
                    path: self.path_buffer[record.path.start..record.path.end].to_owned(),
                }));
            }
        }
        Ok(())
    }
    fn load(&mut self, value: &Bound<'_, PyAny>, path: &str) -> PyResult<Edge> {
        let ptr = value.as_ptr() as usize;
        let record = self
            .records
            .get(&ptr)
            .ok_or_else(|| PyValueError::new_err("object graph changed during transfer"))?;
        let shared = record.count > 1;
        if shared {
            if let Some(edge) = self.loaded.get(&ptr) {
                return Ok(edge.share());
            }
        }
        // A cycle necessarily revisits an object counted as shared by the scan.
        if shared && !self.building.insert(ptr) {
            return Err(PyValueError::new_err("cyclic schema graph"));
        }
        let data = load_data(value, self, path)?;
        if shared {
            self.building.remove(&ptr);
        }
        let edge = self.table.attach(Node::new(data));
        if shared {
            self.loaded.insert(ptr, edge.share());
        }
        Ok(edge)
    }
    fn load_optional(&mut self, value: &Bound<'_, PyAny>, path: &str) -> PyResult<Option<Edge>> {
        if value.is_none() {
            Ok(None)
        } else {
            self.load(value, path).map(Some)
        }
    }
    fn load_list(&mut self, value: &Bound<'_, PyAny>, path: &str) -> PyResult<Vec<Edge>> {
        let mut out = vec![];
        for (i, item) in value.try_iter()?.enumerate() {
            out.push(self.load(&item?, &format!("{path}[{i}]"))?);
        }
        Ok(out)
    }
    fn history_chain(&mut self, tail: &Bound<'_, PyAny>) -> PyResult<Option<usize>> {
        if tail.is_none() {
            return Ok(None);
        }
        let mut chain = vec![];
        let mut current = tail.clone();
        while !current.is_none() && !self.history_keys.contains_key(&(current.as_ptr() as usize)) {
            chain.push(current.clone());
            current = current.getattr("parent")?;
        }
        for item in chain.into_iter().rev() {
            let parent = item.getattr("parent")?;
            let parent = if parent.is_none() {
                None
            } else {
                Some(
                    *self
                        .history_keys
                        .get(&(parent.as_ptr() as usize))
                        .ok_or_else(|| access_error(AccessError::UnknownSlot))?,
                )
            };
            let instruction = item.getattr("instruction")?;
            let pair = instruction.cast::<PyTuple>()?;
            let arg = pair.get_item(1)?;
            let arg = if arg.is_instance_of::<PyString>() {
                HistoryArg::Text(arg.extract()?)
            } else {
                HistoryArg::CtmAware {
                    value: arg.getattr("value")?.extract()?,
                    ctm: <babeldoc_il_types::PythonValue as crate::FromPyValue>::from_py(
                        &arg.getattr("ctm")?,
                        &mut self.scalar_context,
                    )?,
                }
            };
            let key = self.history.push(HistoryNode {
                parent,
                op: pair.get_item(0)?.extract()?,
                arg,
                replace_key: item.getattr("replace_key")?.extract()?,
                version: item.getattr("version")?.extract()?,
                length: item.getattr("length")?.extract()?,
            });
            self.history_keys.insert(item.as_ptr() as usize, key);
        }
        self.history_keys
            .get(&(tail.as_ptr() as usize))
            .copied()
            .map(Some)
            .ok_or_else(|| access_error(AccessError::UnknownSlot))
    }
    fn instructions(&mut self, value: &Bound<'_, PyAny>) -> PyResult<Option<Instructions>> {
        if value.is_none() {
            return Ok(None);
        }
        if value.is_instance_of::<PyString>() {
            return Ok(Some(Instructions::Materialized(value.extract()?)));
        }
        if value.get_type().name()?.to_str()? != "LazyPassthroughInstruction" {
            return Err(PyTypeError::new_err("unexpected instruction value"));
        }
        let snapshot = value.getattr("snapshot")?;
        let tail = self.history_chain(&snapshot.getattr("tail")?)?;
        Ok(Some(Instructions::View(InstructionView {
            tail,
            latest_versions: snapshot.getattr("latest_versions")?.extract()?,
            include_clipping: value.getattr("include_clipping")?.extract()?,
            suffix_parts: value.getattr("suffix_parts")?.extract()?,
        })))
    }
}

include!(concat!(env!("OUT_DIR"), "/m_reader.rs"));

/// Atomic publication: an error returns no M document. `consume` progressively
/// releases Python page slots, and intentionally does not restore them on failure.
pub fn read_document(value: &Bound<'_, PyAny>, consume: bool) -> PyResult<ir::Document> {
    read_document_with_progress(value, consume, None, None)
}

/// The consuming path preserves the production transfer progress protocol.
/// Python source aliases belong exclusively to the transfer caller until return.
pub fn read_document_with_progress(
    value: &Bound<'_, PyAny>,
    consume: bool,
    progress_sink: Option<&Bound<'_, PyAny>>,
    cancel_flag: Option<CancelFlag>,
) -> PyResult<ir::Document> {
    let py = value.py();
    let pages = if consume {
        Some(value.getattr("page")?.cast_into::<PyList>().map_err(|_| {
            PyTypeError::new_err("transfer requires Document.page to be a mutable list")
        })?)
    } else {
        None
    };
    let total = pages.as_ref().map_or(0, |pages| pages.len());
    let total_pages: Option<i64> = if consume {
        crate::FromPyValue::from_py(
            &value.getattr("total_pages")?,
            &mut crate::PyBuildContext::default(),
        )?
    } else {
        None
    };
    let mut context = MBuildContext::new();
    context.cancel_flag = cancel_flag;
    context.checkpoint(py)?;
    scan_node(value, &mut context, ScanContext::default(), "$".into(), 0)?;
    context.validate()?;
    let mut progress_stage = if let Some(sink) = progress_sink {
        Some(
            sink.call_method1("stage_start", ("Rust transfer", total))
                .map_err(|err| {
                    PyRuntimeError::new_err(format!(
                        "transfer failed at page index 0, field path <progress.start>: {err}"
                    ))
                })?,
        )
    } else {
        None
    };
    if let Some(stage) = progress_stage.as_ref() {
        stage.call_method0("__enter__").map_err(|err| {
            PyRuntimeError::new_err(format!(
                "transfer failed at page index 0, field path <progress.start>: {err}"
            ))
        })?;
    }
    let root = if let Some(pages) = pages.as_ref() {
        let mut rust_pages = Vec::with_capacity(total);
        for index in 0..total {
            context.checkpoint(py)?;
            let page = pages.get_item(index)?;
            let rust_page = context
                .load(&page, &format!("$.page[{index}]"))
                .map_err(|err| {
                    PyRuntimeError::new_err(format!(
                        "transfer failed at page index {index}, field path <reader>: {err}"
                    ))
                })?;
            pages.set_item(index, py.None())?;
            drop(page);
            rust_pages.push(rust_page);
            if let Some(stage) = progress_stage.as_ref() {
                if let Err(err) = stage.call_method1("advance", (1,)) {
                    return Err(PyRuntimeError::new_err(format!(
                        "transfer failed at page index {index}, field path <progress.advance>: {err}"
                    )));
                }
            }
        }
        NodeData::Document(DataDocument {
            page: rust_pages,
            total_pages,
        })
    } else {
        load_data(value, &mut context, "$")?
    };
    let document = ir::Document::from_parts(Node::new(root), context.table, context.history)
        .map_err(access_error)?;
    if let Some(pages) = pages {
        pages.call_method0("clear")?;
    }
    drop(context.scalar_context);
    if let Some(stage) = progress_stage.take() {
        if let Err(err) = stage.call_method1("__exit__", (py.None(), py.None(), py.None())) {
            return Err(PyRuntimeError::new_err(format!(
                "transfer failed at page index {}, field path <progress.finish>: {err}",
                total.saturating_sub(1)
            )));
        }
    }
    Ok(document)
}

/// Rust owns the M document; read-only artifacts support handoff verification.
#[pyclass(module = "babeldoc._rust_il")]
pub struct MDocument {
    pub(crate) document: ir::Document,
    pub(crate) page_count: usize,
    pub(crate) diagnostics: babeldoc_parser_core::ParserDiagnostics,
    pub(crate) page_objects: babeldoc_writer_core::PageObjectMap,
}

#[pymethods]
impl MDocument {
    #[getter]
    fn page_count(&self) -> usize {
        self.page_count
    }

    #[getter]
    fn parser_invalid_objstm_members(&self) -> usize {
        self.diagnostics.invalid_objstm_members
    }

    #[getter]
    fn parser_invalid_objstm_offset_members(&self) -> usize {
        self.diagnostics.invalid_objstm_offset_members
    }

    #[getter]
    fn parser_invalid_objstm_segment_unbounded_parse_succeeded(&self) -> usize {
        self.diagnostics
            .invalid_objstm_segment_b20_would_accept
    }

    #[getter]
    fn parser_invalid_objstm_segment_both_fail(&self) -> usize {
        self.diagnostics.invalid_objstm_segment_both_fail
    }

    #[getter]
    fn parser_invalid_encrypted_objects(&self) -> usize {
        self.diagnostics.invalid_encrypted_objects
    }

    #[getter]
    fn parser_invalid_encrypted_offset_objects(&self) -> usize {
        self.diagnostics.invalid_encrypted_offset_objects
    }

    #[getter]
    fn parser_invalid_encrypted_segment_unbounded_parse_succeeded(&self) -> usize {
        self.diagnostics
            .invalid_encrypted_segment_b20_would_accept
    }

    #[getter]
    fn parser_invalid_encrypted_segment_both_fail(&self) -> usize {
        self.diagnostics
            .invalid_encrypted_segment_both_fail
    }

    #[getter]
    fn parser_long_font_alias_skips(&self) -> usize {
        self.diagnostics.long_font_alias_skips
    }

    #[getter]
    fn payload_utf8_bytes(&self) -> usize {
        self.document.payload_utf8_bytes()
    }

    fn artifact(&self) -> PyResult<String> {
        serde_json::to_string(&self.document.artifact().map_err(access_error)?)
            .map_err(|err| PyRuntimeError::new_err(err.to_string()))
    }

    fn parser_artifact_hash(&self) -> PyResult<String> {
        self.document.artifact_sha256().map_err(access_error)
    }

    fn parser_artifact_json(&self) -> PyResult<String> {
        self.document.artifact_json().map_err(access_error)
    }

    fn write_debug_json(&self, path: &str) -> PyResult<()> {
        let artifact = self.document.artifact().map_err(access_error)?;
        let bytes = serde_json::to_vec_pretty(&artifact)
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
        std::fs::write(path, bytes)
            .map_err(|error| pyo3::exceptions::PyOSError::new_err(error.to_string()))
    }

    #[pyo3(signature=(config, shared_context, dispatcher, flag, monitor))]
    fn detect_scanned_file(
        &mut self,
        py: Python<'_>,
        config: &Bound<'_, PyAny>,
        shared_context: &Bound<'_, PyAny>,
        dispatcher: Py<PyAny>,
        flag: CancelFlag,
        monitor: Py<PyAny>,
    ) -> PyResult<bool> {
        crate::scan_detection::process(
            py,
            &mut self.document,
            config,
            shared_context,
            dispatcher,
            flag,
            monitor,
        )
    }

    #[pyo3(signature=(config, shared_context, dispatcher, translator_service, flag, monitor))]
    fn extract_terms(
        &mut self,
        py: Python<'_>,
        config: &Bound<'_, PyAny>,
        shared_context: &Bound<'_, PyAny>,
        dispatcher: Py<PyAny>,
        translator_service: Py<PyAny>,
        flag: CancelFlag,
        monitor: Py<PyAny>,
    ) -> PyResult<()> {
        crate::term_extractor::process(
            py,
            &mut self.document,
            config,
            shared_context,
            dispatcher,
            translator_service,
            flag,
            monitor,
        )
    }

    #[pyo3(signature=(config, shared_context, font_mapper, dispatcher, translator_service, flag, monitor))]
    fn translate_paragraphs(
        &mut self,
        py: Python<'_>,
        config: &Bound<'_, PyAny>,
        shared_context: &Bound<'_, PyAny>,
        font_mapper: PyRef<'_, crate::font_mapper::FontMapperHandle>,
        dispatcher: Py<PyAny>,
        translator_service: Py<PyAny>,
        flag: CancelFlag,
        monitor: Py<PyAny>,
    ) -> PyResult<()> {
        let result = crate::ordinary_translation::process(
            py,
            &mut self.document,
            config,
            shared_context,
            &font_mapper,
            dispatcher,
            translator_service,
            flag,
            monitor,
        );
        if result.is_ok() {
            self.document.compact();
        }
        result
    }

    #[pyo3(signature=(config, shared_context, font_mapper, dispatcher_a, dispatcher_b, translator_service, flag, monitor))]
    fn translate_paragraphs_llm_only(
        &mut self,
        py: Python<'_>,
        config: &Bound<'_, PyAny>,
        shared_context: &Bound<'_, PyAny>,
        font_mapper: PyRef<'_, crate::font_mapper::FontMapperHandle>,
        dispatcher_a: Py<PyAny>,
        dispatcher_b: Py<PyAny>,
        translator_service: Py<PyAny>,
        flag: CancelFlag,
        monitor: Py<PyAny>,
    ) -> PyResult<(usize, usize, usize)> {
        let result = crate::llm_only_translation::process(
            py,
            &mut self.document,
            config,
            shared_context,
            &font_mapper,
            dispatcher_a,
            dispatcher_b,
            translator_service,
            flag,
            monitor,
        );
        if result.is_ok() {
            self.document.compact();
        }
        result
    }

    #[pyo3(signature=(run_context, config, executors, stage_config_json, paths_json, writer_config_json, writer_assets_json, flag, monitor, capture_stages, capture_paths_json))]
    fn run_part(
        &mut self,
        py: Python<'_>,
        run_context: PyRef<'_, crate::run_context::DocumentRunContext>,
        config: &Bound<'_, PyAny>,
        executors: &Bound<'_, PyDict>,
        stage_config_json: &str,
        paths_json: &str,
        writer_config_json: &str,
        writer_assets_json: &str,
        flag: CancelFlag,
        monitor: Py<PyAny>,
        capture_stages: bool,
        capture_paths_json: &str,
    ) -> PyResult<(Option<String>, Option<String>, Option<String>)> {
        #[cfg(not(feature = "stage-capture"))]
        if capture_stages {
            return Err(PyRuntimeError::new_err(
                "stage capture was requested, but this Rust extension was built without the `stage-capture` Cargo feature",
            ));
        }
        #[cfg(not(feature = "stage-capture"))]
        let _ = capture_paths_json;
        if !config.getattr("table_model")?.is_none() {
            return Err(PyValueError::new_err(
                "table_model is not supported by the Rust production entry",
            ));
        }
        let mut stages: serde_json::Value = serde_json::from_str(stage_config_json)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        let paths: serde_json::Value = serde_json::from_str(paths_json)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        #[cfg(feature = "stage-capture")]
        let capture_paths: serde_json::Value = if capture_stages {
            serde_json::from_str(capture_paths_json)
                .map_err(|error| PyValueError::new_err(error.to_string()))?
        } else {
            serde_json::Value::Null
        };
        let mut writer_cfg: crate::writer::WriteConfig =
            serde_json::from_str(writer_config_json)
                .map_err(|error| PyValueError::new_err(error.to_string()))?;
        writer_cfg.keep_pages = writer_keep_pages(&mut self.document, config)?;
        let writer_assets: Vec<crate::writer::AssetSpec> = serde_json::from_str(writer_assets_json)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        let shared_context = run_context.shared_context.clone_ref(py);
        let font_mapper = run_context.font_mapper.clone_ref(py);
        let dispatcher_a = run_context.dispatcher_a.clone_ref(py);
        let dispatcher_b = run_context.dispatcher_b.clone_ref(py);
        drop(run_context);
        let mapper = font_mapper
            .bind(py)
            .extract::<PyRef<'_, crate::font_mapper::FontMapperHandle>>()?;
        let mapper = crate::font_mapper::FontMapperHandle::from_shared(std::sync::Arc::clone(
            &mapper.mapper,
        ));
        let executor = |name: &str| -> PyResult<Py<PyAny>> {
            executors
                .get_item(name)?
                .ok_or_else(|| PyValueError::new_err(format!("missing {name} callback executor")))
                .map(Bound::unbind)
        };
        let debug = config.getattr("debug")?.extract::<bool>()?;
        if config
            .getattr("only_parse_generate_pdf")?
            .extract::<bool>()?
        {
            return write_part(
                py,
                &mut self.document,
                &self.page_objects,
                writer_cfg,
                writer_assets,
                &mapper,
                executor("writer")?,
                flag,
                monitor,
            );
        }
        if !config
            .getattr("skip_scanned_detection")?
            .extract::<bool>()?
        {
            crate::scan_detection::process(
                py,
                &mut self.document,
                config,
                shared_context.bind(py),
                executor("scan")?,
                flag.clone(),
                monitor.clone_ref(py),
            )?;
            if debug {
                dump_at(py, &self.document, &paths, "scan", executor("debug")?)?;
            }
            let ocr_workaround = config.getattr("ocr_workaround")?.extract::<bool>()?;
            let remove_non_formula_lines = config
                .getattr("remove_non_formula_lines")?
                .extract::<bool>()?;
            stages["paragraph"]["ocr_workaround"] = ocr_workaround.into();
            stages["styles"]["config_fields"]["ocr_workaround"] = ocr_workaround.into();
            stages["styles"]["config_fields"]["remove_non_formula_lines"] =
                remove_non_formula_lines.into();
            writer_cfg.ocr_workaround = ocr_workaround;
        }
        #[cfg(feature = "stage-capture")]
        if capture_stages {
            capture_at(&self.document, &capture_paths, "scan")?;
        }
        let layout_executor = executor("layout")?;
        let layout_service = executors
            .get_item("layout")?
            .ok_or_else(|| PyValueError::new_err("missing layout callback executor"))?;
        let characters = layout_service
            .getattr("service")?
            .getattr("characters")?
            .extract::<bool>()?;
        drop(layout_service);
        crate::layout_parser::process(
            py,
            &mut self.document,
            layout_executor,
            flag.clone(),
            monitor.clone_ref(py),
            debug,
            characters,
            debug_path(&paths, "layout"),
        )?;
        if debug {
            dump_at(py, &self.document, &paths, "layout", executor("debug")?)?;
        }
        #[cfg(feature = "stage-capture")]
        if capture_stages {
            capture_at(&self.document, &capture_paths, "layout")?;
        }
        let paragraph_config = stages["paragraph"].clone();
        let paragraph_mapper = crate::font_mapper::FontMapperHandle::from_shared(
            std::sync::Arc::clone(&mapper.mapper),
        );
        let paragraph_monitor = monitor.clone_ref(py);
        let paragraph_flag = flag.clone();
        let document = &mut self.document;
        py.detach(move || {
            crate::production_stages::paragraph_finder(
                document,
                &paragraph_config,
                &paragraph_mapper,
                paragraph_monitor,
                &paragraph_flag,
            )
        })?;
        if debug {
            dump_at(py, &self.document, &paths, "paragraph", executor("debug")?)?;
        }
        #[cfg(feature = "stage-capture")]
        if capture_stages {
            capture_at(&self.document, &capture_paths, "paragraph")?;
        }
        let styles_config = stages["styles"].clone();
        let styles_mapper = crate::font_mapper::FontMapperHandle::from_shared(
            std::sync::Arc::clone(&mapper.mapper),
        );
        let styles_monitor = monitor.clone_ref(py);
        let styles_flag = flag.clone();
        let document = &mut self.document;
        py.detach(move || {
            crate::production_stages::styles_and_formulas(
                document,
                &styles_config,
                &styles_mapper,
                styles_monitor,
                &styles_flag,
            )
        })?;
        if debug {
            dump_at(py, &self.document, &paths, "styles", executor("debug")?)?;
        }
        #[cfg(feature = "stage-capture")]
        if capture_stages {
            capture_at(&self.document, &capture_paths, "styles")?;
        }
        // The 0.6.4 probe happens after styles and before term extraction. Keep
        // it as a Python callback so caller supplied translator objects retain
        // their own behavior and the original exception/logging policy.
        let supports_llm_translate =
            probe_llm_support(py, executor("probe_translate")?, "translation")?;
        let supports_llm_terms = probe_llm_support(py, executor("probe_terms")?, "term")?;
        if supports_llm_terms && config.getattr("auto_extract_glossary")?.extract::<bool>()? {
            let term_executor = executor("term")?;
            let term_service = executors
                .get_item("term")?
                .ok_or_else(|| PyValueError::new_err("missing term callback executor"))?
                .getattr("service")?
                .unbind();
            self.extract_terms(
                py,
                config,
                shared_context.bind(py),
                term_executor,
                term_service,
                flag.clone(),
                monitor.clone_ref(py),
            )?;
        }
        if !config.getattr("skip_translation")?.extract::<bool>()? {
            let translation_mapper = font_mapper
                .bind(py)
                .extract::<PyRef<'_, crate::font_mapper::FontMapperHandle>>()?;
            let translation_service = executors
                .get_item("translation")?
                .ok_or_else(|| PyValueError::new_err("missing translation callback executor"))?
                .getattr("service")?
                .unbind();
            if supports_llm_translate {
                self.translate_paragraphs_llm_only(
                    py,
                    config,
                    shared_context.bind(py),
                    translation_mapper,
                    dispatcher_a.clone_ref(py),
                    dispatcher_b.clone_ref(py),
                    translation_service.clone_ref(py),
                    flag.clone(),
                    monitor.clone_ref(py),
                )?;
            } else {
                self.translate_paragraphs(
                    py,
                    config,
                    shared_context.bind(py),
                    translation_mapper,
                    dispatcher_a.clone_ref(py),
                    translation_service,
                    flag.clone(),
                    monitor.clone_ref(py),
                )?;
            }
        }
        #[cfg(feature = "stage-capture")]
        if capture_stages {
            capture_at(&self.document, &capture_paths, "translation")?;
        }
        if debug {
            dump_at(
                py,
                &self.document,
                &paths,
                "translation",
                executor("debug")?,
            )?;
            crate::production_stages::add_debug_information(&mut self.document, true)?;
            dump_at(
                py,
                &self.document,
                &paths,
                "debug_information",
                executor("debug")?,
            )?;
        }
        let typesetting_config = stages["typesetting"].clone();
        let typesetting_mapper = crate::font_mapper::FontMapperHandle::from_shared(
            std::sync::Arc::clone(&mapper.mapper),
        );
        let typesetting_monitor = monitor.clone_ref(py);
        let typesetting_flag = flag.clone();
        let document = &mut self.document;
        py.detach(move || {
            crate::production_stages::typesetting(
                document,
                &typesetting_config,
                &typesetting_mapper,
                typesetting_monitor,
                &typesetting_flag,
            )
        })?;
        writer_cfg.ocr_workaround = config.getattr("ocr_workaround")?.extract::<bool>()?;
        if debug {
            dump_at(
                py,
                &self.document,
                &paths,
                "typesetting",
                executor("debug")?,
            )?;
        }
        #[cfg(feature = "stage-capture")]
        if capture_stages {
            capture_at(&self.document, &capture_paths, "typesetting")?;
        }
        write_part(
            py,
            &mut self.document,
            &self.page_objects,
            writer_cfg,
            writer_assets,
            &mapper,
            executor("writer")?,
            flag,
            monitor,
        )
    }
}

fn probe_llm_support(
    py: Python<'_>,
    dispatcher: Py<PyAny>,
    service_name: &'static str,
) -> PyResult<bool> {
    use pyo3::types::PyTuple;

    let args = PyTuple::empty(py).unbind();
    let recorder = Recorder::for_dispatcher(py, dispatcher.bind(py));
    let service = PythonService {
        dispatcher,
        recorder,
    };
    crate::callbacks::run(py, 1, move |runtime, _| {
        let value = runtime
            .block_on(service.request(Request {
                stage: "TranslatorProbe".into(),
                job: service_name.into(),
                service: service_name.into(),
                method: "probe".into(),
                args,
                kwargs: None,
            }))
            .map_err(|error| error.into_python())?;
        Python::attach(|py| value.bind(py).extract::<bool>())
    })
}

fn debug_path(paths: &serde_json::Value, name: &str) -> Option<String> {
    paths
        .get(name)
        .and_then(serde_json::Value::as_str)
        .map(str::to_owned)
}

fn dump_at(
    py: Python<'_>,
    document: &ir::Document,
    paths: &serde_json::Value,
    name: &str,
    dispatcher: Py<PyAny>,
) -> PyResult<()> {
    let Some(path) = debug_path(paths, name) else {
        return Ok(());
    };
    let artifact = document
        .artifact()
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    let artifact_json = serde_json::to_string(&artifact)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    let args = PyTuple::new(
        py,
        [
            artifact_json.into_pyobject(py)?.into_any(),
            path.into_pyobject(py)?.into_any(),
        ],
    )?
    .unbind();
    let recorder = Recorder::for_dispatcher(py, dispatcher.bind(py));
    let service = PythonService {
        dispatcher,
        recorder,
    };
    callbacks::run(py, 1, move |runtime, _| {
        runtime
            .block_on(service.request(Request {
                stage: "DebugSerialization".into(),
                job: name.into(),
                service: "debug".into(),
                method: "write_debug_json".into(),
                args,
                kwargs: None,
            }))
            .map_err(|error| error.into_python())?;
        Ok(())
    })
}

#[cfg(feature = "stage-capture")]
pub(crate) fn capture_at(
    document: &ir::Document,
    paths: &serde_json::Value,
    name: &str,
) -> PyResult<()> {
    let Some(path) = debug_path(paths, name) else {
        return Ok(());
    };
    let artifact = document
        .artifact()
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    let mut bytes = serde_json::to_vec_pretty(&artifact)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    bytes.push(b'\n');
    let path = Path::new(&path);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|error| {
            PyOSError::new_err(format!("failed to create stage capture directory: {error}"))
        })?;
    }
    let file = std::fs::File::create(path)
        .map_err(|error| PyOSError::new_err(format!("failed to create stage capture: {error}")))?;
    let mut encoder = zstd::stream::Encoder::new(file, 1).map_err(|error| {
        PyOSError::new_err(format!("failed to start zstd stage capture: {error}"))
    })?;
    encoder.write_all(&bytes).map_err(|error| {
        PyOSError::new_err(format!("failed to compress stage capture: {error}"))
    })?;
    encoder.finish().map_err(|error| {
        PyOSError::new_err(format!("failed to finish zstd stage capture: {error}"))
    })?;
    Ok(())
}

/// Read page identity in Rust and apply the very same one-based predicate as
/// TranslationConfig; no Python IL or duplicated range grammar is involved.
fn writer_keep_pages(document: &mut ir::Document, config: &Bound<'_, PyAny>) -> PyResult<Vec<i64>> {
    let view = document.view::<ir::ReadOnlyView>();
    let mut keep = Vec::new();
    for page in view.document().page_iter(&view).map_err(access_error)? {
        let number = page
            .map_err(access_error)?
            .page_number(&view)
            .map_err(access_error)?
            .ok_or_else(|| PyTypeError::new_err("page_number is None"))?;
        if config
            .call_method1("should_translate_page", (number + 1,))?
            .is_truthy()?
        {
            keep.push(number);
        }
    }
    Ok(keep)
}

fn write_part(
    py: Python<'_>,
    document: &mut ir::Document,
    page_objects: &babeldoc_writer_core::PageObjectMap,
    config: crate::writer::WriteConfig,
    assets: Vec<crate::writer::AssetSpec>,
    mapper: &crate::font_mapper::FontMapperHandle,
    dispatcher: Py<PyAny>,
    flag: CancelFlag,
    monitor: Py<PyAny>,
) -> PyResult<(Option<String>, Option<String>, Option<String>)> {
    let result = crate::writer::process_with_page_objects(
        py,
        document,
        page_objects,
        config,
        mapper,
        assets,
        dispatcher,
        flag,
        monitor,
    )?;
    Ok((
        result
            .get("mono")
            .and_then(serde_json::Value::as_str)
            .map(str::to_owned),
        result
            .get("dual")
            .and_then(serde_json::Value::as_str)
            .map(str::to_owned),
        result
            .get("glossary")
            .and_then(serde_json::Value::as_str)
            .map(str::to_owned),
    ))
}

#[pyfunction]
#[pyo3(signature=(source_document, progress_sink=None, flag=None))]
pub(crate) fn transfer_document(
    source_document: &Bound<'_, PyAny>,
    progress_sink: Option<&Bound<'_, PyAny>>,
    flag: Option<CancelFlag>,
) -> PyResult<MDocument> {
    let page_count = source_document.getattr("page")?.len()?;
    Ok(MDocument {
        document: read_document_with_progress(source_document, true, progress_sink, flag)?,
        page_count,
        diagnostics: babeldoc_parser_core::ParserDiagnostics::default(),
        page_objects: babeldoc_writer_core::PageObjectMap::default(),
    })
}
