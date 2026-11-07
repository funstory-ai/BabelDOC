//! Prepared-PDF harness and product adapters over the same parser core.
use crate::m_transfer::MDocument;
use pyo3::exceptions::{
    PyAssertionError, PyIndexError, PyKeyError, PyOverflowError, PyRecursionError, PyRuntimeError,
    PyTypeError, PyValueError,
};
use pyo3::prelude::*;
use pyo3::types::PyType;

fn parser_error(py: Python<'_>, error: babeldoc_parser_core::Error) -> PyErr {
    if let Some(resource) = error.resource {
        let build = || -> PyResult<PyErr> {
            let kwargs = pyo3::types::PyDict::new(py);
            kwargs.set_item("limit_name", resource.limit_name)?;
            kwargs.set_item("limit", resource.limit)?;
            kwargs.set_item("observed_or_requested", resource.observed_or_requested)?;
            kwargs.set_item("unit", resource.unit)?;
            kwargs.set_item("scope", resource.scope)?;
            kwargs.set_item("phase", resource.phase)?;
            kwargs.set_item("page", resource.page)?;
            kwargs.set_item("object_ref", resource.object_ref)?;
            kwargs.set_item("policy_version", resource.policy_version)?;
            let value = py
                .import("babeldoc.format.pdf.parser_resource")?
                .getattr("ParserResourceLimitError")?
                .call((error.message,), Some(&kwargs))?;
            Ok(PyErr::from_value(value))
        };
        return build().unwrap_or_else(|error| error);
    }
    let message = error.message;
    match error.kind.as_str() {
        "ValueError" => PyValueError::new_err(message),
        "TypeError" => PyTypeError::new_err(message),
        "IndexError" => PyIndexError::new_err(message),
        "KeyError" => PyKeyError::new_err(message),
        "OverflowError" => PyOverflowError::new_err(message),
        "AssertionError" => PyAssertionError::new_err(message),
        "RecursionError" => PyRecursionError::new_err(message),
        kind => {
            let module = match kind {
                "UnsupportedOperatorError" => Some("babeldoc.format.pdf.parser_errors"),
                "TokenizerError" => Some("babeldoc.format.pdf.parser_errors"),
                "PDFTypeError" => Some("babeldoc.format.pdf.parser_errors"),
                "FileDataError" => Some("pymupdf"),
                "FzErrorFormat" => Some("pymupdf.mupdf"),
                "StructError" => Some("struct"),
                "ExtractTextError" => Some("babeldoc.babeldoc_exception.BabelDOCException"),
                _ => None,
            };
            if let Some(module) = module {
                let name = if kind == "StructError" { "error" } else { kind };
                match py
                    .import(module)
                    .and_then(|m| m.getattr(name))
                    .and_then(|class| class.cast_into::<PyType>().map_err(Into::into))
                {
                    Ok(class) => PyErr::from_type(class, message),
                    Err(error) => error,
                }
            } else {
                PyRuntimeError::new_err(format!("{kind}: {message}"))
            }
        }
    }
}

#[pyfunction]
#[pyo3(signature=(path, config_json="{}",font_mapper=None,shared_context=None))]
pub fn parse_prepared_pdf(
    py: Python<'_>,
    path: String,
    config_json: &str,
    font_mapper: Option<PyRef<'_, crate::font_mapper::FontMapperHandle>>,
    shared_context: Option<PyRef<'_, crate::shared_context::PySharedContextCrossSplitPart>>,
) -> PyResult<MDocument> {
    if font_mapper.is_some() != shared_context.is_some() {
        return Err(PyValueError::new_err(
            "font_mapper and shared_context must be supplied together",
        ));
    }
    let config: babeldoc_parser_core::Config =
        serde_json::from_str(config_json).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let mut observer = PythonWarningObserver { error: None };
    let parsed = py
        .detach(|| {
            babeldoc_parser_core::catch_parser_unwind(|| {
                babeldoc_parser_core::parse_observed(path, config, &mut observer)
            })
        })
        .map_err(|e| observer_error(py, e, observer.error.take()))?;
    if let (Some(mapper), Some(context)) = (font_mapper, shared_context) {
        let mapper = std::sync::Arc::clone(&mapper.mapper);
        let counts = py
            .detach(|| {
                babeldoc_parser_core::catch_parser_unwind(|| {
                    babeldoc_parser_core::valid_text::collect(&parsed.page_text, |s| {
                        mapper.has_char(s)
                    })
                })
            })
            .map_err(|e| parser_error(py, e))?;
        for (chars, tokens) in counts {
            if chars > 0 {
                context.root.add_valid_counts(chars, tokens);
            }
        }
    }
    #[cfg(feature = "memory-profile")]
    babeldoc_parser_core::memory_checkpoint("after_valid_text", serde_json::json!({}));
    let page_count = parsed.page_text.len();
    let page_objects = babeldoc_writer_core::PageObjectMap::new(parsed.page_objects);
    Ok(MDocument {
        document: parsed.document,
        page_count,
        diagnostics: parsed.diagnostics,
        page_objects,
    })
}

fn disabled_warning(py: Python<'_>) -> PyResult<()> {
    py.import("logging")?
        .getattr("getLogger")?
        .call1(("babeldoc.parser",))?
        .call_method1("warning", (babeldoc_parser_core::DISABLED_WARNING,))?;
    Ok(())
}
struct PythonWarningObserver {
    error: Option<PyErr>,
}
impl babeldoc_parser_core::ParseObserver for PythonWarningObserver {
    fn resource_limits_disabled(&mut self) -> babeldoc_parser_core::Result<()> {
        Python::attach(disabled_warning).map_err(|error| {
            self.error = Some(error);
            babeldoc_parser_core::Error::new("ProductObserverError", "parser warning failed")
        })
    }
}

struct ProductObserver {
    mapper: std::sync::Arc<babeldoc_p2_stages::font_mapper::FontMapper>,
    shared: babeldoc_p2_stages::runtime::shared_context::SharedContextRoot,
    monitor: Py<PyAny>,
    progress: Option<Py<PyAny>>,
    flag: crate::callbacks::CancelFlag,
    pages: Option<Vec<usize>>,
    completed: usize,
    python_error: Option<PyErr>,
}
impl ProductObserver {
    fn boundary(
        &mut self,
        action: impl FnOnce(Python<'_>, &mut Self) -> PyResult<()>,
    ) -> babeldoc_parser_core::Result<()> {
        let result = Python::attach(|py| action(py, self));
        result.map_err(|error| {
            self.python_error = Some(error);
            babeldoc_parser_core::Error::new("ProductObserverError", "parser observation failed")
        })
    }
}
impl babeldoc_parser_core::ParseObserver for ProductObserver {
    fn resource_limits_disabled(&mut self) -> babeldoc_parser_core::Result<()> {
        self.boundary(|py, _| disabled_warning(py))
    }

    fn start(&mut self, selected: usize) -> babeldoc_parser_core::Result<()> {
        self.boundary(|py, this| {
            let total = (0..selected)
                .filter(|n| this.pages.as_ref().is_none_or(|pages| pages.contains(n)))
                .count();
            this.progress = Some(
                this.monitor
                    .bind(py)
                    .call_method1(
                        "stage_start",
                        ("Parse PDF and Create Intermediate Representation", total),
                    )?
                    .unbind(),
            );
            Ok(())
        })
    }
    fn page_start(&mut self) -> babeldoc_parser_core::Result<()> {
        self.boundary(|_, this| this.flag.checkpoint())
    }
    fn page_end(&mut self, text: &[String]) -> babeldoc_parser_core::Result<()> {
        // The reference accumulates each finished page, even if a later page
        // fails. It logs statistics failures without replacing the parse error.
        match babeldoc_parser_core::valid_text::collect_pages(std::iter::once(text), |s| {
            self.mapper.has_char(s)
        }) {
            Ok(counts) => {
                for (chars, tokens) in counts {
                    if chars > 0 {
                        self.shared.add_valid_counts(chars, tokens);
                    }
                }
            }
            Err(error) => {
                if error.resource.is_some() {
                    return Err(error);
                }
                log::warn!("Failed to accumulate page valid stats: {error}");
            }
        }
        self.completed += 1;
        self.boundary(|py, this| {
            this.progress
                .as_ref()
                .unwrap()
                .bind(py)
                .call_method1("advance", (1,))?;
            Ok(())
        })
    }
    fn retain_page_text(&self) -> bool {
        false
    }
}

#[pyfunction]
#[pyo3(signature=(path, config_json, font_mapper, shared_context, flag, monitor, *, only_include_translated_page=false, capture_path=None, debug_path=None))]
pub fn parse_prepared_pdf_product(
    py: Python<'_>,
    path: String,
    config_json: &str,
    font_mapper: PyRef<'_, crate::font_mapper::FontMapperHandle>,
    shared_context: PyRef<'_, crate::shared_context::PySharedContextCrossSplitPart>,
    flag: crate::callbacks::CancelFlag,
    monitor: Py<PyAny>,
    only_include_translated_page: bool,
    capture_path: Option<String>,
    debug_path: Option<String>,
) -> PyResult<Option<MDocument>> {
    #[cfg(not(feature = "stage-capture"))]
    if capture_path.is_some() {
        return Err(PyRuntimeError::new_err(
            "stage capture was requested, but this Rust extension was built without the `stage-capture` Cargo feature",
        ));
    }
    let config: babeldoc_parser_core::Config =
        serde_json::from_str(config_json).map_err(|e| PyValueError::new_err(e.to_string()))?;
    flag.checkpoint()?;
    // ActiveILCreater constructs its tokenizer before loading page views.
    // Keep initialization failures outside the per-page statistics safeguard.
    py.detach(|| babeldoc_parser_core::valid_text::collect_pages(std::iter::empty(), |_| false))
        .map_err(|error| parser_error(py, error))?;
    let mut observer = ProductObserver {
        mapper: std::sync::Arc::clone(&font_mapper.mapper),
        shared: shared_context.root.clone(),
        monitor,
        progress: None,
        flag,
        pages: config.pages.clone(),
        completed: 0,
        python_error: None,
    };
    let parsed = py.detach(|| {
        babeldoc_parser_core::catch_parser_unwind(|| {
            babeldoc_parser_core::parse_observed(path, config, &mut observer)
        })
    });
    let mut parsed = parsed.map_err(|e| observer_error(py, e, observer.python_error.take()))?;
    observer
        .progress
        .as_ref()
        .unwrap()
        .bind(py)
        .call_method1("__exit__", (py.None(), py.None(), py.None()))?;
    #[cfg(feature = "stage-capture")]
    if let Some(path) = capture_path {
        crate::m_transfer::capture_at(
            &parsed.document,
            &serde_json::json!({"parser": path}),
            "parser",
        )?;
    }
    if only_include_translated_page && observer.completed == 0 {
        return Ok(None);
    }
    if let Some(path) = debug_path {
        let artifact = parsed
            .document
            .artifact()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let artifact_json =
            serde_json::to_string(&artifact).map_err(|e| PyValueError::new_err(e.to_string()))?;
        py.import("babeldoc.rust_debug_service")?
            .getattr("RustDebugSerializerService")?
            .call0()?
            .call_method1("write_debug_json", (artifact_json, path))?;
    }
    check_cid(&mut parsed.document).map_err(|e| parser_error(py, e))?;
    let page_objects = babeldoc_writer_core::PageObjectMap::new(parsed.page_objects);
    Ok(Some(MDocument {
        document: parsed.document,
        page_count: observer.completed,
        diagnostics: parsed.diagnostics,
        page_objects,
    }))
}

fn check_cid(document: &mut babeldoc_ir_core::Document) -> babeldoc_parser_core::Result<()> {
    let view = document.view::<babeldoc_ir_core::ReadOnlyView>();
    let failure = |e: babeldoc_ir_core::AccessError| {
        babeldoc_parser_core::Error::new("IRError", e.to_string())
    };
    let mut total = 0usize;
    let mut cid = 0usize;
    for page in view.document().page_iter(&view).map_err(failure)? {
        let page = page.map_err(failure)?;
        for character in page.pdf_character_iter(&view).map_err(failure)? {
            let character = character.map_err(failure)?;
            total += 1;
            if babeldoc_p2_stages::shared::formula_helper::is_cid_character(
                character
                    .char_unicode(&view)
                    .map_err(failure)?
                    .as_deref()
                    .unwrap_or_default(),
            ) {
                cid += 1;
            }
        }
    }
    if cid as f64 > total as f64 * 0.8 {
        return Err(babeldoc_parser_core::Error::new(
            "ExtractTextError",
            "The document contains too many CID chars.",
        ));
    }
    Ok(())
}

fn observer_error(
    py: Python<'_>,
    error: babeldoc_parser_core::Error,
    observer: Option<PyErr>,
) -> PyErr {
    if let Some(observer) = observer {
        if error.resource.is_some() {
            let resource = parser_error(py, error);
            resource.set_cause(py, observer.cause(py));
            observer.set_cause(py, Some(resource));
        }
        observer
    } else {
        parser_error(py, error)
    }
}

#[cfg(test)]
mod parser_tests;

#[cfg(test)]
#[path = "parser_resource_v2_tests.rs"]
mod resource_v2_tests;

#[cfg(test)]
#[path = "parser_expansion_tests.rs"]
mod expansion_tests;
