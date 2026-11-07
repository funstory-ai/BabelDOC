//! E2 adapter for the DetectScannedFile pure stage.
use crate::callbacks::{
    self, CallbackService, CancelFlag, ProgressError, PythonProgress, PythonService, Request,
};
use babeldoc_ir_core::{Document, ReadOnlyView};
use babeldoc_p2_stages::stages::detect_scanned_file as stage;
use babeldoc_p2_stages::{StageContext, StageError};
use babeldoc_writer_core as writer;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use std::time::Duration;
use tokio::runtime::Runtime;

enum Failure {
    Stage(StageError),
    Progress(ProgressError),
    Python(PyErr),
}
impl From<StageError> for Failure {
    fn from(e: StageError) -> Self {
        Self::Stage(e)
    }
}
impl From<ProgressError> for Failure {
    fn from(e: ProgressError) -> Self {
        Self::Progress(e)
    }
}
impl From<PyErr> for Failure {
    fn from(e: PyErr) -> Self {
        Self::Python(e)
    }
}
impl Failure {
    fn into_python(self, py: Python<'_>) -> PyErr {
        match self {
            Self::Stage(error) => crate::stage_error::to_python(py, &error),
            Self::Progress(error) => error.into_python(),
            Self::Python(error) => error,
        }
    }
}
struct ProgressAdapter(PythonProgress);
impl babeldoc_p2_stages::ProgressReporter for ProgressAdapter {
    type Error = Failure;
    fn report(&self, amount: u64, _: Option<u64>) -> Result<(), Failure> {
        self.0.report(amount, None).map_err(Failure::Progress)
    }
    fn checkpoint(&self) -> Result<(), Failure> {
        self.0.checkpoint().map_err(Failure::Progress)
    }
}

struct Config {
    object: Py<PyAny>,
    auto: bool,
    debug: bool,
    ocr_workaround: bool,
    skip_form_render: bool,
    skip_curve_render: bool,
    ocr: bool,
    skip_detection: bool,
    disable_rich_text: bool,
    remove_non_formula: bool,
}
impl stage::ScanConfig for Config {
    fn auto_enable_ocr_workaround(&self) -> bool {
        self.auto
    }
    fn debug(&self) -> bool {
        self.debug
    }
    fn set_ocr_workaround(&mut self, v: bool) {
        self.ocr = v;
    }
    fn set_skip_scanned_detection(&mut self, v: bool) {
        self.skip_detection = v;
    }
    fn set_disable_rich_text_translate(&mut self, v: bool) {
        self.disable_rich_text = v;
    }
    fn set_remove_non_formula_lines(&mut self, v: bool) {
        self.remove_non_formula = v;
    }
    fn touch_python_config(&mut self) -> Result<(), StageError> {
        callbacks::short_touch(|py| {
            let object = self.object.bind(py);
            object.setattr("ocr_workaround", self.ocr)?;
            object.setattr("skip_scanned_detection", self.skip_detection)?;
            object.setattr("disable_rich_text_translate", self.disable_rich_text)?;
            object.setattr("remove_non_formula_lines", self.remove_non_formula)
        })
        .map_err(|e| StageError::failed(e.to_string()))
    }
}
struct FileLayer<'a> {
    runtime: &'a Runtime,
    service: PythonService,
}
impl FileLayer<'_> {
    fn call<A>(&self, method: &str, args: A) -> PyResult<Py<PyAny>>
    where
        A: for<'py> IntoPyObject<'py, Target = PyTuple>,
    {
        let args = Python::attach(|py| callbacks::into_py_tuple(py, args))?;
        self.runtime
            .block_on(self.service.request(Request {
                stage: "DetectScannedFile".into(),
                job: method.into(),
                service: "scan".into(),
                method: method.into(),
                args,
                kwargs: None,
            }))
            .map_err(|e| e.into_python())
    }
}

fn forms_arg(py: Python<'_>, forms: &[writer::FormOutput]) -> PyResult<Py<PyList>> {
    let forms = forms
        .iter()
        .map(|form| {
            let value = PyDict::new(py);
            value.set_item("xobj_id", form.xobj_id)?;
            value.set_item("xref_id", form.xref_id)?;
            value.set_item("stream", form.stream.as_slice())?;
            Ok::<_, PyErr>(value.into_any())
        })
        .collect::<PyResult<Vec<_>>>()?;
    PyList::new(py, forms).map(Bound::unbind)
}
struct Backend<'a> {
    file: &'a FileLayer<'a>,
    input_path: String,
    options: writer::BuildOptions,
}
impl stage::ScanBackend for Backend<'_> {
    type Error = Failure;
    fn begin(&mut self) -> Result<(), Failure> {
        self.file.call("begin", (self.input_path.clone(),))?;
        Ok(())
    }
    fn render_before(&mut self, page: i64) -> Result<(), Failure> {
        self.file.call("render_before", (page,))?;
        Ok(())
    }
    fn apply_and_compare(
        &mut self,
        page: &babeldoc_ir_core::Cursor<babeldoc_ir_core::types::Page>,
        document: &mut Document,
    ) -> Result<f64, Failure> {
        let view = document.view::<ReadOnlyView>();
        let output = writer::build_page_output(
            &view,
            page,
            &writer::BuildOptions {
                skip_char: true,
                ..self.options.clone()
            },
        )
        .map_err(|e| Failure::Python(pyo3::exceptions::PyValueError::new_err(e.to_string())))?;
        let forms = Python::attach(|py| forms_arg(py, &output.forms))?;
        let result = self.file.call(
            "apply_and_compare",
            (output.page_number, output.stream.as_slice(), forms),
        )?;
        Python::attach(|py| result.bind(py).extract::<f64>()).map_err(Failure::Python)
    }
    fn close(&mut self) -> Result<(), Failure> {
        self.file.call("close", ())?;
        Ok(())
    }
}

#[cfg(test)]
mod scan_detection_tests;

pub fn process(
    py: Python<'_>,
    document: &mut Document,
    config_obj: &Bound<'_, PyAny>,
    shared_context: &Bound<'_, PyAny>,
    dispatcher: Py<PyAny>,
    flag: CancelFlag,
    monitor: Py<PyAny>,
) -> PyResult<bool> {
    let page_count = {
        let view = document.view::<ReadOnlyView>();
        view.document()
            .page_len(&view)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?
    };
    let mut selected = Vec::new();
    for number in 1..=page_count {
        if config_obj
            .call_method1("should_translate_page", (number,))?
            .extract::<bool>()?
        {
            selected.push(number as i64 - 1);
        }
    }
    if selected.is_empty() {
        return Ok(false);
    }
    let input_path: String = config_obj
        .call_method1("get_working_file_path", ("input.pdf",))?
        .str()?
        .to_str()?
        .to_owned();
    let config = Config {
        object: config_obj.clone().unbind(),
        auto: config_obj
            .getattr("auto_enable_ocr_workaround")?
            .extract()?,
        debug: config_obj.getattr("debug")?.extract()?,
        ocr_workaround: config_obj.getattr("ocr_workaround")?.extract()?,
        skip_form_render: config_obj.getattr("skip_form_render")?.extract()?,
        skip_curve_render: config_obj.getattr("skip_curve_render")?.extract()?,
        ocr: config_obj.getattr("ocr_workaround")?.extract()?,
        skip_detection: config_obj.getattr("skip_scanned_detection")?.extract()?,
        disable_rich_text: config_obj
            .getattr("disable_rich_text_translate")?
            .extract()?,
        remove_non_formula: config_obj.getattr("remove_non_formula_lines")?.extract()?,
    };
    let shared = crate::shared_context::root_from_handle(shared_context)?;
    let debug = config.debug;
    let ocr_workaround = config.ocr_workaround;
    let skip_form_render = config.skip_form_render;
    let skip_curve_render = config.skip_curve_render;
    let mut document = document;
    let recorder = callbacks::Recorder::for_dispatcher(py, dispatcher.bind(py));
    callbacks::run(py, 1, move |runtime, progress| {
        let monitor_ref = Python::attach(|py| monitor.clone_ref(py));
        let page_count = selected.len();
        let stage_progress = progress
            .start(
                monitor_ref,
                "DetectScannedFile",
                page_count as u64,
                Duration::from_millis(100),
            )
            .map_err(ProgressError::into_python)?;
        let file = FileLayer {
            runtime,
            service: PythonService {
                dispatcher,
                recorder,
            },
        };
        let mut backend = Backend {
            file: &file,
            input_path,
            options: writer::BuildOptions {
                debug,
                ocr_workaround,
                skip_form_render,
                skip_curve_render,
                ..Default::default()
            },
        };
        let reporter = ProgressAdapter(stage_progress.clone());
        let context = StageContext::new(&flag, &reporter);
        let mut config = config;
        let mut shared = shared;
        let outcome = stage::process(
            &mut document,
            |page| selected.binary_search(&(page - 1)).is_ok(),
            &mut config,
            &mut shared,
            &mut backend,
            &context,
        );
        match outcome {
            Ok(decision) => {
                stage_progress
                    .end(None)
                    .map_err(ProgressError::into_python)?;
                Ok(decision.hit)
            }
            Err(error) => {
                let (pyerr, progress_error) = Python::attach(|py| {
                    let pyerr = error.into_python(py);
                    let progress_error = ProgressError::from(pyerr.clone_ref(py));
                    (pyerr, progress_error)
                });
                let _ = stage_progress.end(Some(progress_error));
                Err(pyerr)
            }
        }
    })
}
