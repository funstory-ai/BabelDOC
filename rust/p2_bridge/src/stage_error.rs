use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use babeldoc_p2_stages::{StageError, StageErrorCategory};

pub fn to_python(py: Python<'_>, error: &StageError) -> PyErr {
    let message = error.to_string();
    match error.category() {
        StageErrorCategory::ValueError => PyValueError::new_err(message),
        StageErrorCategory::Failed => PyRuntimeError::new_err(message),
        StageErrorCategory::Cancelled => {
            let asyncio = PyModule::import(py, "asyncio").expect("asyncio is available");
            let error_type = asyncio
                .getattr("CancelledError")
                .expect("asyncio.CancelledError is available");
            PyErr::from_value(
                error_type
                    .call0()
                    .expect("CancelledError construction succeeds"),
            )
        }
        StageErrorCategory::ScannedPdf => custom_exception(
            py,
            "babeldoc.babeldoc_exception.BabelDOCException",
            "ScannedPDFError",
            &message,
        ),
        StageErrorCategory::ExtractText => custom_exception(
            py,
            "babeldoc.babeldoc_exception.BabelDOCException",
            "ExtractTextError",
            &message,
        ),
        StageErrorCategory::ContentFilter => custom_exception(
            py,
            "babeldoc.babeldoc_exception.BabelDOCException",
            "ContentFilterError",
            &message,
        ),
    }
}

fn custom_exception(py: Python<'_>, module: &str, name: &str, message: &str) -> PyErr {
    let exception_type = PyModule::import(py, module)
        .expect("BabelDOC exception module is importable")
        .getattr(name)
        .expect("BabelDOC exception type is available");
    PyErr::from_value(
        exception_type
            .call1((message,))
            .expect("BabelDOC exception construction succeeds"),
    )
}
