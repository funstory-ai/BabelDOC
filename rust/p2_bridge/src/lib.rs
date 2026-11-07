use babeldoc_il_types::PythonValue;
use babeldoc_il_types::{CurveDebugInfo, RuntimeBool};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyInt, PyList, PyString, PyTuple};
pub mod callbacks;
pub mod font_mapper;
mod formula_patterns;
pub mod layout_parser;
pub mod llm_only_translation;
pub mod logging;
pub mod m_transfer;
pub mod ordinary_translation;
pub mod parser;
pub mod priority_gate;
pub mod production_stages;
pub mod run_context;
pub mod scan_detection;
pub mod shared_context;
pub mod stage_error;
pub mod term_extractor;
pub mod translation;
pub mod translation_service;
pub mod writer;
#[derive(Default)]
pub struct PyBuildContext;
pub trait FromPyValue: Sized {
    fn from_py(value: &Bound<'_, PyAny>, context: &mut PyBuildContext) -> PyResult<Self>;
}

impl FromPyValue for String {
    fn from_py(v: &Bound<'_, PyAny>, _: &mut PyBuildContext) -> PyResult<Self> {
        v.extract()
    }
}
impl FromPyValue for i64 {
    fn from_py(v: &Bound<'_, PyAny>, _: &mut PyBuildContext) -> PyResult<Self> {
        v.extract()
    }
}
impl FromPyValue for f64 {
    fn from_py(v: &Bound<'_, PyAny>, _: &mut PyBuildContext) -> PyResult<Self> {
        v.extract()
    }
}
impl FromPyValue for bool {
    fn from_py(v: &Bound<'_, PyAny>, _: &mut PyBuildContext) -> PyResult<Self> {
        v.extract()
    }
}
impl<T: FromPyValue> FromPyValue for Option<T> {
    fn from_py(v: &Bound<'_, PyAny>, c: &mut PyBuildContext) -> PyResult<Self> {
        if v.is_none() {
            Ok(None)
        } else {
            T::from_py(v, c).map(Some)
        }
    }
}
impl<T: FromPyValue> FromPyValue for std::boxed::Box<T> {
    fn from_py(v: &Bound<'_, PyAny>, c: &mut PyBuildContext) -> PyResult<Self> {
        T::from_py(v, c).map(std::boxed::Box::new)
    }
}
impl<T: FromPyValue> FromPyValue for Vec<T> {
    fn from_py(v: &Bound<'_, PyAny>, c: &mut PyBuildContext) -> PyResult<Self> {
        if let Ok(items) = v.cast::<PyList>() {
            return items.iter().map(|item| T::from_py(&item, c)).collect();
        }
        if let Ok(items) = v.cast::<PyTuple>() {
            return items.iter().map(|item| T::from_py(&item, c)).collect();
        }
        Err(PyTypeError::new_err("expected list or tuple for Rust Vec"))
    }
}
// Scalar carrier code is separate from other bridge responsibilities.
mod python_value;
pub use python_value::{from_python as carrier_from_python, to_python};
impl FromPyValue for PythonValue {
    fn from_py(v: &Bound<'_, PyAny>, _: &mut PyBuildContext) -> PyResult<Self> {
        python_value::from_python(v)
    }
}
impl FromPyValue for RuntimeBool {
    fn from_py(v: &Bound<'_, PyAny>, _: &mut PyBuildContext) -> PyResult<Self> {
        if v.is_instance_of::<PyBool>() {
            Ok(Self::Bool(v.extract()?))
        } else if v.is_instance_of::<PyInt>() {
            Ok(Self::Int(v.extract()?))
        } else {
            Err(PyTypeError::new_err(
                "expected bool or integer runtime flag",
            ))
        }
    }
}
impl FromPyValue for CurveDebugInfo {
    fn from_py(v: &Bound<'_, PyAny>, _: &mut PyBuildContext) -> PyResult<Self> {
        if v.is_instance_of::<PyBool>() {
            Ok(Self::Bool(v.extract()?))
        } else if v.is_instance_of::<PyString>() {
            Ok(Self::Text(v.extract()?))
        } else {
            Err(PyTypeError::new_err(
                "expected bool or string curve debug_info",
            ))
        }
    }
}

#[pymodule]
fn _rust_il(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("ParserResourceLimitError", module.py().import("babeldoc.format.pdf.parser_resource")?.getattr("ParserResourceLimitError")?)?;
    callbacks::register(module)?;
    logging::register(module)?;
    shared_context::register(module)?;
    font_mapper::register(module)?;
    run_context::register(module)?;
    module.add_class::<m_transfer::MDocument>()?;
    module.add_function(wrap_pyfunction!(m_transfer::transfer_document, module)?)?;
    module.add_function(wrap_pyfunction!(stage_capture_enabled, module)?)?;
    module.add_function(wrap_pyfunction!(parser::parse_prepared_pdf, module)?)?;
    module.add_function(wrap_pyfunction!(
        parser::parse_prepared_pdf_product,
        module
    )?)?;
    translation::register(module)?;
    Ok(())
}

#[pyfunction]
fn stage_capture_enabled() -> bool {
    cfg!(feature = "stage-capture")
}
