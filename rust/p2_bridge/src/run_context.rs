use pyo3::prelude::*;

/// One document's immutable native handles plus its independent LLM dispatchers.
#[pyclass(module = "babeldoc._rust_il")]
pub struct DocumentRunContext {
    pub shared_context: Py<PyAny>,
    pub font_mapper: Py<PyAny>,
    pub dispatcher_a: Py<PyAny>,
    pub dispatcher_b: Py<PyAny>,
}

#[pymethods]
impl DocumentRunContext {
    #[new]
    fn new(
        shared_context: Py<PyAny>,
        font_mapper: Py<PyAny>,
        dispatcher_a: Py<PyAny>,
        dispatcher_b: Py<PyAny>,
        _py: Python<'_>,
    ) -> Self {
        Self {
            shared_context,
            font_mapper,
            dispatcher_a,
            dispatcher_b,
        }
    }

    #[getter]
    fn shared_context(&self, py: Python<'_>) -> Py<PyAny> {
        self.shared_context.clone_ref(py)
    }

    #[getter]
    fn font_mapper(&self, py: Python<'_>) -> Py<PyAny> {
        self.font_mapper.clone_ref(py)
    }

    #[getter]
    fn dispatcher_a(&self, py: Python<'_>) -> Py<PyAny> {
        self.dispatcher_a.clone_ref(py)
    }

    #[getter]
    fn dispatcher_b(&self, py: Python<'_>) -> Py<PyAny> {
        self.dispatcher_b.clone_ref(py)
    }
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<DocumentRunContext>()
}
