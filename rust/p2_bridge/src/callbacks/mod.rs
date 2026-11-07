//! Generic I/O callback primitives. CPU stages must not call these primitives.
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use async_trait::async_trait;
use babeldoc_p2_stages::Cancellation;
mod progress;
pub use progress::{ProgressError, ProgressRun, PythonProgress, progress_threads_alive};
use pyo3::BoundObject;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use tokio::sync::oneshot;

pyo3::create_exception!(_rust_il, BridgeError, pyo3::exceptions::PyBaseException);

/// Transport errors cannot enter an adapter's ordinary `except Exception` branch.
#[derive(Debug)]
pub enum CallbackError {
    Python(PyErr),
    Bridge(String),
}
impl CallbackError {
    pub fn into_python(self) -> PyErr {
        match self {
            Self::Python(error) => error,
            Self::Bridge(message) => BridgeError::new_err(message),
        }
    }
}

#[pyclass(from_py_object)]
#[derive(Clone, Default)]
pub struct CancelFlag {
    cancelled: Arc<AtomicBool>,
    times: Arc<Mutex<(Option<Instant>, Option<f64>)>>,
}
#[pymethods]
impl CancelFlag {
    #[new]
    fn new() -> Self {
        Self::default()
    }
    pub fn cancel(&self) {
        let mut times = self.times.lock().unwrap_or_else(|e| e.into_inner());
        if times.0.is_none() {
            times.0 = Some(Instant::now());
        }
        self.cancelled.store(true, Ordering::Release);
    }
    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Acquire)
    }
    /// The checkpoint is called at job start, never used to interrupt an in-flight call.
    pub fn checkpoint(&self) -> PyResult<()> {
        if self.is_cancelled() {
            let mut times = self.times.lock().unwrap_or_else(|e| e.into_inner());
            if times.1.is_none() {
                times.1 = times.0.map(|at| at.elapsed().as_secs_f64());
            }
            Err(Python::attach(|py| {
                match py
                    .import("asyncio")
                    .and_then(|m| m.getattr("CancelledError"))
                    .and_then(|c| c.call0())
                {
                    Ok(exception) => PyErr::from_value(exception),
                    Err(error) => error,
                }
            }))
        } else {
            Ok(())
        }
    }
    pub fn observation_delay(&self) -> Option<f64> {
        self.times.lock().unwrap_or_else(|e| e.into_inner()).1
    }
}
impl Cancellation for CancelFlag {
    fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Acquire)
    }
}

pub struct Request {
    pub stage: String,
    pub job: String,
    pub service: String,
    pub method: String,
    pub args: Py<PyTuple>,
    pub kwargs: Option<Py<PyDict>>,
}

/// Converts a Rust tuple directly into the Python argument tuple for a callback.
pub fn into_py_tuple<'py, T>(py: Python<'py>, args: T) -> PyResult<Py<PyTuple>>
where
    T: IntoPyObject<'py, Target = PyTuple>,
    T::Error: Into<PyErr>,
{
    args.into_pyobject(py)
        .map(|args| args.unbind())
        .map_err(Into::into)
}

pub struct RequestRecord {
    pub sequence: u64,
    pub request: Request,
    pub execution_thread: String,
    pub outcome: Result<Py<PyAny>, PyErr>,
}
pub struct Recorder {
    enabled: bool,
    next: AtomicU64,
    records: Mutex<Vec<RequestRecord>>,
}
impl Recorder {
    pub fn for_dispatcher(_py: Python<'_>, dispatcher: &Bound<'_, PyAny>) -> Arc<Self> {
        let enabled = dispatcher
            .getattr("record_requests")
            .and_then(|value| value.extract::<bool>())
            .unwrap_or(true);
        Arc::new(Self {
            enabled,
            ..Self::default()
        })
    }

    pub fn take(&self) -> Vec<RequestRecord> {
        std::mem::take(&mut *self.records.lock().unwrap_or_else(|e| e.into_inner()))
    }
}
impl Default for Recorder {
    fn default() -> Self {
        Self {
            enabled: true,
            next: AtomicU64::new(0),
            records: Mutex::new(Vec::new()),
        }
    }
}

/// One completion object owns one sender; no global reply-routing table exists.
#[pyclass]
struct Completion {
    sender: Mutex<Option<oneshot::Sender<Result<Py<PyAny>, CallbackError>>>>,
    record: Mutex<Option<(u64, Request)>>,
    recorder: Arc<Recorder>,
}
#[pymethods]
impl Completion {
    fn __call__(&self, py: Python<'_>, future: &Bound<'_, PyAny>) {
        // Future.result preserves the original exception, cause and traceback.
        let outcome = future.call_method0("result").map(Bound::unbind);
        if let Some((sequence, request)) =
            self.record.lock().unwrap_or_else(|e| e.into_inner()).take()
        {
            let recorded = match &outcome {
                Ok(value) => Ok(value.clone_ref(py)),
                Err(error) => Err(error.clone_ref(py)),
            };
            self.recorder
                .records
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .push(RequestRecord {
                    sequence,
                    request,
                    execution_thread: format!("{:?}", std::thread::current().id()),
                    outcome: recorded,
                });
        }
        if let Some(sender) = self.sender.lock().unwrap_or_else(|e| e.into_inner()).take() {
            // A dropped receiver is not an exception in this Python done callback.
            let _ = sender.send(outcome.map_err(CallbackError::Python));
        }
    }
}

#[async_trait]
pub trait CallbackService: Send + Sync {
    async fn request(&self, request: Request) -> Result<Py<PyAny>, CallbackError>;
}

pub struct PythonService {
    pub dispatcher: Py<PyAny>,
    pub recorder: Arc<Recorder>,
}
#[async_trait]
impl CallbackService for PythonService {
    async fn request(&self, request: Request) -> Result<Py<PyAny>, CallbackError> {
        let sequence = self
            .recorder
            .enabled
            .then(|| self.recorder.next.fetch_add(1, Ordering::Relaxed));
        let (sender, receiver) = oneshot::channel();
        Python::attach(|py| -> PyResult<()> {
            let method = request.method.clone();
            let args = request.args.clone_ref(py);
            let kwargs = request.kwargs.as_ref().map(|kw| kw.clone_ref(py));
            let completion = Py::new(
                py,
                Completion {
                    sender: Mutex::new(Some(sender)),
                    record: Mutex::new(sequence.map(|sequence| (sequence, request))),
                    recorder: Arc::clone(&self.recorder),
                },
            )?;
            if let Err(error) = self
                .dispatcher
                .bind(py)
                .call_method1("submit", (method, args, kwargs, &completion))
            {
                if let Some((sequence, request)) = completion
                    .borrow(py)
                    .record
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .take()
                {
                    self.recorder
                        .records
                        .lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .push(RequestRecord {
                            sequence,
                            request,
                            execution_thread: format!("submit:{:?}", std::thread::current().id()),
                            outcome: Err(BridgeError::new_err(format!(
                                "request submission failed: {error}"
                            ))),
                        });
                }
                return Err(error);
            }
            Ok(())
        })
        .map_err(|error| CallbackError::Bridge(format!("request submission failed: {error}")))?;
        receiver
            .await
            .map_err(|error| CallbackError::Bridge(format!("result channel closed: {error}")))?
    }
}

static SHORT_TOUCH: Mutex<()> = Mutex::new(());
/// Only call from a detached I/O orchestration thread. Methods must return promptly.
pub fn short_touch<T>(call: impl for<'py> FnOnce(Python<'py>) -> PyResult<T>) -> PyResult<T> {
    let _guard = SHORT_TOUCH.lock().unwrap_or_else(|e| e.into_inner());
    Python::attach(call)
}

/// One runtime and one progress thread per entry; both are joined before return.
pub fn run<T: Send>(
    py: Python<'_>,
    workers: usize,
    call: impl FnOnce(&tokio::runtime::Runtime, &ProgressRun) -> PyResult<T> + Send,
) -> PyResult<T> {
    crate::logging::begin_call(py)?;
    py.detach(|| {
        let result = (|| {
            let mut progress = ProgressRun::new()?;
            let result = tokio::runtime::Builder::new_multi_thread()
                .worker_threads(workers.max(1))
                .build()
                .map_err(|e| BridgeError::new_err(e.to_string()))
                .and_then(|runtime| call(&runtime, &progress));
            let closed = progress.close();
            closed?;
            result
        })();
        let flushed = crate::logging::flush();
        match result {
            Err(error) => Err(error),
            Ok(value) => {
                flushed?;
                Ok(value)
            }
        }
    })
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<CancelFlag>()?;
    module.add("BridgeError", module.py().get_type::<BridgeError>())?;
    Ok(())
}
