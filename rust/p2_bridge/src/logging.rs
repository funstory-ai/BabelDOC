use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Sender};
use std::sync::{Mutex, OnceLock, RwLock};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use log::{Level, LevelFilter, Log, Metadata, Record};
use pyo3::prelude::*;

const STAGE_TARGETS: [&str; 5] = [
    "babeldoc.format.pdf.document_il.midend.paragraph_finder",
    "babeldoc.format.pdf.document_il.midend.styles_and_formulas",
    "babeldoc.format.pdf.document_il.midend.add_debug_information",
    "babeldoc.format.pdf.document_il.midend.typesetting",
    "babeldoc.format.pdf.document_il.midend.detect_scanned_file",
];

enum Command {
    Record(QueuedRecord),
    Flush(Sender<()>),
    Stop(Sender<()>),
}

struct QueuedRecord {
    target: String,
    level: Level,
    message: String,
    created: f64,
}

struct PythonLogger {
    sender: Mutex<Option<Sender<Command>>>,
    enabled: RwLock<HashMap<String, [bool; 4]>>,
}

impl PythonLogger {
    fn new() -> Self {
        Self {
            sender: Mutex::new(None),
            enabled: RwLock::new(HashMap::new()),
        }
    }
}

impl Log for PythonLogger {
    fn enabled(&self, metadata: &Metadata<'_>) -> bool {
        let Ok(enabled) = self.enabled.read() else {
            return false;
        };
        let mut target = metadata.target();
        loop {
            if let Some(levels) = enabled.get(target) {
                return levels[level_index(metadata.level())];
            }
            if let Some(separator) = target.rfind('.') {
                target = &target[..separator];
            } else {
                return enabled
                    .get("")
                    .is_some_and(|levels| levels[level_index(metadata.level())]);
            }
        }
    }

    fn log(&self, record: &Record<'_>) {
        if !self.enabled(record.metadata()) {
            return;
        }
        let created = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs_f64();
        let command = Command::Record(QueuedRecord {
            target: record.target().to_owned(),
            level: record.level(),
            message: record.args().to_string(),
            created,
        });
        if let Ok(sender) = self.sender.lock()
            && let Some(sender) = sender.as_ref()
        {
            let _ = sender.send(command);
        }
    }

    fn flush(&self) {}
}

static LOGGER: OnceLock<PythonLogger> = OnceLock::new();
static INSTALLED: AtomicBool = AtomicBool::new(false);
#[derive(Default)]
struct RunState {
    worker: Option<JoinHandle<()>>,
    active_runs: usize,
}

static RUN_STATE: OnceLock<Mutex<RunState>> = OnceLock::new();

fn run_state() -> &'static Mutex<RunState> {
    RUN_STATE.get_or_init(|| Mutex::new(RunState::default()))
}

fn level_index(level: Level) -> usize {
    match level {
        Level::Error => 0,
        Level::Warn => 1,
        Level::Info => 2,
        Level::Debug => 3,
        Level::Trace => 3,
    }
}

fn logger() -> &'static PythonLogger {
    LOGGER.get_or_init(PythonLogger::new)
}

pub fn begin_call(py: Python<'_>) -> PyResult<()> {
    prepare_call(py, false)
}

pub fn acquire_run(py: Python<'_>) -> PyResult<()> {
    prepare_call(py, true)
}

pub fn snapshot_targets<'a>(
    py: Python<'_>,
    targets: impl IntoIterator<Item = &'a str>,
) -> PyResult<()> {
    let logging = PyModule::import(py, "logging")?;
    let manager = logging.getattr("root")?.getattr("manager")?;
    let logger_dict = manager.getattr("loggerDict")?;
    let known_targets: Vec<String> = PyModule::import(py, "builtins")?
        .getattr("list")?
        .call1((logger_dict,))?
        .extract()?;
    let mut snapshot = HashMap::new();
    let targets = std::iter::once(String::new())
        .chain(known_targets)
        .chain(targets.into_iter().map(str::to_owned))
        .collect::<std::collections::BTreeSet<_>>();
    for target in targets {
        let logger = logging.call_method1("getLogger", (&target,))?;
        let mut levels = [false; 4];
        for (index, number) in [40, 30, 20, 10].into_iter().enumerate() {
            levels[index] = logger.call_method1("isEnabledFor", (number,))?.extract()?;
        }
        snapshot.insert(target, levels);
    }
    *logger().enabled.write().map_err(|_| {
        pyo3::exceptions::PyRuntimeError::new_err("Rust logger snapshot lock poisoned")
    })? = snapshot;
    Ok(())
}

/// Snapshot before publishing a worker or a lease: a Python logger failure
/// leaves the lifecycle unchanged, including when this is the first call.
fn prepare_call(py: Python<'_>, acquire: bool) -> PyResult<()> {
    snapshot_targets(py, STAGE_TARGETS.iter().copied())?;
    // A final release holds this lock while joining a worker that needs the
    // GIL. A competing acquire must wait without holding the GIL as well.
    py.detach(|| {
        let mut state = run_state().lock().map_err(|_| {
            pyo3::exceptions::PyRuntimeError::new_err("Rust logger run lock poisoned")
        })?;
        install(&mut state)?;
        if acquire {
            state.active_runs += 1;
        }
        Ok(())
    })
}

fn install(state: &mut RunState) -> PyResult<()> {
    if state.worker.is_some() {
        return Ok(());
    }
    if !INSTALLED.load(Ordering::Acquire) {
        log::set_logger(logger())
            .map_err(|error| pyo3::exceptions::PyRuntimeError::new_err(error.to_string()))?;
        log::set_max_level(LevelFilter::Trace);
        INSTALLED.store(true, Ordering::Release);
    }
    let (sender, receiver) = mpsc::channel();
    let mut slot = logger().sender.lock().map_err(|_| {
        pyo3::exceptions::PyRuntimeError::new_err("Rust logger sender lock poisoned")
    })?;
    let worker = thread::Builder::new()
        .name("babeldoc-python-log-flusher".into())
        .spawn(move || flush_worker(receiver))
        .map_err(|error| pyo3::exceptions::PyRuntimeError::new_err(error.to_string()))?;
    *slot = Some(sender);
    state.worker = Some(worker);
    Ok(())
}

fn flush_reply() -> PyResult<mpsc::Receiver<()>> {
    let (reply, wait) = mpsc::channel();
    let sender = logger().sender.lock().map_err(|_| {
        pyo3::exceptions::PyRuntimeError::new_err("Rust logger sender lock poisoned")
    })?;
    if let Some(sender) = sender.as_ref() {
        sender
            .send(Command::Flush(reply))
            .map_err(|error| pyo3::exceptions::PyRuntimeError::new_err(error.to_string()))?;
    } else {
        // No worker means there is nothing queued to flush.
        let _ = reply.send(());
    }
    Ok(wait)
}

pub fn flush() -> PyResult<()> {
    flush_reply()?
        .recv()
        .map_err(|error| pyo3::exceptions::PyRuntimeError::new_err(error.to_string()))
}

fn flush_from_python(py: Python<'_>) -> PyResult<()> {
    let wait = flush_reply()?;
    py.detach(move || wait.recv())
        .map_err(|error| pyo3::exceptions::PyRuntimeError::new_err(error.to_string()))
}

/// Release one production run's logger lease. The final active run flushes,
/// stops, and joins the process-wide worker before its Python teardown begins.
pub fn stop_and_join(py: Python<'_>) -> PyResult<()> {
    py.detach(|| {
        let mut state = run_state().lock().map_err(|_| {
            pyo3::exceptions::PyRuntimeError::new_err("Rust logger run lock poisoned")
        })?;
        if state.active_runs > 0 {
            state.active_runs -= 1;
        }
        if state.active_runs != 0 {
            return Ok(());
        }
        if let Some(handle) = state.worker.take() {
            // Detach the sender and enqueue Stop under its lock so concurrent
            // flushes cannot enqueue behind Stop. Stop drains preceding records.
            let sender = logger()
                .sender
                .lock()
                .map_err(|_| {
                    pyo3::exceptions::PyRuntimeError::new_err("Rust logger sender lock poisoned")
                })?
                .take();
            let (reply, wait) = mpsc::channel();
            let sent = sender
                .ok_or_else(|| {
                    pyo3::exceptions::PyRuntimeError::new_err("Rust logger worker has no sender")
                })
                .and_then(|sender| {
                    sender.send(Command::Stop(reply)).map_err(|error| {
                        pyo3::exceptions::PyRuntimeError::new_err(error.to_string())
                    })
                });
            let joined = handle.join();
            sent?;
            wait.recv()
                .map_err(|error| pyo3::exceptions::PyRuntimeError::new_err(error.to_string()))?;
            joined.map_err(|_| {
                pyo3::exceptions::PyRuntimeError::new_err("Rust logger worker panicked")
            })?;
        }
        Ok(())
    })
}

#[pyfunction]
fn acquire_logging_run(py: Python<'_>) -> PyResult<()> {
    acquire_run(py)
}

#[pyfunction]
fn stop_logging_run(py: Python<'_>) -> PyResult<()> {
    stop_and_join(py)
}

#[pyfunction]
fn flush_logging_run(py: Python<'_>) -> PyResult<()> {
    flush_from_python(py)
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(acquire_logging_run, module)?)?;
    module.add_function(wrap_pyfunction!(stop_logging_run, module)?)?;
    module.add_function(wrap_pyfunction!(flush_logging_run, module)?)?;
    Ok(())
}

fn flush_worker(receiver: mpsc::Receiver<Command>) {
    loop {
        let Ok(first) = receiver.recv() else {
            return;
        };
        let mut flush_requested = matches!(first, Command::Flush(_));
        let mut batch = vec![first];
        let deadline = Instant::now() + Duration::from_millis(50);
        while !flush_requested {
            match receiver.recv_timeout(deadline.saturating_duration_since(Instant::now())) {
                Ok(command) => {
                    flush_requested = matches!(command, Command::Flush(_));
                    batch.push(command);
                }
                Err(mpsc::RecvTimeoutError::Timeout) => break,
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    process_batch(batch);
                    return;
                }
            }
        }
        while let Ok(command) = receiver.try_recv() {
            batch.push(command);
        }
        let stopping = batch
            .iter()
            .any(|command| matches!(command, Command::Stop(_)));
        process_batch(batch);
        if stopping {
            return;
        }
    }
}

fn process_batch(batch: Vec<Command>) {
    let _ = Python::try_attach(|py| {
        for command in batch {
            match command {
                Command::Record(record) => deliver(py, &record),
                Command::Flush(reply) => {
                    let _ = reply.send(());
                }
                Command::Stop(reply) => {
                    let _ = reply.send(());
                }
            }
        }
    });
}

fn deliver(py: Python<'_>, record: &QueuedRecord) {
    let result = (|| -> PyResult<()> {
        let logging = PyModule::import(py, "logging")?;
        let logger = logging.call_method1("getLogger", (&record.target,))?;
        let level = match record.level {
            Level::Error => 40,
            Level::Warn => 30,
            Level::Info => 20,
            Level::Debug | Level::Trace => 10,
        };
        let python_record = logging.call_method1(
            "LogRecord",
            (&record.target, level, "", 0, &record.message, (), py.None()),
        )?;
        python_record.setattr("created", record.created)?;
        logger.call_method1("handle", (python_record,))?;
        Ok(())
    })();
    if let Err(error) = result {
        eprintln!("BabelDOC Python log delivery failed: {error}");
    }
}
