//! Per-run FIFO monitor delivery. CPU workers only touch shared Rust state.
use super::BridgeError;
use babeldoc_p2_stages::{ProgressReporter, StageError};
use pyo3::prelude::*;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, mpsc};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

#[derive(Clone, Debug)]
pub enum ProgressError {
    Python(Arc<PyErr>),
    Bridge(Arc<str>),
    Cancelled,
}
impl From<StageError> for ProgressError {
    fn from(error: StageError) -> Self {
        match error {
            StageError::Cancelled => Self::Cancelled,
            other => Self::bridge(other.to_string()),
        }
    }
}
impl ProgressError {
    fn bridge(message: impl Into<String>) -> Self {
        Self::Bridge(message.into().into())
    }
    pub fn into_python(self) -> PyErr {
        Python::attach(|py| match self {
            Self::Python(error) => error.clone_ref(py),
            Self::Bridge(message) => BridgeError::new_err(message.to_string()),
            Self::Cancelled => match py
                .import("asyncio")
                .and_then(|m| m.getattr("CancelledError"))
                .and_then(|c| c.call0())
            {
                Ok(value) => PyErr::from_value(value),
                Err(error) => error,
            },
        })
    }
}
impl From<PyErr> for ProgressError {
    fn from(error: PyErr) -> Self {
        Self::Python(Arc::new(error))
    }
}

struct State {
    completed: AtomicU64,
    failed: AtomicBool,
    closed: AtomicBool,
    error: Mutex<Option<ProgressError>>,
    total: u64,
}
impl State {
    fn checkpoint(&self) -> Result<(), ProgressError> {
        if self.failed.load(Ordering::Acquire) {
            if let Some(error) = self
                .error
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .as_ref()
            {
                return Err(error.clone());
            }
        }
        Ok(())
    }
    fn save(&self, error: ProgressError) -> ProgressError {
        let mut saved = self.error.lock().unwrap_or_else(|e| e.into_inner());
        let error = saved.get_or_insert(error).clone();
        self.failed.store(true, Ordering::Release);
        error
    }
}
type Reply = mpsc::Sender<Result<(), ProgressError>>;
enum Command {
    Start {
        monitor: Py<PyAny>,
        name: String,
        state: Arc<State>,
        interval: Duration,
        reply: Reply,
    },
    Barrier(Arc<State>, Reply),
    SmallAdvance(Arc<State>, u64, Reply),
    End(Arc<State>, Option<ProgressError>, Reply),
    Finish(Arc<State>, Py<PyAny>, Reply),
    Stop,
}

/// Clone this handle for serial or parallel work. No method here acquires the GIL.
#[derive(Clone)]
pub struct PythonProgress {
    state: Arc<State>,
    sender: mpsc::Sender<Command>,
}
impl PythonProgress {
    fn barrier(&self, make: impl FnOnce(Reply) -> Command) -> Result<(), ProgressError> {
        let (sender, receiver) = mpsc::channel();
        self.sender
            .send(make(sender))
            .map_err(|e| ProgressError::bridge(e.to_string()))?;
        receiver
            .recv()
            .map_err(|e| ProgressError::bridge(e.to_string()))?
    }
    fn updated(&self, completed: u64, amount: u64, first: bool) -> Result<(), ProgressError> {
        if self.state.total <= 3 {
            self.barrier(|reply| Command::SmallAdvance(self.state.clone(), amount, reply))
        } else if first || completed >= self.state.total {
            // Preserve the unthrottled first update without calling Python here.
            self.barrier(|reply| Command::Barrier(self.state.clone(), reply))
        } else {
            self.state.checkpoint()
        }
    }
    pub fn completed(&self) -> u64 {
        self.state.completed.load(Ordering::Acquire)
    }
    pub fn end(&self, error: Option<ProgressError>) -> Result<(), ProgressError> {
        self.barrier(|reply| Command::End(self.state.clone(), error, reply))
    }
    pub fn finish(&self, result: Py<PyAny>) -> Result<(), ProgressError> {
        self.barrier(|reply| Command::Finish(self.state.clone(), result, reply))
    }
}
impl ProgressReporter for PythonProgress {
    type Error = ProgressError;
    fn report(&self, amount: u64, _total: Option<u64>) -> Result<(), ProgressError> {
        self.checkpoint()?;
        let previous = self.state.completed.fetch_add(amount, Ordering::AcqRel);
        self.updated(previous + amount, amount, previous == 0 && amount > 0)
    }
    fn checkpoint(&self) -> Result<(), ProgressError> {
        self.state.checkpoint()?;
        if self.state.closed.load(Ordering::Acquire) {
            return Err(ProgressError::bridge("progress stage already ended"));
        }
        Ok(())
    }
}

struct Stage {
    state: Arc<State>,
    monitor: Py<PyAny>,
    stage: Py<PyAny>,
    delivered: u64,
    interval: Duration,
    next: Instant,
}
impl Stage {
    fn flush(&mut self) -> Result<(), ProgressError> {
        self.state.checkpoint()?;
        let completed = self.state.completed.load(Ordering::Acquire);
        if self.state.total > 3 && completed > self.delivered {
            let result = Python::attach(|py| {
                self.stage
                    .bind(py)
                    .call_method1("advance", (completed - self.delivered,))
                    .map(|_| ())
            });
            self.delivered = completed;
            self.next = Instant::now() + self.interval;
            if let Err(error) = result {
                return Err(self.state.save(error.into()));
            }
        }
        Ok(())
    }
    fn end(&mut self, primary: Option<ProgressError>) -> Result<(), ProgressError> {
        let flushed = self.flush();
        let result = flushed.and(primary.map_or(Ok(()), Err));
        // Even when unwinding, stage end is delivered after every earlier update.
        let exited = Python::attach(|py| -> PyResult<()> {
            match &result {
                Ok(()) => {
                    self.stage
                        .bind(py)
                        .call_method1("__exit__", (py.None(), py.None(), py.None()))?;
                }
                Err(error) => {
                    let error = error.clone().into_python();
                    self.stage.bind(py).call_method1(
                        "__exit__",
                        (error.get_type(py), error.value(py), error.traceback(py)),
                    )?;
                }
            }
            Ok(())
        });
        self.state.closed.store(true, Ordering::Release);
        match exited {
            Err(error) => Err(self.state.save(error.into())),
            Ok(()) => result,
        }
    }
}

static ALIVE: AtomicUsize = AtomicUsize::new(0);
pub fn progress_threads_alive() -> usize {
    ALIVE.load(Ordering::Acquire)
}
struct Alive;
impl Drop for Alive {
    fn drop(&mut self) {
        ALIVE.fetch_sub(1, Ordering::AcqRel);
    }
}

fn find<'a>(stages: &'a mut [Stage], state: &Arc<State>) -> Result<&'a mut Stage, ProgressError> {
    stages
        .iter_mut()
        .find(|stage| Arc::ptr_eq(&stage.state, state))
        .ok_or_else(|| ProgressError::bridge("unknown progress stage"))
}
fn deliver(receiver: mpsc::Receiver<Command>) {
    ALIVE.fetch_add(1, Ordering::AcqRel);
    let _alive = Alive;
    let mut stages: Vec<Stage> = Vec::new();
    loop {
        let timeout = stages
            .iter()
            .filter(|s| !s.state.closed.load(Ordering::Acquire))
            .map(|s| s.next.saturating_duration_since(Instant::now()))
            .min()
            .unwrap_or(Duration::from_secs(60));
        match receiver.recv_timeout(timeout) {
            Ok(Command::Start {
                monitor,
                name,
                state,
                interval,
                reply,
            }) => {
                let result = Python::attach(|py| -> PyResult<Py<PyAny>> {
                    let stage = monitor
                        .bind(py)
                        .call_method1("stage_start", (name, state.total))?;
                    stage.call_method0("__enter__")?;
                    Ok(stage.unbind())
                });
                let result = match result {
                    Ok(stage) => {
                        stages.push(Stage {
                            state,
                            monitor,
                            stage,
                            delivered: 0,
                            interval,
                            next: Instant::now() + interval,
                        });
                        Ok(())
                    }
                    Err(error) => Err(state.save(error.into())),
                };
                let _ = reply.send(result);
            }
            Ok(Command::Barrier(state, reply)) => {
                let _ = reply.send(find(&mut stages, &state).and_then(Stage::flush));
            }
            Ok(Command::SmallAdvance(state, amount, reply)) => {
                let result = find(&mut stages, &state).and_then(|stage| {
                    stage.state.checkpoint()?;
                    if amount == 0 {
                        return Ok(());
                    }
                    let result = Python::attach(|py| {
                        stage
                            .stage
                            .bind(py)
                            .call_method1("advance", (amount,))
                            .map(|_| ())
                    });
                    stage.delivered += amount;
                    result.map_err(|error| stage.state.save(error.into()))
                });
                let _ = reply.send(result);
            }
            Ok(Command::End(state, error, reply)) => {
                let _ = reply.send(find(&mut stages, &state).and_then(|stage| stage.end(error)));
            }
            Ok(Command::Finish(state, result, reply)) => {
                let result = find(&mut stages, &state).and_then(|stage| {
                    stage.state.checkpoint()?;
                    Python::attach(|py| {
                        stage
                            .monitor
                            .bind(py)
                            .call_method1("translate_done", (result,))
                            .map(|_| ())
                    })
                    .map_err(|error| stage.state.save(error.into()))
                });
                let _ = reply.send(result);
            }
            Ok(Command::Stop) | Err(mpsc::RecvTimeoutError::Disconnected) => {
                for stage in &mut stages {
                    if !stage.state.closed.load(Ordering::Acquire) {
                        if let Err(error) = stage.end(None) {
                            stage.state.save(error);
                        }
                    }
                }
                break;
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {}
        }
        let now = Instant::now();
        for stage in &mut stages {
            if !stage.state.closed.load(Ordering::Acquire) && now >= stage.next {
                let _ = stage.flush();
                stage.next = Instant::now() + stage.interval;
            }
        }
    }
}

/// Drop is a fallback for Rust unwinding; ordinary paths explicitly close and join.
pub struct ProgressRun {
    sender: mpsc::Sender<Command>,
    thread: Option<JoinHandle<()>>,
    states: Mutex<Vec<Arc<State>>>,
}
impl ProgressRun {
    pub fn new() -> PyResult<Self> {
        let (sender, receiver) = mpsc::channel();
        let thread = thread::Builder::new()
            .name("p2-progress-report".into())
            .spawn(move || deliver(receiver))
            .map_err(|error| BridgeError::new_err(error.to_string()))?;
        Ok(Self {
            sender,
            thread: Some(thread),
            states: Mutex::new(Vec::new()),
        })
    }
    pub fn start(
        &self,
        monitor: Py<PyAny>,
        name: &str,
        total: u64,
        interval: Duration,
    ) -> Result<PythonProgress, ProgressError> {
        let state = Arc::new(State {
            completed: AtomicU64::new(0),
            failed: AtomicBool::new(false),
            closed: AtomicBool::new(false),
            error: Mutex::new(None),
            total,
        });
        self.states
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push(state.clone());
        let handle = PythonProgress {
            state: state.clone(),
            sender: self.sender.clone(),
        };
        let started = handle.barrier(|reply| Command::Start {
            monitor,
            name: name.into(),
            state,
            interval: interval.max(Duration::from_millis(1)),
            reply,
        });
        if let Err(error) = started {
            // Failed entry creates no Stage and must not be unwound as one.
            handle.state.closed.store(true, Ordering::Release);
            return Err(error);
        }
        Ok(handle)
    }
    /// A caller that caught an ended stage's error may continue (writer retry
    /// or debug fallback). Keep live-stage failures pending for their own end.
    /// No handles are revived: report still rejects every closed stage.
    pub fn acknowledge_ended_errors(&self) {
        for state in self.states.lock().unwrap_or_else(|e| e.into_inner()).iter() {
            if state.closed.load(Ordering::Acquire) {
                *state.error.lock().unwrap_or_else(|e| e.into_inner()) = None;
                state.failed.store(false, Ordering::Release);
            }
        }
    }
    pub fn close(&mut self) -> PyResult<()> {
        if let Some(thread) = self.thread.take() {
            let _ = self.sender.send(Command::Stop);
            thread
                .join()
                .map_err(|_| BridgeError::new_err("progress reporter panicked"))?;
        }
        for state in self.states.lock().unwrap_or_else(|e| e.into_inner()).iter() {
            state.checkpoint().map_err(ProgressError::into_python)?;
        }
        Ok(())
    }
}
impl Drop for ProgressRun {
    fn drop(&mut self) {
        let _ = self.close();
    }
}
