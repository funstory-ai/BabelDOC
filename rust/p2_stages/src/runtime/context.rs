use crate::StageError;

pub trait Cancellation: Send + Sync {
    fn is_cancelled(&self) -> bool;
}

/// Shared progress handle. Implementations own counting and delivery policy.
pub trait ProgressReporter: Send + Sync {
    type Error;
    fn report(&self, amount: u64, total: Option<u64>) -> Result<(), Self::Error>;
    fn checkpoint(&self) -> Result<(), Self::Error> {
        Ok(())
    }
}

#[derive(Default)]
pub struct NoopCancellation;
impl Cancellation for NoopCancellation {
    fn is_cancelled(&self) -> bool {
        false
    }
}

#[derive(Default)]
pub struct NoopProgress;
impl ProgressReporter for NoopProgress {
    type Error = StageError;
    fn report(&self, _completed: u64, _total: Option<u64>) -> Result<(), StageError> {
        Ok(())
    }
}

pub struct StageContext<'a, E = StageError> {
    cancellation: &'a dyn Cancellation,
    progress: &'a dyn ProgressReporter<Error = E>,
}

impl<'a, E: From<StageError>> StageContext<'a, E> {
    pub fn new(
        cancellation: &'a dyn Cancellation,
        progress: &'a dyn ProgressReporter<Error = E>,
    ) -> Self {
        Self {
            cancellation,
            progress,
        }
    }

    /// Deferred callback errors and cancellation are observed without touching Python.
    pub fn check_cancelled(&self) -> Result<(), E> {
        self.progress.checkpoint()?;
        if self.cancellation.is_cancelled() {
            Err(StageError::Cancelled.into())
        } else {
            Ok(())
        }
    }

    /// Add a delta using the same shared API in serial and parallel work.
    pub fn report_progress(&self, amount: u64, total: Option<u64>) -> Result<(), E> {
        self.progress.report(amount, total)
    }
}

/// Test harness uses the same Rayon pool implementation as CPU stages.
pub fn parallel_progress_probe<E: From<StageError> + Send>(
    context: &StageContext<'_, E>,
    total: u64,
    workers: usize,
    delay_ms: u64,
) -> Result<u64, E> {
    use rayon::prelude::*;
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(workers)
        .build()
        .map_err(|error| E::from(StageError::Failed(error.to_string())))?;
    let processed = std::sync::atomic::AtomicU64::new(0);
    let result = pool.install(|| {
        (0..total).into_par_iter().try_for_each(|_| {
            context.check_cancelled()?;
            if delay_ms != 0 {
                std::thread::sleep(std::time::Duration::from_millis(delay_ms));
            }
            context.check_cancelled()?;
            processed.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            context.report_progress(1, Some(total))
        })
    });
    result.map(|_| processed.load(std::sync::atomic::Ordering::Relaxed))
}
