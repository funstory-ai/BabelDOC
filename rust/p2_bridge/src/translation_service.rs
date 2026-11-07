//! Translator requests use the E2 service executor plus an equal-capacity priority gate.
use crate::{
    callbacks::{CallbackError, CallbackService, PythonService, Request},
    priority_gate::PriorityGate,
};
use babeldoc_p2_stages::runtime::shared_context::SharedContextRoot;
use pyo3::{
    prelude::*,
    types::{PyDict, PyTuple},
};
use std::sync::Arc;

pub struct TranslatorPoolPair {
    service: PythonService,
    stage: String,
    gate: PriorityGate,
    shared_context: SharedContextRoot,
}

impl TranslatorPoolPair {
    /// `workers` must match the Python `ServiceExecutor` pool size for this service.
    pub fn new(
        service: PythonService,
        stage: impl Into<String>,
        workers: usize,
        shared_context: SharedContextRoot,
    ) -> Self {
        Self {
            service,
            stage: stage.into(),
            gate: PriorityGate::new(workers),
            shared_context,
        }
    }

    pub fn shared_context(&self) -> &SharedContextRoot {
        &self.shared_context
    }

    pub async fn request(
        &self,
        method: &str,
        job: impl Into<String>,
        args: Py<PyTuple>,
        kwargs: Option<Py<PyDict>>,
        priority: i64,
        admitted: impl FnOnce() -> bool + Send,
    ) -> Result<Option<Py<PyAny>>, CallbackError> {
        let _permit = self.gate.acquire(priority).await;
        if !admitted() {
            return Ok(None);
        }
        self.service
            .request(Request {
                stage: self.stage.clone(),
                job: job.into(),
                service: "translator".into(),
                method: method.into(),
                args,
                kwargs,
            })
            .await
            .map(Some)
    }

    pub fn readings(&self) -> crate::priority_gate::GateReadings {
        self.gate.readings()
    }
}

pub type SharedTranslatorPoolPair = Arc<TranslatorPoolPair>;
