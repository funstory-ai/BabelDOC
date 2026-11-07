//! Async priority admission gate used before translator service requests.
use std::{
    collections::BTreeSet,
    sync::{Arc, Mutex},
};

use tokio::sync::Notify;

#[derive(Clone)]
pub struct PriorityGate {
    inner: Arc<Inner>,
}

struct Inner {
    limit: usize,
    state: Mutex<State>,
    changed: Notify,
}

#[derive(Default)]
struct State {
    next_sequence: u64,
    active: usize,
    peak_active: usize,
    waiting: BTreeSet<(i64, u64)>,
    admission_order: Vec<(i64, u64)>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GateReadings {
    pub in_flight: usize,
    pub peak_in_flight: usize,
    pub queued: usize,
    pub admission_order: Vec<(i64, u64)>,
}

impl PriorityGate {
    pub fn new(limit: usize) -> Self {
        Self {
            inner: Arc::new(Inner {
                limit: limit.max(1),
                state: Mutex::new(State::default()),
                changed: Notify::new(),
            }),
        }
    }

    /// Lower priorities are admitted first; equal priorities use FIFO sequence.
    pub async fn acquire(&self, priority: i64) -> GateLease {
        let key = {
            let mut state = self.inner.state.lock().unwrap_or_else(|e| e.into_inner());
            let key = (priority, state.next_sequence);
            state.next_sequence += 1;
            state.waiting.insert(key);
            key
        };
        let mut ticket = GateTicket {
            inner: Arc::clone(&self.inner),
            key,
            admitted: false,
        };
        loop {
            let notified = self.inner.changed.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            let admitted = {
                let mut state = self.inner.state.lock().unwrap_or_else(|e| e.into_inner());
                if state.active < self.inner.limit && state.waiting.first() == Some(&key) {
                    state.waiting.remove(&key);
                    state.active += 1;
                    state.peak_active = state.peak_active.max(state.active);
                    state.admission_order.push(key);
                    true
                } else {
                    false
                }
            };
            if admitted {
                ticket.admitted = true;
                return GateLease { _ticket: ticket };
            }
            notified.await;
        }
    }

    pub fn readings(&self) -> GateReadings {
        let state = self.inner.state.lock().unwrap_or_else(|e| e.into_inner());
        GateReadings {
            in_flight: state.active,
            peak_in_flight: state.peak_active,
            queued: state.waiting.len(),
            admission_order: state.admission_order.clone(),
        }
    }
}

pub struct GateLease {
    _ticket: GateTicket,
}

struct GateTicket {
    inner: Arc<Inner>,
    key: (i64, u64),
    admitted: bool,
}

impl Drop for GateTicket {
    fn drop(&mut self) {
        let mut state = self.inner.state.lock().unwrap_or_else(|e| e.into_inner());
        if self.admitted {
            state.active -= 1;
        } else {
            state.waiting.remove(&self.key);
        }
        drop(state);
        self.inner.changed.notify_waiters();
    }
}

#[cfg(test)]
mod priority_gate_tests;
