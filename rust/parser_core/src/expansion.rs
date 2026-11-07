//! Serial expansion-root state. Each caller owns and drops its own instance.
use lopdf::{
    Error, ObjectId,
    resource::{Limit, Policy},
};
use std::collections::{HashMap, HashSet};

pub(super) struct Expansion {
    policy: Option<Policy>,
    scope: &'static str,
    limit: Limit,
    // Each root has one lexer dialect. Keys are original document/program
    // identities, never decoded allocation addresses.
    sources: HashMap<usize, usize>,
    source_weight: usize,
    validated: HashSet<usize>,
    font_edges: HashSet<usize>,
    clip_nodes: HashSet<usize>,
    text_clips: HashSet<usize>,
    work: usize,
    decoded_bytes: usize,
}

impl Expansion {
    pub fn new(policy: Option<Policy>, scope: &'static str) -> Self {
        let policy = policy.map(|p| {
            if scope == "form" {
                p.for_form_expansion()
            } else {
                p
            }
        });
        Self {
            policy: policy.filter(|p| !p.is_disabled()),
            scope,
            limit: if scope == "page_tree" {
                Limit::PageTreePageAmplification
            } else {
                Limit::ReferenceExpansionAmplification
            },
            sources: HashMap::new(),
            source_weight: 0,
            validated: HashSet::new(),
            font_edges: HashSet::new(),
            clip_nodes: HashSet::new(),
            text_clips: HashSet::new(),
            work: 0,
            decoded_bytes: 0,
        }
    }
    pub fn decode(
        &mut self,
        doc: &lopdf::Document,
        _key: usize,
        stream: &lopdf::Stream,
    ) -> lopdf::Result<std::rc::Rc<Vec<u8>>> {
        // The caller's frame owns these bytes. Distinct siblings must not retain
        // one another's decoded buffers for the lifetime of the expansion root.
        Ok(std::rc::Rc::new(doc.decode_stream(stream)?))
    }
    /// A successful validation is a zero-payload identity memo. The lexer and
    /// its operation results are still invocation-local.
    #[cfg(test)]
    pub fn validate_program(&mut self, key: usize, bytes: &[u8], object: Option<ObjectId>) -> crate::Result<()> {
        self.decoded(bytes.len(), object)?;
        self.program(key, self.weight(bytes.len(), object)?, object)?;
        self.validate_entered_program(key, bytes)
    }
    /// Validate a program whose decode, source weight, and entry work were
    /// already charged by `enter_program`.
    pub fn validate_entered_program(&mut self, key: usize, bytes: &[u8]) -> crate::Result<()> {
        if self.validated.contains(&key) {
            return Ok(());
        }
        let mut lexer = crate::tokens::Lexer::new(bytes);
        while lexer.operation()?.is_some() {}
        self.validated.insert(key);
        Ok(())
    }
    pub fn enter_program(&mut self, key: usize, bytes: usize, object: Option<ObjectId>) -> lopdf::Result<()> {
        self.decoded(bytes, object)?;
        let weight = self.weight(bytes, object)?;
        self.program(key, weight, object)?;
        self.add_work(weight, object)
    }
    pub fn enter_reference_program(&mut self, id: ObjectId, object: Option<ObjectId>) -> lopdf::Result<()> {
        // Page fallback may retain dangling references whose value cannot be
        // resolved. The xref number remains their stable original identity.
        let key = (1usize << (usize::BITS - 1))
            | ((id.0 as usize) << 16)
            | id.1 as usize;
        self.enter_program(key, 0, object)
    }
    pub fn program(&mut self, key: usize, weight: usize, object: Option<ObjectId>) -> lopdf::Result<()> {
        if self.policy.is_none() || self.sources.contains_key(&key) {
            return Ok(());
        }
        let next = self.source_weight.checked_add(weight)
            .ok_or_else(|| self.overflow(self.limit, object))?;
        self.sources.insert(key, weight);
        self.source_weight = next;
        self.check(self.work, object)
    }
    pub fn work(&mut self, amount: usize, object: Option<ObjectId>) -> lopdf::Result<()> {
        self.add_work(amount, object)
    }
    pub fn structural_program(&mut self, key: usize, slots: usize, object: Option<ObjectId>) -> lopdf::Result<()> {
        let weight = self.weight(slots, object)?;
        self.program(key, weight, object)
    }
    pub fn font_edge(&mut self, key: usize, object: Option<ObjectId>) -> lopdf::Result<()> {
        if self.policy.is_none() {
            return Ok(());
        }
        if self.font_edges.insert(key) {
            let next = self.source_weight.checked_add(1)
                .ok_or_else(|| self.overflow(self.limit, object))?;
            self.source_weight = next;
        }
        self.add_work(1, object)
    }
    pub fn clip_node(
        &mut self,
        key: usize,
        weight: usize,
        object: Option<ObjectId>,
    ) -> lopdf::Result<()> {
        if self.policy.is_none() {
            return Ok(());
        }
        if self.clip_nodes.insert(key) {
            let next = self.source_weight.checked_add(weight)
                .ok_or_else(|| self.overflow(self.limit, object))?;
            self.source_weight = next;
        }
        self.add_work(weight, object)
    }
    pub fn text_clip(
        &mut self,
        key: usize,
        weight: usize,
        object: Option<ObjectId>,
    ) -> lopdf::Result<()> {
        if self.policy.is_none() {
            return Ok(());
        }
        if self.text_clips.insert(key) {
            let next = self.source_weight.checked_add(weight)
                .ok_or_else(|| self.overflow(self.limit, object))?;
            self.source_weight = next;
        }
        self.add_work(weight, object)
    }
    fn add_work(&mut self, amount: usize, object: Option<ObjectId>) -> lopdf::Result<()> {
        if self.policy.is_none() {
            return Ok(());
        }
        let next = self.work.checked_add(amount)
            .ok_or_else(|| self.overflow(self.limit, object))?;
        self.check(next, object)?;
        self.work = next;
        Ok(())
    }
    fn decoded(&mut self, amount: usize, object: Option<ObjectId>) -> lopdf::Result<()> {
        if self.policy.is_none() || amount == 0 {
            return Ok(());
        }
        let next = self
            .decoded_bytes
            .checked_add(amount)
            .ok_or_else(|| self.overflow(Limit::RootDecodedBytes, object))?;
        if let Some(policy) = self.policy {
            policy
                .check(Limit::RootDecodedBytes, next)
                .map_err(|mut error| {
                    if let Error::ResourceLimitExceeded(e) = &mut error {
                        e.scope = self.scope;
                        e.object_ref = object;
                    }
                    error
                })?;
        }
        self.decoded_bytes = next;
        Ok(())
    }
    fn check(&self, work: usize, object: Option<ObjectId>) -> lopdf::Result<()> {
        if let Some(policy) = self.policy {
            let source_weight = self.source_weight.max(1);
            let ratio = work / source_weight + usize::from(work % source_weight != 0);
            policy.check(self.limit, ratio).map_err(|mut error| {
                if let Error::ResourceLimitExceeded(e) = &mut error {
                    e.scope = self.scope;
                    e.object_ref = object;
                }
                error
            })?;
        }
        Ok(())
    }
    pub fn depth(&self, depth: usize, object: Option<ObjectId>) -> lopdf::Result<()> {
        self.check_depth(depth, object)
    }
    fn check_depth(&self, depth: usize, object: Option<ObjectId>) -> lopdf::Result<()> {
        if let Some(policy) = self.policy {
            policy.check(Limit::ReferenceExpansionDepth, depth).map_err(|mut error| {
                if let Error::ResourceLimitExceeded(e) = &mut error {
                    e.scope = self.scope;
                    e.object_ref = object;
                }
                error
            })?;
        }
        Ok(())
    }
    pub fn child_depth(&self, depth: usize, object: Option<ObjectId>) -> lopdf::Result<usize> {
        let next = depth.checked_add(1).ok_or_else(|| {
            if self.policy.is_some() {
                self.overflow(Limit::ReferenceExpansionDepth, object)
            } else {
                Error::NumericCast("reference depth overflow".into())
            }
        })?;
        self.check_depth(next, object)?;
        Ok(next)
    }
    fn overflow(&self, kind: Limit, object: Option<ObjectId>) -> Error {
        // A counter can overflow only after reaching every representable value.
        let mut error = self.policy.unwrap().overflow(kind);
        if let Error::ResourceLimitExceeded(e) = &mut error {
            e.scope = self.scope;
            e.object_ref = object;
        }
        error
    }
    fn weight(&self, bytes: usize, object: Option<ObjectId>) -> lopdf::Result<usize> {
        bytes.checked_add(1).ok_or_else(|| {
            if self.policy.is_some() {
                self.overflow(self.limit, object)
            } else {
                Error::NumericCast("expansion weight overflow".into())
            }
        })
    }
}

#[cfg(test)]
mod expansion_tests;
