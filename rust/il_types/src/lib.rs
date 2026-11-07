pub mod python_value;
pub use python_value::PythonValue;
use std::collections::HashMap;
use std::sync::Arc;

use serde::{Deserialize, Serialize, Serializer};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum RuntimeBool {
    Bool(bool),
    Int(i64),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum CurveDebugInfo {
    Bool(bool),
    Text(String),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum PassthroughArg {
    Text(String),
    CtmAware(CtmAwareArg),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CtmAwareArg {
    pub value: String,
    pub ctm: [f64; 6],
}

#[derive(Clone, Debug)]
pub struct PassthroughNode {
    pub wire_id: u64,
    pub parent: Option<Arc<PassthroughNode>>,
    pub op: String,
    pub arg: PassthroughArg,
    pub replace_key: Option<String>,
    pub version: u64,
    pub length: u64,
}

impl Drop for PassthroughNode {
    fn drop(&mut self) {
        let mut parent = self.parent.take();
        while let Some(arc) = parent {
            match Arc::try_unwrap(arc) {
                Ok(mut node) => {
                    parent = node.parent.take();
                    drop(node);
                }
                Err(shared) => {
                    drop(shared);
                    break;
                }
            }
        }
    }
}

#[cfg(test)]
mod passthrough_stack_tests;

#[derive(Clone, Debug)]
pub struct PassthroughView {
    pub tail_ref: Option<u64>,
    pub tail: Option<Arc<PassthroughNode>>,
    pub latest_versions: Vec<(String, u64)>,
    pub include_clipping: bool,
    pub suffix_parts: Vec<String>,
}

impl Serialize for PassthroughView {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        #[derive(Serialize)]
        struct ViewWire<'a> {
            kind: &'static str,
            tail_ref: Option<u64>,
            latest_versions: &'a [(String, u64)],
            include_clipping: bool,
            suffix_parts: &'a [String],
        }
        ViewWire {
            kind: "view",
            tail_ref: self.tail_ref,
            latest_versions: &self.latest_versions,
            include_clipping: self.include_clipping,
            suffix_parts: &self.suffix_parts,
        }
        .serialize(serializer)
    }
}

#[derive(Clone, Debug, Serialize)]
#[serde(untagged)]
pub enum PassthroughValue {
    Materialized(String),
    View(PassthroughView),
}

impl PassthroughView {
    pub fn write_to(&self, output: &mut String, target_ctm_for_extgstate: Option<[f64; 6]>) {
        let mut chain = Vec::<&PassthroughNode>::new();
        let mut node = self.tail.as_deref();
        while let Some(current) = node {
            chain.push(current);
            node = current.parent.as_deref();
        }
        let latest: HashMap<&str, u64> = self
            .latest_versions
            .iter()
            .map(|(key, version)| (key.as_str(), *version))
            .collect();
        let mut needs_space = false;
        for current in chain.into_iter().rev() {
            if current
                .replace_key
                .as_deref()
                .is_some_and(|key| latest.get(key).copied() != Some(current.version))
            {
                continue;
            }
            if !self.include_clipping && matches!(current.op.as_str(), "W n" | "W* n") {
                continue;
            }
            if current.op == "gs" {
                if let (PassthroughArg::CtmAware(arg), Some(target)) =
                    (&current.arg, target_ctm_for_extgstate)
                {
                    if arg.ctm != target {
                        if let Some(to_source) = multiply(arg.ctm, invert(target)) {
                            if !is_identity(to_source) {
                                if let Some(to_target) = invert(to_source) {
                                    append_token(
                                        output,
                                        &mut needs_space,
                                        &matrix_to_pdf(to_source),
                                    );
                                    append_token(
                                        output,
                                        &mut needs_space,
                                        &format!("{} gs", arg.value),
                                    );
                                    append_token(
                                        output,
                                        &mut needs_space,
                                        &matrix_to_pdf(to_target),
                                    );
                                    continue;
                                }
                            }
                        }
                    }
                }
            }
            match &current.arg {
                PassthroughArg::Text(arg) => {
                    append_token(output, &mut needs_space, &format!("{arg} {}", current.op))
                }
                PassthroughArg::CtmAware(arg) => append_token(
                    output,
                    &mut needs_space,
                    &format!("{} {}", arg.value, current.op),
                ),
            }
        }
        for suffix in &self.suffix_parts {
            append_token(output, &mut needs_space, suffix);
        }
    }
}

fn append_token(output: &mut String, needs_space: &mut bool, token: &str) {
    if token.is_empty() {
        return;
    }
    if *needs_space {
        output.push(' ');
    }
    output.push_str(token);
    *needs_space = true;
}
fn invert(m: [f64; 6]) -> Option<[f64; 6]> {
    let [a, b, c, d, e, f] = m;
    let det = a * d - b * c;
    if det.abs() <= 1e-12 {
        return None;
    }
    Some([
        d / det,
        -b / det,
        -c / det,
        a / det,
        (c * f - d * e) / det,
        (b * e - a * f) / det,
    ])
}
fn multiply(l: [f64; 6], r: Option<[f64; 6]>) -> Option<[f64; 6]> {
    let [a1, b1, c1, d1, e1, f1] = l;
    let [a0, b0, c0, d0, e0, f0] = r?;
    Some([
        a1 * a0 + c1 * b0,
        b1 * a0 + d1 * b0,
        a1 * c0 + c1 * d0,
        b1 * c0 + d1 * d0,
        a1 * e0 + c1 * f0 + e1,
        b1 * e0 + d1 * f0 + f1,
    ])
}
fn is_identity(m: [f64; 6]) -> bool {
    m.iter()
        .zip([1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
        .all(|(a, b)| (a - b).abs() <= 1e-9)
}
fn matrix_to_pdf(m: [f64; 6]) -> String {
    format!(
        "{:.6} {:.6} {:.6} {:.6} {:.6} {:.6} cm",
        m[0], m[1], m[2], m[3], m[4], m[5]
    )
}

#[derive(Clone, Debug)]
pub struct Shared<T> {
    value: Arc<T>,
}

impl<T: Serialize> Serialize for Shared<T> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.value.serialize(serializer)
    }
}

impl<T> Shared<T> {
    pub fn new(value: T) -> Self {
        Self {
            value: Arc::new(value),
        }
    }
    pub fn as_ref(&self) -> &T {
        &self.value
    }
    pub fn arc(&self) -> Arc<T> {
        self.value.clone()
    }
    pub fn same_identity(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.value, &other.value)
    }
}

pub mod schema {
    include!(concat!(env!("OUT_DIR"), "/il_types.rs"));
}
pub use schema::*;
