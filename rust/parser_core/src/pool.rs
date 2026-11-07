//! Parser-local lookup reuses IR instructions as the full equality witness.
use super::*;
use ir::Instructions;
use std::hash::{Hash, Hasher};
#[derive(Default)]
pub(super) struct GraphicPool {
    first: HashMap<u64, Edge>,
    collisions: HashMap<u64, Vec<Edge>>,
    pub ocr_black: Option<Edge>,
}
impl GraphicPool {
    pub fn hash(instruction: &Instructions) -> u64 {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        match instruction {
            Instructions::Materialized(text) => {
                0u8.hash(&mut h);
                text.hash(&mut h);
            }
            Instructions::View(v) => {
                1u8.hash(&mut h);
                v.tail.hash(&mut h);
                v.include_clipping.hash(&mut h);
                v.suffix_parts.hash(&mut h);
            }
        }
        h.finish()
    }
    fn equal(a: &Instructions, b: &Instructions) -> bool {
        match (a, b) {
            (Instructions::Materialized(a), Instructions::Materialized(b)) => a == b,
            (Instructions::View(a), Instructions::View(b)) => {
                a.tail == b.tail
                    && a.include_clipping == b.include_clipping
                    && a.suffix_parts == b.suffix_parts
            }
            _ => false,
        }
    }
    pub fn find(&self, table: &ir::Table, instruction: &Instructions) -> Option<&Edge> {
        self.find_hash(table, instruction, Self::hash(instruction))
    }
    fn find_hash(&self, table: &ir::Table, instruction: &Instructions, hash: u64) -> Option<&Edge> {
        self.first
            .get(&hash)
            .into_iter()
            .chain(self.collisions.get(&hash).into_iter().flatten())
            .find(|e| {
                Self::equal(
                    table
                        .graphic_state_instructions(e)
                        .expect("pool entries are graphic states"),
                    instruction,
                )
            })
    }
    pub fn insert(&mut self, hash: u64, edge: Edge) {
        match self.first.entry(hash) {
            std::collections::hash_map::Entry::Vacant(v) => {
                v.insert(edge);
            }
            std::collections::hash_map::Entry::Occupied(_) => {
                self.collisions.entry(hash).or_default().push(edge);
            }
        }
    }
}
#[cfg(test)]
mod pool_tests;
