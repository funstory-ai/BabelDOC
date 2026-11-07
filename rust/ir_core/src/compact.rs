// Stage-boundary collection. No logical objects are merged or reidentified.

const UNREACHABLE: usize = usize::MAX;

/// Rank among reachable old keys, without another word for every node.
/// The directory remains in old-key order until every edge has been relocated.
enum Relocation {
    Identity,
    Ranked {
        words: Vec<u64>,
        prefixes: Vec<usize>,
    },
}
impl Relocation {
    fn from_owners(owners: &[usize]) -> Self {
        if owners.iter().all(|&owner| owner != UNREACHABLE) {
            return Self::Identity;
        }
        let mut words = vec![0u64; owners.len().div_ceil(64)];
        for (key, &owner) in owners.iter().enumerate() {
            if owner != UNREACHABLE {
                words[key / 64] |= 1u64 << (key % 64);
            }
        }
        let mut count = 0;
        let prefixes = words
            .iter()
            .map(|word| {
                let previous = count;
                count += word.count_ones() as usize;
                previous
            })
            .collect();
        Self::Ranked { words, prefixes }
    }
    fn index(&self, key: usize) -> usize {
        match self {
            Self::Identity => key,
            Self::Ranked { words, prefixes } => {
                let word = words[key / 64];
                let bit = key % 64;
                assert_ne!(word & (1u64 << bit), 0, "live edge must target a live node");
                prefixes[key / 64] + (word & ((1u64 << bit) - 1)).count_ones() as usize
            }
        }
    }
}

impl Document {
    /// Collect and repartition the graph after a complete stage has returned.
    /// All stage-local cursors and prepared field values must have been dropped.
    /// Stored identities, instruction versions and every logical edge survive.
    pub fn compact(&mut self) {
        self.generation = next_generation();
        let mut owners = vec![UNREACHABLE; self.table.nodes.len()];
        let mut page_scopes = HashMap::new();
        let mut work = Vec::new();
        for edge in self.root.data.edges() {
            edge.table_edges(&mut |key| {
                let scope = if self.table.nodes[key].kind == Kind::Page {
                    let next = page_scopes.len() + 1;
                    *page_scopes.entry(key).or_insert(next)
                } else {
                    0
                };
                work.push((key, scope));
            });
        }
        while let Some((key, mut scope)) = work.pop() {
            let previous = owners[key];
            if previous == scope || previous == 0 {
                continue;
            }
            if previous != UNREACHABLE {
                scope = 0;
            }
            owners[key] = scope;
            for edge in self.table.node(key).data.edges() {
                edge.table_edges(&mut |child| work.push((child, scope)));
            }
        }

        // Determine final scope and forward ranks before repurposing locations.
        let mut counts = vec![[0; CLASS_COUNT]; page_scopes.len() + 1];
        for (key, &scope) in owners.iter().enumerate() {
            if scope != UNREACHABLE {
                counts[scope][self.table.nodes[key].kind as usize] += 1;
            }
        }
        let indices = Relocation::from_owners(&owners);
        let old_history_base = self.history.base;
        let history_indices = self.compact_history(&owners);
        let history_base = self.history.base;
        let mut old = std::mem::take(&mut self.table);
        // Reverse physical slots while the old locations are still intact.
        // Exact capacities avoid geometric slack in this temporary lookup.
        let mut source_counts = vec![[0; CLASS_COUNT]; old.pages.len() + 1];
        for location in &old.nodes {
            source_counts[location.scope][location.kind as usize] += 1;
        }
        let mut keys: Vec<[Vec<usize>; CLASS_COUNT]> = source_counts
            .iter()
            .map(|counts| std::array::from_fn(|kind| Vec::with_capacity(counts[kind])))
            .collect();
        for (key, location) in old.nodes.iter().enumerate() {
            let group = &mut keys[location.scope][location.kind as usize];
            if group.len() <= location.index {
                group.resize(location.index + 1, UNREACHABLE);
            }
            debug_assert_eq!(group[location.index], UNREACHABLE);
            group[location.index] = key;
        }
        drop(source_counts);
        let mut directory = std::mem::take(&mut old.nodes);
        for (location, owner) in directory.iter_mut().zip(owners) {
            location.scope = owner;
        }
        // `owners` is consumed. Old-key slots now hold both the final owner and
        // (after moving that node) its new physical location. No second directory.
        self.table = Table {
            nodes: Vec::new(),
            shared: TypedTables::default(),
            pages: counts[1..].iter().map(|_| TypedTables::default()).collect(),
            staging_scope: 0,
        };
        let mut immutable_memo = HashMap::new();
        for edge in self.root.data.edges_mut() {
            edge.relocate(&indices, &mut immutable_memo);
        }
        let kinds: Vec<_> = directory
            .iter()
            .map(|l| l.kind)
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect();
        for kind in kinds {
            // A source vector whose nodes all land in one otherwise-empty
            // destination can be retained in place. In particular, parser page
            // character tables need no second allocation, even on allocators
            // which retain the pages of recently freed small vectors.
            let reuse: Vec<_> = keys
                .iter()
                .map(|source| {
                    let group = &source[kind as usize];
                    let first = *group.first()?;
                    let owner = directory[first].scope;
                    (owner != UNREACHABLE
                        && counts[owner][kind as usize] == group.len()
                        && group.iter().all(|&key| directory[key].scope == owner))
                    .then_some(owner)
                })
                .collect();
            let mut reused = vec![false; counts.len()];
            for &scope in reuse.iter().flatten() {
                reused[scope] = true;
            }
            for (scope, count) in counts.iter().enumerate() {
                if !reused[scope] {
                    self.table
                        .scope_mut(scope)
                        .reserve_kind(kind, count[kind as usize]);
                }
            }
            let mut relocate = |node: &mut Node| {
                for edge in node.data.edges_mut() {
                    edge.relocate(&indices, &mut immutable_memo);
                }
                if let NodeData::GraphicState(state) = &mut node.data
                    && let Some(Instructions::View(view)) =
                        &mut state.passthrough_per_char_instruction
                {
                    view.tail = view
                        .tail
                        .map(|key| history_base + history_indices[key - old_history_base]);
                }
            };
            for source_scope in 0..keys.len() {
                let source_keys = std::mem::take(&mut keys[source_scope][kind as usize]);
                if let Some(scope) = reuse[source_scope] {
                    self.table.scope_mut(scope).reuse_kind(
                        old.scope_mut(source_scope),
                        kind,
                        &source_keys,
                        &mut |key, index, node| {
                            relocate(node);
                            directory[key] = Location { kind, scope, index };
                        },
                    );
                } else {
                    old.scope_mut(source_scope).drain_kind(
                        kind,
                        &source_keys,
                        &mut |key, mut node| {
                            let scope = directory[key].scope;
                            if scope == UNREACHABLE {
                                return;
                            }
                            relocate(&mut node);
                            let index = self.table.scope_mut(scope).push(node);
                            directory[key] = Location { kind, scope, index };
                        },
                    );
                }
            }
        }
        // Retaining old-key order is exactly the rank mapping used by edges.
        // Shrink only after every source payload and relocation memo can drop.
        drop(old);
        drop(keys);
        drop(indices);
        drop(history_indices);
        drop(immutable_memo);
        directory.retain(|location| location.scope != UNREACHABLE);
        directory.shrink_to_fit();
        self.table.nodes = directory;
    }

    fn compact_history(&mut self, owners: &[usize]) -> Vec<usize> {
        let mut reachable = vec![false; self.history.nodes.len()];
        for (key, &owner) in owners.iter().enumerate() {
            if owner == UNREACHABLE || self.table.nodes[key].kind != Kind::GraphicState {
                continue;
            }
            let NodeDataRef::GraphicState(state) = self.table.node(key).data else {
                unreachable!()
            };
            if let Some(Instructions::View(view)) = &state.passthrough_per_char_instruction {
                let mut tail = view.tail.map(|key| {
                    self.history
                        .index(key)
                        .expect("live instruction key must be current")
                });
                while let Some(key) = tail {
                    if reachable[key] {
                        break;
                    }
                    reachable[key] = true;
                    tail = self.history.nodes[key].parent;
                }
            }
        }
        let mut indices = vec![UNREACHABLE; reachable.len()];
        let mut count = 0;
        for (key, &live) in reachable.iter().enumerate() {
            if live {
                indices[key] = count;
                count += 1;
            }
        }
        self.history.base = self
            .history
            .base
            .checked_add(self.history.nodes.len())
            .and_then(|v| v.checked_add(1))
            .expect("history key space exhausted");
        let old = std::mem::replace(&mut self.history.nodes, Vec::with_capacity(count));
        for (key, mut node) in old.into_iter().enumerate() {
            if reachable[key] {
                node.parent = node.parent.map(|parent| indices[parent]);
                self.history.nodes.push(node);
            }
        }
        indices
    }
}

#[cfg(test)]
mod table_tests;
