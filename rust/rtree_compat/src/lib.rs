// Derived from libspatialindex 2.1.0 R*-tree algorithms.
// Copyright (c) 2002, Marios Hadjieleftheriou
//
// All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

use std::cmp::Ordering;

const CAPACITY: usize = 100;
const SPF: f64 = 0.4;
const REINSERT: f64 = 0.3;
const NEAR_MIN_OVERLAP: usize = 32;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rect {
    pub min_x: f64,
    pub min_y: f64,
    pub max_x: f64,
    pub max_y: f64,
}
impl Rect {
    pub fn new(min_x: f64, min_y: f64, max_x: f64, max_y: f64) -> Self {
        Self {
            min_x,
            min_y,
            max_x,
            max_y,
        }
    }
    fn area(self) -> f64 {
        (self.max_x - self.min_x) * (self.max_y - self.min_y)
    }
    fn margin(self) -> f64 {
        2.0 * ((self.max_x - self.min_x) + (self.max_y - self.min_y))
    }
    fn union(self, b: Self) -> Self {
        Self::new(
            self.min_x.min(b.min_x),
            self.min_y.min(b.min_y),
            self.max_x.max(b.max_x),
            self.max_y.max(b.max_y),
        )
    }
    fn overlap(self, b: Self) -> f64 {
        (self.max_x.min(b.max_x) - self.min_x.max(b.min_x)).max(0.0)
            * (self.max_y.min(b.max_y) - self.min_y.max(b.min_y)).max(0.0)
    }
    fn intersects(self, b: Self) -> bool {
        self.min_x <= b.max_x
            && self.max_x >= b.min_x
            && self.min_y <= b.max_y
            && self.max_y >= b.min_y
    }
    fn center(self) -> (f64, f64) {
        (
            (self.min_x + self.max_x) * 0.5,
            (self.min_y + self.max_y) * 0.5,
        )
    }
}
#[derive(Clone, Debug)]
enum Entry {
    Data { id: i64, rect: Rect },
    Child { node: usize, rect: Rect },
}
impl Entry {
    fn rect(&self) -> Rect {
        match self {
            Self::Data { rect, .. } | Self::Child { rect, .. } => *rect,
        }
    }
}
#[derive(Clone, Debug)]
struct Node {
    level: usize,
    entries: Vec<Entry>,
}
impl Node {
    fn mbr(&self) -> Rect {
        let mut it = self.entries.iter();
        let Some(first) = it.next() else {
            return Rect::new(f64::MAX, f64::MAX, -f64::MAX, -f64::MAX);
        };
        it.fold(first.rect(), |a, e| a.union(e.rect()))
    }
}
#[derive(Clone, Debug)]
pub struct RTree {
    nodes: Vec<Node>,
    root: usize,
    pending: Vec<(Entry, usize)>,
}
impl Default for RTree {
    fn default() -> Self {
        Self::new()
    }
}
impl RTree {
    pub fn new() -> Self {
        Self {
            nodes: vec![Node {
                level: 0,
                entries: vec![],
            }],
            root: 0,
            pending: vec![],
        }
    }
    pub fn insert(&mut self, id: i64, rect: Rect) {
        self.insert_entry(Entry::Data { id, rect }, 0);
    }
    fn insert_entry(&mut self, entry: Entry, target_level: usize) {
        let mut reinserted = vec![false; 64];
        let mut queue = vec![(entry, target_level)];
        while let Some((item, level)) = queue.pop() {
            let sibling = self.insert_at(self.root, item, level, &mut reinserted);
            if let Some(sib) = sibling {
                let old = self.root;
                let new_level = self.nodes[old].level + 1;
                let left = Entry::Child {
                    node: old,
                    rect: self.nodes[old].mbr(),
                };
                let right = Entry::Child {
                    node: sib,
                    rect: self.nodes[sib].mbr(),
                };
                let root = self.nodes.len();
                self.nodes.push(Node {
                    level: new_level,
                    entries: vec![left, right],
                });
                self.root = root;
            }
            queue.extend(self.pending.drain(..));
        }
    }
    fn insert_at(
        &mut self,
        node: usize,
        item: Entry,
        target: usize,
        seen: &mut [bool],
    ) -> Option<usize> {
        if self.nodes[node].level == target {
            if self.nodes[node].entries.len() < CAPACITY {
                self.nodes[node].entries.push(item);
                return None;
            }
            let level = self.nodes[node].level;
            if node != self.root && level < seen.len() && !seen[level] {
                seen[level] = true;
                let mut all = self.nodes[node].entries.clone();
                all.push(item);
                let (keep, reins) = self.reinsert_partition(self.nodes[node].mbr(), all);
                self.nodes[node].entries = keep;
                // Ancestor MBRs are refreshed while recursion unwinds, before pending reinserts run.
                // Queue order mirrors insertData_impl's ascending-distance reinserts.
                for x in reins.into_iter().rev() {
                    self.pending.push((x, target));
                }
                return None;
            }
            let mut all = self.nodes[node].entries.clone();
            all.push(item);
            let (a, b) = self.split(all);
            self.nodes[node].entries = a;
            let sibling = self.nodes.len();
            self.nodes.push(Node { level, entries: b });
            return Some(sibling);
        }
        let idx = self.choose_subtree(node, item.rect());
        let Entry::Child { node: child, .. } = self.nodes[node].entries[idx].clone() else {
            unreachable!()
        };
        if let Some(sib) = self.insert_at(child, item, target, seen) {
            self.nodes[node].entries[idx] = Entry::Child {
                node: child,
                rect: self.nodes[child].mbr(),
            };
            let new = Entry::Child {
                node: sib,
                rect: self.nodes[sib].mbr(),
            };
            let mut all = self.nodes[node].entries.clone();
            all.push(new);
            if all.len() <= CAPACITY {
                self.nodes[node].entries = all;
                None
            } else {
                let level = self.nodes[node].level;
                if node != self.root && level < seen.len() && !seen[level] {
                    seen[level] = true;
                    let mbr = bounds(&all);
                    let (keep, reins) = self.reinsert_partition(mbr, all);
                    self.nodes[node].entries = keep;
                    for x in reins.into_iter().rev() {
                        self.pending.push((x, level));
                    }
                    None
                } else {
                    let (a, b) = self.split(all);
                    self.nodes[node].entries = a;
                    let sibling = self.nodes.len();
                    self.nodes.push(Node { level, entries: b });
                    Some(sibling)
                }
            }
        } else {
            self.nodes[node].entries[idx] = Entry::Child {
                node: child,
                rect: self.nodes[child].mbr(),
            };
            None
        }
    }
    fn choose_subtree(&self, node: usize, r: Rect) -> usize {
        let n = &self.nodes[node];
        if n.level == 1 {
            let mut candidates: Vec<(usize, f64, Rect, Rect)> = n
                .entries
                .iter()
                .enumerate()
                .map(|(i, e)| {
                    let b = e.rect();
                    let c = b.union(r);
                    (i, c.area() - b.area(), b, c)
                })
                .collect();
            let min_enl = candidates.iter().map(|x| x.1).fold(f64::INFINITY, f64::min);
            let min_area = candidates
                .iter()
                .filter(|x| x.1 == min_enl)
                .map(|x| x.2.area())
                .fold(f64::INFINITY, f64::min);
            if min_enl.abs() <= f64::EPSILON {
                return candidates
                    .iter()
                    .find(|x| x.1 == min_enl && x.2.area() == min_area)
                    .unwrap()
                    .0;
            }
            if candidates.len() > NEAR_MIN_OVERLAP {
                candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
                candidates.truncate(NEAR_MIN_OVERLAP);
            }
            let mut best = candidates[0].0;
            let mut best_enl = candidates[0].1;
            let mut best_area = candidates[0].2.area();
            let mut least = f64::INFINITY;
            for (i, enl, original, combined) in candidates {
                let mut delta = 0.0;
                for (j, e) in n.entries.iter().enumerate() {
                    if j != i {
                        let other = e.rect();
                        let new = combined.overlap(other);
                        if new != 0.0 {
                            delta += new - original.overlap(other);
                        }
                    }
                }
                if delta < least
                    || (delta == least
                        && (enl < best_enl || (enl == best_enl && original.area() < best_area)))
                {
                    least = delta;
                    best = i;
                    best_enl = enl;
                    best_area = original.area();
                }
            }
            best
        } else {
            let mut best = 0;
            let mut be = f64::INFINITY;
            let mut ba = f64::INFINITY;
            for (i, e) in n.entries.iter().enumerate() {
                let b = e.rect();
                let en = b.union(r).area() - b.area();
                if en < be || (en == be && b.area() < ba) {
                    best = i;
                    be = en;
                    ba = b.area();
                }
            }
            best
        }
    }
    fn reinsert_partition(
        &self,
        node_mbr: Rect,
        mut entries: Vec<Entry>,
    ) -> (Vec<Entry>, Vec<Entry>) {
        let c = node_mbr.center();
        entries.sort_by(|a, b| {
            let ca = a.rect().center();
            let cb = b.rect().center();
            let da = (c.0 - ca.0).powi(2) + (c.1 - ca.1).powi(2);
            let db = (c.0 - cb.0).powi(2) + (c.1 - cb.1).powi(2);
            da.partial_cmp(&db).unwrap_or(Ordering::Equal)
        });
        let n = ((CAPACITY + 1) as f64 * REINSERT).floor() as usize;
        let reins = entries.split_off(entries.len() - n);
        (entries, reins)
    }
    fn split(&self, entries: Vec<Entry>) -> (Vec<Entry>, Vec<Entry>) {
        let node_spf = ((CAPACITY + 1) as f64 * SPF).floor() as usize;
        let distributions = (CAPACITY + 1) - 2 * node_spf + 2;
        let mut min_margin = f64::INFINITY;
        let mut axis = 0;
        let mut low_order = true;
        let mut data_low = entries.clone();
        let mut data_high = entries;
        for dim in 0..2 {
            data_low.sort_by(|a, b| {
                coord(a.rect(), dim, false)
                    .partial_cmp(&coord(b.rect(), dim, false))
                    .unwrap_or(Ordering::Equal)
            });
            data_high.sort_by(|a, b| {
                coord(a.rect(), dim, true)
                    .partial_cmp(&coord(b.rect(), dim, true))
                    .unwrap_or(Ordering::Equal)
            });
            let mut ml = 0.0;
            let mut mh = 0.0;
            for k in 1..=distributions {
                let l = node_spf - 1 + k;
                ml += bounds(&data_low[..l]).margin() + bounds(&data_low[l..]).margin();
                mh += bounds(&data_high[..l]).margin() + bounds(&data_high[l..]).margin();
            }
            let m = ml.min(mh);
            if m < min_margin {
                min_margin = m;
                axis = dim;
                low_order = ml < mh;
            }
        }
        let mut ordered = data_low;
        ordered.sort_by(|a, b| {
            coord(a.rect(), axis, !low_order)
                .partial_cmp(&coord(b.rect(), axis, !low_order))
                .unwrap_or(Ordering::Equal)
        });
        let mut min_ov = f64::INFINITY;
        let mut min_area = f64::INFINITY;
        let mut point = 1;
        for k in 1..=distributions {
            let l = node_spf - 1 + k;
            let a = bounds(&ordered[..l]);
            let b = bounds(&ordered[l..]);
            let ov = a.overlap(b);
            let area = a.area() + b.area();
            if ov < min_ov || (ov == min_ov && area < min_area) {
                point = k;
                min_ov = ov;
                min_area = area;
            }
        }
        let left_count = node_spf - 1 + point;
        let right = ordered.split_off(left_count);
        (ordered, right)
    }
    pub fn intersection(&self, query: Rect) -> Vec<i64> {
        let mut out = Vec::new();
        let mut stack = vec![self.root];
        while let Some(id) = stack.pop() {
            let n = &self.nodes[id];
            for e in &n.entries {
                if e.rect().intersects(query) {
                    match e {
                        Entry::Data { id, .. } => out.push(*id),
                        Entry::Child { node, .. } => stack.push(*node),
                    }
                }
            }
        }
        out
    }
}
fn coord(r: Rect, d: usize, high: bool) -> f64 {
    match (d, high) {
        (0, false) => r.min_x,
        (0, true) => r.max_x,
        (1, false) => r.min_y,
        _ => r.max_y,
    }
}
fn bounds(e: &[Entry]) -> Rect {
    e.iter().map(Entry::rect).reduce(Rect::union).unwrap()
}
