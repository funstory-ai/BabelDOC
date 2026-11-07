//! Lazy first-match index for the permissive Count-guided sibling scan.
//! Signed ranks are translated on the 2^64 ring, exactly as release i64 subtraction.
use super::*;
use std::collections::BTreeMap;
const MODULUS: u128 = 1u128 << 64;
const SIGN: u64 = 1u64 << 63;
fn rank(n: i64) -> u128 { ((n as u64) ^ SIGN) as u128 }
#[derive(Clone, Copy)]
pub(super) enum Action<'a> { Null, Leaf(&'a Object), Branch(&'a Object) }
#[derive(Clone, Copy)]
pub(super) struct Interval<'a> {
    pub end: u128,
    pub prefix: u64,
    pub action: Action<'a>,
}
pub(super) struct Index<'a> {
    kids: &'a [Object],
    next: usize,
    prefix: u64,
    assigned: BTreeMap<u128, Interval<'a>>,
    vacant: BTreeMap<u128, u128>,
}
impl<'a> Index<'a> {
    pub fn new(kids: &'a [Object]) -> Self {
        Self { kids, next: 0, prefix: 0, assigned: BTreeMap::new(), vacant: BTreeMap::from([(0, MODULUS)]) }
    }
    fn assign(&mut self, start: u128, end: u128, action: Action<'a>) {
        if start == end { return; }
        let mut keys = Vec::new();
        if let Some((&begin, &stop)) = self.vacant.range(..start).next_back() {
            if stop > start { keys.push(begin); }
        }
        keys.extend(self.vacant.range(start..end).map(|(&begin, _)| begin));
        for begin in keys {
            let stop = self.vacant.remove(&begin).unwrap();
            let lo = begin.max(start); let hi = stop.min(end);
            if begin < lo { self.vacant.insert(begin, lo); }
            if hi < stop { self.vacant.insert(hi, stop); }
            self.assigned.insert(lo, Interval { end: hi, prefix: self.prefix, action });
        }
    }
    fn translated(&mut self, start: u128, len: u128, action: Action<'a>) {
        if len == 0 { return; }
        let lo = (start + u128::from(self.prefix)) % MODULUS;
        let end = lo + len;
        self.assign(lo, end.min(MODULUS), action);
        if end > MODULUS { self.assign(0, end - MODULUS, action); }
    }
    pub fn query(&mut self, pdf: &Pdf, skip: i64) -> Option<Interval<'a>> {
        let point = rank(skip);
        loop {
            if let Some((&begin, &interval)) = self.assigned.range(..=point).next_back() {
                if begin <= point && point < interval.end { return Some(interval); }
            }
            let kid = self.kids.get(self.next)?;
            self.next += 1;
            if matches!(resolve(pdf, kid), Ok(Object::Null)) {
                self.assign(0, MODULUS, Action::Null);
                continue;
            }
            let empty = Dictionary::new();
            let d = dict(pdf, kid).unwrap_or(&empty);
            let pages = d.get(b"Type").ok().map_or_else(
                || d.get(b"Kids").is_ok() && d.get(b"MediaBox").is_err(),
                |t| t.as_name().ok() == Some(b"Pages"),
            );
            if pages {
                let count = d.get(b"Count").ok().and_then(numeric).unwrap_or(0.) as i64;
                self.translated(0, rank(count), Action::Branch(kid));
                self.prefix = self.prefix.wrapping_add(count as u64);
            } else {
                self.translated(rank(0), 1, Action::Leaf(kid));
                self.prefix = self.prefix.wrapping_add(1);
            }
        }
    }
}
// Each frame covers a contiguous range of global page ranks. Reusing a frame
// repeats no guards: its source and (node, n-skip) occurrence have not changed.
struct Frame<'a> {
    node: &'a Object,
    key: usize,
    offset: u64,
    end: u128,
}
pub(super) fn fallback(pdf: &Pdf, root: &Object, count: i64) -> Result<Vec<(u32, ObjectId)>> {
    fallback_observed(pdf, root, count, || {})
}
fn fallback_observed(pdf: &Pdf, root: &Object, count: i64, mut visit: impl FnMut()) -> Result<Vec<(u32, ObjectId)>> {
    let mut out = Vec::new();
    let mut expansion = expansion::Expansion::new(pdf.parser_policy, "page_tree");
    let mut indexes = HashMap::new();
    let mut frames: Vec<Frame<'_>> = Vec::new();
    let mut active = HashSet::new();
    for n in 0..count {
        while frames.last().is_some_and(|frame| frame.end <= n as u128) {
            active.remove(&frames.pop().unwrap().key);
        }
        let mut next = frames.is_empty().then_some((root, 0u64, count as u128, true));
        loop {
            if let Some((node, offset, end, _root_node)) = next.take() {
                visit();
                let value = resolve(pdf, node)?;
                let key = value as *const Object as usize;
                if !active.insert(key) { return Err(Error::new("FzErrorFormat", "cycle in page tree")); }
                let source = node.as_reference().ok();
                let skip = n.wrapping_sub(offset as i64);
                expansion.enter_program(key, 0, source)?;
                let d = dict(pdf, value)?;
                let object = d.get(b"Kids").ok().and_then(|o| resolve(pdf, o).ok());
                let Some(kids) = object.and_then(|o| o.as_array().ok()).filter(|k| !k.is_empty()) else {
                    return Err(Error::new("FzErrorFormat", "malformed page tree"));
                };
                let identity = object.unwrap() as *const Object as usize;
                expansion.structural_program(identity, kids.len(), source)?;
                expansion.work(1, source)?;
                indexes.entry(identity).or_insert_with(|| Index::new(kids));
                // The occurrence rank changes at signed wrap, even on one route.
                let wrap = n as u128 + (MODULUS - rank(skip));
                frames.push(Frame { node, key, offset, end: end.min(wrap) });
            }
            let frame = frames.last().unwrap();
            let skip = n.wrapping_sub(frame.offset as i64);
            let d = dict(pdf, frame.node)?;
            let object = resolve(pdf, d.get(b"Kids")?)?;
            let index = indexes.get_mut(&(object as *const Object as usize)).unwrap();
            let Some(interval) = index.query(pdf, skip) else {
                return Err(Error::new("FzErrorFormat", format!("cannot find page {} in page tree", n + 1)));
            };
            match interval.action {
                Action::Null => return Err(Error::new("FzErrorFormat", "cycle in page tree")),
                Action::Leaf(kid) => {
                    if let Object::Reference(id) = kid {
                        expansion.enter_reference_program(*id, Some(*id))?;
                        out.push(((n + 1) as u32, *id));
                        break;
                    }
                    return Err(Error::new("ValueError", "page has no indirect xref"));
                }
                Action::Branch(kid) => {
                    let end = frame.end.min(n as u128 + interval.end - rank(skip));
                    next = Some((kid, frame.offset.wrapping_add(interval.prefix), end, false));
                }
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
#[path = "page_rank/page_rank_tests.rs"]
mod page_rank_tests;
