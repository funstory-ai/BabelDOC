//! Owned IR tree, with document indices for shared mutable schema objects.
//! Immutable structural subtrees are borrowed directly; Python memos never escape transfer.
#![forbid(unsafe_code)]
pub use babeldoc_il_types::{PythonValue, RuntimeBool};
mod compact_values;
pub use compact_values::{CurveCtm, DataPdfPath, PathOp};
use serde_json::Value as Json;
use std::io::Read;
use std::{
    collections::{BTreeMap, HashMap, HashSet},
    marker::PhantomData,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};

fn decode_base_operations(value: &str) -> Result<String, AccessError> {
    const ALPHABET: &[u8] =
        b"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!#$%&()*+-;<=>?@^_`{|}~";
    let pad = (5 - value.len() % 5) % 5;
    let mut input = value.as_bytes().to_vec();
    input.extend(std::iter::repeat_n(b'~', pad));
    let mut encoded = Vec::new();
    for chunk in input.chunks(5) {
        let mut n = 0u64;
        for c in chunk {
            let d = ALPHABET
                .iter()
                .position(|a| a == c)
                .ok_or(AccessError::WrongType)?;
            n = n * 85 + d as u64;
        }
        encoded.extend(
            u32::try_from(n)
                .map_err(|_| AccessError::WrongType)?
                .to_be_bytes(),
        );
    }
    encoded.truncate(encoded.len().saturating_sub(pad));
    let mut decoder = ruzstd::decoding::StreamingDecoder::new(encoded.as_slice())
        .map_err(|_| AccessError::WrongType)?;
    let mut output = String::new();
    decoder
        .read_to_string(&mut output)
        .map_err(|_| AccessError::WrongType)?;
    Ok(output)
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
#[serde(tag = "code", rename_all = "snake_case")]
pub enum AccessError {
    UnknownSlot,
    StaleHandle,
    NotOwned,
    Frozen,
    Missing { field: String },
    WrongType,
    CrossPage { kind: String, path: String },
}
impl std::fmt::Display for AccessError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            serde_json::to_string(self).map_err(|_| std::fmt::Error)?
        )
    }
}
impl std::error::Error for AccessError {}
static GENERATIONS: AtomicU64 = AtomicU64::new(1);
fn next_generation() -> u64 {
    GENERATIONS.fetch_add(1, Ordering::Relaxed)
}
static IDENTITIES: AtomicU64 = AtomicU64::new(1);
pub struct Node {
    pub identity: u64,
    pub data: NodeData,
}
impl Node {
    pub fn new(data: NodeData) -> Self {
        Self {
            identity: IDENTITIES.fetch_add(1, Ordering::Relaxed),
            data,
        }
    }
}
/// An immutable node allocation contains only its Arc counts, identity, and typed payload.
pub struct ImmutableNode<D> {
    pub identity: u64,
    pub data: D,
}
/// Fresh owning values become either document indices or immutable subtrees once attached.
/// The generated `Edge` enum gives every immutable type its own single-pointer Arc variant.
impl Edge {
    pub fn share(&self) -> Self {
        share_edge(self)
    }
}
/// Payloads occupy only their schema type's size, in a page or shared table.
struct StoredNode<D> {
    identity: u64,
    data: D,
}
struct NodeRef<'a> {
    identity: u64,
    data: NodeDataRef<'a>,
}
impl Node {
    fn as_ref(&self) -> NodeRef<'_> {
        NodeRef {
            identity: self.identity,
            data: self.data.as_ref(),
        }
    }
}
#[derive(Clone, Copy)]
struct Location {
    kind: Kind,
    scope: usize,
    index: usize,
}
/// The directory keeps edges one word wide; it never contains schema payloads.
/// Scope zero holds shared objects and newly attached values until the next
/// stage boundary. Scope n + 1 holds objects reachable from only page n.
#[derive(Default)]
pub struct Table {
    nodes: Vec<Location>,
    shared: TypedTables,
    pages: Vec<TypedTables>,
    staging_scope: usize,
}
impl Table {
    /// Give newly attached parser nodes a separately releasable source domain.
    /// This is provisional storage only: compaction determines final ownership
    /// from all root paths, including references from later pages.
    pub fn begin_page_scope(&mut self) {
        self.pages.push(TypedTables::default());
        self.staging_scope = self.pages.len();
    }
    /// Finish provisional construction without retaining each page's growth slack.
    pub fn finish_page_scope(&mut self) {
        self.scope_mut(self.staging_scope).shrink_to_fit();
        self.staging_scope = 0;
    }
    fn scope(&self, scope: usize) -> &TypedTables {
        if scope == 0 {
            &self.shared
        } else {
            &self.pages[scope - 1]
        }
    }
    fn scope_mut(&mut self, scope: usize) -> &mut TypedTables {
        if scope == 0 {
            &mut self.shared
        } else {
            &mut self.pages[scope - 1]
        }
    }
    fn node(&self, key: usize) -> NodeRef<'_> {
        let location = self.nodes[key];
        self.scope(location.scope).node(location)
    }
    /// Read a parser pool candidate without cloning its instruction payload.
    pub fn graphic_state_instructions(&self, edge: &Edge) -> Option<&Instructions> {
        let Edge::Shared(key) = edge else {
            return None;
        };
        let NodeDataRef::GraphicState(state) = self.node(*key).data else {
            return None;
        };
        state.passthrough_per_char_instruction.as_ref()
    }
    pub fn prepare_edge(&mut self, edge: Edge) -> Edge {
        match edge {
            Edge::Inline(node) => self.attach(*node),
            edge => edge,
        }
    }
    pub fn attach(&mut self, mut node: Node) -> Edge {
        for edge in node.data.edges_mut() {
            if let Edge::Inline(child) = edge {
                let replacement = self.attach(Node {
                    identity: child.identity,
                    data: std::mem::replace(
                        &mut child.data,
                        NodeData::Document(Default::default()),
                    ),
                });
                *edge = replacement;
            }
        }
        if node.data.kind().is_table() {
            let key = self.nodes.len();
            let kind = node.data.kind();
            let scope = self.staging_scope;
            let index = self.scope_mut(scope).push(node);
            self.nodes.push(Location { kind, scope, index });
            Edge::Shared(key)
        } else {
            node.into_immutable_edge()
        }
    }
}
#[cfg(feature = "memory-profile")]
impl Table {
    pub fn memory_profile(&self, root: Option<&Node>) -> serde_json::Value {
        fn walk(n: NodeRef<'_>, counts: &mut BTreeMap<&'static str, usize>, inline: &mut usize) {
            *counts.entry(n.data.kind().name()).or_default() += 1;
            for e in n.data.edges() {
                if let Edge::Inline(child) = e {
                    *inline += 1;
                    walk(Node::as_ref(child), counts, inline);
                }
            }
        }
        let mut counts = BTreeMap::new();
        let mut inline = 0;
        for key in 0..self.nodes.len() {
            walk(self.node(key), &mut counts, &mut inline);
        }
        if let Some(root) = root {
            walk(root.as_ref(), &mut counts, &mut inline);
        }
        serde_json::json!({"node_size":std::mem::size_of::<Node>(),"edge_size":std::mem::size_of::<Edge>(),"table_len":self.nodes.len(),"table_capacity":self.nodes.capacity(),"table_capacity_bytes":self.nodes.capacity()*std::mem::size_of::<Location>()+self.shared.capacity_bytes()+self.pages.iter().map(TypedTables::capacity_bytes).sum::<usize>(),"inline_nodes":inline,"inline_node_bytes":inline*std::mem::size_of::<Node>(),"node_kinds":counts})
    }
}
pub trait SchemaType {
    const KIND: Kind;
    const NAME: &'static str;
    type Data;
    fn from_immutable_edge(edge: &Edge) -> Option<&Arc<ImmutableNode<Self::Data>>>;
    fn into_immutable_edge(value: Arc<ImmutableNode<Self::Data>>) -> Edge;
    fn data(doc: &Document, index: usize) -> Result<&Self::Data, AccessError>;
    fn data_mut(doc: &mut Document, index: usize) -> Result<&mut Self::Data, AccessError>;
}
pub trait Permission {
    const WRITE: [bool; CLASS_COUNT];
}
pub trait Writes<T: SchemaType>: Permission {}
const ROOT: usize = usize::MAX;
/// A stage-local selection. Compaction is allowed only after all selections
/// from that stage have been discarded. Navigation never walks a schema path.
pub struct Cursor<T: SchemaType> {
    index: usize,
    generation: u64,
    _type: PhantomData<fn() -> T>,
}
impl<T: SchemaType> Copy for Cursor<T> {}
impl<T: SchemaType> Clone for Cursor<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T: SchemaType> std::fmt::Debug for Cursor<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple(T::NAME).field(&self.index).finish()
    }
}
impl<T: SchemaType> Cursor<T> {
    fn new(index: usize, generation: u64) -> Self {
        Self {
            index,
            generation,
            _type: PhantomData,
        }
    }
    fn from_edge(edge: &Edge, generation: u64) -> Result<Self, AccessError> {
        match edge {
            Edge::Shared(i) => Ok(Self::new(*i, generation)),
            _ => Err(AccessError::WrongType),
        }
    }
    pub fn identity(&self) -> u64 {
        self.index as u64
    }
    fn validate<P: Permission>(&self, v: &TreeView<'_, P>) -> Result<(), AccessError> {
        if self.generation != v.document_ref().generation {
            return Err(AccessError::StaleHandle);
        }
        Ok(())
    }
    fn data<'a, P: Permission>(&self, v: &'a TreeView<'_, P>) -> Result<&'a T::Data, AccessError> {
        self.validate(v)?;
        T::data(v.document_ref(), self.index)
    }
    fn data_mut<'a, P: Permission + Writes<T>>(
        &self,
        v: &'a mut TreeView<'_, P>,
    ) -> Result<&'a mut T::Data, AccessError> {
        self.validate(v)?;
        if !P::WRITE[T::KIND as usize] {
            return Err(AccessError::Frozen);
        }
        T::data_mut(v.document_mut()?, self.index)
    }
    pub fn retain<P: Permission>(&self, v: &mut TreeView<'_, P>) -> Result<Self, AccessError> {
        self.validate(v)?;
        Ok(*self)
    }
    pub fn share<P: Permission>(
        &self,
        v: &mut TreeView<'_, P>,
    ) -> Result<FieldValue<T>, AccessError> {
        self.validate(v)?;
        Ok(FieldValue::from_edge(
            Edge::Shared(self.index),
            self.generation,
        ))
    }
}
/// A direct borrow of an immutable structural subtree. Navigation does not clone its Arc.
pub struct Borrowed<'a, T: SchemaType> {
    value: &'a Arc<ImmutableNode<T::Data>>,
    generation: u64,
    _type: PhantomData<fn() -> T>,
}
impl<'a, T: SchemaType> Borrowed<'a, T> {
    fn from_edge(edge: &'a Edge, generation: u64) -> Result<Self, AccessError> {
        let value = T::from_immutable_edge(edge).ok_or(AccessError::WrongType)?;
        Ok(Self {
            value,
            generation,
            _type: PhantomData,
        })
    }
    fn validate<P: Permission>(&self, v: &TreeView<'_, P>) -> Result<(), AccessError> {
        if self.generation != v.document_ref().generation {
            return Err(AccessError::StaleHandle);
        }
        Ok(())
    }
    fn node(&self) -> &'a ImmutableNode<T::Data> {
        self.value
    }
    pub fn identity(&self) -> u64 {
        self.value.identity
    }
    pub fn share<P: Permission>(
        &self,
        v: &mut TreeView<'_, P>,
    ) -> Result<FieldValue<T>, AccessError> {
        self.validate(v)?;
        Ok(FieldValue::from_edge(
            T::into_immutable_edge(Arc::clone(self.value)),
            self.generation,
        ))
    }
}
#[derive(Clone, Copy, Default)]
enum Binding {
    #[default]
    Fresh,
    Generation(u64),
    Conflicting,
}
impl Binding {
    fn merge(&mut self, other: Self) {
        *self = match (*self, other) {
            (Self::Fresh, b) | (b, Self::Fresh) => b,
            (Self::Generation(a), Self::Generation(b)) if a == b => Self::Generation(a),
            _ => Self::Conflicting,
        };
    }
    fn validate(self, generation: u64) -> Result<(), AccessError> {
        match self {
            Self::Fresh => Ok(()),
            Self::Generation(g) if g == generation => Ok(()),
            _ => Err(AccessError::StaleHandle),
        }
    }
}
pub struct FieldValue<T: SchemaType> {
    edge: Edge,
    binding: Binding,
    _type: PhantomData<fn() -> T>,
}
impl<T: SchemaType> FieldValue<T> {
    fn new(data: NodeData, binding: Binding) -> Self {
        Self {
            edge: Edge::Inline(std::boxed::Box::new(Node::new(data))),
            binding,
            _type: PhantomData,
        }
    }
    fn from_edge(edge: Edge, generation: u64) -> Self {
        Self {
            edge,
            binding: Binding::Generation(generation),
            _type: PhantomData,
        }
    }
}
pub struct Document {
    generation: u64,
    root: Node,
    table: Table,
    pub history: History,
}
impl Document {
    #[cfg(feature = "memory-profile")]
    pub fn memory_profile(&self) -> serde_json::Value {
        self.table.memory_profile(None)
    }
    pub fn from_parts(
        mut root: Node,
        mut table: Table,
        history: History,
    ) -> Result<Self, AccessError> {
        if root.data.kind() != Kind::Document {
            return Err(AccessError::WrongType);
        }
        for edge in root.data.edges_mut() {
            if let Edge::Inline(child) = edge {
                *edge = table.attach(Node {
                    identity: child.identity,
                    data: std::mem::replace(
                        &mut child.data,
                        NodeData::Document(Default::default()),
                    ),
                });
            }
        }
        let mut document = Self {
            generation: next_generation(),
            root,
            table,
            history,
        };
        document.compact();
        Ok(document)
    }
    /// Narrowing preserves the compile-time write set.
    /// ```compile_fail
    /// use babeldoc_ir_core::{types,Cursor,TreeView,ReadOnlyView};
    /// fn illegal(p:Cursor<types::PdfParagraph>,v:&mut TreeView<'_,ReadOnlyView>) {
    ///     p.set_scale(v,Some(1.0)).unwrap();
    /// }
    /// ```
    pub fn view<P: Permission>(&mut self) -> TreeView<'_, P> {
        TreeView {
            doc: ViewStorage::Write(self),
            _permission: PhantomData,
        }
    }
    pub fn read(&self) -> TreeView<'_, ReadOnlyView> {
        TreeView {
            doc: ViewStorage::Read(self),
            _permission: PhantomData,
        }
    }
    pub fn payload_utf8_bytes(&self) -> usize {
        fn count(n: NodeRef<'_>, seen: &mut HashSet<u64>) -> usize {
            if !seen.insert(n.identity) {
                return 0;
            }
            n.data.payload_utf8_bytes()
                + n.data
                    .edges()
                    .into_iter()
                    .map(|edge| edge.payload_utf8_bytes(seen))
                    .sum::<usize>()
        }
        let mut seen = HashSet::new();
        count(self.root.as_ref(), &mut seen)
            + self
                .table
                .nodes
                .iter()
                .enumerate()
                .map(|(key, _)| count(self.table.node(key), &mut seen))
                .sum::<usize>()
            + self
                .history
                .nodes
                .iter()
                .map(|n| {
                    n.op.len()
                        + n.replace_key.as_ref().map_or(0, String::len)
                        + match &n.arg {
                            HistoryArg::Text(s) => s.len(),
                            HistoryArg::CtmAware { value, ctm } => value.len() + ctm.utf8_bytes(),
                        }
                })
                .sum::<usize>()
    }
    pub fn artifact(&self) -> Result<Json, AccessError> {
        let mut dump = DumpContext::new(self);
        let document = dump.visit(self.root.as_ref(), "$".into())?;
        let mut groups: BTreeMap<String, Vec<Vec<String>>> = BTreeMap::new();
        for range in alias_ranges(
            &mut dump.aliases,
            &dump.alias_suffixes,
            &dump.alias_prefixes,
        ) {
            let kind = dump.aliases[range.0].kind.name().to_owned();
            let paths = dump.aliases[range.0..range.1]
                .iter()
                .map(|a| alias_path(a, &dump.alias_suffixes, &dump.alias_prefixes))
                .collect();
            groups.entry(kind).or_default().push(paths)
        }
        for entries in groups.values_mut() {
            entries.sort()
        }
        dump.views.sort_by(|a, b| a.path.cmp(&b.path));
        Ok(
            serde_json::json!({"schema_version":1,"document":document,"shared_alias_paths":groups,"graphic_state_nodes":dump.nodes,"graphic_state_views":dump.views}),
        )
    }

    /// Stream the compact artifact JSON without constructing its document subtree.
    pub fn write_artifact<W: std::io::Write>(&self, out: &mut W) -> Result<(), AccessError> {
        let mut dump = DumpContext::new(self);
        dump.discovering = true;
        #[cfg(feature = "memory-profile")]
        dump.checkpoint("start");
        dump.discover_node(self.root.as_ref(), "$")?;
        dump.discovering = false;
        #[cfg(feature = "memory-profile")]
        dump.checkpoint("discovered");
        dump.aliases.clear();
        dump.alias_suffixes.clear();
        dump.alias_prefixes.clear();
        dump.alias_prefix_ids.clear();
        dump.views.clear();
        out.write_all(b"{\"document\":")
            .map_err(|_| AccessError::WrongType)?;
        dump.write_node(self.root.as_ref(), "$", out)?;
        #[cfg(feature = "memory-profile")]
        dump.checkpoint("written_document");
        let ranges = alias_ranges(
            &mut dump.aliases,
            &dump.alias_suffixes,
            &dump.alias_prefixes,
        );
        dump.views.sort_by(|a, b| a.path.cmp(&b.path));
        // serde_json::Map sorts keys, so match its canonical compact order.
        out.write_all(b",\"graphic_state_nodes\":")
            .map_err(|_| AccessError::WrongType)?;
        serde_json::to_writer(&mut *out, &dump.nodes).map_err(|_| AccessError::WrongType)?;
        out.write_all(b",\"graphic_state_views\":")
            .map_err(|_| AccessError::WrongType)?;
        serde_json::to_writer(&mut *out, &dump.views).map_err(|_| AccessError::WrongType)?;
        out.write_all(b",\"schema_version\":1,\"shared_alias_paths\":")
            .map_err(|_| AccessError::WrongType)?;
        write_alias_groups(
            &dump.aliases,
            &ranges,
            &dump.alias_suffixes,
            &dump.alias_prefixes,
            out,
        )?;
        out.write_all(b"}").map_err(|_| AccessError::WrongType)?;
        Ok(())
    }
    pub fn artifact_sha256(&self) -> Result<String, AccessError> {
        use sha2::Digest;
        struct HashSink(sha2::Sha256);
        impl std::io::Write for HashSink {
            fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
                self.0.update(bytes);
                Ok(bytes.len())
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        let mut sink = HashSink(sha2::Sha256::new());
        self.write_artifact(&mut sink)?;
        let digest = sink.0.finalize();
        Ok(digest.iter().map(|byte| format!("{byte:02x}")).collect())
    }
    pub fn artifact_json(&self) -> Result<String, AccessError> {
        let mut bytes = Vec::new();
        self.write_artifact(&mut bytes)?;
        String::from_utf8(bytes).map_err(|_| AccessError::WrongType)
    }
}
enum ViewStorage<'a> {
    Read(&'a Document),
    Write(&'a mut Document),
}
impl ViewStorage<'_> {
    fn reborrow(&mut self) -> ViewStorage<'_> {
        match self {
            Self::Read(d) => ViewStorage::Read(d),
            Self::Write(d) => ViewStorage::Write(d),
        }
    }
}
pub struct TreeView<'a, P: Permission> {
    doc: ViewStorage<'a>,
    _permission: PhantomData<P>,
}
impl<P: Permission> TreeView<'_, P> {
    fn document_ref(&self) -> &Document {
        match &self.doc {
            ViewStorage::Read(d) => d,
            ViewStorage::Write(d) => d,
        }
    }
    fn document_mut(&mut self) -> Result<&mut Document, AccessError> {
        match &mut self.doc {
            ViewStorage::Read(_) => Err(AccessError::Frozen),
            ViewStorage::Write(d) => Ok(d),
        }
    }
    pub fn document(&self) -> Cursor<types::Document> {
        Cursor::new(ROOT, self.document_ref().generation)
    }
    pub fn render_instructions(
        &self,
        value: &Option<Instructions>,
    ) -> Result<Option<String>, AccessError> {
        render_instructions(value, &self.document_ref().history)
    }
    fn prepare<T: SchemaType>(&mut self, value: FieldValue<T>) -> Result<Edge, AccessError> {
        value.binding.validate(self.document_ref().generation)?;
        Ok(self.document_mut()?.table.prepare_edge(value.edge))
    }
    /// Reserve a batch of new table nodes without geometric capacity growth.
    /// This counts only nodes of T, not any inline children they may attach.
    /// Existing cursors and object identities remain valid.
    pub fn reserve_new_nodes<T: SchemaType>(&mut self, additional: usize) -> Result<(), AccessError>
    where
        P: Writes<T>,
    {
        if !T::KIND.is_table() {
            return Err(AccessError::NotOwned);
        }
        let table = &mut self.document_mut()?.table;
        table.nodes.reserve_exact(additional);
        table
            .scope_mut(table.staging_scope)
            .reserve_kind(T::KIND, additional);
        Ok(())
    }
    pub fn create_pool<T: SchemaType>(
        &mut self,
        value: FieldValue<T>,
    ) -> Result<Cursor<T>, AccessError>
    where
        P: Writes<T>,
    {
        if !T::KIND.is_pool() {
            return Err(AccessError::NotOwned);
        }
        let edge = self.prepare(value)?;
        Cursor::from_edge(&edge, self.document_ref().generation)
    }
    pub fn root<T: SchemaType>(&self) -> Result<Cursor<T>, AccessError> {
        if T::KIND != Kind::Document {
            return Err(AccessError::WrongType);
        }
        Ok(Cursor::new(ROOT, self.document_ref().generation))
    }
}
#[derive(serde::Serialize)]
#[serde(untagged)]
pub enum HistoryArg {
    Text(String),
    CtmAware { value: String, ctm: PythonValue },
}
pub struct HistoryNode {
    pub parent: Option<usize>,
    pub op: String,
    pub arg: HistoryArg,
    pub replace_key: Option<String>,
    pub version: u64,
    pub length: u64,
}
#[derive(Default)]
pub struct History {
    nodes: Vec<HistoryNode>,
    base: usize,
}
impl History {
    pub fn push(&mut self, mut node: HistoryNode) -> usize {
        node.parent = node
            .parent
            .map(|key| self.index(key).expect("history parent key must be current"));
        let i = self
            .base
            .checked_add(self.nodes.len())
            .expect("history key space exhausted");
        self.nodes.push(node);
        i
    }
    fn index(&self, key: usize) -> Result<usize, AccessError> {
        let i = key.checked_sub(self.base).ok_or(AccessError::StaleHandle)?;
        if i >= self.nodes.len() {
            return Err(AccessError::UnknownSlot);
        }
        Ok(i)
    }
    /// Parents are stored as local indices; expose only current external keys.
    pub fn parent_key(&self, node: &HistoryNode) -> Option<usize> {
        node.parent.map(|i| self.base + i)
    }
    pub fn get(&self, key: usize) -> Result<&HistoryNode, AccessError> {
        Ok(&self.nodes[self.index(key)?])
    }
    #[cfg(feature = "memory-profile")]
    pub fn memory_profile(&self) -> serde_json::Value {
        serde_json::json!({"nodes":self.nodes.len(),"capacity":self.nodes.capacity(),"node_size":std::mem::size_of::<HistoryNode>()})
    }
}
#[derive(Clone, Debug)]
pub struct InstructionView {
    pub tail: Option<usize>,
    pub latest_versions: Vec<(String, u64)>,
    pub include_clipping: bool,
    pub suffix_parts: Vec<String>,
}
#[derive(Clone, Debug)]
pub enum Instructions {
    Materialized(String),
    View(InstructionView),
}
struct DumpContext<'a> {
    doc: &'a Document,
    aliases: Vec<AliasRecord>,
    alias_suffixes: Vec<u8>,
    alias_prefixes: Vec<String>,
    alias_prefix_ids: HashMap<String, usize>,
    discovering: bool,
    node_ids: BTreeMap<usize, u64>,
    nodes: Vec<StateNodeRecord<'a>>,
    views: Vec<StateViewRecord>,
}

struct AliasRecord {
    path_start: usize,
    path_len: usize,
    identity: u64,
    prefix_id: usize,
    kind: Kind,
}

fn split_alias_path(path: &str) -> Option<(&str, &str)> {
    let bracket = path.rfind('[')?;
    let suffix = &path[bracket..];
    let close = suffix.find(']')?;
    let digits = &suffix[1..close];
    (!digits.is_empty() && digits.bytes().all(|byte| byte.is_ascii_digit()))
        .then_some((&path[..bracket], suffix))
}

fn store_alias_path(
    path: &str,
    prefixes: &mut Vec<String>,
    prefix_ids: &mut HashMap<String, usize>,
    suffixes: &mut Vec<u8>,
) -> (usize, usize, usize) {
    let (prefix_id, suffix) = if let Some((prefix, suffix)) = split_alias_path(path) {
        let id = if let Some(&id) = prefix_ids.get(prefix) {
            id
        } else {
            let id = prefixes.len();
            prefixes.push(prefix.to_owned());
            prefix_ids.insert(prefix.to_owned(), id);
            id
        };
        (id, suffix)
    } else {
        (usize::MAX, path)
    };
    let start = suffixes.len();
    suffixes.extend_from_slice(suffix.as_bytes());
    (prefix_id, start, suffix.len())
}

fn alias_path_parts<'a>(
    alias: &AliasRecord,
    suffixes: &'a [u8],
    prefixes: &'a [String],
) -> (&'a str, &'a str) {
    let suffix_end = alias.path_start + alias.path_len;
    let suffix = std::str::from_utf8(&suffixes[alias.path_start..suffix_end])
        .expect("alias path suffixes originate from UTF-8 strings");
    let prefix = if alias.prefix_id == usize::MAX {
        ""
    } else {
        prefixes[alias.prefix_id].as_str()
    };
    (prefix, suffix)
}

fn compare_alias_paths(
    a: &AliasRecord,
    b: &AliasRecord,
    suffixes: &[u8],
    prefixes: &[String],
) -> std::cmp::Ordering {
    let (a_prefix, a_suffix) = alias_path_parts(a, suffixes, prefixes);
    let (b_prefix, b_suffix) = alias_path_parts(b, suffixes, prefixes);
    if a.prefix_id == b.prefix_id {
        return a_suffix.as_bytes().cmp(b_suffix.as_bytes());
    }
    let prefix_order = a_prefix.as_bytes().cmp(b_prefix.as_bytes());
    if !a_prefix.starts_with(b_prefix) && !b_prefix.starts_with(a_prefix) {
        return prefix_order;
    }
    a_prefix
        .as_bytes()
        .iter()
        .chain(a_suffix.as_bytes())
        .cmp(b_prefix.as_bytes().iter().chain(b_suffix.as_bytes()))
}

fn alias_path(alias: &AliasRecord, suffixes: &[u8], prefixes: &[String]) -> String {
    let (prefix, suffix) = alias_path_parts(alias, suffixes, prefixes);
    let mut path = String::with_capacity(prefix.len() + suffix.len());
    path.push_str(prefix);
    path.push_str(suffix);
    path
}

fn compare_alias_groups(
    a: &[AliasRecord],
    b: &[AliasRecord],
    suffixes: &[u8],
    prefixes: &[String],
) -> std::cmp::Ordering {
    let mut a_iter = a.iter();
    let mut b_iter = b.iter();
    loop {
        match (a_iter.next(), b_iter.next()) {
            (Some(a), Some(b)) => {
                let order = compare_alias_paths(a, b, suffixes, prefixes);
                if order != std::cmp::Ordering::Equal {
                    return order;
                }
            }
            (None, Some(_)) => return std::cmp::Ordering::Less,
            (Some(_), None) => return std::cmp::Ordering::Greater,
            (None, None) => return std::cmp::Ordering::Equal,
        }
    }
}

fn alias_ranges(
    aliases: &mut [AliasRecord],
    suffixes: &[u8],
    prefixes: &[String],
) -> Vec<(usize, usize)> {
    aliases.sort_unstable_by(|a, b| {
        a.kind
            .name()
            .cmp(b.kind.name())
            .then(a.identity.cmp(&b.identity))
            .then_with(|| compare_alias_paths(a, b, suffixes, prefixes))
    });
    let mut ranges = Vec::new();
    let mut start = 0;
    while start < aliases.len() {
        let mut end = start + 1;
        while end < aliases.len()
            && (aliases[end].kind, aliases[end].identity)
                == (aliases[start].kind, aliases[start].identity)
        {
            end += 1;
        }
        ranges.push((start, end));
        start = end;
    }
    ranges.sort_unstable_by(|&(a, b), &(c, d)| {
        aliases[a]
            .kind
            .name()
            .cmp(aliases[c].kind.name())
            .then_with(|| compare_alias_groups(&aliases[a..b], &aliases[c..d], suffixes, prefixes))
    });
    ranges
}

fn write_alias_groups<W: std::io::Write>(
    aliases: &[AliasRecord],
    ranges: &[(usize, usize)],
    suffixes: &[u8],
    prefixes: &[String],
    out: &mut W,
) -> Result<(), AccessError> {
    let io = |_| AccessError::WrongType;
    out.write_all(b"{").map_err(io)?;
    let mut previous = None;
    let mut scratch = String::new();
    for &(start, end) in ranges {
        let kind = aliases[start].kind.name();
        if previous != Some(kind) {
            if previous.is_some() {
                out.write_all(b"],").map_err(io)?;
            }
            serde_json::to_writer(&mut *out, kind).map_err(|_| AccessError::WrongType)?;
            out.write_all(b":[").map_err(io)?;
            previous = Some(kind);
        } else {
            out.write_all(b",").map_err(io)?;
        }
        out.write_all(b"[").map_err(io)?;
        for (i, alias) in aliases[start..end].iter().enumerate() {
            if i > 0 {
                out.write_all(b",").map_err(io)?;
            }
            let (prefix, suffix) = alias_path_parts(alias, suffixes, prefixes);
            scratch.clear();
            scratch.push_str(prefix);
            scratch.push_str(suffix);
            serde_json::to_writer(&mut *out, &scratch).map_err(|_| AccessError::WrongType)?;
        }
        out.write_all(b"]").map_err(io)?;
    }
    if previous.is_some() {
        out.write_all(b"]").map_err(io)?;
    }
    out.write_all(b"}").map_err(io)?;
    Ok(())
}

#[derive(serde::Serialize)]
struct StateViewRecord {
    include_clipping: bool,
    latest_versions: Vec<(String, u64)>,
    path: String,
    suffix_parts: Vec<String>,
    tail_ordinal: Option<u64>,
}

#[derive(serde::Serialize)]
struct StateNodeRecord<'a> {
    arg: HistoryArgRecord<'a>,
    length: u64,
    op: &'a str,
    ordinal: u64,
    parent_ordinal: Option<u64>,
    replace_key: Option<&'a str>,
    version: u64,
}

#[derive(serde::Serialize)]
#[serde(untagged)]
enum HistoryArgRecord<'a> {
    Text(&'a str),
    CtmAware {
        ctm: &'a PythonValue,
        value: &'a str,
    },
}

impl<'a> From<&'a HistoryArg> for HistoryArgRecord<'a> {
    fn from(arg: &'a HistoryArg) -> Self {
        match arg {
            HistoryArg::Text(value) => Self::Text(value),
            HistoryArg::CtmAware { value, ctm } => Self::CtmAware { ctm, value },
        }
    }
}

impl<'a> DumpContext<'a> {
    fn alias(&mut self, kind: Kind, identity: u64, path: &str) {
        if !self.discovering {
            let (prefix_id, path_start, path_len) = store_alias_path(
                path,
                &mut self.alias_prefixes,
                &mut self.alias_prefix_ids,
                &mut self.alias_suffixes,
            );
            self.aliases.push(AliasRecord {
                path_start,
                path_len,
                identity,
                prefix_id,
                kind,
            });
        }
    }
    #[cfg(feature = "memory-profile")]
    fn checkpoint(&self, phase: &str) {
        let status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
        let memory: Vec<_> = status
            .lines()
            .filter(|s| s.starts_with("VmRSS:") || s.starts_with("VmHWM:"))
            .collect();
        eprintln!(
            "HASH_MEMORY {}",
            serde_json::json!({"phase":phase,"memory":memory,"aliases":self.aliases.len(),"paths":self.aliases.len(),"path_bytes":self.alias_suffixes.len()+self.alias_prefixes.iter().map(String::len).sum::<usize>(),"nodes":self.nodes.len(),"views":self.views.len()})
        );
    }
    fn new(doc: &'a Document) -> Self {
        Self {
            doc,
            aliases: Vec::new(),
            alias_suffixes: Vec::new(),
            alias_prefixes: Vec::new(),
            alias_prefix_ids: HashMap::new(),
            discovering: false,
            node_ids: BTreeMap::new(),
            nodes: vec![],
            views: vec![],
        }
    }
    fn visit(&mut self, node: NodeRef<'_>, path: String) -> Result<Json, AccessError> {
        self.alias(node.data.kind(), node.identity, &path);
        node.data.dump(self, &path)
    }
    fn write_node<W: std::io::Write>(
        &mut self,
        node: NodeRef<'_>,
        path: &str,
        out: &mut W,
    ) -> Result<(), AccessError> {
        self.alias(node.data.kind(), node.identity, path);
        node.data.write(self, path, out)
    }
    fn discover_node(&mut self, node: NodeRef<'_>, path: &str) -> Result<(), AccessError> {
        node.data.discover(self, path)
    }
    fn discover_edge(&mut self, edge: &Edge, path: &str) -> Result<(), AccessError> {
        match edge {
            Edge::Inline(node) => self.discover_node(Node::as_ref(node), path),
            Edge::Shared(index) => self.discover_node(self.doc.table.node(*index), path),
            _ => discover_immutable_edge(self, edge, path),
        }
    }
    fn edge(&mut self, edge: &Edge, path: String) -> Result<Json, AccessError> {
        match edge {
            Edge::Inline(n) => self.visit(Node::as_ref(n), path),
            Edge::Shared(index) => self.visit(self.doc.table.node(*index), path),
            _ => dump_immutable_edge(self, edge, &path),
        }
    }
    fn write_edge<W: std::io::Write>(
        &mut self,
        edge: &Edge,
        path: &str,
        out: &mut W,
    ) -> Result<(), AccessError> {
        match edge {
            Edge::Inline(n) => self.write_node(Node::as_ref(n), path, out),
            Edge::Shared(i) => self.write_node(self.doc.table.node(*i), path, out),
            _ => write_immutable_edge(self, edge, path, out),
        }
    }
    fn instructions(
        &mut self,
        value: &Option<Instructions>,
        path: String,
    ) -> Result<Json, AccessError> {
        let Some(value) = value else {
            return Ok(Json::Null);
        };
        let Instructions::View(view) = value else {
            if self.discovering {
                return Ok(Json::Null);
            }
            let Instructions::Materialized(s) = value else {
                unreachable!()
            };
            return Ok(Json::String(s.clone()));
        };
        let mut chain = vec![];
        let mut current = view.tail;
        while let Some(key) = current {
            if self.node_ids.contains_key(&key) {
                break;
            }
            chain.push(key);
            current = self.doc.history.parent_key(self.doc.history.get(key)?);
        }
        for key in chain.into_iter().rev() {
            let node = self.doc.history.get(key)?;
            let parent = self
                .doc
                .history
                .parent_key(node)
                .map(|r| {
                    self.node_ids
                        .get(&r)
                        .copied()
                        .ok_or(AccessError::UnknownSlot)
                })
                .transpose()?;
            let ordinal = self.nodes.len() as u64;
            self.node_ids.insert(key, ordinal);
            self.nodes.push(StateNodeRecord {
                ordinal,
                parent_ordinal: parent,
                op: &node.op,
                arg: HistoryArgRecord::from(&node.arg),
                replace_key: node.replace_key.as_deref(),
                version: node.version,
                length: node.length,
            });
        }
        let tail = view
            .tail
            .map(|r| {
                self.node_ids
                    .get(&r)
                    .copied()
                    .ok_or(AccessError::UnknownSlot)
            })
            .transpose()?;
        if self.discovering {
            return Ok(Json::Null);
        }
        let mut value = serde_json::json!({"kind":"view","tail_ref":tail,"latest_versions":view.latest_versions,"include_clipping":view.include_clipping,"suffix_parts":view.suffix_parts});
        self.views.push(StateViewRecord {
            include_clipping: view.include_clipping,
            latest_versions: view.latest_versions.clone(),
            path,
            suffix_parts: view.suffix_parts.clone(),
            tail_ordinal: tail,
        });
        Ok(value.take())
    }
}

fn json_utf8_bytes(v: &Json) -> usize {
    match v {
        Json::String(s) => s.len(),
        Json::Array(v) => v.iter().map(json_utf8_bytes).sum(),
        Json::Object(v) => v.iter().map(|(k, v)| k.len() + json_utf8_bytes(v)).sum(),
        _ => 0,
    }
}
fn instruction_utf8_bytes(v: &Option<Instructions>) -> usize {
    match v {
        None => 0,
        Some(Instructions::Materialized(s)) => s.len(),
        Some(Instructions::View(v)) => {
            v.latest_versions
                .iter()
                .map(|(k, _)| k.len())
                .sum::<usize>()
                + v.suffix_parts.iter().map(String::len).sum::<usize>()
        }
    }
}
include!(concat!(env!("OUT_DIR"), "/schema.rs"));
fn render_instructions(
    value: &Option<Instructions>,
    history: &History,
) -> Result<Option<String>, AccessError> {
    let Some(value) = value else { return Ok(None) };
    let Instructions::View(view) = value else {
        let Instructions::Materialized(s) = value else {
            unreachable!()
        };
        return Ok(Some(s.clone()));
    };
    let mut chain = vec![];
    let mut current = view.tail;
    while let Some(key) = current {
        let node = history.get(key)?;
        chain.push(node);
        current = history.parent_key(node);
    }
    let latest: std::collections::HashMap<_, _> = view
        .latest_versions
        .iter()
        .map(|(k, v)| (k.as_str(), *v))
        .collect();
    let mut parts = vec![];
    for node in chain.into_iter().rev() {
        if node
            .replace_key
            .as_deref()
            .is_some_and(|k| latest.get(k).copied() != Some(node.version))
        {
            continue;
        }
        if !view.include_clipping && matches!(node.op.as_str(), "W n" | "W* n") {
            continue;
        }
        let arg = match &node.arg {
            HistoryArg::Text(s) => s,
            HistoryArg::CtmAware { value, .. } => value,
        };
        parts.push(format!("{arg} {}", node.op));
    }
    parts.extend(view.suffix_parts.iter().filter(|s| !s.is_empty()).cloned());
    Ok(Some(parts.join(" ")))
}

fn float_artifact(n: f64) -> Json {
    if n.is_nan() {
        Json::String("nan".into())
    } else if n == f64::INFINITY {
        Json::String("inf".into())
    } else if n == f64::NEG_INFINITY {
        Json::String("-inf".into())
    } else {
        serde_json::json!(n)
    }
}

include!("compact.rs");

#[cfg(test)]
mod capacity_tests;

#[cfg(test)]
mod lib_tests;
