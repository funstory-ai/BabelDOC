use super::*;
use babeldoc_il_types::{CurveDebugInfo, RuntimeBool};
use ir::{InstructionView, Instructions};
use std::sync::Arc;

pub(super) const MAX_FONT_ALIAS_BYTES: usize = 4096;

impl ScopeFonts {
    fn find_entry(&mut self, name: &str) -> (Option<usize>, bool) {
        if let Some(resolved) = self.resolved.get(name) {
            return (*resolved, false);
        }
        if let Some(index) = self.exact.get(name).copied() {
            self.resolved.insert(name.to_owned(), Some(index));
            return (Some(index), false);
        }
        let mut skipped_overlong = name.len() > MAX_FONT_ALIAS_BYTES;
        let resolved = if skipped_overlong {
            None
        } else {
            self.entries.iter().position(|font| {
                if font.name.len() > MAX_FONT_ALIAS_BYTES {
                    skipped_overlong = true;
                    return false;
                }
                let mut alias = font.name.trim_start_matches('/').to_owned();
                loop {
                    let next = tokens::decode_name(&alias);
                    if next == alias {
                        return false;
                    }
                    if next == name {
                        return true;
                    }
                    alias = next;
                }
            })
        };
        self.resolved.insert(name.to_owned(), resolved);
        (resolved, skipped_overlong)
    }
}
impl ClipList {
    fn formatted(&self, target: Matrix) -> Option<String> {
        let target_bits = target.map(f64::to_bits);
        {
            let cached = self
                .formatted
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            if let Some(cached) = cached.as_ref()
                && cached.target_bits == target_bits
            {
                return cached.text.clone();
            }
        }
        #[cfg(test)]
        {
            self.format_misses
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        let text = format_clip(&self.clip, target);
        let mut cached = self
            .formatted
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        *cached = Some(FormattedClip { target_bits, text });
        cached.as_ref().and_then(|cached| cached.text.clone())
    }
}
fn format_clip(clip: &Clip, target: Matrix) -> Option<String> {
    let transform = mul(clip.ctm, inv(target));
    let mut args = Vec::new();
    for seg in clip.path.iter() {
        if clip.ctm == target {
            for x in &seg.coords {
                args.push(format!("{x:.6}"))
            }
        } else {
            for p in seg.coords.chunks(2) {
                let q = pt(transform, [p[0], p[1]]);
                args.extend(q.iter().map(|x| format!("{x:.6}")))
            }
        }
        args.push(seg.op.clone())
    }
    (!args.is_empty()).then(|| {
        format!(
            "{} {}",
            args.join(" "),
            if clip.evenodd { "W* n" } else { "W n" }
        )
    })
}
fn formatted_clips(clips: Option<&ClipList>, target: Option<Matrix>) -> Vec<String> {
    let (Some(clips), Some(target)) = (clips, target) else {
        return Vec::new();
    };
    let mut nodes = Vec::new();
    let mut current = Some(clips);
    while let Some(node) = current {
        nodes.push(node);
        current = node.parent.as_deref();
    }
    nodes
        .into_iter()
        .rev()
        .filter_map(|node| node.formatted(target))
        .collect()
}
pub(super) enum Buffered {
    // Projection filters these chars, but they still count toward a native batch.
    FilteredChar,
    Char {
        bbox: [f64; 4],
        size: f64,
        advance: f64,
        cid: u32,
        text: String,
        font: String,
        font_descent: f64,
        font_xref: i64,
        vertical: bool,
        xid: i64,
        order: i64,
        snap: Snapshot,
        tm: Matrix,
        mode: i64,
    },
    Curve {
        path: Arc<[Segment]>,
        primitive: Option<Vec<f64>>,
        ctm: Matrix,
        xid: i64,
        order: i64,
        snap: Snapshot,
        clips: Option<Arc<ClipList>>,
        stroke: bool,
        fill: bool,
        evenodd: bool,
    },
    Shading {
        name: String,
        ctm: Matrix,
        xid: i64,
        order: i64,
        snap: Snapshot,
        clips: Option<Arc<ClipList>>,
    },
}
// The reference stores font xrefs and Form ids in the same dictionary. A Form
// can therefore overwrite a glyph entry in a cached font map, including making
// cycles. Integer arena links retain that aliasing without leaking Rc cycles.
#[derive(Clone, Copy)]
enum BboxValue {
    Box([f64; 4]),
    Map(usize),
}
#[derive(Default)]
pub(super) struct BboxMap {
    // Initial glyph entries are immutable in the backend. Only dictionary
    // overwrites and appended keys need storage in this aliased map.
    base: Option<Arc<fonts::FontBackend>>,
    values: HashMap<Option<i64>, BboxValue>,
    order: Vec<Option<i64>>,
}
impl BboxMap {
    fn base_len(&self) -> usize {
        self.base.as_ref().map_or(0, |b| b.projection_bboxes.len())
    }
    fn base_index(&self, key: Option<i64>) -> Option<usize> {
        usize::try_from(key?).ok().filter(|&i| i < self.base_len())
    }
    fn get(&self, key: Option<i64>) -> Option<BboxValue> {
        self.values.get(&key).copied().or_else(|| {
            let i = self.base_index(key)?;
            Some(BboxValue::Box(self.base.as_ref()?.projection_bboxes[i]))
        })
    }
    fn len(&self) -> usize {
        self.base_len() + self.order.len()
    }
    fn key(&self, index: usize) -> Option<i64> {
        let base_len = self.base_len();
        if index < base_len {
            Some(index as i64)
        } else {
            self.order[index - base_len]
        }
    }
    #[cfg(feature = "memory-profile")]
    pub(super) fn memory_profile(&self) -> serde_json::Value {
        serde_json::json!({"entries":self.len(), "base_entries":self.base_len(), "overrides":self.values.len(), "capacity":self.values.capacity(), "order_capacity":self.order.capacity()})
    }
    fn insert(&mut self, key: Option<i64>, value: BboxValue) {
        let in_base = self.base_index(key).is_some();
        if self.values.insert(key, value).is_none() && !in_base {
            self.order.push(key);
        }
    }
}

impl Parser {
    fn curve_ctm(&mut self, matrix: Matrix) -> ir::CurveCtm {
        self.curve_ctms
            .entry(matrix.map(f64::to_bits))
            .or_insert_with(|| ir::CurveCtm::from_matrix(matrix))
            .clone()
    }

    pub(super) fn prepare_form_resources(&mut self, resources: &Arc<Object>) {
        let graph_key = self
            .inherited_objects
            .object_ids
            .get(&(Arc::as_ptr(resources) as usize))
            .copied();
        if let Some(graph_key) = graph_key {
            if let Some(prepared) = self.prepared_resource_graphs.get(&graph_key) {
                self.prepared_form_resources = Arc::clone(prepared);
                return;
            }
        }
        // PreparedXObject is cached by xref in a depth-first walk of the entire
        // page resource graph. A missing Resources therefore retains the first
        // fallback, even when the same Form is later used under another parent.
        let root_resources = Arc::clone(resources);
        let mut prepared = HashMap::new();
        let mut resource_objects = HashMap::new();
        let mut stack = vec![(None, Arc::clone(&root_resources))];
        let mut expanded_resources = HashSet::new();
        while let Some((form, resources)) = stack.pop() {
            if let Some(id) = form {
                if prepared.contains_key(&id) {
                    continue;
                }
                prepared.insert(id, Arc::clone(&resources));
            }
            let Ok(d) = dict(&self.pdf, &resources) else {
                continue;
            };
            // Like resource_map_cache in the reference, expand each resolved
            // resource dictionary once. Register every Form's first fallback
            // above even when its resource graph was already expanded.
            // The root Arc and prepared map keep direct dictionaries alive.
            if !expanded_resources.insert(d as *const Dictionary as usize) {
                continue;
            }
            let children = d.get(b"XObject").ok().and_then(|o| dict(&self.pdf, o).ok());
            let Some(children) = children else { continue };
            // Push one resource graph at a time in PDF dictionary order.
            let mut pending = Vec::new();
            for (_, object) in children.iter() {
                let Object::Reference(id) = object else {
                    continue;
                };
                if prepared.contains_key(id) {
                    continue;
                }
                let Ok(child) = dict(&self.pdf, object) else {
                    continue;
                };
                if child.get(b"Subtype").ok().and_then(|o| o.as_name().ok()) != Some(b"Form") {
                    continue;
                }
                let next = child
                    .get(b"Resources")
                    .ok()
                    .filter(|o| object_truthy(o))
                    .map(|direct| {
                        let key = direct as *const Object as usize;
                        resource_objects
                            .entry(key)
                            .or_insert_with(|| Arc::new(direct.clone()))
                            .clone()
                    })
                    .unwrap_or_else(|| Arc::clone(&resources));
                pending.push((*id, next));
            }
            // Defer insertion until a child is visited, not when queued: earlier
            // siblings' descendants must win over later siblings.
            for (id, next) in pending.into_iter().rev() {
                stack.push((Some(id), next));
            }
        }
        let prepared = Arc::new(prepared);
        self.prepared_form_resources = Arc::clone(&prepared);
        if let Some(graph_key) = graph_key {
            self.prepared_resource_graphs
                .insert(graph_key, Arc::clone(&prepared));
        }
    }
    pub(super) fn fonts(&mut self, resources: &Object) -> Result<ScopeFonts> {
        if let Some(fonts) = self.scope_font_cache.get(&self.scope_path) {
            return Ok(fonts.clone());
        }
        if self.scope_path.is_empty() {
            let fonts = self.resolve_scope_fonts(resources)?;
            self.scope_font_cache.insert(Vec::new(), fonts.clone());
            return Ok(fonts);
        }
        // PageResourceBundle.get_direct_font_map re-resolves the ancestors on
        // a cache miss, overwriting intermediate direct maps along that path.
        // A repeated cached path retains its direct backend identities.
        let path = self.scope_path.clone();
        let mut current_resources = self.root_resources.clone();
        let mut current = self.resolve_scope_fonts(&current_resources)?;
        self.scope_font_cache
            .entry(Vec::new())
            .or_insert_with(|| current.clone());
        let mut prefix = Vec::new();
        for name in path {
            let object = dict(&self.pdf, &current_resources)
                .ok()
                .and_then(|d| d.get(b"XObject").ok())
                .and_then(|o| dict(&self.pdf, o).ok())
                .and_then(|d| d.get(name.as_bytes()).ok());
            let cached = object
                .and_then(|o| o.as_reference().ok())
                .and_then(|id| self.prepared_form_resources.get(&id));
            let next = cached
                .cloned()
                .or_else(|| {
                    object
                        .and_then(|o| dict(&self.pdf, o).ok())
                        .and_then(|d| d.get(b"Resources").ok())
                        .filter(|o| object_truthy(o))
                        .map(|resources| Arc::new(resources.clone()))
                })
                .unwrap_or_else(|| current_resources.clone());
            current = self.resolve_scope_fonts(&next)?;
            prefix.push(name);
            self.scope_font_cache
                .insert(prefix.clone(), current.clone());
            current_resources = next;
        }
        Ok(current)
    }
    fn resolve_scope_fonts(&mut self, resources: &Object) -> Result<ScopeFonts> {
        let mut entries = Vec::new();
        let specs = dict(&self.pdf, resources)
            .ok()
            .and_then(|d| d.get(b"Font").ok())
            .and_then(|o| dict(&self.pdf, o).ok());
        if let Some(specs) = specs {
            for (name, obj) in specs.iter() {
                // Prepared resource construction ignores non-dictionary fonts.
                if !matches!(resolve(&self.pdf, obj), Ok(Object::Dictionary(_))) {
                    continue;
                }
                let id = if let Object::Reference(id) = obj {
                    Some(*id)
                } else {
                    None
                };
                let was_cached = id.is_some_and(|i| self.font_cache.contains_key(&i));
                let backend = if let Some(c) = id.and_then(|i| self.font_cache.get(&i)) {
                    Arc::clone(c)
                } else {
                    let f = fonts::FontBackend::from_object_with_cache(
                        &self.pdf,
                        obj,
                        &mut self.font_resources,
                    )
                    .map_err(|e| {
                        if let fonts::FontError::Resource(mut resource) = e {
                            if let Some(id) = id {
                                resource.object_ref.get_or_insert(id);
                            }
                            return Error::from(resource);
                        }
                        Error::new(
                            e.kind(),
                            format!(
                                "font {:?} xref {:?}: {e}",
                                String::from_utf8_lossy(name),
                                id
                            ),
                        )
                    })?;
                    let f = Arc::new(f);
                    if let Some(id) = id {
                        self.font_cache.insert(id, Arc::clone(&f));
                    }
                    f
                };
                let font_name = String::from_utf8_lossy(name).to_string();
                let xref = id.map(|x| x.0 as i64).unwrap_or(0);
                let identity = if id.is_some() {
                    xref
                } else {
                    self.next_direct_font -= 1;
                    self.next_direct_font
                };
                let legacy_descent = *self
                    .legacy_cache
                    .entry((identity, font_name.clone()))
                    .or_insert(if was_cached {
                        0.
                    } else {
                        backend.legacy_descent
                    });
                entries.push(FontEntry {
                    name: font_name,
                    xref,
                    identity,
                    backend,
                    legacy_descent,
                });
            }
        }
        let mut exact = HashMap::new();
        for (index, font) in entries.iter().enumerate() {
            exact.entry(font.name.clone()).or_insert(index);
        }
        Ok(ScopeFonts {
            entries,
            exact,
            resolved: HashMap::new(),
        })
    }
    fn font_boxes(&mut self, f: &FontEntry, descent: f64) -> Vec<Edge> {
        let key = (f.identity, f.name.clone(), descent.to_bits());
        if let Some(v) = self.box_cache.get(&key) {
            return v.iter().map(Edge::share).collect();
        }
        let b = &f.backend;
        let mut boxes = Vec::new();
        if f.xref == 0 {
            return boxes;
        }
        let default_keys = [0, 256];
        let keys = if b.projection_cmap.is_empty() {
            &default_keys[..]
        } else {
            &b.projection_cmap
        };
        for cid in fonts::projection_codes(keys, b.projection_bboxes.len()) {
            if let Some(b) = b.projection_bboxes.get(cid as usize) {
                if *b == [0., 0., 500., 698.] || *b == [0.; 4] {
                    continue;
                }
                boxes.push(
                    self.table
                        .attach(Node::new(NodeData::PdfFontCharBoundingBox(
                            ir::DataPdfFontCharBoundingBox {
                                x: Some(b[0]),
                                y: Some(b[1]),
                                x2: Some(b[2]),
                                y2: Some(b[3]),
                                char_id: Some(cid as i64),
                            },
                        ))),
                )
            }
        }
        self.box_cache
            .insert(key, boxes.iter().map(Edge::share).collect());
        boxes
    }
    fn record_bbox_map(&mut self, f: &FontEntry, descent: f64, form: Option<i64>) {
        let key = (f.identity, f.name.clone(), descent.to_bits());
        let map = if let Some(&map) = self.bbox_map_cache.get(&key) {
            map
        } else {
            let values = BboxMap {
                base: (f.xref != 0).then(|| Arc::clone(&f.backend)),
                ..BboxMap::default()
            };
            let map = self.bbox_maps.len();
            self.bbox_maps.push(values);
            self.bbox_map_cache.insert(key, map);
            map
        };
        if let Some(xid) = form {
            let scope = if let Some(&scope) = self.page_bbox_maps.get(&Some(xid)) {
                scope
            } else {
                let scope = self.bbox_maps.len();
                self.bbox_maps.push(BboxMap::default());
                self.page_bbox_maps.insert(Some(xid), scope);
                scope
            };
            self.bbox_maps[scope].insert((f.xref != 0).then_some(f.xref), BboxValue::Map(map));
        } else {
            self.page_bbox_maps
                .insert((f.xref != 0).then_some(f.xref), map);
        }
    }
    fn character_bbox(&self, xid: i64, xref: i64, cid: u32) -> Result<Option<[f64; 4]>> {
        let xref = (xref != 0).then_some(xref);
        let map = if let Some(&scope) = self.page_bbox_maps.get(&Some(xid)) {
            match self.bbox_maps[scope].get(xref) {
                Some(BboxValue::Map(map)) => map,
                _ => return Ok(None),
            }
        } else {
            let Some(&map) = self.page_bbox_maps.get(&xref) else {
                return Ok(None);
            };
            map
        };
        match self.bbox_maps[map].get(Some(i64::from(cid))) {
            Some(BboxValue::Box(bbox)) => Ok(Some(bbox)),
            Some(BboxValue::Map(map)) => {
                // Python unpacks dictionary keys when this accidental value has
                // length four; otherwise its len(...) == 4 test rejects it.
                let keys = &self.bbox_maps[map];
                if keys.len() != 4 {
                    return Ok(None);
                }
                let mut bbox = [0.; 4];
                for (i, coord) in bbox.iter_mut().enumerate() {
                    *coord = keys.key(i).ok_or_else(|| {
                        Error::new("TypeError", "unsupported operand type for None font xref")
                    })? as f64;
                }
                Ok(Some(bbox))
            }
            None => Ok(None),
        }
    }
    pub(super) fn install_fonts(
        &mut self,
        fs: &ScopeFonts,
        slot: Option<usize>,
        _legacy: bool,
    ) -> Result<()> {
        for f in &fs.entries {
            let b = &f.backend;
            let descent = if self.emitted_fonts.insert((f.identity, f.name.clone())) {
                f.legacy_descent
            } else {
                0.
            };
            let xid = slot.map(|i| self.xobjects[i].xobj_id.unwrap()).unwrap_or(0);
            self.projected_descents.insert(
                (xid, f.name.clone()),
                b.type3_metrics
                    .as_ref()
                    .filter(|_| f.xref != 0)
                    .map(|m| m.descent)
                    .unwrap_or(descent),
            );
            self.record_bbox_map(f, descent, slot.map(|_| xid));
            let boxes = self.font_boxes(f, descent);
            let font = ir::DataPdfFont {
                name: Some(b.font_name.clone()),
                font_id: Some(tokens::canonical_name(&f.name)),
                xref_id: (f.xref != 0).then_some(f.xref),
                encoding_length: Some(b.encoding_length as i64),
                font_subtype: (f.xref != 0).then(|| b.font_subtype.clone()),
                ascent: Some(
                    b.type3_metrics
                        .as_ref()
                        .filter(|_| f.xref != 0)
                        .map(|m| m.ascent)
                        .unwrap_or(b.ascent),
                ),
                descent: Some(
                    b.type3_metrics
                        .as_ref()
                        .filter(|_| f.xref != 0)
                        .map(|m| m.descent)
                        .unwrap_or(descent),
                ),
                type3_em_height: b
                    .type3_metrics
                    .as_ref()
                    .filter(|_| f.xref != 0)
                    .map(|m| m.em_height),
                type3_font_matrix: b
                    .type3_metrics
                    .as_ref()
                    .filter(|_| f.xref != 0)
                    .map(|m| m.matrix_text.clone()),
                type3_font_bbox: b
                    .type3_metrics
                    .as_ref()
                    .filter(|_| f.xref != 0)
                    .map(|m| m.bbox_text.clone()),
                bold: b
                    .style
                    .filter(|_| f.xref != 0)
                    .map(|s| RuntimeBool::Int(s.bold as i64)),
                italic: b
                    .style
                    .filter(|_| f.xref != 0)
                    .map(|s| RuntimeBool::Int(s.italic as i64)),
                monospace: b
                    .style
                    .filter(|_| f.xref != 0)
                    .map(|s| RuntimeBool::Int(s.mono as i64)),
                serif: b
                    .style
                    .filter(|_| f.xref != 0)
                    .map(|s| RuntimeBool::Int(s.serif as i64)),
                pdf_font_char_bounding_box: boxes.iter().map(Edge::share).collect(),
                ..Default::default()
            };
            let e = self.table.attach(Node::new(NodeData::PdfFont(font)));
            if let Some((_, value)) = self
                .available_fonts
                .iter_mut()
                .find(|(name, _)| name == &f.name)
            {
                *value = e.share();
            } else {
                self.available_fonts.push((f.name.clone(), e.share()));
            }
            let fonts = if let Some(slot) = slot {
                &mut self.xobjects[slot].pdf_font
            } else {
                &mut self.page.pdf_font
            }; // Resource keys are unique; inherited fonts are replaced by same id.
               // Keep source order; Python removes the prior font before appending.
               // The inherited font names are tracked by the enclosing scope.
            let names = &mut self.font_names[slot.map(|x| x + 1).unwrap_or(0)];
            if let Some(i) = names.iter().position(|n| n == &f.name) {
                names.remove(i);
                fonts.remove(i);
            }
            names.push(f.name.clone());
            fonts.push(e);
        }
        Ok(())
    }
    pub(super) fn inherited_font_edges(
        &self,
        _fonts: &ScopeFonts,
        _slot: Option<usize>,
    ) -> Vec<Edge> {
        self.available_fonts
            .iter()
            .map(|(_, edge)| edge.share())
            .collect()
    }
    pub(super) fn ensure_text_font(&mut self, st: &State, fonts: &mut ScopeFonts) -> Result<()> {
        let Some(name) = st.font.as_ref() else {
            return Ok(());
        };
        if self.find_font(st, fonts).is_some() {
            return Ok(());
        }
        // PageResourceBundle.get_font creates this backend only when an absent
        // font is actually used, without registering a resource font in the IL.
        let mut spec = lopdf::Dictionary::new();
        spec.set("Subtype", Object::Name(b"Type1".to_vec()));
        spec.set("BaseFont", Object::Name(name.as_bytes().to_vec()));
        let backend = Arc::new(fonts::FontBackend::from_object_with_cache(
            &self.pdf,
            &Object::Dictionary(spec),
            &mut self.font_resources,
        )?);
        self.next_direct_font -= 1;
        self.fallback_fonts.insert(
            name.clone(),
            FontEntry {
                name: "UNKNOW".into(),
                xref: 0,
                identity: self.next_direct_font,
                legacy_descent: backend.legacy_descent,
                backend,
            },
        );
        Ok(())
    }
    fn find_font<'a>(
        &'a mut self,
        st: &State,
        fonts: &'a mut ScopeFonts,
    ) -> Option<&'a FontEntry> {
        let name = st.font.as_ref()?;
        let (resolved, skipped_overlong) = fonts.find_entry(name);
        if skipped_overlong {
            self.diagnostics.long_font_alias_skips += 1;
        }
        resolved
            .and_then(|index| fonts.entries.get(index))
            .or_else(|| self.fallback_fonts.get(name))
    }
    pub(super) fn advance(
        &mut self,
        st: &mut State,
        seq: &[Token],
        fonts: &mut ScopeFonts,
    ) -> Result<()> {
        if let Some(f) = self.find_font(st, fonts).cloned() {
            let b = &f.backend;
            let scale = st.tz * 0.01;
            let cs = st.tc * scale;
            let ws = if b.multibyte { 0. } else { st.tw * scale };
            let dx = 0.001 * st.size * scale;
            let mut need = false;
            for t in seq {
                match t {
                    Token::Int(_) | Token::Real(_) => {
                        if b.vertical {
                            st.line[1] -= t.number()? * dx
                        } else {
                            st.line[0] -= t.number()? * dx
                        }
                        need = true
                    }
                    Token::Str(bytes, _) => {
                        for cid in b.decode(bytes) {
                            let i = if b.vertical { 1 } else { 0 };
                            if need {
                                st.line[i] += cs
                            }
                            st.line[i] += b
                                .width_with_policy(cid, self.pdf.parser_policy.as_ref())?
                                * st.size
                                * scale;
                            if cid == 32 && ws != 0. {
                                st.line[i] += ws
                            }
                            need = true
                        }
                    }
                    _ => {
                        return Err(Error::new(
                            "ValueError",
                            "Expected string or numeric text segment",
                        ));
                    }
                }
            }
        }
        Ok(())
    }
    pub(super) fn text(
        &mut self,
        st: &State,
        seq: &[Token],
        fonts: &mut ScopeFonts,
        xid: i64,
        snap: &Snapshot,
        _clips: Option<&ClipList>,
        buffer: &mut Vec<Buffered>,
    ) -> Result<()> {
        let Some(f) = self.find_font(st, fonts).cloned() else {
            return Ok(());
        };
        let b = &f.backend;
        let matrix = mul(st.tm, st.ctm);
        let scale = st.tz * 0.01;
        let cs = st.tc * scale;
        let ws = if b.multibyte { 0. } else { st.tw * scale };
        let dx = 0.001 * st.size * scale;
        let mut pos = st.line;
        let mut need = false;
        for t in seq {
            match t {
                Token::Int(_) | Token::Real(_) => {
                    let i = if b.vertical { 1 } else { 0 };
                    pos[i] -= t.number()? * dx;
                    need = true
                }
                Token::Str(bytes, _) => {
                    for cid in b.decode(bytes) {
                        let i = if b.vertical { 1 } else { 0 };
                        if need {
                            pos[i] += cs
                        }
                        let m = translate(matrix, pos);
                        let adv = b.width_with_policy(cid, self.pdf.parser_policy.as_ref())?
                            * st.size
                            * scale;
                        if st.mode != 7 {
                            let text = b.unicode_text_with_policy(
                                cid,
                                &format!("(cid:{cid})"),
                                self.pdf.parser_policy.as_ref(),
                            )?;
                            let (ll, ur) = if b.vertical {
                                let (vx, vy) = b.displacement(cid)?;
                                let vx = vx.map(|x| x * st.size * 0.001).unwrap_or(st.size * 0.5);
                                let vy = (1000. - vy) * st.size * 0.001;
                                ([-vx, vy + st.rise + adv], [-vx + st.size, vy + st.rise])
                            } else {
                                let descent = 0.0 * 0.001 * st.size;
                                ([0., descent + st.rise], [adv, descent + st.rise + st.size])
                            };
                            let a = pt(m, ll);
                            let z = pt(m, ur);
                            let bbox = [
                                a[0].min(z[0]),
                                a[1].min(z[1]),
                                a[0].max(z[0]),
                                a[1].max(z[1]),
                            ];
                            let size = if b.vertical || m[0] == 0. {
                                bbox[2] - bbox[0]
                            } else {
                                bbox[3] - bbox[1]
                            };
                            let size = size
                                * b.type3_metrics
                                    .as_ref()
                                    .filter(|_| f.xref != 0)
                                    .map(|m| m.em_height)
                                    .unwrap_or(1.);
                            let rotation = m[1].atan2(m[0]).to_degrees();
                            let order = self.order(); // Even filtered chars consume render order.
                            if (!self
                                .config
                                .visual_roundtrip_mode
                                || st.mode != 3)
                                && ((-0.1..=0.1).contains(&rotation)
                                    || (89.9..=90.1).contains(&rotation))
                            {
                                buffer.push(Buffered::Char {
                                    bbox,
                                    size,
                                    advance: adv,
                                    cid,
                                    text,
                                    font: if f.name == "UNKNOW" {
                                        f.name.clone()
                                    } else {
                                        tokens::canonical_name(
                                            st.font.as_deref().unwrap_or(&f.name),
                                        )
                                    },
                                    font_descent: self
                                        .projected_descents
                                        .get(&(xid, f.name.clone()))
                                        .copied()
                                        .unwrap_or(0.),
                                    font_xref: f.xref,
                                    vertical: m[0] == 0. && m[3] == 0.,
                                    xid,
                                    order,
                                    snap: snap.clone(),
                                    tm: st.tm,
                                    mode: st.mode,
                                })
                            } else {
                                buffer.push(Buffered::FilteredChar);
                            }
                        }
                        pos[i] += adv;
                        if cid == 32 && ws != 0. {
                            pos[i] += ws
                        }
                        need = true
                    }
                }
                _ => {
                    return Err(Error::new(
                        "ValueError",
                        "Expected string or numeric text segment",
                    ));
                }
            }
        }
        Ok(())
    }
    fn render_snapshot(&self, snap: &Snapshot, include: bool, target: Option<Matrix>) -> String {
        let mut kept = Vec::new();
        let mut at = snap.tail;
        while let Some(i) = at {
            let node = self
                .history
                .get(i)
                .expect("snapshot history indices are owned by this parser");
            let ir::HistoryNode {
                op,
                arg,
                replace_key: key,
                version: v,
                ..
            } = node;
            if key
                .as_ref()
                .is_none_or(|k| snap.latest.iter().any(|(x, y)| x == k && y == v))
            {
                if include || !matches!(op.as_str(), "W n" | "W* n") {
                    let text = match arg {
                        ir::HistoryArg::Text(s) => s.clone(),
                        ir::HistoryArg::CtmAware { value, .. } => value.clone(),
                    };
                    let mut result = format!("{text} {op}");
                    if let (Some(target), ir::HistoryArg::CtmAware { ctm, .. }) = (target, arg) {
                        if let PythonValue::Tuple(v) = ctm {
                            let source: Matrix = v
                                .iter()
                                .map(|x| x.as_f64().unwrap())
                                .collect::<Vec<_>>()
                                .try_into()
                                .unwrap();
                            if source != target {
                                let m = mul(inv(target), source);
                                if !m.iter().zip(ID).all(|(x, y)| (*x - y).abs() <= 1e-9) {
                                    result = format!("{} cm {result} {} cm", cm(m), cm(inv(m)))
                                }
                            }
                        }
                    }
                    kept.push(result)
                }
            }
            at = self.history.parent_key(node)
        }
        kept.reverse();
        kept.join(" ")
    }
    fn graphic(
        &mut self,
        snap: &Snapshot,
        include: bool,
        target: Option<Matrix>,
        clips: Option<&ClipList>,
        extra: Option<String>,
        preserve: bool,
    ) -> Edge {
        let mut suffix = if include {
            formatted_clips(clips, target)
        } else {
            Vec::new()
        };
        if let Some(extra) = extra {
            if !extra.is_empty() {
                suffix.push(extra)
            }
        }
        let lazy = snap.count > 256 && !preserve;
        let instruction = if lazy {
            Instructions::View(InstructionView {
                tail: snap.tail,
                latest_versions: snap.latest.clone(),
                include_clipping: include,
                suffix_parts: suffix,
            })
        } else {
            let mut parts = Vec::new();
            let base = self.render_snapshot(snap, include, if preserve { target } else { None });
            if !base.is_empty() {
                parts.push(base)
            }
            parts.extend(suffix);
            let text = parts.join(" ");
            Instructions::Materialized(text)
        };
        if let Some(e) = self.pool.find(&self.table, &instruction) {
            return e.share();
        }
        let hash = pool::GraphicPool::hash(&instruction);
        let e = self
            .table
            .attach(Node::new(NodeData::GraphicState(ir::DataGraphicState {
                passthrough_per_char_instruction: Some(instruction),
            })));
        self.pool.insert(hash, e.share());
        e
    }
    // Commit each completed subtree immediately. Keeping Inline<Box<Node>>
    // until document finalization pads every path to the largest schema payload
    // and makes the temporary tree coexist with the final table/typed Arcs.
    pub(super) fn flush(
        &mut self,
        buffer: &mut Vec<Buffered>,
        expansion: &mut expansion::Expansion,
    ) -> Result<()> {
        for item in buffer.drain(..) {
            match item {
                Buffered::FilteredChar => continue,
                Buffered::Char {
                    bbox,
                    size,
                    advance,
                    cid,
                    text,
                    font,
                    font_descent,
                    font_xref,
                    vertical,
                    xid,
                    order,
                    snap,
                    tm,
                    mode,
                } => {
                    self.valid_text.last_mut().unwrap().push(text.clone());
                    let gs = if self.config.ocr_workaround {
                        if let Some(e) = &self.pool.ocr_black {
                            e.share()
                        } else {
                            let e = self.table.attach(Node::new(NodeData::GraphicState(
                                ir::DataGraphicState {
                                    passthrough_per_char_instruction: Some(
                                        Instructions::Materialized("0 g 0 G".into()),
                                    ),
                                },
                            )));
                            self.pool.ocr_black = Some(e.share());
                            e
                        }
                    } else {
                        self.graphic(&snap, false, None, None, None, false)
                    };
                    let descent = font_descent * size / 1000.;
                    let mut vb = if vertical {
                        [bbox[0] - descent, bbox[1], bbox[2] - descent, bbox[3]]
                    } else {
                        [bbox[0], bbox[1] + descent, bbox[2], bbox[3] + descent]
                    };
                    if let Some(b) = self.character_bbox(xid, font_xref, cid)? {
                        let factor = 1. / 1000. * size;
                        let ll = [bbox[0] + b[0] * factor, bbox[1] + b[1] * factor];
                        let ur = [bbox[0] + b[2] * factor, bbox[1] + b[3] * factor];
                        if (ur[0] - ll[0]) * (ur[1] - ll[1]) > 1. {
                            vb = [ll[0], ll[1], ur[0], ur[1]]
                        }
                    }
                    if size == 0. {
                        continue;
                    }
                    let debug = self
                        .config
                        .visual_roundtrip_mode;
                    let ch=ir::DataPdfCharacter{r#box:Some(box_edge(bbox)),visual_bbox:Some(edge(NodeData::VisualBbox(ir::DataVisualBbox{r#box:Some(box_edge(vb))}))),pdf_style:Some(edge(NodeData::PdfStyle(ir::DataPdfStyle{graphic_state:Some(gs),font_id:Some(font),font_size:Some(size)}))),pdf_character_id:Some(cid as i64),advance:Some(advance),char_unicode:Some(if is_space(&text){" ".into()}else{text}),vertical:Some(vertical),xobj_id:Some(xid),render_order:if self.config.ocr_workaround{None}else{Some(order)},sub_render_order:Some(0),visual_roundtrip_text_scale_x:debug.then_some(tm[0]),visual_roundtrip_text_scale_y:debug.then_some(tm[3]),visual_roundtrip_text_render_mode:debug.then_some(mode),..Default::default()};
                    self.page
                        .pdf_character
                        .push(self.table.attach(Node::new(NodeData::PdfCharacter(ch))));
                }
                Buffered::Curve {
                    path,
                    primitive,
                    ctm,
                    xid,
                    order,
                    snap,
                    clips,
                    stroke,
                    fill,
                    evenodd,
                } => {
                    if !self.config.enable_graphic_element_process {
                        continue;
                    }
                    self.charge_clip_output(clips.as_deref(), expansion, None)?;
                    let mut pts = Vec::new();
                    let mut paths = Vec::new();
                    let mut raw = Vec::new();
                    for seg in path.iter() {
                        let p = if seg.op == "h" {
                            [path[0].coords[0], path[0].coords[1]]
                        } else {
                            let n = seg.coords.len();
                            [seg.coords[n - 2], seg.coords[n - 1]]
                        };
                        pts.push(pt(ctm, p));
                        if seg.coords.is_empty() {
                            paths.push(path_edge("h", None, None, false));
                            raw.push(edge(NodeData::PdfOriginalPath(ir::DataPdfOriginalPath {
                                pdf_path: Some(path_edge("h", Some(0.), Some(0.), false)),
                            })));
                        } else {
                            let count = seg.coords.len() / 2;
                            for (i, p) in seg.coords.chunks(2).enumerate() {
                                let q = pt(ctm, [p[0], p[1]]);
                                let op = if i + 1 == count { seg.op.as_str() } else { "" };
                                paths.push(path_edge(op, Some(q[0]), Some(q[1]), true));
                                raw.push(edge(NodeData::PdfOriginalPath(ir::DataPdfOriginalPath {
                                    pdf_path: Some(path_edge(op, Some(p[0]), Some(p[1]), true)),
                                })))
                            }
                        }
                    }
                    let shape = path.iter().map(|s| s.op.as_str()).collect::<String>();
                    if shape.len() > 3 && shape.ends_with("lh") && pts[pts.len() - 2] == pts[0] {
                        pts.pop();
                    }
                    let bbox = bounds(pts.into_iter());
                    let gs = self.graphic(
                        &snap,
                        true,
                        Some(ctm),
                        clips.as_deref(),
                        None,
                        false,
                    );
                    let curve_ctm = self.curve_ctm(ctm);
                    self.page
                        .pdf_curve
                        .push(
                            self.table
                                .attach(Node::new(NodeData::PdfCurve(ir::DataPdfCurve {
                                    r#box: Some(box_edge(bbox)),
                                    graphic_state: Some(gs),
                                    pdf_path: paths,
                                    pdf_original_path: raw,
                                    pdf_original_path_primitive: primitive.map(|args| {
                                        edge(NodeData::PdfOriginalPathPrimitive(
                                            ir::DataPdfOriginalPathPrimitive {
                                                op: Some("re".into()),
                                                args,
                                            },
                                        ))
                                    }),
                                    fill_background: Some(fill),
                                    stroke_path: Some(stroke),
                                    evenodd: Some(evenodd),
                                    debug_info: Some(CurveDebugInfo::Text("a".into())),
                                    xobj_id: Some(xid),
                                    render_order: Some(order),
                                    ctm: curve_ctm,
                                    ..Default::default()
                                }))),
                        );
                }
                Buffered::Shading {
                    name,
                    ctm,
                    xid,
                    order,
                    snap,
                    clips,
                } => {
                    if !self.config.enable_graphic_element_process {
                        continue;
                    }
                    self.charge_clip_output(clips.as_deref(), expansion, None)?;
                    let gs = self.graphic(
                        &snap,
                        true,
                        Some(ctm),
                        clips.as_deref(),
                        Some(format!("{name} sh")),
                        true,
                    );
                    let curve_ctm = self.curve_ctm(ctm);
                    self.page
                        .pdf_curve
                        .push(
                            self.table
                                .attach(Node::new(NodeData::PdfCurve(ir::DataPdfCurve {
                                    r#box: Some(box_edge([0.; 4])),
                                    graphic_state: Some(gs),
                                    fill_background: Some(false),
                                    stroke_path: Some(false),
                                    evenodd: Some(false),
                                    passthrough_paint: Some(true),
                                    debug_info: Some(CurveDebugInfo::Bool(true)),
                                    xobj_id: Some(xid),
                                    render_order: Some(order),
                                    ctm: curve_ctm,
                                    ..Default::default()
                                }))),
                        );
                }
            }
        }
        Ok(())
    }
    pub(super) fn form(
        &mut self,
        ctm: Matrix,
        xid: i64,
        xref: i64,
        kind: &str,
        name: &str,
        bbox: [f64; 4],
        matrix: Matrix,
        snap: &Snapshot,
        clips: Option<&ClipList>,
        extra: Option<String>,
    ) -> Result<()> {
        let combined = mul(matrix, ctm);
        let b = bounds(
            [
                [bbox[0], bbox[1]],
                [bbox[0] + bbox[2], bbox[1]],
                [bbox[0], bbox[1] + bbox[3]],
                [bbox[0] + bbox[2], bbox[1] + bbox[3]],
            ]
            .into_iter()
            .map(|p| pt(combined, p)),
        );
        let subtype = edge(NodeData::PdfFormSubtype(ir::DataPdfFormSubtype {
            pdf_xobj_form: Some(edge(NodeData::PdfXobjForm(ir::DataPdfXobjForm {
                xref_id: Some(xref),
                do_args: Some(name.into()),
            }))),
            ..Default::default()
        }));
        let gs = self.graphic(snap, true, Some(ctm), clips, extra, false);
        let order = self.order();
        self.page.pdf_form.push(
            self.table
                .attach(Node::new(NodeData::PdfForm(ir::DataPdfForm {
                    r#box: Some(box_edge(b)),
                    graphic_state: Some(gs),
                    pdf_matrix: Some(matrix_edge(ctm)),
                    pdf_affine_transform: Some(affine(ctm)),
                    pdf_form_subtype: Some(subtype),
                    xobj_id: Some(xid),
                    ctm: py_matrix(ctm),
                    render_order: Some(order),
                    form_type: Some(kind.into()),
                    ..Default::default()
                }))),
        );
        Ok(())
    }
    pub(super) fn inline_image(
        &mut self,
        ctm: Matrix,
        xid: i64,
        snap: &Snapshot,
        clips: Option<&ClipList>,
        params: &[(String, Token)],
        data: &[u8],
    ) -> Result<()> {
        let image_parameters = format!(
            "{{{}}}",
            params
                .iter()
                .map(|(k, v)| format!("{}: {}", json_string(k), inline_json(v)))
                .collect::<Vec<_>>()
                .join(", ")
        );
        let b = bounds(
            [[0., 0.], [1., 0.], [0., 1.], [1., 1.]]
                .into_iter()
                .map(|p| pt(ctm, p)),
        );
        let gs = self.graphic(snap, true, Some(ctm), clips, None, false);
        let order = self.order();
        let subtype = edge(NodeData::PdfFormSubtype(ir::DataPdfFormSubtype {
            pdf_inline_form: Some(edge(NodeData::PdfInlineForm(ir::DataPdfInlineForm {
                form_data: Some(base64(data)),
                image_parameters: Some(image_parameters),
            }))),
            ..Default::default()
        }));
        self.page.pdf_form.push(
            self.table
                .attach(Node::new(NodeData::PdfForm(ir::DataPdfForm {
                    r#box: Some(box_edge(b)),
                    graphic_state: Some(gs),
                    pdf_matrix: Some(matrix_edge(ctm)),
                    pdf_affine_transform: Some(affine(ctm)),
                    pdf_form_subtype: Some(subtype),
                    xobj_id: Some(xid),
                    ctm: py_matrix(ctm),
                    render_order: Some(order),
                    form_type: Some("image".into()),
                    ..Default::default()
                }))),
        );
        Ok(())
    }
}

#[cfg(test)]
mod projection_tests;
fn path_edge(op: &str, x: Option<f64>, y: Option<f64>, has: bool) -> Edge {
    edge(NodeData::PdfPath(ir::DataPdfPath::new(
        x,
        y,
        Some(op.into()),
        Some(has),
    )))
}
fn matrix_edge(m: Matrix) -> Edge {
    edge(NodeData::PdfMatrix(ir::DataPdfMatrix {
        a: Some(m[0]),
        b: Some(m[1]),
        c: Some(m[2]),
        d: Some(m[3]),
        e: Some(m[4]),
        f: Some(m[5]),
    }))
}
// Compensated sum of two squares, followed by a square-root correction.
// Python math.hypot uses compensated arithmetic rather than the platform hypot.
fn python_hypot(a: f64, b: f64) -> f64 {
    let max = a.abs().max(b.abs());
    if a.is_infinite() || b.is_infinite() {
        return f64::INFINITY;
    }
    if !a.is_finite() || !b.is_finite() {
        return f64::NAN;
    }
    if max == 0. {
        return 0.;
    }
    let exponent = ((max.to_bits() >> 52) & 2047) as i32 - 1023;
    let scale = 2f64.powi(-exponent.clamp(-1022, 1022));
    let x = a * scale;
    let y = b * scale;
    let xx = x * x;
    let yy = y * y;
    let sum = xx + yy;
    let z = sum - xx;
    let error = (xx - (sum - z)) + (yy - z) + x.mul_add(x, -xx) + y.mul_add(y, -yy);
    let h = sum.sqrt();
    let correction = h.mul_add(-h, sum) + error;
    (h + correction / (2. * h)) / scale
}
fn affine(m: Matrix) -> Edge {
    let [a, b, c, d, e, f] = m;
    let sx = python_hypot(a, b);
    let det = a * d - c * b;
    let (sy, shear, rotation) = if sx < 1e-12 {
        let sy = python_hypot(c, d);
        (if det < 0. { -sy } else { sy }, 0., 0.)
    } else {
        let (rx, ry) = (a / sx, b / sx);
        let mut shear = rx * c + ry * d;
        let mut sy = python_hypot(c - shear * rx, d - shear * ry);
        if det < 0. {
            sy = -sy;
            shear = -shear
        }
        (sy, shear, b.atan2(a))
    };
    edge(NodeData::PdfAffineTransform(ir::DataPdfAffineTransform {
        translation_x: Some(e),
        translation_y: Some(f),
        rotation: Some(rotation),
        scale_x: Some(sx),
        scale_y: Some(sy),
        shear: Some(shear),
    }))
}
fn is_space(s: &str) -> bool {
    crate::valid_text::is_space(s)
}
fn json_string(text: &str) -> String {
    let mut out = String::from("\"");
    for c in text.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            c if c < ' ' || c as u32 >= 128 => {
                for u in c.encode_utf16(&mut [0; 2]) {
                    out.push_str(&format!("\\u{u:04x}"))
                }
            }
            _ => out.push(c),
        }
    }
    out.push('"');
    out
}
fn inline_json(token: &Token) -> String {
    match token {
        Token::Int(n) => n.to_string(),
        Token::Real(n) => crate::python_float_str(*n),
        Token::Array(v) => format!(
            "[{}]",
            v.iter().map(inline_json).collect::<Vec<_>>().join(", ")
        ),
        Token::Dict(v) => format!(
            "{{{}}}",
            v.iter()
                .map(|(k, v)| format!("{}: {}", json_string(k), inline_json(v)))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        Token::Word(s) if ["true", "false", "null"].contains(&s.as_str()) => s.clone(),
        Token::Str(bytes, _) => json_string(&format!(
            "<{}>",
            bytes.iter().map(|x| format!("{x:02X}")).collect::<String>()
        )),
        _ => json_string(&token.pdf()),
    }
}
fn base64(data: &[u8]) -> String {
    let a = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut s = String::new();
    for b in data.chunks(3) {
        let n = ((b[0] as u32) << 16)
            | ((b.get(1).copied().unwrap_or(0) as u32) << 8)
            | b.get(2).copied().unwrap_or(0) as u32;
        s.push(a[((n >> 18) & 63) as usize] as char);
        s.push(a[((n >> 12) & 63) as usize] as char);
        s.push(if b.len() > 1 {
            a[((n >> 6) & 63) as usize] as char
        } else {
            '='
        });
        s.push(if b.len() > 2 {
            a[(n & 63) as usize] as char
        } else {
            '='
        });
    }
    s
}
#[cfg(test)]
mod hypot_tests;
