use crate::{AvailableFonts, PageOutput, Result, WriterError};
use lopdf::{Dictionary, Document, Object, ObjectId, Stream};
use std::collections::BTreeSet;

/// Parser-selected source-page objects in full-document ordinal order.
///
/// This is runtime writer metadata, not IR. It deliberately has no serde
/// implementation so callers cannot accidentally add it to an IR artifact.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct PageObjectMap {
    ids: Vec<ObjectId>,
}

impl PageObjectMap {
    pub fn new(ids: Vec<ObjectId>) -> Self {
        Self { ids }
    }

    pub fn len(&self) -> usize {
        self.ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }
}

/// Page lookup for production writing. Missing sidecar data and missing source
/// objects are fatal so writer cannot silently select a different page tree.
pub fn page_object_with_map(
    doc: &Document,
    page_objects: &PageObjectMap,
    number: i64,
) -> Result<ObjectId> {
    if page_objects.is_empty() {
        return Err(WriterError::Invalid("page object map is empty".into()));
    }
    let index =
        usize::try_from(number).map_err(|_| WriterError::Invalid("invalid page index".into()))?;
    let id = page_objects.ids.get(index).copied().ok_or_else(|| {
        WriterError::Invalid(format!(
            "page {number} out of range for page object map (len {})",
            page_objects.len()
        ))
    })?;
    if !doc.objects.contains_key(&id) {
        return Err(WriterError::Invalid(format!(
            "page {number} object {id:?} missing from writer PDF"
        )));
    }
    Ok(id)
}

/// Source-page discovery retained only for the frozen `writer_probe`.
///
/// Production bridge entries must carry parser page objects and use
/// [`page_object_with_map`] or another `*_with_page_objects` writer entry.
#[doc(hidden)]
pub fn legacy_page_objects(doc: &Document) -> PageObjectMap {
    PageObjectMap::new(doc.get_pages().into_values().collect())
}

#[derive(Debug, Default)]
pub struct ApplyCounts {
    pub gs_added: usize,
    pub sh_added: usize,
    pub form_failures: Vec<(i64, String)>,
}
/// Compatibility helper for the frozen `p2_stage_diff::writer_probe`.
///
/// Production writers use [`page_object_with_map`].
pub fn page_object(doc: &Document, number: i64) -> Result<ObjectId> {
    let n =
        u32::try_from(number + 1).map_err(|_| WriterError::Invalid("invalid page index".into()))?;
    doc.get_pages()
        .get(&n)
        .copied()
        .ok_or_else(|| WriterError::Invalid(format!("page {number} missing")))
}
pub(crate) fn xref(doc: &Document, number: i64) -> Result<ObjectId> {
    let n = u32::try_from(number).map_err(|_| WriterError::Invalid("invalid xref".into()))?;
    doc.objects
        .keys()
        .find(|(id, _)| *id == n)
        .copied()
        .ok_or_else(|| WriterError::Invalid(format!("xref {number} missing")))
}
pub(crate) fn dict(o: &Object) -> Result<&Dictionary> {
    match o {
        Object::Stream(s) => Ok(&s.dict),
        _ => Ok(o.as_dict()?),
    }
}
fn dict_mut(o: &mut Object) -> Result<&mut Dictionary> {
    match o {
        Object::Stream(s) => Ok(&mut s.dict),
        _ => Ok(o.as_dict_mut()?),
    }
}
fn deref<'a>(doc: &'a Document, o: &'a Object) -> Result<&'a Object> {
    match o {
        Object::Reference(id) => Ok(doc.get_object(*id)?),
        _ => Ok(o),
    }
}
pub(crate) fn path<'a>(doc: &'a Document, id: ObjectId, keys: &[&[u8]]) -> Result<&'a Object> {
    let mut o = doc.get_object(id)?;
    for key in keys {
        o = dict(deref(doc, o)?)?.get(key)?;
    }
    Ok(o)
}
/// Resources are read from the object's own dictionary without page-tree inheritance.
pub fn available_font_names(doc: &Document, id: ObjectId) -> BTreeSet<String> {
    let read = || -> Result<BTreeSet<String>> {
        let fonts = deref(doc, path(doc, id, &[b"Resources", b"Font"])?)?;
        Ok(dict(fonts)?
            .iter()
            .map(|(k, _)| String::from_utf8_lossy(k).into_owned())
            .collect())
    };
    read().unwrap_or_default()
}
/// Compatibility entry for the frozen `p2_stage_diff::writer_probe`.
///
/// Production bridge writing uses [`available_fonts_with_page_objects`].
pub fn available_fonts(doc: &Document, output: &PageOutput) -> Result<AvailableFonts> {
    let page = available_font_names(doc, page_object(doc, output.page_number)?);
    Ok(available_fonts_at(doc, output, page))
}
pub fn available_fonts_with_page_objects(
    doc: &Document,
    output: &PageOutput,
    page_objects: &PageObjectMap,
) -> Result<AvailableFonts> {
    let page = available_font_names(
        doc,
        page_object_with_map(doc, page_objects, output.page_number)?,
    );
    Ok(available_fonts_at(doc, output, page))
}
fn available_fonts_at(
    doc: &Document,
    output: &PageOutput,
    page: BTreeSet<String>,
) -> AvailableFonts {
    let mut forms = std::collections::HashMap::new();
    for f in &output.forms {
        let mut names = page.clone();
        if let Ok(id) = xref(doc, f.xref_id) {
            names.extend(available_font_names(doc, id));
        }
        forms.insert(f.xobj_id, names);
    }
    AvailableFonts { page, forms }
}
pub fn replace_stream(doc: &mut Document, id: ObjectId, bytes: Vec<u8>) -> Result<()> {
    let object = doc.get_object_mut(id)?;
    match object {
        Object::Stream(s) => s.set_plain_content(bytes),
        Object::Dictionary(d) => {
            let mut s = Stream::new(d.clone(), bytes);
            s.dict.remove(b"Filter");
            s.dict.remove(b"DecodeParms");
            *object = Object::from(s);
        }
        _ => {
            return Err(WriterError::Invalid(
                "update_stream requires dictionary".into(),
            ));
        }
    }
    Ok(())
}
/// Resource reads cross references, writes deliberately reject intermediate references.
pub(crate) fn set_direct(
    doc: &mut Document,
    id: ObjectId,
    keys: &[&[u8]],
    value: Object,
) -> Result<()> {
    fn write(o: &mut Object, keys: &[&[u8]], value: Object) -> Result<()> {
        let d = dict_mut(o)?;
        if keys.len() == 1 {
            d.set(keys[0], value);
            return Ok(());
        }
        if d.get(keys[0]).is_err() || matches!(d.get(keys[0]), Ok(Object::Null)) {
            d.set(keys[0], Dictionary::new());
        }
        let child = d.get_mut(keys[0])?;
        if matches!(child, Object::Reference(_)) {
            return Err(WriterError::Invalid("path has indirects".into()));
        }
        // MuPDF replaces non-dictionary intermediates (including names) while
        // creating a nested key. Keep the special rejection for references.
        if !matches!(child, Object::Dictionary(_) | Object::Stream(_)) {
            *child = Object::Dictionary(Dictionary::new());
        }
        write(child, &keys[1..], value)
    }
    write(doc.get_object_mut(id)?, keys, value)
}
fn set_resource(
    doc: &mut Document,
    id: ObjectId,
    kind: &[u8],
    name: &[u8],
    value: Object,
) -> Result<()> {
    match set_direct(doc, id, &[b"Resources", kind, name], value.clone()) {
        Ok(()) => return Ok(()),
        Err(e) if !e.to_string().contains("has indirects") => return Err(e),
        Err(_) => (),
    }
    // Reproduce the Python fallback, including its generation-zero-only parse.
    let Object::Reference(resources) = path(doc, id, &[b"Resources"])? else {
        return Err(WriterError::Invalid("path has indirects".into()));
    };
    let resources = *resources;
    if resources.1 != 0 {
        return Err(WriterError::Invalid("path has indirects".into()));
    }
    match path(doc, resources, &[kind]) {
        Ok(Object::Reference(k)) if k.1 == 0 => set_direct(doc, *k, &[name], value),
        _ => set_direct(doc, resources, &[kind, name], value),
    }
}
fn name_char(c: u8) -> bool {
    c.is_ascii_alphanumeric() || b"!#$%&'*+,-.:;=?@\\^_`{|}~".contains(&c)
}
fn names(stream: &[u8], op: &[u8]) -> BTreeSet<Vec<u8>> {
    let mut result = BTreeSet::new();
    for (i, c) in stream.iter().enumerate() {
        if *c != b'/' {
            continue;
        }
        let mut end = i + 1;
        while stream.get(end).is_some_and(|c| name_char(*c)) {
            end += 1;
        }
        if end == i + 1 {
            continue;
        }
        let mut at = end;
        while stream
            .get(at)
            .is_some_and(|c| matches!(c, b' ' | b'\t' | b'\n' | b'\r' | b'\x0b' | b'\x0c'))
        {
            at += 1;
        }
        if at == end {
            continue;
        }
        if stream.get(at..at + op.len()) == Some(op)
            && !stream
                .get(at + op.len())
                .is_some_and(|c| c.is_ascii_alphanumeric() || *c == b'_')
        {
            result.insert(stream[i + 1..end].to_vec());
        }
    }
    result
}
fn ensure(
    doc: &mut Document,
    id: ObjectId,
    stream: &[u8],
    candidates: &[ObjectId],
    kind: &[u8],
    op: &[u8],
    count: &mut usize,
) -> Result<()> {
    for name in names(stream, op) {
        if path(doc, id, &[b"Resources", kind, &name]).is_ok_and(|v| !matches!(v, Object::Null)) {
            continue;
        }
        let found = candidates.iter().find_map(|c| {
            path(doc, *c, &[b"Resources", kind, &name])
                .ok()
                .filter(|v| !matches!(v, Object::Null))
                .cloned()
        });
        if let Some(value) = found {
            set_resource(doc, id, kind, &name, value)?;
            *count += 1;
        }
    }
    Ok(())
}
/// Apply one page. The caller must invoke this in IR page order; no parallel mutation.
///
/// Compatibility entry for the frozen `p2_stage_diff::writer_probe`.
/// Production bridge writing uses [`apply_page_output_with_page_objects`].
pub fn apply_page_output(doc: &mut Document, output: &PageOutput) -> Result<ApplyCounts> {
    let page = page_object(doc, output.page_number)?;
    apply_page_output_at(doc, output, page)
}
pub fn apply_page_output_with_page_objects(
    doc: &mut Document,
    output: &PageOutput,
    page_objects: &PageObjectMap,
) -> Result<ApplyCounts> {
    let page = page_object_with_map(doc, page_objects, output.page_number)?;
    apply_page_output_at(doc, output, page)
}
fn apply_page_output_at(
    doc: &mut Document,
    output: &PageOutput,
    page: ObjectId,
) -> Result<ApplyCounts> {
    let mut candidates = vec![page];
    // Invalid Form IDs remain skippable in the candidate search, as Python's getter errors are swallowed.
    candidates.extend(
        output
            .forms
            .iter()
            .filter_map(|f| xref(doc, f.xref_id).ok()),
    );
    let mut counts = ApplyCounts::default();
    for form in &output.forms {
        let mut update = || -> Result<()> {
            let id = xref(doc, form.xref_id)?;
            ensure(
                doc,
                id,
                &form.stream,
                &candidates,
                b"ExtGState",
                b"gs",
                &mut counts.gs_added,
            )?;
            ensure(
                doc,
                id,
                &form.stream,
                &candidates,
                b"Shading",
                b"sh",
                &mut counts.sh_added,
            )?;
            replace_stream(doc, id, form.stream.clone())
        };
        if let Err(e) = update() {
            log::warn!("update xref {} stream fail, continue", form.xref_id);
            counts.form_failures.push((form.xref_id, e.to_string()));
        }
    }
    // Allocate before supplementing page resources, preserving failed-attempt side effects.
    let id = doc.add_object(Dictionary::new());
    ensure(
        doc,
        page,
        &output.stream,
        &candidates,
        b"ExtGState",
        b"gs",
        &mut counts.gs_added,
    )?;
    ensure(
        doc,
        page,
        &output.stream,
        &candidates,
        b"Shading",
        b"sh",
        &mut counts.sh_added,
    )?;
    replace_stream(doc, id, output.stream.clone())?;
    doc.get_object_mut(page)?
        .as_dict_mut()?
        .set("Contents", Object::Reference(id));
    Ok(counts)
}
/// Compatibility entry for the frozen `p2_stage_diff::writer_probe`.
///
/// Production bridge writing uses [`first_content_stream_with_page_objects`].
pub fn first_content_stream(doc: &Document, page_number: i64) -> Result<(ObjectId, Vec<u8>)> {
    first_content_stream_at(doc, page_object(doc, page_number)?)
}
pub fn first_content_stream_with_page_objects(
    doc: &Document,
    page_number: i64,
    page_objects: &PageObjectMap,
) -> Result<(ObjectId, Vec<u8>)> {
    first_content_stream_at(doc, page_object_with_map(doc, page_objects, page_number)?)
}
fn first_content_stream_at(doc: &Document, page: ObjectId) -> Result<(ObjectId, Vec<u8>)> {
    let contents = path(doc, page, &[b"Contents"])?;
    let first = match contents {
        Object::Array(a) => a
            .first()
            .ok_or_else(|| WriterError::Invalid("empty Contents".into()))?,
        _ => contents,
    };
    let id = first.as_reference()?;
    Ok((id, doc.get_object(id)?.as_stream()?.get_plain_content()?))
}

#[cfg(test)]
mod pdf_tests;
