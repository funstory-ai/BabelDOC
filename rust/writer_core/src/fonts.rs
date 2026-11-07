//! Serial font installation. Assets are borrowed; this module does not create,
//! own, or decide the lifetime of a FontMapper or its Python handle.
use crate::{Result, WriterError, page_object, page_object_with_map, pdf};
pub use babeldoc_font_core::FontFace;
use babeldoc_ir_core::{Cursor, PdfFontInit, RuntimeBool, TreeView, WriterView, types};
use lopdf::{Dictionary, Document, Object, ObjectId, Stream, dictionary};
use std::collections::{HashMap, HashSet};

pub struct FontMetadata {
    pub font_name: String,
    pub ascent: f64,
    pub descent: f64,
    pub encoding_length: i64,
}
pub struct FontAsset<'a> {
    /// Resource name / IR font_id, including the Python mapper's "base" alias.
    pub id: &'a str,
    pub face: &'a FontFace,
    pub metadata: &'a FontMetadata,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PageFontRef {
    pub xref_id: i64,
    pub refname: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum PdfObjectIdentity {
    Indirect(ObjectId),
    Direct(usize),
}

fn resolve_object<'a>(
    pdf: &'a Document,
    object: &'a Object,
) -> Option<(&'a Object, Option<ObjectId>)> {
    let mut current = object;
    let mut first_reference = None;
    let mut seen = HashSet::new();
    while let Object::Reference(id) = current {
        if !seen.insert(*id) {
            return None;
        }
        first_reference.get_or_insert(*id);
        current = pdf.get_object(*id).ok()?;
    }
    Some((current, first_reference))
}

fn object_dict(object: &Object) -> Option<&Dictionary> {
    match object {
        Object::Dictionary(dict) => Some(dict),
        Object::Stream(stream) => Some(&stream.dict),
        _ => None,
    }
}

fn object_identity(pdf: &Document, object: &Object) -> Option<PdfObjectIdentity> {
    let (resolved, reference) = resolve_object(pdf, object)?;
    if let Some(reference) = reference {
        return Some(PdfObjectIdentity::Indirect(reference));
    }
    let dict = object_dict(resolved)?;
    Some(PdfObjectIdentity::Direct(std::ptr::from_ref(dict) as usize))
}

fn resolved_dict<'a>(pdf: &'a Document, object: &'a Object) -> Option<&'a Dictionary> {
    object_dict(resolve_object(pdf, object)?.0)
}

fn inheritable_page_resources(pdf: &Document, page: ObjectId) -> Option<&Object> {
    let mut current = pdf.get_object(page).ok()?;
    let mut seen = HashSet::new();
    loop {
        let identity = object_identity(pdf, &current)?;
        if !seen.insert(identity) {
            return None;
        }
        let (resolved, _) = resolve_object(pdf, &current)?;
        let dict = object_dict(resolved)?;
        if let Ok(resources) = dict.get(b"Resources") {
            if !matches!(resources, Object::Null) {
                return Some(resources);
            }
        }
        current = dict.get(b"Parent").ok()?;
    }
}

struct ResourceScanFrame<'a> {
    identity: PdfObjectIdentity,
    xobjects: Vec<&'a Object>,
    next_xobject: usize,
}

fn enter_resource_frame<'a>(
    pdf: &'a Document,
    resources: &'a Object,
    active_resources: &mut HashSet<PdfObjectIdentity>,
    fonts: &mut Vec<PageFontRef>,
) -> Option<ResourceScanFrame<'a>> {
    let Some(identity) = object_identity(pdf, resources) else {
        return None;
    };
    if !active_resources.insert(identity) {
        return None;
    }

    let Some(resources_dict) = resolved_dict(pdf, resources) else {
        active_resources.remove(&identity);
        return None;
    };
    if let Ok(font_object) = resources_dict.get(b"Font") {
        if let Some(font_dict) = resolved_dict(pdf, font_object) {
            for (refname, font_object) in font_dict.iter() {
                let Some((resolved_font, xref)) = resolve_object(pdf, font_object) else {
                    continue;
                };
                if object_dict(resolved_font).is_some() {
                    fonts.push(PageFontRef {
                        xref_id: xref.map_or(0, |id| i64::from(id.0)),
                        refname: String::from_utf8_lossy(refname).into_owned(),
                    });
                }
            }
        }
    }

    let xobjects = resources_dict
        .get(b"XObject")
        .ok()
        .and_then(|object| resolved_dict(pdf, object))
        .map(|dict| dict.iter().map(|(_, object)| object).collect())
        .unwrap_or_default();
    Some(ResourceScanFrame {
        identity,
        xobjects,
        next_xobject: 0,
    })
}

fn scan_resources(
    pdf: &Document,
    resources: &Object,
    active_resources: &mut HashSet<PdfObjectIdentity>,
    tracer: &mut Vec<i64>,
    fonts: &mut Vec<PageFontRef>,
) {
    let Some(frame) = enter_resource_frame(pdf, resources, active_resources, fonts) else {
        return;
    };
    let mut stack = vec![frame];

    // PyMuPDF's JM_scan_resources recursively follows XObject Resources in
    // Python. Use an explicit heap-backed stack here so a valid deep form tree
    // cannot exhaust the native Rust call stack.
    while !stack.is_empty() {
        let xobject = {
            let frame = stack.last_mut().expect("nonempty resource stack");
            if frame.next_xobject == frame.xobjects.len() {
                None
            } else {
                let xobject = frame.xobjects[frame.next_xobject];
                frame.next_xobject += 1;
                Some(xobject)
            }
        };
        let Some(xobject) = xobject else {
            let frame = stack.pop().expect("nonempty resource stack");
            active_resources.remove(&frame.identity);
            continue;
        };

        let Some((resolved_xobject, reference)) = resolve_object(pdf, xobject) else {
            continue;
        };
        let stream_xref = if matches!(resolved_xobject, Object::Stream(_)) {
            reference.map_or(0, |id| i64::from(id.0))
        } else {
            0
        };
        let Some(xobject_dict) = object_dict(resolved_xobject) else {
            continue;
        };
        let Ok(subresources) = xobject_dict.get(b"Resources") else {
            continue;
        };
        if matches!(subresources, Object::Null) {
            continue;
        }
        if tracer.contains(&stream_xref) {
            // JM_scan_resources returns from this resource level on a
            // repeated XObject xref, including the sxref=0 non-stream case.
            let frame = stack.last_mut().expect("nonempty resource stack");
            frame.next_xobject = frame.xobjects.len();
            continue;
        }
        tracer.push(stream_xref);
        if let Some(child) = enter_resource_frame(pdf, subresources, active_resources, fonts) {
            stack.push(child);
        }
    }
}

/// Enumerate the font rows that `Page.get_fonts()` exposes for page 0, in the
/// order used by PyMuPDF's recursive `JM_scan_resources` walk.
///
/// Compatibility entry for the frozen `p2_stage_diff::writer_probe`.
/// Production font registration uses [`page0_font_refs_with_page_objects`].
pub fn page0_font_refs(pdf: &Document) -> Result<Vec<PageFontRef>> {
    page0_font_refs_at(pdf, page_object(pdf, 0)?)
}
pub fn page0_font_refs_with_page_objects(
    pdf: &Document,
    page_objects: &pdf::PageObjectMap,
) -> Result<Vec<PageFontRef>> {
    page0_font_refs_at(pdf, page_object_with_map(pdf, page_objects, 0)?)
}
fn page0_font_refs_at(pdf: &Document, page: ObjectId) -> Result<Vec<PageFontRef>> {
    let mut fonts = Vec::new();
    if let Some(resources) = inheritable_page_resources(pdf, page) {
        scan_resources(
            pdf,
            resources,
            &mut HashSet::new(),
            &mut Vec::new(),
            &mut fonts,
        );
    }
    Ok(fonts)
}
#[derive(Debug)]
pub struct InstalledFont {
    pub font_id: String,
    pub xref_id: i64,
}
#[derive(Debug)]
pub struct FontInstallResult {
    pub total: u64,
    pub installed: Vec<InstalledFont>,
}

/// The formula-only blind spot is intentional 0.6.4 behavior.
pub fn used_font_ids(view: &TreeView<'_, WriterView>) -> Result<HashSet<String>> {
    let mut used = HashSet::new();
    let mut visit = |char: Cursor<types::PdfCharacter>| -> Result<()> {
        if let Some(style) = char.pdf_style(view)? {
            if let Some(id) = style.font_id(view)? {
                if !id.is_empty() {
                    used.insert(id.clone());
                }
            }
        }
        Ok(())
    };
    for page in view.document().page_iter(view)? {
        let page = page?;
        for character in page.pdf_character_iter(view)? {
            let char = character?;
            visit(char)?;
        }
        for paragraph in page.pdf_paragraph_iter(view)? {
            let para = paragraph?;
            for composition in para.pdf_paragraph_composition_iter(view)? {
                let comp = composition?;
                if let Some(char) = comp.pdf_character(view)? {
                    visit(char)?;
                }
            }
        }
    }
    Ok(used)
}

fn tounicode(map: &std::collections::BTreeMap<u32, u32>) -> Vec<u8> {
    let mut text = String::from(
        "/CIDInit /ProcSet findresource begin\n12 dict begin\nbegincmap\n/CIDSystemInfo <</Registry(Adobe)/Ordering(UCS)/Supplement 0>> def\n/CMapName /Adobe-Identity-UCS def\n/CMapType 2 def\n1 begincodespacerange\n<0000> <FFFF>\nendcodespacerange\n",
    );
    // Use MuPDF's hexadecimal scalar representation, including its >FFFF
    // destinations. Do not introduce a UTF-16 normalization in the port.
    let entries: Vec<_> = map.iter().map(|(&g, &u)| (g, u)).collect();
    let mut ranges = vec![];
    let mut chars = vec![];
    let mut i = 0;
    while i < entries.len() {
        let (first, code) = entries[i];
        let mut j = i + 1;
        while j < entries.len()
            && entries[j].0 == entries[j - 1].0 + 1
            && entries[j].1 == entries[j - 1].1 + 1
            && entries[j].0 >> 8 == first >> 8
        {
            j += 1;
        }
        if j > i + 1 {
            ranges.push((first, entries[j - 1].0, code));
        } else {
            chars.push((first, code));
        }
        i = j;
    }
    // MuPDF's range vs char split is observable for its >FFFF hex values:
    // identical integer maps in different operators can extract differently.
    for chunk in ranges.chunks(100) {
        text.push_str(&format!("{} beginbfrange\n", chunk.len()));
        for (first, last, code) in chunk {
            text.push_str(&format!("<{first:04x}> <{last:04x}> <{code:04x}>\n"));
        }
        text.push_str("endbfrange\n");
    }
    for chunk in chars.chunks(100) {
        text.push_str(&format!("{} beginbfchar\n", chunk.len()));
        for (gid, code) in chunk {
            text.push_str(&format!("<{gid:04x}> <{code:04x}>\n"));
        }
        text.push_str("endbfchar\n");
    }
    text.push_str("endcmap\nCMapName currentdict /CMap defineresource pop\nend\nend\n");
    text.into_bytes()
}

/// Construct the six Type0 / Identity-H objects. Resource installation is a
/// separate serial step so this constructor also serves the 34-font unit oracle.
pub fn embed_font(pdf: &mut Document, asset: &FontAsset<'_>) -> Result<ObjectId> {
    asset
        .face
        .pdf_font_flags()
        .map_err(|e| WriterError::Invalid(e.to_string()))?;
    let data = asset
        .face
        .pdf_embedding()
        .map_err(|e| WriterError::Invalid(e.to_string()))?;
    let id = pdf.new_object_id();
    let unicode = pdf.add_object(Stream::new(Dictionary::new(), tounicode(&data.unicode)));
    let file = pdf.add_object(Stream::new(
        dictionary! {"Length1"=>data.data.len() as i64},
        data.data.to_vec(),
    ));
    let descriptor = pdf.add_object(dictionary! {
        "Type"=>"FontDescriptor", "FontName"=>Object::Name(asset.metadata.font_name.as_bytes().to_vec()),
        "FontBBox"=>Object::Array(data.bbox.iter().map(|v| Object::Real(*v)).collect()),
        "ItalicAngle"=>0, "Ascent"=>asset.metadata.ascent as i64,
        "Descent"=>asset.metadata.descent as i64, "StemV"=>80, "Flags"=>32,
        "FontFile2"=>file,
    });
    let widths = pdf.add_object(Object::Array(vec![
        0.into(),
        Object::Array(data.widths.into_iter().map(Object::Integer).collect()),
    ]));
    let descendant = pdf.add_object(dictionary! {
        "Type"=>"Font", "Subtype"=>"CIDFontType2",
        "CIDSystemInfo"=>dictionary!{"Registry"=>Object::string_literal("Adobe"),"Ordering"=>Object::string_literal("Identity"),"Supplement"=>0},
        "BaseFont"=>Object::Name(data.postscript_name.into_bytes()),"FontDescriptor"=>descriptor,"W"=>widths,
    });
    pdf.objects.insert(id,Object::Dictionary(dictionary! {
        "Type"=>"Font", "Subtype"=>"Type0","BaseFont"=>Object::Name(asset.metadata.font_name.as_bytes().to_vec()),
        "Encoding"=>"Identity-H","ToUnicode"=>unicode,"DescendantFonts"=>vec![Object::Reference(descendant)],
    }));
    Ok(id)
}

fn resource_owner(pdf: &Document, mut page: ObjectId, page0: ObjectId) -> Result<ObjectId> {
    let mut seen = HashSet::new();
    while seen.insert(page) {
        let object = pdf::dict(pdf.get_object(page)?)?;
        if object
            .get(b"Resources")
            .is_ok_and(|v| !matches!(v, Object::Null))
        {
            return Ok(page);
        }
        match object.get(b"Parent") {
            Ok(Object::Reference(parent)) => page = *parent,
            _ => return Ok(page0),
        }
    }
    Err(WriterError::Invalid("cyclic page tree".into()))
}
fn existing_page_font<'a>(id: &str, page_fonts: &'a [PageFontRef]) -> Option<&'a PageFontRef> {
    let refname = id.strip_prefix('/').unwrap_or(id);
    page_fonts.iter().find(|font| font.refname == refname)
}
fn insert_page0(
    pdf: &mut Document,
    asset: &FontAsset<'_>,
    cached: &mut HashMap<std::path::PathBuf, i64>,
    page_fonts: &[PageFontRef],
    page0: ObjectId,
) -> Result<i64> {
    // PyMuPDF Page.insert_font strips one optional leading slash before
    // CheckFont compares the name against Page.get_fonts()[i][4]. Keep the
    // supplied order so duplicate refnames retain first-match behavior.
    if let Some(font) = existing_page_font(asset.id, page_fonts) {
        if font.xref_id == 0 {
            return Err(WriterError::Invalid("xref is not a font".into()));
        }
        return Ok(font.xref_id);
    }

    let owner = resource_owner(pdf, page0, page0)?;
    let (target, prefix): (ObjectId, Vec<&[u8]>) = match pdf::path(pdf, owner, &[b"Resources"]) {
        Ok(Object::Reference(r)) => (*r, vec![]),
        _ => (owner, vec![b"Resources"]),
    };
    let mut keys = prefix.clone();
    keys.push(b"Font");
    let (target, mut keys) = match pdf::path(pdf, target, &keys) {
        Ok(Object::Reference(r)) => (*r, vec![]),
        _ => (target, keys),
    };
    keys.push(asset.id.as_bytes());
    let xref_id = if let Some(id) = cached.get(asset.face.path()) {
        *id
    } else {
        let id = embed_font(pdf, asset)?;
        let xref_id = i64::from(id.0);
        cached.insert(asset.face.path().to_owned(), xref_id);
        xref_id
    };
    pdf::set_direct(pdf, target, &keys, Object::Reference((xref_id as u32, 0)))?;
    Ok(xref_id)
}

/// Caller supplies assets in Python's normal/script/fallback/base order,
/// deduplicated by resource ID. Progress is synchronous and does not fix any
/// callback bridge or handle lifetime policy.
///
/// Compatibility entry for the frozen `p2_stage_diff::writer_probe`.
pub fn add_fonts(
    pdf: &mut Document,
    view: &mut TreeView<'_, WriterView>,
    assets: &[FontAsset<'_>],
    advance: impl FnMut(u64) -> Result<()>,
) -> Result<FontInstallResult> {
    let used = used_font_ids(view)?;
    add_fonts_with_used_ids(pdf, view, assets, &used, advance)
}

/// Install fonts using the IDs already collected by the caller for progress.
///
/// Compatibility entry for the frozen `p2_stage_diff::writer_probe`.
pub fn add_fonts_with_used_ids(
    pdf: &mut Document,
    view: &mut TreeView<'_, WriterView>,
    assets: &[FontAsset<'_>],
    used: &HashSet<String>,
    advance: impl FnMut(u64) -> Result<()>,
) -> Result<FontInstallResult> {
    let page0 = page_object(pdf, 0)?;
    add_fonts_with_used_ids_at_page0(pdf, view, assets, used, page0, advance)
}

/// Production font registration keyed by the parser-selected full-document
/// page map. Page 0 remains full-document ordinal zero even when its page is
/// absent from the translated IR subset.
pub fn add_fonts_with_used_ids_from_page_objects(
    pdf: &mut Document,
    view: &mut TreeView<'_, WriterView>,
    assets: &[FontAsset<'_>],
    used: &HashSet<String>,
    page_objects: &pdf::PageObjectMap,
    advance: impl FnMut(u64) -> Result<()>,
) -> Result<FontInstallResult> {
    let page0 = page_object_with_map(pdf, page_objects, 0)?;
    add_fonts_with_used_ids_at_page0(pdf, view, assets, used, page0, advance)
}

fn add_fonts_with_used_ids_at_page0(
    pdf: &mut Document,
    view: &mut TreeView<'_, WriterView>,
    assets: &[FontAsset<'_>],
    used: &HashSet<String>,
    page0: ObjectId,
    mut advance: impl FnMut(u64) -> Result<()>,
) -> Result<FontInstallResult> {
    let assets: Vec<_> = assets.iter().filter(|a| used.contains(a.id)).collect();
    let page_count = view.document().page_len(view)?;
    let initial = pdf.max_id;
    let total = u64::from(initial) + 2 * assets.len() as u64 + page_count as u64;
    let mut result = FontInstallResult {
        total,
        installed: vec![],
    };
    if page_count == 0 {
        advance(total)?;
        return Ok(result);
    }
    let page_fonts = page0_font_refs_at(pdf, page0)?;
    let mut cached = HashMap::new();
    for asset in &assets {
        let xref_id = insert_page0(pdf, asset, &mut cached, &page_fonts, page0)?;
        result.installed.push(InstalledFont {
            font_id: asset.id.into(),
            xref_id,
        });
        advance(1)?;
    }
    for number in 1..=initial {
        advance(1)?;
        let Ok(mut target) = pdf::xref(pdf, i64::from(number)) else {
            continue;
        };
        for prefix in [true, false] {
            let keys: Vec<&[u8]> = if prefix {
                vec![b"Resources", b"Font"]
            } else {
                vec![b"Font"]
            };
            // The Python try block includes every font in this one path.
            let update = || -> Result<()> {
                let found = pdf::path(pdf, target, &keys)?;
                let mut key_prefix = keys;
                if let Object::Reference(id) = found {
                    if id.1 != 0 {
                        return Err(WriterError::Invalid(
                            "font ref generation is not zero".into(),
                        ));
                    }
                    target = *id;
                    key_prefix.clear();
                }
                if pdf::dict(pdf::path(pdf, target, &key_prefix)?).is_err() {
                    return Ok(());
                }
                for installed in &result.installed {
                    let mut key = key_prefix.clone();
                    key.push(installed.font_id.as_bytes());
                    if pdf::path(pdf, target, &key).map_or(true, |v| matches!(v, Object::Null)) {
                        pdf::set_direct(
                            pdf,
                            target,
                            &key,
                            Object::Reference((installed.xref_id as u32, 0)),
                        )?;
                    }
                }
                Ok(())
            };
            let _ = update();
        }
    }
    // Seed on the first page, then share the exact same M object with every
    // holder. Appending on a later call creates a second distinct batch.
    let first = view.document().page_at(view, 0)?;
    // This batch adds one shared IR node per font. Avoid doubling the compacted
    // document directory (and the shared font table) for just a few new rows.
    view.reserve_new_nodes::<types::PdfFont>(assets.len())?;
    let mut fonts = vec![];
    for (asset, installed) in assets.iter().zip(&result.installed) {
        let metadata = asset.metadata;
        let flags = asset
            .face
            .pdf_font_flags()
            .map_err(|e| WriterError::Invalid(e.to_string()))?;
        let index = first.pdf_font_len(view)?;
        first.push_pdf_font(
            view,
            PdfFontInit {
                name: Some(asset.id.into()),
                font_id: Some(asset.id.into()),
                xref_id: Some(installed.xref_id),
                encoding_length: Some(metadata.encoding_length),
                bold: Some(RuntimeBool::Int(flags.bold as i64)),
                italic: Some(RuntimeBool::Int(flags.italic as i64)),
                monospace: Some(RuntimeBool::Int(flags.monospace as i64)),
                serif: Some(RuntimeBool::Int(flags.serif as i64)),
                ascent: Some(metadata.ascent),
                descent: Some(metadata.descent),
                ..Default::default()
            }
            .into_value(),
        )?;
        fonts.push(first.pdf_font_at(view, index)?.retain(view)?);
        advance(1)?;
    }
    for index in 0..page_count {
        let page = view.document().page_at(view, index)?;
        if index != 0 {
            for font in &fonts {
                let value = font.share(view)?;
                page.push_pdf_font(view, value)?;
            }
        }
        for child_index in 0..page.pdf_xobject_len(view)? {
            let xobj = page.pdf_xobject_at(view, child_index)?;
            for font in &fonts {
                let value = font.share(view)?;
                xobj.push_pdf_font(view, value)?;
            }
        }
        advance(1)?;
    }
    Ok(result)
}

#[cfg(test)]
mod fonts_tests;
