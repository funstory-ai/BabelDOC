//! Page access follows the prepared reader's PyMuPDF enumeration and fallback geometry.
use super::*;
#[path = "page_rank.rs"]
mod rank;

fn references(object: &Object, pending: &mut Vec<ObjectId>) {
    match object {
        Object::Reference(id) => pending.push(*id),
        Object::Array(values) => values.iter().for_each(|o| references(o, pending)),
        Object::Dictionary(d) => d.iter().for_each(|(_, o)| references(o, pending)),
        Object::Stream(s) => s.dict.iter().for_each(|(_, o)| references(o, pending)),
        _ => {}
    }
}

#[cfg(test)]
fn projection_references(pdf: &Pdf, object: &Object, pending: &mut Vec<ObjectId>) {
    match object {
        Object::Array(values) => values
            .iter()
            .for_each(|o| projection_references(pdf, o, pending)),
        Object::Dictionary(d) => {
            if let Some(id) = fonts::projection_descendant_id(pdf, d) {
                pending.push(id);
            }
            d.iter()
                .for_each(|(_, o)| projection_references(pdf, o, pending));
        }
        Object::Stream(s) => {
            s.dict
                .iter()
                .for_each(|(_, o)| projection_references(pdf, o, pending));
        }
        _ => {}
    }
}

#[derive(Default)]
struct ProjectionWork {
    scanned: HashSet<ObjectId>,
    waiting: HashMap<ObjectId, Vec<(ObjectId, usize, ObjectId)>>,
}
impl ProjectionWork {
    fn extract(pdf: &Pdf, object: &Object, probes: &mut Vec<ProjectionProbe>) {
        match object {
            Object::Array(values) => values.iter().for_each(|o| Self::extract(pdf, o, probes)),
            Object::Dictionary(d) => {
                if let Ok(Object::Reference(id)) = d.get(b"DescendantFonts") {
                    probes.push(ProjectionProbe::Reference(*id));
                } else if let Some(id) = fonts::projection_descendant_id(pdf, d) {
                    probes.push(ProjectionProbe::Ready(id));
                }
                d.iter().for_each(|(_, o)| Self::extract(pdf, o, probes));
            }
            Object::Stream(s) => s.dict.iter().for_each(|(_, o)| Self::extract(pdf, o, probes)),
            _ => {}
        }
    }
    fn enqueue(&mut self, pdf: &Pdf, added: &mut Vec<ObjectId>, pending: &mut Vec<ObjectId>) {
        let mut events = std::collections::BTreeMap::new();
        for source in added.drain(..) {
            if self.scanned.insert(source) {
                if let Some(object) = pdf.objects.get(&source) {
                    let mut probes = Vec::new();
                    Self::extract(pdf, object, &mut probes);
                    for (ordinal, probe) in probes.into_iter().enumerate() {
                        events.insert((source, ordinal), probe);
                    }
                }
            }
            if let Some(waiters) = self.waiting.remove(&source) {
                for (owner, ordinal, reference) in waiters {
                    events.insert((owner, ordinal), ProjectionProbe::Reference(reference));
                }
            }
        }
        for ((source, ordinal), probe) in events {
            match probe {
                ProjectionProbe::Ready(id) => pending.push(id),
                ProjectionProbe::Reference(reference) => {
                    match pdf.get_object(reference) {
                        Err(lopdf::Error::ObjectNotFound(missing)) => {
                            self.waiting.entry(missing).or_default().push((source, ordinal, reference));
                        }
                        Ok(_) => {
                            let d = lopdf::dictionary! {"DescendantFonts"=>reference};
                            if let Some(id) = fonts::projection_descendant_id(pdf, &d) { pending.push(id); }
                        }
                        Err(_) => {} // An immutable cycle/reference-limit failure cannot become valid.
                    }
                }
            }
        }
    }
}
enum ProjectionProbe { Ready(ObjectId), Reference(ObjectId) }

pub(super) fn load_selected(
    reader: &mut lopdf::Reader<'_>,
    pages: Option<&[usize]>,
) -> lopdf::Result<()> {
    let mut newly_loaded = Vec::new();
    // Page enumeration needs only the catalog, page dictionaries and Kids.
    let root = reader.document.trailer.get(b"Root")?.as_reference()?;
    lopdf::resource::optional(reader.load_object_with_new_ids(root, &mut newly_loaded))?;
    let tree = reader.document.catalog()?.get(b"Pages")?.clone();
    let mut stack = vec![tree];
    let mut tree_seen = HashSet::new();
    while let Some(object) = stack.pop() {
        match object {
            Object::Reference(id) => {
                if !tree_seen.insert(id) {
                    continue;
                }
                lopdf::resource::optional(reader.load_object_with_new_ids(id, &mut newly_loaded))?;
                if let Ok(object) = reader.document.get_object(id) {
                    stack.push(object.clone());
                }
            }
            Object::Array(values) => stack.extend(values),
            Object::Dictionary(d) => {
                if let Ok(kids) = d.get(b"Kids") {
                    stack.push(kids.clone());
                }
            }
            _ => {}
        }
    }
    let ids = match page_ids(&reader.document) {
        Ok(ids) => ids,
        Err(error) if error.resource.is_some() => return Err((*error.resource.unwrap()).into()),
        Err(_) => Vec::new(),
    };
    let mut pending = Vec::new();
    let mut scanned_ancestors = HashSet::new();
    for (one, id) in ids {
        if pages.is_some_and(|p| !p.contains(&((one - 1) as usize))) {
            continue;
        }
        let mut current = Some(id);
        let mut seen = HashSet::new();
        while let Some(id) = current {
            if !seen.insert(id) {
                break;
            }
            lopdf::resource::optional(reader.load_object_with_new_ids(id, &mut newly_loaded))?;
            if !scanned_ancestors.insert(id) {
                break;
            }
            let Ok(d) = reader.document.get_dictionary(id) else {
                break;
            };
            for key in [
                b"Resources".as_slice(),
                b"Contents",
                b"CropBox",
                b"MediaBox",
                b"Rotate",
            ] {
                if let Ok(value) = d.get(key) {
                    references(value, &mut pending);
                }
            }
            current = d.get(b"Parent").ok().and_then(|o| o.as_reference().ok());
        }
    }
    // Every dependency of a resource/content object stays available, including
    // embedded fonts, nested Forms, CMaps and indirect numeric arrays.
    let mut seen = HashSet::new();
    while let Some(id) = pending.pop() {
        if !seen.insert(id) {
            continue;
        }
        lopdf::resource::optional(reader.load_object_with_new_ids(id, &mut newly_loaded))?;
        if let Ok(object) = reader.document.get_object(id) {
            references(object, &mut pending);
        }
    }
    // The reference's CID projection regex can infer an xref from a decimal in
    // an inline descendant, even when there is no PDF reference to that object.
    // Retain those semantic dependencies and their ordinary transitive closure.
    let mut projection = ProjectionWork::default();
    loop {
        projection.enqueue(&reader.document, &mut newly_loaded, &mut pending);
        let mut added = false;
        while let Some(id) = pending.pop() {
            if !seen.insert(id) { continue; }
            added = true;
            lopdf::resource::optional(reader.load_object_with_new_ids(id, &mut newly_loaded))?;
            if let Ok(object) = reader.document.get_object(id) { references(object, &mut pending); }
        }
        if !added { break; }
    }
    Ok(())
}

pub(super) fn page_ids(pdf: &Pdf) -> Result<Vec<(u32, ObjectId)>> {
    let root = pdf.catalog()?.get(b"Pages")?;
    let root_dict = dict(pdf, root)?;
    let count = root_dict.get(b"Count").ok().and_then(numeric).unwrap_or(0.) as i64;
    let size = pdf
        .trailer
        .get(b"Size")
        .ok()
        .and_then(numeric)
        .unwrap_or((pdf.max_id + 1) as f64) as i64;
    if count < 0 || count >= size {
        return Err(Error::new("FzErrorFormat", "Invalid number of pages"));
    }
    // Repeated siblings retain their order. Once the strict traversal reaches
    // the resource boundary, the whole parse fails.
    let strict = (|| -> Result<Option<Vec<ObjectId>>> {
    let mut stack = vec![(root, false, true)];
    let mut active = HashSet::new();
    let mut expansion = expansion::Expansion::new(pdf.parser_policy, "page_tree");
    let mut fast = Vec::new();
    let mut valid = true;
    while let Some((node, leaving, _root_node)) = stack.pop() {
        let Ok(value) = resolve(pdf, node) else {
            valid = false;
            break;
        };
        let key = value as *const Object as usize;
        if leaving {
            active.remove(&key);
            continue;
        }
        let Ok(d) = dict(pdf, value) else {
            valid = false;
            break;
        };
        match d.get(b"Type").ok().and_then(|o| o.as_name().ok()) {
            Some(b"Pages") => {
                if !active.insert(key) {
                    valid = false;
                    break;
                }
                let source = node.as_reference().ok();
                expansion.enter_program(key, 0, source)?;
                stack.push((value, true, false));
                let kids = d.get(b"Kids").ok().and_then(|o| resolve(pdf, o).ok());
                if let Some((object, items)) = kids.and_then(|o| o.as_array().ok().map(|items| (o, items))) {
                    // Distinct Pages nodes may share one physical Kids array.
                    expansion.structural_program(object as *const Object as usize, items.len(), source)?;
                    expansion.work(1, source)?;
                    if !items.is_empty() {
                        stack.extend(items.iter().rev().map(|o| (o, false, false)));
                    }
                }
            }
            Some(b"Page") => {
                if let Object::Reference(id) = node {
                    expansion.enter_program(key, 0, Some(*id))?;
                    fast.push(*id);
                } else {
                    valid = false;
                    break;
                }
            }
            _ => {
                valid = false;
                break;
            }
        }
    }
    Ok((valid && fast.len() == count as usize).then_some(fast))
    })();
    match strict {
        Ok(Some(fast)) => return Ok(fast.into_iter().enumerate().map(|(n,id)| ((n+1) as u32,id)).collect()),
        Ok(None) => {}
        Err(error) => return Err(error),
    }
    // Keep the legacy Count-guided semantics; index only the sibling routing.
    rank::fallback(pdf, root, count)
}

fn inherited_rect(
    pdf: &Pdf,
    id: ObjectId,
    key: &[u8],
    inherited_objects: &mut InheritedObjectCache,
) -> Option<[f32; 4]> {
    let value = inherited_cached(pdf, id, key, inherited_objects).ok()??;
    let a = resolve(pdf, &value).ok()?.as_array().ok()?;
    if a.len() != 4 {
        return None;
    }
    let n = |i: usize| match resolve(pdf, &a[i]).ok() {
        Some(Object::Integer(n)) => *n as f32,
        Some(Object::Real(n)) => *n as f32,
        _ => 0.,
    };
    let [a, b, c, d] = [n(0), n(1), n(2), n(3)];
    Some([a.min(c), b.min(d), a.max(c), b.max(d)])
}
fn valid_rect(b: [f32; 4]) -> bool {
    b.iter().all(|n| n.is_finite() && n.abs() < 2147483520.) && b[0] < b[2] && b[1] < b[3]
}

pub(super) fn cropbox(
    pdf: &Pdf,
    id: ObjectId,
    inherited_objects: &mut InheritedObjectCache,
) -> Result<[f64; 4]> {
    let page = pdf.get_dictionary(id)?;
    let direct = page
        .get(b"CropBox")
        .ok()
        .filter(|o| object_truthy(o))
        .or_else(|| page.get(b"MediaBox").ok());
    if let Some(a) = direct
        .and_then(|o| o.as_array().ok())
        .filter(|a| a.len() >= 4)
    {
        let mut out = [0.; 4];
        for (i, o) in a[..4].iter().enumerate() {
            out[i] = match resolve(pdf, o)? {
                Object::Integer(n) => *n as f64,
                Object::Real(n) => object_numbers::printed_real(*n),
                Object::Boolean(b) => {
                    if *b {
                        1.
                    } else {
                        0.
                    }
                }
                Object::String(s, _) => std::str::from_utf8(s)
                    .ok()
                    .and_then(|s| s.trim().parse().ok())
                    .ok_or_else(|| {
                        Error::new("ValueError", "could not convert box coordinate to float")
                    })?,
                _ => {
                    return Err(Error::new(
                        "TypeError",
                        "float() argument must be a string or a real number",
                    ));
                }
            };
        }
        return Ok(out);
    }
    // JM_cropbox uses inherited rectangles only for the fallback; direct PDF
    // arrays above intentionally retain the product reader's raw coordinates.
    let mut media = inherited_rect(pdf, id, b"MediaBox", inherited_objects)
        .filter(|b| valid_rect(*b))
        .unwrap_or([0., 0., 612., 792.]);
    if media[2] - media[0] < 1. || media[3] - media[1] < 1. {
        media = [0., 0., 1., 1.];
    }
    let crop = inherited_rect(pdf, id, b"CropBox", inherited_objects)
        .filter(|b| valid_rect(*b))
        .unwrap_or(media);
    Ok([
        crop[0] as f64,
        (media[3] - crop[3]) as f64,
        crop[2] as f64,
        (media[3] - crop[1]) as f64,
    ])
}

#[cfg(test)]
mod page_access_tests;

#[cfg(test)]
mod shared_input_tests;

#[cfg(test)]
#[path = "page_access_resource_tests.rs"]
mod resource_tests;

#[cfg(test)]
#[path = "page_projection_v3_tests.rs"]
mod projection_v3_tests;
