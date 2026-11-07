use lopdf::{Document, Object, Result, dictionary};

mod utils;

/// One page whose `/Annots` mixes the two legal entry forms: a reference to
/// an annotation dictionary, and the dictionary written directly into the
/// array. Returns the document and the page's id.
fn page_with_mixed_annots(annots_indirect: bool) -> (Document, lopdf::ObjectId) {
    let mut doc = Document::with_version("1.7");
    let referenced = doc.add_object(dictionary! {
        "Type" => "Annot",
        "Subtype" => "Text",
        "Contents" => Object::string_literal("referenced"),
    });
    let entries = vec![
        Object::Reference(referenced),
        Object::Dictionary(dictionary! {
            "Type" => "Annot",
            "Subtype" => "Text",
            "Contents" => Object::string_literal("direct"),
        }),
    ];
    // `/Annots` is itself allowed to be either an array or a reference to one.
    let annots = if annots_indirect {
        Object::Reference(doc.add_object(Object::Array(entries)))
    } else {
        Object::Array(entries)
    };
    let page_id = doc.add_object(dictionary! {
        "Type" => "Page",
        "Annots" => annots,
    });
    (doc, page_id)
}

fn contents_of(doc: &Document, page_id: lopdf::ObjectId) -> Vec<String> {
    doc.get_page_annotations(page_id)
        .unwrap()
        .iter()
        .map(|a| String::from_utf8_lossy(a.get(b"Contents").unwrap().as_str().unwrap()).into_owned())
        .collect()
}

/// An entry of `/Annots` may be the annotation dictionary itself rather than
/// a reference to one — ISO 32000-1, 12.5.2 requires an indirect object only
/// for an annotation that carries a `/Popup` or is an `/IRT` target. Direct
/// dictionaries used to be dropped without a word, so a page could report
/// fewer annotations than it has.
#[test]
fn page_annotations_include_direct_dictionaries() {
    let (doc, page_id) = page_with_mixed_annots(false);
    assert_eq!(contents_of(&doc, page_id), vec!["referenced", "direct"]);
}

/// The same, with `/Annots` written as a reference to the array.
#[test]
fn page_annotations_include_direct_dictionaries_behind_an_indirect_annots() {
    let (doc, page_id) = page_with_mixed_annots(true);
    assert_eq!(contents_of(&doc, page_id), vec!["referenced", "direct"]);
}

/// A reference that does not resolve costs its own annotation, not the page:
/// the surviving entries still come back.
#[test]
fn page_annotations_skip_a_dangling_reference() {
    let mut doc = Document::with_version("1.7");
    let good = doc.add_object(dictionary! {
        "Type" => "Annot",
        "Subtype" => "Text",
        "Contents" => Object::string_literal("kept"),
    });
    let page_id = doc.add_object(dictionary! {
        "Type" => "Page",
        "Annots" => vec![
            Object::Reference((9999, 0)),
            Object::Reference(good),
            Object::Null,
        ],
    });
    assert_eq!(contents_of(&doc, page_id), vec!["kept"]);
}

/// A page without `/Annots` has no annotations, and that is not an error.
#[test]
fn page_annotations_without_annots_is_empty() {
    let mut doc = Document::with_version("1.7");
    let page_id = doc.add_object(dictionary! { "Type" => "Page" });
    assert!(doc.get_page_annotations(page_id).unwrap().is_empty());
}

#[test]
fn annotation_count() -> Result<()> {
    // This test file from the pdfcpu repository,
    // https://github.com/pdfcpu/pdfcpu/blob/master/pkg/samples/basic/AnnotationDemo.pdf
    let doc = utils::load_document("assets/AnnotationDemo.pdf")?;
    assert_eq!(doc.version, "1.7".to_string());
    assert_eq!(doc.page_iter().count(), 1);
    assert_eq!(doc.get_page_annotations(doc.page_iter().next().unwrap())?.len(), 33);
    Ok(())
}
