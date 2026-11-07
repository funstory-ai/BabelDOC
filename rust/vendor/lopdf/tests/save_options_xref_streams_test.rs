//! `SaveOptions::use_xref_streams` on its own, without object streams.
//!
//! The two features are independent in the PDF specification, so asking for a
//! cross-reference stream should not require asking for object streams as well.

#![cfg(not(feature = "async"))]

use lopdf::xref::XrefType;
use lopdf::{Document, Object, SaveOptions, Stream, dictionary};

/// A one page document written with a classic cross-reference table, as a document
/// loaded from a pre-1.5 file would be.
fn sample_document() -> Document {
    let mut doc = Document::with_version("1.4");
    doc.reference_table.cross_reference_type = XrefType::CrossReferenceTable;

    let pages_id = doc.new_object_id();
    let content_id = doc.add_object(Stream::new(dictionary! {}, b"BT ET".to_vec()));
    let page_id = doc.add_object(dictionary! {
        "Type" => "Page",
        "Parent" => Object::Reference(pages_id),
        "MediaBox" => vec![0.into(), 0.into(), 612.into(), 792.into()],
        "Contents" => Object::Reference(content_id),
    });
    doc.objects.insert(
        pages_id,
        Object::Dictionary(dictionary! {
            "Type" => "Pages",
            "Kids" => vec![Object::Reference(page_id)],
            "Count" => 1,
        }),
    );
    let catalog_id = doc.add_object(dictionary! {
        "Type" => "Catalog",
        "Pages" => Object::Reference(pages_id),
    });
    doc.trailer.set("Root", Object::Reference(catalog_id));

    doc
}

fn contains(haystack: &[u8], needle: &[u8]) -> bool {
    haystack.windows(needle.len()).any(|window| window == needle)
}

#[test]
fn xref_streams_are_written_without_object_streams() {
    let mut doc = sample_document();

    let options = SaveOptions::builder()
        .use_object_streams(false)
        .use_xref_streams(true)
        .build();
    let mut buffer = Vec::new();
    doc.save_with_options(&mut buffer, options).unwrap();

    assert!(
        contains(&buffer, b"/Type/XRef") || contains(&buffer, b"/Type /XRef"),
        "a requested cross-reference stream should be written even with object streams off"
    );
    assert!(
        !contains(&buffer, b"\nxref\n"),
        "the classic cross-reference table should not be written as well"
    );
    assert!(
        !contains(&buffer, b"/ObjStm"),
        "object streams were not requested and must not appear"
    );

    // A cross-reference stream is a PDF 1.5 construct, so the header has to say so.
    assert!(
        buffer.starts_with(b"%PDF-1.5"),
        "expected the version to be raised to 1.5, got {:?}",
        String::from_utf8_lossy(&buffer[..8.min(buffer.len())])
    );

    let reloaded = Document::load_mem(&buffer).unwrap();
    assert_eq!(reloaded.get_pages().len(), 1, "the saved document should still load");
}

#[test]
fn cross_reference_table_is_kept_when_xref_streams_are_not_requested() {
    let mut doc = sample_document();

    let options = SaveOptions::builder()
        .use_object_streams(false)
        .use_xref_streams(false)
        .build();
    let mut buffer = Vec::new();
    doc.save_with_options(&mut buffer, options).unwrap();

    assert!(
        contains(&buffer, b"\nxref\n"),
        "without the option the document keeps its cross-reference table"
    );
    assert!(
        !contains(&buffer, b"/Type/XRef") && !contains(&buffer, b"/Type /XRef"),
        "no cross-reference stream should appear when it was not requested"
    );
    assert!(
        buffer.starts_with(b"%PDF-1.4"),
        "the version should be left alone when no 1.5 feature is used"
    );
}

#[test]
fn both_options_together_still_write_object_and_xref_streams() {
    let mut doc = sample_document();

    let options = SaveOptions::builder()
        .use_object_streams(true)
        .use_xref_streams(true)
        .build();
    let mut buffer = Vec::new();
    doc.save_with_options(&mut buffer, options).unwrap();

    assert!(
        contains(&buffer, b"/ObjStm"),
        "object streams were requested and should still be written"
    );
    assert!(
        contains(&buffer, b"/Type/XRef") || contains(&buffer, b"/Type /XRef"),
        "cross-reference streams were requested and should still be written"
    );

    let reloaded = Document::load_mem(&buffer).unwrap();
    assert_eq!(reloaded.get_pages().len(), 1, "the saved document should still load");
}
