//! A cross-reference stream declares how many entries it holds in its `/Index`
//! array, but the entries themselves live in the (decoded) stream body. If the
//! declared count is allowed to run past what the body can hold, `load` spins on
//! an attacker-controlled number instead of the bytes actually present.

#![cfg(not(feature = "async"))]

use std::io::Write;

use flate2::{Compression, write::ZlibEncoder};
use lopdf::{Document, Error, LoadOptions, ParseError, SaveOptions, Stream, dictionary};

/// A PDF whose only cross-reference is an xref stream with the given `/W` widths,
/// `/Index [0 count]`, and `body` as an uncompressed stream body.
fn xref_stream_pdf(w: [i64; 3], count: i64, body: &[u8]) -> Vec<u8> {
    xref_stream_pdf_with_index(w, &format!("0 {count}"), body)
}

fn xref_stream_pdf_with_index(w: [i64; 3], index: &str, body: &[u8]) -> Vec<u8> {
    let mut pdf = Vec::new();
    pdf.extend_from_slice(b"%PDF-1.5\n");
    let obj_offset = pdf.len();
    pdf.extend_from_slice(b"1 0 obj\n");
    pdf.extend_from_slice(
        format!(
            "<< /Type /XRef /Size 4 /W [{} {} {}] /Index [{index}] /Root 1 0 R /Length {} >>\n",
            w[0],
            w[1],
            w[2],
            body.len()
        )
        .as_bytes(),
    );
    pdf.extend_from_slice(b"stream\n");
    pdf.extend_from_slice(body);
    pdf.extend_from_slice(b"\nendstream\nendobj\n");
    pdf.extend_from_slice(format!("startxref\n{obj_offset}\n%%EOF").as_bytes());
    pdf
}

fn compressed_xref_stream_pdf(entries: usize) -> Vec<u8> {
    let mut pdf = b"%PDF-1.5\n".to_vec();
    let catalog_offset = pdf.len();
    pdf.extend_from_slice(b"1 0 obj\n<< /Type /Catalog >>\nendobj\n");
    let xref_offset = pdf.len();

    let mut body = Vec::with_capacity(entries * 5);
    for id in 0..entries {
        let (entry_type, offset): (u8, u32) = match id {
            1 => (1, catalog_offset as u32),
            2 => (1, xref_offset as u32),
            _ => (0, 0),
        };
        body.push(entry_type);
        body.extend_from_slice(&offset.to_be_bytes()[1..4]);
        body.push(0);
    }
    let mut encoder = ZlibEncoder::new(Vec::new(), Compression::default());
    encoder.write_all(&body).unwrap();
    let compressed_body = encoder.finish().unwrap();

    pdf.extend_from_slice(b"2 0 obj\n");
    pdf.extend_from_slice(
        format!(
            "<< /Type /XRef /Size {entries} /W [1 3 1] /Index [0 {entries}] /Root 1 0 R \
             /Filter /FlateDecode /Length {} >>\n",
            compressed_body.len()
        )
        .as_bytes(),
    );
    pdf.extend_from_slice(b"stream\n");
    pdf.extend_from_slice(&compressed_body);
    pdf.extend_from_slice(b"\nendstream\nendobj\n");
    pdf.extend_from_slice(format!("startxref\n{xref_offset}\n%%EOF").as_bytes());
    pdf
}

fn incremental_classic_xref_pdf(objects: usize) -> Vec<u8> {
    let mut pdf = b"%PDF-1.4\n".to_vec();
    let mut offsets = Vec::with_capacity(objects);
    offsets.push(pdf.len());
    pdf.extend_from_slice(b"1 0 obj\n<< /Type /Catalog >>\nendobj\n");
    for id in 2..=objects {
        offsets.push(pdf.len());
        pdf.extend_from_slice(format!("{id} 0 obj\n{id}\nendobj\n").as_bytes());
    }

    let first_xref = pdf.len();
    pdf.extend_from_slice(format!("xref\n0 {}\n0000000000 65535 f \n", objects + 1).as_bytes());
    for offset in offsets {
        pdf.extend_from_slice(format!("{offset:010} 00000 n \n").as_bytes());
    }
    pdf.extend_from_slice(format!("trailer\n<< /Size {} /Root 1 0 R >>\n", objects + 1).as_bytes());
    pdf.extend_from_slice(format!("startxref\n{first_xref}\n%%EOF\n").as_bytes());

    let new_offset = pdf.len();
    let new_id = objects + 1;
    pdf.extend_from_slice(format!("{new_id} 0 obj\n99\nendobj\n").as_bytes());
    let second_xref = pdf.len();
    pdf.extend_from_slice(format!("xref\n{new_id} 1\n{new_offset:010} 00000 n \n").as_bytes());
    pdf.extend_from_slice(format!("trailer\n<< /Size {} /Root 1 0 R /Prev {first_xref} >>\n", new_id + 1).as_bytes());
    pdf.extend_from_slice(format!("startxref\n{second_xref}\n%%EOF").as_bytes());
    pdf
}

// With `/W [0 0 0]` every entry reads zero bytes, so before the bound the loop ran
// purely on `/Index`. A 137-byte file claiming a million entries returned Ok with a
// million-node xref table; `i64::MAX` never returned at all. It must be rejected,
// and quickly.
#[test]
fn zero_width_xref_stream_is_rejected() {
    let pdf = xref_stream_pdf([0, 0, 0], 1_000_000, b"");
    match Document::load_mem(&pdf) {
        Err(Error::Parse(ParseError::InvalidXref)) => {}
        Err(other) => panic!("expected InvalidXref, got {other:?}"),
        Ok(doc) => panic!(
            "a 137-byte file claiming 1,000,000 xref entries loaded with {} entries",
            doc.reference_table.entries.len()
        ),
    }
}

// Non-zero widths already terminate once the reader runs dry, but only after
// inserting one map entry per byte first. Bounding the count against the bytes
// present stops that up front: three body bytes at `/W [1 1 1]` hold one entry, so
// a section claiming a million is malformed.
#[test]
fn index_count_past_stream_length_is_rejected() {
    let pdf = xref_stream_pdf([1, 1, 1], 1_000_000, &[1, 0, 0]);
    match Document::load_mem(&pdf) {
        Err(Error::Parse(ParseError::InvalidXref)) => {}
        Err(other) => panic!("expected InvalidXref, got {other:?}"),
        Ok(_) => panic!("an /Index count far past the stream length was accepted"),
    }
}

#[test]
fn negative_index_count_is_rejected() {
    let pdf = xref_stream_pdf([1, 1, 1], -1, &[]);
    assert!(matches!(
        Document::load_mem(&pdf),
        Err(Error::Parse(ParseError::InvalidXref))
    ));
}

#[test]
fn three_byte_xref_records_are_accepted() {
    let pdf = xref_stream_pdf([1, 2, 0], 2, &[0, 0, 0, 1, 0, 9]);
    Document::load_mem(&pdf).expect("/W [1 2 0] records should remain accepted");
}

#[test]
fn single_index_section_matching_stream_length_is_accepted() {
    let body = [1, 0, 0, 0, 9, 0, 0];
    let pdf = xref_stream_pdf_with_index([1, 4, 2], "0 1", &body);
    Document::load_mem(&pdf).expect("one seven-byte record should fit a seven-byte body");
}

#[test]
fn total_index_count_past_stream_length_is_rejected() {
    let body = [1, 0, 0, 0, 9, 0, 0];
    let pdf = xref_stream_pdf_with_index([1, 4, 2], "0 1 2 1", &body);
    assert!(matches!(
        Document::load_mem(&pdf),
        Err(Error::Parse(ParseError::InvalidXref))
    ));
}

// `/W [0 1 0]` records consume one decoded byte but retain a full map entry.
// Floor the effective width so a valid-sized body cannot amplify into a much
// larger in-memory xref table, independently of any decompression budget.
#[test]
fn degenerate_narrow_xref_records_are_rejected() {
    let pdf = xref_stream_pdf([0, 1, 0], 4, &[0; 4]);
    let result = Document::load_mem(&pdf);
    assert!(matches!(result, Err(Error::Parse(ParseError::InvalidXref))));
}

#[test]
fn realistic_compressed_xref_stream_loads_under_budget() {
    let pdf = compressed_xref_stream_pdf(300);
    Document::load_mem_with_options(&pdf, LoadOptions::with_max_decompressed_size(2_000))
        .expect("a 1,500-byte xref stream should load under a 2,000-byte budget");
}

#[test]
fn incremental_classic_xref_loads_under_budget() {
    let pdf = incremental_classic_xref_pdf(400);
    Document::load_mem_with_options(&pdf, LoadOptions::with_max_decompressed_size(2_000))
        .expect("classic xref entries should not consume the decompression budget");
}

// The bound is derived from the real per-entry width, so a genuine xref stream
// whose `/Index` matches its body still loads. Written by lopdf's own writer to be
// sure the shape is exactly what the reader expects.
#[test]
fn valid_xref_stream_still_loads() {
    let mut doc = Document::with_version("1.5");
    let content_id = doc.add_object(Stream::new(dictionary! {}, b"BT ET".to_vec()));
    let pages_id = doc.new_object_id();
    let page_id = doc.add_object(dictionary! {
        "Type" => "Page",
        "Parent" => lopdf::Object::Reference(pages_id),
        "MediaBox" => vec![0.into(), 0.into(), 612.into(), 792.into()],
        "Contents" => lopdf::Object::Reference(content_id),
    });
    doc.objects.insert(
        pages_id,
        lopdf::Object::Dictionary(dictionary! {
            "Type" => "Pages",
            "Kids" => vec![lopdf::Object::Reference(page_id)],
            "Count" => 1,
        }),
    );
    let catalog_id = doc.add_object(dictionary! {
        "Type" => "Catalog",
        "Pages" => lopdf::Object::Reference(pages_id),
    });
    doc.trailer.set("Root", lopdf::Object::Reference(catalog_id));

    let options = SaveOptions::builder()
        .use_object_streams(false)
        .use_xref_streams(true)
        .build();
    let mut buffer = Vec::new();
    doc.save_with_options(&mut buffer, options).unwrap();

    let reloaded = Document::load_mem(&buffer).unwrap();
    assert!(reloaded.get_object(catalog_id).is_ok());
    assert_eq!(reloaded.get_pages().len(), 1);
}
