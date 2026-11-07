use std::collections::HashMap;

use lopdf::{Document, LoadOptions, Object};

fn pdf_with_stream(declared_length: usize, content: &[u8], trailer: &[u8]) -> Vec<u8> {
    let mut pdf = Vec::new();
    let mut offsets = Vec::new();

    pdf.extend_from_slice(b"%PDF-1.7\n");
    offsets.push(pdf.len());
    pdf.extend_from_slice(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n");
    offsets.push(pdf.len());
    pdf.extend_from_slice(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n");
    offsets.push(pdf.len());
    pdf.extend_from_slice(
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 10 10] /Resources << /XObject << /Im0 4 0 R >> >> >>\nendobj\n",
    );
    offsets.push(pdf.len());
    pdf.extend_from_slice(
        format!(
            "4 0 obj\n<< /Type /XObject /Subtype /Image /Width 1 /Height 1 /ColorSpace /DeviceRGB \
             /BitsPerComponent 8 /Filter [/FlateDecode /DCTDecode] /Length {declared_length} >>\nstream\n"
        )
        .as_bytes(),
    );
    pdf.extend_from_slice(content);
    pdf.extend_from_slice(trailer);

    let xref_offset = pdf.len();
    pdf.extend_from_slice(b"xref\n0 5\n0000000000 65535 f \n");
    for offset in offsets {
        pdf.extend_from_slice(format!("{offset:010} 00000 n \n").as_bytes());
    }
    pdf.extend_from_slice(format!("trailer\n<< /Size 5 /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n").as_bytes());
    pdf
}

#[test]
fn lenient_load_recovers_an_unambiguous_stream_length_mismatch() {
    let content = b"actual stream bytes";
    let pdf = pdf_with_stream(0, content, b"\nendstream\nendobj\n");

    let document = Document::load_mem(&pdf).unwrap();
    let Object::Stream(stream) = document.get_object((4, 0)).unwrap() else {
        panic!("recovered object must remain a stream");
    };
    assert_eq!(stream.content, content);
    assert_eq!(
        stream.dict.get(b"Length").unwrap().as_i64().unwrap(),
        content.len() as i64
    );
}

#[test]
fn strict_load_rejects_a_stream_length_mismatch() {
    let content = b"actual stream bytes";
    let malformed = pdf_with_stream(0, content, b"\nendstream\nendobj\n");
    let conforming = pdf_with_stream(content.len(), content, b"\nendstream\nendobj\n");
    let strict = LoadOptions {
        strict: true,
        ..Default::default()
    };

    assert!(Document::load_mem_with_options(&conforming, strict.clone()).is_ok());
    assert!(Document::load_mem_with_options(&malformed, strict).is_err());
}

#[test]
fn lenient_load_does_not_guess_an_ambiguous_stream_boundary() {
    let pdf = pdf_with_stream(
        0,
        b"first candidate\nendstream\nendobj\nbytes after the false boundary",
        b"\nendstream\nendobj\n",
    );

    let document = Document::load_mem(&pdf).unwrap();
    assert!(document.get_object((4, 0)).is_err());
}

#[test]
fn lenient_load_does_not_recover_without_endstream() {
    let pdf = pdf_with_stream(0, b"unterminated stream bytes", b"\nendobj\n");

    let document = Document::load_mem(&pdf).unwrap();
    assert!(document.get_object((4, 0)).is_err());
}

/// Assemble a PDF from whole `"<id> 0 obj ... endobj"` blobs laid out in the
/// given order. `xref_overrides` lets tests record an offset that differs from
/// the physical one; the returned map holds the true physical offsets.
fn pdf_from_objects(objects: &[(u32, &[u8])], xref_overrides: &[(u32, usize)]) -> (Vec<u8>, HashMap<u32, usize>) {
    let mut pdf = Vec::new();
    let mut physical = HashMap::new();

    pdf.extend_from_slice(b"%PDF-1.7\n");
    for (id, body) in objects {
        physical.insert(*id, pdf.len());
        pdf.extend_from_slice(format!("{id} 0 obj\n").as_bytes());
        pdf.extend_from_slice(body);
        pdf.extend_from_slice(b"endobj\n");
    }

    let xref_offset = pdf.len();
    let size = objects.iter().map(|(id, _)| *id).max().unwrap_or(0) + 1;
    pdf.extend_from_slice(b"xref\n");
    pdf.extend_from_slice(format!("0 {size}\n0000000000 65535 f \n").as_bytes());
    for object_number in 1..size {
        let offset = xref_overrides
            .iter()
            .find(|&&(id, _)| id == object_number)
            .map(|&(_, offset)| offset)
            .or_else(|| physical.get(&object_number).copied())
            .expect("xref entry missing for object");
        pdf.extend_from_slice(format!("{offset:010} 00000 n \n").as_bytes());
    }
    pdf.extend_from_slice(
        format!("trailer\n<< /Size {size} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n").as_bytes(),
    );
    (pdf, physical)
}

fn catalog() -> &'static [u8] {
    b"<< /Type /Catalog /Pages 2 0 R >>\n"
}

fn pages(kids: &str) -> Vec<u8> {
    format!("<< /Type /Pages /Kids [{kids}] /Count 1 >>\n").into_bytes()
}

fn page(xobjects: &str) -> Vec<u8> {
    format!("<< /Type /Page /Parent 2 0 R /MediaBox [0 0 10 10] /Resources << /XObject << {xobjects} >> >> >>\n")
        .into_bytes()
}

/// Body of an image stream object with a *correct* `/Length`, ending right
/// before the `endobj` keyword appended by `pdf_from_objects`.
fn image_stream_body(content: &[u8]) -> Vec<u8> {
    let mut body = format!(
        "<< /Type /XObject /Subtype /Image /Width {} /Height 1 /ColorSpace /DeviceRGB \
         /BitsPerComponent 8 /Length {} >>\nstream\n",
        content.len() / 3,
        content.len()
    )
    .into_bytes();
    body.extend_from_slice(content);
    body.extend_from_slice(b"\nendstream\n");
    body
}

#[test]
fn accurate_but_out_of_order_xref_offsets_load_every_object() {
    // Physical order: catalog, pages, image stream, page. The xref rows are
    // still emitted in ascending object-number order, so the recorded offsets
    // do not increase in file order.
    let content = vec![b'A'; 48];
    let objects: Vec<(u32, Vec<u8>)> = vec![
        (1, catalog().to_vec()),
        (2, pages("3 0 R")),
        (4, image_stream_body(&content)),
        (3, page("/Im0 4 0 R")),
    ];
    let bodies: Vec<(u32, &[u8])> = objects.iter().map(|(id, body)| (*id, body.as_slice())).collect();
    let (pdf, _) = pdf_from_objects(&bodies, &[]);

    let document = Document::load_mem(&pdf).unwrap();
    assert!(document.get_object((1, 0)).is_ok());
    assert!(document.get_object((2, 0)).is_ok());
    assert!(document.get_object((3, 0)).is_ok());
    let Object::Stream(stream) = document.get_object((4, 0)).unwrap() else {
        panic!("image object must remain a stream");
    };
    assert_eq!(stream.content, content);
}

/// Byte position just after the only `stream` keyword's EOL in the document.
fn stream_data_start(pdf: &[u8]) -> usize {
    pdf.windows(7)
        .position(|window| window == b"stream\n")
        .expect("no stream keyword")
        + 7
}

#[test]
fn inaccurate_xref_offset_inside_a_stream_only_loses_the_mislocated_object() {
    // Object 5 physically follows object 4, but its xref offset is miswritten
    // to point into object 4's stream payload. Object 4 itself is well-formed
    // (its /Length is correct), so it still loads; only the mislocated object
    // is lost.
    let content = vec![b'A'; 48];
    let objects: Vec<(u32, Vec<u8>)> = vec![
        (1, catalog().to_vec()),
        (2, pages("3 0 R")),
        (3, page("/Im0 4 0 R /F1 5 0 R")),
        (4, image_stream_body(&content)),
        (5, b"<< /Type /Font >>\n".to_vec()),
    ];
    let bodies: Vec<(u32, &[u8])> = objects.iter().map(|(id, body)| (*id, body.as_slice())).collect();
    let (pdf, _) = pdf_from_objects(&bodies, &[]);
    let false_offset = stream_data_start(&pdf) + 16;
    let (pdf, _) = pdf_from_objects(&bodies, &[(5, false_offset)]);

    let document = Document::load_mem(&pdf).unwrap();
    assert!(document.get_object((1, 0)).is_ok());
    assert!(document.get_object((2, 0)).is_ok());
    assert!(document.get_object((3, 0)).is_ok());
    let Object::Stream(stream) = document.get_object((4, 0)).unwrap() else {
        panic!("image object must remain a stream");
    };
    assert_eq!(stream.content, content);
    assert!(document.get_object((5, 0)).is_err());
}

#[test]
fn strict_mode_aborts_when_an_xref_offset_is_unparseable() {
    // Strict mode fails the whole load when any object cannot be parsed —
    // here object 5, whose xref offset points into object 4's payload.
    // Object 4 itself survives parsing; the abort is caused solely by the
    // unparseable neighbor offset.
    let content = vec![b'A'; 48];
    let objects: Vec<(u32, Vec<u8>)> = vec![
        (1, catalog().to_vec()),
        (2, pages("3 0 R")),
        (3, page("/Im0 4 0 R")),
        (4, image_stream_body(&content)),
        (5, b"<< /Type /Font >>\n".to_vec()),
    ];
    let bodies: Vec<(u32, &[u8])> = objects.iter().map(|(id, body)| (*id, body.as_slice())).collect();
    let (pdf, _) = pdf_from_objects(&bodies, &[]);
    let false_offset = stream_data_start(&pdf) + 16;
    let (pdf, _) = pdf_from_objects(&bodies, &[(5, false_offset)]);
    let strict = LoadOptions {
        strict: true,
        ..Default::default()
    };

    assert!(Document::load_mem_with_options(&pdf, strict).is_err());
}

#[test]
fn out_of_order_xref_with_offset_inside_a_stream_only_loses_the_mislocated_object() {
    // Object 3 physically sits after the image stream of object 4, but its
    // recorded xref offset lands inside object 4's payload. Object 4 parses
    // fine; only the mislocated object disappears.
    let content = vec![b'A'; 48];
    let objects: Vec<(u32, Vec<u8>)> = vec![
        (1, catalog().to_vec()),
        (2, pages("3 0 R")),
        (4, image_stream_body(&content)),
        (3, page("/Im0 4 0 R")),
    ];
    let bodies: Vec<(u32, &[u8])> = objects.iter().map(|(id, body)| (*id, body.as_slice())).collect();
    let (pdf, _) = pdf_from_objects(&bodies, &[]);
    let false_offset = stream_data_start(&pdf) + 16;
    let (pdf, _) = pdf_from_objects(&bodies, &[(3, false_offset)]);

    let document = Document::load_mem(&pdf).unwrap();
    assert!(document.get_object((1, 0)).is_ok());
    assert!(document.get_object((2, 0)).is_ok());
    assert!(document.get_object((3, 0)).is_err());
    let Object::Stream(stream) = document.get_object((4, 0)).unwrap() else {
        panic!("image object must remain a stream");
    };
    assert_eq!(stream.content, content);
}

/// Body of a stream whose declared `/Length` is deliberately wrong (too
/// small), so loading has to rely on boundary recovery.
fn wrong_length_stream_body(content: &[u8]) -> Vec<u8> {
    let mut body = format!(
        "<< /Type /XObject /Subtype /Image /Width {} /Height 1 /ColorSpace /DeviceRGB \
             /BitsPerComponent 8 /Length 0 >>\nstream\n",
        content.len() / 3
    )
    .into_bytes();
    body.extend_from_slice(content);
    body.extend_from_slice(b"\nendstream\n");
    body
}

#[test]
fn stream_length_recovery_stops_at_the_next_recorded_xref_offset() {
    // A stream with a wrong /Length can only be recovered if its unambiguous
    // `endstream` lies before the next recorded xref offset: recovery must
    // not cross into a neighboring object even when that offset is itself
    // inaccurate. Here object 5's offset points into object 4's payload, so
    // the recovery window ends mid-data and object 4 is dropped.
    let content = vec![b'A'; 48];
    let objects: Vec<(u32, Vec<u8>)> = vec![
        (1, catalog().to_vec()),
        (2, pages("3 0 R")),
        (3, page("/Im0 4 0 R")),
        (4, wrong_length_stream_body(&content)),
        (5, b"<< /Type /Font >>\n".to_vec()),
    ];
    let bodies: Vec<(u32, &[u8])> = objects.iter().map(|(id, body)| (*id, body.as_slice())).collect();
    let (pdf, _) = pdf_from_objects(&bodies, &[]);
    let false_offset = stream_data_start(&pdf) + 16;
    let (pdf, _) = pdf_from_objects(&bodies, &[(5, false_offset)]);

    let document = Document::load_mem(&pdf).unwrap();
    assert!(document.get_object((1, 0)).is_ok());
    assert!(document.get_object((2, 0)).is_ok());
    assert!(document.get_object((3, 0)).is_ok());
    assert!(document.get_object((4, 0)).is_err());
    assert!(document.get_object((5, 0)).is_err());
}
