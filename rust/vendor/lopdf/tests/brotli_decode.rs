use lopdf::{DecompressError, Dictionary, Document, Error, LoadOptions, Object, Stream};

const PAYLOAD: &[u8] = b"BrotliDecode works across filter layers.";
const COMPRESSED_PAYLOAD: &[u8] = &[
    27, 39, 0, 224, 28, 169, 83, 159, 59, 116, 45, 84, 246, 38, 39, 65, 136, 232, 26, 196, 37, 141, 13, 108, 192, 129,
    75, 84, 35, 221, 153, 78, 105, 229, 184, 47, 49, 2, 83, 8, 107, 4, 164,
];
const COMPRESSED_ASCII_HEX: &[u8] = &[11, 5, 128, 52, 56, 54, 53, 54, 99, 54, 99, 54, 102, 62, 3];
const COMPRESSED_XREF: &[u8] = &[
    27, 34, 0, 0, 4, 54, 224, 26, 71, 93, 117, 151, 221, 6, 135, 28, 200, 5, 14, 196, 81, 11, 176, 12, 144, 100, 228,
    47, 106, 153, 208, 15, 0,
];

fn brotli_stream(content: &[u8]) -> Stream {
    let mut dict = Dictionary::new();
    dict.set("Filter", "BrotliDecode");
    Stream::new(dict, content.to_vec())
}

fn brotli_xref_pdf() -> Vec<u8> {
    let mut pdf = Vec::new();
    pdf.extend_from_slice(b"%PDF-1.5\n");
    pdf.extend_from_slice(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n");
    pdf.extend_from_slice(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n");
    pdf.extend_from_slice(b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 10 10] >>\nendobj\n");
    let xref_offset = pdf.len();
    assert_eq!(xref_offset, 184, "compressed xref offsets must match the fixture body");
    pdf.extend_from_slice(
        b"4 0 obj\n<< /Type /XRef /Size 5 /Root 1 0 R /W [1 4 2] /Length 33 /Filter /BrotliDecode >>\nstream\n",
    );
    pdf.extend_from_slice(COMPRESSED_XREF);
    pdf.extend_from_slice(b"\nendstream\nendobj\nstartxref\n184\n%%EOF\n");
    pdf
}

#[test]
fn brotli_filter_decodes_and_honors_output_limit() {
    let stream = brotli_stream(COMPRESSED_PAYLOAD);

    assert_eq!(stream.decompressed_content().unwrap(), PAYLOAD);
    assert_eq!(stream.decompressed_content_with_limit(PAYLOAD.len()).unwrap(), PAYLOAD);
    assert!(matches!(
        stream.decompressed_content_with_limit(PAYLOAD.len() - 1),
        Err(Error::Decompress(DecompressError::MemoryLimitExceeded { limit })) if limit == PAYLOAD.len() - 1
    ));
}

#[test]
fn brotli_filter_is_bounded_before_the_next_filter_layer() {
    let mut dict = Dictionary::new();
    dict.set(
        "Filter",
        vec![Object::from("BrotliDecode"), Object::from("ASCIIHexDecode")],
    );
    let stream = Stream::new(dict, COMPRESSED_ASCII_HEX.to_vec());

    assert_eq!(stream.decompressed_content_with_limit(11).unwrap(), b"Hello");
    assert!(matches!(
        stream.decompressed_content_with_limit(5),
        Err(Error::Decompress(DecompressError::MemoryLimitExceeded { limit: 5 }))
    ));
}

#[test]
fn brotli_xref_stream_decodes_during_load_with_the_same_limit() {
    let pdf = brotli_xref_pdf();
    let document = Document::load_mem_with_options(&pdf, LoadOptions::with_max_decompressed_size(35)).unwrap();
    assert_eq!(document.get_pages().len(), 1);

    assert!(matches!(
        Document::load_mem_with_options(&pdf, LoadOptions::with_max_decompressed_size(34)),
        Err(Error::Decompress(DecompressError::MemoryLimitExceeded { limit: 34 }))
    ));
}

#[test]
fn brotli_filter_ignores_decode_parms() {
    // Producers do not pair /BrotliDecode with predictors — the PDF
    // Association's prototype files carry no /DecodeParms on any Brotli
    // stream (their xref streams store raw W entries) — and pdf.js/pypdf
    // likewise ignore the parameters. The decompressed bytes must therefore
    // pass through unmodified even when a /DecodeParms dictionary is present.
    let mut parms = Dictionary::new();
    parms.set("Predictor", 12);
    parms.set("Columns", 4);
    let mut dict = Dictionary::new();
    dict.set("Filter", Object::from("BrotliDecode"));
    dict.set("DecodeParms", Object::Dictionary(parms));
    let stream = Stream::new(dict, COMPRESSED_PAYLOAD.to_vec());

    assert_eq!(stream.decompressed_content().unwrap(), PAYLOAD);
}
