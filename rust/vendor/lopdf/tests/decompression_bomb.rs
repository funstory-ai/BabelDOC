//! Tests for bounded stream decompression (decompression-bomb handling).
//!
//! By default `Stream::decompressed_content` is unbounded, so a small compressed
//! stream can inflate to an arbitrarily large output (a decompression bomb).
//! These tests cover both the unbounded behavior and the bounded API
//! (`decompressed_content_with_limit`, `decompress_to_writer`, and
//! `LoadOptions::max_decompressed_size`) that rejects oversized output,
//! including nested filters and streams decoded during `Document::load`.
//!
//! Bombs are built by streaming zeros through the compressor, so the test
//! process itself never allocates the full (large) plaintext.

use flate2::Compression;
use flate2::write::ZlibEncoder;
use std::io::Write;

use lopdf::{DecompressError, Dictionary, Document, Error, LoadOptions, Object, ObjectId, ObjectStream, Stream};

const MIB: usize = 1024 * 1024;

/// A zlib stream that decodes to `target` zero-bytes, built without ever holding
/// `target` bytes in memory at once.
fn flate_bomb(target: usize) -> Vec<u8> {
    let mut encoder = ZlibEncoder::new(Vec::new(), Compression::best());
    let zeros = [0u8; 64 * 1024];
    let mut remaining = target;
    while remaining > 0 {
        let n = remaining.min(zeros.len());
        encoder.write_all(&zeros[..n]).unwrap();
        remaining -= n;
    }
    encoder.finish().unwrap()
}

/// Compress arbitrary bytes with zlib (always emits a valid stream).
fn zlib_compress(data: &[u8]) -> Vec<u8> {
    let mut encoder = ZlibEncoder::new(Vec::new(), Compression::best());
    encoder.write_all(data).unwrap();
    encoder.finish().unwrap()
}

fn flate_stream(content: Vec<u8>) -> Stream {
    let mut dict = Dictionary::new();
    dict.set("Filter", "FlateDecode");
    Stream::new(dict, content)
}

/// Assert a result is the memory-limit error for `expected_limit`, without ever
/// `Debug`-printing a multi-megabyte success value.
fn assert_limit_err(result: lopdf::Result<Vec<u8>>, expected_limit: usize) {
    match result.map(|v| v.len()) {
        Err(Error::Decompress(DecompressError::MemoryLimitExceeded { limit })) => {
            assert_eq!(limit, expected_limit, "error reported the wrong limit");
        }
        Ok(n) => panic!("expected MemoryLimitExceeded, but decompressed {n} bytes"),
        Err(other) => panic!("expected MemoryLimitExceeded, got {other:?}"),
    }
}

/// A single `FlateDecode` stream inflates ~32 KiB of input to 32 MiB of output
/// (~1000x) with no cap — the library materializes the whole thing.
#[test]
fn single_filter_bomb_expands_without_limit() {
    let bomb = flate_stream(flate_bomb(32 * MIB));
    assert!(
        bomb.content.len() < 128 * 1024,
        "input should be tiny: {} bytes",
        bomb.content.len()
    );

    let output = bomb.decompressed_content().expect("decodes");
    assert_eq!(output.len(), 32 * MIB, "the full output was materialized with no cap");
}

/// Chained filters multiply the amplification: under 8 KiB of input expands to
/// 32 MiB through `[FlateDecode FlateDecode]`. Additional layers reach petabyte
/// scale.
#[test]
fn chained_filter_bomb_expands_without_limit() {
    let inner = flate_bomb(32 * MIB); // layer 2 decodes this -> 32 MiB
    let outer = zlib_compress(&inner); // layer 1 decodes to `inner` (tiny)

    let mut dict = Dictionary::new();
    dict.set("Filter", vec![Object::from("FlateDecode"), Object::from("FlateDecode")]);
    let stream = Stream::new(dict, outer);

    assert!(
        stream.content.len() < 8 * 1024,
        "input should be under 8 KiB: {} bytes",
        stream.content.len()
    );

    let output = stream.decompressed_content().expect("decodes");
    assert_eq!(output.len(), 32 * MIB, "nested filters expanded fully, no cap");
}

/// With a limit supplied, the same bomb is rejected before the full 32 MiB is
/// allocated.
#[test]
fn flate_bomb_rejected_with_limit() {
    let bomb = flate_stream(flate_bomb(32 * MIB));
    assert_limit_err(bomb.decompressed_content_with_limit(4 * MIB), 4 * MIB);
}

/// A stream comfortably under the limit still decodes correctly.
#[test]
fn legit_stream_decompresses_under_limit() {
    let payload: Vec<u8> = (0..64 * 1024).map(|i| (i % 251) as u8).collect();
    let stream = flate_stream(zlib_compress(&payload));

    let output = stream
        .decompressed_content_with_limit(MIB)
        .expect("stream under the limit should decode");
    assert_eq!(output, payload);
}

/// The same stream passes or fails based on the chosen limit.
#[test]
fn custom_limit_is_enforced() {
    let stream = flate_stream(flate_bomb(2 * MIB));

    // Limit below the output size: rejected.
    assert_limit_err(stream.decompressed_content_with_limit(MIB), MIB);

    // Limit above the output size: decoded in full.
    let output = stream
        .decompressed_content_with_limit(4 * MIB)
        .expect("2 MiB output fits under a 4 MiB limit");
    assert_eq!(output.len(), 2 * MIB);
}

/// Nested filters (`/Filter [FlateDecode FlateDecode]`) amplify per layer. The
/// limit bounds each layer, so the bomb is rejected at the second layer instead
/// of expanding.
#[test]
fn chained_flate_filters_are_bounded() {
    let inner = flate_bomb(32 * MIB); // layer 2 decodes this -> 32 MiB
    let outer = zlib_compress(&inner); // layer 1 decodes to `inner` (tiny)

    let mut dict = Dictionary::new();
    dict.set("Filter", vec![Object::from("FlateDecode"), Object::from("FlateDecode")]);
    let stream = Stream::new(dict, outer);

    assert_limit_err(stream.decompressed_content_with_limit(4 * MIB), 4 * MIB);
}

/// `decompress_to_writer` enforces the same bound and leaves the caller's sink
/// untouched on rejection.
#[test]
fn decompress_to_writer_bounds_output() {
    let bomb = flate_stream(flate_bomb(32 * MIB));

    let mut sink: Vec<u8> = Vec::new();
    let result = bomb.decompress_to_writer(&mut sink, 4 * MIB);

    match result {
        Err(Error::Decompress(DecompressError::MemoryLimitExceeded { limit })) => {
            assert_eq!(limit, 4 * MIB);
        }
        other => panic!(
            "expected MemoryLimitExceeded, got {:?}",
            other.map(|n| format!("Ok({n})"))
        ),
    }
    assert!(
        sink.is_empty(),
        "sink should be untouched on rejection, got {} bytes",
        sink.len()
    );
}

// Object streams and cross-reference streams are decompressed while the document
// is loaded, so the limit is supplied through `LoadOptions` to reach them.

/// Build a PDF whose cross-reference stream (`/Type /XRef`) is a FlateDecode
/// bomb. The reference stream is decoded during `Document::load` to build the
/// xref table.
fn xref_stream_bomb_pdf(bomb: &[u8]) -> Vec<u8> {
    let mut pdf = Vec::new();
    pdf.extend_from_slice(b"%PDF-1.5\n");
    let obj_offset = pdf.len();
    pdf.extend_from_slice(b"1 0 obj\n");
    pdf.extend_from_slice(
        format!(
            "<< /Type /XRef /Size 1 /W [1 1 1] /Root 1 0 R /Filter /FlateDecode /Length {} >>\n",
            bomb.len()
        )
        .as_bytes(),
    );
    // Framing matches lopdf's parser: `stream<eol><exactly Length bytes><eol>endstream`.
    pdf.extend_from_slice(b"stream\n");
    pdf.extend_from_slice(bomb);
    pdf.extend_from_slice(b"\nendstream\nendobj\n");
    pdf.extend_from_slice(format!("startxref\n{obj_offset}\n%%EOF").as_bytes());
    pdf
}

/// A bomb in the cross-reference stream is rejected during load when
/// `max_decompressed_size` is set.
#[test]
fn load_time_xref_stream_bomb_is_rejected_with_limit() {
    let pdf = xref_stream_bomb_pdf(&flate_bomb(64 * MIB));

    let result = Document::load_mem_with_options(&pdf, LoadOptions::with_max_decompressed_size(4 * MIB));

    match result {
        Err(Error::Decompress(DecompressError::MemoryLimitExceeded { limit })) => {
            assert_eq!(limit, 4 * MIB);
        }
        Err(other) => panic!("expected MemoryLimitExceeded during load, got {other:?}"),
        Ok(_) => panic!("bomb PDF loaded without hitting the decompression limit"),
    }
}

// Page content is decompressed on demand by `Document::get_page_content` (and
// thus `extract_text`), which is unbounded. `get_page_content_with_limit` bounds
// the whole concatenated page content to a caller-supplied budget.

/// Build a single-page document whose `/Contents` is the given streams, and
/// return the document plus the page's object id. No page tree / fonts, so this
/// exercises `get_page_content_with_limit` directly (not the full extract path).
fn doc_with_page_streams(streams: Vec<Stream>) -> (Document, ObjectId) {
    let mut doc = Document::with_version("1.5");
    let content_refs: Vec<Object> = streams.into_iter().map(|s| doc.add_object(s).into()).collect();
    let mut page = Dictionary::new();
    page.set("Type", "Page");
    page.set("Contents", Object::Array(content_refs));
    let page_id = doc.add_object(page);
    (doc, page_id)
}

/// A bomb in a page's content stream is rejected by `get_page_content_with_limit`
/// before the full output is allocated.
#[test]
fn page_content_bomb_is_rejected_with_limit() {
    let (doc, page_id) = doc_with_page_streams(vec![flate_stream(flate_bomb(32 * MIB))]);
    assert_limit_err(doc.get_page_content_with_limit(page_id, 4 * MIB), 4 * MIB);
}

/// A page comfortably under the limit yields exactly the same bytes as the
/// unbounded `get_page_content` (including the inter-stream `\n` separators).
#[test]
fn page_content_under_limit_matches_unbounded() {
    let (doc, page_id) = doc_with_page_streams(vec![
        flate_stream(zlib_compress(b"1 0 0 1 50 100 cm")),
        flate_stream(zlib_compress(b"BT /F1 12 Tf (hi) Tj ET")),
    ]);

    let bounded = doc
        .get_page_content_with_limit(page_id, MIB)
        .expect("small page content should decode under the limit");
    assert_eq!(bounded, doc.get_page_content(page_id));
}

/// The limit is a *total-page* budget, not a per-stream one: two streams that are
/// each individually under the limit but together exceed it are rejected. (A
/// per-stream bound would have let this through.)
#[test]
fn page_content_limit_is_total_across_streams() {
    let (doc, page_id) = doc_with_page_streams(vec![
        flate_stream(flate_bomb(3 * MIB)),
        flate_stream(flate_bomb(3 * MIB)),
    ]);

    // Each stream (3 MiB) is under 4 MiB, but 3 + 3 > 4, so the page is rejected.
    assert_limit_err(doc.get_page_content_with_limit(page_id, 4 * MIB), 4 * MIB);
}

/// A large *uncompressed* (no `/Filter`) content stream is also bounded: the raw
/// bytes count against the budget, so it can't bypass the guard.
#[test]
fn page_content_uncompressed_over_limit_is_rejected() {
    let raw = Stream::new(Dictionary::new(), vec![b'x'; 2 * MIB]);
    let (doc, page_id) = doc_with_page_streams(vec![raw]);
    assert_limit_err(doc.get_page_content_with_limit(page_id, MIB), MIB);
}

/// When a stream can't be decoded for a reason *other* than the size limit (here
/// an unimplemented filter), the code falls back to the raw bytes — but that
/// fallback is still bounded, so oversized raw content is rejected rather than
/// silently appended.
#[test]
fn page_content_raw_fallback_stays_bounded() {
    let mut dict = Dictionary::new();
    dict.set("Filter", "DCTDecode"); // not implemented → decode fails with a non-limit error
    let undecodable = Stream::new(dict, vec![b'x'; 2 * MIB]);

    let (doc, page_id) = doc_with_page_streams(vec![undecodable]);
    assert_limit_err(doc.get_page_content_with_limit(page_id, MIB), MIB);
}

/// A zero limit rejects any non-empty page content without panicking (the
/// `remaining` budget uses `saturating_sub`).
#[test]
fn page_content_zero_limit_rejects_nonempty() {
    let (doc, page_id) = doc_with_page_streams(vec![flate_stream(zlib_compress(b"BT ET"))]);
    assert_limit_err(doc.get_page_content_with_limit(page_id, 0), 0);
}

/// An object stream (`/ObjStm`) is decompressed eagerly during load; the bounded
/// constructor rejects an oversized stream.
#[test]
fn object_stream_bomb_is_rejected_with_limit() {
    let mut dict = Dictionary::new();
    dict.set("Type", "ObjStm");
    dict.set("N", 1i64);
    dict.set("First", 0i64);
    dict.set("Filter", "FlateDecode");
    let stream = Stream::new(dict, flate_bomb(64 * MIB));

    match ObjectStream::new_with_limit(&stream, Some(4 * MIB)) {
        Err(Error::Decompress(DecompressError::MemoryLimitExceeded { limit })) => {
            assert_eq!(limit, 4 * MIB);
        }
        Err(other) => panic!("expected MemoryLimitExceeded, got {other:?}"),
        Ok(_) => panic!("object stream bomb was not caught"),
    }
}
