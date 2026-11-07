use lopdf::Document;

/// Build a minimal PDF where two ObjStm streams contain conflicting versions
/// of the same object (the Pages tree root). The xref Compressed entry points
/// to the higher-numbered ObjStm as authoritative, but the lower-numbered one
/// is processed first during loading.
///
/// Without the fix in load_objects_raw, the stale copy from the lower-numbered
/// ObjStm would win via or_insert, causing the document to report fewer pages.
fn build_conflicting_objstm_pdf() -> Vec<u8> {
    let mut buf = Vec::new();

    // Header
    buf.extend_from_slice(b"%PDF-1.5\n");

    // Object 1: Catalog
    let off_1 = buf.len();
    buf.extend_from_slice(b"1 0 obj\n<</Type/Catalog/Pages 3 0 R>>\nendobj\n");

    // Object 2: ObjStm with STALE copy of object 3 (Count=1, only 1 kid)
    let off_2 = buf.len();
    let stale_data = b"3 0 <</Type/Pages/Count 1/Kids[5 0 R]>>";
    buf.extend_from_slice(
        format!(
            "2 0 obj\n<</Type/ObjStm/N 1/First 4/Length {}>>\nstream\n",
            stale_data.len()
        )
        .as_bytes(),
    );
    buf.extend_from_slice(stale_data);
    buf.extend_from_slice(b"\nendstream\nendobj\n");

    // Object 4: ObjStm with CORRECT copy of object 3 (Count=2, 2 kids)
    let off_4 = buf.len();
    let correct_data = b"3 0 <</Type/Pages/Count 2/Kids[5 0 R 6 0 R]>>";
    buf.extend_from_slice(
        format!(
            "4 0 obj\n<</Type/ObjStm/N 1/First 4/Length {}>>\nstream\n",
            correct_data.len()
        )
        .as_bytes(),
    );
    buf.extend_from_slice(correct_data);
    buf.extend_from_slice(b"\nendstream\nendobj\n");

    // Object 5: Page
    let off_5 = buf.len();
    buf.extend_from_slice(b"5 0 obj\n<</Type/Page/Parent 3 0 R/MediaBox[0 0 612 792]>>\nendobj\n");

    // Object 6: Page
    let off_6 = buf.len();
    buf.extend_from_slice(b"6 0 obj\n<</Type/Page/Parent 3 0 R/MediaBox[0 0 612 792]>>\nendobj\n");

    // Object 7: Cross-reference stream
    // W=[1 3 1]: type (1 byte), field2 (3 bytes), field3 (1 byte)
    // Entries for objects 0-7:
    //   0: Free
    //   1: Normal (Catalog)
    //   2: Normal (stale ObjStm)
    //   3: Compressed in container=4, index=0  <-- authoritative
    //   4: Normal (correct ObjStm)
    //   5: Normal (Page)
    //   6: Normal (Page)
    //   7: Normal (xref stream itself)
    let off_7 = buf.len();

    let mut xref_data = Vec::new();
    let offsets = [
        0u32,
        off_1 as u32,
        off_2 as u32,
        0,
        off_4 as u32,
        off_5 as u32,
        off_6 as u32,
        off_7 as u32,
    ];

    // Object 0: Free
    xref_data.extend_from_slice(&[0, 0, 0, 0, 0]);

    // Objects 1, 2: Normal
    for &off in &offsets[1..=2] {
        xref_data.push(1);
        xref_data.push((off >> 16) as u8);
        xref_data.push((off >> 8) as u8);
        xref_data.push(off as u8);
        xref_data.push(0);
    }

    // Object 3: Compressed { container: 4, index: 0 }
    xref_data.push(2);
    xref_data.extend_from_slice(&[0, 0, 4]); // container = 4
    xref_data.push(0); // index = 0

    // Objects 4, 5, 6, 7: Normal
    for &off in &offsets[4..=7] {
        xref_data.push(1);
        xref_data.push((off >> 16) as u8);
        xref_data.push((off >> 8) as u8);
        xref_data.push(off as u8);
        xref_data.push(0);
    }

    buf.extend_from_slice(
        format!(
            "7 0 obj\n<</Type/XRef/Size 8/W[1 3 1]/Root 1 0 R/Length {}>>\nstream\n",
            xref_data.len()
        )
        .as_bytes(),
    );
    buf.extend_from_slice(&xref_data);
    buf.extend_from_slice(b"\nendstream\nendobj\n");

    buf.extend_from_slice(format!("startxref\n{}\n%%EOF", off_7).as_bytes());
    buf
}

#[test]
fn test_conflicting_objstm_uses_xref_container() {
    let pdf = build_conflicting_objstm_pdf();
    let doc = Document::load_mem(&pdf).unwrap();

    // The xref says object 3 (Pages root) belongs to ObjStm 4 (Count=2).
    // ObjStm 2 has a stale copy with Count=1. The correct version must win.
    assert_eq!(
        doc.get_pages().len(),
        2,
        "Should load 2 pages from ObjStm 4, not 1 from stale ObjStm 2"
    );
}
