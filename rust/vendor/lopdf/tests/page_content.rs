use lopdf::{Document, Stream, content::Content, dictionary};

#[test]
fn get_page_content_separates_streams_at_token_boundary() {
    let mut doc = Document::with_version("1.5");
    // Producer writes stream that ends without trailing whitespace
    let s1 = doc.add_object(Stream::new(dictionary! {}, b"1 0 0 1 50".to_vec()));
    let s2 = doc.add_object(Stream::new(dictionary! {}, b"100 cm".to_vec()));
    let page_id = doc.add_object(dictionary! {
        "Type" => "Page",
        "Contents" => vec![s1.into(), s2.into()],
    });

    let bytes = doc.get_page_content(page_id);
    let content = Content::decode(&bytes).unwrap();

    assert_eq!(content.operations.len(), 1);
    let op = &content.operations[0];
    assert_eq!(op.operator, "cm");
    let operands: Vec<f32> = op.operands.iter().map(|o| o.as_float().unwrap()).collect();

    assert_eq!(operands, [1.0, 0.0, 0.0, 1.0, 50.0, 100.0]);
}

#[test]
fn get_page_content_terminates_trailing_comment() {
    let mut doc = Document::with_version("1.5");
    // Producer writes stream that ends with a comment without EOL
    let s1 = doc.add_object(Stream::new(dictionary! {}, b"% a comment".to_vec()));
    let s2 = doc.add_object(Stream::new(dictionary! {}, b"1 0 0 1 50 100 cm".to_vec()));
    let page_id = doc.add_object(dictionary! {
        "Type" => "Page",
        "Contents" => vec![s1.into(), s2.into()],
    });

    let bytes = doc.get_page_content(page_id);
    let content = Content::decode(&bytes).unwrap();

    assert_eq!(content.operations.len(), 1);
    let op = &content.operations[0];
    assert_eq!(op.operator, "cm");
    let operands: Vec<f32> = op.operands.iter().map(|o| o.as_float().unwrap()).collect();

    assert_eq!(operands, [1.0, 0.0, 0.0, 1.0, 50.0, 100.0]);
}
