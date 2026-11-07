use lopdf::{Document, Object, ObjectId, ObjectStream, Stream, dictionary};

#[test]
fn test_object_stream_builder() {
    let builder = ObjectStream::builder();
    assert_eq!(builder.get_max_objects(), 100);
    assert_eq!(builder.get_compression_level(), 6);
}

#[test]
fn test_object_stream_add_and_build() {
    let mut obj_stream = ObjectStream::builder().build();

    // Add some objects
    obj_stream.add_object((1, 0), Object::Integer(42)).unwrap();
    obj_stream.add_object((2, 0), Object::Boolean(true)).unwrap();
    obj_stream.add_object((3, 0), Object::Name(b"Test".to_vec())).unwrap();

    // Build the stream content
    let content = obj_stream.build_stream_content().unwrap();
    assert!(!content.is_empty());

    // Convert to stream object
    let stream_obj = obj_stream.to_stream_object().unwrap();
    assert_eq!(stream_obj.dict.get(b"Type").unwrap(), &Object::Name(b"ObjStm".to_vec()));
    assert_eq!(stream_obj.dict.get(b"N").unwrap(), &Object::Integer(3));
}

#[test]
fn test_save_with_object_streams() {
    let mut doc = Document::with_version("1.5");
    let pages_id = doc.new_object_id();

    let font_id = doc.add_object(dictionary! {
        "Type" => "Font",
        "Subtype" => "Type1",
        "BaseFont" => "Helvetica"
    });

    let resources_id = doc.add_object(dictionary! {
        "Font" => dictionary! {
            "F1" => font_id,
        },
    });

    let content = lopdf::content::Content {
        operations: vec![
            lopdf::content::Operation::new("BT", vec![]),
            lopdf::content::Operation::new("Tf", vec!["F1".into(), 12.into()]),
            lopdf::content::Operation::new("Td", vec![50.into(), 700.into()]),
            lopdf::content::Operation::new("Tj", vec![Object::string_literal("Test Document")]),
            lopdf::content::Operation::new("ET", vec![]),
        ],
    };

    let content_id = doc.add_object(Stream::new(dictionary! {}, content.encode().unwrap()));

    let page_id = doc.add_object(dictionary! {
        "Type" => "Page",
        "Parent" => pages_id,
        "Contents" => content_id,
        "Resources" => resources_id,
        "MediaBox" => vec![0.into(), 0.into(), 612.into(), 792.into()],
    });

    doc.objects.insert(
        pages_id,
        Object::Dictionary(dictionary! {
            "Type" => "Pages",
            "Kids" => vec![page_id.into()],
            "Count" => 1,
        }),
    );

    let catalog_id = doc.add_object(dictionary! {
        "Type" => "Catalog",
        "Pages" => pages_id,
    });

    doc.trailer.set("Root", catalog_id);

    // Test save_modern
    let mut buffer = Vec::new();
    let result = doc.save_modern(&mut buffer);
    assert!(result.is_ok());

    // Verify PDF was created
    let content = String::from_utf8_lossy(&buffer);
    assert!(content.starts_with("%PDF-1.5"));

    // Verify object streams were created
    assert!(content.contains("/ObjStm"), "Object streams should be created");

    // Verify that structural objects are NOT present as individual objects
    assert!(
        !content.contains("2 0 obj\n<</Type/Pages"),
        "Pages object should be compressed"
    );
    assert!(
        !content.contains("3 0 obj\n<</Type/Page"),
        "Page object should be compressed"
    );
}

/// Construct a raw ObjStm whose embedded objects are preceded by newlines
/// (as seen in real PDFs like MAECI documents). Without the whitespace-
/// skipping fix in ObjectStream::new, all objects would silently fail to
/// parse and the returned ObjectStream would be empty.
#[test]
fn test_object_stream_parses_objects_with_leading_whitespace() {
    // Build the object data portion: two dictionaries preceded by \n
    // Object 10: a Font dictionary
    // Object 11: a FontDescriptor dictionary
    let obj10_bytes = b"\n<< /Type /Font /Subtype /TrueType /BaseFont /Calibri >>";
    let obj11_bytes = b"\n<< /Type /FontDescriptor /FontName /Calibri >>";

    let obj10_offset = 0usize;
    let obj11_offset = obj10_bytes.len();

    // Build the index block: "obj_num offset obj_num offset "
    let index = format!("10 {} 11 {} ", obj10_offset, obj11_offset);
    let first_offset = index.len();

    // Assemble the full stream content: index block + object data
    let mut content = Vec::new();
    content.extend_from_slice(index.as_bytes());
    content.extend_from_slice(obj10_bytes);
    content.extend_from_slice(obj11_bytes);

    let dict = dictionary! {
        "Type" => "ObjStm",
        "N" => 2,
        "First" => first_offset as i64,
    };

    let stream = Stream::new(dict, content);
    let obj_stream = ObjectStream::new(&stream).expect("should parse object stream");

    // Both objects must be present
    assert_eq!(
        obj_stream.objects.len(),
        2,
        "expected 2 objects but got {}; leading whitespace was not skipped",
        obj_stream.objects.len()
    );

    // Verify object 10 is a Font dictionary
    let obj10 = obj_stream
        .objects
        .get(&(10u32, 0u16) as &ObjectId)
        .expect("object 10 missing");
    if let Object::Dictionary(dict) = obj10 {
        assert_eq!(dict.get(b"BaseFont").unwrap().as_name().unwrap(), b"Calibri");
    } else {
        panic!("object 10 should be a Dictionary, got {:?}", obj10);
    }

    // Verify object 11 is a FontDescriptor dictionary
    let obj11 = obj_stream
        .objects
        .get(&(11u32, 0u16) as &ObjectId)
        .expect("object 11 missing");
    if let Object::Dictionary(dict) = obj11 {
        assert_eq!(dict.get(b"FontName").unwrap().as_name().unwrap(), b"Calibri");
    } else {
        panic!("object 11 should be a Dictionary, got {:?}", obj11);
    }
}
