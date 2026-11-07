use lopdf::{Document, Error, Object, ObjectStream, SaveOptions, Stream, dictionary};

#[test]
fn test_object_stream_parse_empty() {
    let stream = Stream::new(
        dictionary! {
            "Type" => "ObjStm",
            "N" => 0,
            "First" => 0
        },
        vec![],
    );

    let obj_stream = ObjectStream::new(&stream).unwrap();
    assert_eq!(obj_stream.objects.len(), 0);
}

#[test]
fn test_object_stream_parse_preserves_encoded_content() {
    let decoded = b"1 0 << /Kind /ArchiveEntry /Padding (aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa) >>";
    let mut stream = Stream::new(
        dictionary! {
            "Type" => "ObjStm",
            "N" => 1,
            "First" => 4,
        },
        decoded.to_vec(),
    );
    stream.compress().unwrap();
    let encoded = stream.content.clone();

    let object_stream = ObjectStream::new_with_limit(&stream, Some(decoded.len())).unwrap();

    assert!(object_stream.objects.contains_key(&(1, 0)));
    assert_eq!(stream.content, encoded);
    assert_eq!(stream.dict.get(b"Filter").unwrap().as_name().unwrap(), b"FlateDecode");
}

#[test]
fn test_object_stream_parse_reports_unsupported_filters_without_mutating_input() {
    let encoded = b"1 0 << >>".to_vec();
    let stream = Stream::new(
        dictionary! {
            "Type" => "ObjStm",
            "N" => 1,
            "First" => 4,
            "Filter" => "DCTDecode",
        },
        encoded.clone(),
    );

    let result = ObjectStream::new(&stream);

    assert!(matches!(result, Err(Error::Unimplemented("decompression algorithms"))));
    assert_eq!(stream.content, encoded);
    assert_eq!(stream.dict.get(b"Filter").unwrap().as_name().unwrap(), b"DCTDecode");
}

#[test]
fn test_object_stream_parse_with_empty_filter_array() {
    let stream = Stream::new(
        dictionary! {
            "Type" => "ObjStm",
            "N" => 1,
            "First" => 4,
            "Filter" => Object::Array(vec![]),
        },
        b"1 0 42".to_vec(),
    );

    let object_stream = ObjectStream::new(&stream).unwrap();

    assert_eq!(object_stream.objects.get(&(1, 0)), Some(&Object::Integer(42)));
}

#[test]
fn test_object_stream_config_default() {
    let config = lopdf::ObjectStreamConfig::default();
    assert_eq!(config.max_objects_per_stream, 100);
    assert_eq!(config.compression_level, 6);
}

#[test]
fn test_object_stream_to_stream_object() {
    let mut obj_stream = ObjectStream::builder().build();

    // Add some objects
    obj_stream.add_object((1, 0), Object::Integer(42)).unwrap();
    obj_stream.add_object((2, 0), Object::Boolean(true)).unwrap();
    obj_stream.add_object((3, 0), Object::Name(b"Test".to_vec())).unwrap();

    // Convert to stream object
    let stream_obj = obj_stream.to_stream_object().unwrap();

    // Verify stream properties
    assert_eq!(stream_obj.dict.get(b"Type").unwrap(), &Object::Name(b"ObjStm".to_vec()));
    assert_eq!(stream_obj.dict.get(b"N").unwrap(), &Object::Integer(3));
    assert!(stream_obj.dict.has(b"First"));
    assert!(stream_obj.dict.has(b"Length"));

    // Verify content is not empty
    assert!(!stream_obj.content.is_empty());
}

#[test]
fn test_save_options_with_object_streams() {
    let mut doc = Document::with_version("1.5");
    let pages_id = doc.new_object_id();

    let page_id = doc.add_object(dictionary! {
        "Type" => "Page",
        "Parent" => pages_id,
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

    // Save with object streams enabled
    let save_options = SaveOptions {
        use_object_streams: true,
        use_xref_streams: false,
        linearize: false,
        object_stream_config: Default::default(),
    };

    let mut buffer = Vec::new();
    doc.save_with_options(&mut buffer, save_options).unwrap();

    let content = String::from_utf8_lossy(&buffer);

    // Verify object streams were created
    assert!(content.contains("/ObjStm"), "Object streams should be created");

    // Verify PDF version is 1.5 or higher
    assert!(
        content.starts_with("%PDF-1.5") || content.starts_with("%PDF-1.6") || content.starts_with("%PDF-1.7"),
        "PDF version should be 1.5 or higher for object streams"
    );
}

#[test]
fn test_save_options_without_object_streams() {
    let mut doc = Document::with_version("1.4");
    let pages_id = doc.new_object_id();

    let page_id = doc.add_object(dictionary! {
        "Type" => "Page",
        "Parent" => pages_id,
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

    // Save without object streams
    let save_options = SaveOptions {
        use_object_streams: false,
        use_xref_streams: false,
        linearize: false,
        object_stream_config: Default::default(),
    };

    let mut buffer = Vec::new();
    doc.save_with_options(&mut buffer, save_options).unwrap();

    let content = String::from_utf8_lossy(&buffer);

    // Verify no object streams were created
    assert!(!content.contains("/ObjStm"), "Object streams should not be created");

    // Verify objects exist as individual objects
    assert!(content.contains(&format!("{} 0 obj", catalog_id.0)));
    assert!(content.contains(&format!("{} 0 obj", pages_id.0)));
    assert!(content.contains(&format!("{} 0 obj", page_id.0)));
}

#[test]
fn test_encrypted_document_object_streams() {
    let mut doc = Document::with_version("1.5");
    let pages_id = doc.new_object_id();

    // Create encryption dictionary
    let encrypt_id = doc.add_object(dictionary! {
        "Filter" => "Standard",
        "V" => 2,
        "R" => 3,
        "Length" => 128,
        "P" => -1,
        "O" => Object::String(vec![0; 32], lopdf::StringFormat::Hexadecimal),
        "U" => Object::String(vec![0; 32], lopdf::StringFormat::Hexadecimal),
    });

    doc.trailer.set("Encrypt", encrypt_id);

    let page_id = doc.add_object(dictionary! {
        "Type" => "Page",
        "Parent" => pages_id,
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

    // Verify encryption dictionary cannot be compressed
    let encrypt_obj = doc.objects.get(&encrypt_id).unwrap();
    assert!(
        !ObjectStream::can_be_compressed(encrypt_id, encrypt_obj, &doc),
        "Encryption dictionary must not be compressible"
    );

    let mut buffer = Vec::new();
    let result = doc.save_modern(&mut buffer);
    assert!(result.is_ok());

    let content = String::from_utf8_lossy(&buffer);

    // Encryption dictionary should remain as top-level object
    assert!(
        content.contains(&format!("{} 0 obj", encrypt_id.0)),
        "Encryption dictionary must remain as individual object"
    );
}

#[test]
fn test_linearized_pdf_catalog_handling() {
    let mut doc = Document::with_version("1.5");

    // Add linearization parameters
    let _lin_id = doc.add_object(dictionary! {
        "Linearized" => 1,
        "L" => 50000,  // File length
        "H" => vec![Object::Integer(1000), Object::Integer(2000)],  // Hint stream location
        "O" => 10,  // First page object number
        "E" => 5000,  // End of first page
        "N" => 1,  // Number of pages
        "T" => 45000  // First entry in main cross-reference table
    });

    let pages_id = doc.new_object_id();

    let page_id = doc.add_object(dictionary! {
        "Type" => "Page",
        "Parent" => pages_id,
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

    // In linearized PDF, catalog should not be compressible
    let catalog_obj = doc.objects.get(&catalog_id).unwrap();
    assert!(
        !ObjectStream::can_be_compressed(catalog_id, catalog_obj, &doc),
        "Catalog must not be compressible in linearized PDFs"
    );

    // But Pages and Page should still be compressible
    let pages_obj = doc.objects.get(&pages_id).unwrap();
    let page_obj = doc.objects.get(&page_id).unwrap();
    assert!(
        ObjectStream::can_be_compressed(pages_id, pages_obj, &doc),
        "Pages should still be compressible in linearized PDFs"
    );
    assert!(
        ObjectStream::can_be_compressed(page_id, page_obj, &doc),
        "Page should still be compressible in linearized PDFs"
    );
}

#[test]
fn test_object_stream_with_references() {
    let mut obj_stream = ObjectStream::builder().build();

    // Add objects with references
    obj_stream
        .add_object(
            (1, 0),
            Object::Dictionary(dictionary! {
                "Type" => "Font",
                "BaseFont" => "Helvetica",
                "Encoding" => Object::Reference((5, 0))  // Reference to another object
            }),
        )
        .unwrap();

    obj_stream
        .add_object(
            (2, 0),
            Object::Array(vec![
                Object::Integer(100),
                Object::Reference((10, 0)), // Reference
                Object::String(b"Test".to_vec(), lopdf::StringFormat::Literal),
            ]),
        )
        .unwrap();

    let stream_obj = obj_stream.to_stream_object().unwrap();
    assert_eq!(stream_obj.dict.get(b"N").unwrap(), &Object::Integer(2));

    // Verify content contains the references
    let content = String::from_utf8_lossy(&stream_obj.content);
    assert!(content.contains("5 0 R")); // Reference syntax
    assert!(content.contains("10 0 R"));
}

#[test]
fn test_object_stream_with_nested_dictionaries() {
    let mut obj_stream = ObjectStream::builder().build();

    // Add complex nested dictionary
    obj_stream
        .add_object(
            (1, 0),
            Object::Dictionary(dictionary! {
                "Type" => "ExtGState",
                "CA" => 0.5,
                "ca" => 0.5,
                "BM" => Object::Array(vec![
                    Object::Name(b"Normal".to_vec()),
                    Object::Name(b"Multiply".to_vec())
                ]),
                "SMask" => dictionary! {
                    "Type" => "Mask",
                    "S" => "Alpha",
                    "G" => Object::Reference((10, 0))
                }
            }),
        )
        .unwrap();

    let stream_obj = obj_stream.to_stream_object().unwrap();
    assert!(!stream_obj.content.is_empty());
}

#[test]
fn test_multiple_object_streams_in_document() {
    let mut doc = Document::with_version("1.5");
    let pages_id = doc.new_object_id();

    // Add many objects to force multiple object streams
    let mut page_ids = vec![];
    for i in 0..250 {
        // More than default max_objects_per_stream
        let page_id = doc.add_object(dictionary! {
            "Type" => "Page",
            "Parent" => pages_id,
            "MediaBox" => vec![0.into(), 0.into(), 612.into(), 792.into()],
            "Contents" => Object::Reference((1000 + i, 0))
        });
        page_ids.push(page_id);
    }

    doc.objects.insert(
        pages_id,
        Object::Dictionary(dictionary! {
            "Type" => "Pages",
            "Kids" => page_ids.into_iter().map(Object::Reference).collect::<Vec<_>>(),
            "Count" => 250,
        }),
    );

    let catalog_id = doc.add_object(dictionary! {
        "Type" => "Catalog",
        "Pages" => pages_id,
    });

    doc.trailer.set("Root", catalog_id);

    let mut buffer = Vec::new();
    doc.save_modern(&mut buffer).unwrap();

    let content = String::from_utf8_lossy(&buffer);

    // Count object streams
    let objstm_count = content.matches("/ObjStm").count();
    assert!(
        objstm_count >= 3,
        "Should have multiple object streams for 250+ objects"
    );
}

#[test]
fn test_loads_catalog_from_later_generated_object_stream() {
    let mut doc = Document::with_version("1.5");
    let pages_id = doc.new_object_id();

    let page_id = doc.add_object(dictionary! {
        "Type" => "Page",
        "Parent" => pages_id,
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

    // Fill the document with enough objects to force multiple object streams.
    let max_objects_per_stream = lopdf::ObjectStreamConfig::default().max_objects_per_stream;
    for _ in 0..=max_objects_per_stream {
        doc.add_object(dictionary! {
            "Dummy" => true,
        });
    }

    let catalog_id = doc.add_object(dictionary! {
        "Type" => "Catalog",
        "Pages" => pages_id,
    });

    doc.trailer.set("Root", catalog_id);

    let mut buffer = Vec::new();
    doc.save_modern(&mut buffer).unwrap();

    let content = String::from_utf8_lossy(&buffer);
    let objstm_count = content.matches("/ObjStm").count();
    assert!(objstm_count >= 2, "should have multiple object streams");

    let loaded = Document::load_mem(&buffer).unwrap();
    loaded
        .catalog()
        .expect("catalog should remain reachable from a later generated object stream");
    assert_eq!(loaded.get_pages().len(), 1);
}
