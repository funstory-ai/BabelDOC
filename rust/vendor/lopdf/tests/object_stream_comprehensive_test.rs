use lopdf::{Document, Object, ObjectStream, Stream, dictionary};

#[test]
fn test_can_be_compressed_stream_objects() {
    let mut doc = Document::new();
    let stream_id = doc.add_object(Stream::new(dictionary! { "Type" => "XObject" }, vec![1, 2, 3]));

    let stream_obj = doc.objects.get(&stream_id).unwrap();
    assert!(
        !ObjectStream::can_be_compressed(stream_id, stream_obj, &doc),
        "Stream objects should not be compressible"
    );
}

#[test]
fn test_can_be_compressed_non_zero_generation() {
    let mut doc = Document::new();
    let obj_id = (1, 5); // Non-zero generation
    doc.objects.insert(obj_id, Object::Integer(42));

    let obj = doc.objects.get(&obj_id).unwrap();
    assert!(
        !ObjectStream::can_be_compressed(obj_id, obj, &doc),
        "Objects with non-zero generation should not be compressible"
    );
}

#[test]
fn test_can_be_compressed_trailer_referenced() {
    let mut doc = Document::new();
    let obj_id = doc.add_object(Object::Dictionary(dictionary! {
        "Test" => "Value"
    }));

    // Add to trailer (non-encryption reference)
    doc.trailer.set("TestRef", obj_id);

    let obj = doc.objects.get(&obj_id).unwrap();
    assert!(
        ObjectStream::can_be_compressed(obj_id, obj, &doc),
        "Non-encryption objects referenced in trailer should be compressible"
    );
}

#[test]
fn test_can_be_compressed_xref_stream() {
    let mut doc = Document::new();
    let xref_id = doc.add_object(Object::Dictionary(dictionary! {
        "Type" => "XRef"
    }));

    let obj = doc.objects.get(&xref_id).unwrap();
    assert!(
        !ObjectStream::can_be_compressed(xref_id, obj, &doc),
        "XRef streams should not be compressible"
    );
}

#[test]
fn test_can_be_compressed_objstm() {
    let mut doc = Document::new();
    let objstm_id = doc.add_object(Object::Dictionary(dictionary! {
        "Type" => "ObjStm"
    }));

    let obj = doc.objects.get(&objstm_id).unwrap();
    assert!(
        !ObjectStream::can_be_compressed(objstm_id, obj, &doc),
        "Object streams should not be compressible"
    );
}

#[test]
fn test_can_be_compressed_catalog_non_linearized() {
    let mut doc = Document::new();
    let catalog_id = doc.add_object(Object::Dictionary(dictionary! {
        "Type" => "Catalog",
        "Pages" => Object::Reference((2, 0))
    }));

    let obj = doc.objects.get(&catalog_id).unwrap();
    assert!(
        ObjectStream::can_be_compressed(catalog_id, obj, &doc),
        "Catalog should be compressible in non-linearized PDFs"
    );
}

#[test]
fn test_can_be_compressed_catalog_linearized() {
    let mut doc = Document::new();

    // Add linearization dictionary
    let _lin_id = doc.add_object(Object::Dictionary(dictionary! {
        "Linearized" => 1
    }));

    let catalog_id = doc.add_object(Object::Dictionary(dictionary! {
        "Type" => "Catalog",
        "Pages" => Object::Reference((2, 0))
    }));

    let obj = doc.objects.get(&catalog_id).unwrap();
    assert!(
        !ObjectStream::can_be_compressed(catalog_id, obj, &doc),
        "Catalog should not be compressible in linearized PDFs"
    );
}

#[test]
fn test_can_be_compressed_pages() {
    let mut doc = Document::new();
    let pages_id = doc.add_object(Object::Dictionary(dictionary! {
        "Type" => "Pages",
        "Kids" => vec![Object::Reference((3, 0))],
        "Count" => 1
    }));

    let obj = doc.objects.get(&pages_id).unwrap();
    assert!(
        ObjectStream::can_be_compressed(pages_id, obj, &doc),
        "Pages objects should be compressible"
    );
}

#[test]
fn test_can_be_compressed_page() {
    let mut doc = Document::new();
    let page_id = doc.add_object(Object::Dictionary(dictionary! {
        "Type" => "Page",
        "Parent" => Object::Reference((2, 0)),
        "MediaBox" => vec![0.into(), 0.into(), 612.into(), 792.into()]
    }));

    let obj = doc.objects.get(&page_id).unwrap();
    assert!(
        ObjectStream::can_be_compressed(page_id, obj, &doc),
        "Page objects should be compressible"
    );
}

#[test]
fn test_can_be_compressed_encryption_dict() {
    let mut doc = Document::new();
    let encrypt_id = doc.add_object(Object::Dictionary(dictionary! {
        "Filter" => "Standard",
        "V" => 1,
        "R" => 2
    }));

    // Set as encryption dictionary
    doc.trailer.set("Encrypt", encrypt_id);

    let obj = doc.objects.get(&encrypt_id).unwrap();
    assert!(
        !ObjectStream::can_be_compressed(encrypt_id, obj, &doc),
        "Encryption dictionary should not be compressible"
    );
}

#[test]
fn test_catalog_can_be_compressed_with_trailer_reference() {
    let mut doc = Document::with_version("1.5");
    let catalog_id = doc.add_object(dictionary! {
        "Type" => "Catalog",
        "Pages" => Object::Reference((2, 0))
    });

    // Set as root in trailer (standard PDF structure)
    doc.trailer.set("Root", catalog_id);

    let catalog_obj = doc.objects.get(&catalog_id).unwrap();
    assert!(
        ObjectStream::can_be_compressed(catalog_id, catalog_obj, &doc),
        "Catalog should be compressible even when referenced in trailer"
    );
}

#[test]
fn test_info_dict_can_be_compressed_with_trailer_reference() {
    let mut doc = Document::with_version("1.5");
    let info_id = doc.add_object(dictionary! {
        "Title" => "Test PDF",
        "Author" => "Test Author",
        "CreationDate" => "D:20250807120000Z"
    });

    // Set as info in trailer
    doc.trailer.set("Info", info_id);

    let info_obj = doc.objects.get(&info_id).unwrap();
    assert!(
        ObjectStream::can_be_compressed(info_id, info_obj, &doc),
        "Info dictionary should be compressible even when referenced in trailer"
    );
}

#[test]
fn test_linearized_detection_via_catalog_compression() {
    let mut doc = Document::new();

    // Create catalog in non-linearized document
    let catalog_id = doc.add_object(Object::Dictionary(dictionary! {
        "Type" => "Catalog",
        "Pages" => Object::Reference((2, 0))
    }));

    // Initially should be compressible (not linearized)
    {
        let obj = doc.objects.get(&catalog_id).unwrap();
        assert!(
            ObjectStream::can_be_compressed(catalog_id, obj, &doc),
            "Catalog should be compressible in non-linearized PDF"
        );
    }

    // Add linearization dictionary
    let _lin_id = doc.add_object(Object::Dictionary(dictionary! {
        "Linearized" => 1,
        "L" => 12345,
        "H" => vec![Object::Integer(100), Object::Integer(200)]
    }));

    // Now catalog should NOT be compressible
    let obj = doc.objects.get(&catalog_id).unwrap();
    assert!(
        !ObjectStream::can_be_compressed(catalog_id, obj, &doc),
        "Catalog should not be compressible after linearization"
    );
}

#[test]
fn test_can_be_compressed_regular_objects() {
    let mut doc = Document::new();

    // Test various regular object types that should be compressible
    let int_id = doc.add_object(Object::Integer(42));
    let bool_id = doc.add_object(Object::Boolean(true));
    let string_id = doc.add_object(Object::String(b"Hello".to_vec(), lopdf::StringFormat::Literal));
    let name_id = doc.add_object(Object::Name(b"Test".to_vec()));
    let array_id = doc.add_object(Object::Array(vec![Object::Integer(1), Object::Integer(2)]));
    let dict_id = doc.add_object(Object::Dictionary(dictionary! {
        "Key" => "Value"
    }));

    // All should be compressible
    assert!(ObjectStream::can_be_compressed(
        int_id,
        doc.objects.get(&int_id).unwrap(),
        &doc
    ));
    assert!(ObjectStream::can_be_compressed(
        bool_id,
        doc.objects.get(&bool_id).unwrap(),
        &doc
    ));
    assert!(ObjectStream::can_be_compressed(
        string_id,
        doc.objects.get(&string_id).unwrap(),
        &doc
    ));
    assert!(ObjectStream::can_be_compressed(
        name_id,
        doc.objects.get(&name_id).unwrap(),
        &doc
    ));
    assert!(ObjectStream::can_be_compressed(
        array_id,
        doc.objects.get(&array_id).unwrap(),
        &doc
    ));
    assert!(ObjectStream::can_be_compressed(
        dict_id,
        doc.objects.get(&dict_id).unwrap(),
        &doc
    ));
}

#[test]
fn test_save_with_object_streams_empty_document() {
    let mut doc = Document::with_version("1.5");
    doc.trailer.set("Root", Object::Reference((1, 0)));

    let mut buffer = Vec::new();
    let result = doc.save_modern(&mut buffer);
    assert!(result.is_ok(), "Empty document should save successfully");
}

#[test]
fn test_save_with_object_streams_single_page() {
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

    let mut buffer = Vec::new();
    let result = doc.save_modern(&mut buffer);
    assert!(result.is_ok());

    let content = String::from_utf8_lossy(&buffer);
    assert!(
        content.contains("/ObjStm"),
        "Object streams should be created for single page PDF"
    );
}

#[test]
fn test_object_stream_builder_custom_config() {
    let builder = ObjectStream::builder().max_objects(50).compression_level(9);

    assert_eq!(builder.get_max_objects(), 50);
    assert_eq!(builder.get_compression_level(), 9);

    let _obj_stream = builder.build();
    // Note: max_objects and compression_level are private fields
    // We've already tested the builder getter methods above
}

#[test]
fn test_object_stream_add_max_objects() {
    let mut obj_stream = ObjectStream::builder().max_objects(3).build();

    // Add objects up to max
    assert!(obj_stream.add_object((1, 0), Object::Integer(1)).is_ok());
    assert!(obj_stream.add_object((2, 0), Object::Integer(2)).is_ok());
    assert!(obj_stream.add_object((3, 0), Object::Integer(3)).is_ok());

    // Should fail when exceeding max
    assert!(obj_stream.add_object((4, 0), Object::Integer(4)).is_err());
}

#[test]
fn test_object_stream_invalid_generation() {
    let mut obj_stream = ObjectStream::builder().build();

    // Note: The current implementation doesn't validate generation in add_object
    // It only checks during can_be_compressed. This test documents current behavior.
    let result = obj_stream.add_object((1, 1), Object::Integer(42));
    assert!(result.is_ok(), "add_object currently doesn't validate generation");
}

#[test]
fn test_save_modern_with_mixed_objects() {
    let mut doc = Document::with_version("1.5");
    let pages_id = doc.new_object_id();

    // Add various object types
    let font_id = doc.add_object(dictionary! {
        "Type" => "Font",
        "Subtype" => "Type1",
        "BaseFont" => "Helvetica"
    });

    let xobject_stream_id = doc.add_object(Stream::new(
        dictionary! { "Type" => "XObject", "Subtype" => "Image" },
        vec![0; 100],
    ));

    let annotation_id = doc.add_object(dictionary! {
        "Type" => "Annot",
        "Subtype" => "Text",
        "Rect" => vec![100.into(), 100.into(), 200.into(), 200.into()]
    });

    let page_id = doc.add_object(dictionary! {
        "Type" => "Page",
        "Parent" => pages_id,
        "MediaBox" => vec![0.into(), 0.into(), 612.into(), 792.into()],
        "Annots" => vec![annotation_id.into()],
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

    let mut buffer = Vec::new();
    let result = doc.save_modern(&mut buffer);
    assert!(result.is_ok());

    let content = String::from_utf8_lossy(&buffer);

    // Verify object streams were created
    assert!(content.contains("/ObjStm"));

    // Verify stream objects are NOT in object streams
    assert!(
        content.contains(&format!("{} 0 obj", xobject_stream_id.0)),
        "Stream objects should remain as top-level objects"
    );

    // Verify compressible objects are NOT present as individual objects
    assert!(
        !content.contains(&format!("{} 0 obj\n<</Type/Font", font_id.0)),
        "Font object should be compressed"
    );
    assert!(
        !content.contains(&format!("{} 0 obj\n<</Type/Annot", annotation_id.0)),
        "Annotation object should be compressed"
    );
}

#[test]
fn test_parse_existing_object_stream() {
    // Create a simple object stream content
    let content = b"1 0 2 50 3 100\n\
                    <</Type/Font/Subtype/Type1/BaseFont/Helvetica>>\n\
                    <</Type/Annot/Subtype/Text/Rect[100 100 200 200]>>\n\
                    42";

    let stream = Stream::new(
        dictionary! {
            "Type" => "ObjStm",
            "N" => 3,
            "First" => 15
        },
        content.to_vec(),
    );

    let obj_stream = ObjectStream::new(&stream);
    assert!(obj_stream.is_ok());

    let obj_stream = obj_stream.unwrap();
    assert_eq!(obj_stream.objects.len(), 3);
    assert!(obj_stream.objects.contains_key(&(1, 0)));
    assert!(obj_stream.objects.contains_key(&(2, 0)));
    assert!(obj_stream.objects.contains_key(&(3, 0)));
}

#[test]
fn test_regression_structural_objects_compression() {
    // This test ensures the fix for structural object compression doesn't regress
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

    // Set trailer reference
    doc.trailer.set("Root", catalog_id);

    // Verify objects can be compressed even after trailer reference
    let catalog_obj = doc.objects.get(&catalog_id).unwrap();
    let pages_obj = doc.objects.get(&pages_id).unwrap();
    let page_obj = doc.objects.get(&page_id).unwrap();

    assert!(
        ObjectStream::can_be_compressed(catalog_id, catalog_obj, &doc),
        "Catalog must be compressible in non-linearized PDFs even with trailer reference"
    );
    assert!(
        ObjectStream::can_be_compressed(pages_id, pages_obj, &doc),
        "Pages object must be compressible"
    );
    assert!(
        ObjectStream::can_be_compressed(page_id, page_obj, &doc),
        "Page object must be compressible"
    );

    // Save and verify compression
    let mut buffer = Vec::new();
    doc.save_modern(&mut buffer).unwrap();

    let content = String::from_utf8_lossy(&buffer);

    // Must have object streams
    assert!(
        content.contains("/ObjStm"),
        "Object streams must be created when saving with modern format"
    );

    // All structural objects should be compressed now
    assert!(
        !content.contains(&format!("{} 0 obj\n<</Type/Catalog", catalog_id.0)),
        "Catalog should be in object stream, not as individual object"
    );
    assert!(
        !content.contains(&format!("{} 0 obj\n<</Type/Pages", pages_id.0)),
        "Pages should be in object stream, not as individual object"
    );
    assert!(
        !content.contains(&format!("{} 0 obj\n<</Type/Page", page_id.0)),
        "Page should be in object stream, not as individual object"
    );
}
