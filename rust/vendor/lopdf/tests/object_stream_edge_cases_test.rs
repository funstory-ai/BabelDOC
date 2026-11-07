use lopdf::{Document, Object, ObjectStream, dictionary};

#[test]
fn test_empty_trailer() {
    let doc = Document::with_version("1.5");
    let obj_id = (1, 0);
    let obj = Object::Dictionary(dictionary! { "Test" => "Value" });

    // With empty trailer, all objects should be compressible
    assert!(
        ObjectStream::can_be_compressed(obj_id, &obj, &doc),
        "Object should be compressible with empty trailer"
    );
}

#[test]
fn test_malformed_encrypt_reference() {
    let mut doc = Document::with_version("1.5");

    let obj_id = doc.add_object(dictionary! {
        "Type" => "Catalog"
    });

    // Set Encrypt to a malformed reference (wrong generation)
    doc.trailer.set("Encrypt", Object::Reference((999, 99)));

    // Object should still be compressible (not the encryption dict)
    assert!(
        ObjectStream::can_be_compressed(obj_id, doc.objects.get(&obj_id).unwrap(), &doc),
        "Object should be compressible even with malformed Encrypt reference"
    );
}

#[test]
fn test_self_referencing_object() {
    let mut doc = Document::with_version("1.5");

    let obj_id = (5, 0);
    doc.objects.insert(
        obj_id,
        Object::Dictionary(dictionary! {
            "Type" => "Test",
            "Self" => Object::Reference(obj_id)  // Self reference
        }),
    );

    doc.trailer.set("Test", obj_id);

    // Should be compressible (not encryption dict)
    assert!(
        ObjectStream::can_be_compressed(obj_id, doc.objects.get(&obj_id).unwrap(), &doc),
        "Self-referencing object should be compressible"
    );
}

#[test]
fn test_circular_references() {
    let mut doc = Document::with_version("1.5");

    let obj1_id = (1, 0);
    let obj2_id = (2, 0);

    doc.objects.insert(
        obj1_id,
        Object::Dictionary(dictionary! {
            "Next" => Object::Reference(obj2_id)
        }),
    );

    doc.objects.insert(
        obj2_id,
        Object::Dictionary(dictionary! {
            "Next" => Object::Reference(obj1_id)  // Circular reference
        }),
    );

    doc.trailer.set("Start", obj1_id);

    // Both should be compressible
    assert!(
        ObjectStream::can_be_compressed(obj1_id, doc.objects.get(&obj1_id).unwrap(), &doc),
        "First object in circular reference should be compressible"
    );
    assert!(
        ObjectStream::can_be_compressed(obj2_id, doc.objects.get(&obj2_id).unwrap(), &doc),
        "Second object in circular reference should be compressible"
    );
}

#[test]
fn test_trailer_with_all_pdf_types() {
    let mut doc = Document::with_version("1.5");

    let obj_id = doc.add_object(dictionary! { "Type" => "Test" });

    // Add all possible PDF types to trailer
    doc.trailer.set("Root", obj_id);
    doc.trailer.set("Null", Object::Null);
    doc.trailer.set("Bool", Object::Boolean(true));
    doc.trailer.set("Int", Object::Integer(42));
    doc.trailer.set("Real", Object::Real(1.25));
    doc.trailer
        .set("String", Object::String(b"test".to_vec(), lopdf::StringFormat::Literal));
    doc.trailer.set("Name", Object::Name(b"Test".to_vec()));
    doc.trailer
        .set("Array", Object::Array(vec![Object::Integer(1), Object::Integer(2)]));
    doc.trailer
        .set("Dict", Object::Dictionary(dictionary! { "Key" => "Value" }));

    // Object should still be compressible
    assert!(
        ObjectStream::can_be_compressed(obj_id, doc.objects.get(&obj_id).unwrap(), &doc),
        "Object should be compressible with diverse trailer entries"
    );
}

#[test]
fn test_unicode_trailer_keys() {
    let mut doc = Document::with_version("1.5");

    let obj_id = doc.add_object(dictionary! { "Test" => "Value" });

    // Use Unicode/non-ASCII keys in trailer
    doc.trailer.set("Root", obj_id);
    doc.trailer.set(
        "Ünïcödé",
        Object::String(b"test".to_vec(), lopdf::StringFormat::Literal),
    );
    doc.trailer.set("日本語", Object::Integer(42));
    doc.trailer.set("🎯", Object::Boolean(true));

    // Should still work correctly
    assert!(
        ObjectStream::can_be_compressed(obj_id, doc.objects.get(&obj_id).unwrap(), &doc),
        "Object should be compressible with Unicode trailer keys"
    );
}

#[test]
fn test_very_large_trailer() {
    let mut doc = Document::with_version("1.5");

    let obj_id = doc.add_object(dictionary! { "Type" => "Catalog" });

    // Add many entries to trailer
    doc.trailer.set("Root", obj_id);
    for i in 0..1000 {
        doc.trailer
            .set(format!("Custom{:04}", i).as_bytes(), Object::Integer(i));
    }

    // Should still be efficient
    let start = std::time::Instant::now();
    let compressible = ObjectStream::can_be_compressed(obj_id, doc.objects.get(&obj_id).unwrap(), &doc);
    let duration = start.elapsed();

    assert!(compressible, "Object should be compressible even with large trailer");
    assert!(
        duration.as_micros() < 100,
        "Check should be fast even with large trailer"
    );
}

#[test]
fn test_concurrent_modification_safety() {
    // This test verifies the function doesn't panic with concurrent-like access patterns
    let mut doc = Document::with_version("1.5");

    let obj1_id = doc.add_object(dictionary! { "Type" => "Test1" });
    let obj2_id = doc.add_object(dictionary! { "Type" => "Test2" });

    doc.trailer.set("Ref1", obj1_id);

    // Check first object
    let result1 = ObjectStream::can_be_compressed(obj1_id, doc.objects.get(&obj1_id).unwrap(), &doc);

    // Modify trailer between checks
    doc.trailer.set("Ref2", obj2_id);
    doc.trailer.set("Encrypt", obj1_id); // Now obj1 is encryption dict

    // Check again - result should change
    let result2 = ObjectStream::can_be_compressed(obj1_id, doc.objects.get(&obj1_id).unwrap(), &doc);

    assert!(result1, "Initially should be compressible");
    assert!(!result2, "Should not be compressible after becoming encryption dict");
}
