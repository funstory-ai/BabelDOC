use lopdf::{Document, Object, ObjectStream, dictionary};

#[test]
fn test_multiple_trailer_references_all_compressible() {
    let mut doc = Document::with_version("1.5");

    // Create multiple objects that will be referenced in trailer
    let catalog_id = doc.add_object(dictionary! {
        "Type" => "Catalog",
        "Pages" => Object::Reference((10, 0))
    });

    let info_id = doc.add_object(dictionary! {
        "Title" => "Test",
        "Author" => "Test Author"
    });

    let metadata_id = doc.add_object(dictionary! {
        "Type" => "Metadata",
        "Subtype" => "XML"
    });

    let outlines_id = doc.add_object(dictionary! {
        "Type" => "Outlines",
        "Count" => 0
    });

    // Add all to trailer
    doc.trailer.set("Root", catalog_id);
    doc.trailer.set("Info", info_id);
    doc.trailer.set("Metadata", metadata_id);
    doc.trailer.set("Outlines", outlines_id);

    // All should be compressible
    assert!(
        ObjectStream::can_be_compressed(catalog_id, doc.objects.get(&catalog_id).unwrap(), &doc),
        "Catalog should be compressible"
    );
    assert!(
        ObjectStream::can_be_compressed(info_id, doc.objects.get(&info_id).unwrap(), &doc),
        "Info should be compressible"
    );
    assert!(
        ObjectStream::can_be_compressed(metadata_id, doc.objects.get(&metadata_id).unwrap(), &doc),
        "Metadata should be compressible"
    );
    assert!(
        ObjectStream::can_be_compressed(outlines_id, doc.objects.get(&outlines_id).unwrap(), &doc),
        "Outlines should be compressible"
    );
}

#[test]
fn test_encryption_dict_with_other_trailer_refs() {
    let mut doc = Document::with_version("1.5");

    // Create catalog and encryption dictionary
    let catalog_id = doc.add_object(dictionary! {
        "Type" => "Catalog",
        "Pages" => Object::Reference((10, 0))
    });

    let encrypt_id = doc.add_object(dictionary! {
        "Filter" => "Standard",
        "V" => 2,
        "R" => 3,
        "Length" => 128
    });

    let info_id = doc.add_object(dictionary! {
        "Title" => "Encrypted PDF"
    });

    // Add to trailer
    doc.trailer.set("Root", catalog_id);
    doc.trailer.set("Encrypt", encrypt_id);
    doc.trailer.set("Info", info_id);

    // Catalog and Info should be compressible, Encrypt should not
    assert!(
        ObjectStream::can_be_compressed(catalog_id, doc.objects.get(&catalog_id).unwrap(), &doc),
        "Catalog should be compressible even with encryption"
    );
    assert!(
        !ObjectStream::can_be_compressed(encrypt_id, doc.objects.get(&encrypt_id).unwrap(), &doc),
        "Encryption dictionary should NOT be compressible"
    );
    assert!(
        ObjectStream::can_be_compressed(info_id, doc.objects.get(&info_id).unwrap(), &doc),
        "Info should be compressible even with encryption"
    );
}

#[test]
fn test_trailer_with_non_reference_values() {
    let mut doc = Document::with_version("1.5");

    let catalog_id = doc.add_object(dictionary! {
        "Type" => "Catalog"
    });

    // Add various types to trailer
    doc.trailer.set("Root", catalog_id);
    doc.trailer.set("Size", Object::Integer(100));
    doc.trailer.set("Prev", Object::Integer(1234));
    doc.trailer.set(
        "ID",
        Object::Array(vec![
            Object::String(vec![1, 2, 3, 4], lopdf::StringFormat::Hexadecimal),
            Object::String(vec![5, 6, 7, 8], lopdf::StringFormat::Hexadecimal),
        ]),
    );

    // Catalog should still be compressible
    assert!(
        ObjectStream::can_be_compressed(catalog_id, doc.objects.get(&catalog_id).unwrap(), &doc),
        "Catalog should be compressible with non-reference trailer entries"
    );
}

#[test]
fn test_same_object_referenced_multiple_times() {
    let mut doc = Document::with_version("1.5");

    let shared_dict_id = doc.add_object(dictionary! {
        "Shared" => "Dictionary"
    });

    // Reference the same object from multiple trailer keys
    doc.trailer.set("Custom1", shared_dict_id);
    doc.trailer.set("Custom2", shared_dict_id);
    doc.trailer.set("Custom3", shared_dict_id);

    // Should still be compressible (not encryption dict)
    assert!(
        ObjectStream::can_be_compressed(shared_dict_id, doc.objects.get(&shared_dict_id).unwrap(), &doc),
        "Object referenced multiple times in trailer should be compressible"
    );
}

#[test]
fn test_encrypt_key_with_non_reference_value() {
    let mut doc = Document::with_version("1.5");

    let catalog_id = doc.add_object(dictionary! {
        "Type" => "Catalog"
    });

    // Set Encrypt to non-reference value (invalid but should not crash)
    doc.trailer.set("Root", catalog_id);
    doc.trailer.set("Encrypt", Object::Null);

    // Catalog should be compressible
    assert!(
        ObjectStream::can_be_compressed(catalog_id, doc.objects.get(&catalog_id).unwrap(), &doc),
        "Catalog should be compressible when Encrypt is not a reference"
    );
}

#[test]
fn test_missing_encrypt_key() {
    let mut doc = Document::with_version("1.5");

    let catalog_id = doc.add_object(dictionary! {
        "Type" => "Catalog"
    });

    let some_dict_id = doc.add_object(dictionary! {
        "Filter" => "Standard"  // Looks like encryption dict but not referenced as one
    });

    doc.trailer.set("Root", catalog_id);
    // No Encrypt key in trailer

    // Both should be compressible
    assert!(
        ObjectStream::can_be_compressed(catalog_id, doc.objects.get(&catalog_id).unwrap(), &doc),
        "Catalog should be compressible without Encrypt in trailer"
    );
    assert!(
        ObjectStream::can_be_compressed(some_dict_id, doc.objects.get(&some_dict_id).unwrap(), &doc),
        "Dictionary that looks like encryption should be compressible if not referenced as Encrypt"
    );
}

#[test]
fn test_linearized_with_trailer_references() {
    let mut doc = Document::with_version("1.5");

    // Add linearization dictionary
    let _lin_id = doc.add_object(dictionary! {
        "Linearized" => 1
    });

    let catalog_id = doc.add_object(dictionary! {
        "Type" => "Catalog"
    });

    let info_id = doc.add_object(dictionary! {
        "Title" => "Linearized PDF"
    });

    doc.trailer.set("Root", catalog_id);
    doc.trailer.set("Info", info_id);

    // In linearized PDF, catalog cannot be compressed but info can
    assert!(
        !ObjectStream::can_be_compressed(catalog_id, doc.objects.get(&catalog_id).unwrap(), &doc),
        "Catalog should NOT be compressible in linearized PDF"
    );
    assert!(
        ObjectStream::can_be_compressed(info_id, doc.objects.get(&info_id).unwrap(), &doc),
        "Info should be compressible even in linearized PDF"
    );
}

#[test]
fn test_indirect_object_chain() {
    let mut doc = Document::with_version("1.5");

    // Create a chain of objects
    let obj3_id = doc.add_object(dictionary! {
        "Level" => 3
    });

    let obj2_id = doc.add_object(dictionary! {
        "Level" => 2,
        "Next" => obj3_id
    });

    let obj1_id = doc.add_object(dictionary! {
        "Level" => 1,
        "Next" => obj2_id
    });

    // Only reference the first in trailer
    doc.trailer.set("Chain", obj1_id);

    // All should be compressible (trailer only directly references obj1)
    assert!(
        ObjectStream::can_be_compressed(obj1_id, doc.objects.get(&obj1_id).unwrap(), &doc),
        "Object directly referenced in trailer should be compressible"
    );
    assert!(
        ObjectStream::can_be_compressed(obj2_id, doc.objects.get(&obj2_id).unwrap(), &doc),
        "Object indirectly referenced should be compressible"
    );
    assert!(
        ObjectStream::can_be_compressed(obj3_id, doc.objects.get(&obj3_id).unwrap(), &doc),
        "Object indirectly referenced should be compressible"
    );
}

#[test]
fn test_real_world_trailer_structure() {
    let mut doc = Document::with_version("1.5");

    // Simulate a real PDF trailer structure
    let pages_id = doc.new_object_id();

    let catalog_id = doc.add_object(dictionary! {
        "Type" => "Catalog",
        "Pages" => pages_id,
        "Version" => "/1.5"
    });

    let info_id = doc.add_object(dictionary! {
        "Title" => "Real World PDF",
        "Author" => "Test Suite",
        "Subject" => "Testing trailer compression",
        "Creator" => "lopdf",
        "Producer" => "lopdf test",
        "CreationDate" => "D:20250101120000Z",
        "ModDate" => "D:20250807120000Z"
    });

    // Typical trailer entries
    doc.trailer.set("Root", catalog_id);
    doc.trailer.set("Info", info_id);
    doc.trailer.set("Size", Object::Integer(50));
    doc.trailer.set(
        "ID",
        Object::Array(vec![
            Object::String(b"<4E4F204944454153>".to_vec(), lopdf::StringFormat::Hexadecimal),
            Object::String(b"<4E4F204944454153>".to_vec(), lopdf::StringFormat::Hexadecimal),
        ]),
    );

    // Both catalog and info should be compressible
    assert!(
        ObjectStream::can_be_compressed(catalog_id, doc.objects.get(&catalog_id).unwrap(), &doc),
        "Real-world catalog should be compressible"
    );
    assert!(
        ObjectStream::can_be_compressed(info_id, doc.objects.get(&info_id).unwrap(), &doc),
        "Real-world info dictionary should be compressible"
    );
}

#[test]
fn test_compression_with_save_integration() {
    use lopdf::SaveOptions;

    let mut doc = Document::with_version("1.5");
    let pages_id = doc.new_object_id();

    let page_id = doc.add_object(dictionary! {
        "Type" => "Page",
        "Parent" => pages_id,
        "MediaBox" => vec![0.into(), 0.into(), 612.into(), 792.into()]
    });

    doc.objects.insert(
        pages_id,
        Object::Dictionary(dictionary! {
            "Type" => "Pages",
            "Kids" => vec![page_id.into()],
            "Count" => 1
        }),
    );

    let catalog_id = doc.add_object(dictionary! {
        "Type" => "Catalog",
        "Pages" => pages_id
    });

    let info_id = doc.add_object(dictionary! {
        "Title" => "Compression Test",
        "Keywords" => "test, compression, object streams"
    });

    doc.trailer.set("Root", catalog_id);
    doc.trailer.set("Info", info_id);

    // Save with object streams
    let options = SaveOptions::builder().use_object_streams(true).build();

    let mut output = Vec::new();
    doc.save_with_options(&mut output, options).unwrap();

    let content = String::from_utf8_lossy(&output);

    // Verify object streams exist
    assert!(content.contains("/ObjStm"), "Object streams should be created");

    // Verify trailer-referenced objects are compressed
    assert!(
        !content.contains(&format!("{} 0 obj\n<</Type/Catalog", catalog_id.0)),
        "Catalog should be in object stream"
    );
    assert!(
        !content.contains(&format!("{} 0 obj\n<</Title", info_id.0)),
        "Info should be in object stream"
    );

    // Load and verify the PDF is valid
    let loaded = Document::load_mem(&output).unwrap();
    assert!(loaded.trailer.get(b"Root").is_ok(), "Loaded PDF should have valid Root");
    assert!(loaded.trailer.get(b"Info").is_ok(), "Loaded PDF should have valid Info");
}
